"""Nodes: write_balance_code, execute_balance_code — Balance training data."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from auto_llm_predictor.prompts.balance import BALANCE_SYSTEM, format_balance_prompt
from auto_llm_predictor.state import PipelineState
from auto_llm_predictor.utils import run_script

logger = logging.getLogger(__name__)


def write_balance_code(state: PipelineState, *, llm) -> dict:
    """Generate a Python script to balance the training data.

    Writes: balance_code, balance_code_path, balance_attempts, messages
    """
    output_dir = state["output_dir"]
    data_dir = str(Path(output_dir) / "data")
    data_json_path = state.get("all_data_path", str(Path(data_dir) / "all_data.json"))
    output_json_path = str(Path(data_dir) / "balanced_data.json")

    # Get balance strategy from the prep plan
    plan = json.loads(state.get("prep_plan", "{}"))
    balance_strategy = plan.get("balance_strategy", "none")

    # If the user specified a balance strategy via feedback, use it
    balance_feedback = state.get("balance_feedback", "")
    strategy_changed = False
    if balance_feedback:
        import re
        match = re.search(
            r"(?:balance(?:_strategy)?|(?:use\s+))(oversample|undersample|none)",
            balance_feedback, re.IGNORECASE,
        )
        if match:
            new_strategy = match.group(1).lower()
            if new_strategy != balance_strategy:
                balance_strategy = new_strategy
                strategy_changed = True

    if balance_strategy == "none":
        # No balancing needed — skip code generation
        logger.info("Balance strategy is 'none', skipping balance code generation.")
        return {
            "balance_code": "",
            "balance_code_path": "",
            "balance_attempts": 0,
            "balance_succeeded": True,
            "messages": [
                HumanMessage(content="[write_balance_code] No balancing needed (strategy=none)."),
            ],
        }

    # Build error context from previous attempt
    previous_error = ""
    if state.get("balance_result") and not state.get("balance_succeeded", True):
        previous_error = state["balance_result"]

    user_prompt = format_balance_prompt(
        data_json_path=data_json_path,
        output_json_path=output_json_path,
        task_type=state.get("task_type", "binary"),
        balance_strategy=balance_strategy,
        target_mapping=state.get("target_mapping", {}),
        previous_error=previous_error,
        user_feedback=balance_feedback,
    )

    messages = [
        SystemMessage(content=BALANCE_SYSTEM),
        HumanMessage(content=user_prompt),
    ]

    response = llm.invoke(messages)
    code = response.content.strip()

    # Strip markdown fences
    if code.startswith("```"):
        code = code.split("\n", 1)[1]
        if code.endswith("```"):
            code = code[: code.rfind("```")]
        code = code.strip()

    # Save the script
    script_dir = Path(output_dir) / "scripts"
    script_dir.mkdir(parents=True, exist_ok=True)
    script_path = script_dir / "balance_data.py"
    script_path.write_text(code)

    attempts = state.get("balance_attempts", 0) + 1
    logger.info("Saved balance script to %s (attempt %d)", script_path, attempts)

    ret = {
        "balance_code": code,
        "balance_code_path": str(script_path),
        "balance_attempts": attempts,
        "messages": [
            HumanMessage(
                content=f"[write_balance_code] Generated balance_data.py "
                f"(attempt {attempts}, strategy={balance_strategy})"
            ),
        ],
    }
    
    # If the user altered the balancing strategy, save it back to prep_plan
    # so the `.pipeline_state.json` accurately reflects the chosen action
    if strategy_changed:
        plan["balance_strategy"] = balance_strategy
        ret["prep_plan"] = json.dumps(plan, indent=2)

    return ret


def execute_balance_code(state: PipelineState) -> dict:
    """Execute the balancing script and validate the result.

    Writes: balance_result, balance_succeeded, messages
    """
    script_path = state.get("balance_code_path", "")

    # If no script was generated (strategy=none), just pass through
    if not script_path:
        return {
            "balance_result": "No balancing script to run.",
            "balance_succeeded": True,
            "messages": [
                HumanMessage(content="[execute_balance_code] Skipped (no balancing needed)."),
            ],
        }

    logger.info("Running balance script: %s", script_path)
    success, output = run_script(script_path, timeout=120)

    if success:
        # Verify balanced_data.json exists and is valid
        data_dir = Path(state["output_dir"]) / "data"
        balanced_data_path = data_dir / "balanced_data.json"
        if balanced_data_path.exists():
            with open(balanced_data_path) as f:
                data = json.load(f)
            logger.info("Balanced data saved to %s: %d examples", balanced_data_path.name, len(data))
        else:
            success = False
            output += f"\nERROR: {balanced_data_path.name} not found after balancing."

    logger.info("Balance script %s. Output:\n%s",
                "succeeded" if success else "FAILED", output[:1000])

    ret = {
        "balance_result": output,
        "balance_succeeded": success,
        "messages": [
            HumanMessage(
                content=f"[execute_balance_code] {'SUCCESS' if success else 'FAILED'}. "
                f"Output: {output[:500]}"
            ),
        ],
    }
    
    # Update the path to point to the balanced data for downstream splitting
    if success:
        ret["all_data_path"] = str(balanced_data_path)
        
    return ret


def check_balance_result(state: PipelineState) -> str:
    """Conditional edge: route after balance execution.

    Returns 'review_balanced_data' if succeeded,
    'write_balance_code' to retry on failure.
    Gives up after 3 attempts.
    """
    if state.get("balance_succeeded"):
        return "review_balanced_data"

    if state.get("balance_attempts", 0) >= 3:
        logger.warning("Max balance attempts reached. Proceeding to review.")
        return "review_balanced_data"

    return "write_balance_code"
