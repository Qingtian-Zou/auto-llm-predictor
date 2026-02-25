"""Node: write_prep_code — LLM generates a data preparation Python script."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from auto_llm_predictor.prompts.codegen import CODEGEN_SYSTEM, format_codegen_prompt
from auto_llm_predictor.state import PipelineState

logger = logging.getLogger(__name__)


def write_prep_code(state: PipelineState, *, llm) -> dict:
    """Generate a self-contained Python script to convert CSV → JSON.

    Writes: prep_code, prep_code_path, prep_attempts, messages
    """
    output_dir = state["output_dir"]
    output_data_dir = str(Path(output_dir) / "data")

    # Parse the plan (it was stored as a JSON string)
    plan = json.loads(state["prep_plan"])

    # Build the error context from the previous attempt if any
    previous_error = ""
    if state.get("prep_result") and not state.get("prep_succeeded", True):
        previous_error = state["prep_result"]

    user_prompt = format_codegen_prompt(
        csv_path=state["csv_path"],
        data_profile=state["data_profile"],
        target_column=state["target_column"],
        task_type=state["task_type"],
        target_mapping=state["target_mapping"],
        selected_features=plan["selected_features"],
        instruction_template=plan["instruction_template"],
        input_format=plan["input_format"],
        output_format=plan["output_format"],
        data_cleaning_steps=plan.get("data_cleaning_steps", []),
        output_data_dir=output_data_dir,
        test_csv_path=state.get("test_csv_path", ""),
        previous_error=previous_error,
        user_feedback=state.get("user_feedback", ""),
    )

    messages = [
        SystemMessage(content=CODEGEN_SYSTEM),
        HumanMessage(content=user_prompt),
    ]

    response = llm.invoke(messages)
    code = response.content.strip()

    # Strip markdown fences if present
    if code.startswith("```"):
        # Remove first line (```python or ```)
        code = code.split("\n", 1)[1]
        if code.endswith("```"):
            code = code[: code.rfind("```")]
        code = code.strip()

    # Save the script
    script_dir = Path(output_dir) / "scripts"
    script_dir.mkdir(parents=True, exist_ok=True)
    script_path = script_dir / "prepare_data.py"
    script_path.write_text(code)
    logger.info("Saved prep script to %s (%d lines)", script_path, code.count("\n") + 1)

    attempts = state.get("prep_attempts", 0) + 1

    return {
        "prep_code": code,
        "prep_code_path": str(script_path),
        "prep_attempts": attempts,
        "messages": [
            HumanMessage(content=f"[write_prep_code] Generated prepare_data.py (attempt {attempts}, "
                        f"{code.count(chr(10)) + 1} lines)"),
        ],
    }
