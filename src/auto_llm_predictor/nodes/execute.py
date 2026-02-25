"""Node: execute_prep_code — Run the generated data preparation script."""

from __future__ import annotations

import ast
import json
import logging
import re
from collections import Counter
from pathlib import Path

from langchain_core.messages import HumanMessage

from auto_llm_predictor.state import PipelineState
from auto_llm_predictor.utils import run_script

logger = logging.getLogger(__name__)


def _print_class_distribution(data: list[dict], label: str) -> str:
    """Compute and print class distribution from Alpaca JSON entries."""
    counts = Counter(entry.get("output", "") for entry in data)
    total = sum(counts.values())
    lines = [f"\n{label} class distribution ({total} examples):"]
    for cls, count in sorted(counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / total if total else 0
        lines.append(f"  {cls}: {count} ({pct:.1f}%)")
    text = "\n".join(lines)
    print(text, flush=True)
    return text


def _extract_target_mapping_from_code(code: str) -> dict | None:
    """Attempt to parse target_mapping = {...} out of the generated prep script."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "target_mapping":
                        if isinstance(node.value, ast.Dict):
                            return ast.literal_eval(node.value)
    except Exception as e:
        logger.warning("Failed to AST parse prep_code for target_mapping: %s", e)
        
    # Fallback regex if AST parsing fails or doesn't find it exactly
    match = re.search(r"target_mapping\s*=\s*(\{.*?\})", code, re.DOTALL)
    if match:
        try:
            # literal_eval handles standard dicts well
            return ast.literal_eval(match.group(1))
        except Exception:
            pass
            
    return None


def execute_prep_code(state: PipelineState) -> dict:
    """Run the generated prepare_data.py and check outputs.

    Writes: prep_result, prep_succeeded, all_data_path, dataset_info_path, target_mapping, messages
    """
    script_path = state["prep_code_path"]
    output_data_dir = Path(state["output_dir"]) / "data"

    logger.info("Executing prep script: %s", script_path)
    success, output = run_script(script_path, timeout=300)

    # Check required output files
    all_data_path = output_data_dir / "all_data.json"
    info_path = output_data_dir / "dataset_info.json"

    files_exist = all_data_path.exists() and info_path.exists()

    # Optionally check test_data.json when test CSV was provided
    test_data_path = output_data_dir / "test_data.json"
    has_test_csv = bool(state.get("test_csv_path"))
    if has_test_csv and not test_data_path.exists():
        output += "\n\nERROR: Test CSV was provided but test_data.json not found."
        success = False

    if success and not files_exist:
        output += "\n\nERROR: Script succeeded but output files not found. "
        output += f"Expected: {all_data_path}, {info_path}"
        success = False

    # Print class distributions
    if success and files_exist:
        logger.info("Prep script succeeded. Output files in %s", output_data_dir)
        try:
            with open(all_data_path) as f:
                all_data = json.load(f)
            label = "Train" if has_test_csv else "All data"
            output += _print_class_distribution(all_data, label)
        except Exception as e:
            logger.warning("Could not compute class dist: %s", e)

        if has_test_csv and test_data_path.exists():
            try:
                with open(test_data_path) as f:
                    test_data = json.load(f)
                output += _print_class_distribution(test_data, "Test")
            except Exception as e:
                logger.warning("Could not compute test class dist: %s", e)
    else:
        logger.warning("Prep script failed (attempt %d). Output:\n%s",
                       state.get("prep_attempts", 1), output[-2000:])

    ret = {
        "prep_result": output[-3000:],
        "prep_succeeded": success and files_exist,
        "all_data_path": str(all_data_path) if files_exist else "",
        "dataset_info_path": str(info_path) if files_exist else "",
        "messages": [
            HumanMessage(
                content=f"[execute_prep_code] {'SUCCESS' if success and files_exist else 'FAILED'}. "
                + (f"Files created in {output_data_dir}"
                   if files_exist
                   else f"Error: {output[-500:]}")
            ),
        ],
    }

    # Extract target_mapping from generated script if it succeeds
    # This ensures any overrides from the review stage are persisted back to state
    if files_exist:
        prep_code = state.get("prep_code", "")
        if prep_code:
            updated_mapping = _extract_target_mapping_from_code(prep_code)
            if updated_mapping is not None:
                # Convert keys to strings so json serialization doesn't choke later
                str_mapping = {str(k): v for k, v in updated_mapping.items()}
                if str_mapping != state.get("target_mapping", {}):
                    logger.info("Extracted updated target_mapping from generated script: %s", str_mapping)
                    ret["target_mapping"] = str_mapping

    return ret


def check_prep_result(state: PipelineState) -> str:
    """Conditional edge: route to data verification or retry code generation.

    Returns 'verify_prepared_data' if prep succeeded, otherwise 'write_prep_code'.
    Gives up after 3 attempts.
    """
    if state.get("prep_succeeded"):
        return "verify_prepared_data"

    if state.get("prep_attempts", 0) >= 3:
        logger.warning("Max prep attempts reached. Proceeding to review anyway.")
        return "verify_prepared_data"

    return "write_prep_code"
