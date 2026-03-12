"""Node: explore_data — Profile the CSV and identify the prediction target."""

from __future__ import annotations

import json
import logging
import pandas as pd

from langchain_core.messages import HumanMessage, SystemMessage

from auto_llm_predictor.prompts.explore import EXPLORE_SYSTEM, format_explore_prompt
from auto_llm_predictor.state import PipelineState
from auto_llm_predictor.utils import profile_csv

logger = logging.getLogger(__name__)


def explore_data(state: PipelineState, *, llm) -> dict:
    """Profile the CSV and use the LLM to identify target, task type, and mapping.

    Writes: data_profile, target_column, task_type, target_mapping, messages
    """
    csv_path = state["csv_path"]
    logger.info("Profiling CSV: %s", csv_path)

    data_profile = profile_csv(csv_path)

    # If a separate test CSV is provided, verify header alignment.
    test_csv_path = state.get("test_csv_path")
    if test_csv_path:
        try:
            train_cols = set(pd.read_csv(csv_path, nrows=0).columns)
            test_cols = set(pd.read_csv(test_csv_path, nrows=0).columns)
            
            missing_in_test = sorted(list(train_cols - test_cols))
            missing_in_train = sorted(list(test_cols - train_cols))
            
            if missing_in_test or missing_in_train:
                warning_lines = ["\n⚠️ WARNING: Feature mismatch between Train and Test CSVs!"]
                if missing_in_test:
                    warning_lines.append(f"   Missing in TEST: {missing_in_test}")
                if missing_in_train:
                    warning_lines.append(f"   Missing in TRAIN: {missing_in_train}")
                warning_lines.append("   Make sure to handle these missing columns during data preparation (e.g., dropping them or filling with defaults).")
                
                warning_text = "\n".join(warning_lines)
                logger.warning(warning_text)
                data_profile += f"\n\n{warning_text}"
        except Exception as e:
            logger.warning("Failed to compare train/test CSV headers: %s", e)

    user_prompt = format_explore_prompt(
        data_profile=data_profile,
        target_column=state.get("target_column", ""),
    )

    messages = [
        SystemMessage(content=EXPLORE_SYSTEM),
        HumanMessage(content=user_prompt),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()
    logger.info("LLM explore response: %s", raw[:500])

    # Parse the JSON response — extract JSON object robustly
    # Strip markdown fences if present
    if raw.startswith("```"):
        parts = raw.split("\n", 1)
        raw = parts[1] if len(parts) > 1 else ""
        if raw.endswith("```"):
            raw = raw[: raw.rfind("```")]
        raw = raw.strip()

    # Use regex to extract JSON object (handles conversational text around JSON)
    import re
    json_match = re.search(r"(\{.*\})", raw, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        json_str = raw.strip()

    try:
        analysis = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse LLM explore response as JSON. Raw response:\n%s", raw[:1000])
        raise ValueError(
            f"LLM returned unparseable response for data exploration. "
            f"JSON error: {e}. Response preview: {raw[:200]}"
        ) from e

    # For regression tasks, target_mapping is not meaningful — force it empty.
    if analysis["task_type"] == "regression":
        if analysis.get("target_mapping"):
            logger.info(
                "task_type is 'regression'; clearing target_mapping "
                "(%d entries).", len(analysis["target_mapping"]),
            )
        analysis["target_mapping"] = {}
    else:
        # Validate target_mapping completeness against actual CSV values
        target_col = analysis["target_column"]
        try:
            df = pd.read_csv(csv_path, usecols=[target_col], low_memory=False)
            actual_values = set(str(v) for v in df[target_col].dropna().unique())
            mapped_keys = set(str(k) for k in analysis["target_mapping"].keys())
            missing = actual_values - mapped_keys
            if missing:
                logger.warning(
                    "target_mapping is missing %d value(s): %s. Auto-filling.",
                    len(missing), missing,
                )
                for val in sorted(missing):
                    analysis["target_mapping"][val] = val
            # Reconcile task_type with actual class count
            n_classes = len(analysis["target_mapping"])
            if analysis["task_type"] == "binary" and n_classes > 2:
                logger.warning(
                    "task_type was 'binary' but found %d classes. Correcting to 'multiclass'.",
                    n_classes,
                )
                analysis["task_type"] = "multiclass"
            elif analysis["task_type"] == "multiclass" and n_classes == 2:
                logger.warning(
                    "task_type was 'multiclass' but found only 2 classes. Correcting to 'binary'.",
                )
                analysis["task_type"] = "binary"
        except Exception as e:
            logger.warning("Could not validate target_mapping completeness: %s", e)

    return {
        "data_profile": data_profile,
        "target_column": analysis["target_column"],
        "task_type": analysis["task_type"],
        "target_mapping": analysis["target_mapping"],
        "messages": [
            HumanMessage(content=f"[explore_data] CSV profiled. Target: {analysis['target_column']}, "
                        f"Task: {analysis['task_type']}. {analysis.get('reasoning', '')}"),
        ],
    }
