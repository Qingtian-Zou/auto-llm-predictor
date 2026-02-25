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

    # Parse the JSON response
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw[: raw.rfind("```")]
        raw = raw.strip()

    analysis = json.loads(raw)

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
