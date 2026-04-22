# Copyright 2024-2026 Qingtian Zou
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Node: split_input_csv — Split the input CSV into train/test before any
data-dependent preprocessing runs.

Runs early in the pipeline (after ``explore_data`` so the target column is
known, before ``select_features`` / ``plan_preparation``).  This guarantees
that every later step — feature selection, planning, the LLM-generated
``prepare_data.py``, balancing — only sees training data, so any fitted
transformer (encoders, scalers, imputers, vocabularies) is fit on train
and then applied to test.

If the user already supplied a separate ``--test-csv``, this node is a
no-op and the user-supplied test set flows through unchanged.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from langchain_core.messages import HumanMessage

from auto_llm_predictor.state import PipelineState

logger = logging.getLogger(__name__)


def split_input_csv(state: PipelineState) -> dict:
    """Stratified-split the input CSV into train.csv / test.csv.

    No-op if ``state["test_csv_path"]`` is already set.

    Writes: csv_path, test_csv_path, messages
    """
    if state.get("test_csv_path"):
        logger.info(
            "split_input_csv: test_csv_path already set (%s) — skipping split.",
            state["test_csv_path"],
        )
        return {}

    csv_path = state["csv_path"]
    target_column = state.get("target_column", "")
    task_type = state.get("task_type", "multiclass")
    training_config = state.get("training_config", {})
    test_ratio = float(training_config.get("test_ratio", 0.2))

    data_dir = Path(state["output_dir"]) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_csv = data_dir / "train.csv"
    test_csv = data_dir / "test.csv"

    df = pd.read_csv(csv_path, low_memory=False)
    n_total = len(df)

    from sklearn.model_selection import train_test_split

    stratify = None
    if task_type != "regression" and target_column and target_column in df.columns:
        # Only stratify on classes with at least 2 samples (sklearn requirement).
        counts = df[target_column].value_counts(dropna=False)
        if (counts >= 2).all() and len(counts) >= 2:
            stratify = df[target_column]

    try:
        train_df, test_df = train_test_split(
            df,
            test_size=test_ratio,
            random_state=42,
            stratify=stratify,
        )
    except ValueError as exc:
        logger.warning(
            "Stratified split failed (%s) — falling back to random split.", exc,
        )
        train_df, test_df = train_test_split(
            df, test_size=test_ratio, random_state=42,
        )

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    summary = (
        f"Split input CSV ({n_total} rows, test_ratio={test_ratio}) → "
        f"train.csv ({len(train_df)} rows), test.csv ({len(test_df)} rows). "
        f"Stratified by '{target_column}'." if stratify is not None
        else f"Split input CSV ({n_total} rows, test_ratio={test_ratio}) → "
        f"train.csv ({len(train_df)} rows), test.csv ({len(test_df)} rows). "
        f"Random split (no stratification)."
    )
    logger.info(summary)
    print(f"\n{'=' * 50}\nSPLIT INPUT CSV\n{'=' * 50}\n{summary}", flush=True)

    return {
        "csv_path": str(train_csv),
        "test_csv_path": str(test_csv),
        "messages": [HumanMessage(content=f"[split_input_csv] {summary}")],
    }
