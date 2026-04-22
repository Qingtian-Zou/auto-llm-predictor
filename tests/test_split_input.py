"""Tests for auto_llm_predictor.nodes.split_input.

Covers: split_input_csv no-op when test_csv provided, stratified split,
fallback to random split, state mutation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _make_state(tmp_path: Path, **overrides) -> dict:
    csv_path = tmp_path / "input.csv"
    # Balanced 20-row dataset with two classes so stratification is possible.
    df = pd.DataFrame({
        "age": list(range(20)),
        "bmi": [20.0 + i * 0.1 for i in range(20)],
        "target": ["A", "B"] * 10,
    })
    df.to_csv(csv_path, index=False)

    state = {
        "csv_path": str(csv_path),
        "test_csv_path": "",
        "target_column": "target",
        "task_type": "binary",
        "output_dir": str(tmp_path),
        "training_config": {"test_ratio": 0.25},
    }
    state.update(overrides)
    return state


class TestSplitInputCsv:
    def test_noop_when_test_csv_provided(self, tmp_path):
        from auto_llm_predictor.nodes.split_input import split_input_csv

        state = _make_state(tmp_path, test_csv_path="/some/external/test.csv")
        result = split_input_csv(state)

        assert result == {}
        assert not (tmp_path / "data" / "train.csv").exists()
        assert not (tmp_path / "data" / "test.csv").exists()

    def test_stratified_split_creates_files_and_updates_paths(self, tmp_path):
        from auto_llm_predictor.nodes.split_input import split_input_csv

        state = _make_state(tmp_path)
        result = split_input_csv(state)

        train_csv = tmp_path / "data" / "train.csv"
        test_csv = tmp_path / "data" / "test.csv"

        assert train_csv.exists()
        assert test_csv.exists()
        assert result["csv_path"] == str(train_csv)
        assert result["test_csv_path"] == str(test_csv)

        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
        assert len(train_df) + len(test_df) == 20
        # test_ratio=0.25 → 5 test rows
        assert len(test_df) == 5
        # Stratification preserves both classes on both sides
        assert set(train_df["target"].unique()) == {"A", "B"}
        assert set(test_df["target"].unique()) == {"A", "B"}

    def test_random_split_fallback_for_regression(self, tmp_path):
        """task_type='regression' skips stratification but still splits."""
        from auto_llm_predictor.nodes.split_input import split_input_csv

        csv_path = tmp_path / "input.csv"
        df = pd.DataFrame({
            "x": list(range(30)),
            "y": [float(i) for i in range(30)],  # continuous target
        })
        df.to_csv(csv_path, index=False)

        state = {
            "csv_path": str(csv_path),
            "test_csv_path": "",
            "target_column": "y",
            "task_type": "regression",
            "output_dir": str(tmp_path),
            "training_config": {"test_ratio": 0.2},
        }

        result = split_input_csv(state)
        assert Path(result["csv_path"]).exists()
        assert Path(result["test_csv_path"]).exists()

    def test_random_split_fallback_for_single_sample_class(self, tmp_path):
        """Classes with only one sample must not break the split."""
        from auto_llm_predictor.nodes.split_input import split_input_csv

        csv_path = tmp_path / "input.csv"
        df = pd.DataFrame({
            "x": list(range(11)),
            "target": ["A"] * 5 + ["B"] * 5 + ["rare"],  # rare class has 1 sample
        })
        df.to_csv(csv_path, index=False)

        state = {
            "csv_path": str(csv_path),
            "test_csv_path": "",
            "target_column": "target",
            "task_type": "multiclass",
            "output_dir": str(tmp_path),
            "training_config": {"test_ratio": 0.2},
        }

        # Must not raise — falls back to random split
        result = split_input_csv(state)
        assert Path(result["csv_path"]).exists()
        assert Path(result["test_csv_path"]).exists()

    def test_default_test_ratio(self, tmp_path):
        """When training_config has no test_ratio, default 0.2 is used."""
        from auto_llm_predictor.nodes.split_input import split_input_csv

        state = _make_state(tmp_path)
        state["training_config"] = {}  # no test_ratio
        result = split_input_csv(state)

        test_df = pd.read_csv(result["test_csv_path"])
        # 20 rows * 0.2 = 4 test rows
        assert len(test_df) == 4
