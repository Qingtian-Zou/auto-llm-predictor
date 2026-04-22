"""Tests for auto_llm_predictor.nodes.data_registration.

Covers: register_dataset copies all_data.json/test_data.json to canonical
train.json/test.json names and writes a valid LlamaFactory dataset_info.json.
"""

from __future__ import annotations

import json

import pytest


class TestRegisterDataset:
    def _setup(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "all_data.json").write_text(json.dumps([
            {"instruction": "I", "input": "a", "output": "A"},
            {"instruction": "I", "input": "b", "output": "B"},
            {"instruction": "I", "input": "c", "output": "A"},
        ]))
        (data_dir / "test_data.json").write_text(json.dumps([
            {"instruction": "I", "input": "x", "output": "A"},
            {"instruction": "I", "input": "y", "output": "B"},
        ]))
        return data_dir

    def test_copies_files_and_writes_dataset_info(self, tmp_path):
        from auto_llm_predictor.nodes.data_registration import register_dataset

        data_dir = self._setup(tmp_path)
        state = {
            "output_dir": str(tmp_path),
            "all_data_path": str(data_dir / "all_data.json"),
        }

        result = register_dataset(state)

        train_path = data_dir / "train.json"
        test_path = data_dir / "test.json"
        info_path = data_dir / "dataset_info.json"

        assert train_path.exists()
        assert test_path.exists()
        assert info_path.exists()

        train_data = json.loads(train_path.read_text())
        test_data = json.loads(test_path.read_text())
        assert len(train_data) == 3
        assert len(test_data) == 2

        info = json.loads(info_path.read_text())
        assert info == {
            "train": {"file_name": "train.json"},
            "test": {"file_name": "test.json"},
        }

        assert result["train_data_path"] == str(train_path)
        assert result["test_data_path"] == str(test_path)
        assert result["dataset_info_path"] == str(info_path)

    def test_raises_when_test_data_missing(self, tmp_path):
        """register_dataset assumes split_input_csv ran upstream — test_data.json
        must exist."""
        from auto_llm_predictor.nodes.data_registration import register_dataset

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "all_data.json").write_text(json.dumps([
            {"instruction": "I", "input": "a", "output": "A"},
        ]))
        # No test_data.json

        state = {
            "output_dir": str(tmp_path),
            "all_data_path": str(data_dir / "all_data.json"),
        }

        with pytest.raises(FileNotFoundError, match="test_data.json"):
            register_dataset(state)
