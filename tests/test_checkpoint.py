"""Tests for auto_llm_predictor.checkpoint.

Covers: save_state/load_state round-trip, error handling, and atomic writes.
"""

from __future__ import annotations

import json

import pytest


# ---------------------------------------------------------------------------
# save_state / load_state
# ---------------------------------------------------------------------------

class TestCheckpoint:
    def test_round_trip(self, tmp_path):
        from auto_llm_predictor.checkpoint import load_state, save_state

        state = {
            "csv_path": "/tmp/data.csv",
            "target_column": "response",
            "training_config": {"lora_rank": 64},
            "messages": ["should be excluded"],
        }
        save_state(state, str(tmp_path))
        loaded = load_state(str(tmp_path))
        assert loaded["csv_path"] == "/tmp/data.csv"
        assert loaded["training_config"]["lora_rank"] == 64
        # messages should be a fresh empty list
        assert loaded["messages"] == []

    def test_missing_state_file(self, tmp_path):
        from auto_llm_predictor.checkpoint import load_state

        with pytest.raises(FileNotFoundError):
            load_state(str(tmp_path))

    def test_corrupted_state_file(self, tmp_path):
        from auto_llm_predictor.checkpoint import load_state

        state_file = tmp_path / ".pipeline_state.json"
        state_file.write_text("{broken json!!")
        with pytest.raises(ValueError, match="corrupted"):
            load_state(str(tmp_path))


# ---------------------------------------------------------------------------
# Atomic checkpoint writes
# ---------------------------------------------------------------------------

class TestCheckpointAtomicWrite:
    """Tests for the atomic write logic in checkpoint.save_state."""

    def test_no_partial_file_on_serialization_error(self, tmp_path):
        """If serialization fails, the state file should not be created."""
        from auto_llm_predictor.checkpoint import save_state

        state_path = tmp_path / ".pipeline_state.json"

        # First, do a successful save
        save_state({"foo": "bar"}, str(tmp_path))
        assert state_path.exists()
        original = state_path.read_text()

        # Now attempt to save — but monkeypatch json.dump to raise mid-write
        import auto_llm_predictor.checkpoint as cp
        real_dump = json.dump

        def broken_dump(*args, **kwargs):
            raise IOError("simulated disk full")

        cp.json.dump = broken_dump
        try:
            with pytest.raises(IOError, match="simulated disk full"):
                save_state({"new": "data"}, str(tmp_path))
        finally:
            cp.json.dump = real_dump

        # Original file should be intact — not corrupted
        assert state_path.read_text() == original

    def test_no_leftover_tmp_files(self, tmp_path):
        """Successful saves should not leave .tmp files behind."""
        from auto_llm_predictor.checkpoint import save_state

        save_state({"a": 1}, str(tmp_path))
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == []

    def test_round_trip_atomic(self, tmp_path):
        """Basic round-trip still works with atomic writes."""
        from auto_llm_predictor.checkpoint import load_state, save_state

        save_state({"csv_path": "/data.csv", "target_column": "y"}, str(tmp_path))
        loaded = load_state(str(tmp_path))
        assert loaded["csv_path"] == "/data.csv"
        assert loaded["target_column"] == "y"
