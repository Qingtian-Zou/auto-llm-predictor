"""Tests for auto_llm_predictor.nodes.cutoff.

Covers: determine_cutoff_len, _parse_cutoff_choice, and _round_up_to_multiple.
"""

from __future__ import annotations

import json


# ---------------------------------------------------------------------------
# determine_cutoff_len
# ---------------------------------------------------------------------------

class TestDetermineCutoffLen:
    """Tests for the determine_cutoff_len node (no LLM / GPU required)."""

    def _make_state(self, tmp_path, data, auto_cutoff=True, cutoff_len=4096):
        """Write data to train.json and return a minimal state dict."""
        train_path = tmp_path / "train.json"
        train_path.write_text(json.dumps(data))
        return {
            "output_dir": str(tmp_path),
            "train_data_path": str(train_path),
            "auto_cutoff": auto_cutoff,
            # Use a non-existent model name so the tokenizer load fails gracefully
            # and falls back to the character heuristic — correct for offline tests.
            "base_model": "test-model-that-does-not-exist",
            "training_config": {"cutoff_len": cutoff_len},
        }

    def test_auto_detect_is_multiple_of_512(self, tmp_path):
        """Auto-detected value is always a multiple of 512."""
        from auto_llm_predictor.nodes.cutoff import determine_cutoff_len

        data = [{"instruction": "a" * 40, "input": "", "output": "b"} for _ in range(20)]
        state = self._make_state(tmp_path, data, auto_cutoff=True)
        result = determine_cutoff_len(state)
        cl = result["cutoff_len"]
        assert cl >= 512
        assert cl % 512 == 0

    def test_auto_detect_covers_max(self, tmp_path):
        """Auto-detected cutoff_len must be >= maximum token count in data."""
        from auto_llm_predictor.nodes.cutoff import determine_cutoff_len, _count_tokens

        entries = [
            {"instruction": "x" * 100, "input": "", "output": "y"},
            {"instruction": "x" * 2000, "input": "", "output": "y"},
        ]
        state = self._make_state(tmp_path, entries, auto_cutoff=True)
        result = determine_cutoff_len(state)
        # tokenizer=None triggers the character-heuristic fallback in test environments
        max_tok = max(
            _count_tokens(e["instruction"] + e["input"] + e["output"], None)
            for e in entries
        )
        assert result["cutoff_len"] >= max_tok

    def test_user_override_respected(self, tmp_path):
        """When auto_cutoff=False, the user-supplied cutoff_len is used unchanged."""
        from auto_llm_predictor.nodes.cutoff import determine_cutoff_len

        data = [{"instruction": "hello", "input": "", "output": "world"}]
        state = self._make_state(tmp_path, data, auto_cutoff=False, cutoff_len=2048)
        result = determine_cutoff_len(state)
        assert result["cutoff_len"] == 2048

    def test_empty_train_json_falls_back(self, tmp_path):
        """An empty train.json should return cutoff_len=1024 instead of crashing."""
        from auto_llm_predictor.nodes.cutoff import determine_cutoff_len

        state = self._make_state(tmp_path, [], auto_cutoff=True)
        result = determine_cutoff_len(state)
        assert result["cutoff_len"] == 1024

    def test_missing_train_json_falls_back(self, tmp_path):
        """Missing train.json should return cutoff_len=1024 instead of crashing."""
        from auto_llm_predictor.nodes.cutoff import determine_cutoff_len

        state = {
            "output_dir": str(tmp_path),
            "train_data_path": str(tmp_path / "nonexistent.json"),
            "auto_cutoff": True,
            "base_model": "test-model-that-does-not-exist",
            "training_config": {"cutoff_len": 4096},
        }
        result = determine_cutoff_len(state)
        assert result["cutoff_len"] == 1024

    def test_result_is_multiple_of_512_various_sizes(self, tmp_path):
        """Result is always a multiple of 512 regardless of data size mix."""
        from auto_llm_predictor.nodes.cutoff import determine_cutoff_len

        data = [
            {"instruction": "x" * n, "input": "", "output": "y"}
            for n in [10, 500, 999, 4001]
        ]
        state = self._make_state(tmp_path, data, auto_cutoff=True)
        result = determine_cutoff_len(state)
        assert result["cutoff_len"] % 512 == 0


# ---------------------------------------------------------------------------
# _parse_cutoff_choice
# ---------------------------------------------------------------------------

class TestParseCutoffChoice:
    """Unit tests for the _parse_cutoff_choice helper."""

    def _alts(self):
        return {"p95": 9728, "p90": 9216, "p85": 8704, "p80": 8192}

    def _parse(self, user_input):
        from auto_llm_predictor.nodes.cutoff import _parse_cutoff_choice
        return _parse_cutoff_choice(user_input, 12288, self._alts())

    def test_approve_returns_primary(self):
        assert self._parse("approve") == 12288

    def test_empty_returns_primary(self):
        assert self._parse("") == 12288

    def test_named_percentile_p95(self):
        assert self._parse("p95") == 9728

    def test_named_percentile_p80(self):
        assert self._parse("p80") == 8192

    def test_custom_integer_rounded_to_512(self):
        # 7000 → next multiple of 512 = 7168
        result = self._parse("7000")
        assert result == 7168
        assert result % 512 == 0

    def test_exact_multiple_unchanged(self):
        assert self._parse("8192") == 8192

    def test_unrecognised_falls_back_to_primary(self):
        assert self._parse("gobbledygook") == 12288


# ---------------------------------------------------------------------------
# _round_up_to_multiple
# ---------------------------------------------------------------------------

class TestRoundUpToMultiple:
    """Unit tests for _round_up_to_multiple."""

    def _round(self, v, m=512):
        from auto_llm_predictor.nodes.cutoff import _round_up_to_multiple
        return _round_up_to_multiple(v, m)

    def test_already_a_multiple(self):
        assert self._round(512) == 512
        assert self._round(1024) == 1024

    def test_rounds_up_correctly(self):
        assert self._round(513) == 1024
        assert self._round(1) == 512
        assert self._round(1023) == 1024

    def test_custom_multiple(self):
        assert self._round(100, 256) == 256
        assert self._round(256, 256) == 256
        assert self._round(257, 256) == 512
