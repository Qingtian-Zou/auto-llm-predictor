"""Unit tests for robustness improvements.

Tests utility functions and hardened error-handling paths
that do NOT require an LLM API or GPU.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# profile_csv
# ---------------------------------------------------------------------------

class TestProfileCSV:
    """Tests for utils.profile_csv."""

    def test_normal_csv(self, tmp_path):
        """profile_csv returns a meaningful summary for a valid CSV."""
        from auto_llm_predictor.utils import profile_csv

        csv = tmp_path / "data.csv"
        csv.write_text("a,b,target\n1,2,yes\n3,4,no\n5,6,yes\n")
        result = profile_csv(str(csv))
        assert "data.csv" in result
        assert "3 rows" in result
        assert "3 columns" in result

    def test_malformed_csv(self, tmp_path):
        """profile_csv returns an error message instead of crashing on a bad file."""
        from auto_llm_predictor.utils import profile_csv

        bad = tmp_path / "bad.csv"
        bad.write_bytes(b"\x00\x01\x02\x03\xff\xfe")
        result = profile_csv(str(bad))
        assert "ERROR" in result or "error" in result.lower()

    def test_empty_csv(self, tmp_path):
        """profile_csv handles an empty file gracefully."""
        from auto_llm_predictor.utils import profile_csv

        empty = tmp_path / "empty.csv"
        empty.write_text("")
        result = profile_csv(str(empty))
        # Should return something (error or degenerate summary), not crash
        assert isinstance(result, str)
        assert len(result) > 0

    def test_operator_precedence_fix(self, tmp_path):
        """Categorical column detection uses correct precedence.

        A numeric column with <=10 unique values should be included,
        and an object column with >20 unique values should NOT be included
        in the value-counts section.
        """
        from auto_llm_predictor.utils import profile_csv

        # Build a CSV with a numeric low-cardinality column (should show up)
        # and a string high-cardinality column (should NOT show up for value counts)
        rows = ["id,score\n"]
        for i in range(30):
            rows.append(f"name_{i},{i % 5}\n")
        csv = tmp_path / "prec.csv"
        csv.write_text("".join(rows))

        result = profile_csv(str(csv))
        assert "Value counts for 'score'" in result
        # 'id' has 30 unique string values — should not show value counts
        assert "Value counts for 'id'" not in result


# ---------------------------------------------------------------------------
# run_script — uses sys.executable
# ---------------------------------------------------------------------------

class TestRunScript:
    """Tests for utils.run_script."""

    def test_successful_script(self, tmp_path):
        from auto_llm_predictor.utils import run_script

        script = tmp_path / "ok.py"
        script.write_text("print('hello')\n")
        success, output = run_script(str(script))
        assert success is True
        assert "hello" in output

    def test_failing_script(self, tmp_path):
        from auto_llm_predictor.utils import run_script

        script = tmp_path / "fail.py"
        script.write_text("raise ValueError('boom')\n")
        success, output = run_script(str(script))
        assert success is False
        assert "boom" in output

    def test_timeout(self, tmp_path):
        from auto_llm_predictor.utils import run_script

        script = tmp_path / "slow.py"
        script.write_text("import time; time.sleep(60)\n")
        success, output = run_script(str(script), timeout=1)
        assert success is False
        assert "timed out" in output.lower()


# ---------------------------------------------------------------------------
# _extract_label (evaluate.py)
# ---------------------------------------------------------------------------

class TestExtractLabel:
    """Tests for nodes.evaluate._extract_label."""

    def _extract(self, text, mapping):
        from auto_llm_predictor.nodes.evaluate import _extract_label
        return _extract_label(text, mapping)

    def test_exact_match(self):
        mapping = {"0": "No", "1": "Yes"}
        assert self._extract("Yes", mapping) == "Yes"
        assert self._extract("No", mapping) == "No"

    def test_case_insensitive(self):
        mapping = {"0": "No", "1": "Yes"}
        assert self._extract("yes", mapping) == "Yes"
        assert self._extract("NO", mapping) == "No"

    def test_prefix_match(self):
        mapping = {"0": "No", "1": "Yes"}
        assert self._extract("Yes, I think so", mapping) == "Yes"

    def test_substring_match(self):
        mapping = {"0": "No Response", "1": "Response"}
        assert self._extract("The answer is Response.", mapping) == "Response"

    def test_empty_mapping_returns_raw(self):
        """With an empty mapping, raw non-empty text is returned as-is."""
        assert self._extract("SomeLabel", {}) == "SomeLabel"

    def test_empty_text_returns_none(self):
        assert self._extract("", {"0": "A"}) is None

    def test_whitespace_only(self):
        assert self._extract("   ", {"0": "A"}) is None


# ---------------------------------------------------------------------------
# _coerce_value, _parse_overrides (review.py)
# ---------------------------------------------------------------------------

class TestCoerceValue:
    def _coerce(self, value):
        from auto_llm_predictor.nodes.review import _coerce_value
        return _coerce_value(value)

    def test_int(self):
        assert self._coerce("42") == 42
        assert isinstance(self._coerce("42"), int)

    def test_float(self):
        assert self._coerce("3.14") == pytest.approx(3.14)

    def test_bool_true(self):
        assert self._coerce("true") is True
        assert self._coerce("True") is True

    def test_bool_false(self):
        assert self._coerce("false") is False

    def test_string_passthrough(self):
        assert self._coerce("cosine") == "cosine"

    def test_scientific_notation(self):
        assert self._coerce("2.0e-5") == pytest.approx(2.0e-5)


class TestParseOverrides:
    def _parse(self, text):
        from auto_llm_predictor.nodes.review import _parse_overrides
        return _parse_overrides(text)

    def test_single_override(self):
        result = self._parse("lora_rank: 32")
        assert result == {"lora_rank": "32"}

    def test_multiple_comma_separated(self):
        result = self._parse("lora_rank: 32, num_train_epochs: 5")
        assert result == {"lora_rank": "32", "num_train_epochs": "5"}

    def test_newline_separated(self):
        result = self._parse("lora_rank: 32\nlearning_rate: 1.0e-5")
        assert result == {"lora_rank": "32", "learning_rate": "1.0e-5"}

    def test_empty_string(self):
        assert self._parse("") == {}

    def test_no_colons(self):
        assert self._parse("just some text without overrides") == {}


# ---------------------------------------------------------------------------
# load_jsonl
# ---------------------------------------------------------------------------

class TestLoadJSONL:
    def test_normal(self, tmp_path):
        from auto_llm_predictor.utils import load_jsonl

        f = tmp_path / "data.jsonl"
        f.write_text('{"a": 1}\n{"b": 2}\n')
        result = load_jsonl(str(f))
        assert len(result) == 2
        assert result[0] == {"a": 1}

    def test_empty_file(self, tmp_path):
        from auto_llm_predictor.utils import load_jsonl

        f = tmp_path / "empty.jsonl"
        f.write_text("")
        assert load_jsonl(str(f)) == []

    def test_blank_lines(self, tmp_path):
        from auto_llm_predictor.utils import load_jsonl

        f = tmp_path / "blanks.jsonl"
        f.write_text('{"a": 1}\n\n\n{"b": 2}\n\n')
        result = load_jsonl(str(f))
        assert len(result) == 2


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
# _apply_feedback_overrides (plan.py)
# ---------------------------------------------------------------------------

class TestFeedbackOverrides:
    def _apply(self, plan, feedback):
        from auto_llm_predictor.nodes.plan import _apply_feedback_overrides
        return _apply_feedback_overrides(plan, feedback)

    def test_drop_features(self):
        plan = {"selected_features": ["age", "bmi", "smoker"], "dropped_features": []}
        result = self._apply(plan, "drop features: smoker")
        assert "smoker" not in result["selected_features"]
        assert "smoker" in result["dropped_features"]

    def test_add_features(self):
        plan = {"selected_features": ["age"], "dropped_features": ["weight"]}
        result = self._apply(plan, "add features: weight")
        assert "weight" in result["selected_features"]
        assert "weight" not in result["dropped_features"]

    def test_keep_only(self):
        plan = {"selected_features": ["age", "bmi", "smoker", "height"], "dropped_features": []}
        result = self._apply(plan, "keep only features: age, bmi")
        assert set(result["selected_features"]) == {"age", "bmi"}

    def test_balance_strategy(self):
        plan = {"selected_features": [], "dropped_features": []}
        result = self._apply(plan, "use oversample")
        assert result["balance_strategy"] == "oversample"

    def test_test_ratio(self):
        plan = {"selected_features": [], "dropped_features": []}
        result = self._apply(plan, "test_ratio: 0.3")
        assert result["test_ratio"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# _repair_json (plan.py)
# ---------------------------------------------------------------------------

class TestRepairJSON:
    """Tests for the _repair_json bracket/brace repair utility."""

    def _repair(self, s):
        from auto_llm_predictor.nodes.plan import _repair_json
        return _repair_json(s)

    def test_valid_json_passthrough(self):
        """Valid JSON should be returned unchanged."""
        valid = '{"a": 1, "b": [1, 2, 3]}'
        assert self._repair(valid) == valid
        json.loads(self._repair(valid))  # should not raise

    def test_unclosed_array(self):
        """An unclosed array should get ] appended."""
        broken = '{"a": ["x", "reasoning": "ok"}'
        repaired = self._repair(broken)
        result = json.loads(repaired)
        assert "a" in result

    def test_unclosed_array_real_world(self):
        """The exact failure pattern from the user report:
        data_cleaning_steps opens [ but never closes it."""
        broken = (
            '{"selected_features": ["f1", "f2"], '
            '"data_cleaning_steps": ["step one; step two", '
            '"reasoning": "everything is fine"}'
        )
        repaired = self._repair(broken)
        result = json.loads(repaired)
        assert "selected_features" in result

    def test_unclosed_brace(self):
        """An unclosed brace should get } appended."""
        broken = '{"a": {"nested": 1}'
        repaired = self._repair(broken)
        result = json.loads(repaired)
        assert result["a"]["nested"] == 1

    def test_strings_with_brackets_ignored(self):
        """Brackets inside quoted strings should not be counted."""
        valid = '{"msg": "use [these] brackets {here}"}'
        assert self._repair(valid) == valid
        json.loads(self._repair(valid))

    def test_trailing_comma_stripped(self):
        """Trailing comma before appended closers should be removed."""
        broken = '{"a": [1, 2,'
        repaired = self._repair(broken)
        result = json.loads(repaired)
        assert result["a"] == [1, 2]
