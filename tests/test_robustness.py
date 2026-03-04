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
# _build_edit_feedback (review.py)
# ---------------------------------------------------------------------------

class TestBuildEditFeedback:
    """Tests for _build_edit_feedback — concrete feedback for re-planning."""

    def _build(self, state_updates, edited_plan):
        from auto_llm_predictor.nodes.review import _build_edit_feedback
        return _build_edit_feedback(state_updates, edited_plan)

    def test_includes_target_mapping(self):
        updates = {"target_mapping": {"0": "No Response", "1": "Response"}}
        result = self._build(updates, {})
        assert '"No Response"' in result
        assert '"Response"' in result
        assert "target_mapping" in result

    def test_includes_selected_features(self):
        updates = {"selected_features": ["age", "bmi"]}
        result = self._build(updates, {})
        assert '"age"' in result
        assert '"bmi"' in result
        assert "selected_features" in result

    def test_includes_plan_level_keys(self):
        updates = {}
        plan = {"instruction_template": "Predict the response", "balance_strategy": "oversample"}
        result = self._build(updates, plan)
        assert "instruction_template" in result
        assert "Predict the response" in result
        assert "balance_strategy" in result
        assert "oversample" in result

    def test_no_clarifying_questions_instruction(self):
        result = self._build({"target_mapping": {"0": "A"}}, {})
        assert "do NOT ask clarifying questions" in result

    def test_empty_updates_still_valid(self):
        result = self._build({}, {})
        assert isinstance(result, str)
        assert len(result) > 0


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


# ---------------------------------------------------------------------------
# determine_cutoff_len (nodes/cutoff.py)
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


# ---------------------------------------------------------------------------
# XAI — run_xai / check_xai_enabled (nodes/explain.py)
# ---------------------------------------------------------------------------

class TestCheckXaiEnabled:
    """Tests for the check_xai_enabled routing function."""

    def _check(self, state):
        from auto_llm_predictor.nodes.explain import check_xai_enabled
        return check_xai_enabled(state)

    def test_enabled(self):
        assert self._check({"xai_enabled": True}) == "run_xai"

    def test_disabled(self):
        assert self._check({"xai_enabled": False}) == "__end__"

    def test_missing_key(self):
        assert self._check({}) == "__end__"


class TestRunXaiSkips:
    """Tests for run_xai skip guards (no LLM/GPU needed)."""

    def _run(self, state):
        from auto_llm_predictor.nodes.explain import run_xai
        return run_xai(state)

    def test_skipped_when_disabled(self):
        result = self._run({"xai_enabled": False})
        assert result == {}

    def test_skipped_when_no_adapter(self, tmp_path):
        result = self._run({
            "xai_enabled": True,
            "adapter_path": str(tmp_path / "nonexistent"),
            "finetune_succeeded": True,
        })
        assert result["xai_report_path"] == ""

    def test_skipped_when_finetune_failed(self, tmp_path):
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        result = self._run({
            "xai_enabled": True,
            "adapter_path": str(adapter),
            "finetune_succeeded": False,
        })
        assert result["xai_report_path"] == ""

    def test_skipped_when_no_test_data(self, tmp_path):
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        result = self._run({
            "xai_enabled": True,
            "adapter_path": str(adapter),
            "finetune_succeeded": True,
            "test_data_path": str(tmp_path / "nonexistent.json"),
        })
        assert result["xai_report_path"] == ""

    def test_skipped_when_test_data_empty(self, tmp_path):
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        test_json = tmp_path / "test.json"
        test_json.write_text("[]")
        result = self._run({
            "xai_enabled": True,
            "adapter_path": str(adapter),
            "finetune_succeeded": True,
            "test_data_path": str(test_json),
        })
        assert result["xai_report_path"] == ""


class TestBuildPrompt:
    """Tests for the _build_prompt helper."""

    def _build(self, entry):
        from auto_llm_predictor.nodes.explain import _build_prompt
        return _build_prompt(entry)

    def test_instruction_only(self):
        result = self._build({"instruction": "Predict X", "input": ""})
        assert result == "Predict X"

    def test_instruction_with_input(self):
        result = self._build({"instruction": "Predict X", "input": "age: 50"})
        assert "Predict X" in result
        assert "age: 50" in result
        assert "\n\n" in result

    def test_empty_entry(self):
        result = self._build({})
        assert result == ""
