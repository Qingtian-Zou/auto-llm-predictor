"""Tests for auto_llm_predictor.nodes.review.

Covers: _coerce_value, _parse_overrides, and _build_edit_feedback.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# _coerce_value
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


# ---------------------------------------------------------------------------
# _parse_overrides
# ---------------------------------------------------------------------------

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
# _build_edit_feedback
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
