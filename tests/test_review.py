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

    def _build(self, state_updates, edited_plan,
               original_state=None, original_plan=None):
        from auto_llm_predictor.nodes.review import _build_edit_feedback
        return _build_edit_feedback(
            state_updates, edited_plan,
            original_state or {}, original_plan or {},
        )

    # --- Existing behaviour (empty originals → everything looks changed) ---

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

    # --- Diff behaviour (only changed fields reported) ---

    def test_unchanged_state_field_excluded(self):
        """A state field identical to the original should NOT appear as changed."""
        orig = {"target_mapping": {"0": "A", "1": "B"}}
        edits = {"target_mapping": {"0": "A", "1": "B"}}
        result = self._build(edits, {}, original_state=orig)
        # target_mapping value unchanged — must not be listed as a change
        assert "- target_mapping:" not in result

    def test_changed_state_field_included(self):
        """A state field that differs from the original SHOULD appear."""
        orig = {"target_mapping": {"0": "A"}}
        edits = {"target_mapping": {"0": "B"}}
        result = self._build(edits, {}, original_state=orig)
        assert "target_mapping" in result
        assert '"B"' in result

    def test_unchanged_plan_field_excluded(self):
        """An unchanged plan field should not be in the changed section
        but should appear in the 'adapt' section."""
        orig_plan = {"instruction_template": "Predict X"}
        edited_plan = {"instruction_template": "Predict X"}
        # Need at least one change so we don't hit the no-changes fallback
        result = self._build(
            {"selected_features": ["age"]}, edited_plan,
            original_state={}, original_plan=orig_plan,
        )
        assert "- instruction_template:" not in result
        assert "instruction_template" in result  # in the adapt list

    def test_changed_plan_field_included(self):
        orig_plan = {"instruction_template": "Predict X"}
        edited_plan = {"instruction_template": "Predict Y"}
        result = self._build({}, edited_plan, original_plan=orig_plan)
        assert "- instruction_template:" in result
        assert "Predict Y" in result

    def test_feature_move_scenario(self):
        """Moving a feature from selected to dropped should report both as
        changed and list data_cleaning_steps in the adapt section."""
        orig_state = {
            "selected_features": ["age", "bmi", "smoker"],
            "dropped_features": ["id"],
        }
        edits_state = {
            "selected_features": ["age", "bmi"],
            "dropped_features": ["id", "smoker"],
        }
        orig_plan = {"data_cleaning_steps": ["Fill missing age", "Encode smoker"]}
        edited_plan = {"data_cleaning_steps": ["Fill missing age", "Encode smoker"]}
        result = self._build(
            edits_state, edited_plan,
            original_state=orig_state, original_plan=orig_plan,
        )
        assert "selected_features" in result
        assert "dropped_features" in result
        # data_cleaning_steps unchanged → should be in adapt list, not changed
        assert "- data_cleaning_steps:" not in result
        assert "data_cleaning_steps" in result

    def test_no_changes_fallback(self):
        """If the user re-submits identical JSON, a valid fallback is returned."""
        orig_state = {"selected_features": ["age"]}
        orig_plan = {"instruction_template": "X"}
        result = self._build(
            {"selected_features": ["age"]},
            {"instruction_template": "X"},
            original_state=orig_state,
            original_plan=orig_plan,
        )
        assert isinstance(result, str)
        assert len(result) > 0
        assert "without meaningful changes" in result
