"""Tests for auto_llm_predictor.nodes.plan.

Covers: _apply_feedback_overrides, _repair_json, and _validated_target_mapping.
"""

from __future__ import annotations

import json


# ---------------------------------------------------------------------------
# _apply_feedback_overrides
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


# ---------------------------------------------------------------------------
# _repair_json
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
# _validated_target_mapping
# ---------------------------------------------------------------------------

class TestValidatedTargetMapping:
    """Tests for _validated_target_mapping (multiclass preservation)."""

    def test_rejects_class_reductions(self):
        """_validated_target_mapping rejects class reductions."""
        from auto_llm_predictor.nodes.plan import _validated_target_mapping

        original = {"0": "A", "1": "B", "2": "C"}
        reduced = {"0": "A", "1": "B"}
        assert _validated_target_mapping(reduced, original) == original

    def test_accepts_same_or_more(self):
        from auto_llm_predictor.nodes.plan import _validated_target_mapping

        original = {"0": "A", "1": "B"}
        refined = {"0": "ClassA", "1": "ClassB"}
        assert _validated_target_mapping(refined, original) == refined

    def test_accepts_more_classes(self):
        from auto_llm_predictor.nodes.plan import _validated_target_mapping

        original = {"0": "A", "1": "B"}
        expanded = {"0": "A", "1": "B", "2": "C"}
        assert _validated_target_mapping(expanded, original) == expanded

    def test_none_falls_back(self):
        from auto_llm_predictor.nodes.plan import _validated_target_mapping

        original = {"0": "A", "1": "B"}
        assert _validated_target_mapping(None, original) == original

    def test_empty_falls_back(self):
        from auto_llm_predictor.nodes.plan import _validated_target_mapping

        original = {"0": "A", "1": "B"}
        assert _validated_target_mapping({}, original) == original
