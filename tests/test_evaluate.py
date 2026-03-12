"""Tests for auto_llm_predictor.nodes.evaluate.

Covers: _extract_label and _compute_metrics (including multiclass preservation).
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# _extract_label
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
# Multiclass preservation — _compute_metrics
# ---------------------------------------------------------------------------

class TestMulticlassPreservation:
    """Tests for the multiclass-to-binary conversion bug fix."""

    def test_evaluate_uses_task_type_for_multiclass(self):
        """_compute_metrics uses macro_f1 when task_type='multiclass' even with 2 labels."""
        from auto_llm_predictor.nodes.evaluate import _compute_metrics

        y_true = ["A", "B", "A", "B"]
        y_pred = ["A", "B", "B", "A"]
        labels = ["A", "B"]
        result = _compute_metrics(y_true, y_pred, labels, task_type="multiclass")
        assert "macro_f1" in result
        assert "f1" not in result

    def test_evaluate_binary_task_type(self):
        from auto_llm_predictor.nodes.evaluate import _compute_metrics

        y_true = ["A", "B", "A", "B"]
        y_pred = ["A", "B", "B", "A"]
        labels = ["A", "B"]
        result = _compute_metrics(y_true, y_pred, labels, task_type="binary")
        assert "f1" in result
        assert "macro_f1" not in result

    def test_evaluate_fallback_when_no_task_type(self):
        """Without task_type, falls back to len(labels) heuristic."""
        from auto_llm_predictor.nodes.evaluate import _compute_metrics

        y_true = ["A", "B", "C", "A"]
        y_pred = ["A", "B", "C", "C"]
        labels = ["A", "B", "C"]
        result = _compute_metrics(y_true, y_pred, labels)
        assert "macro_f1" in result
        assert "f1" not in result
