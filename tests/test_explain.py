"""Tests for auto_llm_predictor.nodes.explain.

Covers: check_xai_enabled, run_xai skip guards, and _build_prompt.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# check_xai_enabled
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


# ---------------------------------------------------------------------------
# run_xai — skip guards
# ---------------------------------------------------------------------------

class TestRunXaiSkips:
    """Tests for run_xai skip guards (no LLM/GPU needed)."""

    def _run(self, state):
        from auto_llm_predictor.nodes.explain import run_xai
        config = {"configurable": {}}
        return run_xai(state, config)

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


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------

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
