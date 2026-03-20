"""Tests for auto_llm_predictor.nodes.explain.

Covers: _build_prompt helper (used by standalone XAI).
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    """Tests for the _build_prompt helper."""

    def _build(self, entry, tokenizer=None):
        from auto_llm_predictor.nodes.explain import _build_prompt
        return _build_prompt(entry, tokenizer)

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

    def test_no_tokenizer_returns_raw(self):
        """Without a tokenizer, the raw Alpaca prompt is returned."""
        result = self._build({"instruction": "Predict", "input": "x: 1"})
        assert result == "Predict\n\nx: 1"

    def test_tokenizer_without_chat_template_returns_raw(self):
        """A tokenizer without chat_template falls back to raw prompt."""
        class NoTemplateTok:
            chat_template = None
        result = self._build({"instruction": "Predict", "input": "x: 1"}, NoTemplateTok())
        assert result == "Predict\n\nx: 1"

    def test_tokenizer_with_chat_template_wraps_prompt(self):
        """When tokenizer has a chat_template, the prompt is wrapped."""
        class MockTokenizer:
            chat_template = "mock"
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                content = messages[0]["content"]
                return f"<|user|>{content}<|assistant|>"

        result = self._build(
            {"instruction": "Predict", "input": "x: 1"}, MockTokenizer(),
        )
        assert result == "<|user|>Predict\n\nx: 1<|assistant|>"
