"""Tests for code generation related utilities.

Covers: markdown fence stripping and balance script timeout.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Markdown fence stripping edge cases
# ---------------------------------------------------------------------------

def _strip_fences(code: str) -> str:
    """Reproduce the fence-stripping logic used in codegen/balance/explore nodes."""
    if code.startswith("```"):
        parts = code.split("\n", 1)
        code = parts[1] if len(parts) > 1 else ""
        if code.endswith("```"):
            code = code[: code.rfind("```")]
        code = code.strip()
    return code


class TestMarkdownFenceStripping:
    """Tests for the markdown fence stripping logic in codegen/balance/explore."""

    def test_normal_python_fence(self):
        code = "```python\nprint('hello')\n```"
        assert _strip_fences(code) == "print('hello')"

    def test_bare_fence(self):
        code = "```\nprint('hello')\n```"
        assert _strip_fences(code) == "print('hello')"

    def test_single_line_fence_no_crash(self):
        """A single-line ``` should not IndexError."""
        code = "```"
        result = _strip_fences(code)
        assert result == ""

    def test_single_line_fence_with_language(self):
        """```python with no newline should not IndexError."""
        code = "```python"
        result = _strip_fences(code)
        assert result == ""

    def test_no_fences(self):
        code = "print('hello')"
        assert _strip_fences(code) == "print('hello')"

    def test_fence_with_no_closing(self):
        code = "```python\nprint('hello')"
        assert _strip_fences(code) == "print('hello')"

    def test_empty_body(self):
        code = "```python\n```"
        assert _strip_fences(code) == ""


# ---------------------------------------------------------------------------
# Codegen prompt: structured CLI + transformer persistence
# ---------------------------------------------------------------------------

class TestCodegenPromptCli:
    """The codegen prompt must instruct the LLM to expose --predict-only,
    --input-csv, --test-csv, --output-dir, and to persist transformers.pkl."""

    def _format(self, **overrides):
        from auto_llm_predictor.prompts.codegen import format_codegen_prompt

        defaults = dict(
            csv_path="/tmp/train.csv",
            data_profile="profile",
            target_column="t",
            task_type="binary",
            target_mapping={"0": "no", "1": "yes"},
            selected_features=["a"],
            instruction_template="i",
            input_format="f",
            output_format="o",
            data_cleaning_steps=[],
            output_data_dir="/tmp/data",
            test_csv_path="/tmp/test.csv",
        )
        defaults.update(overrides)
        return format_codegen_prompt(**defaults)

    def test_prompt_mentions_predict_only_flag(self):
        prompt = self._format()
        assert "--predict-only" in prompt

    def test_prompt_mentions_required_cli_flags(self):
        prompt = self._format()
        for flag in ("--input-csv", "--test-csv", "--output-dir"):
            assert flag in prompt, f"missing {flag} in codegen prompt"

    def test_prompt_mentions_transformers_pickle(self):
        prompt = self._format()
        assert "transformers.pkl" in prompt

    def test_prompt_emphasizes_fit_on_train(self):
        prompt = self._format()
        # The prompt should explicitly forbid refitting on test data.
        assert ".fit(" in prompt or "fit_transform" in prompt
        # And mention applying to test
        assert "transform" in prompt.lower()

    def test_prompt_handles_no_test_csv(self):
        """When no test CSV is supplied, the prompt should say so explicitly
        and still require --predict-only / transformers.pkl."""
        prompt = self._format(test_csv_path="")
        assert "--predict-only" in prompt
        assert "transformers.pkl" in prompt


# ---------------------------------------------------------------------------
# Balance timeout constant
# ---------------------------------------------------------------------------

class TestBalanceTimeout:
    """Verify the balance script timeout was increased."""

    def test_balance_timeout_is_at_least_600(self):
        """The execute_balance_code function should use a timeout >= 600s."""
        import inspect
        import re
        from auto_llm_predictor.nodes.balance import execute_balance_code

        source = inspect.getsource(execute_balance_code)
        # Find timeout= argument — should be at least 600
        match = re.search(r"timeout=(\d+)", source)
        assert match, "No timeout= found in execute_balance_code"
        assert int(match.group(1)) >= 600
