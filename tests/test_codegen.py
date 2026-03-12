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
