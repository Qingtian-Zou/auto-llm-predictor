"""Tests for the verbose LLM wrapper.

Covers: VerboseLLM delegation, stderr output when verbose, bind_tools re-wrapping.
"""

from __future__ import annotations

import sys
from io import StringIO
from unittest.mock import MagicMock

from auto_llm_predictor.verbose import VerboseLLM


def _make_mock_llm(response_content: str = "mock response") -> MagicMock:
    """Create a mock LLM that returns a fixed response from invoke()."""
    llm = MagicMock()
    response = MagicMock()
    response.content = response_content
    llm.invoke.return_value = response
    llm.some_attr = "test_value"
    return llm


class TestVerboseLLMDelegation:
    """VerboseLLM should transparently delegate to the wrapped LLM."""

    def test_invoke_delegates(self):
        llm = _make_mock_llm("hello")
        wrapper = VerboseLLM(llm, label="TEST")

        messages = [MagicMock(type="human", content="prompt")]
        result = wrapper.invoke(messages)

        llm.invoke.assert_called_once_with(messages)
        assert result.content == "hello"

    def test_getattr_delegates(self):
        llm = _make_mock_llm()
        wrapper = VerboseLLM(llm, label="TEST")
        assert wrapper.some_attr == "test_value"

    def test_bind_tools_returns_verbose_wrapper(self):
        llm = _make_mock_llm()
        bound = MagicMock()
        llm.bind_tools.return_value = bound

        wrapper = VerboseLLM(llm, label="TEST")
        result = wrapper.bind_tools(["tool1"])

        llm.bind_tools.assert_called_once_with(["tool1"])
        assert isinstance(result, VerboseLLM)


class TestVerboseLLMOutput:
    """VerboseLLM should print prompts and responses to stderr."""

    def test_invoke_prints_to_stderr(self):
        llm = _make_mock_llm("the answer is 42")
        wrapper = VerboseLLM(llm, label="AGENT")

        messages = [
            MagicMock(type="system", content="You are helpful."),
            MagicMock(type="human", content="What is 6*7?"),
        ]

        old_stderr = sys.stderr
        captured = StringIO()
        sys.stderr = captured
        try:
            wrapper.invoke(messages)
        finally:
            sys.stderr = old_stderr

        output = captured.getvalue()

        # Request section
        assert "AGENT REQUEST" in output
        assert "[SYSTEM]" in output
        assert "You are helpful." in output
        assert "[HUMAN]" in output
        assert "What is 6*7?" in output

        # Response section
        assert "AGENT RESPONSE" in output
        assert "the answer is 42" in output

    def test_invoke_with_kwargs(self):
        """Extra kwargs should be forwarded to the underlying LLM."""
        llm = _make_mock_llm("ok")
        wrapper = VerboseLLM(llm, label="TEST")

        messages = [MagicMock(type="human", content="hi")]

        old_stderr = sys.stderr
        sys.stderr = StringIO()
        try:
            wrapper.invoke(messages, temperature=0.5)
        finally:
            sys.stderr = old_stderr

        llm.invoke.assert_called_once_with(messages, temperature=0.5)


class TestBuildGraphVerbose:
    """Verify that build_graph accepts the verbose parameter."""

    def test_verbose_parameter_exists(self):
        """build_graph should accept verbose as a keyword argument."""
        import inspect
        from auto_llm_predictor.graph import build_graph

        sig = inspect.signature(build_graph)
        assert "verbose" in sig.parameters
        param = sig.parameters["verbose"]
        assert param.default is False
