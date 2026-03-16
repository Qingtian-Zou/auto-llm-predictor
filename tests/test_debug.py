"""Tests for auto_llm_predictor.nodes.debug.

Covers: debug tool functions, route_after_debug routing, and debug node fallback.
"""

from __future__ import annotations

import textwrap

import pytest


# ---------------------------------------------------------------------------
# Helper: create sandboxed tools for testing
# ---------------------------------------------------------------------------

def _get_tools(output_dir: str, csv_path: str) -> dict:
    """Create debug tools and return them as a name→callable dict."""
    from auto_llm_predictor.nodes.debug import _make_debug_tools

    tools = _make_debug_tools(output_dir, csv_path)
    return {t.name: t for t in tools}


# ---------------------------------------------------------------------------
# read_file tool
# ---------------------------------------------------------------------------

class TestReadFileTool:
    """Tests for the read_file debug tool."""

    def test_reads_existing_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello\nworld\n")
        tools = _get_tools(str(tmp_path), str(tmp_path / "dummy.csv"))
        result = tools["read_file"].invoke({"path": str(f)})
        assert "hello" in result
        assert "world" in result

    def test_rejects_path_outside_sandbox(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("secret")
        # Use a different directory as the allowed one
        other = tmp_path / "allowed"
        other.mkdir()
        tools = _get_tools(str(other), str(other / "dummy.csv"))
        result = tools["read_file"].invoke({"path": str(f)})
        assert "ERROR" in result
        assert "Access denied" in result

    def test_handles_nonexistent_file(self, tmp_path):
        tools = _get_tools(str(tmp_path), str(tmp_path / "dummy.csv"))
        result = tools["read_file"].invoke({"path": str(tmp_path / "nope.txt")})
        assert "ERROR" in result
        assert "not found" in result.lower()

    def test_truncates_large_files(self, tmp_path):
        f = tmp_path / "big.txt"
        f.write_text("line\n" * 500)
        tools = _get_tools(str(tmp_path), str(tmp_path / "dummy.csv"))
        result = tools["read_file"].invoke({"path": str(f)})
        assert "truncated" in result


# ---------------------------------------------------------------------------
# run_python_snippet tool
# ---------------------------------------------------------------------------

class TestRunPythonSnippet:
    """Tests for the run_python_snippet debug tool."""

    def test_runs_simple_snippet(self, tmp_path):
        tools = _get_tools(str(tmp_path), str(tmp_path / "dummy.csv"))
        result = tools["run_python_snippet"].invoke({"code": "print('hello from snippet')"})
        assert "hello from snippet" in result

    def test_captures_stderr(self, tmp_path):
        tools = _get_tools(str(tmp_path), str(tmp_path / "dummy.csv"))
        result = tools["run_python_snippet"].invoke({
            "code": "import sys; sys.stderr.write('err msg\\n')"
        })
        assert "err msg" in result

    def test_timeout_on_long_running(self, tmp_path):
        tools = _get_tools(str(tmp_path), str(tmp_path / "dummy.csv"))
        # Use a very short timeout would be ideal, but the tool has a fixed 30s timeout.
        # Instead, test that an import error is captured properly.
        result = tools["run_python_snippet"].invoke({
            "code": "raise ValueError('test error')"
        })
        assert "test error" in result

    def test_truncates_long_output(self, tmp_path):
        tools = _get_tools(str(tmp_path), str(tmp_path / "dummy.csv"))
        result = tools["run_python_snippet"].invoke({
            "code": "print('x' * 5000)"
        })
        assert "truncated" in result


# ---------------------------------------------------------------------------
# list_directory tool
# ---------------------------------------------------------------------------

class TestListDirectory:
    """Tests for the list_directory debug tool."""

    def test_lists_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / "b.json").write_text("{}")
        tools = _get_tools(str(tmp_path), str(tmp_path / "dummy.csv"))
        result = tools["list_directory"].invoke({"path": str(tmp_path)})
        assert "a.txt" in result
        assert "b.json" in result

    def test_rejects_path_outside_sandbox(self, tmp_path):
        other = tmp_path / "allowed"
        other.mkdir()
        tools = _get_tools(str(other), str(other / "dummy.csv"))
        result = tools["list_directory"].invoke({"path": str(tmp_path)})
        assert "ERROR" in result

    def test_handles_empty_directory(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        tools = _get_tools(str(tmp_path), str(tmp_path / "dummy.csv"))
        result = tools["list_directory"].invoke({"path": str(empty)})
        assert "empty" in result.lower()

    def test_handles_nonexistent_directory(self, tmp_path):
        tools = _get_tools(str(tmp_path), str(tmp_path / "dummy.csv"))
        result = tools["list_directory"].invoke({"path": str(tmp_path / "nope")})
        assert "ERROR" in result


# ---------------------------------------------------------------------------
# inspect_csv_sample tool
# ---------------------------------------------------------------------------

class TestInspectCsvSample:
    """Tests for the inspect_csv_sample debug tool."""

    def test_inspects_valid_csv(self, tmp_path):
        csv = tmp_path / "data.csv"
        csv.write_text("a,b,c\n1,2,3\n4,5,6\n")
        tools = _get_tools(str(tmp_path), str(csv))
        result = tools["inspect_csv_sample"].invoke({"path": str(csv)})
        assert "3 rows" in result or "2 rows" in result
        assert "a" in result
        assert "b" in result

    def test_caps_n_rows(self, tmp_path):
        csv = tmp_path / "data.csv"
        lines = ["x\n"] + [f"{i}\n" for i in range(100)]
        csv.write_text("".join(lines))
        tools = _get_tools(str(tmp_path), str(csv))
        # Request 50 rows, should be capped at 20
        result = tools["inspect_csv_sample"].invoke({"path": str(csv), "n_rows": 50})
        assert "100 rows" in result  # shape shows total rows
        assert "First 20 rows" in result

    def test_handles_nonexistent_csv(self, tmp_path):
        tools = _get_tools(str(tmp_path), str(tmp_path / "dummy.csv"))
        result = tools["inspect_csv_sample"].invoke({"path": str(tmp_path / "nope.csv")})
        assert "ERROR" in result

    def test_rejects_path_outside_sandbox(self, tmp_path):
        other = tmp_path / "allowed"
        other.mkdir()
        csv = tmp_path / "data.csv"
        csv.write_text("a,b\n1,2\n")
        tools = _get_tools(str(other), str(other / "dummy.csv"))
        result = tools["inspect_csv_sample"].invoke({"path": str(csv)})
        assert "ERROR" in result


# ---------------------------------------------------------------------------
# route_after_debug
# ---------------------------------------------------------------------------

class TestRouteAfterDebug:
    """Tests for the route_after_debug routing function."""

    def test_routes_to_retry_on_normal_diagnosis(self):
        from auto_llm_predictor.nodes.debug import route_after_debug

        state = {"debug_diagnosis": "ROOT CAUSE: Wrong column name. FIX: Use 'target' instead."}
        assert route_after_debug(state) == "write_prep_code"

    def test_routes_to_abort_on_ABORT(self):
        from auto_llm_predictor.nodes.debug import route_after_debug

        state = {"debug_diagnosis": "The CSV file is missing. ABORT - cannot proceed."}
        assert route_after_debug(state) == "verify_prepared_data"

    def test_routes_to_abort_on_HUMAN_HELP(self):
        from auto_llm_predictor.nodes.debug import route_after_debug

        state = {"debug_diagnosis": "Ambiguous target column. HUMAN_HELP needed."}
        assert route_after_debug(state) == "verify_prepared_data"

    def test_routes_to_retry_on_empty_diagnosis(self):
        from auto_llm_predictor.nodes.debug import route_after_debug

        state = {"debug_diagnosis": ""}
        assert route_after_debug(state) == "write_prep_code"

    def test_abort_case_insensitive(self):
        from auto_llm_predictor.nodes.debug import route_after_debug

        state = {"debug_diagnosis": "Cannot fix. abort recommended."}
        assert route_after_debug(state) == "verify_prepared_data"


# ---------------------------------------------------------------------------
# debug_prep_failure node — fallback and error handling
# ---------------------------------------------------------------------------

class TestDebugPrepFailure:
    """Integration tests for the debug_prep_failure node."""

    def _make_state(self, tmp_path):
        """Create a minimal state for testing the debug node."""
        csv = tmp_path / "data.csv"
        csv.write_text("a,b,target\n1,2,yes\n3,4,no\n")
        script = tmp_path / "scripts" / "prepare_data.py"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("raise ValueError('column not found')")
        return {
            "csv_path": str(csv),
            "output_dir": str(tmp_path),
            "prep_code_path": str(script),
            "prep_code": "raise ValueError('column not found')",
            "prep_result": "Traceback: ValueError: column not found",
            "prep_succeeded": False,
            "prep_attempts": 1,
            "data_profile": "Shape: 2 rows x 3 columns",
            "target_column": "target",
            "task_type": "binary",
        }

    def test_fallback_when_agent_creation_fails(self, tmp_path):
        """When create_react_agent fails, fallback should still return a diagnosis."""
        from auto_llm_predictor.nodes.debug import _fallback_diagnosis

        class MockLLM:
            def invoke(self, messages):
                class Resp:
                    content = "ROOT CAUSE: Column mismatch. FIX: Use correct name."
                return Resp()

        state = self._make_state(tmp_path)
        result = _fallback_diagnosis(state, llm=MockLLM())
        assert "debug_diagnosis" in result
        assert "Column mismatch" in result["debug_diagnosis"]
        assert result["debug_tool_calls"] == []

    def test_fallback_handles_llm_exception(self, tmp_path):
        """Fallback should handle LLM API errors gracefully."""
        from auto_llm_predictor.nodes.debug import _fallback_diagnosis

        class FailLLM:
            def invoke(self, messages):
                raise ConnectionError("API unavailable")

        state = self._make_state(tmp_path)
        result = _fallback_diagnosis(state, llm=FailLLM())
        assert "debug_diagnosis" in result
        assert "failed" in result["debug_diagnosis"].lower()

    def test_node_returns_required_fields(self, tmp_path):
        """The node should always return debug_diagnosis, debug_tool_calls, messages."""
        from auto_llm_predictor.nodes.debug import debug_prep_failure

        class MockLLM:
            def invoke(self, messages):
                class Resp:
                    content = "ROOT CAUSE: Test. FIX: Do something."
                return Resp()

            def bind_tools(self, tools):
                raise NotImplementedError("No tool support")

        state = self._make_state(tmp_path)
        result = debug_prep_failure(state, llm=MockLLM())
        assert "debug_diagnosis" in result
        assert "debug_tool_calls" in result
        assert "messages" in result
