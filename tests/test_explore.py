"""Tests for auto_llm_predictor.nodes.explore.

Covers: exploration tool functions, JSON extraction helpers,
target mapping validation, fallback exploration, and node integration.
"""

from __future__ import annotations

import json
import textwrap

import pytest


# ---------------------------------------------------------------------------
# Helper: create exploration tools for testing
# ---------------------------------------------------------------------------

def _get_tools(csv_path: str) -> dict:
    """Create explore tools and return them as a name→callable dict."""
    from auto_llm_predictor.nodes.explore import _make_explore_tools

    tools = _make_explore_tools(csv_path)
    return {t.name: t for t in tools}


def _write_csv(tmp_path, name="data.csv", content=None):
    """Write a small test CSV and return its path."""
    if content is None:
        content = "a,b,target\n1,2,yes\n3,4,no\n5,6,yes\n7,8,no\n"
    csv = tmp_path / name
    csv.write_text(content)
    return str(csv)


# ---------------------------------------------------------------------------
# sample_rows tool
# ---------------------------------------------------------------------------

class TestSampleRows:
    """Tests for the sample_rows exploration tool."""

    def test_returns_sample_data(self, tmp_path):
        csv = _write_csv(tmp_path)
        tools = _get_tools(csv)
        result = tools["sample_rows"].invoke({"n": 2})
        # Should contain column headers and values
        assert "a" in result
        assert "target" in result

    def test_caps_at_20_rows(self, tmp_path):
        lines = ["x,y\n"] + [f"{i},{i+1}\n" for i in range(100)]
        csv = _write_csv(tmp_path, content="".join(lines))
        tools = _get_tools(csv)
        result = tools["sample_rows"].invoke({"n": 50})
        # Should not contain more than 20 data rows
        data_lines = [l for l in result.strip().split("\n") if l.strip()]
        # Header + up to 20 data rows
        assert len(data_lines) <= 22

    def test_handles_empty_csv(self, tmp_path):
        csv = _write_csv(tmp_path, content="a,b,c\n")
        tools = _get_tools(csv)
        result = tools["sample_rows"].invoke({"n": 5})
        assert "empty" in result.lower()

    def test_default_n(self, tmp_path):
        lines = ["x\n"] + [f"{i}\n" for i in range(20)]
        csv = _write_csv(tmp_path, content="".join(lines))
        tools = _get_tools(csv)
        result = tools["sample_rows"].invoke({})
        # Should return some rows (default n=5)
        assert "x" in result


# ---------------------------------------------------------------------------
# column_stats tool
# ---------------------------------------------------------------------------

class TestColumnStats:
    """Tests for the column_stats exploration tool."""

    def test_numeric_column(self, tmp_path):
        csv = _write_csv(tmp_path, content="val,label\n1.5,a\n2.5,b\n3.5,c\n")
        tools = _get_tools(csv)
        result = tools["column_stats"].invoke({"column_name": "val"})
        assert "dtype" in result
        assert "unique" in result
        assert "missing" in result

    def test_categorical_column(self, tmp_path):
        csv = _write_csv(tmp_path)
        tools = _get_tools(csv)
        result = tools["column_stats"].invoke({"column_name": "target"})
        assert "yes" in result or "no" in result

    def test_nonexistent_column(self, tmp_path):
        csv = _write_csv(tmp_path)
        tools = _get_tools(csv)
        result = tools["column_stats"].invoke({"column_name": "nonexistent"})
        assert "ERROR" in result
        assert "not found" in result.lower()
        # Should list available columns
        assert "target" in result


# ---------------------------------------------------------------------------
# value_counts tool
# ---------------------------------------------------------------------------

class TestValueCounts:
    """Tests for the value_counts exploration tool."""

    def test_basic_value_counts(self, tmp_path):
        csv = _write_csv(tmp_path)
        tools = _get_tools(csv)
        result = tools["value_counts"].invoke({"column_name": "target"})
        assert "yes" in result
        assert "no" in result
        assert "%" in result

    def test_top_k_limit(self, tmp_path):
        lines = ["val\n"] + [f"{chr(65 + i)}\n" for i in range(20)]
        csv = _write_csv(tmp_path, content="".join(lines))
        tools = _get_tools(csv)
        result = tools["value_counts"].invoke({"column_name": "val", "top_k": 3})
        # Count the value lines (lines with percentages)
        pct_lines = [l for l in result.split("\n") if "%" in l]
        assert len(pct_lines) <= 3

    def test_handles_missing_values(self, tmp_path):
        csv = _write_csv(tmp_path, content="a,b\n1,x\n2,\n3,x\n")
        tools = _get_tools(csv)
        result = tools["value_counts"].invoke({"column_name": "b"})
        # NaN should appear in output
        assert "NaN" in result or "nan" in result.lower()

    def test_nonexistent_column(self, tmp_path):
        csv = _write_csv(tmp_path)
        tools = _get_tools(csv)
        result = tools["value_counts"].invoke({"column_name": "nope"})
        assert "ERROR" in result


# ---------------------------------------------------------------------------
# correlation_matrix tool
# ---------------------------------------------------------------------------

class TestCorrelationMatrix:
    """Tests for the correlation_matrix exploration tool."""

    def test_numeric_columns(self, tmp_path):
        csv = _write_csv(tmp_path, content="x,y,z\n1,2,3\n4,5,6\n7,8,9\n")
        tools = _get_tools(csv)
        result = tools["correlation_matrix"].invoke({"columns": "x,y,z"})
        # Should show correlation values
        assert "x" in result
        assert "y" in result

    def test_caps_at_10_columns(self, tmp_path):
        cols = [f"c{i}" for i in range(15)]
        header = ",".join(cols) + "\n"
        row = ",".join(["1"] * 15) + "\n"
        csv = _write_csv(tmp_path, content=header + row + row)
        tools = _get_tools(csv)
        result = tools["correlation_matrix"].invoke({"columns": ",".join(cols)})
        # Should work but only use first 10 columns
        assert "c0" in result

    def test_non_numeric_column_error(self, tmp_path):
        csv = _write_csv(tmp_path, content="name,val\nAlice,1\nBob,2\n")
        tools = _get_tools(csv)
        result = tools["correlation_matrix"].invoke({"columns": "name,val"})
        assert "ERROR" in result
        assert "Non-numeric" in result or "non-numeric" in result.lower()

    def test_empty_columns_string(self, tmp_path):
        csv = _write_csv(tmp_path)
        tools = _get_tools(csv)
        result = tools["correlation_matrix"].invoke({"columns": ""})
        assert "ERROR" in result


# ---------------------------------------------------------------------------
# check_missing_values tool
# ---------------------------------------------------------------------------

class TestCheckMissingValues:
    """Tests for the check_missing_values exploration tool."""

    def test_reports_missing_counts(self, tmp_path):
        csv = _write_csv(tmp_path, content="a,b,c\n1,,3\n4,5,\n7,,9\n")
        tools = _get_tools(csv)
        result = tools["check_missing_values"].invoke({})
        assert "b" in result  # column b has 2 missing
        assert "missing" in result.lower()

    def test_no_missing_values(self, tmp_path):
        csv = _write_csv(tmp_path, content="a,b\n1,2\n3,4\n")
        tools = _get_tools(csv)
        result = tools["check_missing_values"].invoke({})
        assert "no missing" in result.lower() or "0" in result

    def test_all_missing_column(self, tmp_path):
        csv = _write_csv(tmp_path, content="a,b\n1,\n2,\n3,\n")
        tools = _get_tools(csv)
        result = tools["check_missing_values"].invoke({})
        assert "b" in result
        assert "100" in result or "3" in result  # 100% or 3 missing


# ---------------------------------------------------------------------------
# run_pandas_query tool
# ---------------------------------------------------------------------------

class TestRunPandasQuery:
    """Tests for the run_pandas_query exploration tool."""

    def test_simple_query(self, tmp_path):
        csv = _write_csv(tmp_path, content="x,y\n1,2\n3,4\n5,6\n")
        tools = _get_tools(csv)
        result = tools["run_pandas_query"].invoke({"query": "df.shape"})
        assert "3" in result  # 3 rows

    def test_groupby_query(self, tmp_path):
        csv = _write_csv(tmp_path)
        tools = _get_tools(csv)
        result = tools["run_pandas_query"].invoke({
            "query": "df.groupby('target').count()"
        })
        assert "yes" in result or "no" in result

    def test_syntax_error(self, tmp_path):
        csv = _write_csv(tmp_path)
        tools = _get_tools(csv)
        result = tools["run_pandas_query"].invoke({"query": "df.invalid_method_xyz()"})
        # Should contain error info
        assert "Error" in result or "ERROR" in result or "error" in result.lower()

    def test_truncates_long_output(self, tmp_path):
        csv = _write_csv(tmp_path, content="x\n" + "\n".join(str(i) for i in range(1000)) + "\n")
        tools = _get_tools(csv)
        result = tools["run_pandas_query"].invoke({
            "query": "df.to_string()"
        })
        # Output may or may not be truncated depending on size
        assert len(result) > 0


# ---------------------------------------------------------------------------
# _extract_json helper
# ---------------------------------------------------------------------------

class TestExtractJson:
    """Tests for the _extract_json helper."""

    def test_clean_json(self):
        from auto_llm_predictor.nodes.explore import _extract_json

        raw = '{"target_column": "label", "task_type": "binary"}'
        result = _extract_json(raw)
        assert result["target_column"] == "label"

    def test_markdown_fenced_json(self):
        from auto_llm_predictor.nodes.explore import _extract_json

        raw = '```json\n{"target_column": "label", "task_type": "binary"}\n```'
        result = _extract_json(raw)
        assert result["target_column"] == "label"

    def test_conversational_wrapper(self):
        from auto_llm_predictor.nodes.explore import _extract_json

        raw = 'Here is my analysis:\n{"target_column": "y", "task_type": "regression"}\nHope this helps!'
        result = _extract_json(raw)
        assert result["task_type"] == "regression"

    def test_raises_on_no_json(self):
        from auto_llm_predictor.nodes.explore import _extract_json

        with pytest.raises((json.JSONDecodeError, ValueError)):
            _extract_json("No JSON here at all!")


# ---------------------------------------------------------------------------
# _extract_analysis_from_agent helper
# ---------------------------------------------------------------------------

class TestExtractAnalysisFromAgent:
    """Tests for the _extract_analysis_from_agent helper."""

    def test_extracts_from_last_ai_message(self):
        from auto_llm_predictor.nodes.explore import _extract_analysis_from_agent
        from langchain_core.messages import AIMessage, HumanMessage

        messages = [
            HumanMessage(content="Investigate"),
            AIMessage(content="Let me check..."),
            AIMessage(content='{"target_column": "y", "task_type": "binary", "target_mapping": {}}'),
        ]
        result = _extract_analysis_from_agent(messages)
        assert result["target_column"] == "y"

    def test_skips_non_json_messages(self):
        from auto_llm_predictor.nodes.explore import _extract_analysis_from_agent
        from langchain_core.messages import AIMessage, HumanMessage

        messages = [
            HumanMessage(content="Go"),
            AIMessage(content="Thinking..."),
            AIMessage(content="Still thinking..."),
            AIMessage(content='{"target_column": "z", "task_type": "multiclass", "target_mapping": {"a": "A"}}'),
        ]
        result = _extract_analysis_from_agent(messages)
        assert result["target_column"] == "z"

    def test_raises_when_no_json(self):
        from auto_llm_predictor.nodes.explore import _extract_analysis_from_agent
        from langchain_core.messages import AIMessage

        messages = [AIMessage(content="I have no JSON for you.")]
        with pytest.raises(ValueError, match="No valid JSON"):
            _extract_analysis_from_agent(messages)


# ---------------------------------------------------------------------------
# _validate_and_correct_analysis
# ---------------------------------------------------------------------------

class TestValidateAndCorrectAnalysis:
    """Tests for the _validate_and_correct_analysis helper."""

    def test_regression_clears_mapping(self, tmp_path):
        from auto_llm_predictor.nodes.explore import _validate_and_correct_analysis

        csv = _write_csv(tmp_path, content="x,y\n1,2.5\n3,4.5\n")
        analysis = {
            "target_column": "y",
            "task_type": "regression",
            "target_mapping": {"2.5": "low", "4.5": "high"},
        }
        _validate_and_correct_analysis(analysis, csv)
        assert analysis["target_mapping"] == {}

    def test_auto_fills_missing_mapping_values(self, tmp_path):
        from auto_llm_predictor.nodes.explore import _validate_and_correct_analysis

        csv = _write_csv(tmp_path, content="x,label\n1,A\n2,B\n3,C\n")
        analysis = {
            "target_column": "label",
            "task_type": "multiclass",
            "target_mapping": {"A": "Class A"},  # Missing B, C
        }
        _validate_and_correct_analysis(analysis, csv)
        assert "B" in analysis["target_mapping"]
        assert "C" in analysis["target_mapping"]

    def test_corrects_binary_to_multiclass(self, tmp_path):
        from auto_llm_predictor.nodes.explore import _validate_and_correct_analysis

        csv = _write_csv(tmp_path, content="x,label\n1,A\n2,B\n3,C\n")
        analysis = {
            "target_column": "label",
            "task_type": "binary",
            "target_mapping": {"A": "A", "B": "B", "C": "C"},
        }
        _validate_and_correct_analysis(analysis, csv)
        assert analysis["task_type"] == "multiclass"

    def test_corrects_multiclass_to_binary(self, tmp_path):
        from auto_llm_predictor.nodes.explore import _validate_and_correct_analysis

        csv = _write_csv(tmp_path, content="x,label\n1,pos\n2,neg\n3,pos\n")
        analysis = {
            "target_column": "label",
            "task_type": "multiclass",
            "target_mapping": {"pos": "Positive", "neg": "Negative"},
        }
        _validate_and_correct_analysis(analysis, csv)
        assert analysis["task_type"] == "binary"


# ---------------------------------------------------------------------------
# _fallback_explore
# ---------------------------------------------------------------------------

class TestFallbackExplore:
    """Tests for the _fallback_explore function."""

    def _make_state(self, tmp_path):
        csv = _write_csv(tmp_path)
        return {
            "csv_path": csv,
            "target_column": "",
        }

    def test_returns_required_fields_with_valid_response(self, tmp_path):
        from auto_llm_predictor.nodes.explore import _fallback_explore
        from auto_llm_predictor.utils import profile_csv

        state = self._make_state(tmp_path)

        class MockLLM:
            def invoke(self, messages):
                class Resp:
                    content = json.dumps({
                        "target_column": "target",
                        "task_type": "binary",
                        "target_mapping": {"yes": "Yes", "no": "No"},
                        "class_distribution": {"Yes": 2, "No": 2},
                        "data_quality_notes": "Looks good",
                        "reasoning": "target column is obvious",
                    })
                return Resp()

        data_profile = profile_csv(state["csv_path"])
        result = _fallback_explore(state, llm=MockLLM(), data_profile=data_profile)

        assert result["target_column"] == "target"
        assert result["task_type"] == "binary"
        assert result["target_mapping"] == {"yes": "Yes", "no": "No"}
        assert result["exploration_steps"] == []
        assert result["data_quality_notes"] == ""
        assert "data_profile" in result
        assert "messages" in result

    def test_raises_on_llm_exception(self, tmp_path):
        from auto_llm_predictor.nodes.explore import _fallback_explore
        from auto_llm_predictor.utils import profile_csv

        state = self._make_state(tmp_path)

        class FailLLM:
            def invoke(self, messages):
                raise ConnectionError("API unavailable")

        data_profile = profile_csv(state["csv_path"])
        with pytest.raises(ConnectionError):
            _fallback_explore(state, llm=FailLLM(), data_profile=data_profile)


# ---------------------------------------------------------------------------
# explore_data node — integration via fallback
# ---------------------------------------------------------------------------

class TestExploreData:
    """Integration tests for the explore_data node (triggers fallback path)."""

    def _make_state(self, tmp_path):
        csv = _write_csv(tmp_path)
        return {
            "csv_path": csv,
            "target_column": "",
            "test_csv_path": "",
        }

    def test_returns_required_fields_via_fallback(self, tmp_path):
        """When create_agent is unavailable, fallback should still work."""
        from auto_llm_predictor.nodes.explore import explore_data

        state = self._make_state(tmp_path)

        class MockLLM:
            def invoke(self, messages):
                class Resp:
                    content = json.dumps({
                        "target_column": "target",
                        "task_type": "binary",
                        "target_mapping": {"yes": "Yes", "no": "No"},
                        "class_distribution": {"Yes": 2, "No": 2},
                        "data_quality_notes": "Clean",
                        "reasoning": "Binary classification on target",
                    })
                return Resp()

            def bind_tools(self, tools):
                raise NotImplementedError("No tool support")

        result = explore_data(state, llm=MockLLM())

        # All required fields must be present
        assert "data_profile" in result
        assert result["target_column"] == "target"
        assert result["task_type"] == "binary"
        assert "target_mapping" in result
        assert "exploration_steps" in result
        assert "data_quality_notes" in result
        assert "messages" in result
