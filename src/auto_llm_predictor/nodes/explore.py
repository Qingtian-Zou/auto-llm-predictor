"""Node: explore_data — ReAct agent that profiles and investigates CSV data."""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from auto_llm_predictor.prompts.explore import (
    EXPLORE_AGENT_SYSTEM,
    EXPLORE_SYSTEM,
    format_explore_agent_prompt,
    format_explore_prompt,
)
from auto_llm_predictor.state import PipelineState
from auto_llm_predictor.utils import profile_csv

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_SAMPLE_ROWS = 20
_MAX_VALUE_COUNTS = 50
_MAX_CORR_COLS = 10
_QUERY_TIMEOUT = 30
_MAX_OUTPUT_CHARS = 3000

# ---------------------------------------------------------------------------
# Exploration tool factory
# ---------------------------------------------------------------------------


def _make_explore_tools(csv_path: str) -> list:
    """Create exploration tools bound to a specific CSV file.

    The CSV is loaded once; all tools close over the DataFrame for fast
    repeated access.  ``run_pandas_query`` is the exception — it executes
    in a subprocess for sandboxing.
    """
    try:
        _df = pd.read_csv(csv_path, low_memory=False)
    except Exception as exc:
        logger.warning("Failed to load CSV for exploration tools: %s", exc)
        _df = pd.DataFrame()

    _csv_path = csv_path  # captured for run_pandas_query subprocess

    @tool
    def sample_rows(n: int = 5) -> str:
        """Get n random sample rows from the dataset to see actual data values."""
        if _df.empty:
            return "Dataset is empty — no rows to sample."
        n = min(max(1, n), _MAX_SAMPLE_ROWS)
        sample = _df.sample(n=min(n, len(_df)))
        text = sample.to_string()
        if len(text) > _MAX_OUTPUT_CHARS:
            text = text[:_MAX_OUTPUT_CHARS] + "\n... (truncated)"
        return text

    @tool
    def column_stats(column_name: str) -> str:
        """Get detailed statistics for a specific column: dtype, unique count, missing count, descriptive stats, and top values."""
        if column_name not in _df.columns:
            available = ", ".join(_df.columns[:30])
            return f"ERROR: Column '{column_name}' not found. Available columns: {available}"
        col = _df[column_name]
        parts = [
            f"Column: {column_name}",
            f"  dtype: {col.dtype}",
            f"  unique: {col.nunique()}",
            f"  missing: {int(col.isna().sum())} ({col.isna().mean():.1%})",
        ]
        try:
            desc = col.describe()
            parts.append(f"  describe:\n{desc.to_string()}")
        except Exception:
            pass
        # Top 5 values
        vc = col.value_counts(dropna=False).head(5)
        parts.append(f"  top values:\n{vc.to_string()}")
        return "\n".join(parts)

    @tool
    def value_counts(column_name: str, top_k: int = 20) -> str:
        """See the value distribution for a column, with counts and percentages."""
        if column_name not in _df.columns:
            available = ", ".join(_df.columns[:30])
            return f"ERROR: Column '{column_name}' not found. Available columns: {available}"
        top_k = min(max(1, top_k), _MAX_VALUE_COUNTS)
        col = _df[column_name]
        vc = col.value_counts(dropna=False).head(top_k)
        total = len(col)
        lines = [f"Value counts for '{column_name}' (total={total}):"]
        for val, count in vc.items():
            pct = count / total * 100
            lines.append(f"  {val!r}: {count} ({pct:.1f}%)")
        return "\n".join(lines)

    @tool
    def correlation_matrix(columns: str) -> str:
        """Compute correlation between named numeric columns. Pass column names as a comma-separated string, e.g. 'col1,col2,col3'."""
        col_list = [c.strip() for c in columns.split(",") if c.strip()]
        if not col_list:
            return "ERROR: No column names provided. Pass comma-separated names."
        col_list = col_list[:_MAX_CORR_COLS]
        missing = [c for c in col_list if c not in _df.columns]
        if missing:
            return f"ERROR: Columns not found: {missing}"
        subset = _df[col_list]
        non_numeric = [c for c in col_list if not pd.api.types.is_numeric_dtype(subset[c])]
        if non_numeric:
            return f"ERROR: Non-numeric columns cannot be correlated: {non_numeric}"
        corr = subset.corr()
        return corr.to_string()

    @tool
    def check_missing_values() -> str:
        """Analyze missing values across all columns in the dataset."""
        if _df.empty:
            return "Dataset is empty — no columns to analyze."
        missing = _df.isna().sum()
        total = len(_df)
        lines = [f"Missing value analysis ({total} rows, {len(_df.columns)} columns):"]
        cols_with_missing = missing[missing > 0].sort_values(ascending=False)
        if cols_with_missing.empty:
            lines.append("  No missing values found in any column.")
        else:
            lines.append(f"  {len(cols_with_missing)} column(s) with missing values:")
            for col, count in cols_with_missing.items():
                pct = count / total * 100
                lines.append(f"    {col}: {count} missing ({pct:.1f}%)")
        return "\n".join(lines)

    @tool
    def run_pandas_query(query: str) -> str:
        """Run an arbitrary pandas expression on the dataframe. The dataframe is available as 'df'. Example: 'df.groupby(\"col\").mean()'"""
        script = (
            f"import pandas as pd\n"
            f"df = pd.read_csv({_csv_path!r}, low_memory=False)\n"
            f"result = {query}\n"
            f"print(result)\n"
        )
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False,
            ) as f:
                f.write(script)
                tmp_path = f.name

            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=_QUERY_TIMEOUT,
            )
            output = result.stdout
            if result.stderr:
                output += "\n--- STDERR ---\n" + result.stderr
            if len(output) > _MAX_OUTPUT_CHARS:
                output = output[:_MAX_OUTPUT_CHARS] + "\n... (truncated)"
            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return f"ERROR: Query timed out after {_QUERY_TIMEOUT}s"
        except Exception as e:
            return f"ERROR: Failed to run query: {e}"
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

    return [sample_rows, column_stats, value_counts, correlation_matrix,
            check_missing_values, run_pandas_query]


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

def _extract_json(raw: str) -> dict:
    """Extract a JSON object from LLM text, stripping fences and wrappers."""
    # Strip markdown fences if present
    if raw.startswith("```"):
        parts = raw.split("\n", 1)
        raw = parts[1] if len(parts) > 1 else ""
        if raw.endswith("```"):
            raw = raw[: raw.rfind("```")]
        raw = raw.strip()

    json_match = re.search(r"(\{.*\})", raw, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        json_str = raw.strip()

    return json.loads(json_str)


def _extract_analysis_from_agent(messages: list) -> dict:
    """Extract the JSON analysis from the agent's final AI message."""
    for msg in reversed(messages):
        content = getattr(msg, "content", "")
        if not content or not isinstance(content, str):
            continue
        try:
            return _extract_json(content)
        except (json.JSONDecodeError, ValueError):
            continue

    raise ValueError("No valid JSON analysis found in agent messages.")


# ---------------------------------------------------------------------------
# Target mapping validation (shared by agent and fallback paths)
# ---------------------------------------------------------------------------

def _validate_and_correct_analysis(analysis: dict, csv_path: str) -> None:
    """Validate and correct target_mapping and task_type in-place.

    For regression tasks, forces target_mapping to empty dict.
    For classification tasks, validates mapping completeness against
    actual CSV values, auto-fills missing entries, and reconciles
    binary/multiclass task_type with actual class count.
    """
    if analysis["task_type"] == "regression":
        if analysis.get("target_mapping"):
            logger.info(
                "task_type is 'regression'; clearing target_mapping (%d entries).",
                len(analysis["target_mapping"]),
            )
        analysis["target_mapping"] = {}
    else:
        target_col = analysis["target_column"]
        try:
            df = pd.read_csv(csv_path, usecols=[target_col], low_memory=False)
            actual_values = set(str(v) for v in df[target_col].dropna().unique())
            mapped_keys = set(str(k) for k in analysis["target_mapping"].keys())
            missing = actual_values - mapped_keys
            if missing:
                logger.warning(
                    "target_mapping is missing %d value(s): %s. Auto-filling.",
                    len(missing), missing,
                )
                for val in sorted(missing):
                    analysis["target_mapping"][val] = val
            # Reconcile task_type with actual class count
            n_classes = len(analysis["target_mapping"])
            if analysis["task_type"] == "binary" and n_classes > 2:
                logger.warning(
                    "task_type was 'binary' but found %d classes. Correcting to 'multiclass'.",
                    n_classes,
                )
                analysis["task_type"] = "multiclass"
            elif analysis["task_type"] == "multiclass" and n_classes == 2:
                logger.warning(
                    "task_type was 'multiclass' but found only 2 classes. Correcting to 'binary'.",
                )
                analysis["task_type"] = "binary"
        except Exception as e:
            logger.warning("Could not validate target_mapping completeness: %s", e)


# ---------------------------------------------------------------------------
# Fallback single-shot exploration (when tool calling is not supported)
# ---------------------------------------------------------------------------

def _fallback_explore(state: PipelineState, *, llm, data_profile: str) -> dict:
    """Single-shot LLM exploration without tool calling.

    Preserves the original explore_data behavior for models that do not
    support tool calling.
    """
    user_prompt = format_explore_prompt(
        data_profile=data_profile,
        target_column=state.get("target_column", ""),
    )

    messages = [
        SystemMessage(content=EXPLORE_SYSTEM),
        HumanMessage(content=user_prompt),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip()
    except Exception as e:
        logger.error("Fallback exploration LLM call failed: %s", e)
        raise

    logger.info("LLM explore response (fallback): %s", raw[:500])

    analysis = _extract_json(raw)
    _validate_and_correct_analysis(analysis, state["csv_path"])

    return {
        "data_profile": data_profile,
        "target_column": analysis["target_column"],
        "task_type": analysis["task_type"],
        "target_mapping": analysis["target_mapping"],
        "exploration_steps": [],
        "data_quality_notes": "",
        "messages": [
            HumanMessage(
                content=f"[explore_data] CSV profiled (single-shot fallback). "
                f"Target: {analysis['target_column']}, "
                f"Task: {analysis['task_type']}. {analysis.get('reasoning', '')}",
            ),
        ],
    }


# ---------------------------------------------------------------------------
# Main explore node
# ---------------------------------------------------------------------------

def explore_data(state: PipelineState, *, llm) -> dict:
    """Profile the CSV and use a ReAct agent to investigate data quality,
    identify the prediction target, task type, and mapping.

    Falls back to single-shot LLM call if the model does not support
    tool calling.

    Writes: data_profile, target_column, task_type, target_mapping,
            exploration_steps, data_quality_notes, messages
    """
    csv_path = state["csv_path"]
    logger.info("Profiling CSV: %s", csv_path)

    data_profile = profile_csv(csv_path)

    # If a separate test CSV is provided, verify header alignment.
    test_csv_path = state.get("test_csv_path")
    if test_csv_path:
        try:
            train_cols = set(pd.read_csv(csv_path, nrows=0).columns)
            test_cols = set(pd.read_csv(test_csv_path, nrows=0).columns)

            missing_in_test = sorted(list(train_cols - test_cols))
            missing_in_train = sorted(list(test_cols - train_cols))

            if missing_in_test or missing_in_train:
                warning_lines = ["\n⚠️ WARNING: Feature mismatch between Train and Test CSVs!"]
                if missing_in_test:
                    warning_lines.append(f"   Missing in TEST: {missing_in_test}")
                if missing_in_train:
                    warning_lines.append(f"   Missing in TRAIN: {missing_in_train}")
                warning_lines.append(
                    "   Make sure to handle these missing columns during data preparation "
                    "(e.g., dropping them or filling with defaults)."
                )

                warning_text = "\n".join(warning_lines)
                logger.warning(warning_text)
                data_profile += f"\n\n{warning_text}"
        except Exception as e:
            logger.warning("Failed to compare train/test CSV headers: %s", e)

    # --- Create exploration tools ---
    tools = _make_explore_tools(csv_path)

    # --- Build agent prompt ---
    user_prompt = format_explore_agent_prompt(
        data_profile=data_profile,
        target_column=state.get("target_column", ""),
    )

    # --- Try ReAct agent ---
    try:
        from langchain.agents import create_agent

        explore_agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=EXPLORE_AGENT_SYSTEM,
        )

        result = explore_agent.invoke(
            {"messages": [HumanMessage(content=user_prompt)]},
            config={"recursion_limit": 20},
        )

        # Extract JSON analysis from the agent's final message
        agent_messages = result.get("messages", [])
        analysis = _extract_analysis_from_agent(agent_messages)

        # Extract tool call log
        tool_call_log = []
        for msg in agent_messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_call_log.append({
                        "tool": tc.get("name", ""),
                        "args": tc.get("args", {}),
                    })

        logger.info(
            "Explore agent completed: %d tool calls", len(tool_call_log),
        )

    except Exception as e:
        logger.warning(
            "ReAct explore agent failed (%s: %s), falling back to single-shot.",
            type(e).__name__, e,
        )
        print(
            f"[explore_data] WARNING: Model does not support tool calling "
            f"({type(e).__name__}). Falling back to single-shot exploration.",
            flush=True,
        )
        return _fallback_explore(state, llm=llm, data_profile=data_profile)

    # --- Validate target_mapping ---
    _validate_and_correct_analysis(analysis, csv_path)

    # --- Extract data quality report ---
    data_quality_notes = analysis.get("data_quality_notes", "")

    return {
        "data_profile": data_profile,
        "target_column": analysis["target_column"],
        "task_type": analysis["task_type"],
        "target_mapping": analysis["target_mapping"],
        "exploration_steps": tool_call_log,
        "data_quality_notes": data_quality_notes,
        "messages": [
            HumanMessage(
                content=f"[explore_data] Agent explored CSV ({len(tool_call_log)} tool calls). "
                f"Target: {analysis['target_column']}, Task: {analysis['task_type']}. "
                f"{analysis.get('reasoning', '')}",
            ),
        ],
    }
