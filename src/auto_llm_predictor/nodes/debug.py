"""Node: debug_prep_failure — ReAct agent that diagnoses code generation failures."""

from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from auto_llm_predictor.prompts.debug import DEBUG_SYSTEM, format_debug_prompt
from auto_llm_predictor.state import PipelineState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sandboxed debug tool factory
# ---------------------------------------------------------------------------

_MAX_FILE_LINES = 200
_MAX_FILE_BYTES = 10_240
_MAX_DIR_ENTRIES = 100
_MAX_CSV_ROWS = 20
_SNIPPET_TIMEOUT = 30


def _make_debug_tools(output_dir: str, csv_path: str) -> list:
    """Create sandboxed debug tools bound to specific allowed directories."""
    allowed_dirs = set()
    if output_dir:
        allowed_dirs.add(Path(output_dir).resolve())
    if csv_path:
        allowed_dirs.add(Path(csv_path).resolve().parent)

    def _is_allowed(path: str) -> bool:
        try:
            resolved = Path(path).resolve()
            return any(
                resolved == d or d in resolved.parents
                for d in allowed_dirs
            )
        except (OSError, ValueError):
            return False

    @tool
    def read_file(path: str) -> str:
        """Read a file's contents. Use this to inspect generated scripts, data files, or config files."""
        if not _is_allowed(path):
            return f"ERROR: Access denied. Path must be within allowed directories."
        p = Path(path)
        if not p.exists():
            return f"ERROR: File not found: {path}"
        if not p.is_file():
            return f"ERROR: Not a file: {path}"
        try:
            text = p.read_text(errors="replace")
            lines = text.splitlines(keepends=True)
            if len(lines) > _MAX_FILE_LINES:
                text = "".join(lines[:_MAX_FILE_LINES])
                text += f"\n... (truncated, {len(lines)} total lines)"
            if len(text) > _MAX_FILE_BYTES:
                text = text[:_MAX_FILE_BYTES] + "\n... (truncated)"
            return text
        except Exception as e:
            return f"ERROR: Could not read file: {e}"

    @tool
    def run_python_snippet(code: str) -> str:
        """Execute a short Python snippet to test hypotheses about the data or environment. Returns stdout and stderr."""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False,
                dir=output_dir if Path(output_dir).is_dir() else None,
            ) as f:
                f.write(code)
                tmp_path = f.name

            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=_SNIPPET_TIMEOUT,
            )
            output = result.stdout
            if result.stderr:
                output += "\n--- STDERR ---\n" + result.stderr
            if len(output) > 3000:
                output = output[:3000] + "\n... (truncated)"
            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return f"ERROR: Snippet timed out after {_SNIPPET_TIMEOUT}s"
        except Exception as e:
            return f"ERROR: Failed to run snippet: {e}"
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

    @tool
    def list_directory(path: str) -> str:
        """List files in a directory with sizes. Use this to check what files exist."""
        if not _is_allowed(path):
            return f"ERROR: Access denied. Path must be within allowed directories."
        p = Path(path)
        if not p.exists():
            return f"ERROR: Directory not found: {path}"
        if not p.is_dir():
            return f"ERROR: Not a directory: {path}"
        try:
            entries = sorted(p.iterdir(), key=lambda x: x.name)
            if not entries:
                return "(empty directory)"
            lines = []
            for entry in entries[:_MAX_DIR_ENTRIES]:
                if entry.is_file():
                    size = entry.stat().st_size
                    lines.append(f"  {entry.name}  ({size:,} bytes)")
                elif entry.is_dir():
                    lines.append(f"  {entry.name}/")
            if len(entries) > _MAX_DIR_ENTRIES:
                lines.append(f"  ... and {len(entries) - _MAX_DIR_ENTRIES} more entries")
            return "\n".join(lines)
        except Exception as e:
            return f"ERROR: Could not list directory: {e}"

    @tool
    def inspect_csv_sample(path: str, n_rows: int = 5) -> str:
        """Preview CSV data with dtypes and missing value counts. Use this to verify CSV structure."""
        if not _is_allowed(path):
            return f"ERROR: Access denied. Path must be within allowed directories."
        if not Path(path).exists():
            return f"ERROR: File not found: {path}"
        n_rows = min(max(1, n_rows), _MAX_CSV_ROWS)
        try:
            import pandas as pd
            df = pd.read_csv(path, low_memory=False)
            parts = [
                f"Shape: {df.shape[0]} rows x {df.shape[1]} columns",
                "",
                "Columns and dtypes:",
            ]
            for col in df.columns:
                missing = int(df[col].isna().sum())
                parts.append(f"  {col}: {df[col].dtype} (missing={missing})")
            parts.append(f"\nFirst {n_rows} rows:")
            parts.append(df.head(n_rows).to_string())
            return "\n".join(parts)
        except Exception as e:
            return f"ERROR: Could not read CSV: {e}"

    return [read_file, run_python_snippet, list_directory, inspect_csv_sample]


# ---------------------------------------------------------------------------
# Fallback single-shot diagnosis (when tool calling is not supported)
# ---------------------------------------------------------------------------

def _fallback_diagnosis(state: PipelineState, *, llm) -> dict:
    """Single-shot LLM diagnosis without tool calling."""
    prompt = format_debug_prompt(
        script_path=state.get("prep_code_path", ""),
        error_output=state.get("prep_result", ""),
        script_code=state.get("prep_code", ""),
        data_profile=state.get("data_profile", ""),
        csv_path=state.get("csv_path", ""),
        output_dir=state.get("output_dir", ""),
        attempt_number=state.get("prep_attempts", 0),
        target_column=state.get("target_column", ""),
        task_type=state.get("task_type", ""),
    )
    messages = [
        SystemMessage(content=DEBUG_SYSTEM),
        HumanMessage(content=prompt),
    ]
    try:
        response = llm.invoke(messages)
        diagnosis = response.content.strip()
    except Exception as e:
        logger.warning("Fallback diagnosis LLM call failed: %s", e)
        diagnosis = f"Debug agent fallback also failed: {e}. Retrying with error context only."

    return {
        "debug_diagnosis": diagnosis,
        "debug_tool_calls": [],
        "messages": [
            HumanMessage(content=f"[debug_prep_failure] Fallback diagnosis: {diagnosis[:200]}"),
        ],
    }


# ---------------------------------------------------------------------------
# Main debug node
# ---------------------------------------------------------------------------

def debug_prep_failure(state: PipelineState, *, llm=None) -> dict:
    """Invoke a ReAct debug agent to diagnose why the prep script failed.

    Writes: debug_diagnosis, debug_tool_calls, messages
    """
    logger.info(
        "Debugging prep failure (attempt %d)", state.get("prep_attempts", 0),
    )

    output_dir = state.get("output_dir", "")
    csv_path = state.get("csv_path", "")
    tools = _make_debug_tools(output_dir, csv_path)

    # Build the debug prompt
    debug_prompt = format_debug_prompt(
        script_path=state.get("prep_code_path", ""),
        error_output=state.get("prep_result", ""),
        script_code=state.get("prep_code", ""),
        data_profile=state.get("data_profile", ""),
        csv_path=csv_path,
        output_dir=output_dir,
        attempt_number=state.get("prep_attempts", 0),
        target_column=state.get("target_column", ""),
        task_type=state.get("task_type", ""),
    )

    # Try to create and run the ReAct agent
    try:
        from langchain.agents import create_agent

        debug_agent = create_agent(
            model=llm,
            tools=tools,
            prompt=DEBUG_SYSTEM,
        )

        result = debug_agent.invoke(
            {"messages": [HumanMessage(content=debug_prompt)]},
            config={"recursion_limit": 25},
        )

        # Extract diagnosis from the agent's final message
        agent_messages = result.get("messages", [])
        diagnosis = ""
        if agent_messages:
            last_msg = agent_messages[-1]
            diagnosis = (
                last_msg.content if hasattr(last_msg, "content") else str(last_msg)
            )

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
            "Debug agent completed: %d tool calls, diagnosis length %d",
            len(tool_call_log), len(diagnosis),
        )

        return {
            "debug_diagnosis": diagnosis,
            "debug_tool_calls": tool_call_log,
            "messages": [
                HumanMessage(
                    content=f"[debug_prep_failure] Diagnosis ({len(tool_call_log)} tool calls): "
                    + diagnosis[:200],
                ),
            ],
        }

    except Exception as e:
        logger.warning(
            "ReAct debug agent failed (%s: %s), falling back to single-shot diagnosis.",
            type(e).__name__, e,
        )
        print(
            f"[debug_prep_failure] WARNING: Model does not support tool calling "
            f"({type(e).__name__}). Falling back to single-shot diagnosis.",
            flush=True,
        )
        return _fallback_diagnosis(state, llm=llm)


# ---------------------------------------------------------------------------
# Routing function
# ---------------------------------------------------------------------------

def route_after_debug(state: PipelineState) -> str:
    """Route after debug: retry code generation or abort.

    Returns 'write_prep_code' to retry with the diagnosis, or
    'verify_prepared_data' to give up (when diagnosis says ABORT or HUMAN_HELP).
    """
    diagnosis = state.get("debug_diagnosis", "")
    upper = diagnosis.upper()

    if "ABORT" in upper or "HUMAN_HELP" in upper:
        logger.warning("Debug agent recommends aborting: %s", diagnosis[:200])
        return "verify_prepared_data"

    return "write_prep_code"
