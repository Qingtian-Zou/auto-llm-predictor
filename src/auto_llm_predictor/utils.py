"""Shared utility functions."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd


def profile_csv(csv_path: str, max_rows: int = 5, max_cols: int = 60) -> str:
    """Build an LLM-readable text summary of a CSV file.

    Includes shape, column names/types, descriptive statistics,
    value counts of likely-categorical columns, and sample rows.
    """
    path = Path(csv_path)
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        return (
            f"File: {path.name}\n"
            f"ERROR: Failed to read CSV file: {e}\n"
            f"The file may be malformed, use a non-CSV format, or have encoding issues."
        )

    parts: list[str] = []
    parts.append(f"File: {path.name}")
    parts.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

    # Column listing (truncate if huge)
    col_info = []
    cols = list(df.columns)
    for c in cols[:max_cols]:
        dtype = str(df[c].dtype)
        n_unique = df[c].nunique()
        n_missing = int(df[c].isna().sum())
        col_info.append(f"  {c}  (dtype={dtype}, unique={n_unique}, missing={n_missing})")
    if len(cols) > max_cols:
        col_info.append(f"  ... and {len(cols) - max_cols} more columns")
    parts.append("Columns:\n" + "\n".join(col_info) + "\n")

    # Descriptive stats on numeric columns (first 20)
    numeric_cols = df.select_dtypes("number").columns[:20]
    if len(numeric_cols):
        desc = df[numeric_cols].describe().round(4).to_string()
        parts.append("Numeric summary (first 20 numeric columns):\n" + desc + "\n")

    # Value counts on low-cardinality columns (likely targets / categories)
    cat_cols = [c for c in df.columns
                if (df[c].dtype == "object" and df[c].nunique() <= 20)
                or df[c].nunique() <= 10]
    for c in cat_cols[:10]:
        vc = df[c].value_counts(dropna=False).head(10).to_dict()
        parts.append(f"Value counts for '{c}': {vc}")

    # Sample rows
    parts.append(f"\nSample rows (first {max_rows}):")
    sample = df.head(max_rows).to_string(max_cols=max_cols)
    parts.append(sample)

    return "\n".join(parts)


def run_script(script_path: str, timeout: int = 300) -> tuple[bool, str]:
    """Run a Python script and return (success, combined_output)."""
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout
        if result.stderr:
            output += "\n--- STDERR ---\n" + result.stderr
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, f"Script timed out after {timeout}s"
    except Exception as e:
        return False, f"Failed to run script: {e}"


def run_llamafactory(
    yaml_path: str,
    timeout: int = 7200,
    stream: bool = True,
    tail_chars: int = 5000,
    log_callback: callable | None = None,
) -> tuple[bool, str]:
    """Run llamafactory-cli train with a YAML config.

    When *stream* is True (default), stdout and stderr are printed in
    real-time so the user can monitor the long-running process.  The
    last *tail_chars* characters of the combined output are returned
    for downstream state storage.

    Returns (success, output_tail).  Timeout defaults to 2 hours.
    """
    import sys
    import threading
    from collections import deque

    cmd = ["llamafactory-cli", "train", str(yaml_path)]

    if not stream:
        # Simple buffered mode (for short tasks or testing)
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
            )
            output = result.stdout[-tail_chars:] if len(result.stdout) > tail_chars else result.stdout
            if result.stderr:
                stderr_tail = result.stderr[-3000:] if len(result.stderr) > 3000 else result.stderr
                output += "\n--- STDERR (tail) ---\n" + stderr_tail
            return result.returncode == 0, output
        except subprocess.TimeoutExpired:
            return False, f"llamafactory-cli timed out after {timeout}s"
        except Exception as e:
            return False, f"Failed to run llamafactory-cli: {e}"

    # ── Streaming mode: print output live ─────────────────────
    output_lines: deque[str] = deque(maxlen=200)  # keep last 200 lines

    def _reader(pipe, prefix=""):
        """Read lines from a pipe, print them, and store in buffer."""
        try:
            for line in iter(pipe.readline, ""):
                line = line.rstrip("\n")
                print(f"{prefix}{line}", flush=True)
                output_lines.append(line)
                if log_callback:
                    log_callback(f"{prefix}{line}")
        except ValueError:
            pass  # pipe closed
        finally:
            pipe.close()

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
        )

        # Read stdout and stderr in parallel threads
        stdout_thread = threading.Thread(
            target=_reader, args=(proc.stdout,), daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_reader, args=(proc.stderr, "[stderr] "), daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        proc.wait(timeout=timeout)
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)

        output_tail = "\n".join(output_lines)
        if len(output_tail) > tail_chars:
            output_tail = output_tail[-tail_chars:]

        return proc.returncode == 0, output_tail

    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return False, f"llamafactory-cli timed out after {timeout}s"
    except Exception as e:
        return False, f"Failed to run llamafactory-cli: {e}"


def save_yaml(data: dict | str, path: str) -> None:
    """Save a YAML config (either a dict or raw string) to a file."""
    import yaml

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        if isinstance(data, str):
            f.write(data)
        else:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries
