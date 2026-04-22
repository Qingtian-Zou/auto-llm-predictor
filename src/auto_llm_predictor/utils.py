# Copyright 2024-2026 Qingtian Zou
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared utility functions."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def normalize_path(p: str) -> str:
    """Normalize a path string to use OS-native separators.

    Replaces backslashes with forward slashes so that paths originating
    from Windows-style input work correctly on POSIX systems (where ``\\``
    is a legal filename character rather than a separator).
    """
    return str(Path(p.replace("\\", "/")))


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


def run_script(
    script_path: str,
    timeout: int = 300,
    args: list[str] | None = None,
) -> tuple[bool, str]:
    """Run a Python script and return (success, combined_output)."""
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    try:
        result = subprocess.run(
            cmd,
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
    idle_timeout: int | None = None,
) -> tuple[bool, int | None, str]:
    """Run llamafactory-cli train with a YAML config.

    When *stream* is True (default), stdout and stderr are printed in
    real-time so the user can monitor the long-running process.  The
    last *tail_chars* characters of the combined output are returned
    for downstream state storage.

    *idle_timeout* enables activity-based timeout: if the subprocess
    produces no output for *idle_timeout* seconds it is killed.  When
    set, *timeout* acts as a wall-clock safety net.  When ``None``
    (default), the legacy behaviour of a single wall-clock timeout is
    used.

    Returns (success, return_code, output_tail).
    """
    import os
    import sys
    import time
    import threading
    from collections import deque

    cmd = ["llamafactory-cli", "train", str(yaml_path)]

    # Disable Weights & Biases so no remote API key is required
    env = {**os.environ, "WANDB_DISABLED": "true"}

    if not stream:
        # Simple buffered mode (for short tasks or testing)
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, env=env,
            )
            output = result.stdout[-tail_chars:] if len(result.stdout) > tail_chars else result.stdout
            if result.stderr:
                stderr_tail = result.stderr[-3000:] if len(result.stderr) > 3000 else result.stderr
                output += "\n--- STDERR (tail) ---\n" + stderr_tail
            return result.returncode == 0, result.returncode, output
        except subprocess.TimeoutExpired:
            return False, None, f"llamafactory-cli timed out after {timeout}s"
        except Exception as e:
            return False, None, f"Failed to run llamafactory-cli: {e}"

    # ── Streaming mode: print output live ─────────────────────
    output_lines: deque[str] = deque(maxlen=200)  # keep last 200 lines

    # Activity tracking for idle-timeout watchdog
    last_activity = time.monotonic()
    activity_lock = threading.Lock()

    def _reader(pipe, prefix=""):
        """Read lines from a pipe, print them, and store in buffer."""
        nonlocal last_activity
        try:
            for line in iter(pipe.readline, ""):
                line = line.rstrip("\n")
                print(f"{prefix}{line}", flush=True)
                output_lines.append(line)
                with activity_lock:
                    last_activity = time.monotonic()
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
            env=env,
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

        if idle_timeout is not None:
            # Activity-based watchdog: poll for process exit while
            # checking that output is still being produced.
            poll_interval = 5
            start_time = time.monotonic()
            while True:
                try:
                    proc.wait(timeout=poll_interval)
                    break  # process exited
                except subprocess.TimeoutExpired:
                    now = time.monotonic()
                    with activity_lock:
                        idle_seconds = now - last_activity
                    if idle_seconds > idle_timeout:
                        raise subprocess.TimeoutExpired(
                            cmd, idle_timeout,
                            output=f"No output for {idle_seconds:.0f}s "
                                   f"(idle_timeout={idle_timeout}s)",
                        )
                    if now - start_time > timeout:
                        raise subprocess.TimeoutExpired(cmd, timeout)
        else:
            # Legacy behaviour: single wall-clock wait
            proc.wait(timeout=timeout)

        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)

        output_tail = "\n".join(output_lines)
        if len(output_tail) > tail_chars:
            output_tail = output_tail[-tail_chars:]

        return proc.returncode == 0, proc.returncode, output_tail

    except subprocess.TimeoutExpired as e:
        proc.kill()
        proc.wait()
        # Ensure reader threads finish before returning
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
        msg = str(e.output) if e.output else f"llamafactory-cli timed out after {timeout}s"
        return False, proc.returncode, msg
    except Exception as e:
        return False, None, f"Failed to run llamafactory-cli: {e}"


def find_latest_checkpoint(sft_dir: str) -> str | None:
    """Find the latest HuggingFace Trainer checkpoint directory.

    LlamaFactory saves checkpoints as ``checkpoint-{step}/`` inside the
    output directory.  Returns the path to the highest-numbered checkpoint,
    or ``None`` if no checkpoints exist.
    """
    import re

    sft_path = Path(sft_dir)
    if not sft_path.exists():
        return None

    checkpoints = []
    for d in sft_path.iterdir():
        if d.is_dir():
            m = re.match(r"checkpoint-(\d+)$", d.name)
            if m:
                checkpoints.append((int(m.group(1)), d))

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return str(checkpoints[0][1])


def set_resume_in_yaml(yaml_path: str, resume: bool = True) -> None:
    """Set or remove ``resume_from_checkpoint`` in a LlamaFactory YAML config.

    Reads the file, parses as YAML, sets the key, and writes back.
    """
    import yaml

    path = Path(yaml_path)
    with open(path) as f:
        config = yaml.safe_load(f) or {}

    if resume:
        config["resume_from_checkpoint"] = True
    else:
        config.pop("resume_from_checkpoint", None)

    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


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


# ---------------------------------------------------------------------------
# Local model utilities
# ---------------------------------------------------------------------------

def is_local_model(model: str) -> bool:
    """Check whether a model specifier is a local filesystem path."""
    return Path(model).is_dir()


_WEIGHT_FILES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)

_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
)


def validate_local_model(model_path: str) -> list[str]:
    """Validate that a local model directory has the required files.

    Returns a list of error strings (empty means the directory is valid).
    """
    errors: list[str] = []
    p = Path(model_path)

    if not (p / "config.json").is_file():
        errors.append("Missing config.json (required for model architecture).")

    if not any((p / f).is_file() for f in _WEIGHT_FILES):
        errors.append(
            "Missing model weights. Expected at least one of: "
            + ", ".join(_WEIGHT_FILES)
        )

    if not any((p / f).is_file() for f in _TOKENIZER_FILES):
        errors.append(
            "Missing tokenizer files. Expected at least one of: "
            + ", ".join(_TOKENIZER_FILES)
        )

    return errors


# Maps config.json "model_type" values to LlamaFactory chat templates.
_MODEL_TYPE_MAP = {
    "llama": "llama3",
    "qwen2": "qwen",
    "qwen": "qwen",
    "gemma": "gemma",
    "gemma2": "gemma",
    "mistral": "mistral",
    "phi": "phi",
    "phi3": "phi",
    "deepseek": "deepseek",
}


def detect_template_from_config(model_path: str) -> str:
    """Detect the LlamaFactory chat template from a local model's config.json.

    Reads the ``model_type`` field and maps it to a known template name.
    Returns ``"default"`` if detection fails.
    """
    config_file = Path(model_path) / "config.json"
    try:
        with open(config_file) as f:
            config = json.load(f)
        model_type = config.get("model_type", "")
        return _MODEL_TYPE_MAP.get(model_type, "default")
    except Exception:
        return "default"
