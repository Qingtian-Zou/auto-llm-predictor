"""Pipeline state checkpoint — save/load state for resuming experiments."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from auto_llm_predictor.utils import normalize_path

logger = logging.getLogger(__name__)

_STATE_FILE = ".pipeline_state.json"

# Fields to exclude from serialization (non-JSON-serializable or transient)
_SKIP_FIELDS = {"messages"}

# State fields that contain filesystem paths and should be normalized
_PATH_FIELDS = {
    "csv_path", "test_csv_path", "output_dir", "run_dir",
    "adapter_path", "prep_code_path", "all_data_path",
    "train_data_path", "test_data_path", "dataset_info_path",
    "lmf_train_yaml", "lmf_predict_train_yaml", "lmf_predict_test_yaml",
    "train_predictions_path", "test_predictions_path",
}

# Path fields that refer to external files (not under output_dir) — keep absolute
_EXTERNAL_PATH_FIELDS = {"csv_path", "test_csv_path", "output_dir"}

# Path fields that are always under output_dir — stored as relative for portability
_RELATIVE_PATH_FIELDS = _PATH_FIELDS - _EXTERNAL_PATH_FIELDS


def save_state(state: dict[str, Any], output_dir: str) -> str:
    """Save pipeline state to JSON in the output directory.

    Internal paths (under ``output_dir``) are stored as relative paths for
    portability across different mount points and machines.  External paths
    (``csv_path``, ``test_csv_path``) remain absolute.

    Returns the path to the saved state file.
    """
    state_path = Path(output_dir) / _STATE_FILE
    resolved_output = normalize_path(str(Path(output_dir).resolve()))

    # Filter out non-serializable fields and normalize path values
    serializable = {}
    for key, value in state.items():
        if key in _SKIP_FIELDS:
            continue
        try:
            json.dumps(value)  # test serializability
            if key in _PATH_FIELDS and isinstance(value, str) and value:
                value = normalize_path(value)
                # Convert internal paths to relative (relative to output_dir)
                if key in _RELATIVE_PATH_FIELDS and os.path.isabs(value):
                    try:
                        rel = os.path.relpath(value, resolved_output)
                        # Only use relative if it stays under output_dir
                        # (doesn't start with ".." traversals that leave it)
                        if not rel.startswith(".."):
                            value = rel
                    except ValueError:
                        pass  # different drives on Windows — keep absolute
            serializable[key] = value
        except (TypeError, ValueError):
            logger.debug("Skipping non-serializable field: %s", key)

    # Write atomically: temp file + rename to avoid corruption on crash
    state_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=state_path.parent, suffix=".tmp")
    try:
        with open(fd, "w") as f:
            json.dump(serializable, f, indent=2)
        Path(tmp_path).replace(state_path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise
    logger.info("Saved pipeline state to %s (%d fields)", state_path, len(serializable))
    return str(state_path)


def load_state(output_dir: str) -> dict[str, Any]:
    """Load pipeline state from a previous experiment's output directory.

    Relative paths (stored by ``save_state``) are resolved against the
    user-provided ``output_dir``.  Absolute paths from older checkpoints
    are normalized as before — this ensures backward compatibility.

    Returns the deserialized state dict with an empty ``messages`` list.

    Raises
    ------
    FileNotFoundError
        If no saved state file exists in the directory.
    """
    state_path = Path(output_dir) / _STATE_FILE
    if not state_path.exists():
        raise FileNotFoundError(
            f"No saved pipeline state found at {state_path}. "
            f"Run the pipeline at least once before using --start-from."
        )

    try:
        state = json.loads(state_path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Pipeline state file is corrupted: {state_path}. "
            f"JSON parse error: {e}. "
            f"Delete the file and re-run the pipeline from scratch, "
            f"or fix the JSON manually."
        ) from e

    resolved_output = normalize_path(str(Path(output_dir).resolve()))

    for key in _PATH_FIELDS:
        val = state.get(key)
        if not isinstance(val, str) or not val:
            continue
        if key in _RELATIVE_PATH_FIELDS and not os.path.isabs(val):
            # Relative path — resolve against user-provided output_dir
            state[key] = normalize_path(
                str(Path(resolved_output) / val),
            )
        else:
            # Absolute path (external or legacy checkpoint) — normalize only
            state[key] = normalize_path(val)

    state["messages"] = []  # fresh message list for the new session
    logger.info("Loaded pipeline state from %s (%d fields)", state_path, len(state))
    return state
