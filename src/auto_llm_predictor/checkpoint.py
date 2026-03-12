"""Pipeline state checkpoint — save/load state for resuming experiments."""

from __future__ import annotations

import json
import logging
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


def save_state(state: dict[str, Any], output_dir: str) -> str:
    """Save pipeline state to JSON in the output directory.

    Returns the path to the saved state file.
    """
    state_path = Path(output_dir) / _STATE_FILE

    # Filter out non-serializable fields and normalize path values
    serializable = {}
    for key, value in state.items():
        if key in _SKIP_FIELDS:
            continue
        try:
            json.dumps(value)  # test serializability
            if key in _PATH_FIELDS and isinstance(value, str) and value:
                value = normalize_path(value)
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

    # Normalize any path fields that may contain mixed separators
    for key in _PATH_FIELDS:
        val = state.get(key)
        if isinstance(val, str) and val:
            state[key] = normalize_path(val)

    state["messages"] = []  # fresh message list for the new session
    logger.info("Loaded pipeline state from %s (%d fields)", state_path, len(state))
    return state
