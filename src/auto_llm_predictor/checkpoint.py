"""Pipeline state checkpoint — save/load state for resuming experiments."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_STATE_FILE = ".pipeline_state.json"

# Fields to exclude from serialization (non-JSON-serializable or transient)
_SKIP_FIELDS = {"messages"}


def save_state(state: dict[str, Any], output_dir: str) -> str:
    """Save pipeline state to JSON in the output directory.

    Returns the path to the saved state file.
    """
    state_path = Path(output_dir) / _STATE_FILE

    # Filter out non-serializable fields
    serializable = {}
    for key, value in state.items():
        if key in _SKIP_FIELDS:
            continue
        try:
            json.dumps(value)  # test serializability
            serializable[key] = value
        except (TypeError, ValueError):
            logger.debug("Skipping non-serializable field: %s", key)

    state_path.write_text(json.dumps(serializable, indent=2))
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

    state["messages"] = []  # fresh message list for the new session
    logger.info("Loaded pipeline state from %s (%d fields)", state_path, len(state))
    return state
