"""Node: verify_prepared_data — Automated LLM review of prepared JSON data."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from auto_llm_predictor.prompts.verify import VERIFY_DATA_SYSTEM, VERIFY_DATA_USER
from auto_llm_predictor.state import PipelineState

logger = logging.getLogger(__name__)


def _sample_jsonl(file_path: Path, num_samples: int = 3) -> str:
    """Read a JSON array file and return a formatted string of random samples."""
    if not file_path.exists():
        return "Not provided."
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        if not data:
            return "File exists but is empty."
            
        samples = random.sample(data, min(len(data), num_samples))
        return json.dumps(samples, indent=2)
    except Exception as e:
        logger.warning(f"Failed to read samples from {file_path}: {e}")
        return f"Error reading file: {e}"


def verify_prepared_data(state: PipelineState, *, llm) -> dict:
    """Use the LLM to verify the structure and consistency of the generated JSON data.
    
    This node acts as an automated second opinion before the human-in-the-loop review.
    
    Writes: prep_data_verification, messages
    """
    logger.info("Executing automated LLM verification of prepared JSON data.")
    
    data_dir = Path(state["output_dir"]) / "data"
    all_data_path = data_dir / "all_data.json"
    test_data_path = data_dir / "test_data.json"
    
    # Extract properties
    task_desc = state.get("prep_plan", "Task description not provided in state.")
    target_mapping = state.get("target_mapping", {})
    
    # Grab 3 random samples
    train_samples_str = _sample_jsonl(all_data_path, num_samples=3)
    test_samples_str = _sample_jsonl(test_data_path, num_samples=3)
        
    user_prompt = VERIFY_DATA_USER.format(
        task_description=task_desc,
        target_mapping=json.dumps(target_mapping, indent=2),
        train_samples=train_samples_str,
        test_samples=test_samples_str
    )
    
    messages = [
        SystemMessage(content=VERIFY_DATA_SYSTEM),
        HumanMessage(content=user_prompt)
    ]
    
    # Execute LLM to verify the data
    try:
        response = llm.invoke(messages)
        critique = response.content.strip()
        logger.info("Data verification complete. LLM Critique:\n%s", critique)
    except Exception as e:
        logger.error("Failed to execute automated test verification: %s", e)
        critique = f"Verification failed due to LLM error: {e}"
        
    return {
        "prep_data_verification": critique,
        "messages": [
            HumanMessage(
                content=f"[verify_prepared_data] Verification generated ({len(critique)} chars)."
            )
        ]
    }
