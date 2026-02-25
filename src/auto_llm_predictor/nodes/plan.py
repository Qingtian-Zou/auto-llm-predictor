"""Node: plan_preparation — Decide on features, prompt format, and cleaning."""

from __future__ import annotations

import json
import logging
import re

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from auto_llm_predictor.prompts.plan import PLAN_SYSTEM, format_plan_prompt
from auto_llm_predictor.state import PipelineState

logger = logging.getLogger(__name__)


def _apply_feedback_overrides(plan: dict, feedback: str) -> dict:
    """Parse and programmatically enforce common user feedback patterns.

    This runs *after* the LLM generates its plan, applying hard overrides
    so that user requests are guaranteed to be respected even if the LLM
    didn't fully comply.

    Supported patterns (case-insensitive):
        - "drop features: col1, col2"  /  "remove features: col1, col2"
        - "add features: col1, col2"   /  "include features: col1, col2"
        - "keep only features: col1, col2"
        - "balance: oversample"  /  "use undersample"  /  "balance_strategy: none"
        - "test_ratio: 0.3"  /  "test ratio: 0.3"
        - "instruction: <new instruction text>"
    """
    fb = feedback.lower()

    features = list(plan.get("selected_features", []))
    dropped = list(plan.get("dropped_features", []))

    # ── Drop / remove features ────────────────────────────────
    drop_match = re.search(
        r"(?:drop|remove|exclude)\s+features?\s*[:\-]\s*(.+?)(?:\n|$)",
        fb, re.IGNORECASE,
    )
    if drop_match:
        to_drop = {f.strip().lower() for f in drop_match.group(1).split(",")}
        before = len(features)
        
        # move matched features to dropped
        for f in features:
            if f.lower() in to_drop and f not in dropped:
                dropped.append(f)
                
        features = [f for f in features if f.lower() not in to_drop]
        logger.info("User requested drop features: removed %d → %d", before, len(features))

    # ── Add / include features ────────────────────────────────
    add_match = re.search(
        r"(?:add|include)\s+features?\s*[:\-]\s*(.+?)(?:\n|$)",
        fb, re.IGNORECASE,
    )
    if add_match:
        to_add = [f.strip() for f in add_match.group(1).split(",")]
        existing = {f.lower() for f in features}
        for f in to_add:
            if f.lower() not in existing:
                features.append(f)
                logger.info("User requested add feature: %s", f)
            # Remove it from dropped if it's there
            dropped = [d for d in dropped if d.lower() != f.lower()]

    # ── Keep only specific features ───────────────────────────
    keep_match = re.search(
        r"keep\s+only\s+features?\s*[:\-]\s*(.+?)(?:\n|$)",
        fb, re.IGNORECASE,
    )
    if keep_match:
        to_keep = {f.strip().lower() for f in keep_match.group(1).split(",")}
        
        # Everything not kept gets moved to dropped
        for f in features:
            if f.lower() not in to_keep and f not in dropped:
                dropped.append(f)
                
        features = [f for f in features if f.lower() in to_keep]
        logger.info("User requested keep only: %d features", len(features))

    plan["selected_features"] = features
    plan["dropped_features"] = dropped

    # ── Balance strategy ──────────────────────────────────────
    balance_match = re.search(
        r"(?:balance(?:_strategy)?|(?:use\s+))(oversample|undersample|none)",
        fb, re.IGNORECASE,
    )
    if balance_match:
        strategy = balance_match.group(1).lower()
        plan["balance_strategy"] = strategy
        logger.info("User requested balance strategy: %s", strategy)

    # ── Test ratio ────────────────────────────────────────────
    ratio_match = re.search(
        r"test[\s_]ratio\s*[:\-]\s*([\d.]+)",
        fb, re.IGNORECASE,
    )
    if ratio_match:
        plan["test_ratio"] = float(ratio_match.group(1))
        logger.info("User requested test ratio: %s", plan["test_ratio"])

    # ── Instruction template ──────────────────────────────────
    instr_match = re.search(
        r"(?:change\s+)?instruction(?:\s+(?:to|template))?\s*[:\-]\s*(.+?)(?:\n|$)",
        fb, re.IGNORECASE,
    )
    if instr_match:
        new_instruction = instr_match.group(1).strip().strip("'\"")
        if len(new_instruction) > 10:  # Avoid false positives
            plan["instruction_template"] = new_instruction
            logger.info("User requested instruction: %s", new_instruction[:80])

    return plan


def plan_preparation(state: PipelineState, *, llm) -> dict:
    """Ask the LLM to create a data preparation plan based on the data profile.

    If ``selected_features`` is already populated (e.g. from ensemble feature
    selection), the LLM is told to use those features and only decide on
    formatting, cleaning, and balancing.

    After the LLM responds, any explicit user feedback is programmatically
    enforced via ``_apply_feedback_overrides`` to guarantee compliance.

    Writes: selected_features, prep_plan, messages
    """
    # Count columns from the profile
    df = pd.read_csv(state["csv_path"], nrows=0, low_memory=False)
    n_columns = len(df.columns)

    pre_selected = state.get("selected_features", [])

    user_prompt = format_plan_prompt(
        data_profile=state["data_profile"],
        target_column=state["target_column"],
        task_type=state["task_type"],
        target_mapping=state["target_mapping"],
        n_columns=n_columns,
        pre_selected_features=pre_selected,
    )

    # If the user provided feedback from a review cycle, append it
    user_feedback = state.get("user_feedback", "")
    if user_feedback:
        user_prompt += (
            f"\n\n=== USER FEEDBACK (MUST FOLLOW) ===\n"
            f"The user reviewed the previous data preparation and requested changes:\n"
            f"{user_feedback}\n"
            f"Please revise the plan to incorporate this feedback.\n"
        )

    messages = [
        SystemMessage(content=PLAN_SYSTEM),
        HumanMessage(content=user_prompt),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()
    logger.info("LLM plan response: %s", raw[:500])

    # Extract JSON object using regex (handles conversational text and missing fences)
    json_match = re.search(r"(\{.*\})", raw, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        json_str = raw.strip()

    try:
        plan = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse LLM plan as JSON. String was:\n%s", json_str)
        raise e

    # If features were pre-selected by ensemble and no user feedback,
    # preserve them exactly.
    if pre_selected and not user_feedback:
        plan["selected_features"] = pre_selected
        plan["feature_selection_reason"] = (
            f"Features pre-selected by ensemble feature selection "
            f"({len(pre_selected)} features). "
            + plan.get("feature_selection_reason", "")
        )

    # Programmatically enforce explicit user feedback
    if user_feedback:
        plan = _apply_feedback_overrides(plan, user_feedback)

    # Apply hard overrides for human feedback
    # (e.g., if LLM ignored a 'drop features' command)
    feedback_history = state.get("prep_feedback")
    if feedback_history:
        plan = _apply_feedback_overrides(plan, feedback_history[-1])

    # If ensemble selection ran, preserve its original dropped_features since they 
    # override LLM hallucations unless the user explicitly overrode them.
    final_dropped = plan.get("dropped_features", [])
    if pre_selected and not plan.get("dropped_features"): # Assuming 'preset_features' in snippet refers to 'pre_selected'
        final_dropped = state.get("dropped_features", [])

    prep_plan = json.dumps(plan, indent=2)
    logger.info("Prep plan: selected %d features, balance=%s",
                len(plan["selected_features"]), plan.get("balance_strategy", "none"))

    return {
        "prep_plan": json.dumps(plan, indent=2),
        "selected_features": plan["selected_features"],
        "dropped_features": final_dropped,
        "target_mapping": plan.get("target_mapping", state.get("target_mapping", {})),
        "messages": [
            HumanMessage(content="[plan_preparation] Plan generated."),
        ],
    }
