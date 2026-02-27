"""Node: review_prep_data — Human-in-the-loop review of prepared data.

Uses LangGraph's ``interrupt()`` to pause execution and present the
prepared data summary to the user.  The user can:
  - Approve and continue to fine-tuning
  - Request changes (drop features, change target mapping, etc.)
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.types import interrupt

from auto_llm_predictor.checkpoint import save_state
from auto_llm_predictor.state import PipelineState

logger = logging.getLogger(__name__)


# ── Plan review ───────────────────────────────────────────────────


def _build_plan_summary(state: PipelineState) -> str:
    """Build a human-readable summary of the LLM preparation plan for review."""
    plan_json = state.get("prep_plan", "{}")
    try:
        plan = json.loads(plan_json)
    except json.JSONDecodeError:
        plan = {}

    parts = [
        "=" * 60,
        "DATA PREPARATION PLAN REVIEW",
        "=" * 60,
        "",
        f"Target column:   {state.get('target_column', 'N/A')}",
        f"Task type:       {state.get('task_type', 'N/A')}",
        f"Target mapping:  {json.dumps(state.get('target_mapping', {}), indent=2)}",
        "",
    ]

    # Selected features
    features = plan.get("selected_features", state.get("selected_features", []))
    parts.append(f"Selected features ({len(features)}):")
    for feat in features:
        parts.append(f"  - {feat}")

    # Dropped features (single line)
    dropped = plan.get("dropped_features", state.get("dropped_features", []))
    if dropped:
        parts.extend(["", f"Dropped features ({len(dropped)}): {', '.join(dropped)}"])

    parts.extend([
        "",
        f"Feature selection reason: {plan.get('feature_selection_reason', 'N/A')}",
        "",
        "-" * 40,
        "INSTRUCTION TEMPLATE",
        "-" * 40,
        plan.get("instruction_template", "(not set)"),
        "",
        f"Input format:    {plan.get('input_format', 'N/A')}",
        f"Output format:   {plan.get('output_format', 'N/A')}",
        f"Balance strategy: {plan.get('balance_strategy', 'none')}",
        f"Test ratio:      {plan.get('test_ratio', 0.2)}",
    ])

    cleaning = plan.get("data_cleaning_steps", [])
    if cleaning:
        parts.append("")
        parts.append("Data cleaning steps:")
        for i, step in enumerate(cleaning, 1):
            parts.append(f"  {i}. {step}")

    reasoning = plan.get("reasoning", "")
    if reasoning:
        parts.extend(["", f"Reasoning: {reasoning}"])

    parts.extend([
        "",
        "=" * 60,
        "Please review the above plan and respond with one of:",
        "  'approve'   — Proceed to code generation",
        "  Or provide feedback to revise the plan, e.g.:",
        "    'drop features: patient_id, smoker'",
        "    'change instruction to: Predict the drug response...'",
        "    'change target mapping: 0=No Response, 1=Response'",
        "    'use oversample'",
        "=" * 60,
    ])

    return "\n".join(parts)


def review_prep_plan(state: PipelineState) -> dict:
    """Pause for human review of the data preparation plan.

    Uses ``interrupt()`` to present the plan summary to the user and wait
    for approval, revision feedback, or a directly-edited plan JSON.
    """
    summary = _build_plan_summary(state)

    user_response = interrupt(summary)

    user_response = str(user_response).strip()
    approved = user_response.lower() in ("approve", "approved", "ok", "yes", "y", "lgtm", "")

    # Check if the user submitted an edited plan JSON directly
    if not approved and user_response.startswith("{"):
        try:
            edited_plan = json.loads(user_response)
            logger.info("User submitted edited plan JSON directly")
            # Extract state-level variables that may have been injected
            # for editing (keeps CLI and webUI in sync)
            state_updates: dict = {
                # Treat manual JSON edits as feedback so it routes back to plan_preparation
                # for LLM validation and instruction_template regeneration.
                "user_feedback": (
                    "I have manually edited the preparation plan JSON (e.g. changed target mapping "
                    "or selected features). Please verify the new plan. Crucially, strictly ensure "
                    "your 'instruction_template' perfectly reflects any new target_mapping or features "
                    "I just provided."
                ),
            }
            # Extract state-level variables
            for key in ("target_column", "target_mapping", "task_type", "selected_features", "dropped_features"):
                if key in edited_plan:
                    state_updates[key] = edited_plan.pop(key)
            
            # The LLM will regenerate the plan, but we can pass the edited one
            # forward so it's in the message history for context.
            state_updates["prep_plan"] = json.dumps(edited_plan)
            state_updates["messages"] = [
                HumanMessage(
                    content="[review_prep_plan] User edited the plan directly. "
                    "Routing back to plan_preparation for validation and instruction update."
                ),
            ]
            return state_updates
        except json.JSONDecodeError:
            pass  # Not valid JSON — treat as text feedback

    logger.info("User plan review response: %s (approved=%s)", user_response[:100], approved)

    return {
        "user_feedback": user_response if not approved else "",
        "messages": [
            HumanMessage(
                content=f"[review_prep_plan] User {'approved' if approved else 'requested changes'}: "
                + (user_response[:200] if not approved else "Proceeding to code generation.")
            ),
        ],
    }


def route_after_plan_review(state: PipelineState) -> str:
    """Conditional edge: route based on plan review.

    Returns 'write_prep_code' if approved,
    'plan_preparation' to re-plan with user feedback.
    """
    feedback = state.get("user_feedback", "")
    if not feedback:
        return "write_prep_code"
    return "plan_preparation"


# ── Data review ───────────────────────────────────────────────────


def _build_review_summary(state: PipelineState) -> str:
    """Build a human-readable summary of the prepared data for review."""
    output_dir = Path(state["output_dir"])
    data_dir = output_dir / "data"

    parts = [
        "=" * 60,
        "DATA PREPARATION REVIEW",
        "=" * 60,
        "",
        f"CSV source:      {state.get('csv_path', 'N/A')}",
    ]

    test_csv = state.get("test_csv_path", "")
    if test_csv:
        parts.append(f"Test CSV:        {test_csv}")

    parts.extend([
        f"Target column:   {state.get('target_column', 'N/A')}",
        f"Task type:       {state.get('task_type', 'N/A')}",
        f"Target mapping:  {json.dumps(state.get('target_mapping', {}), indent=2)}",
        "",
        f"Selected features ({len(state.get('selected_features', []))}):",
    ])

    for feat in state.get("selected_features", []):
        parts.append(f"  - {feat}")
        
    dropped_features = state.get("dropped_features", [])
    if dropped_features:
        parts.extend([
            "",
            f"Dropped features ({len(dropped_features)}): {', '.join(dropped_features)}"
        ])

    # Show class distributions from generated files
    all_data_path = data_dir / "all_data.json"
    test_data_path = data_dir / "test_data.json"

    if all_data_path.exists():
        with open(all_data_path) as f:
            all_data = json.load(f)
        label = "Train data" if test_csv else "All data"
        parts.append(f"\n{label}: {len(all_data)} examples")

        label_counts = Counter(entry.get("output", "") for entry in all_data)
        total = sum(label_counts.values())
        parts.append(f"{label} class distribution:")
        for lbl, count in sorted(label_counts.items(), key=lambda x: -x[1]):
            pct = 100.0 * count / total if total else 0
            parts.append(f"  {lbl}: {count} ({pct:.1f}%)")

    if test_data_path.exists():
        with open(test_data_path) as f:
            test_data = json.load(f)
        parts.append(f"\nTest data: {len(test_data)} examples")

        label_counts = Counter(entry.get("output", "") for entry in test_data)
        total = sum(label_counts.values())
        parts.append("Test data class distribution:")
        for lbl, count in sorted(label_counts.items(), key=lambda x: -x[1]):
            pct = 100.0 * count / total if total else 0
            parts.append(f"  {lbl}: {count} ({pct:.1f}%)")

    # Show a sample entry
    if all_data_path.exists() and all_data:
        parts.append("\nSample entry (all_data[0]):")
        parts.append(json.dumps(all_data[0], indent=2))

    # Show the generated script path
    script_path = state.get("prep_code_path", "")
    if script_path:
        parts.append(f"\nGenerated script: {script_path}")
        
    # Show the automated LLM data verification feedback
    verification_feedback = state.get("prep_data_verification", "")
    if verification_feedback:
        parts.extend([
            "",
            "-" * 40,
            "AUTOMATED DATA VERIFICATION (LLM CRITIQUE)",
            "-" * 40,
            verification_feedback
        ])

    parts.extend([
        "",
        "=" * 60,
        "Please review the above and respond with one of:",
        "  'approve'   — Continue to splitting & config generation",
        "  'balance'   — Proceed to balance training data (oversample/undersample)",
        "  Or provide feedback to revise the data preparation, e.g.:",
        "    'drop features: patient_id, smoker'",
        "    'change target mapping: 0=No Response, 1=Response'",
        "    'change instruction to: Predict the drug response...'",
        "=" * 60,
    ])

    return "\n".join(parts)


def review_prep_data(state: PipelineState) -> dict:
    """Pause for human review of the prepared data.

    Uses ``interrupt()`` to present a summary to the user and wait
    for approval or revision feedback.

    Returns
    -------
    dict with ``user_feedback`` and routing info.
    """
    summary = _build_review_summary(state)

    # Persist state so this experiment can be resumed with --start-from
    save_state(state, state["output_dir"])

    # This pauses the graph and returns the summary to the caller.
    # When the user resumes, ``interrupt()`` returns their response.
    user_response = interrupt(summary)

    user_response = str(user_response).strip()
    approved = user_response.lower() in ("approve", "approved", "ok", "yes", "y", "lgtm", "")

    logger.info("User review response: %s (approved=%s)", user_response[:100], approved)

    return {
        "user_feedback": user_response if not approved else "",
        "messages": [
            HumanMessage(
                content=f"[review_prep_data] User {'approved' if approved else 'requested changes'}: "
                + (user_response[:200] if not approved else "Proceeding to fine-tuning.")
            ),
        ],
    }


def route_after_review(state: PipelineState) -> str:
    """Conditional edge: route based on user review.

    Returns 'split_data' if approved (skip balancing),
    'write_balance_code' if user wants to balance,
    'plan_preparation' to re-plan with user feedback.
    """
    feedback = state.get("user_feedback", "")
    if not feedback:
        # Approved — skip balancing, go to split
        return "split_data"
    if feedback.lower() in ("balance", "balance data", "yes balance", "oversample", "undersample"):
        return "write_balance_code"
    # Check for explicit balance request in free-text
    if re.search(r'\b(balance|oversample|undersample)\b', feedback, re.IGNORECASE):
        return "write_balance_code"
    return "plan_preparation"


# ── Config review ─────────────────────────────────────────────────

# Parameters that only apply to the training config
_TRAIN_ONLY_KEYS = {
    "lora_rank", "lora_alpha", "lora_dropout", "lora_target", "use_dora",
    "num_train_epochs", "learning_rate", "lr_scheduler_type",
    "warmup_ratio", "gradient_accumulation_steps",
    "per_device_train_batch_size", "save_steps", "save_strategy",
    "save_total_limit", "logging_steps", "logging_dir",
    "val_size", "eval_steps", "eval_strategy", "plot_loss",
    "report_to", "ddp_timeout",
}

# Key parameters to highlight in the review summary
_KEY_PARAMS = [
    "lora_rank", "lora_alpha", "lora_dropout", "use_dora",
    "num_train_epochs", "learning_rate", "lr_scheduler_type",
    "warmup_ratio", "per_device_train_batch_size",
    "gradient_accumulation_steps", "cutoff_len",
    "val_size", "eval_steps",
    "save_steps", "save_strategy", "save_total_limit",
    "bf16", "flash_attn", "quantization_bit", "report_to",
]


def _extract_yaml_values(yaml_path: str) -> dict[str, str]:
    """Extract key-value pairs from a YAML file."""
    import re
    result = {}
    path = Path(yaml_path)
    if not path.exists():
        return result
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^(\S+)\s*:\s*(.+)$", line)
        if m:
            result[m.group(1)] = m.group(2)
    return result


def _build_config_summary(state: PipelineState) -> str:
    """Build a human-readable summary of the LlamaFactory configs."""
    parts = [
        "=" * 60,
        "LLAMAFACTORY CONFIG REVIEW",
        "=" * 60,
        "",
    ]

    # Show key tunable parameters from the train config
    train_yaml_path = state.get("lmf_train_yaml", "")
    if train_yaml_path:
        values = _extract_yaml_values(train_yaml_path)
        parts.append("KEY TRAINING PARAMETERS:")
        parts.append("-" * 40)
        for param in _KEY_PARAMS:
            val = values.get(param, "(not set)")
            parts.append(f"  {param}: {val}")
        parts.append("")

    # Show full configs
    for label, key in [
        ("TRAIN CONFIG", "lmf_train_yaml"),
        ("PREDICT-TRAIN CONFIG", "lmf_predict_train_yaml"),
        ("PREDICT-TEST CONFIG", "lmf_predict_test_yaml"),
    ]:
        yaml_path = state.get(key, "")
        if yaml_path and Path(yaml_path).exists():
            parts.append(f"── {label}: {yaml_path}")
            parts.append(Path(yaml_path).read_text())
        else:
            parts.append(f"── {label}: (not generated)")
        parts.append("")

    parts.extend([
        "=" * 60,
        "Please review the configs and respond with one of:",
        "  'approve'  — Start fine-tuning with these settings",
        "",
        "  Or specify changes as key: value pairs, e.g.:",
        "    'lora_rank: 32'",
        "    'num_train_epochs: 5'",
        "    'learning_rate: 1.0e-5'",
        "    'per_device_train_batch_size: 4'",
        "    'lora_rank: 32, num_train_epochs: 5'",
        "",
        "  Configurable training parameters:",
        "    lora_rank, lora_alpha, lora_dropout, lora_target, use_dora,",
        "    num_train_epochs, learning_rate, lr_scheduler_type, warmup_ratio,",
        "    per_device_train_batch_size, gradient_accumulation_steps,",
        "    save_steps, save_strategy, save_total_limit, logging_steps,",
        "    val_size, eval_steps, plot_loss, report_to, ddp_timeout",
        "",
        "  Configurable shared parameters (applied to all YAMLs):",
        "    cutoff_len, bf16, fp16, flash_attn, quantization_bit,",
        "    per_device_eval_batch_size, preprocessing_num_workers, template",
        "=" * 60,
    ])

    return "\n".join(parts)


def _parse_overrides(text: str) -> dict[str, str]:
    """Parse user input like 'lora_rank: 32, num_train_epochs: 5' into a dict."""
    import re
    overrides = {}
    # Split on comma or newline, then parse key: value pairs
    for part in re.split(r"[,\n]", text):
        part = part.strip()
        if ":" in part:
            key, _, value = part.partition(":")
            key = key.strip()
            value = value.strip()
            if key and value:
                overrides[key] = value
    return overrides


def _coerce_value(value: str):
    """Try to coerce a string value to int, float, or bool."""
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def review_lmf_config(state: PipelineState) -> dict:
    """Pause for human review of LlamaFactory configs.

    Users can approve or specify hyperparameter overrides.
    Overrides are merged into ``training_config`` and the pipeline
    regenerates all YAML files from scratch.

    Writes: config_feedback, training_config, messages
    """
    summary = _build_config_summary(state)

    save_state(state, state["output_dir"])

    user_response = interrupt(summary)
    user_response = str(user_response).strip()
    approved = user_response.lower() in ("approve", "approved", "ok", "yes", "y", "lgtm", "")

    result: dict = {
        "config_feedback": user_response if not approved else "",
        "messages": [
            HumanMessage(
                content=f"[review_lmf_config] User {'approved' if approved else 'requested changes'}: "
                + (user_response[:200] if not approved else "Proceeding to fine-tuning.")
            ),
        ],
    }

    if not approved:
        # Check if the user submitted an edited YAML directly
        if "model_name_or_path:" in user_response or "### model" in user_response:
            train_yaml_path = state.get("lmf_train_yaml", "")
            if train_yaml_path and Path(train_yaml_path).exists():
                with open(train_yaml_path, "w") as f:
                    f.write(user_response)

                # Parse the edited YAML to sync state variables
                state_updates: dict = {
                    "config_feedback": "",  # clear so routing goes to finetuning
                    "messages": [
                        HumanMessage(
                            content="[review_lmf_config] User edited the config directly. "
                            "Proceeding to fine-tuning with updated config."
                        ),
                    ],
                }
                try:
                    import yaml
                    edited_config = yaml.safe_load(user_response) or {}
                    # Extract base_model
                    if "model_name_or_path" in edited_config:
                        state_updates["base_model"] = edited_config["model_name_or_path"]
                    # Sync training_config with any recognized training params
                    tc = dict(state.get("training_config", {}))
                    _tc_keys = set(_TRAIN_ONLY_KEYS) | {
                        "cutoff_len", "bf16", "fp16", "flash_attn",
                        "quantization_bit", "per_device_eval_batch_size",
                        "preprocessing_num_workers", "template",
                    }
                    for k in _tc_keys:
                        if k in edited_config:
                            tc[k] = edited_config[k]
                    state_updates["training_config"] = tc
                except Exception:
                    logger.warning("Could not parse edited YAML to sync state", exc_info=True)

                logger.info("User edited train config directly")
                return state_updates

        # Parse overrides and merge into training_config
        overrides = _parse_overrides(user_response)
        if overrides:
            tc = dict(state.get("training_config", {}))
            for key, value in overrides.items():
                tc[key] = _coerce_value(value)
            result["training_config"] = tc
            logger.info("Updated training_config with overrides: %s — will regenerate YAMLs",
                        list(overrides.keys()))
        else:
            logger.info("User provided feedback but no parseable overrides: %s",
                        user_response[:100])

    logger.info("Config review: %s (approved=%s)", user_response[:100], approved)

    return result


def route_after_config_review(state: PipelineState) -> str:
    """Conditional edge: route based on config review.

    Returns 'run_finetuning' if approved,
    'generate_lmf_config' to regenerate with updated training_config.
    """
    feedback = state.get("config_feedback", "")
    if not feedback:
        return "run_finetuning"
    return "generate_lmf_config"


# ── Balanced data review ──────────────────────────────────────────


def _build_balance_summary(state: PipelineState) -> str:
    """Build a summary of the balanced training data."""
    output_dir = Path(state["output_dir"])
    data_dir = output_dir / "data"
    train_path = data_dir / "train.json"
    test_path = data_dir / "test.json"

    plan = json.loads(state.get("prep_plan", "{}"))
    balance_strategy = plan.get("balance_strategy", "none")

    parts = [
        "=" * 60,
        "BALANCED DATA REVIEW",
        "=" * 60,
        "",
        f"Balance strategy: {balance_strategy}",
        "",
    ]

    if train_path.exists():
        with open(train_path) as f:
            train_data = json.load(f)
        parts.append(f"Train set: {len(train_data)} examples")

        # Show class distribution
        label_counts = Counter(entry.get("output", "") for entry in train_data)
        total = sum(label_counts.values())
        parts.append("Train class distribution:")
        for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
            pct = 100.0 * count / total if total else 0
            parts.append(f"  {label}: {count} ({pct:.1f}%)")

    if test_path.exists():
        with open(test_path) as f:
            test_data = json.load(f)
        parts.append(f"\nTest set: {len(test_data)} examples")

        label_counts = Counter(entry.get("output", "") for entry in test_data)
        total = sum(label_counts.values())
        parts.append("Test class distribution:")
        for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
            pct = 100.0 * count / total if total else 0
            parts.append(f"  {label}: {count} ({pct:.1f}%)")

    # Show balance result output
    balance_result = state.get("balance_result", "")
    if balance_result:
        parts.append(f"\nBalance script output:\n{balance_result[:1000]}")

    parts.extend([
        "",
        "=" * 60,
        "Please review the balanced data and respond with one of:",
        "  'approve'  — Continue to LlamaFactory config generation",
        "  Or provide feedback to re-balance, e.g.:",
        "    'use undersample instead'",
        "    'use oversample'",
        "    'balance_strategy: none'",
        "=" * 60,
    ])

    return "\n".join(parts)


def review_balanced_data(state: PipelineState) -> dict:
    """Pause for human review of the balanced data.

    Uses ``interrupt()`` to present the class distribution and let
    the user approve or request a different balancing strategy.

    Writes: balance_feedback, messages
    """
    summary = _build_balance_summary(state)

    save_state(state, state["output_dir"])
    user_response = interrupt(summary)
    user_response = str(user_response).strip()
    approved = user_response.lower() in ("approve", "approved", "ok", "yes", "y", "lgtm", "")

    logger.info("Balance review response: %s (approved=%s)", user_response[:100], approved)

    return {
        "balance_feedback": user_response if not approved else "",
        "messages": [
            HumanMessage(
                content=f"[review_balanced_data] User {'approved' if approved else 'requested changes'}: "
                + (user_response[:200] if not approved else "Proceeding to config.")
            ),
        ],
    }


def route_after_balance_review(state: PipelineState) -> str:
    """Conditional edge: route based on balance review.

    Returns 'split_data' if approved,
    'write_balance_code' to re-balance with feedback.
    """
    feedback = state.get("balance_feedback", "")
    if not feedback:
        return "split_data"
    return "write_balance_code"
