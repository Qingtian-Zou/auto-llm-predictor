"""Node: run_finetuning — Execute LlamaFactory fine-tuning."""

from __future__ import annotations

import logging
from pathlib import Path

from langgraph.types import Command
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from auto_llm_predictor.state import PipelineState
from auto_llm_predictor.utils import (
    find_latest_checkpoint,
    run_llamafactory,
    set_resume_in_yaml,
)

logger = logging.getLogger(__name__)


def run_finetuning(state: PipelineState, config: RunnableConfig) -> dict:
    """Run LlamaFactory SFT fine-tuning with auto-resume on failure.

    When fine-tuning fails and checkpoints exist in the output directory,
    ``resume_from_checkpoint: true`` is added to the training YAML and
    the process is retried.  When no checkpoints exist, the training is
    retried from scratch to handle transient errors.

    Writes: finetune_succeeded, finetune_attempts, messages
    """
    yaml_path = state["lmf_train_yaml"]
    sft_dir = state.get("adapter_path", "")
    configurable = config.get("configurable", {})
    log_callback = configurable.get("log_callback")
    cancelled_check = configurable.get("cancelled_check")

    tc = state.get("training_config", {})
    max_retries = tc.get("finetune_max_retries", 3)

    attempt = 0
    success = False
    output = ""

    while attempt <= max_retries:
        attempt += 1
        is_retry = attempt > 1

        # Check for user cancellation before retrying
        if is_retry and cancelled_check and cancelled_check():
            logger.info("Fine-tuning cancelled by user — skipping retry.")
            break

        if is_retry:
            latest_ckpt = find_latest_checkpoint(sft_dir)
            if latest_ckpt:
                logger.info(
                    "Retrying fine-tuning (attempt %d/%d) — resuming from %s",
                    attempt, max_retries + 1, latest_ckpt,
                )
                set_resume_in_yaml(yaml_path, resume=True)

                print(f"\n{'=' * 60}")
                print(f"FINE-TUNING RETRY (attempt {attempt}/{max_retries + 1})")
                print(f"Resuming from checkpoint: {Path(latest_ckpt).name}")
                print(f"{'=' * 60}\n", flush=True)

                if log_callback:
                    log_callback(
                        f"Fine-tuning retry {attempt}/{max_retries + 1} — "
                        f"resuming from {Path(latest_ckpt).name}"
                    )
            else:
                logger.info(
                    "Retrying fine-tuning from scratch (attempt %d/%d) — "
                    "no checkpoints found",
                    attempt, max_retries + 1,
                )
                set_resume_in_yaml(yaml_path, resume=False)

                print(f"\n{'=' * 60}")
                print(f"FINE-TUNING RETRY (attempt {attempt}/{max_retries + 1})")
                print("Restarting from scratch (no checkpoints available)")
                print(f"{'=' * 60}\n", flush=True)

                if log_callback:
                    log_callback(
                        f"Fine-tuning retry {attempt}/{max_retries + 1} — "
                        f"restarting from scratch"
                    )
        else:
            logger.info("Starting fine-tuning with config: %s", yaml_path)

            print("\n" + "=" * 60)
            print("FINE-TUNING — llamafactory-cli train")
            print(f"Config: {yaml_path}")
            print("=" * 60 + "\n", flush=True)

        success, output = run_llamafactory(
            yaml_path,
            timeout=7200,
            stream=True,
            log_callback=log_callback,
        )

        if success:
            print("\n" + "=" * 60)
            print("FINE-TUNING COMPLETE ✓")
            print("=" * 60 + "\n", flush=True)
            logger.info("Fine-tuning completed successfully (attempt %d).", attempt)
            break

        # Failed
        logger.error("Fine-tuning failed (attempt %d):\n%s", attempt, output[-2000:])

        if attempt > max_retries:
            print("\n" + "=" * 60)
            print("FINE-TUNING FAILED ✗ (all retries exhausted)")
            print("=" * 60 + "\n", flush=True)

    # Clean up: remove resume_from_checkpoint from YAML so the saved
    # config stays clean for potential manual re-runs
    if attempt > 1:
        try:
            set_resume_in_yaml(yaml_path, resume=False)
        except Exception:
            pass  # non-critical cleanup

    return {
        "finetune_succeeded": success,
        "finetune_attempts": attempt,
        "messages": [
            HumanMessage(
                content=f"[run_finetuning] {'SUCCESS' if success else 'FAILED'} "
                f"after {attempt} attempt(s). "
                f"Adapter at: {state.get('adapter_path', 'unknown')}. "
                f"Output tail: {output[-300:]}"
            ),
        ],
    }
