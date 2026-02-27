"""Node: run_finetuning — Execute LlamaFactory fine-tuning."""

from __future__ import annotations

import logging

from langgraph.types import Command
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from auto_llm_predictor.state import PipelineState
from auto_llm_predictor.utils import run_llamafactory

logger = logging.getLogger(__name__)


def run_finetuning(state: PipelineState, config: RunnableConfig) -> dict:
    """Run LlamaFactory SFT fine-tuning.

    Writes: adapter_path, messages
    """
    yaml_path = state["lmf_train_yaml"]
    logger.info("Starting fine-tuning with config: %s", yaml_path)

    log_callback = config.get("configurable", {}).get("log_callback")

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
        logger.info("Fine-tuning completed successfully.")
    else:
        print("\n" + "=" * 60)
        print("FINE-TUNING FAILED ✗")
        print("=" * 60 + "\n", flush=True)
        logger.error("Fine-tuning failed:\n%s", output[-2000:])

    return {
        "finetune_succeeded": success,
        "messages": [
            HumanMessage(
                content=f"[run_finetuning] {'SUCCESS' if success else 'FAILED'}. "
                f"Adapter at: {state.get('adapter_path', 'unknown')}. "
                f"Output tail: {output[-300:]}"
            ),
        ],
    }
