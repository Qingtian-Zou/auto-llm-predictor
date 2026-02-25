"""Node: run_prediction — Execute LlamaFactory prediction."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.messages import HumanMessage

from auto_llm_predictor.state import PipelineState
from auto_llm_predictor.utils import run_llamafactory

logger = logging.getLogger(__name__)


def run_prediction(state: PipelineState) -> dict:
    """Run LlamaFactory prediction on both train and test sets.

    Writes: train_predictions_path, test_predictions_path, messages
    """
    run_dir = Path(state.get("run_dir", state["output_dir"]))

    results = {}
    for split, yaml_key, pred_dir_name in [
        ("train", "lmf_predict_train_yaml", "predict_train"),
        ("test", "lmf_predict_test_yaml", "predict_test"),
    ]:
        yaml_path = state.get(yaml_key)
        if not yaml_path:
            logger.warning("No YAML config for %s prediction, skipping.", split)
            continue

        logger.info("Running prediction on %s set with config: %s", split, yaml_path)

        print("\n" + "=" * 60)
        print(f"PREDICTION ({split.upper()}) — llamafactory-cli train")
        print(f"Config: {yaml_path}")
        print("=" * 60 + "\n", flush=True)

        success, output = run_llamafactory(yaml_path, timeout=3600, stream=True)

        pred_path = run_dir / pred_dir_name / "generated_predictions.jsonl"
        if success and pred_path.exists():
            results[f"{split}_predictions_path"] = str(pred_path)
            print(f"\n✓ {split.capitalize()} predictions saved to {pred_path}\n", flush=True)
            logger.info("%s predictions saved to %s", split.capitalize(), pred_path)
        else:
            print(f"\n✗ {split.capitalize()} prediction failed\n", flush=True)
            logger.error("%s prediction failed:\n%s", split.capitalize(), output[-1000:])
            results[f"{split}_predictions_path"] = ""

    return {
        **results,
        "messages": [
            HumanMessage(
                content=f"[run_prediction] Train: {bool(results.get('train_predictions_path'))}, "
                f"Test: {bool(results.get('test_predictions_path'))}"
            ),
        ],
    }
