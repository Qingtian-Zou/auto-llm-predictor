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

"""Standalone baseline evaluation module for the auto-LLM-predictor.

Runs prediction and evaluation using a **non-finetuned** (baseline) model
on data prepared by a previous training run.  This enables comparing
finetuned vs. baseline performance.

The model can be any HuggingFace model — it does not have to match the
one used during training.  Results are stored in the output directory
(not inside a specific run folder).

Both a programmatic API (``run_baseline_evaluation``) and a CLI entry
point (``auto-llm-predictor-baseline``) are provided.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections.abc import Callable
from pathlib import Path

from auto_llm_predictor.utils import normalize_path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_pipeline_state(output_dir: str) -> dict:
    """Load the saved pipeline state from a training output directory."""
    from auto_llm_predictor.checkpoint import load_state

    return load_state(output_dir)


def _guess_template(model_name: str) -> str:
    """Guess the chat template from the model name."""
    from auto_llm_predictor.nodes.config import _guess_template

    return _guess_template(model_name)


def _sanitize_model_name(model_name: str) -> str:
    """Sanitize a model name for use as a directory name."""
    return re.sub(r"[/\\:*?\"<>|]", "_", model_name)


# ---------------------------------------------------------------------------
# YAML generation
# ---------------------------------------------------------------------------

def generate_baseline_yaml(
    *,
    base_model: str,
    data_dir: str,
    dataset_name: str,
    template: str,
    cutoff_len: int,
    output_dir: str,
    precision: str = "bf16",
    flash_attn: str = "auto",
    quantization_bit: int | None = None,
) -> str:
    """Generate a LlamaFactory predict YAML for baseline (no adapter).

    Returns the path to the saved YAML file.
    """
    from auto_llm_predictor.utils import save_yaml

    quant_line = f"quantization_bit: {quantization_bit}\n" if quantization_bit else ""

    yaml_content = f"""\
### model
model_name_or_path: {base_model}
trust_remote_code: true
{quant_line}
### method
stage: sft
do_predict: true

### dataset
dataset_dir: {data_dir}
dataset: {dataset_name}
eval_dataset: {dataset_name}
template: {template}
cutoff_len: {cutoff_len}
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: {output_dir}
overwrite_output_dir: true

### eval
per_device_eval_batch_size: 1
predict_with_generate: true
{precision}: true
flash_attn: {flash_attn}
"""
    config_dir = Path(output_dir) / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = str(config_dir / f"baseline_predict_{dataset_name}.yaml")
    save_yaml(yaml_content, yaml_path)
    logger.info("Generated baseline YAML at %s", yaml_path)
    return yaml_path


# ---------------------------------------------------------------------------
# Evaluation helpers (reuse from nodes/evaluate.py)
# ---------------------------------------------------------------------------

def _evaluate_predictions(pred_path: str, target_mapping: dict,
                          task_type: str) -> dict:
    """Evaluate a predictions JSONL file and return metrics dict."""
    from auto_llm_predictor.nodes.evaluate import (
        _compute_metrics,
        _compute_regression_metrics,
        _extract_label,
    )
    from auto_llm_predictor.utils import load_jsonl

    predictions = load_jsonl(pred_path)
    logger.info("Loaded %d predictions for evaluation", len(predictions))

    if task_type == "regression":
        y_true_f: list[float] = []
        y_pred_f: list[float] = []

        for entry in predictions:
            raw_pred = entry.get("predict", entry.get("prediction", "")).strip()
            raw_label = entry.get("label", entry.get("ground_truth", "")).strip()
            try:
                y_true_f.append(float(raw_label))
            except (ValueError, TypeError):
                continue
            try:
                y_pred_f.append(float(raw_pred))
            except (ValueError, TypeError):
                y_pred_f.append(float("nan"))

        valid_pairs = [
            (t, p) for t, p in zip(y_true_f, y_pred_f)
            if not (p != p)  # NaN check
        ]
        if valid_pairs:
            yt, yp = zip(*valid_pairs)
            metrics = _compute_regression_metrics(list(yt), list(yp))
            metrics["invalid_predictions"] = len(y_true_f) - len(valid_pairs)
            metrics["total_samples"] = len(predictions)
            return metrics
        return {"error": "No valid numeric predictions to evaluate"}

    # Classification
    labels = list(target_mapping.values())
    y_true: list[str] = []
    y_pred: list[str] = []

    for entry in predictions:
        raw_pred = entry.get("predict", entry.get("prediction", ""))
        raw_label = entry.get("label", entry.get("ground_truth", ""))

        true_label = _extract_label(raw_label, target_mapping)
        pred_label = _extract_label(raw_pred, target_mapping)

        if true_label is not None:
            y_true.append(true_label)
            y_pred.append(pred_label if pred_label is not None else "UNPARSED")

    if y_true:
        all_labels = labels if labels else sorted(set(y_true))
        if "UNPARSED" not in all_labels:
            all_labels_with_unparsed = all_labels + ["UNPARSED"]
        else:
            all_labels_with_unparsed = all_labels
        return _compute_metrics(y_true, y_pred, all_labels_with_unparsed,
                                task_type=task_type)

    return {"error": "No valid labels found in predictions"}


# ---------------------------------------------------------------------------
# Core baseline evaluation function
# ---------------------------------------------------------------------------

def run_baseline_evaluation(
    *,
    output_dir: str,
    model: str | None = None,
    baseline_dir: str | None = None,
    precision: str = "bf16",
    flash_attn: str = "auto",
    quantization_bit: int | None = None,
    splits: list[str] | None = None,
    log_callback: Callable[[str], None] | None = None,
) -> dict:
    """Run baseline (non-finetuned) prediction and evaluation.

    Parameters
    ----------
    output_dir : str
        Training output directory containing ``.pipeline_state.json``
        and ``data/`` with prepared datasets.
    model : str, optional
        HuggingFace model ID or local path.  Defaults to the model
        used during training (from pipeline state).
    baseline_dir : str, optional
        Custom output directory for baseline results.  Defaults to
        ``output_dir/baseline_<sanitized_model_name>/``.
    precision : str
        Precision for inference (``bf16`` or ``fp16``).
    flash_attn : str
        Flash attention setting (``auto``, ``fa2``, or ``disabled``).
    quantization_bit : int, optional
        Quantization bit width (4 or 8), or None for no quantization.
    splits : list[str], optional
        Data splits to evaluate.  Defaults to ``["test"]``.
    log_callback : callable, optional
        Callback for progress messages (used by web UI SSE).

    Returns
    -------
    dict
        Keys: ``results`` (per-split metrics), ``results_path``,
        ``model``, ``baseline_dir``.
    """
    from auto_llm_predictor.utils import run_llamafactory

    def _log(msg: str) -> None:
        logger.info(msg)
        if log_callback:
            log_callback(msg)

    if splits is None:
        splits = ["test"]

    # 1. Load pipeline state
    _log("Loading pipeline state...")
    state = _load_pipeline_state(output_dir)

    task_type = state.get("task_type", "")
    target_mapping = state.get("target_mapping", {})
    cutoff_len = state.get("cutoff_len") or state.get("training_config", {}).get("cutoff_len", 4096)
    data_dir = str(Path(output_dir) / "data")

    # 2. Resolve model
    if not model:
        model = state.get("base_model", "")
        if not model:
            raise ValueError("No model specified and no base_model found in pipeline state.")
        _log(f"Using model from training run: {model}")
    else:
        _log(f"Using user-specified model: {model}")

    # 3. Guess template
    template = _guess_template(model)
    _log(f"Using template: {template}")

    # 4. Resolve baseline output directory
    if not baseline_dir:
        sanitized = _sanitize_model_name(model)
        baseline_dir = str(Path(output_dir) / f"baseline_{sanitized}")
    Path(baseline_dir).mkdir(parents=True, exist_ok=True)
    _log(f"Baseline output directory: {baseline_dir}")

    # 5. Run prediction and evaluation for each split
    all_results: dict[str, dict] = {}

    for split in splits:
        _log(f"--- Running baseline prediction on {split} split ---")

        predict_output = str(Path(baseline_dir) / f"predict_{split}")

        # Generate YAML
        yaml_path = generate_baseline_yaml(
            base_model=model,
            data_dir=data_dir,
            dataset_name=split,
            template=template,
            cutoff_len=cutoff_len,
            output_dir=predict_output,
            precision=precision,
            flash_attn=flash_attn,
            quantization_bit=quantization_bit,
        )
        _log(f"Generated baseline YAML: {yaml_path}")

        # Run LlamaFactory prediction
        _log(f"Running LlamaFactory prediction on {split} split...")
        success, ret_code, output_tail = run_llamafactory(
            yaml_path, timeout=86400, stream=True,
            log_callback=log_callback, idle_timeout=300,
        )

        if not success:
            msg = f"Baseline prediction failed on {split} split (exit code {ret_code})"
            _log(msg)
            all_results[split] = {"error": msg, "output_tail": output_tail}
            continue

        # Find predictions file
        pred_path = str(Path(predict_output) / "generated_predictions.jsonl")
        if not Path(pred_path).exists():
            msg = f"Predictions file not found: {pred_path}"
            _log(msg)
            all_results[split] = {"error": msg}
            continue

        _log(f"Evaluating {split} predictions...")
        metrics = _evaluate_predictions(pred_path, target_mapping, task_type)
        all_results[split] = metrics

        # Log key metrics
        if "error" not in metrics:
            if task_type == "regression":
                _log(f"{split}: MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}")
            else:
                _log(f"{split}: accuracy={metrics.get('accuracy', 0):.4f}, "
                     f"valid={metrics.get('valid_predictions', 0)}/{metrics.get('total_samples', 0)}")

    # 6. Save results
    eval_dir = Path(baseline_dir) / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    results_path = str(eval_dir / "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    _log(f"Saved baseline evaluation results to {results_path}")

    return {
        "results": all_results,
        "results_path": results_path,
        "model": model,
        "baseline_dir": baseline_dir,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """CLI entry point for baseline evaluation."""
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run baseline (non-finetuned) model evaluation on prepared data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Training output directory (must contain .pipeline_state.json and data/).",
    )
    parser.add_argument(
        "--model", default=None,
        help="HuggingFace model ID or local path. Defaults to the model from the training run.",
    )
    parser.add_argument(
        "--baseline-dir", default=None,
        help="Custom output directory for baseline results. "
             "Defaults to output_dir/baseline_<model_name>/.",
    )
    parser.add_argument(
        "--precision", default="bf16", choices=["bf16", "fp16"],
        help="Precision for inference (default: bf16).",
    )
    parser.add_argument(
        "--quantization-bit", type=int, default=None, choices=[4, 8],
        help="Quantization bit width (4 or 8). None for no quantization.",
    )
    parser.add_argument(
        "--flash-attn", default="auto",
        help="Flash attention setting (default: auto).",
    )
    parser.add_argument(
        "--splits", default="test",
        help="Comma-separated data splits to evaluate (default: test). "
             "E.g.: test, train, or test,train.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    output_dir = normalize_path(str(Path(args.output_dir).resolve()))
    if not Path(output_dir).exists():
        print(f"Error: Output directory not found: {output_dir}", file=sys.stderr)
        sys.exit(1)

    baseline_dir = None
    if args.baseline_dir:
        baseline_dir = normalize_path(str(Path(args.baseline_dir).resolve()))

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        print("Error: No valid splits specified.", file=sys.stderr)
        sys.exit(1)

    try:
        result = run_baseline_evaluation(
            output_dir=output_dir,
            model=args.model,
            baseline_dir=baseline_dir,
            precision=args.precision,
            flash_attn=args.flash_attn,
            quantization_bit=args.quantization_bit,
            splits=splits,
        )
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        logger.debug("Full traceback:", exc_info=True)
        sys.exit(1)

    # Print summary
    print("\n" + "=" * 60)
    print("Baseline Evaluation Results")
    print(f"Model: {result['model']}")
    print("=" * 60)

    for split, metrics in result["results"].items():
        if isinstance(metrics, dict) and "error" not in metrics:
            print(f"\n{split.upper()} Results:")
            if "mae" in metrics:
                print(f"  MAE:                {metrics['mae']:.4f}")
                print(f"  MSE:                {metrics['mse']:.4f}")
                print(f"  RMSE:               {metrics['rmse']:.4f}")
                print(f"  R²:                 {metrics['r2']:.4f}")
                print(f"  Valid predictions:   {metrics['valid_predictions']}/{metrics['total_samples']}")
            else:
                print(f"  Accuracy:           {metrics['accuracy']:.4f}")
                print(f"  Valid predictions:   {metrics['valid_predictions']}/{metrics['total_samples']}")
                if "f1" in metrics:
                    print(f"  F1 Score:           {metrics['f1']:.4f}")
                if "macro_f1" in metrics:
                    print(f"  Macro F1:           {metrics['macro_f1']:.4f}")
                    print(f"  Weighted F1:        {metrics['weighted_f1']:.4f}")
        elif isinstance(metrics, dict) and "error" in metrics:
            print(f"\n{split.upper()}: ERROR — {metrics['error']}")

    print("=" * 60)
    print(f"\nResults saved to: {result['results_path']}")
