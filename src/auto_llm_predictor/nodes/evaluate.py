"""Node: run_evaluation — Compute metrics from predictions."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from langchain_core.messages import HumanMessage

from auto_llm_predictor.state import PipelineState
from auto_llm_predictor.utils import load_jsonl

logger = logging.getLogger(__name__)


def _extract_label(text: str, target_mapping: dict[str, str]) -> str | None:
    """Extract a label from model output using the target mapping.

    Tries exact match first, then case-insensitive prefix/substring match.
    Returns the mapped label string, or None if unparseable.
    """
    text = text.strip()
    labels = list(target_mapping.values())

    # Exact match
    for label in labels:
        if text.lower() == label.lower():
            return label

    # Prefix match
    for label in labels:
        if text.lower().startswith(label.lower()):
            return label

    # Substring match
    for label in labels:
        if label.lower() in text.lower():
            return label

    # If no match in mapping, but we have text, return the raw text
    # This handles empty target_mappings or unmapped string categories
    if text:
        return text

    return None


def _compute_metrics(y_true: list[str], y_pred: list[str], labels: list[str], task_type: str = "") -> dict:
    """Compute classification metrics."""
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    # Convert to indices for sklearn
    label_to_idx = {l: i for i, l in enumerate(labels)}
    y_true_idx = [label_to_idx.get(y, -1) for y in y_true]
    y_pred_idx = [label_to_idx.get(y, -1) for y in y_pred]

    # Filter valid
    valid = [(t, p) for t, p in zip(y_true_idx, y_pred_idx) if t >= 0 and p >= 0]
    if not valid:
        return {"error": "No valid predictions to evaluate"}

    yt, yp = zip(*valid)

    results = {
        "total_samples": len(y_true),
        "valid_predictions": len(valid),
        "invalid_predictions": len(y_true) - len(valid),
        "accuracy": accuracy_score(yt, yp),
    }

    # Use explicit task_type when available; fall back to label count
    is_binary = task_type == "binary" if task_type else len(labels) == 2
    if is_binary:
        results["f1"] = f1_score(yt, yp, average="binary", pos_label=label_to_idx[labels[0]])
    else:
        results["macro_f1"] = f1_score(yt, yp, average="macro")
        results["weighted_f1"] = f1_score(yt, yp, average="weighted")

    cm = confusion_matrix(yt, yp, labels=list(range(len(labels))))
    results["confusion_matrix"] = cm.tolist()
    results["labels"] = labels

    # Classification report as a dict
    report = classification_report(
        yt, yp, labels=list(range(len(labels))),
        target_names=labels, output_dict=True, zero_division=0,
    )
    results["classification_report"] = report

    return results


def _compute_regression_metrics(y_true: list[float], y_pred: list[float]) -> dict:
    """Compute regression metrics (MAE, MSE, RMSE, R²)."""
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )
    import math

    mse = mean_squared_error(y_true, y_pred)
    return {
        "total_samples": len(y_true),
        "valid_predictions": len(y_true),
        "invalid_predictions": 0,
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "r2": r2_score(y_true, y_pred),
    }


def run_evaluation(state: PipelineState) -> dict:
    """Evaluate predictions on the test set.

    Writes: eval_results, messages
    """
    test_pred_path = state.get("test_predictions_path", "")
    target_mapping = state.get("target_mapping", {})
    task_type = state.get("task_type", "")
    labels = list(target_mapping.values())

    results = {}

    for split, pred_path_key in [("train", "train_predictions_path"),
                                  ("test", "test_predictions_path")]:
        pred_path = state.get(pred_path_key, "")
        if not pred_path or not Path(pred_path).exists():
            logger.warning("No predictions file for %s split.", split)
            continue

        predictions = load_jsonl(pred_path)
        logger.info("Loaded %d predictions for %s set", len(predictions), split)

        if task_type == "regression":
            # ── Regression evaluation ──────────────────────────────
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

            # Drop pairs where prediction is NaN
            valid_pairs = [
                (t, p) for t, p in zip(y_true_f, y_pred_f)
                if not (p != p)  # NaN check
            ]
            if valid_pairs:
                yt, yp = zip(*valid_pairs)
                metrics = _compute_regression_metrics(list(yt), list(yp))
                metrics["invalid_predictions"] = len(y_true_f) - len(valid_pairs)
                metrics["total_samples"] = len(predictions)
                results[split] = metrics
                logger.info("%s evaluation: MAE=%.4f, R²=%.4f",
                            split, metrics["mae"], metrics["r2"])
            else:
                results[split] = {"error": "No valid numeric predictions to evaluate"}
        else:
            # ── Classification evaluation ──────────────────────────
            y_true = []
            y_pred = []

            for entry in predictions:
                raw_pred = entry.get("predict", entry.get("prediction", ""))
                raw_label = entry.get("label", entry.get("ground_truth", ""))

                true_label = _extract_label(raw_label, target_mapping)
                pred_label = _extract_label(raw_pred, target_mapping)

                if true_label is not None:
                    y_true.append(true_label)
                    y_pred.append(pred_label if pred_label is not None else "UNPARSED")

            if y_true:
                # If target_mapping was empty, infer the valid labels from the ground truth
                all_labels = labels if labels else sorted(list(set(y_true)))

                # Identify valid labels based on the true distribution
                if "UNPARSED" not in all_labels:
                    all_labels_with_unparsed = all_labels + ["UNPARSED"]
                else:
                    all_labels_with_unparsed = all_labels

                metrics = _compute_metrics(y_true, y_pred, all_labels_with_unparsed, task_type=task_type)
                results[split] = metrics
                logger.info("%s evaluation: accuracy=%.4f", split, metrics.get("accuracy", 0))
            else:
                results[split] = {"error": "No valid labels found in predictions"}

    # Save evaluation results
    run_dir = Path(state.get("run_dir", state["output_dir"]))
    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_path = eval_dir / "results.json"
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved evaluation results to %s", eval_path)

    # Print results immediately so they're visible before optional XAI
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    for split, metrics in results.items():
        if isinstance(metrics, dict) and "error" not in metrics:
            print(f"\n{split.upper()} Results:")
            if task_type == "regression":
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
    print("=" * 60)

    # Build summary message
    summary_parts = []
    for split, metrics in results.items():
        if "error" not in metrics:
            if task_type == "regression":
                summary_parts.append(
                    f"{split}: MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}, "
                    f"valid={metrics['valid_predictions']}/{metrics['total_samples']}"
                )
            else:
                summary_parts.append(
                    f"{split}: accuracy={metrics['accuracy']:.4f}, "
                    f"valid={metrics['valid_predictions']}/{metrics['total_samples']}"
                )
    summary = "; ".join(summary_parts) if summary_parts else "No valid evaluations"

    return {
        "eval_results": results,
        "messages": [
            HumanMessage(content=f"[run_evaluation] {summary}. Results at {eval_path}"),
        ],
    }
