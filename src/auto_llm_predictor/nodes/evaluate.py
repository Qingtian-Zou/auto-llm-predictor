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


def _compute_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
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

    if len(labels) == 2:
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


def run_evaluation(state: PipelineState) -> dict:
    """Evaluate predictions on the test set.

    Writes: eval_results, messages
    """
    test_pred_path = state.get("test_predictions_path", "")
    target_mapping = state.get("target_mapping", {})
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

        y_true = []
        y_pred = []

        for entry in predictions:
            # LlamaFactory predict format: {"predict": "...", "label": "..."}
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
            # If the user-provided a mapped target list, we use that. Else the unique truth labels.
            if "UNPARSED" not in all_labels:
                all_labels_with_unparsed = all_labels + ["UNPARSED"]
            else:
                all_labels_with_unparsed = all_labels

            metrics = _compute_metrics(y_true, y_pred, all_labels_with_unparsed)
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

    # Build summary message
    summary_parts = []
    for split, metrics in results.items():
        if "accuracy" in metrics:
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
