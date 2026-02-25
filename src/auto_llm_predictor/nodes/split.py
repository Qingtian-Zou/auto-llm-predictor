"""Node: split_data — Deterministic train/test splitting."""

from __future__ import annotations

import json
import logging
import shutil
from collections import Counter
from pathlib import Path

from langchain_core.messages import HumanMessage

from auto_llm_predictor.state import PipelineState

logger = logging.getLogger(__name__)


def _class_distribution_str(data: list[dict], label: str) -> str:
    """Return a formatted class distribution string."""
    counts = Counter(entry.get("output", "") for entry in data)
    total = sum(counts.values())
    lines = [f"{label}: {total} examples"]
    for cls, count in sorted(counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / total if total else 0
        lines.append(f"  {cls}: {count} ({pct:.1f}%)")
    return "\n".join(lines)


def split_data(state: PipelineState) -> dict:
    """Split or assign train/test data.

    - If ``test_data.json`` exists (separate test CSV was provided),
      copies ``all_data.json`` → ``train.json`` and
      ``test_data.json`` → ``test.json`` (no splitting).
    - Otherwise, performs stratified train/test splitting using
      ``test_ratio`` from the preparation plan.

    Writes: train_data_path, test_data_path, dataset_info_path, messages
    """
    data_dir = Path(state["output_dir"]) / "data"
    all_data_path = Path(state.get("all_data_path", data_dir / "all_data.json"))
    test_data_path = data_dir / "test_data.json"
    train_path = data_dir / "train.json"
    test_path = data_dir / "test.json"

    with open(all_data_path) as f:
        all_data = json.load(f)

    has_separate_test = test_data_path.exists()

    if has_separate_test:
        # ── Two-CSV mode: no splitting needed ──────────────────────
        shutil.copy2(all_data_path, train_path)
        shutil.copy2(test_data_path, test_path)

        with open(test_data_path) as f:
            test_data = json.load(f)

        logger.info(
            "Two-CSV mode: train=%d, test=%d (no splitting)",
            len(all_data), len(test_data),
        )
        summary = (
            f"Using separate test CSV — no splitting.\n"
            f"{_class_distribution_str(all_data, 'Train')}\n"
            f"{_class_distribution_str(test_data, 'Test')}"
        )
    else:
        # ── Single-CSV mode: stratified split ──────────────────────
        plan = json.loads(state.get("prep_plan", "{}"))
        tc = state.get("training_config", {})
        test_ratio = tc.get("test_ratio", plan.get("test_ratio", 0.2))
        task_type = state.get("task_type", "binary")

        from sklearn.model_selection import train_test_split

        labels = [entry.get("output", "") for entry in all_data]

        try:
            if task_type != "regression":
                train_data, test_data = train_test_split(
                    all_data,
                    test_size=test_ratio,
                    random_state=42,
                    stratify=labels,
                )
            else:
                train_data, test_data = train_test_split(
                    all_data,
                    test_size=test_ratio,
                    random_state=42,
                )
        except ValueError:
            # Fallback: random split if stratification fails (rare classes)
            logger.warning("Stratified split failed, using random split.")
            train_data, test_data = train_test_split(
                all_data,
                test_size=test_ratio,
                random_state=42,
            )

        with open(train_path, "w") as f:
            json.dump(train_data, f, indent=2)
        with open(test_path, "w") as f:
            json.dump(test_data, f, indent=2)

        logger.info(
            "Split %d examples → train=%d, test=%d (ratio=%.2f)",
            len(all_data), len(train_data), len(test_data), test_ratio,
        )
        summary = (
            f"Split {len(all_data)} examples (test_ratio={test_ratio}).\n"
            f"{_class_distribution_str(train_data, 'Train')}\n"
            f"{_class_distribution_str(test_data, 'Test')}"
        )

    # Update dataset_info.json for LlamaFactory
    info_path = data_dir / "dataset_info.json"
    dataset_info = {
        "train": {"file_name": "train.json"},
        "test": {"file_name": "test.json"},
    }
    with open(info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)

    print(f"\n{'=' * 50}\nTRAIN/TEST SPLIT\n{'=' * 50}\n{summary}", flush=True)

    return {
        "train_data_path": str(train_path),
        "test_data_path": str(test_path),
        "dataset_info_path": str(info_path),
        "messages": [
            HumanMessage(content=f"[split_data] {summary}"),
        ],
    }
