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

"""Node: register_dataset — Register the prepared train/test JSON files with
LlamaFactory.

Once the early ``split_input_csv`` node has split the input CSV and the
LLM-generated ``prepare_data.py`` has produced ``all_data.json`` (from the
train half) and ``test_data.json`` (from the test half), this node simply
copies them to the canonical ``train.json`` / ``test.json`` filenames and
writes the ``dataset_info.json`` registry that LlamaFactory consumes.

The actual train/test splitting happens upstream in ``split_input_csv`` —
this node only handles registration.
"""

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


def register_dataset(state: PipelineState) -> dict:
    """Copy all_data.json / test_data.json → train.json / test.json and write
    dataset_info.json for LlamaFactory.

    Writes: train_data_path, test_data_path, dataset_info_path, messages
    """
    data_dir = Path(state["output_dir"]) / "data"
    all_data_path = Path(state.get("all_data_path", data_dir / "all_data.json"))
    test_data_path = data_dir / "test_data.json"
    train_path = data_dir / "train.json"
    test_path = data_dir / "test.json"

    if not test_data_path.exists():
        raise FileNotFoundError(
            f"Expected test_data.json at {test_data_path} — was the input CSV "
            f"split by split_input_csv and processed by prepare_data.py? "
            f"This node assumes the two-CSV pipeline is in effect."
        )

    shutil.copy2(all_data_path, train_path)
    shutil.copy2(test_data_path, test_path)

    with open(all_data_path) as f:
        train_data = json.load(f)
    with open(test_data_path) as f:
        test_data = json.load(f)

    logger.info(
        "Registered dataset with LlamaFactory: train=%d, test=%d.",
        len(train_data), len(test_data),
    )
    summary = (
        f"Registered prepared data with LlamaFactory.\n"
        f"{_class_distribution_str(train_data, 'Train')}\n"
        f"{_class_distribution_str(test_data, 'Test')}"
    )

    info_path = data_dir / "dataset_info.json"
    dataset_info = {
        "train": {"file_name": "train.json"},
        "test": {"file_name": "test.json"},
    }
    with open(info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)

    print(f"\n{'=' * 50}\nDATASET REGISTRATION\n{'=' * 50}\n{summary}", flush=True)

    return {
        "train_data_path": str(train_path),
        "test_data_path": str(test_path),
        "dataset_info_path": str(info_path),
        "messages": [
            HumanMessage(content=f"[register_dataset] {summary}"),
        ],
    }
