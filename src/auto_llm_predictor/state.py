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

"""Pipeline state schema for the LangGraph auto-LLM-predictor."""

from __future__ import annotations

from typing import Annotated, Any

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class PipelineState(TypedDict):
    """Shared state that flows through every node in the graph.

    Fields are grouped by pipeline stage.  Nodes only write the fields
    they "own" and read whatever they need.
    """

    # ── Inputs (set once at graph invocation) ──────────────────────
    csv_path: str
    """Absolute path to the raw CSV file (training data, or all data if no test CSV)."""

    test_csv_path: str
    """Optional absolute path to a separate test CSV file."""

    target_column: str
    """Column name of the prediction target (may be empty if auto-detected)."""

    base_model: str
    """HuggingFace model ID or local path for fine-tuning, e.g. 'Qwen/Qwen2.5-7B-Instruct' or '/path/to/model'."""

    output_dir: str
    """Root directory for all generated artefacts."""

    training_config: dict[str, Any]
    """Training hyperparameters from CLI (lora_rank, epochs, learning_rate, etc.)."""

    start_from: str
    """Pipeline entry point (e.g. 'explore_data', 'review_prep', 'split', 'config')."""

    # ── Data exploration ───────────────────────────────────────────
    data_profile: str
    """LLM-readable text summary of the CSV (columns, types, stats, samples)."""

    selected_features: list[str]
    """Column names chosen as input features for the LLM prompt."""
    
    dropped_features: list[str]
    """Column names explicitly omitted from prediction."""

    target_mapping: dict[str, str]
    """Maps raw target values → human-readable labels used in prompts."""

    task_type: str
    """'binary', 'multiclass', or 'regression'."""

    exploration_steps: list[dict]
    """Log of tool calls made by the exploration agent during data investigation."""

    data_quality_notes: str
    """Data quality notes produced by the exploration agent."""

    # ── Code generation ────────────────────────────────────────────
    prep_plan: str
    """Natural-language description of the data preparation strategy."""

    prep_code: str
    """Generated Python script source code."""

    prep_code_path: str
    """Path where the generated script was saved."""

    prep_result: str
    """stdout + stderr captured from running the script."""

    prep_succeeded: bool
    """Whether the prep script executed successfully."""

    prep_attempts: int
    """Number of code-generation attempts (for retry limiting)."""
    
    prep_data_verification: str
    """LLM critique of the generated Alpaca JSON data formats."""

    # ── JSON paths (produced by the prep script) ──────────────────
    all_data_path: str
    """Path to all_data.json (main CSV → Alpaca format, pre-split)."""

    train_data_path: str
    """Path to train.json (created by split_data)."""

    test_data_path: str
    """Path to test.json (created by split_data)."""

    dataset_info_path: str
    """Path to dataset_info.json for LlamaFactory."""

    # ── Run directory (timestamped per experiment run) ──────────────
    run_dir: str
    """Timestamped subdirectory for configs, sft, predictions, evaluation."""

    # ── Cutoff length ──────────────────────────────────────────────
    cutoff_len: int
    """Resolved cutoff length (set by determine_cutoff_len; overrides training_config value)."""

    auto_cutoff: bool
    """Whether to auto-detect cutoff_len from training data statistics (--auto-cutoff flag)."""

    # ── LlamaFactory configuration ─────────────────────────────────
    lmf_train_yaml: str
    """Path to the generated train YAML config."""

    lmf_predict_train_yaml: str
    """Path to the generated predict-on-train YAML config."""

    lmf_predict_test_yaml: str
    """Path to the generated predict-on-test YAML config."""

    # ── Training & evaluation ──────────────────────────────────────
    adapter_path: str
    """Path to the LoRA adapter directory after fine-tuning."""

    finetune_succeeded: bool
    """Whether fine-tuning completed successfully."""

    finetune_attempts: int
    """Number of fine-tuning attempts including auto-resume retries."""

    train_predictions_path: str
    """Path to predictions JSONL on the training set."""

    test_predictions_path: str
    """Path to predictions JSONL on the test set."""

    eval_results: dict[str, Any]
    """Evaluation metrics dictionary (accuracy, F1, etc.)."""


    # ── Balancing ──────────────────────────────────────────────────
    balance_code: str
    """Generated Python script for balancing training data."""

    balance_code_path: str
    """Path where the balance script was saved."""

    balance_result: str
    """stdout + stderr from running the balance script."""

    balance_succeeded: bool
    """Whether the balance script executed successfully."""

    balance_attempts: int
    """Number of balance code-generation attempts."""

    # ── Debug agent ─────────────────────────────────────────────
    debug_diagnosis: str
    """Structured diagnosis from the debug agent when code generation fails."""

    debug_tool_calls: list[dict]
    """Log of tool calls made by the debug agent during failure investigation."""

    # ── Human-in-the-loop ────────────────────────────────────────
    user_feedback: str
    """User revision feedback from the data review breakpoint (empty if approved)."""

    balance_feedback: str
    """User revision feedback from the balance review breakpoint (empty if approved)."""

    config_feedback: str
    """User revision feedback from the config review breakpoint (empty if approved)."""

    # ── LLM reasoning chain ────────────────────────────────────────
    messages: Annotated[list, add_messages]
    """LangChain message list used for agent LLM conversations."""
