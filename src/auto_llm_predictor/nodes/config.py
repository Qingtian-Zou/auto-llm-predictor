"""Node: generate_lmf_config — Create LlamaFactory YAML configs."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.messages import HumanMessage

from auto_llm_predictor.state import PipelineState
from auto_llm_predictor.utils import save_yaml

logger = logging.getLogger(__name__)

# Template → model mapping for common models
TEMPLATE_MAP = {
    "llama": "llama3",
    "qwen": "qwen",
    "gemma": "gemma",
    "mistral": "mistral",
    "phi": "phi",
    "deepseek": "deepseek",
}


def _guess_template(model_name: str) -> str:
    """Guess the chat template from the model name."""
    model_lower = model_name.lower()
    for key, template in TEMPLATE_MAP.items():
        if key in model_lower:
            return template
    return "default"


def generate_lmf_config(state: PipelineState) -> dict:
    """Generate LlamaFactory YAML configs for training and prediction.

    Reads training hyperparameters from ``state["training_config"]``
    (populated from CLI arguments).

    Creates a timestamped run directory for configs, SFT output,
    predictions, and evaluation so past runs are preserved.

    Writes: run_dir, lmf_train_yaml, lmf_predict_train_yaml,
            lmf_predict_test_yaml, adapter_path, messages
    """
    from datetime import datetime

    output_dir = Path(state["output_dir"])
    data_dir = output_dir / "data"

    # Reuse existing run directory if we are regenerating configs
    # (prevents creating orphaned run folders when user rejects a config)
    existing_run_dir = state.get("run_dir")
    if existing_run_dir:
        run_dir = Path(existing_run_dir)
        logger.info("Reusing existing run directory: %s", run_dir)
    else:
        # Create new timestamped run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_dir / f"run_{timestamp}"
        
    run_dir.mkdir(parents=True, exist_ok=True)

    config_dir = run_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    model = state["base_model"]
    template = _guess_template(model)
    sft_dir = str(run_dir / "sft")
    predict_train_dir = str(run_dir / "predict_train")
    predict_test_dir = str(run_dir / "predict_test")

    # Pull training hyperparameters from CLI (with defaults)
    tc = state.get("training_config", {})
    lora_rank = tc.get("lora_rank", 64)
    lora_alpha = tc.get("lora_alpha", 128)
    use_dora = str(tc.get("use_dora", False)).lower()
    cutoff_len = tc.get("cutoff_len", 4096)
    num_train_epochs = tc.get("num_train_epochs", 3.0)
    learning_rate = tc.get("learning_rate", "2.0e-5")
    batch_size = tc.get("per_device_train_batch_size", 2)
    grad_accum = tc.get("gradient_accumulation_steps", 8)
    logging_steps = tc.get("logging_steps", 10)
    save_steps = tc.get("save_steps", 500)
    flash_attn = tc.get("flash_attn", "auto")
    precision = tc.get("precision", "bf16")
    quant_bit = tc.get("quantization_bit", None)

    # Build quantization line (only if set)
    quant_line = f"quantization_bit: {quant_bit}\n" if quant_bit else ""

    # ── Train YAML ──────────────────────────────────────────────
    train_yaml = f"""\
### model
model_name_or_path: {model}
trust_remote_code: true
{quant_line}
### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: {lora_rank}
lora_alpha: {lora_alpha}
lora_dropout: 0.05
use_dora: {use_dora}

### dataset
dataset_dir: {data_dir}
dataset: train
template: {template}
cutoff_len: {cutoff_len}
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: {sft_dir}
logging_dir: {sft_dir}/logs
logging_steps: {logging_steps}
save_steps: {save_steps}
save_strategy: steps
save_total_limit: 3
plot_loss: true
overwrite_output_dir: true
report_to: tensorboard

### train
per_device_train_batch_size: {batch_size}
gradient_accumulation_steps: {grad_accum}
learning_rate: {learning_rate}
num_train_epochs: {num_train_epochs}
lr_scheduler_type: cosine
warmup_ratio: 0.1
{precision}: true
flash_attn: {flash_attn}
ddp_timeout: 180000000

### eval
val_size: 0.05
per_device_eval_batch_size: 1
eval_steps: 200
eval_strategy: steps
"""

    # ── Predict-on-train YAML ───────────────────────────────────
    predict_train_yaml = f"""\
### model
model_name_or_path: {model}
trust_remote_code: true
{quant_line}adapter_name_or_path: {sft_dir}

### method
stage: sft
do_predict: true
finetuning_type: lora

### dataset
dataset_dir: {data_dir}
dataset: train
eval_dataset: train
template: {template}
cutoff_len: {cutoff_len}
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: {predict_train_dir}
overwrite_output_dir: true

### eval
per_device_eval_batch_size: 1
predict_with_generate: true
{precision}: true
flash_attn: {flash_attn}
"""

    # ── Predict-on-test YAML ────────────────────────────────────
    predict_test_yaml = f"""\
### model
model_name_or_path: {model}
trust_remote_code: true
{quant_line}adapter_name_or_path: {sft_dir}

### method
stage: sft
do_predict: true
finetuning_type: lora

### dataset
dataset_dir: {data_dir}
dataset: test
eval_dataset: test
template: {template}
cutoff_len: {cutoff_len}
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: {predict_test_dir}
overwrite_output_dir: true

### eval
per_device_eval_batch_size: 1
predict_with_generate: true
{precision}: true
flash_attn: {flash_attn}
"""

    # Save configs
    train_path = str(config_dir / "train.yaml")
    predict_train_path = str(config_dir / "predict_train.yaml")
    predict_test_path = str(config_dir / "predict_test.yaml")

    save_yaml(train_yaml, train_path)
    save_yaml(predict_train_yaml, predict_train_path)
    save_yaml(predict_test_yaml, predict_test_path)

    logger.info("Generated LlamaFactory configs in %s", config_dir)

    return {
        "run_dir": str(run_dir),
        "lmf_train_yaml": train_path,
        "lmf_predict_train_yaml": predict_train_path,
        "lmf_predict_test_yaml": predict_test_path,
        "adapter_path": sft_dir,
        "messages": [
            HumanMessage(
                content=f"[generate_lmf_config] Created 3 YAML configs in {config_dir}. "
                f"Run dir: {run_dir}. Model: {model}, Template: {template}"
            ),
        ],
    }
