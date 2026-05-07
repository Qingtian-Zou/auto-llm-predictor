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

"""Standalone inference module for the auto-LLM-predictor.

Provides two modes independent of the training pipeline:

1. **Batch inference** — process a new CSV through the same data
   preparation script used during training, then run LlamaFactory
   prediction with the fine-tuned adapter.
2. **Single inference** — accept one sample's feature values, format
   as an Alpaca prompt, run the merged model directly, and optionally
   produce XAI explanations.

Both modes require a completed training output directory (containing
``scripts/prepare_data.py`` and ``.pipeline_state.json``) and a run
directory (containing the LoRA adapter).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from auto_llm_predictor.utils import normalize_path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_qbit(value: str) -> int | None:
    """argparse type for --infer-quantization-bit accepting 4, 8, or 'none'."""
    if value is None:
        return None
    if value.lower() in ("none", "0", "off"):
        return None
    if value in ("4", "8"):
        return int(value)
    raise argparse.ArgumentTypeError(
        f"quantization-bit must be 4, 8, or 'none' — got {value!r}"
    )


def _load_pipeline_state(output_dir: str) -> dict:
    """Load the saved pipeline state from a training output directory."""
    from auto_llm_predictor.checkpoint import load_state

    return load_state(output_dir)


def _guess_template(model_name: str) -> str:
    """Guess the chat template from the model name."""
    from auto_llm_predictor.nodes.config import _guess_template

    return _guess_template(model_name)


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------

def generate_inference_yaml(
    *,
    base_model: str,
    adapter_path: str,
    data_dir: str,
    dataset_name: str,
    template: str,
    cutoff_len: int,
    output_dir: str,
    precision: str = "bf16",
    flash_attn: str = "auto",
    quantization_bit: int | None = None,
) -> str:
    """Generate a LlamaFactory predict YAML for inference.

    Returns the path to the saved YAML file.
    """
    from auto_llm_predictor.utils import save_yaml

    quant_line = f"quantization_bit: {quantization_bit}\n" if quantization_bit else ""

    yaml_content = f"""\
### model
model_name_or_path: {base_model}
trust_remote_code: true
{quant_line}adapter_name_or_path: {adapter_path}

### method
stage: sft
do_predict: true
finetuning_type: lora

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
    yaml_path = str(config_dir / "infer_predict.yaml")
    save_yaml(yaml_content, yaml_path)
    logger.info("Generated inference YAML at %s", yaml_path)
    return yaml_path


def run_batch_inference(
    *,
    output_dir: str,
    run_dir: str,
    csv_path: str,
    infer_output: str = "",
    precision: str = "bf16",
    flash_attn: str = "auto",
    quantization_bit: int | None = None,
    xai: bool = False,
    log_callback: callable | None = None,
) -> dict:
    """Run batch inference on a new CSV using the trained adapter.

    Steps:
      1. Load pipeline state from ``output_dir``
      2. Copy new CSV → temp location, run existing ``prepare_data.py``
      3. Generate LlamaFactory predict YAML
      4. Run ``llamafactory-cli train`` (predict mode)
      5. Return results dict

    Parameters
    ----------
    output_dir : str
        Training output directory (contains ``scripts/``, ``.pipeline_state.json``).
    run_dir : str
        Training run directory (contains the LoRA adapter under ``sft/``).
    csv_path : str
        Path to the new CSV file for inference.
    infer_output : str
        Output directory for predictions.  Defaults to
        ``<run_dir>/inference_<timestamp>``.
    """
    from auto_llm_predictor.utils import run_llamafactory, run_script

    # ── Resolve paths ──────────────────────────────────────────
    output_dir = normalize_path(str(Path(output_dir).resolve()))
    run_dir = normalize_path(str(Path(run_dir).resolve()))
    csv_path = normalize_path(str(Path(csv_path).resolve()))

    if not infer_output:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        infer_output = str(Path(run_dir) / f"inference_{ts}")
    infer_output = normalize_path(str(Path(infer_output).resolve()))
    Path(infer_output).mkdir(parents=True, exist_ok=True)

    # ── Load pipeline state ────────────────────────────────────
    state = _load_pipeline_state(output_dir)
    base_model = state.get("base_model", "")
    adapter_path = normalize_path(state.get("adapter_path", str(Path(run_dir) / "sft")))
    template = state.get("training_config", {}).get("template") or _guess_template(base_model)
    cutoff_len = state.get("cutoff_len") or state.get("training_config", {}).get("cutoff_len", 4096)
    training_config = state.get("training_config", {})

    if not base_model:
        raise ValueError("Pipeline state is missing 'base_model'.")

    # ── Validate adapter (fallback to user-provided run_dir) ───
    if not Path(adapter_path).exists():
        adapter_path = normalize_path(str(Path(run_dir) / "sft"))
    if not Path(adapter_path).exists():
        raise FileNotFoundError(f"Adapter not found at {adapter_path}")

    # ── Prepare inference data using existing script ───────────
    prep_script = Path(output_dir) / "scripts" / "prepare_data.py"
    if not prep_script.exists():
        raise FileNotFoundError(
            f"Data preparation script not found at {prep_script}. "
            f"Ensure the training pipeline completed successfully."
        )

    infer_data_dir = Path(infer_output) / "data"
    infer_data_dir.mkdir(parents=True, exist_ok=True)

    # Copy fitted transformers from the training run so the prep script can
    # reload them in --predict-only mode instead of refitting.
    train_transformers = Path(output_dir) / "data" / "transformers.pkl"
    if not train_transformers.exists():
        raise FileNotFoundError(
            f"Training transformers not found at {train_transformers}. "
            f"This run was likely produced by an older pipeline that did not "
            f"persist fitted transformers. Re-train with the current pipeline."
        )
    shutil.copy2(train_transformers, infer_data_dir / "transformers.pkl")

    print("=" * 60)
    print("BATCH INFERENCE — Data Preparation")
    print(f"CSV:        {csv_path}")
    print(f"Script:     {prep_script}")
    print(f"Output:     {infer_data_dir}")
    print("=" * 60 + "\n", flush=True)

    success, output = run_script(
        str(prep_script),
        timeout=600,
        args=[
            "--input-csv", csv_path,
            "--output-dir", str(infer_data_dir),
            "--predict-only",
        ],
    )
    if not success:
        raise RuntimeError(f"Data preparation failed:\n{output[-2000:]}")

    # Check that the inference data file was created
    infer_json = infer_data_dir / "all_data.json"
    if not infer_json.exists():
        raise FileNotFoundError(
            f"Data preparation script did not produce {infer_json}. "
            f"Script output:\n{output[-2000:]}"
        )

    print(f"✓ Inference data prepared: {infer_json}\n", flush=True)

    # Write dataset_info.json for LlamaFactory
    dataset_info = {"infer": {"file_name": "all_data.json"}}
    info_path = infer_data_dir / "dataset_info.json"
    info_path.write_text(json.dumps(dataset_info, indent=2))

    # ── Generate predict YAML ──────────────────────────────────
    yaml_path = generate_inference_yaml(
        base_model=base_model,
        adapter_path=adapter_path,
        data_dir=str(infer_data_dir),
        dataset_name="infer",
        template=template,
        cutoff_len=cutoff_len,
        output_dir=infer_output,
        precision=precision or training_config.get("precision", "bf16"),
        flash_attn=flash_attn or training_config.get("flash_attn", "auto"),
        quantization_bit=quantization_bit,
    )

    # ── Run LlamaFactory prediction ────────────────────────────
    print("=" * 60)
    print("BATCH INFERENCE — LlamaFactory Prediction")
    print(f"Config: {yaml_path}")
    print("=" * 60 + "\n", flush=True)

    success, _, lmf_output = run_llamafactory(
        yaml_path, timeout=86400, stream=True,
        log_callback=log_callback, idle_timeout=300,
    )

    predictions_path = Path(infer_output) / "generated_predictions.jsonl"
    if success and predictions_path.exists():
        print(f"\n✓ Predictions saved to {predictions_path}\n", flush=True)
    else:
        print(f"\n✗ Prediction failed\n", flush=True)
        raise RuntimeError(
            f"LlamaFactory prediction failed. Output:\n{lmf_output[-2000:]}"
        )

    num_samples = sum(1 for _ in open(predictions_path))

    # ── Optional XAI analysis ─────────────────────────────────
    xai_report_path = ""
    xai_results = None
    if xai:
        try:
            xai_result = run_batch_xai(
                output_dir=output_dir,
                run_dir=run_dir,
                infer_output=infer_output,
                predictions_path=str(predictions_path),
                log_callback=log_callback,
            )
            xai_report_path = xai_result.get("xai_report_path", "")
            xai_results = xai_result.get("xai_results")
        except Exception as exc:
            logger.warning("Batch XAI failed: %s", exc, exc_info=True)
            msg = f"XAI analysis failed (predictions are still available): {exc}"
            print(f"\n⚠ {msg}\n", flush=True)
            if log_callback:
                log_callback(msg)

    return {
        "predictions_path": str(predictions_path),
        "infer_output": infer_output,
        "num_samples": num_samples,
        "xai_report_path": xai_report_path,
        "xai_results": xai_results,
    }


# ---------------------------------------------------------------------------
# Batch XAI
# ---------------------------------------------------------------------------

def _build_xai_samples(
    infer_data_path: str,
    predictions_path: str,
    max_samples: int = 50,
) -> list[dict]:
    """Pair inference data with predictions to build XAI-compatible samples.

    Each returned sample has ``instruction``, ``input``, and ``output`` fields.
    The ``output`` is populated from the corresponding prediction so that
    SHAP's TeacherForcing has a target to work with.
    """
    with open(infer_data_path) as f:
        infer_data = json.load(f)

    predictions: list[dict] = []
    with open(predictions_path) as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))

    pool = min(len(infer_data), len(predictions))
    n = min(pool, max_samples)
    indices = random.sample(range(pool), n) if n < pool else list(range(pool))
    samples: list[dict] = []
    for i in indices:
        entry = infer_data[i]
        pred = predictions[i].get("predict", "")
        samples.append({
            "instruction": entry.get("instruction", ""),
            "input": entry.get("input", ""),
            "output": pred,
        })
    return samples


def run_batch_xai(
    *,
    output_dir: str,
    run_dir: str,
    infer_output: str,
    predictions_path: str,
    log_callback: Callable[[str], None] | None = None,
) -> dict:
    """Run XAI analysis on batch inference results.

    Loads the model in-process, pairs inference data with predictions,
    and runs SHAP -> TransformerLens -> Attention (fallback).

    Returns
    -------
    dict
        ``{"xai_report_path": str, "xai_results": list, "methods_succeeded": list}``
    """
    from auto_llm_predictor.nodes.explain import (
        _merge_and_load,
        _run_shap,
        _run_transformer_lens,
        _run_attention,
        _release_model,
        _cleanup_gpu,
        _save_heatmap,
        _TOP_K_TOKENS,
    )

    def _log(msg: str) -> None:
        print(msg, flush=True)
        if log_callback:
            log_callback(msg)

    # ── Load pipeline state ────────────────────────────────────
    state = _load_pipeline_state(output_dir)
    base_model = state.get("base_model", "")
    adapter_path = normalize_path(
        state.get("adapter_path", str(Path(run_dir) / "sft")),
    )
    training_config = dict(state.get("training_config", {}))

    if not base_model:
        raise ValueError("Pipeline state is missing 'base_model'.")
    # Fallback: prefer user-provided run_dir when state path is stale
    if not Path(adapter_path).exists():
        adapter_path = normalize_path(str(Path(run_dir) / "sft"))
    if not Path(adapter_path).exists():
        raise FileNotFoundError(f"Adapter not found at {adapter_path}")

    # ── Build XAI samples ──────────────────────────────────────
    infer_data_path = str(Path(infer_output) / "data" / "all_data.json")
    if not Path(infer_data_path).exists():
        raise FileNotFoundError(
            f"Inference data not found at {infer_data_path}"
        )

    samples = _build_xai_samples(infer_data_path, predictions_path)
    if not samples:
        raise ValueError("No samples available for XAI analysis.")

    # ── XAI hardware defaults: fp16 + 8-bit quantization ──────
    if training_config.get("precision") in (None, "bf16"):
        training_config["precision"] = "fp16"
    if training_config.get("quantization_bit") is None:
        training_config["quantization_bit"] = 8

    # ── Load and merge model ───────────────────────────────────
    header = (
        "\n" + "=" * 60 + "\n"
        "BATCH XAI — Explainability Analysis\n"
        f"Model: {base_model}\n"
        f"Adapter: {adapter_path}\n"
        f"Samples: {len(samples)}\n"
        "Methods: SHAP -> TransformerLens -> Attention (fallback)\n"
        + "=" * 60 + "\n"
    )
    _log(header)

    model, tokenizer = _merge_and_load(base_model, adapter_path, training_config)

    # ── Run XAI methods ────────────────────────────────────────
    xai_dir = Path(infer_output) / "xai"
    xai_dir.mkdir(parents=True, exist_ok=True)

    method_results: list[dict] = []

    # 1. SHAP
    _log("Starting SHAP explanation...")
    shap_result = _run_shap(model, tokenizer, samples, xai_dir, log_callback)
    if shap_result:
        method_results.append(shap_result)

    # 2. TransformerLens
    _log("Starting TransformerLens explanation...")
    tl_result = _run_transformer_lens(
        model, tokenizer, base_model, samples, xai_dir, log_callback,
    )
    if tl_result:
        method_results.append(tl_result)

    # 3. Attention fallback — only if both SHAP and TransformerLens failed
    if not method_results:
        _log("Both SHAP and TransformerLens unavailable — trying attention fallback...")
        attn_result = _run_attention(model, tokenizer, samples, log_callback)
        if attn_result:
            method_results.append(attn_result)

    # ── Unload model ───────────────────────────────────────────
    _release_model(model)
    del model, tokenizer
    _cleanup_gpu()
    _log("Model unloaded, GPU memory freed\n")

    if not method_results:
        _log("All XAI methods failed.")
        return {"xai_report_path": "", "xai_results": [], "methods_succeeded": []}

    # ── Build unified report ───────────────────────────────────
    report = {
        "model": base_model,
        "adapter_path": adapter_path,
        "num_samples": len(samples),
        "top_k_tokens": _TOP_K_TOKENS,
        "methods_succeeded": [r["method"] for r in method_results],
        "results": method_results,
    }

    report_path = xai_dir / "xai_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Saved batch XAI report to %s", report_path)

    # ── Heatmap visualisation ──────────────────────────────────
    heatmap_path = xai_dir / "xai_heatmap.png"
    _save_heatmap(method_results, str(heatmap_path))

    methods = ", ".join(r["method"] for r in method_results)
    _log(f"\nBatch XAI complete ({methods}). Report at {report_path}\n")

    return {
        "xai_report_path": str(report_path),
        "xai_results": method_results,
        "methods_succeeded": [r["method"] for r in method_results],
    }


# ---------------------------------------------------------------------------
# Single inference
# ---------------------------------------------------------------------------

def run_single_inference(
    *,
    output_dir: str,
    run_dir: str,
    sample_features: dict[str, str],
    xai: bool = False,
    precision: str = "bf16",
    quantization_bit: int | None = None,
    log_callback: Callable[[str], None] | None = None,
    prediction_callback: Callable[[dict], None] | None = None,
) -> dict:
    """Run inference on a single sample.

    Parameters
    ----------
    output_dir : str
        Training output directory (contains ``.pipeline_state.json``).
    run_dir : str
        Training run directory (contains the LoRA adapter).
    sample_features : dict
        Feature name → value mapping for the single sample.
    xai : bool
        If True, run XAI explanations on the prediction.
    prediction_callback : callable, optional
        Called with ``{"prediction": str, "target_mapping": dict}`` as soon
        as the prediction is ready (before XAI starts).  Useful for
        streaming the prediction to the UI without waiting for XAI.

    Returns
    -------
    dict
        ``{"prediction": str, "xai_results": list[dict] | None}``
    """
    import torch

    # ── Load pipeline state ────────────────────────────────────
    state = _load_pipeline_state(output_dir)
    base_model = state.get("base_model", "")
    adapter_path = normalize_path(state.get("adapter_path", str(Path(run_dir) / "sft")))
    training_config = state.get("training_config", {})
    target_mapping = state.get("target_mapping", {})

    if not base_model:
        raise ValueError("Pipeline state is missing 'base_model'.")
    # Fallback: prefer user-provided run_dir when state path is stale
    if not Path(adapter_path).exists():
        adapter_path = normalize_path(str(Path(run_dir) / "sft"))
    if not Path(adapter_path).exists():
        raise FileNotFoundError(f"Adapter not found at {adapter_path}")

    # ── Build prompt from sample features ──────────────────────
    prompt = _build_single_prompt(state, sample_features)

    def _log(msg: str) -> None:
        print(msg, flush=True)
        if log_callback:
            log_callback(msg)

    _log("=" * 60)
    _log("SINGLE INFERENCE")
    _log(f"Model:    {base_model}")
    _log(f"Adapter:  {adapter_path}")
    _log(f"Prompt:   {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
    _log("=" * 60 + "\n")

    # ── Load and merge model ───────────────────────────────────
    _log("Loading model and adapter...")
    from auto_llm_predictor.nodes.explain import _merge_and_load, _release_model, _cleanup_gpu

    prec = precision or training_config.get("precision", "bf16")
    qbit = quantization_bit or training_config.get("quantization_bit")
    tc = {**training_config, "precision": prec, "quantization_bit": qbit}
    model, tokenizer = _merge_and_load(base_model, adapter_path, tc)

    # ── Apply chat template ───────────────────────────────────
    # The model was fine-tuned with LlamaFactory's template-wrapped format,
    # so we must wrap the prompt the same way for single inference.
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        templated_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    else:
        templated_prompt = prompt

    cutoff_len = state.get("cutoff_len") or training_config.get("cutoff_len", 4096)
    inputs = tokenizer(templated_prompt, return_tensors="pt", truncation=True, max_length=cutoff_len)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Free intermediate GPU tensors now that we have the string
    del outputs, generated_ids, inputs

    _log(f"\n✓ Prediction: {prediction}\n")

    # ── Notify caller that prediction is ready ─────────────────
    if prediction_callback:
        prediction_callback({
            "prediction": prediction,
            "target_mapping": target_mapping,
        })

    # ── Optional XAI ───────────────────────────────────────────
    xai_results = None
    if xai:
        xai_results = _run_single_xai(
            model, tokenizer, base_model, prompt, state, run_dir,
            prediction=prediction,
            log_callback=log_callback,
        )

    # ── Cleanup ────────────────────────────────────────────────
    _release_model(model)
    del model, tokenizer
    _cleanup_gpu()

    return {
        "prediction": prediction,
        "target_mapping": target_mapping,
        "xai_results": xai_results,
    }


def _build_single_prompt(state: dict, sample_features: dict[str, str]) -> str:
    """Build an Alpaca-style prompt for a single sample.

    Reads the instruction template and input format from an existing
    data entry in the training output, then fills in the sample features.
    """
    output_dir = state.get("output_dir", "")
    data_dir = Path(output_dir) / "data"

    # Try to load a sample entry from all_data.json to get the format
    all_data_path = data_dir / "all_data.json"
    if all_data_path.exists():
        with open(all_data_path) as f:
            entries = json.load(f)
        if entries:
            sample_entry = entries[0]
            instruction = sample_entry.get("instruction", "")

            # Build input string from features using the same format
            # Parse the existing input format to understand the pattern
            existing_input = sample_entry.get("input", "")
            if existing_input:
                # Try to replicate the input format pattern
                input_text = _format_features_like_example(
                    existing_input, sample_features,
                )
                return f"{instruction}\n\n{input_text}" if instruction else input_text
            else:
                # Features are embedded in the instruction itself
                feature_lines = [f"{k}: {v}" for k, v in sample_features.items()]
                return instruction + "\n\n" + "\n".join(feature_lines)

    # Fallback: simple feature listing
    feature_lines = [f"{k}: {v}" for k, v in sample_features.items()]
    return "Based on the following information, make a prediction:\n\n" + "\n".join(feature_lines)


def _format_features_like_example(example_input: str, features: dict[str, str]) -> str:
    """Reproduce the input format from the example, substituting new values."""
    # Common patterns: "key: value\nkey: value" or "key=value, ..."
    import re

    # Detect separator pattern
    if "\n" in example_input and ": " in example_input:
        # Line-separated "key: value" format
        return "\n".join(f"{k}: {v}" for k, v in features.items())
    elif ", " in example_input and ": " in example_input:
        # Comma-separated "key: value" format
        return ", ".join(f"{k}: {v}" for k, v in features.items())
    elif "=" in example_input:
        # Key=value format
        return ", ".join(f"{k}={v}" for k, v in features.items())
    else:
        # Default: newline-separated
        return "\n".join(f"{k}: {v}" for k, v in features.items())


def _run_single_xai(
    model, tokenizer, base_model, prompt, state, run_dir,
    prediction: str = "",
    log_callback: Callable[[str], None] | None = None,
) -> list[dict]:
    """Run XAI methods on a single sample prediction."""
    from auto_llm_predictor.nodes.explain import (
        _run_shap,
        _run_transformer_lens,
        _run_attention,
    )

    def _log(msg: str) -> None:
        print(msg, flush=True)
        if log_callback:
            log_callback(msg)

    xai_dir = Path(run_dir) / "single_xai"
    xai_dir.mkdir(parents=True, exist_ok=True)

    sample = {"instruction": prompt, "input": "", "output": prediction}

    _log("=" * 60)
    _log("XAI — Explaining single prediction")
    _log("=" * 60 + "\n")

    results = []

    # 1. SHAP
    _log("Starting SHAP explanation...")
    shap_result = _run_shap(model, tokenizer, [sample], xai_dir, log_callback=log_callback)
    if shap_result:
        results.append(shap_result)

    # 2. TransformerLens
    _log("Starting TransformerLens explanation...")
    tl_result = _run_transformer_lens(
        model, tokenizer, base_model, [sample], xai_dir, log_callback=log_callback,
    )
    if tl_result:
        results.append(tl_result)

    # 3. Attention fallback
    if not results:
        _log("Starting Attention fallback explanation...")
        attn_result = _run_attention(model, tokenizer, [sample], log_callback=log_callback)
        if attn_result:
            results.append(attn_result)

    _log("XAI analysis complete.")

    # Final GPU cleanup — XAI methods may leave residual tensors or caches
    # that keep VRAM occupied even after the model is deleted by the caller.
    from auto_llm_predictor.nodes.explain import _cleanup_gpu
    _cleanup_gpu()

    return results


def _parse_input_to_features(input_text: str) -> dict[str, str]:
    """Reverse of _format_features_like_example: parse formatted input back to a dict.

    Supports newline-separated ``key: value``, comma-separated ``key: value``,
    and ``key=value`` formats.
    """
    features: dict[str, str] = {}
    if not input_text:
        return features

    if "\n" in input_text and ": " in input_text:
        for line in input_text.split("\n"):
            if ": " in line:
                k, v = line.split(": ", 1)
                features[k.strip()] = v.strip()
    elif ", " in input_text and ": " in input_text:
        for pair in input_text.split(", "):
            if ": " in pair:
                k, v = pair.split(": ", 1)
                features[k.strip()] = v.strip()
    elif "=" in input_text:
        for pair in input_text.split(", "):
            if "=" in pair:
                k, v = pair.split("=", 1)
                features[k.strip()] = v.strip()
    return features


def get_feature_names(output_dir: str) -> list[str]:
    """Get the list of feature names from a training output directory.

    Used by the web UI to populate input fields for single inference.
    """
    state = _load_pipeline_state(output_dir)
    return state.get("selected_features", [])


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """CLI entry point for inference mode."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Auto LLM Predictor — Inference Mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Batch inference on a new CSV
  auto-llm-predictor-infer --infer-output-dir output/my_dataset \\
    --infer-run-dir output/my_dataset/run_20260307_120000 \\
    --infer-csv data/new_data.csv

  # Single interactive inference
  auto-llm-predictor-infer --infer-output-dir output/my_dataset \\
    --infer-run-dir output/my_dataset/run_20260307_120000 \\
    --infer-single

  # Single inference with XAI
  auto-llm-predictor-infer --infer-output-dir output/my_dataset \\
    --infer-run-dir output/my_dataset/run_20260307_120000 \\
    --infer-single --infer-xai

  # Batch inference with XAI
  auto-llm-predictor-infer --infer-output-dir output/my_dataset \\
    --infer-run-dir output/my_dataset/run_20260307_120000 \\
    --infer-csv data/new_data.csv --infer-xai
""",
    )

    # Required
    parser.add_argument(
        "--infer-output-dir", required=True,
        help="Training output directory (contains scripts/ and .pipeline_state.json)",
    )
    parser.add_argument(
        "--infer-run-dir", required=True,
        help="Training run directory (contains the LoRA adapter under sft/)",
    )

    # Mode selection
    parser.add_argument(
        "--infer-csv", default="",
        help="Path to new CSV file for batch inference",
    )
    parser.add_argument(
        "--infer-single", action="store_true",
        help="Enter single-inference mode (interactive prompt for feature values)",
    )

    # Options
    parser.add_argument(
        "--infer-output", default="",
        help="Output directory for predictions (default: <run_dir>/inference_<timestamp>)",
    )
    parser.add_argument(
        "--infer-xai", action="store_true",
        help="Run XAI explanations on predictions (batch or single mode)",
    )
    parser.add_argument(
        "--infer-quantization-bit", type=_parse_qbit, default=None,
        help="Quantization bits: 4, 8, or 'none' to disable (default: 8 when --infer-xai is set, else off)",
    )
    parser.add_argument(
        "--infer-flash-attn", default="auto", choices=["auto", "fa2", "disabled"],
        help="Flash attention mode (default: auto)",
    )
    parser.add_argument(
        "--infer-precision", default="bf16", choices=["bf16", "fp16"],
        help="Precision (default: bf16)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # ── XAI hardware defaults: fp16 + 8-bit quantization ──────
    if args.infer_xai:
        if args.infer_precision == "bf16":
            args.infer_precision = "fp16"
        if args.infer_quantization_bit is None:
            args.infer_quantization_bit = 8

    # ── Setup logging ──────────────────────────────────────────
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Validate ───────────────────────────────────────────────
    output_dir = normalize_path(str(Path(args.infer_output_dir).resolve()))
    run_dir = normalize_path(str(Path(args.infer_run_dir).resolve()))

    if not Path(output_dir).exists():
        print(f"Error: Output directory not found: {output_dir}", file=sys.stderr)
        sys.exit(1)
    if not Path(run_dir).exists():
        print(f"Error: Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    if not args.infer_csv and not args.infer_single:
        print("Error: Specify either --infer-csv for batch mode or --infer-single for single mode.",
              file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("Auto LLM Predictor — Inference Mode")
    print("=" * 60)
    print(f"Output dir:  {output_dir}")
    print(f"Run dir:     {run_dir}")
    print(f"Mode:        {'single' if args.infer_single else 'batch'}")
    print("=" * 60 + "\n")

    try:
        if args.infer_single:
            _run_single_interactive(
                output_dir=output_dir,
                run_dir=run_dir,
                xai=args.infer_xai,
                precision=args.infer_precision,
                quantization_bit=args.infer_quantization_bit,
            )
        else:
            csv_path = normalize_path(str(Path(args.infer_csv).resolve()))
            if not Path(csv_path).exists():
                print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
                sys.exit(1)

            result = run_batch_inference(
                output_dir=output_dir,
                run_dir=run_dir,
                csv_path=csv_path,
                infer_output=args.infer_output,
                precision=args.infer_precision,
                flash_attn=args.infer_flash_attn,
                quantization_bit=args.infer_quantization_bit,
                xai=args.infer_xai,
            )

            print("\n" + "=" * 60)
            print("Batch Inference Complete!")
            print("=" * 60)
            print(f"Predictions:  {result['predictions_path']}")
            print(f"Samples:      {result['num_samples']}")
            print(f"Output:       {result['infer_output']}")
            if result.get("xai_report_path"):
                print(f"XAI Report:   {result['xai_report_path']}")
            print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nInference interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logging.exception("Inference failed")
        print(f"\nInference failed: {e}", file=sys.stderr)
        sys.exit(1)


def _run_single_interactive(
    *,
    output_dir: str,
    run_dir: str,
    xai: bool,
    precision: str,
    quantization_bit: int | None,
):
    """Interactive single-inference mode: prompt user for feature values."""
    features = get_feature_names(output_dir)
    if not features:
        print("Error: No features found in pipeline state.", file=sys.stderr)
        sys.exit(1)

    print(f"Features ({len(features)}): {', '.join(features)}\n")
    print("Enter values for each feature (or 'quit' to exit):\n")

    while True:
        sample = {}
        for feat in features:
            val = input(f"  {feat}: ").strip()
            if val.lower() == "quit":
                print("\nExiting.")
                return
            sample[feat] = val

        print()
        result = run_single_inference(
            output_dir=output_dir,
            run_dir=run_dir,
            sample_features=sample,
            xai=xai,
            precision=precision,
            quantization_bit=quantization_bit,
        )

        print(f"\n{'=' * 40}")
        print(f"  Prediction: {result['prediction']}")

        if result.get("target_mapping"):
            print(f"  Target mapping: {result['target_mapping']}")

        if result.get("xai_results"):
            for xai_result in result["xai_results"]:
                method = xai_result.get("method", "unknown")
                explanations = xai_result.get("sample_explanations", [])
                if explanations:
                    top = explanations[0].get("token_scores", [])[:10]
                    print(f"\n  XAI ({method}) — Top tokens:")
                    for ts in top:
                        print(f"    {ts['token']:20s}  {ts['score']:.6f}")

        print(f"{'=' * 40}\n")

        again = input("Run another prediction? (y/n): ").strip().lower()
        if again != "y":
            break

    print("\nDone.")


if __name__ == "__main__":
    main()
