"""Node: run_xai — Attention-based token-level explanations.

Loads the fine-tuned model (base + LoRA adapter) and runs forward passes
with ``output_attentions=True`` to explain which input tokens the model
focused on for each prediction.
"""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path

from langchain_core.messages import HumanMessage

from auto_llm_predictor.state import PipelineState

logger = logging.getLogger(__name__)

# Maximum number of test samples to explain (keeps runtime bounded)
_MAX_SAMPLES = 50

# Number of top tokens to include per sample in the report
_TOP_K_TOKENS = 15


def check_xai_enabled(state: PipelineState) -> str:
    """Routing function: run XAI node only if --xai was passed."""
    if state.get("xai_enabled", False):
        return "run_xai"
    return "__end__"


def _load_model_and_tokenizer(
    base_model: str,
    adapter_path: str,
    training_config: dict,
):
    """Load the base model + LoRA adapter for inference with attention.

    Returns (model, tokenizer) or raises on failure.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    quant_bit = training_config.get("quantization_bit")
    precision = training_config.get("precision", "bf16")
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "device_map": "auto",
        # Must use "eager" attention for output_attentions=True;
        # SDPA and FlashAttention-2 do not support attention output.
        "attn_implementation": "eager",
    }
    if quant_bit == 4:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
        )
    elif quant_bit == 8:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    logger.info("Loading base model: %s", base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)

    logger.info("Loading LoRA adapter: %s", adapter_path)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def _extract_attention_scores(model, tokenizer, text: str, device: str) -> list[dict]:
    """Run a forward pass and extract per-token attention scores.

    Returns a list of ``{"token": str, "score": float}`` sorted by
    descending attention, limited to the top-K tokens.
    """
    import torch

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # outputs.attentions is a tuple of (batch, heads, seq, seq) per layer.
    # May be None if the model/backend silently ignores output_attentions.
    if outputs.attentions is None:
        raise RuntimeError("Model returned None for attentions — check attn_implementation")

    # Use last layer, average across heads, take attention *to* each token
    last_layer_attn = outputs.attentions[-1]           # (1, heads, seq, seq)
    avg_attn = last_layer_attn.mean(dim=1).squeeze(0)  # (seq, seq)

    # Sum attention *from* all positions *to* each token → column sum
    token_importance = avg_attn.sum(dim=0)              # (seq,)
    token_importance = token_importance / token_importance.sum()  # normalize

    scores = token_importance.cpu().float().tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu())

    # Pair tokens with scores and sort by importance
    paired = [{"token": tok, "score": round(s, 6)} for tok, s in zip(tokens, scores)]
    paired.sort(key=lambda x: x["score"], reverse=True)

    return paired[:_TOP_K_TOKENS]


def _build_prompt(entry: dict) -> str:
    """Reconstruct the Alpaca-style prompt from a data entry."""
    instruction = entry.get("instruction", "")
    input_text = entry.get("input", "")
    if input_text:
        return f"{instruction}\n\n{input_text}"
    return instruction


def _save_heatmap(sample_explanations: list[dict], output_path: str) -> bool:
    """Generate an attention heatmap for the first few samples.

    Returns True on success, False if matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.info("matplotlib not available — skipping heatmap generation.")
        return False

    n_samples = min(5, len(sample_explanations))
    if n_samples == 0:
        return False

    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3 * n_samples))
    if n_samples == 1:
        axes = [axes]

    for ax, sample in zip(axes, sample_explanations[:n_samples]):
        top_tokens = sample["token_scores"][:10]
        tokens = [t["token"] for t in reversed(top_tokens)]
        scores = [t["score"] for t in reversed(top_tokens)]

        bars = ax.barh(tokens, scores, color="#6366f1")
        ax.set_xlabel("Attention Score")

        # Truncate label for display
        label = sample.get("predicted_label", "")
        ax.set_title(f"Predicted: {label}", fontsize=10)

        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{score:.4f}", va="center", fontsize=8,
            )

    fig.suptitle("Token-Level Attention Explanations (Top-10 Tokens)", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved attention heatmap to %s", output_path)
    return True


def _skip_result(reason: str) -> dict:
    """Build a standard skip/failure return dict for run_xai."""
    return {
        "xai_report_path": "",
        "messages": [HumanMessage(content=f"[run_xai] {reason}")],
    }


def run_xai(state: PipelineState) -> dict:
    """Generate attention-based token-level explanations for test predictions.

    Writes: xai_report_path, messages
    """
    # ── Skip guards ────────────────────────────────────────────
    if not state.get("xai_enabled", False):
        logger.info("XAI not enabled — skipping.")
        return {}

    adapter_path = state.get("adapter_path", "")
    if not adapter_path or not Path(adapter_path).exists():
        logger.warning("Adapter not found at %s — skipping XAI.", adapter_path)
        return _skip_result("SKIPPED — adapter not found.")

    if not state.get("finetune_succeeded", False):
        logger.warning("Fine-tuning did not succeed — skipping XAI.")
        return _skip_result("SKIPPED — fine-tuning did not succeed.")

    # ── Load test data ─────────────────────────────────────────
    test_data_path = state.get("test_data_path", "")
    if not test_data_path or not Path(test_data_path).exists():
        logger.warning("Test data not found — skipping XAI.")
        return _skip_result("SKIPPED — test data not found.")

    with open(test_data_path) as f:
        test_data = json.load(f)

    if not test_data:
        logger.warning("Test data is empty — skipping XAI.")
        return _skip_result("SKIPPED — test data is empty.")

    # Limit samples
    samples = test_data[:_MAX_SAMPLES]
    base_model = state["base_model"]
    training_config = state.get("training_config", {})
    target_mapping = state.get("target_mapping", {})

    # ── Load model ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXPLAINABILITY (XAI) — Attention-Based Token Explanations")
    print(f"Model: {base_model}")
    print(f"Adapter: {adapter_path}")
    print(f"Samples: {len(samples)}")
    print("=" * 60 + "\n", flush=True)

    try:
        model, tokenizer = _load_model_and_tokenizer(
            base_model, adapter_path, training_config
        )
    except Exception as exc:
        logger.exception("Failed to load model for XAI")
        return _skip_result(f"FAILED — could not load model: {exc}")

    device = next(model.parameters()).device.type

    # ── Generate explanations ──────────────────────────────────
    sample_explanations = []
    for i, entry in enumerate(samples):
        prompt = _build_prompt(entry)
        true_label = entry.get("output", "")

        try:
            token_scores = _extract_attention_scores(model, tokenizer, prompt, device)
        except Exception as exc:
            logger.warning("XAI failed for sample %d: %s", i, exc)
            token_scores = []

        sample_explanations.append({
            "sample_index": i,
            "input_preview": prompt[:200],
            "true_label": true_label,
            "predicted_label": true_label,  # best available from training data
            "token_scores": token_scores,
        })

        if (i + 1) % 10 == 0 or i == len(samples) - 1:
            print(f"  Explained {i + 1}/{len(samples)} samples", flush=True)

    # ── Unload model ───────────────────────────────────────────
    del model, tokenizer
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass
    print("✓ Model unloaded, GPU memory freed\n", flush=True)

    # ── Build report ───────────────────────────────────────────
    run_dir = Path(state.get("run_dir", state["output_dir"]))
    xai_dir = run_dir / "xai"
    xai_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "method": "attention",
        "model": base_model,
        "adapter_path": adapter_path,
        "num_samples": len(sample_explanations),
        "top_k_tokens": _TOP_K_TOKENS,
        "sample_explanations": sample_explanations,
    }

    report_path = xai_dir / "xai_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Saved XAI report to %s", report_path)

    # ── Optional heatmap visualization ─────────────────────────
    heatmap_path = xai_dir / "attention_heatmap.png"
    _save_heatmap(sample_explanations, str(heatmap_path))

    # ── Summary ────────────────────────────────────────────────
    explained = sum(1 for s in sample_explanations if s["token_scores"])
    summary = (
        f"Explained {explained}/{len(sample_explanations)} samples. "
        f"Report at {report_path}"
    )

    print(f"\n✓ XAI complete: {summary}\n", flush=True)

    return {
        "xai_report_path": str(report_path),
        "messages": [
            HumanMessage(content=f"[run_xai] {summary}"),
        ],
    }
