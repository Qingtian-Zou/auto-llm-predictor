"""Node: run_xai — Explainable AI with SHAP, TransformerLens, and attention.

Generates token-level explanations for model predictions using three
methods in priority order:

1. **SHAP** — Partition-based SHAP values via a text-generation pipeline
   wrapper (``shap.Explainer``).  Produces per-token Shapley attribution.
2. **TransformerLens** — Logit attribution via ``HookedTransformer``.
   Decomposes the residual stream to find each token's contribution to
   the final prediction logit.
3. **Attention** (fallback) — Eager-mode attention weights from the last
   transformer layer.

The node loads the fine-tuned model (base + LoRA adapter merged), runs
the available methods, and saves a unified JSON report plus optional
visualisation PNGs.
"""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from auto_llm_predictor.state import PipelineState

logger = logging.getLogger(__name__)

# Maximum number of test samples to explain (keeps runtime bounded)
_MAX_SAMPLES = 50

# Number of top tokens to include per sample in the report
_TOP_K_TOKENS = 15


# ── Routing ────────────────────────────────────────────────────────

def check_xai_enabled(state: PipelineState) -> str:
    """Routing function: run XAI node only if --xai was passed."""
    if state.get("xai_enabled", False):
        return "run_xai"
    return "__end__"


# ── Model loading ──────────────────────────────────────────────────

def _merge_and_load(base_model: str, adapter_path: str, training_config: dict):
    """Load base model + LoRA adapter, merge weights, return (model, tokenizer).

    Merging is required for both TransformerLens and SHAP (they need a
    plain ``AutoModelForCausalLM`` without PEFT wrappers).

    When ``quantization_bit`` is set in *training_config*, the model is
    loaded with 8-bit (or 4-bit) quantization via ``bitsandbytes`` to
    reduce GPU memory usage — especially useful for XAI workloads.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    precision = training_config.get("precision", "bf16")
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    quantization_bit = training_config.get("quantization_bit")

    load_kwargs: dict = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "device_map": "auto",
    }

    if quantization_bit in (4, 8):
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=(quantization_bit == 8),
            load_in_4bit=(quantization_bit == 4),
        )
        logger.info("Using %d-bit quantization with %s precision", quantization_bit, precision)

    logger.info("Loading base model: %s", base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)

    logger.info("Loading and merging LoRA adapter: %s", adapter_path)
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def _release_model(model) -> None:
    """Aggressively free a model's VRAM.

    Quantized models (``bitsandbytes`` Int8Params) and models loaded with
    ``device_map="auto"`` (``accelerate`` hooks) often survive a simple
    ``del model`` due to circular references.  This helper replaces every
    parameter and buffer tensor with a tiny CPU tensor, which immediately
    releases the underlying CUDA storage regardless of remaining refs.
    """
    try:
        import torch
        empty = torch.empty(0)
        for p in model.parameters():
            p.data = empty
        for b in model.buffers():
            b.data = empty
    except Exception:          # noqa: BLE001
        pass                   # best-effort; the del + gc below will catch the rest


def _cleanup_gpu():
    """Free GPU memory after model use."""
    # Two gc passes: the first breaks simple cycles, the second catches
    # weak-ref / weak-value-dict chains that only become collectable after
    # the first pass frees their referents.
    gc.collect()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ── Prompt helpers ─────────────────────────────────────────────────

def _build_prompt(entry: dict, tokenizer=None) -> str:
    """Reconstruct the Alpaca-style prompt from a data entry.

    When *tokenizer* is provided and has a ``chat_template``, the raw
    prompt is wrapped in the model's chat format so that XAI methods
    analyse the same input the model actually sees.
    """
    instruction = entry.get("instruction", "")
    input_text = entry.get("input", "")
    raw = f"{instruction}\n\n{input_text}" if input_text else instruction

    if tokenizer and hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": raw}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    return raw


# ── Log helper ─────────────────────────────────────────────────────

def _log(msg: str, callback=None) -> None:
    """Print to stdout and, if available, send to the web UI via callback."""
    print(msg, flush=True)
    if callback:
        callback(msg)


# ── Method 1: SHAP ─────────────────────────────────────────────────

def _run_shap(model, tokenizer, samples: list[dict], xai_dir: Path, log_callback=None) -> dict | None:
    """Run SHAP text explainer on the fine-tuned model.

    Uses ``shap.Explainer`` with a HuggingFace text-generation pipeline.
    Returns a result dict or None on failure.
    """
    try:
        import shap
        import transformers
    except ImportError:
        logger.info("shap not installed — skipping SHAP method.")
        return None

    try:
        import torch
        import numpy as np

        # Use SHAP's TeacherForcing model for text-generation.
        # TeacherForcing computes log-odds of generating the target tokens
        # given (possibly masked) input — returning a proper numeric numpy
        # array that SHAP's numba-compiled internals can handle.
        masker = shap.maskers.Text(tokenizer)

        prompts = [_build_prompt(s, tokenizer) for s in samples]
        true_labels = [s.get("output", "") for s in samples]

        # Estimate token count for progress tracking
        n_tokens = len(tokenizer.encode(prompts[0])) if prompts else 0

        shap_model = shap.models.TeacherForcing(
            model, tokenizer,
        )

        # Wrap the shap model call to report progress.
        _orig_call = shap_model.__call__

        _shap_calls = {"n": 0, "last_pct": -1}
        _est_total = 2 * n_tokens + 1

        def _tracked_call(*args, **kwargs):
            _shap_calls["n"] += 1
            pct = min(100, int(_shap_calls["n"] / _est_total * 100))
            if log_callback and pct >= _shap_calls["last_pct"] + 5:
                _shap_calls["last_pct"] = pct
                log_callback(
                    f"    SHAP evaluating masks: {_shap_calls['n']}/{_est_total} "
                    f"(~{pct}%)"
                )
            return _orig_call(*args, **kwargs)
        shap_model.__call__ = _tracked_call

        explainer = shap.Explainer(shap_model, masker)

        _log("  Running SHAP explanations...", log_callback)
        if n_tokens:
            _log(f"    Prompt has ~{n_tokens} tokens → ~{2 * n_tokens + 1} mask evaluations", log_callback)

        # TeacherForcing expects both input prompts and target outputs.
        # The explainer passes (masked_X, Y) to the model at each step.
        shap_prompts = np.array(prompts)
        shap_targets = np.array(true_labels)
        shap_result = explainer(shap_prompts, shap_targets, silent=True)

        # TeacherForcing with two args returns [input_explanation,
        # target_explanation].  We want the input side (index 0).
        if isinstance(shap_result, list):
            shap_values = shap_result[0]
        else:
            shap_values = shap_result

        # Extract per-sample token-level SHAP values
        sample_explanations = []
        for i in range(len(prompts)):
            sv = shap_values[i] if len(prompts) > 1 else shap_values
            # sv.values may be 1-D (single output) or 2-D (multi-output)
            values = sv.values
            tokens = sv.data

            if values is None or tokens is None:
                sample_explanations.append({
                    "sample_index": i,
                    "input_preview": prompts[i][:200],
                    "true_label": true_labels[i],
                    "token_scores": [],
                })
                continue

            # Collapse to 1-D: one importance score per input token.
            # TeacherForcing produces scores per (input_token, output_token),
            # so there may be multiple trailing dimensions to reduce.
            while values.ndim > 1:
                values = np.abs(values).max(axis=-1)

            # Build token→score list sorted by importance
            paired = [
                {"token": str(tok), "score": round(float(np.abs(val)), 6)}
                for tok, val in zip(tokens, values)
                if str(tok).strip()
            ]
            paired.sort(key=lambda x: x["score"], reverse=True)

            sample_explanations.append({
                "sample_index": i,
                "input_preview": prompts[i][:200],
                "true_label": true_labels[i],
                "token_scores": paired[:_TOP_K_TOKENS],
            })

            if (i + 1) % 10 == 0 or i == len(prompts) - 1:
                _log(f"    SHAP: {i + 1}/{len(prompts)} samples", log_callback)

        # Save SHAP HTML visualisation if possible
        shap_html: str | None = None
        try:
            html_content = shap.plots.text(shap_values, display=False)
            if html_content:
                shap_html = str(html_content)
                html_path = xai_dir / "shap_text.html"
                Path(html_path).write_text(shap_html)
                logger.info("Saved SHAP text plot to %s", html_path)
        except Exception as exc:
            logger.debug("Could not save SHAP HTML plot: %s", exc)

        explained = sum(1 for s in sample_explanations if s["token_scores"])
        _log(f"  ✓ SHAP complete: {explained}/{len(sample_explanations)} samples\n", log_callback)

        result: dict = {
            "method": "shap",
            "num_samples": len(sample_explanations),
            "sample_explanations": sample_explanations,
        }
        if shap_html:
            result["html"] = shap_html
        return result

    except Exception as exc:
        logger.warning("SHAP method failed: %s", exc, exc_info=True)
        _log(f"  ✗ SHAP failed: {exc}\n", log_callback)
        return None
    finally:
        # Release references that keep the model pinned on GPU.
        # These objects (pipeline, explainer, masker) hold refs to the
        # model/tokenizer and will prevent VRAM from being freed.
        try:
            del pipe, explainer, masker
        except NameError:
            pass
        try:
            del shap_values
        except NameError:
            pass
        _cleanup_gpu()


# ── Method 2: TransformerLens ──────────────────────────────────────

def _run_transformer_lens(
    model, tokenizer, base_model: str, samples: list[dict], xai_dir: Path,
    log_callback=None,
) -> dict | None:
    """Run TransformerLens logit attribution on the merged model.

    Computes per-token logit contributions via the residual stream
    decomposition.  Returns a result dict or None on failure.
    """
    try:
        from transformer_lens import HookedTransformer
    except ImportError:
        logger.info("transformer_lens not installed — skipping TransformerLens method.")
        return None

    try:
        import torch

        _log("  Loading HookedTransformer...", log_callback)
        hooked = HookedTransformer.from_pretrained(
            base_model,
            hf_model=model,
            tokenizer=tokenizer,
            device=model.device,
        )

        sample_explanations = []
        for i, entry in enumerate(samples):
            prompt = _build_prompt(entry, tokenizer)
            true_label = entry.get("output", "")

            try:
                tokens = hooked.to_tokens(prompt, prepend_bos=True)
                str_tokens = hooked.to_str_tokens(prompt, prepend_bos=True)

                # Run with cache to get residual stream per layer
                logits, cache = hooked.run_with_cache(tokens)

                # Logit attribution: project each token's residual contribution
                # through the unembedding matrix to get logit contribution
                # We use the residual stream at the final position
                final_residual = cache["resid_post", -1]  # (1, seq, d_model)

                # Get the unembedding for the predicted token (last position)
                predicted_token_id = logits[0, -1].argmax().item()
                unembed_vec = hooked.W_U[:, predicted_token_id]  # (d_model,)

                # Per-position contribution to the predicted logit
                contributions = (final_residual[0] * unembed_vec).sum(dim=-1)  # (seq,)
                contributions = contributions.float().cpu()

                # Normalise to relative importance
                abs_c = contributions.abs()
                if abs_c.sum() > 0:
                    importance = (abs_c / abs_c.sum()).tolist()
                else:
                    importance = [0.0] * len(str_tokens)

                paired = [
                    {"token": str(tok), "score": round(s, 6)}
                    for tok, s in zip(str_tokens, importance)
                    if str(tok).strip()
                ]
                paired.sort(key=lambda x: x["score"], reverse=True)

                sample_explanations.append({
                    "sample_index": i,
                    "input_preview": prompt[:200],
                    "true_label": true_label,
                    "token_scores": paired[:_TOP_K_TOKENS],
                })

            except Exception as exc:
                logger.warning("TransformerLens failed for sample %d: %s", i, exc)
                sample_explanations.append({
                    "sample_index": i,
                    "input_preview": prompt[:200],
                    "true_label": true_label,
                    "token_scores": [],
                })

            if (i + 1) % 10 == 0 or i == len(samples) - 1:
                _log(f"    TransformerLens: {i + 1}/{len(samples)} samples", log_callback)

        del hooked
        _cleanup_gpu()

        explained = sum(1 for s in sample_explanations if s["token_scores"])
        _log(f"  ✓ TransformerLens complete: {explained}/{len(sample_explanations)} samples\n", log_callback)

        return {
            "method": "transformer_lens",
            "num_samples": len(sample_explanations),
            "sample_explanations": sample_explanations,
        }

    except Exception as exc:
        logger.warning("TransformerLens method failed: %s", exc, exc_info=True)
        _log(f"  ✗ TransformerLens failed: {exc}\n", log_callback)
        return None


# ── Method 3: Attention fallback ───────────────────────────────────

def _run_attention(model, tokenizer, samples: list[dict], log_callback=None) -> dict | None:
    """Fallback: extract last-layer attention weights.

    Reloading with ``attn_implementation='eager'`` is NOT needed here
    because the model was already loaded without SDPA/FA2 by
    ``_merge_and_load`` (device_map='auto' picks the best backend, but
    we override to eager in the forward call if needed).

    Returns a result dict or None on failure.
    """
    try:
        import torch
    except ImportError:
        return None

    try:
        # Force eager attention for this forward pass
        if hasattr(model.config, "_attn_implementation"):
            model.config._attn_implementation = "eager"

        device = next(model.parameters()).device

        sample_explanations = []
        for i, entry in enumerate(samples):
            prompt = _build_prompt(entry, tokenizer)
            true_label = entry.get("output", "")

            try:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs, output_attentions=True)

                if outputs.attentions is None:
                    raise RuntimeError("Model returned None for attentions")

                last_layer_attn = outputs.attentions[-1]
                avg_attn = last_layer_attn.mean(dim=1).squeeze(0)
                token_importance = avg_attn.sum(dim=0)
                token_importance = token_importance / token_importance.sum()

                scores = token_importance.cpu().float().tolist()
                tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu())

                paired = [
                    {"token": tok, "score": round(s, 6)}
                    for tok, s in zip(tokens, scores)
                ]
                paired.sort(key=lambda x: x["score"], reverse=True)

                sample_explanations.append({
                    "sample_index": i,
                    "input_preview": prompt[:200],
                    "true_label": true_label,
                    "token_scores": paired[:_TOP_K_TOKENS],
                })

            except Exception as exc:
                logger.warning("Attention failed for sample %d: %s", i, exc)
                sample_explanations.append({
                    "sample_index": i,
                    "input_preview": prompt[:200],
                    "true_label": true_label,
                    "token_scores": [],
                })

            if (i + 1) % 10 == 0 or i == len(samples) - 1:
                _log(f"    Attention: {i + 1}/{len(samples)} samples", log_callback)

        explained = sum(1 for s in sample_explanations if s["token_scores"])
        _log(f"  ✓ Attention complete: {explained}/{len(sample_explanations)} samples\n", log_callback)

        return {
            "method": "attention",
            "num_samples": len(sample_explanations),
            "sample_explanations": sample_explanations,
        }

    except Exception as exc:
        logger.warning("Attention fallback failed: %s", exc, exc_info=True)
        _log(f"  ✗ Attention fallback failed: {exc}\n", log_callback)
        return None


# ── Visualisation ──────────────────────────────────────────────────

def _save_heatmap(results: list[dict], output_path: str) -> bool:
    """Generate a combined token-importance bar chart for the first few samples."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.info("matplotlib not available — skipping heatmap generation.")
        return False

    # Find the first result with actual explanations
    best = None
    for r in results:
        if r and r.get("sample_explanations"):
            best = r
            break
    if not best:
        return False

    explanations = [s for s in best["sample_explanations"] if s.get("token_scores")]
    n_samples = min(5, len(explanations))
    if n_samples == 0:
        return False

    method_name = best.get("method", "unknown").upper()
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3 * n_samples))
    if n_samples == 1:
        axes = [axes]

    for ax, sample in zip(axes, explanations[:n_samples]):
        top_tokens = sample["token_scores"][:10]
        tokens = [t["token"] for t in reversed(top_tokens)]
        scores = [t["score"] for t in reversed(top_tokens)]

        bars = ax.barh(tokens, scores, color="#6366f1")
        ax.set_xlabel("Attribution Score")

        label = sample.get("true_label", "")
        ax.set_title(f"True: {label}", fontsize=10)

        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{score:.4f}", va="center", fontsize=8,
            )

    fig.suptitle(f"Token-Level Explanations — {method_name} (Top-10 Tokens)", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved heatmap to %s", output_path)
    return True


# ── Skip result helper ─────────────────────────────────────────────

def _skip_result(reason: str) -> dict:
    """Build a standard skip/failure return dict for run_xai."""
    return {
        "xai_report_path": "",
        "messages": [HumanMessage(content=f"[run_xai] {reason}")],
    }


# ── Main node ──────────────────────────────────────────────────────

def run_xai(state: PipelineState, config: RunnableConfig) -> dict:
    """Generate token-level explanations using SHAP, TransformerLens, and attention.

    Priority: SHAP → TransformerLens → Attention (fallback).
    All methods that succeed are included in the report.

    Writes: xai_report_path, messages
    """
    log_callback = config.get("configurable", {}).get("log_callback")

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

    samples = test_data[:_MAX_SAMPLES]
    base_model = state["base_model"]
    training_config = state.get("training_config", {})

    # XAI hardware defaults: fp16 + 8-bit quantization
    if training_config.get("precision") in (None, "bf16"):
        training_config = {**training_config, "precision": "fp16"}
    if training_config.get("quantization_bit") is None:
        training_config = {**training_config, "quantization_bit": 8}

    # ── Load and merge model ───────────────────────────────────
    header = (
        "\n" + "=" * 60 + "\n"
        "EXPLAINABILITY (XAI)\n"
        f"Model: {base_model}\n"
        f"Adapter: {adapter_path}\n"
        f"Samples: {len(samples)}\n"
        "Methods: SHAP → TransformerLens → Attention (fallback)\n"
        + "=" * 60 + "\n"
    )
    _log(header, log_callback)

    try:
        model, tokenizer = _merge_and_load(base_model, adapter_path, training_config)
    except Exception as exc:
        logger.exception("Failed to load model for XAI")
        return _skip_result(f"FAILED — could not load model: {exc}")

    # ── Run all methods in priority order ──────────────────────
    run_dir = Path(state.get("run_dir", state["output_dir"]))
    xai_dir = run_dir / "xai"
    xai_dir.mkdir(parents=True, exist_ok=True)

    method_results = []

    # 1. SHAP
    shap_result = _run_shap(model, tokenizer, samples, xai_dir, log_callback)
    if shap_result:
        method_results.append(shap_result)

    # 2. TransformerLens
    tl_result = _run_transformer_lens(model, tokenizer, base_model, samples, xai_dir, log_callback)
    if tl_result:
        method_results.append(tl_result)

    # 3. Attention fallback — only if both SHAP and TransformerLens failed
    if not method_results:
        _log("  Both SHAP and TransformerLens unavailable — trying attention fallback...", log_callback)
        attn_result = _run_attention(model, tokenizer, samples, log_callback)
        if attn_result:
            method_results.append(attn_result)

    # ── Unload model ───────────────────────────────────────────
    _release_model(model)
    del model, tokenizer
    _cleanup_gpu()
    _log("✓ Model unloaded, GPU memory freed\n", log_callback)

    if not method_results:
        return _skip_result("FAILED — all XAI methods failed.")

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
    logger.info("Saved XAI report to %s", report_path)

    # ── Heatmap visualisation ──────────────────────────────────
    heatmap_path = xai_dir / "xai_heatmap.png"
    _save_heatmap(method_results, str(heatmap_path))

    # ── Summary ────────────────────────────────────────────────
    methods = ", ".join(r["method"] for r in method_results)
    summary = f"XAI complete ({methods}). Report at {report_path}"

    _log(f"\n✓ {summary}\n", log_callback)

    return {
        "xai_report_path": str(report_path),
        "messages": [
            HumanMessage(content=f"[run_xai] {summary}"),
        ],
    }
