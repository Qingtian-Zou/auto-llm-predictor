"""Node: run_xai — Explainable AI with SHAP, TransformerLens, and attention.

Generates token-level explanations for model predictions using three
methods in priority order:

1. **SHAP** — Partition-based SHAP values via a text-generation pipeline
   wrapper (``shap.Explainer``).  Produces per-token Shapley attribution.
2. **TransformerLens** — Logit attribution via ``TransformerBridge``.
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
        "dtype": dtype,
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

        # Subclass TeacherForcing so __call__ is on the class (Python's MRO
        # finds it) and SHAP's isinstance checks still pass.
        _est_total = 2 * n_tokens + 1
        _shap_progress = {"calls": 0, "last_pct": -1}

        class _TrackedTeacherForcing(shap.models.TeacherForcing):
            """TeacherForcing that reports mask-evaluation progress."""

            def __call__(self, X, Y):
                _shap_progress["calls"] += 1
                pct = min(100, int(_shap_progress["calls"] / _est_total * 100))
                if log_callback and pct >= _shap_progress["last_pct"] + 5:
                    _shap_progress["last_pct"] = pct
                    log_callback(
                        f"    SHAP evaluating masks: "
                        f"{_shap_progress['calls']}/{_est_total} (~{pct}%)"
                    )
                return super().__call__(X, Y)

        shap_model = _TrackedTeacherForcing(model, tokenizer)
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

        # Save per-token SHAP values for all tokens (not just top-K).
        # We use sv.values (partition-level scores replicated to each
        # BPE token within a partition) rather than
        # sv.hierarchical_values[:M] (tree leaf nodes which are often
        # zero because attribution is stored at inner-node level).
        # The file is named shap_leaf_values.json for backward compat.
        try:
            leaf_data = []
            for idx in range(len(prompts)):
                sv = shap_values[idx] if len(prompts) > 1 else shap_values
                tokens_arr = sv.data
                values_arr = sv.values
                if values_arr is not None and tokens_arr is not None:
                    leaf_data.append({
                        "tokens": [str(t) for t in tokens_arr],
                        "values": values_arr.tolist(),
                    })
            if leaf_data:
                leaf_path = xai_dir / "shap_leaf_values.json"
                import json as _json
                Path(leaf_path).write_text(
                    _json.dumps(leaf_data, ensure_ascii=False),
                )
                logger.info("Saved per-token SHAP values to %s", leaf_path)
        except Exception as exc:
            logger.debug("Could not save per-token SHAP values: %s", exc)

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
                "all_token_scores": paired,
            })

            if (i + 1) % 10 == 0 or i == len(prompts) - 1:
                _log(f"    SHAP: {i + 1}/{len(prompts)} samples", log_callback)

        # Save SHAP HTML visualisation.
        # shap.plots.text() crashes on batched TeacherForcing explanations
        # because ragged token counts produce None in _compute_shape.
        # Workaround: call per-sample (each has a consistent 2-D shape).
        shap_html: str | None = None
        try:
            n_html = len(prompts)
            fragments: list[str] = []
            for idx in range(n_html):
                sv = shap_values[idx] if len(prompts) > 1 else shap_values
                fragment = shap.plots.text(sv, display=False)
                if fragment:
                    fragments.append(str(fragment))
            if fragments:
                shap_html = "\n<br>\n".join(fragments)
                html_path = xai_dir / "shap_text.html"
                Path(html_path).write_text(shap_html)
                logger.info("Saved SHAP text plot to %s", html_path)
        except Exception as exc:
            logger.debug("shap.plots.text() failed (%s), falling back to custom HTML", exc)
            try:
                explained_samples = [s for s in sample_explanations if s.get("token_scores")]
                if explained_samples:
                    shap_html = _build_shap_html(explained_samples)
                    html_path = xai_dir / "shap_text.html"
                    Path(html_path).write_text(shap_html)
                    logger.info("Saved SHAP text plot (fallback) to %s", html_path)
            except Exception as exc2:
                logger.debug("Fallback HTML also failed: %s", exc2)
                _log(f"  ✗ Could not save SHAP HTML plot: {exc}\n", log_callback)

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
        from transformer_lens.model_bridge import TransformerBridge
    except ImportError:
        logger.info("transformer_lens not installed — skipping TransformerLens method.")
        return None

    try:
        import torch

        # TransformerLens 3.1.0's set_processed_weights wraps every weight in
        # nn.Parameter() (requires_grad=True), which fails on int8/uint8
        # tensors from BitsAndBytes-quantized models. Detect and skip cleanly
        # rather than crash mid-processing.
        if any(not p.dtype.is_floating_point for p in model.parameters()):
            logger.info(
                "TransformerLens requires a non-quantized model; the merged "
                "model contains integer-dtype weights. Skipping — re-run with "
                "--quantization-bit none to enable."
            )
            _log("  ⚠ TransformerLens skipped (model is quantized)\n", log_callback)
            return None

        _log("  Loading TransformerBridge...", log_callback)
        hooked = TransformerBridge.boot_transformers(
            base_model,
            hf_model=model,
            tokenizer=tokenizer,
            device=model.device,
        )
        # Apply HookedTransformer-equivalent weight processing so W_U and
        # the resid_post cache key match the legacy logit-attribution math.
        try:
            hooked.enable_compatibility_mode(disable_warnings=True)
        except RuntimeError as exc:
            logger.warning(
                "enable_compatibility_mode failed (%s); retrying with "
                "no_processing=True — logit attribution will skip LN folding "
                "and weight centering.", exc,
            )
            hooked.enable_compatibility_mode(
                disable_warnings=True, no_processing=True,
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


# ── SHAP HTML helper ──────────────────────────────────────────────

def _build_shap_html(sample_explanations: list[dict]) -> str:
    """Build a self-contained HTML page showing per-token SHAP attributions.

    This replaces ``shap.plots.text()`` which crashes on TeacherForcing
    explanations due to a shape-computation bug in SHAP (``_compute_shape``
    returns ``(None,)`` for string data, causing ``range(None)``).
    """
    import html as html_mod

    lines = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        "<title>SHAP Token Attributions</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;margin:2em;background:#fafafa}",
        ".sample{background:#fff;border:1px solid #ddd;border-radius:6px;"
        "padding:1em;margin-bottom:1.5em}",
        ".sample h3{margin:0 0 .4em}",
        ".tokens{display:flex;flex-wrap:wrap;gap:2px;margin:.6em 0}",
        ".tok{padding:2px 4px;border-radius:3px;font-size:13px;font-family:monospace}",
        "table{border-collapse:collapse;font-size:13px}",
        "td,th{padding:3px 8px;border:1px solid #ddd;text-align:left}",
        "th{background:#f5f5f5}",
        "</style></head><body>",
        "<h1>SHAP Token-Level Attributions</h1>",
    ]

    for s in sample_explanations:
        tokens = s.get("token_scores", [])
        if not tokens:
            continue
        idx = s.get("sample_index", "?")
        label = html_mod.escape(str(s.get("true_label", "")))
        preview = html_mod.escape(s.get("input_preview", "")[:200])

        max_score = max((t["score"] for t in tokens), default=1) or 1

        lines.append(f"<div class='sample'><h3>Sample {idx}</h3>")
        lines.append(f"<p><b>True label:</b> {label}</p>")
        lines.append(f"<p><b>Input:</b> <code>{preview}</code></p>")

        # Token ribbon — colour intensity proportional to score
        lines.append("<div class='tokens'>")
        for t in tokens:
            score = t["score"]
            intensity = int(255 * (1 - score / max_score))
            bg = f"rgb(255,{intensity},{intensity})"
            tok_text = html_mod.escape(t["token"])
            lines.append(
                f"<span class='tok' style='background:{bg}' "
                f"title='score={score:.4f}'>{tok_text}</span>"
            )
        lines.append("</div>")

        # Score table
        lines.append("<table><tr><th>Token</th><th>Score</th></tr>")
        for t in tokens:
            tok_text = html_mod.escape(t["token"])
            lines.append(f"<tr><td>{tok_text}</td><td>{t['score']:.6f}</td></tr>")
        lines.append("</table></div>")

    lines.append("</body></html>")
    return "\n".join(lines)


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

