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

"""Standalone XAI (Explainability) module for the auto-LLM-predictor.

Runs token-level explainability analysis on a completed training run
without re-running the pipeline.  Loads the fine-tuned model (base +
LoRA adapter), reads the test data and predictions, and produces SHAP,
TransformerLens, and/or attention-based explanations.

Both a programmatic API (``run_standalone_xai``) and a CLI entry point
(``auto-llm-predictor-xai``) are provided.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path

from auto_llm_predictor.utils import normalize_path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_pipeline_state(output_dir: str) -> dict:
    """Load the saved pipeline state from a training output directory."""
    from auto_llm_predictor.checkpoint import load_state

    return load_state(output_dir)


# ---------------------------------------------------------------------------
# Core XAI function
# ---------------------------------------------------------------------------

def run_standalone_xai(
    *,
    output_dir: str,
    run_dir: str,
    max_samples: int = 50,
    precision: str = "fp16",
    quantization_bit: int | None = 8,
    flash_attn: str = "auto",
    log_callback: Callable[[str], None] | None = None,
) -> dict:
    """Run standalone XAI analysis on a completed training run.

    Loads the fine-tuned model, reads test data produced during training,
    and runs SHAP → TransformerLens → Attention (fallback) explanations.

    Parameters
    ----------
    output_dir : str
        Training output directory (contains ``.pipeline_state.json``).
    run_dir : str
        Training run directory (contains the LoRA adapter under ``sft/``).
    max_samples : int
        Maximum number of test samples to explain (default 50).
    precision : str
        Model precision — ``"fp16"`` (default) or ``"bf16"``.
    quantization_bit : int | None
        Quantization bits (4 or 8, default 8).
    flash_attn : str
        Flash attention mode (default ``"auto"``).
    log_callback : callable, optional
        Callback for progress messages (used by web UI SSE).

    Returns
    -------
    dict
        ``{"xai_report_path": str, "xai_results": list,
        "methods_succeeded": list, "num_samples": int}``
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

    # ── Load test data ─────────────────────────────────────────
    test_data_path = state.get("test_data_path", "")
    # Fallback: reconstruct from output_dir when state path is stale
    if not test_data_path or not Path(test_data_path).exists():
        test_data_path = str(Path(output_dir) / "data" / "test.json")
    if not Path(test_data_path).exists():
        raise FileNotFoundError(
            f"Test data not found at {test_data_path}"
        )

    with open(test_data_path) as f:
        test_data = json.load(f)

    if not test_data:
        raise ValueError("Test data is empty — nothing to explain.")

    samples = test_data[:max_samples]

    # ── XAI hardware defaults ──────────────────────────────────
    if training_config.get("precision") in (None, "bf16"):
        training_config["precision"] = precision
    if training_config.get("quantization_bit") is None:
        training_config["quantization_bit"] = quantization_bit

    # ── Load and merge model ───────────────────────────────────
    header = (
        "\n" + "=" * 60 + "\n"
        "STANDALONE XAI — Explainability Analysis\n"
        f"Model: {base_model}\n"
        f"Adapter: {adapter_path}\n"
        f"Samples: {len(samples)}\n"
        "Methods: SHAP → TransformerLens → Attention (fallback)\n"
        + "=" * 60 + "\n"
    )
    _log(header)

    model, tokenizer = _merge_and_load(base_model, adapter_path, training_config)

    # ── Run XAI methods ────────────────────────────────────────
    xai_dir = Path(run_dir) / "xai"
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
        return {
            "xai_report_path": "",
            "xai_results": [],
            "methods_succeeded": [],
            "num_samples": len(samples),
        }

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

    methods = ", ".join(r["method"] for r in method_results)
    _log(f"\nStandalone XAI complete ({methods}). Report at {report_path}\n")

    return {
        "xai_report_path": str(report_path),
        "xai_results": method_results,
        "methods_succeeded": [r["method"] for r in method_results],
        "num_samples": len(samples),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """CLI entry point for standalone XAI mode."""
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Auto LLM Predictor — Standalone XAI Mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Run XAI on a completed training run
  auto-llm-predictor-xai --output-dir output/my_dataset \\
    --run-dir output/my_dataset/run_20260307_120000

  # With custom sample cap and precision
  auto-llm-predictor-xai --output-dir output/my_dataset \\
    --run-dir output/my_dataset/run_20260307_120000 \\
    --max-samples 20 --precision bf16
""",
    )

    # Required
    parser.add_argument(
        "--output-dir", required=True,
        help="Training output directory (contains .pipeline_state.json)",
    )
    parser.add_argument(
        "--run-dir", required=True,
        help="Training run directory (contains the LoRA adapter under sft/)",
    )

    # Options
    parser.add_argument(
        "--max-samples", type=int, default=50,
        help="Maximum number of test samples to explain (default: 50)",
    )
    parser.add_argument(
        "--precision", default="fp16", choices=["bf16", "fp16"],
        help="Precision (default: fp16)",
    )
    parser.add_argument(
        "--quantization-bit", type=int, choices=[4, 8], default=8,
        help="Quantization bits (default: 8)",
    )
    parser.add_argument(
        "--flash-attn", default="auto", choices=["auto", "fa2", "disabled"],
        help="Flash attention mode (default: auto)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # ── Setup logging ──────────────────────────────────────────
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Validate ───────────────────────────────────────────────
    output_dir = normalize_path(str(Path(args.output_dir).resolve()))
    run_dir = normalize_path(str(Path(args.run_dir).resolve()))

    if not Path(output_dir).exists():
        print(f"Error: Output directory not found: {output_dir}", file=sys.stderr)
        sys.exit(1)
    if not Path(run_dir).exists():
        print(f"Error: Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("Auto LLM Predictor — Standalone XAI Mode")
    print("=" * 60)
    print(f"Output dir:    {output_dir}")
    print(f"Run dir:       {run_dir}")
    print(f"Max samples:   {args.max_samples}")
    print(f"Precision:     {args.precision}")
    print(f"Quantization:  {args.quantization_bit}-bit")
    print("=" * 60 + "\n")

    try:
        result = run_standalone_xai(
            output_dir=output_dir,
            run_dir=run_dir,
            max_samples=args.max_samples,
            precision=args.precision,
            quantization_bit=args.quantization_bit,
            flash_attn=args.flash_attn,
        )

        print("\n" + "=" * 60)
        print("Standalone XAI Complete!")
        print("=" * 60)
        if result.get("xai_report_path"):
            print(f"Report:     {result['xai_report_path']}")
        print(f"Samples:    {result['num_samples']}")
        methods = result.get("methods_succeeded", [])
        print(f"Methods:    {', '.join(methods) if methods else 'none succeeded'}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nXAI analysis interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logging.exception("XAI analysis failed")
        print(f"\nXAI analysis failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
