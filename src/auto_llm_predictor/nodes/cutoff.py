"""Node: determine_cutoff_len — Auto-detect the appropriate cutoff length from training data.

Algorithm
---------
1. Load ``train.json`` (Alpaca-format list of dicts with 'instruction', 'input', 'output').
2. Load the ``AutoTokenizer`` for ``state["base_model"]`` and encode each example's text.
   If the tokenizer cannot be loaded (model not cached, network unavailable, etc.), fall
   back gracefully to the character-count heuristic (``len(text) // 4``).
3. Compute the 100th percentile (maximum) of all token lengths and round up to the next
   multiple of 512.  This is the *primary* recommendation.
4. If the primary recommendation exceeds 10 000 tokens, also compute the 95th, 90th, 85th,
   and 80th percentile alternatives and surface them to the user via ``interrupt()`` so
   they can choose a lower value if desired.
5. If the user-supplied ``cutoff_len`` (from ``--cutoff-len`` CLI flag) is non-zero, skip
   auto-detection entirely and use that value.
6. If ``auto_cutoff`` is False (i.e., ``--auto-cutoff`` was not passed), skip auto-
   detection and keep the CLI-provided value.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.types import interrupt

from auto_llm_predictor.state import PipelineState

logger = logging.getLogger(__name__)

# Threshold above which the user is asked to choose a lower percentile
_HIGH_CUTOFF_THRESHOLD = 10_000

# Percentile alternatives offered when the max exceeds the threshold
_ALTERNATIVE_PERCENTILES = [95, 90, 85, 80]


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

def _load_tokenizer(base_model: str):
    """Try to load the HuggingFace tokenizer for *base_model*.

    Returns the tokenizer on success, or ``None`` if the model is not cached /
    unavailable.  Callers should fall back to :func:`_estimate_tokens` when
    ``None`` is returned.
    """
    try:
        from transformers import AutoTokenizer  # type: ignore[import]

        logger.info("Loading tokenizer for '%s' ...", base_model)
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True,
            use_fast=True,
        )
        logger.info("Tokenizer loaded successfully.")
        return tokenizer
    except Exception as exc:
        logger.warning(
            "Could not load tokenizer for %r (%s) — "
            "falling back to character heuristic (len(text) // 4).",
            base_model,
            exc,
        )
        return None


def _estimate_tokens(text: str) -> int:
    """Estimate token count via the 4-characters-per-token heuristic (fallback only)."""
    return max(1, len(text) // 4)


def _count_tokens(text: str, tokenizer) -> int:
    """Return the exact token count using *tokenizer*, or fall back to the heuristic."""
    if tokenizer is None:
        return _estimate_tokens(text)
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception as exc:
        logger.debug("tokenizer.encode failed (%s) — using heuristic", exc)
        return _estimate_tokens(text)


# ---------------------------------------------------------------------------
# Rounding / percentile helpers
# ---------------------------------------------------------------------------

def _round_up_to_multiple(value: int, multiple: int = 512) -> int:
    """Round *value* up to the nearest multiple of *multiple*."""
    return math.ceil(value / multiple) * multiple


def _percentile(sorted_values: list[int], pct: float) -> int:
    """Return the *pct*-th percentile of a pre-sorted list (0-100 scale)."""
    if not sorted_values:
        return 0
    idx = math.ceil(pct / 100.0 * len(sorted_values)) - 1
    idx = max(0, min(idx, len(sorted_values) - 1))
    return sorted_values[idx]


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def _default_cutoff(reason: str) -> dict:
    """Return a safe default cutoff_len=1024 with an explanatory message."""
    logger.warning("%s — using safe default cutoff_len=1024", reason)
    return {
        "cutoff_len": 1024,
        "messages": [
            HumanMessage(content=f"[determine_cutoff_len] {reason}; using default cutoff_len=1024."),
        ],
    }

def determine_cutoff_len(state: PipelineState) -> dict:
    """Determine the cutoff length to use for LlamaFactory training.

    - If ``auto_cutoff`` is False in state, skips auto-detection and keeps
      ``training_config["cutoff_len"]`` unchanged.
    - Otherwise, encodes every training example with the base model's
      ``AutoTokenizer`` (falling back to a character heuristic if the
      tokenizer is unavailable) and stores the resolved value as
      ``cutoff_len`` on the state.
    - If the computed value exceeds 10 000, presents the user with percentile
      alternatives via an interrupt and uses their chosen value.

    Writes: cutoff_len, messages
    """
    tc = state.get("training_config", {})
    auto_cutoff = state.get("auto_cutoff", False)

    # ── Skip: user explicitly chose a cutoff length ───────────────────────────
    if not auto_cutoff:
        fixed = tc.get("cutoff_len", 4096)
        logger.info("Auto-cutoff disabled — using user-supplied cutoff_len=%d", fixed)
        print(
            f"\n{'=' * 50}\nCUTOFF LENGTH\n{'=' * 50}\n"
            f"Using user-specified cutoff_len: {fixed}\n"
            f"(Pass --auto-cutoff to let the pipeline compute this automatically.)",
            flush=True,
        )
        return {
            "cutoff_len": fixed,
            "messages": [
                HumanMessage(
                    content=f"[determine_cutoff_len] User-specified cutoff_len={fixed}."
                ),
            ],
        }

    # ── Load training data ────────────────────────────────────────────────────
    train_path = Path(state.get("train_data_path", ""))
    if not train_path.exists():
        output_dir = Path(state["output_dir"])
        train_path = output_dir / "data" / "train.json"

    if not train_path.exists():
        return _default_cutoff(f"train.json not found at {train_path}")

    try:
        with open(train_path) as f:
            data = json.load(f)
    except Exception as exc:
        return _default_cutoff(f"Could not read train.json ({exc})")

    if not data:
        return _default_cutoff("train.json is empty")

    # ── Load tokenizer ────────────────────────────────────────────────────────
    base_model = state.get("base_model", "")
    tokenizer = _load_tokenizer(base_model) if base_model else None
    token_source = "AutoTokenizer" if tokenizer is not None else "character heuristic (len // 4)"

    # ── Count token lengths ───────────────────────────────────────────────────
    lengths: list[int] = []
    for entry in data:
        text = (
            str(entry.get("instruction", ""))
            + str(entry.get("input", ""))
            + str(entry.get("output", ""))
        )
        lengths.append(_count_tokens(text, tokenizer))

    lengths.sort()

    max_len = lengths[-1]
    primary = _round_up_to_multiple(max_len)

    p50 = _percentile(lengths, 50)
    p90 = _percentile(lengths, 90)
    p95 = _percentile(lengths, 95)

    stats_lines = [
        f"Token length statistics over {len(lengths)} training examples (source: {token_source}):",
        f"  50th percentile : {p50}",
        f"  90th percentile : {p90}",
        f"  95th percentile : {p95}",
        f"  100th (maximum) : {max_len}",
        f"  → Primary recommendation (max, rounded to 512): {primary}",
    ]
    stats_str = "\n".join(stats_lines)
    logger.info(stats_str)

    # ── If primary ≤ 10 000, accept automatically ─────────────────────────────
    if primary <= _HIGH_CUTOFF_THRESHOLD:
        print(
            f"\n{'=' * 50}\nCUTOFF LENGTH (auto-detected)\n{'=' * 50}\n{stats_str}\n",
            flush=True,
        )
        return {
            "cutoff_len": primary,
            "messages": [
                HumanMessage(
                    content=f"[determine_cutoff_len] Auto-detected cutoff_len={primary} "
                    f"(max tokens={max_len}, n={len(lengths)}, source={token_source})."
                ),
            ],
        }

    # ── Primary > 10 000 — ask user to pick a percentile ─────────────────────
    alternatives: dict[str, int] = {}
    for pct in _ALTERNATIVE_PERCENTILES:
        raw = _percentile(lengths, pct)
        alternatives[f"p{pct}"] = _round_up_to_multiple(raw)

    alt_lines = [
        f"  p{pct:3d} ({pct}th percentile): {val}"
        for pct, val in zip(_ALTERNATIVE_PERCENTILES, alternatives.values())
    ]

    prompt = (
        f"\n{'=' * 60}\n"
        f"CUTOFF LENGTH AUTO-DETECTION\n"
        f"{'=' * 60}\n\n"
        f"{stats_str}\n\n"
        f"⚠️  The recommended cutoff_len ({primary}) exceeds 10 000 tokens.\n"
        f"   Training with a very large cutoff may require significantly more GPU memory.\n\n"
        f"Percentile alternatives (rounded to nearest 512):\n"
        + "\n".join(alt_lines)
        + "\n\n"
        f"Options:\n"
        f"  • Type 'approve' or press Enter to accept {primary} (100th percentile).\n"
        f"  • Type a percentile key to use that value: p95, p90, p85, p80\n"
        f"  • Type a custom integer (e.g. '8192') to use that value directly.\n"
    )

    user_choice = interrupt(prompt)
    chosen = _parse_cutoff_choice(user_choice, primary, alternatives)

    logger.info("User chose cutoff_len=%d (input=%r)", chosen, user_choice)
    print(f"\n→ Using cutoff_len={chosen}\n", flush=True)

    return {
        "cutoff_len": chosen,
        "messages": [
            HumanMessage(
                content=f"[determine_cutoff_len] cutoff_len={chosen} (user chose from "
                f"percentile options; max was {max_len}, primary was {primary}, "
                f"source={token_source})."
            ),
        ],
    }


# ---------------------------------------------------------------------------
# Choice parsing helper
# ---------------------------------------------------------------------------

def _parse_cutoff_choice(
    user_input: str,
    primary: int,
    alternatives: dict[str, int],
) -> int:
    """Parse the user's percentile choice and return the resolved integer cutoff."""
    raw = (user_input or "").strip().lower()

    if not raw or raw == "approve":
        return primary

    if raw in alternatives:
        return alternatives[raw]

    try:
        val = int(raw)
        if val > 0:
            return _round_up_to_multiple(val)
    except ValueError:
        pass

    logger.warning(
        "Unrecognised cutoff choice %r — falling back to primary (%d)", user_input, primary
    )
    return primary
