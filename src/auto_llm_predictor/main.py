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

"""CLI entry point for the auto-LLM-predictor pipeline."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langgraph.types import Command

from auto_llm_predictor.checkpoint import load_state
from auto_llm_predictor.graph import build_graph
from auto_llm_predictor.utils import is_local_model, normalize_path, validate_local_model


def _build_training_config(args) -> dict:
    """Build the training_config dict from parsed CLI arguments."""
    config = {
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "use_dora": args.use_dora,
        "cutoff_len": args.cutoff_len,
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "quantization_bit": args.quantization_bit,
        "flash_attn": args.flash_attn,
        "precision": args.precision,
        "test_ratio": args.test_ratio,
        "finetune_max_retries": args.finetune_retries,
    }
    if getattr(args, "template", ""):
        config["template"] = args.template
    return config


def main():
    # Load .env from project root (or current directory)
    load_dotenv()

    # Read env vars with fallbacks
    env_endpoint = os.getenv("openAI_endpoint", "")
    env_llm_provider = os.getenv("llm_provider", "openai")
    if env_llm_provider == "ollama":
        env_api_base = f"http://{env_endpoint}" if env_endpoint else ""
    else:
        env_api_base = f"http://{env_endpoint}/v1" if env_endpoint else ""
    env_api_key = os.getenv("auth_key", "")
    env_agent_model = os.getenv("agent_LLM", "")
    env_coder_model = os.getenv("coder_LLM", "")

    parser = argparse.ArgumentParser(
        description="Auto LLM Predictor — Automatically build a fine-tuned LLM predictor from CSV data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Minimal (auto-detect target)
  auto-llm-predictor --csv data/my_dataset.csv --model mistralai/Mistral-7B-Instruct-v0.3

  # Specify target and output directory
  auto-llm-predictor --csv data/patients.csv --target response --model mistralai/Mistral-7B-Instruct-v0.3 --output output/exp1

  # Use a different agent LLM
  auto-llm-predictor --csv data/patients.csv --model mistralai/Mistral-7B-Instruct-v0.3 \\
    --agent-model gpt-4o --agent-api-base https://api.openai.com/v1 --agent-api-key sk-...
""",
    )

    # Required
    parser.add_argument("--csv", required=True, help="Path to the raw CSV file")
    parser.add_argument("--model", required=True,
                        help="HuggingFace model ID or local model directory path for fine-tuning")

    parser.add_argument("--template", default="",
                        help="LlamaFactory chat template (e.g. llama3, qwen, mistral). "
                             "Auto-detected from model name or config.json if not set.")

    # Optional
    parser.add_argument("--target", default="", help="Target column name (auto-detected if empty)")
    parser.add_argument("--output", default="", help="Output directory (default: output/<csv_stem>)")
    parser.add_argument("--test-csv", default="", help="Optional separate test CSV file")
    parser.add_argument("--test-ratio", type=float, default=0.2,
                        help="Test split ratio when no test CSV is provided (default: 0.2)")
    parser.add_argument("--start-from",
                        choices=["review_prep", "register", "config"],
                        default=None,
                        help="Resume from a previous run (requires --output pointing to existing experiment)")

    # Agent LLM configuration (defaults from .env)
    parser.add_argument("--agent-api-base", default=env_api_base or None,
                        help="OpenAI-compatible API base URL (env: openAI_endpoint)")
    parser.add_argument("--agent-api-key", default=env_api_key or None,
                        help="API key for the agent LLM (env: auth_key)")
    parser.add_argument("--agent-model", default=env_agent_model or None,
                        help="Model ID for reasoning/planning (env: agent_LLM)")
    parser.add_argument("--coder-model", default=env_coder_model or None,
                        help="Model ID for code generation (env: coder_LLM; falls back to agent-model)")
    parser.add_argument("--agent-temperature", type=float, default=0.2,
                        help="Temperature for the agent LLM")
    parser.add_argument("--llm-provider", default=env_llm_provider,
                        choices=["openai", "ollama"],
                        help="LLM backend: 'openai' for OpenAI-compatible APIs (LM Studio, vLLM), "
                             "'ollama' for Ollama native API (env: llm_provider; default: openai)")

    # Logging
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging (print LLM prompts and responses to stderr)")

    # Training hyperparameters (override defaults in generated YAML)
    hp = parser.add_argument_group("Training Hyperparameters")
    hp.add_argument("--lora-rank", type=int, default=64, help="LoRA rank (default: 64)")
    hp.add_argument("--lora-alpha", type=int, default=128, help="LoRA alpha (default: 128)")
    hp.add_argument("--use-dora", action="store_true", default=False, help="Enable DoRA adapter")
    hp.add_argument("--cutoff-len", type=int, default=4096, help="Max input token length (default: 4096; ignored when --auto-cutoff is set)")
    hp.add_argument("--auto-cutoff", action="store_true", default=False,
                    help="Automatically determine cutoff_len from training data statistics instead of using --cutoff-len")

    hp.add_argument("--epochs", type=float, default=3.0, help="Number of training epochs (default: 3.0)")
    hp.add_argument("--learning-rate", type=str, default="2.0e-5", help="Learning rate (default: 2.0e-5)")
    hp.add_argument("--batch-size", type=int, default=1, help="Per-device train batch size (default: 2)")
    hp.add_argument("--grad-accumulation", type=int, default=16, help="Gradient accumulation steps (default: 8)")
    hp.add_argument("--logging-steps", type=int, default=10, help="Logging interval in steps (default: 10)")
    hp.add_argument("--save-steps", type=int, default=500, help="Checkpoint save interval (default: 500)")
    hp.add_argument("--quantization-bit", type=int, choices=[4, 8], default=None, help="Quantization bits (4 or 8)")
    hp.add_argument("--flash-attn", default="auto", choices=["auto", "fa2", "disabled"], help="Flash attention mode (default: auto)")
    hp.add_argument("--precision", default="bf16", choices=["bf16", "fp16"], help="Training precision (default: bf16)")
    hp.add_argument("--finetune-retries", type=int, default=3,
                    help="Max auto-resume attempts when fine-tuning fails mid-training (default: 3; -1 for unlimited)")

    args = parser.parse_args()

    # ── Setup logging ──────────────────────────────────────────
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Validate inputs ────────────────────────────────────────
    csv_path = Path(normalize_path(args.csv)).resolve()
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    test_csv_path = ""
    if args.test_csv:
        test_csv_path = normalize_path(str(Path(args.test_csv).resolve()))
        if not Path(test_csv_path).exists():
            print(f"Error: Test CSV file not found: {test_csv_path}", file=sys.stderr)
            sys.exit(1)

    # Validate agent LLM configuration
    missing = []
    if not args.agent_api_base:
        missing.append("--agent-api-base (or set openAI_endpoint in .env)")
    if not args.agent_api_key and args.llm_provider != "ollama":
        missing.append("--agent-api-key (or set auth_key in .env)")
    if not args.agent_model:
        missing.append("--agent-model (or set agent_LLM in .env)")
    if missing:
        print("Error: Missing agent LLM configuration:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        sys.exit(1)

    # Ollama does not require an API key; set a placeholder to avoid None issues
    if args.llm_provider == "ollama" and not args.agent_api_key:
        args.agent_api_key = "ollama"

    # Validate test ratio
    if not (0 < args.test_ratio < 1):
        print(f"Error: --test-ratio must be between 0 and 1 (exclusive), got {args.test_ratio}",
              file=sys.stderr)
        sys.exit(1)

    # Validate local model path (if applicable)
    if is_local_model(args.model):
        errors = validate_local_model(args.model)
        if errors:
            print("Error: Local model directory is missing required files:", file=sys.stderr)
            for e in errors:
                print(f"  {e}", file=sys.stderr)
            sys.exit(1)
        args.model = str(Path(args.model).resolve())

    output_dir = args.output if args.output else str(Path("output") / csv_path.stem)
    output_dir = normalize_path(str(Path(output_dir).resolve()))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── Build and run the graph ────────────────────────────────
    print("=" * 60)
    print("Auto LLM Predictor")
    print("=" * 60)
    print(f"CSV:          {csv_path}")
    if test_csv_path:
        print(f"Test CSV:     {test_csv_path}")
    else:
        print(f"Test ratio:   {args.test_ratio}")
    print(f"Target:       {args.target or '(auto-detect)'}")
    model_display = f"{args.model} (local)" if is_local_model(args.model) else args.model
    print(f"Base model:   {model_display}")
    print(f"Output dir:   {output_dir}")
    cutoff_display = '(auto-detect)' if args.auto_cutoff else str(args.cutoff_len)
    print(f"Cutoff len:   {cutoff_display}")
    coder_model = args.coder_model or args.agent_model
    print(f"LLM provider: {args.llm_provider}")
    print(f"Agent LLM:    {args.agent_model} @ {args.agent_api_base}")
    print(f"Coder LLM:    {coder_model}")
    print("=" * 60)

    app = build_graph(
        api_base=args.agent_api_base,
        api_key=args.agent_api_key,
        agent_model=args.agent_model,
        coder_model=coder_model,
        temperature=args.agent_temperature,
        verbose=args.verbose,
        llm_provider=args.llm_provider,
    )

    # ── Build initial state ─────────────────────────────────────
    training_config = _build_training_config(args)

    if args.start_from:
        # Resume mode: load saved state, override with CLI args
        try:
            initial_state = load_state(output_dir)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        initial_state["start_from"] = args.start_from
        initial_state["output_dir"] = output_dir
        initial_state["base_model"] = args.model
        initial_state["messages"] = []
        initial_state["auto_cutoff"] = args.auto_cutoff


        # Merge CLI training_config overrides into the saved state
        saved_config = initial_state.get("training_config", {})
        saved_config.update(training_config)
        initial_state["training_config"] = saved_config

        print(f"Resuming from: {args.start_from}")
    else:
        initial_state = {
            "csv_path": str(csv_path),
            "test_csv_path": test_csv_path,
            "target_column": args.target,
            "base_model": args.model,
            "output_dir": output_dir,
            "start_from": "explore_data",
            "messages": [],
            "prep_attempts": 0,
            "finetune_attempts": 0,
            "auto_cutoff": args.auto_cutoff,

            "training_config": training_config,
        }

    # Thread config for checkpointer (required for interrupt/resume)
    thread_config = {"configurable": {"thread_id": uuid.uuid4().hex}}

    print("\nStarting pipeline...\n")

    try:
        final_state = _run_with_review_loop(app, initial_state, thread_config)
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logging.exception("Pipeline failed")
        print(f"\nPipeline failed: {e}", file=sys.stderr)
        sys.exit(1)

    # ── Print results ──────────────────────────────────────────
    _print_results(final_state, output_dir)


def _run_with_review_loop(app, initial_state: dict, config: dict) -> dict:
    """Run the graph, handling interrupt()s for human-in-the-loop review.

    When the graph pauses at ``review_prep_data``, the interrupt value
    (a data summary) is printed to the console and the user is prompted
    for input.  Their response is sent back via ``Command(resume=...)``.
    """
    # First invocation
    result = app.invoke(initial_state, config=config)

    # Check for interrupts and handle them in a loop
    while True:
        state = app.get_state(config)

        # If there are no pending tasks, the graph has finished
        if not state.tasks:
            return state.values

        # Find the interrupt value from pending tasks
        interrupt_value = None
        for task in state.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                interrupt_value = task.interrupts[0].value
                break

        if interrupt_value is None:
            # No interrupt, graph is done
            return state.values

        # Print the review summary to the console
        print("\n" + str(interrupt_value))

        # Get user input
        print()
        user_input = input("Your response: ").strip()

        if not user_input:
            user_input = "approve"

        print(f"\n→ Resuming pipeline with: {user_input!r}\n")

        # Resume the graph with the user's response
        result = app.invoke(Command(resume=user_input), config=config)

    return result


def _print_results(final_state: dict, output_dir: str):
    """Print pipeline completion summary (eval metrics already shown by node)."""
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)

    run_dir = final_state.get("run_dir", output_dir)
    print(f"\nRun artefacts:    {run_dir}")
    print(f"Data directory:   {Path(output_dir) / 'data'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
