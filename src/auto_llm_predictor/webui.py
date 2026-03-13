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

"""Web UI server for the auto-LLM-predictor pipeline.

Provides a browser-based interface that wraps the existing LangGraph
pipeline.  The backend streams pipeline progress via Server-Sent Events
and pauses at review checkpoints for user interaction.

Usage:
    pip install -e ".[webui]"
    auto-llm-predictor-webui          # starts on http://localhost:8000
    auto-llm-predictor-webui --port 9000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import StreamingResponse

from auto_llm_predictor.utils import is_local_model, normalize_path, validate_local_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Run state
# ---------------------------------------------------------------------------

@dataclass
class RunState:
    """Tracks a single pipeline run."""

    run_id: str
    status: str = "pending"          # pending | running | interrupted | completed | error
    cancelled: bool = False
    current_node: str = ""
    error: str = ""
    interrupt_summary: str = ""       # review text shown to the user
    events: list[dict] = field(default_factory=list)
    _events_lock: threading.Lock = field(default_factory=threading.Lock)
    queue: asyncio.Queue | None = None
    thread: threading.Thread | None = None
    # LangGraph handles
    app: Any = None
    thread_config: dict = field(default_factory=dict)
    loop: asyncio.AbstractEventLoop | None = None
    # pipeline state for results
    final_state: dict = field(default_factory=dict)
    # review synchronization
    _review_event: threading.Event = field(default_factory=threading.Event)
    _review_response: str | None = None


_runs: dict[str, RunState] = {}
_MAX_COMPLETED_RUNS = 50


def _prune_completed_runs() -> None:
    """Remove oldest completed/error runs when the history exceeds the limit."""
    finished = [
        rid for rid, r in _runs.items()
        if r.status in ("completed", "error")
    ]
    excess = len(finished) - _MAX_COMPLETED_RUNS
    if excess > 0:
        for rid in finished[:excess]:
            del _runs[rid]


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------

def _emit(run: RunState, event_type: str, data: dict | str) -> None:
    """Push an event both to the in-memory log and the async queue."""
    payload = data if isinstance(data, dict) else {"message": data}
    evt = {"event": event_type, **payload}
    with run._events_lock:
        run.events.append(evt)
    if run.queue and run.loop:
        run.loop.call_soon_threadsafe(run.queue.put_nowait, evt)


# ---------------------------------------------------------------------------
# Pipeline runner (runs in a background thread)
# ---------------------------------------------------------------------------

def _stream_graph(run: RunState, input_data: dict | Any) -> None:
    """Stream the graph, emitting node_start / node_complete SSE events.

    Uses stream_mode=["updates", "debug"] so that the "debug" stream fires a
    "task" event *before* a node begins executing and a "task_result" event
    *after* it completes.  This ensures node_start reaches the browser while
    the node is still running, enabling the breathing-light animation to
    display for the full duration of the stage.

    Returns when the graph finishes or hits an interrupt.
    """
    for mode, chunk in run.app.stream(
        input_data, config=run.thread_config, stream_mode=["updates", "debug"]
    ):
        if run.cancelled:
            break

        if mode == "debug":
            # chunk is a dict with keys: type, timestamp, step, payload
            event_type = chunk.get("type")
            payload = chunk.get("payload", {})
            node_name = payload.get("name", "")

            # Skip internal LangGraph bookkeeping nodes
            if not node_name or node_name.startswith("__") or node_name == "route_start":
                continue

            if event_type == "task":
                # Node is about to start — fire start event immediately
                run.current_node = node_name
                _emit(run, "node_start", {"node": node_name, "message": f"Running: {node_name}"})

            elif event_type == "task_result":
                # Node just finished
                _emit(run, "node_complete", {"node": node_name, "message": f"Completed: {node_name}"})

        # Intercept evaluation results from updates so we can surface them
        # immediately, before the optional XAI node runs.
        if mode == "updates" and isinstance(chunk, dict) and "run_evaluation" in chunk:
            eval_output = chunk["run_evaluation"]
            if "eval_results" in eval_output:
                run_dir = run.app.get_state(run.thread_config).values.get("run_dir", "")
                _emit(run, "eval_results", _format_results({
                    "eval_results": eval_output["eval_results"],
                    "run_dir": run_dir,
                }))


def _run_pipeline(run: RunState, initial_state: dict) -> None:
    """Execute the LangGraph pipeline, emitting SSE events."""
    from langgraph.types import Command

    try:
        run.status = "running"
        _emit(run, "status", {"status": "running", "message": "Pipeline started"})

        # First invocation — stream node-by-node
        _stream_graph(run, initial_state)

        # After streaming finishes, check state for interrupts or completion
        while not run.cancelled:
            state = run.app.get_state(run.thread_config)

            if not state.tasks:
                run.status = "completed"
                run.final_state = state.values
                _emit(run, "complete", _format_results(state.values))
                return

            # Look for an interrupt
            interrupt_value = None
            interrupted_node = ""
            for task in state.tasks:
                if hasattr(task, "interrupts") and task.interrupts:
                    interrupt_value = task.interrupts[0].value
                    interrupted_node = getattr(task, "name", "")
                    break

            if interrupt_value is None:
                run.status = "completed"
                run.final_state = state.values
                _emit(run, "complete", _format_results(state.values))
                return

            # Surface the interrupt to the browser
            run.status = "interrupted"
            run.interrupt_summary = str(interrupt_value)
            run.current_node = interrupted_node
            interrupt_data = {
                "summary": run.interrupt_summary,
                "message": "Pipeline paused — waiting for your review.",
                "node": interrupted_node,
            }
            if interrupted_node == "review_prep_plan":
                # Inject state-level vars into the plan JSON so the user
                # can edit them inline (keeps CLI and webUI in sync).
                try:
                    plan = json.loads(state.values.get("prep_plan", "{}"))
                except json.JSONDecodeError:
                    plan = {}
                plan["target_column"] = state.values.get("target_column", "")
                plan["target_mapping"] = state.values.get("target_mapping", {})
                plan["task_type"] = state.values.get("task_type", "")
                interrupt_data["prep_plan"] = json.dumps(plan)
            elif interrupted_node == "review_lmf_config":
                train_yaml_path = state.values.get("lmf_train_yaml", "")
                if train_yaml_path and Path(train_yaml_path).exists():
                    interrupt_data["lmf_train_yaml"] = Path(train_yaml_path).read_text()
            _emit(run, "interrupt", interrupt_data)

            # Park this thread until the user responds via POST /api/review
            run._review_event = threading.Event()
            run._review_response = None
            run._review_event.wait()               # blocks pipeline thread

            user_input = run._review_response or "approve"
            run.status = "running"
            _emit(run, "status", {
                "status": "running",
                "message": f"Resuming with: {user_input!r}",
            })

            # Resume — stream again for subsequent nodes
            _stream_graph(run, Command(resume=user_input))

    except Exception as exc:
        logger.exception("Pipeline error for run %s", run.run_id)
        run.status = "error"
        run.error = str(exc)
        _emit(run, "error", {"message": str(exc)})


def _format_results(final_state: dict) -> dict:
    """Extract evaluation results into a JSON-friendly dict."""
    eval_results = final_state.get("eval_results", {})
    parts = []
    for split, metrics in eval_results.items():
        if not isinstance(metrics, dict) or "error" in metrics:
            continue
        entry = {
            "split": split,
            "valid_predictions": metrics.get("valid_predictions"),
            "total_samples": metrics.get("total_samples"),
        }
        # Classification metrics
        if "accuracy" in metrics:
            entry["accuracy"] = metrics["accuracy"]
            entry["f1"] = metrics.get("f1")
            entry["macro_f1"] = metrics.get("macro_f1")
            entry["weighted_f1"] = metrics.get("weighted_f1")
        # Regression metrics
        if "mae" in metrics:
            entry["mae"] = metrics["mae"]
            entry["mse"] = metrics["mse"]
            entry["rmse"] = metrics["rmse"]
            entry["r2"] = metrics["r2"]
        parts.append(entry)
    return {
        "message": "Pipeline complete!",
        "run_dir": final_state.get("run_dir", ""),
        "results": parts,
    }


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

    return app


# ---------------------------------------------------------------------------
# Inference runner (runs in a background thread)
# ---------------------------------------------------------------------------

def _run_batch_inference(run: RunState, output_dir: str, run_dir: str,
                         csv_path: str, infer_output: str,
                         precision: str, flash_attn: str,
                         quantization_bit: int | None) -> None:
    """Execute batch inference, emitting SSE events."""
    from auto_llm_predictor.inference import run_batch_inference

    try:
        run.status = "running"
        _emit(run, "status", {"status": "running", "message": "Batch inference started"})

        result = run_batch_inference(
            output_dir=output_dir,
            run_dir=run_dir,
            csv_path=csv_path,
            infer_output=infer_output,
            precision=precision,
            flash_attn=flash_attn,
            quantization_bit=quantization_bit,
            log_callback=lambda msg: _emit(run, "log", {"message": msg}),
        )

        run.status = "completed"
        run.final_state = result
        _emit(run, "complete", {
            "message": "Batch inference complete!",
            "predictions_path": result.get("predictions_path", ""),
            "num_samples": result.get("num_samples", 0),
            "infer_output": result.get("infer_output", ""),
        })

    except Exception as exc:
        logger.exception("Batch inference error for run %s", run.run_id)
        run.status = "error"
        run.error = str(exc)
        _emit(run, "error", {"message": str(exc)})


def create_app() -> FastAPI:
    """Build the FastAPI application."""
    app = FastAPI(title="Auto LLM Predictor — Web UI")

    # Serve static files (HTML/CSS/JS)
    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # ── Landing page ──────────────────────────────────────────
    @app.get("/", response_class=HTMLResponse)
    async def index():
        index_path = static_dir / "index.html"
        return index_path.read_text()

    # ── Start a pipeline run ──────────────────────────────────
    @app.post("/api/run")
    async def start_run(
        csv_file: UploadFile = File(...),
        test_csv_file: UploadFile | None = File(None),
        model: str = Form(...),
        template: str = Form(""),
        target: str = Form(""),
        output: str = Form(""),
        test_ratio: float = Form(0.2),
        # Agent config
        agent_api_base: str = Form(""),
        agent_api_key: str = Form(""),
        agent_model: str = Form(""),
        coder_model: str = Form(""),
        agent_temperature: float = Form(0.2),
        # Hyperparameters
        lora_rank: int = Form(64),
        lora_alpha: int = Form(128),
        use_dora: bool = Form(False),
        cutoff_len: int = Form(4096),
        auto_cutoff: bool = Form(False),
        epochs: float = Form(3.0),
        learning_rate: str = Form("2.0e-5"),
        batch_size: int = Form(2),
        grad_accumulation: int = Form(8),
        logging_steps: int = Form(10),
        save_steps: int = Form(500),
        quantization_bit: int | None = Form(None),
        flash_attn: str = Form("auto"),
        precision: str = Form("bf16"),
        finetune_retries: int = Form(3),
        xai: bool = Form(False),
    ):
        from auto_llm_predictor.graph import build_graph
        from auto_llm_predictor.main import _build_training_config
        from types import SimpleNamespace

        # Resolve env defaults
        load_dotenv()
        env_endpoint = os.getenv("openAI_endpoint", "")
        api_base = agent_api_base or (f"http://{env_endpoint}/v1" if env_endpoint else "")
        api_key = agent_api_key or os.getenv("auth_key", "")
        a_model = agent_model or os.getenv("agent_LLM", "")
        c_model = coder_model or os.getenv("coder_LLM", "") or a_model

        if not api_base or not api_key or not a_model:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing agent LLM configuration. "
                         "Provide agent_api_base, agent_api_key, agent_model "
                         "or set openAI_endpoint / auth_key / agent_LLM in .env"},
            )

        # Validate local model path (if applicable)
        if is_local_model(model):
            errors = validate_local_model(model)
            if errors:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Local model directory is missing required files: "
                             + "; ".join(errors)},
                )
            model = str(Path(model).resolve())

        # Save uploaded CSV
        run_id = uuid.uuid4().hex[:12]
        csv_stem = Path(csv_file.filename or "dataset").stem
        output_dir = output if output else str(Path("output") / csv_stem)
        output_dir = normalize_path(str(Path(output_dir).resolve()))
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        csv_path = Path(output_dir) / (csv_file.filename or "upload.csv")
        with open(csv_path, "wb") as f:
            shutil.copyfileobj(csv_file.file, f)

        # Save optional test CSV
        test_csv_path = ""
        if test_csv_file and test_csv_file.filename:
            test_csv_dest = Path(output_dir) / test_csv_file.filename
            with open(test_csv_dest, "wb") as f:
                shutil.copyfileobj(test_csv_file.file, f)
            test_csv_path = normalize_path(str(test_csv_dest))

        # Build graph
        graph_app = build_graph(
            api_base=api_base,
            api_key=api_key,
            agent_model=a_model,
            coder_model=c_model,
            temperature=agent_temperature,
        )

        hp_args = SimpleNamespace(
            lora_rank=lora_rank, lora_alpha=lora_alpha, use_dora=use_dora,
            cutoff_len=cutoff_len, epochs=epochs, learning_rate=learning_rate,
            batch_size=batch_size, grad_accumulation=grad_accumulation,
            logging_steps=logging_steps, save_steps=save_steps,
            quantization_bit=quantization_bit, flash_attn=flash_attn,
            precision=precision, test_ratio=test_ratio,
            finetune_retries=finetune_retries, template=template,
        )

        initial_state = {
            "csv_path": str(csv_path),
            "test_csv_path": test_csv_path,
            "target_column": target,
            "base_model": model,
            "output_dir": output_dir,
            "start_from": "explore_data",
            "messages": [],
            "prep_attempts": 0,
            "finetune_attempts": 0,
            "auto_cutoff": auto_cutoff,
            "xai_enabled": xai,
            "training_config": _build_training_config(hp_args),
        }

        loop = asyncio.get_event_loop()
        run = RunState(
            run_id=run_id,
            app=graph_app,
            thread_config={},
            queue=asyncio.Queue(),
            loop=loop,
        )

        run.thread_config = {
            "configurable": {
                "thread_id": uuid.uuid4().hex,
                "log_callback": lambda msg: _emit(run, "log", {"message": msg}),
                "cancelled_check": lambda: run.cancelled,
            }
        }
        _prune_completed_runs()
        _runs[run_id] = run

        t = threading.Thread(
            target=_run_pipeline,
            args=(run, initial_state),
            daemon=True,
        )
        run.thread = t
        t.start()

        return {"run_id": run_id, "output_dir": output_dir}

    # ── SSE event stream ──────────────────────────────────────
    @app.get("/api/events/{run_id}")
    async def event_stream(run_id: str):
        run = _runs.get(run_id)
        if not run:
            return JSONResponse(status_code=404, content={"error": "Run not found"})

        async def generate():
            # Replay any events that already happened
            with run._events_lock:
                replay_events = list(run.events)
            for evt in replay_events:
                yield f"data: {json.dumps(evt)}\n\n"

            # Drain stale events from the queue (since we already replayed them from run.events)
            while not run.queue.empty():
                try:
                    run.queue.get_nowait()
                except getattr(asyncio, "QueueEmpty", Exception):
                    break

            # Stream new events
            while run.status not in ("completed", "error"):
                try:
                    evt = await asyncio.wait_for(run.queue.get(), timeout=30)
                    yield f"data: {json.dumps(evt)}\n\n"
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'event': 'heartbeat'})}\n\n"

            # Drain remaining
            while not run.queue.empty():
                evt = await run.queue.get()
                yield f"data: {json.dumps(evt)}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ── Submit review response ────────────────────────────────
    @app.post("/api/review/{run_id}")
    async def submit_review(run_id: str, response: str = Form("approve")):
        run = _runs.get(run_id)
        if not run:
            return JSONResponse(status_code=404, content={"error": "Run not found"})
        if run.status != "interrupted":
            return JSONResponse(
                status_code=400,
                content={"error": f"Run is not waiting for review (status: {run.status})"},
            )

        run._review_response = response.strip() or "approve"
        run._review_event.set()

        return {"status": "resumed", "response": run._review_response}

    # ── Poll-based status ─────────────────────────────────────
    @app.get("/api/status/{run_id}")
    async def get_status(run_id: str):
        run = _runs.get(run_id)
        if not run:
            return JSONResponse(status_code=404, content={"error": "Run not found"})
        return {
            "run_id": run.run_id,
            "status": run.status,
            "current_node": run.current_node,
            "error": run.error,
            "interrupt_summary": run.interrupt_summary if run.status == "interrupted" else "",
        }

    # ── Cancel a pipeline run ─────────────────────────────────
    @app.post("/api/cancel/{run_id}")
    async def cancel_run(run_id: str):
        run = _runs.get(run_id)
        if not run:
            return JSONResponse(status_code=404, content={"error": "Run not found"})
        if run.status in ("completed", "error"):
            return JSONResponse(status_code=400, content={"error": "Run already finished"})

        run.cancelled = True
        run.status = "error"
        run.error = "Cancelled by user"

        # If waiting for review, unblock it
        if not run._review_event.is_set():
            run._review_response = "cancel"
            run._review_event.set()

        _emit(run, "error", {"message": "Pipeline cancelled by user."})
        return {"status": "cancelled"}

    # ── Artifacts ─────────────────────────────────────────────
    @app.get("/api/artifacts/{run_id}")
    async def get_artifacts(run_id: str):
        run = _runs.get(run_id)
        if not run:
            return JSONResponse(status_code=404, content={"error": "Run not found"})
        try:
            state = run.app.get_state(run.thread_config).values
        except Exception:
            state = run.final_state
        
        # Keys mappings from state -> human readable labels
        artifact_keys = {
            "prep_script_path": "Data Prep Script (prepare_data.py)",
            "prepared_data_path": "Prepared Data (prepared_data.csv)",
            "balance_script_path": "Balance Script (balance_data.py)",
            "train_data_path": "Train Data (train.jsonl)",
            "test_data_path": "Test Data (test.jsonl)",
            "lmf_train_yaml": "Training Config (train.yaml)",
            "lmf_predict_train_yaml": "Predict Train Config (predict_train.yaml)",
            "lmf_predict_test_yaml": "Predict Test Config (predict_test.yaml)",
            "lmf_eval_yaml": "Evaluation Config (eval.yaml)",
            "xai_report_path": "XAI Report (xai_report.json)",
        }
        
        available = []
        for key, label in artifact_keys.items():
            path_str = state.get(key)
            # Ensure path exists before marking it as available
            if path_str and Path(path_str).is_file():
                available.append({"key": key, "label": label})
                
        return {"run_id": run_id, "artifacts": available}

    @app.get("/api/download/{run_id}")
    async def download_artifact(run_id: str, key: str):
        run = _runs.get(run_id)
        if not run:
            return JSONResponse(status_code=404, content={"error": "Run not found"})
        try:
            state = run.app.get_state(run.thread_config).values
        except Exception:
            state = run.final_state
            
        path_str = state.get(key)
        if not path_str or not Path(path_str).is_file():
            return JSONResponse(status_code=404, content={"error": "Artifact file not found or not yet available."})

        path = Path(path_str).resolve()
        # Ensure the file is within the run's output directory
        output_dir = state.get("output_dir", "")
        if output_dir:
            allowed_base = Path(output_dir).resolve()
            if not str(path).startswith(str(allowed_base)):
                return JSONResponse(status_code=403, content={"error": "Access denied: file is outside the run directory."})

        return FileResponse(
            path=path,
            filename=path.name,
            media_type="application/octet-stream"
        )

    # ── Active run discovery (for resume on page reload) ──────
    @app.get("/api/runs/active")
    async def get_active_run():
        """Return the most recent run that is still in progress."""
        for run in reversed(list(_runs.values())):
            if run.status in ("running", "interrupted", "pending"):
                return {
                    "run_id": run.run_id,
                    "status": run.status,
                    "current_node": run.current_node,
                }
        return {"run_id": None}

    # ── Inference: batch ──────────────────────────────────────
    @app.post("/api/infer/batch")
    async def start_batch_inference(
        csv_file: UploadFile = File(...),
        output_dir: str = Form(...),
        run_dir: str = Form(...),
        infer_output: str = Form(""),
        precision: str = Form("bf16"),
        flash_attn: str = Form("auto"),
        quantization_bit: int | None = Form(None),
    ):
        output_dir = normalize_path(str(Path(output_dir).resolve()))
        run_dir = normalize_path(str(Path(run_dir).resolve()))

        if not Path(output_dir).exists():
            return JSONResponse(status_code=400, content={"error": f"Output directory not found: {output_dir}"})
        if not Path(run_dir).exists():
            return JSONResponse(status_code=400, content={"error": f"Run directory not found: {run_dir}"})

        run_id = uuid.uuid4().hex[:12]

        # Save uploaded CSV to a temp location
        tmp_dir = Path(normalize_path(infer_output)) if infer_output else Path(run_dir) / "inference_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        csv_path = tmp_dir / (csv_file.filename or "infer_upload.csv")
        with open(csv_path, "wb") as f:
            shutil.copyfileobj(csv_file.file, f)

        loop = asyncio.get_event_loop()
        run = RunState(
            run_id=run_id,
            queue=asyncio.Queue(),
            loop=loop,
        )
        _prune_completed_runs()
        _runs[run_id] = run

        t = threading.Thread(
            target=_run_batch_inference,
            args=(run, output_dir, run_dir, str(csv_path), infer_output,
                  precision, flash_attn, quantization_bit),
            daemon=True,
        )
        run.thread = t
        t.start()

        return {"run_id": run_id}

    # ── Inference: single ─────────────────────────────────────
    @app.post("/api/infer/single")
    async def single_inference(
        output_dir: str = Form(...),
        run_dir: str = Form(...),
        features_json: str = Form(...),
        xai: bool = Form(False),
        precision: str = Form("bf16"),
        quantization_bit: int | None = Form(None),
    ):
        from auto_llm_predictor.inference import run_single_inference

        output_dir = normalize_path(str(Path(output_dir).resolve()))
        run_dir = normalize_path(str(Path(run_dir).resolve()))

        if not Path(output_dir).exists():
            return JSONResponse(status_code=400, content={"error": f"Output directory not found: {output_dir}"})
        if not Path(run_dir).exists():
            return JSONResponse(status_code=400, content={"error": f"Run directory not found: {run_dir}"})

        try:
            sample_features = json.loads(features_json)
        except json.JSONDecodeError as e:
            return JSONResponse(status_code=400, content={"error": f"Invalid features JSON: {e}"})

        try:
            result = run_single_inference(
                output_dir=output_dir,
                run_dir=run_dir,
                sample_features=sample_features,
                xai=xai,
                precision=precision,
                quantization_bit=quantization_bit,
            )
            return result
        except Exception as e:
            logger.exception("Single inference failed")
            return JSONResponse(status_code=500, content={"error": str(e)})

    # ── Inference: get feature names ──────────────────────────
    @app.get("/api/infer/features")
    async def get_features(output_dir: str):
        from auto_llm_predictor.inference import get_feature_names

        output_dir = normalize_path(str(Path(output_dir).resolve()))
        if not Path(output_dir).exists():
            return JSONResponse(status_code=400, content={"error": f"Output directory not found: {output_dir}"})

        try:
            features = get_feature_names(output_dir)
            return {"features": features}
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    # ── Inference: download predictions ───────────────────────
    @app.get("/api/infer/download/{run_id}")
    async def download_predictions(run_id: str):
        run = _runs.get(run_id)
        if not run:
            return JSONResponse(status_code=404, content={"error": "Run not found"})

        pred_path = run.final_state.get("predictions_path", "")
        if not pred_path or not Path(pred_path).is_file():
            return JSONResponse(status_code=404, content={"error": "Predictions not available yet."})

        return FileResponse(
            path=pred_path,
            filename="predictions.jsonl",
            media_type="application/octet-stream",
        )

    return app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Launch the web UI server."""
    import uvicorn

    parser = argparse.ArgumentParser(description="Auto LLM Predictor — Web UI")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 50)
    print("  Auto LLM Predictor — Web UI")
    print(f"  http://localhost:{args.port}")
    print("=" * 50)

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
