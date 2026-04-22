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

"""LangGraph definition — wires all nodes into the pipeline graph."""

from __future__ import annotations

import functools
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from auto_llm_predictor.nodes.balance import (
    check_balance_result,
    execute_balance_code,
    write_balance_code,
)
from auto_llm_predictor.nodes.codegen import write_prep_code
from auto_llm_predictor.nodes.debug import debug_prep_failure, route_after_debug
from auto_llm_predictor.nodes.config import generate_lmf_config
from auto_llm_predictor.nodes.cutoff import determine_cutoff_len
from auto_llm_predictor.nodes.evaluate import run_evaluation
from auto_llm_predictor.nodes.execute import check_prep_result, execute_prep_code

from auto_llm_predictor.nodes.explore import explore_data
from auto_llm_predictor.nodes.feature_selection import check_feature_complexity, select_features
from auto_llm_predictor.nodes.finetune import run_finetuning
from auto_llm_predictor.nodes.plan import plan_preparation
from auto_llm_predictor.nodes.predict import run_prediction
from auto_llm_predictor.nodes.review import (
    review_balanced_data,
    review_lmf_config,
    review_prep_data,
    review_prep_plan,
    route_after_balance_review,
    route_after_config_review,
    route_after_plan_review,
    route_after_review,
)
from auto_llm_predictor.nodes.data_registration import register_dataset
from auto_llm_predictor.nodes.split_input import split_input_csv
from auto_llm_predictor.nodes.verify import verify_prepared_data
from auto_llm_predictor.state import PipelineState


def _create_llm(
    *,
    provider: str,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int = 8192,
) -> Any:
    """Instantiate the correct LangChain chat model for the chosen provider."""
    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError(
                "langchain-ollama is required for Ollama support. "
                "Install it with: pip install auto-llm-predictor[ollama]"
            )
        return ChatOllama(
            base_url=base_url,
            model=model,
            temperature=temperature,
            num_predict=max_tokens,
        )
    return ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _bind_llm(node_fn, llm):
    """Wrap a node function so it receives the LLM as a keyword argument."""
    @functools.wraps(node_fn)
    def wrapper(state: PipelineState) -> dict:
        return node_fn(state, llm=llm)
    return wrapper


def build_graph(
    *,
    api_base: str = "",
    api_key: str = "",
    agent_model: str = "",
    coder_model: str = "",
    temperature: float = 0.2,
    verbose: bool = False,
    llm_provider: str = "openai",
) -> StateGraph:
    """Construct and compile the LangGraph pipeline.

    Parameters
    ----------
    api_base : str
        OpenAI-compatible API endpoint (shared by both LLMs).
        For Ollama, use ``http://host:port`` (no ``/v1`` suffix).
    api_key : str
        API key / token for the LLM endpoint.  Ignored for Ollama.
    agent_model : str
        Model ID for reasoning, planning, and data exploration.
    coder_model : str
        Model ID for code generation. Falls back to *agent_model* if empty.
    temperature : float
        Sampling temperature.
    llm_provider : str
        LLM backend: ``"openai"`` (default) or ``"ollama"``.

    Returns
    -------
    Compiled LangGraph application (with MemorySaver checkpointer for
    human-in-the-loop support via ``interrupt()``).
    """
    agent_llm = _create_llm(
        provider=llm_provider,
        base_url=api_base,
        api_key=api_key,
        model=agent_model,
        temperature=temperature,
    )

    coder_llm = _create_llm(
        provider=llm_provider,
        base_url=api_base,
        api_key=api_key,
        model=coder_model or agent_model,
        temperature=temperature,
    )

    if verbose:
        from auto_llm_predictor.verbose import VerboseLLM

        agent_llm = VerboseLLM(agent_llm, label="AGENT")
        coder_llm = VerboseLLM(coder_llm, label="CODER")

    graph = StateGraph(PipelineState)

    # ── Dispatcher for --start-from ────────────────────────────
    def route_start(state: PipelineState) -> dict:
        """Pass-through node used as entry point for conditional routing."""
        return {}

    def check_start_from(state: PipelineState) -> str:
        """Route to the requested starting step."""
        target = state.get("start_from", "explore_data")
        routes = {
            "explore_data": "explore_data",
            "review_prep": "review_prep_data",
            "register": "data_registration",
            "config": "generate_lmf_config",
        }
        return routes.get(target, "explore_data")

    graph.add_node("route_start", route_start)

    # ── Add nodes ──────────────────────────────────────────────
    # Agent LLM: reasoning, planning, data exploration
    graph.add_node("explore_data", _bind_llm(explore_data, agent_llm))
    graph.add_node("split_input_csv", split_input_csv)
    graph.add_node("select_features", select_features)
    graph.add_node("plan_preparation", _bind_llm(plan_preparation, agent_llm))
    graph.add_node("review_prep_plan", review_prep_plan)

    # Coder LLM: code generation
    graph.add_node("write_prep_code", _bind_llm(write_prep_code, coder_llm))
    graph.add_node("execute_prep_code", execute_prep_code)
    graph.add_node("debug_prep_failure", _bind_llm(debug_prep_failure, agent_llm))
    graph.add_node("verify_prepared_data", _bind_llm(verify_prepared_data, agent_llm))
    graph.add_node("review_prep_data", review_prep_data)
    graph.add_node("write_balance_code", _bind_llm(write_balance_code, coder_llm))
    graph.add_node("execute_balance_code", execute_balance_code)
    graph.add_node("review_balanced_data", review_balanced_data)

    # No LLM needed
    graph.add_node("data_registration", register_dataset)
    graph.add_node("determine_cutoff_len", determine_cutoff_len)
    graph.add_node("generate_lmf_config", generate_lmf_config)
    graph.add_node("review_lmf_config", review_lmf_config)
    graph.add_node("run_finetuning", run_finetuning)
    graph.add_node("run_prediction", run_prediction)
    graph.add_node("run_evaluation", run_evaluation)


    # ── Wire edges ─────────────────────────────────────────────
    graph.set_entry_point("route_start")

    # Entry dispatcher: jump to the requested step
    graph.add_conditional_edges(
        "route_start",
        check_start_from,
        {
            "explore_data": "explore_data",
            "review_prep_data": "review_prep_data",
            "data_registration": "data_registration",
            "generate_lmf_config": "generate_lmf_config",
        },
    )

    # explore_data → split_input_csv → (feature selection or plan)
    graph.add_edge("explore_data", "split_input_csv")

    # Conditional: high-dimensional data gets ensemble feature selection.
    # Runs on training-only data because split_input_csv has updated csv_path.
    graph.add_conditional_edges(
        "split_input_csv",
        check_feature_complexity,
        {
            "select_features": "select_features",
            "plan_preparation": "plan_preparation",
        },
    )
    graph.add_edge("select_features", "plan_preparation")

    graph.add_edge("plan_preparation", "review_prep_plan")
    graph.add_conditional_edges(
        "review_prep_plan",
        route_after_plan_review,
        {
            "write_prep_code": "write_prep_code",
            "plan_preparation": "plan_preparation",
        },
    )
    graph.add_edge("write_prep_code", "execute_prep_code")

    # Conditional: on failure, debug then retry (up to 3 times)
    graph.add_conditional_edges(
        "execute_prep_code",
        check_prep_result,
        {
            "verify_prepared_data": "verify_prepared_data",
            "debug_prep_failure": "debug_prep_failure",
        },
    )
    graph.add_conditional_edges(
        "debug_prep_failure",
        route_after_debug,
        {
            "write_prep_code": "write_prep_code",
            "verify_prepared_data": "verify_prepared_data",
        },
    )
    graph.add_edge("verify_prepared_data", "review_prep_data")

    # Human-in-the-loop: review prepared data
    # 3-way routing: approve → register, balance, or revise
    graph.add_conditional_edges(
        "review_prep_data",
        route_after_review,
        {
            "data_registration": "data_registration",
            "write_balance_code": "write_balance_code",
            "plan_preparation": "plan_preparation",
        },
    )

    # Balance step: generate → execute → retry on failure
    graph.add_edge("write_balance_code", "execute_balance_code")
    graph.add_conditional_edges(
        "execute_balance_code",
        check_balance_result,
        {
            "review_balanced_data": "review_balanced_data",
            "write_balance_code": "write_balance_code",
        },
    )

    # Human-in-the-loop: review balanced data
    graph.add_conditional_edges(
        "review_balanced_data",
        route_after_balance_review,
        {
            "data_registration": "data_registration",
            "write_balance_code": "write_balance_code",
        },
    )

    # registration → cutoff detection → config generation
    graph.add_edge("data_registration", "determine_cutoff_len")
    graph.add_edge("determine_cutoff_len", "generate_lmf_config")

    graph.add_edge("generate_lmf_config", "review_lmf_config")

    # Human-in-the-loop: review configs before fine-tuning
    graph.add_conditional_edges(
        "review_lmf_config",
        route_after_config_review,
        {
            "run_finetuning": "run_finetuning",
            "generate_lmf_config": "generate_lmf_config",
        },
    )

    graph.add_edge("run_finetuning", "run_prediction")
    graph.add_edge("run_prediction", "run_evaluation")
    graph.add_edge("run_evaluation", END)

    # MemorySaver is required for interrupt() to work
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)

