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
from auto_llm_predictor.nodes.config import generate_lmf_config
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
from auto_llm_predictor.nodes.split import split_data
from auto_llm_predictor.nodes.verify import verify_prepared_data
from auto_llm_predictor.state import PipelineState


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
) -> StateGraph:
    """Construct and compile the LangGraph pipeline.

    Parameters
    ----------
    api_base : str
        OpenAI-compatible API endpoint (shared by both LLMs).
    api_key : str
        API key / token for the LLM endpoint.
    agent_model : str
        Model ID for reasoning, planning, and data exploration.
    coder_model : str
        Model ID for code generation. Falls back to *agent_model* if empty.
    temperature : float
        Sampling temperature.

    Returns
    -------
    Compiled LangGraph application (with MemorySaver checkpointer for
    human-in-the-loop support via ``interrupt()``).
    """
    agent_llm = ChatOpenAI(
        base_url=api_base,
        api_key=api_key,
        model=agent_model,
        temperature=temperature,
        max_tokens=8192,
    )

    coder_llm = ChatOpenAI(
        base_url=api_base,
        api_key=api_key,
        model=coder_model or agent_model,
        temperature=temperature,
        max_tokens=8192,
    )

    graph = StateGraph(PipelineState)

    # ── Dispatcher for --start-from ────────────────────────────
    def route_start(state: PipelineState) -> dict:
        """Pass-through node used as entry point for conditional routing."""
        return {}

    def check_start_from(state: PipelineState) -> str:
        """Route to the requested starting step."""
        target = state.get("start_from", "explore_data")
        _VALID = {
            "explore_data", "review_prep", "split", "config",
        }
        if target not in _VALID:
            return "explore_data"
        if target == "review_prep":
            return "review_prep_data"
        if target == "split":
            return "split_data"
        if target == "config":
            return "generate_lmf_config"
        return "explore_data"

    graph.add_node("route_start", route_start)

    # ── Add nodes ──────────────────────────────────────────────
    # Agent LLM: reasoning, planning, data exploration
    graph.add_node("explore_data", _bind_llm(explore_data, agent_llm))
    graph.add_node("select_features", select_features)
    graph.add_node("plan_preparation", _bind_llm(plan_preparation, agent_llm))
    graph.add_node("review_prep_plan", review_prep_plan)

    # Coder LLM: code generation
    graph.add_node("write_prep_code", _bind_llm(write_prep_code, coder_llm))
    graph.add_node("execute_prep_code", execute_prep_code)
    graph.add_node("verify_prepared_data", _bind_llm(verify_prepared_data, agent_llm))
    graph.add_node("review_prep_data", review_prep_data)
    graph.add_node("write_balance_code", _bind_llm(write_balance_code, coder_llm))
    graph.add_node("execute_balance_code", execute_balance_code)
    graph.add_node("review_balanced_data", review_balanced_data)

    # No LLM needed
    graph.add_node("split_data", split_data)
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
            "split_data": "split_data",
            "generate_lmf_config": "generate_lmf_config",
        },
    )


    # Conditional: high-dimensional data gets ensemble feature selection
    graph.add_conditional_edges(
        "explore_data",
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

    # Conditional: retry code generation on failure (up to 3 times)
    graph.add_conditional_edges(
        "execute_prep_code",
        check_prep_result,
        {
            "verify_prepared_data": "verify_prepared_data",
            "write_prep_code": "write_prep_code",
        },
    )
    graph.add_edge("verify_prepared_data", "review_prep_data")

    # Human-in-the-loop: review prepared data
    # 3-way routing: approve → split, balance, or revise
    graph.add_conditional_edges(
        "review_prep_data",
        route_after_review,
        {
            "split_data": "split_data",
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
            "split_data": "split_data",
            "write_balance_code": "write_balance_code",
        },
    )

    # split → config generation
    graph.add_edge("split_data", "generate_lmf_config")

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

