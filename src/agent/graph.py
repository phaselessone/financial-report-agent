"""Agent graph wiring (checklist v3.0 §P2).

Serial flow: analyze_query -> retrieve -> grade_evidence ->
{rewrite_query -> retrieve}* -> synthesize -> verify_answer -> finalize.

Routing and budget policy live in :mod:`src.agent.policies` and the node
guards; this module only wires LangGraph edges.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from src.agent.config import AgentConfig
from src.agent.nodes.analyze_query import make_analyze_query
from src.agent.nodes.finalize import make_finalize
from src.agent.nodes.grade_evidence import make_grade_evidence
from src.agent.nodes.retrieve import make_retrieve
from src.agent.nodes.rewrite_query import make_rewrite_query
from src.agent.nodes.synthesize import make_synthesize
from src.agent.nodes.verify_answer import make_verify_answer


def build_agent_graph(runtime, answerer, llm, config: AgentConfig):
    """Build the compiled agentic-RAG graph."""

    def route_after_grade(state: dict[str, Any]) -> str:
        if state.get("termination_reason"):
            return "finalize"
        if state.get("evidence_sufficient"):
            return "synthesize"
        if state.get("no_improvement"):
            return "synthesize"  # stop searching; try generating with the pooled evidence
        if (
            int(state.get("rewrite_count", 0)) < config.max_query_rewrites
            and int(state.get("llm_call_count", 0)) < config.max_llm_calls
        ):
            return "rewrite_query"
        return "synthesize"

    def route_after_synthesize(state: dict[str, Any]) -> str:
        if state.get("termination_reason"):
            return "finalize"
        return "verify_answer"

    def route_after_verify(state: dict[str, Any]) -> str:
        if state.get("termination_reason"):
            return "finalize"
        draft = state.get("draft_answer") or {}
        if draft.get("abstained") or (draft.get("support_validation") or {}).get("supported", False):
            return "finalize"
        return "synthesize"  # verification retry

    graph = StateGraph(dict)
    graph.add_node("analyze_query", make_analyze_query(config))
    graph.add_node("retrieve", make_retrieve(runtime, config))
    graph.add_node("grade_evidence", make_grade_evidence(config))
    graph.add_node("rewrite_query", make_rewrite_query(llm, config))
    graph.add_node("synthesize", make_synthesize(answerer, config))
    graph.add_node("verify_answer", make_verify_answer(config))
    graph.add_node("finalize", make_finalize())

    graph.set_entry_point("analyze_query")
    graph.add_edge("analyze_query", "retrieve")
    graph.add_edge("retrieve", "grade_evidence")
    graph.add_conditional_edges(
        "grade_evidence",
        route_after_grade,
        {"synthesize": "synthesize", "rewrite_query": "rewrite_query", "finalize": "finalize"},
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_conditional_edges(
        "synthesize",
        route_after_synthesize,
        {"verify_answer": "verify_answer", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "verify_answer",
        route_after_verify,
        {"synthesize": "synthesize", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)
    return graph.compile()


def run_agentic_rag(*, runtime, answerer, llm, config: AgentConfig, query: str, domain_hint: str = "", question_type: str = "") -> dict[str, Any]:
    """One-shot helper: run the graph and return the final state."""
    from src.agent.state import new_agent_state

    graph = build_agent_graph(runtime, answerer, llm, config)
    return graph.invoke(new_agent_state(query=query, domain_hint=domain_hint, question_type=question_type))
