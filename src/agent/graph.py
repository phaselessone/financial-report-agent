"""Agent graph wiring for unified single-hop and multi-hop reasoning.

Every query receives a ReasoningPlan. Multi-hop/calculation plans run through
plan-next/execute/observe/coverage. One-step plans use the same executor and
may enter the bounded evidence-grade/rewrite recovery loop only after coverage.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from src.agent.config import AgentConfig
from src.agent.nodes.analyze_query import make_analyze_query
from src.agent.nodes.build_reasoning_plan import make_build_reasoning_plan
from src.agent.nodes.dependency_gate import make_dependency_gate
from src.agent.nodes.execute_step import make_execute_step
from src.agent.nodes.extract_claims import make_extract_claims
from src.agent.nodes.finalize import make_finalize
from src.agent.nodes.grade_evidence import make_grade_evidence
from src.agent.nodes.observe_step_result import make_observe_step_result
from src.agent.nodes.plan_next_step import make_plan_next_step
from src.agent.nodes.rewrite_query import make_rewrite_query
from src.agent.nodes.synthesize import make_synthesize
from src.agent.nodes.verify_answer import make_verify_answer
from src.agent.semantic_policy import CalibratedSemanticScorer, SemanticActivationError
from src.agent.trajectory import trace_graph_node


def build_agent_graph(
    runtime,
    answerer,
    llm,
    config: AgentConfig,
    *,
    fact_store=None,
    company_aliases=None,
    semantic_scorer=None,
    llm_judge=None,
):
    """Build the compiled agentic-RAG graph."""

    capabilities = config.resolved_execution_capabilities()
    effective_fact_store = fact_store if capabilities.structured_facts else None

    if semantic_scorer is not None and not isinstance(
        semantic_scorer, CalibratedSemanticScorer
    ):
        raise SemanticActivationError(
            "semantic scorer must be activated from a reviewed calibration report"
        )

    def route_after_analyze(state: dict[str, Any]) -> str:
        if state.get("termination_reason"):
            return "finalize"
        if state.get("is_multi_hop") and not capabilities.multi_hop_reasoning:
            state["termination_reason"] = "multi_hop_reasoning_disabled"
            return "finalize"
        return "build_reasoning_plan"

    def route_after_build_plan(state: dict[str, Any]) -> str:
        if state.get("termination_reason"):
            return "finalize"
        return "plan_next_step"

    def route_after_plan_next_step(state: dict[str, Any]) -> str:
        if state.get("termination_reason"):
            return "finalize"
        return "dependency_gate" if state.get("reasoning_plan_complete") else "execute_step"

    def route_after_dependency_gate(state: dict[str, Any]) -> str:
        if state.get("termination_reason"):
            return "finalize"
        steps = list((state.get("reasoning_plan") or {}).get("steps") or [])
        if len(steps) == 1 and str((steps[0] or {}).get("kind") or "") in {"LOOKUP", "SEARCH"}:
            step_id = str((steps[0] or {}).get("step_id") or "")
            artifact = (state.get("reasoning_step_artifacts") or {}).get(step_id) or {}
            results = list(state.get("reasoning_step_results") or [])
            if (
                artifact.get("source") == "structured"
                and results
                and str((results[-1] or {}).get("status") or "") == "SUCCESS"
            ):
                return "synthesize"
            return "grade_evidence"
        return "synthesize"

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
        # Atomic extraction plus deterministic provenance verification is an
        # integrity invariant for every graph-backed profile.  The ablation's
        # claim_verification capability controls enhanced verification and its
        # recovery loop, not whether a non-abstained answer may bypass claims.
        return "extract_claims"

    def route_after_verify(state: dict[str, Any]) -> str:
        if state.get("termination_reason"):
            return "finalize"
        if not capabilities.claim_verification:
            return "finalize"
        if state.get("is_multi_hop"):
            # v1 multi-hop skips the rewrite/verify retry loop; the synthesized
            # draft (or deterministic abstain) is final.
            return "finalize"
        if state.get("unsupported_retry"):
            return "rewrite_query"  # verification retry: rewrite + re-retrieve
        draft = state.get("draft_answer") or {}
        if draft.get("abstained") or (draft.get("support_validation") or {}).get("supported", False):
            return "finalize"
        return "synthesize"  # legacy regenerate-on-same-evidence retry

    graph = StateGraph(dict)
    add_node = lambda name, node: graph.add_node(name, trace_graph_node(name, node))
    add_node("analyze_query", make_analyze_query(config))
    add_node(
        "build_reasoning_plan",
        make_build_reasoning_plan(llm, effective_fact_store, company_aliases, config),
    )
    add_node("dependency_gate", make_dependency_gate())
    add_node("plan_next_step", make_plan_next_step())
    add_node(
        "execute_step",
        make_execute_step(runtime, effective_fact_store, company_aliases, config),
    )
    add_node("observe_step_result", make_observe_step_result())
    add_node("grade_evidence", make_grade_evidence(config))
    add_node("rewrite_query", make_rewrite_query(llm, config))
    add_node("synthesize", make_synthesize(answerer, config))
    add_node("extract_claims", make_extract_claims(llm, config))
    add_node(
        "verify_answer",
        make_verify_answer(
            config,
            runtime=runtime,
            company_aliases=company_aliases or {},
            semantic_scorer=semantic_scorer,
            llm_judge=llm_judge,
        ),
    )
    add_node("finalize", make_finalize(config))

    graph.set_entry_point("analyze_query")
    graph.add_conditional_edges(
        "analyze_query",
        route_after_analyze,
        {
            "build_reasoning_plan": "build_reasoning_plan",
            "finalize": "finalize",
        },
    )
    graph.add_conditional_edges(
        "build_reasoning_plan",
        route_after_build_plan,
        {
            "plan_next_step": "plan_next_step",
            "finalize": "finalize",
        },
    )
    graph.add_conditional_edges(
        "plan_next_step",
        route_after_plan_next_step,
        {
            "execute_step": "execute_step",
            "dependency_gate": "dependency_gate",
            "finalize": "finalize",
        },
    )
    graph.add_edge("execute_step", "observe_step_result")
    graph.add_edge("observe_step_result", "plan_next_step")
    graph.add_conditional_edges(
        "dependency_gate",
        route_after_dependency_gate,
        {
            "grade_evidence": "grade_evidence",
            "synthesize": "synthesize",
            "finalize": "finalize",
        },
    )
    graph.add_conditional_edges(
        "grade_evidence",
        route_after_grade,
        {"synthesize": "synthesize", "rewrite_query": "rewrite_query", "finalize": "finalize"},
    )
    graph.add_edge("rewrite_query", "build_reasoning_plan")
    graph.add_conditional_edges(
        "synthesize",
        route_after_synthesize,
        {"extract_claims": "extract_claims", "finalize": "finalize"},
    )
    graph.add_edge("extract_claims", "verify_answer")
    graph.add_conditional_edges(
        "verify_answer",
        route_after_verify,
        {"synthesize": "synthesize", "rewrite_query": "rewrite_query", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)
    return graph.compile()


def run_agentic_rag(
    *,
    runtime,
    answerer,
    llm,
    config: AgentConfig,
    query: str,
    domain_hint: str = "",
    question_type: str = "",
    fact_store=None,
    company_aliases=None,
    semantic_scorer=None,
    llm_judge=None,
) -> dict[str, Any]:
    """One-shot helper: run the graph and return the final state."""
    from src.agent.state import new_agent_state

    graph = build_agent_graph(
        runtime,
        answerer,
        llm,
        config,
        fact_store=fact_store,
        company_aliases=company_aliases,
        semantic_scorer=semantic_scorer,
        llm_judge=llm_judge,
    )
    return graph.invoke(new_agent_state(query=query, domain_hint=domain_hint, question_type=question_type))
