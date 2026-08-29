"""Agent state schema (checklist v3.0 §P2 Agent State).

The state is a plain dictionary shaped by :data:`AgentState` (TypedDict).
All node functions mutate-and-return the dict, matching LangGraph's
single-serial-flow convention.
"""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    # input / routing
    query: str
    active_query: str
    domain_hint: str
    question_type: str
    answer_mode: str
    fact_subtype: str
    numeric_query: bool
    query_domain_buckets: list[str]
    query_domain_bucket: str

    # search progress
    rewritten_queries: list[str]
    retrieval_history: list[dict[str, Any]]
    evidence_pool: dict[str, dict[str, Any]]  # chunk_id -> retrieval row
    seen_chunk_ids: set[str]
    last_retrieval_result: dict[str, Any] | None
    last_new_ids: int

    # grading
    evidence_sufficient: bool
    missing_information: str
    no_improvement: bool
    failed_rewrite_rounds: int
    missing_reason_history: list[str]
    unsupported_retry: bool

    # multi-hop (P6)
    is_multi_hop: bool
    sub_questions: list[dict[str, Any]]  # [{"id": "q1", "query": "...", "required_fields": []}]
    sub_question_results: list[dict[str, Any]]
    structured_facts: list[dict[str, Any]]
    dependency_edges: list[dict[str, Any]]
    dependency_coverage: dict[str, Any]
    coverage_report: dict[str, Any]
    deterministic_conclusion_allowed: bool
    partial_answer: bool

    # unified single/multi-hop reasoning plan (Phase B)
    reasoning_plan: dict[str, Any] | None
    reasoning_step_results: list[dict[str, Any]]
    reasoning_step_artifacts: dict[str, dict[str, Any]]
    pending_reasoning_step: dict[str, Any] | None
    pending_step_observation: dict[str, Any] | None
    reasoning_plan_complete: bool
    reasoning_coverage: dict[str, Any]
    reasoning_conclusion: dict[str, Any] | None

    # controlled financial tools (M3)
    pending_tool_plan: dict[str, Any] | None
    pending_tool_observation: dict[str, Any] | None
    tool_calls: list[dict[str, Any]]
    tool_call_keys: set[str]
    tool_call_count: int
    tool_outcome: str
    consecutive_empty_tool_results: int
    trajectory_events: list[dict[str, Any]]
    calculations: dict[str, dict[str, Any]]
    calculation_intent: dict[str, Any] | None
    calculation_facts: dict[str, dict[str, Any]]

    # generation
    draft_answer: dict[str, Any] | None
    support_validation: dict[str, Any] | None

    # claim-level provenance (P7)
    claims: list[dict[str, Any]]  # [{"claim_id", "text", "claim_type", "evidence_ids", "calculation_id", "parent_claim_ids", "supported"}]
    claim_verifications: list[dict[str, Any]]
    claim_verification_summary: dict[str, Any]
    claim_retrieval_count: int
    claim_llm_judge_count: int

    # budgets / counters
    step_count: int
    retrieval_count: int
    rewrite_count: int
    generation_count: int
    llm_call_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    llm_calls_log: list[dict[str, Any]]

    # termination
    termination_reason: str
    final_answer: dict[str, Any] | None


def new_agent_state(*, query: str, domain_hint: str = "", question_type: str = "") -> AgentState:
    return AgentState(
        query=query,
        active_query=query,
        domain_hint=domain_hint,
        question_type=question_type,
        answer_mode="",
        fact_subtype="",
        numeric_query=False,
        rewritten_queries=[],
        retrieval_history=[],
        evidence_pool={},
        seen_chunk_ids=set(),
        evidence_sufficient=False,
        missing_information="",
        no_improvement=False,
        failed_rewrite_rounds=0,
        missing_reason_history=[],
        unsupported_retry=False,
        is_multi_hop=False,
        sub_questions=[],
        sub_question_results=[],
        structured_facts=[],
        dependency_edges=[],
        dependency_coverage={},
        coverage_report={},
        deterministic_conclusion_allowed=True,
        partial_answer=False,
        reasoning_plan=None,
        reasoning_step_results=[],
        reasoning_step_artifacts={},
        pending_reasoning_step=None,
        pending_step_observation=None,
        reasoning_plan_complete=False,
        reasoning_coverage={},
        reasoning_conclusion=None,
        pending_tool_plan=None,
        pending_tool_observation=None,
        tool_calls=[],
        tool_call_keys=set(),
        tool_call_count=0,
        tool_outcome="",
        consecutive_empty_tool_results=0,
        trajectory_events=[],
        calculations={},
        calculation_intent=None,
        calculation_facts={},
        draft_answer=None,
        support_validation=None,
        claims=[],
        claim_verifications=[],
        claim_verification_summary={},
        claim_retrieval_count=0,
        claim_llm_judge_count=0,
        step_count=0,
        retrieval_count=0,
        rewrite_count=0,
        generation_count=0,
        llm_call_count=0,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        llm_calls_log=[],
        termination_reason="",
        final_answer=None,
    )
