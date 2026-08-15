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

    # generation
    draft_answer: dict[str, Any] | None
    support_validation: dict[str, Any] | None

    # budgets / counters
    step_count: int
    retrieval_count: int
    rewrite_count: int
    generation_count: int
    llm_call_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

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
        draft_answer=None,
        support_validation=None,
        step_count=0,
        retrieval_count=0,
        rewrite_count=0,
        generation_count=0,
        llm_call_count=0,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        termination_reason="",
        final_answer=None,
    )
