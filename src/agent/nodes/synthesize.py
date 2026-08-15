"""synthesize node: generate the draft answer via the existing AnswerService."""

from __future__ import annotations

from typing import Any

from src.agent.config import AgentConfig
from src.agent.policies import begin_node, record_llm_response, token_budget_exceeded


def make_synthesize(answerer, config: AgentConfig):
    def synthesize(state: dict[str, Any]) -> dict[str, Any]:
        if not begin_node(state, config):
            return state
        if int(state.get("generation_count", 0)) >= config.max_generation_attempts:
            state["termination_reason"] = "max_generation_attempts"
            return state
        if int(state.get("llm_call_count", 0)) >= config.max_llm_calls:
            state["termination_reason"] = "max_llm_calls"
            return state
        if token_budget_exceeded(state, config):
            return state

        retrieval_result = state.get("last_retrieval_result") or {"rerank_rows": [], "hybrid_rows": []}
        calls_before = len(getattr(answerer, "llm_calls", []))

        draft = answerer.answer(
            query=state["active_query"],
            question_type=state.get("question_type", ""),
            retrieval_result=retrieval_result,
            query_domain_hint=state.get("domain_hint", ""),
        )
        state["draft_answer"] = draft
        state["support_validation"] = draft.get("support_validation") or {}
        state["generation_count"] = int(state.get("generation_count", 0)) + 1

        # Account for every LLM call the AnswerService made under the hood.
        new_calls = list(getattr(answerer, "llm_calls", [])[calls_before:])
        for response in new_calls:
            record_llm_response(state, response, node="synthesize")
        return state

    return synthesize
