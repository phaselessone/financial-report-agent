"""verify_answer node: reuse the existing validate_answer_support verdict.

Verification retry: an unsupported (non-abstained) draft may regenerate up to
``max_generation_attempts``; abstained drafts and supported drafts finalize.
"""

from __future__ import annotations

from typing import Any

from src.agent.config import AgentConfig
from src.agent.policies import begin_node


def make_verify_answer(config: AgentConfig):
    def verify_answer(state: dict[str, Any]) -> dict[str, Any]:
        if not begin_node(state, config):
            return state

        draft = state.get("draft_answer")
        if draft is None:
            state["termination_reason"] = "no_draft"
            return state

        support_validation = draft.get("support_validation") or {}
        supported = bool(support_validation.get("supported", False))
        abstained = bool(draft.get("abstained", False))
        if abstained:
            # Retry policy (gate remediation): abstaining despite having seen
            # evidence is worth one rewrite + re-retrieval attempt; without any
            # evidence the abstention is final (P2 semantics preserved).
            has_evidence = bool(state.get("evidence_pool"))
            if (
                has_evidence
                and int(state.get("rewrite_count", 0)) < config.max_query_rewrites
                and int(state.get("retrieval_count", 0)) < config.max_retrieval_rounds
                and int(state.get("llm_call_count", 0)) < config.max_llm_calls
            ):
                state["missing_information"] = state.get("missing_information") or "abstained_despite_evidence"
                state["unsupported_retry"] = True
                return state  # rewrite_query
            state["unsupported_retry"] = False
            return state  # finalize
        low_confidence = str(draft.get("confidence_label", "high") or "high").strip().lower() == "low"
        if supported and not low_confidence:
            state["unsupported_retry"] = False
            return state  # finalize

        # Retry policy (gate remediation): an unsupported OR low-confidence draft
        # first tries to rewrite + re-retrieve fresher evidence; only when the
        # retrieval budget is exhausted does it fall back to regenerating on the
        # same evidence.
        if (
            int(state.get("rewrite_count", 0)) < config.max_query_rewrites
            and int(state.get("retrieval_count", 0)) < config.max_retrieval_rounds
            and int(state.get("llm_call_count", 0)) < config.max_llm_calls
        ):
            state["missing_information"] = state.get("missing_information") or (
                "unsupported_answer" if not supported else "low_confidence_answer"
            )
            state["unsupported_retry"] = True
            return state  # rewrite_query
        state["unsupported_retry"] = False
        if (
            int(state.get("generation_count", 0)) < config.max_generation_attempts
            and int(state.get("llm_call_count", 0)) < config.max_llm_calls
        ):
            state["missing_information"] = state.get("missing_information") or (
                "unsupported_answer" if not supported else "low_confidence_answer"
            )
            return state  # back to synthesize
        state["termination_reason"] = "max_generation_attempts"
        return state

    return verify_answer
