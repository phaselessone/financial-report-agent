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
        if abstained or supported:
            return state  # finalize

        # Verification retry: regenerate while attempts and LLM budget remain.
        if (
            int(state.get("generation_count", 0)) < config.max_generation_attempts
            and int(state.get("llm_call_count", 0)) < config.max_llm_calls
        ):
            state["missing_information"] = state.get("missing_information") or "unsupported_answer"
            return state  # back to synthesize
        state["termination_reason"] = "max_generation_attempts"
        return state

    return verify_answer
