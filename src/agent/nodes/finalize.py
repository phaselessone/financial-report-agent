"""finalize node: emit the final answer (draft, or deterministic abstain)."""

from __future__ import annotations

from typing import Any

from src.generation.routing import FALLBACK_ANSWER


def _abstain_row(state: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "query": state.get("query", ""),
        "question_type": state.get("question_type", ""),
        "answer_mode": state.get("answer_mode", ""),
        "final_answer": FALLBACK_ANSWER,
        "evidence_summary": "",
        "abstained": True,
        "abstain_reason": reason or "no_evidence",
        "used_evidence_ids": [],
        "citations": [],
        "support_validation": {},
        "selected_doc_ids": [],
    }


def make_finalize():
    def finalize(state: dict[str, Any]) -> dict[str, Any]:
        state["step_count"] = int(state.get("step_count", 0)) + 1
        draft = state.get("draft_answer")
        if draft is not None:
            final = draft
        else:
            final = _abstain_row(state, state.get("termination_reason") or "no_evidence")
        if not state.get("termination_reason"):
            state["termination_reason"] = "completed"
        state["final_answer"] = final
        return state

    return finalize
