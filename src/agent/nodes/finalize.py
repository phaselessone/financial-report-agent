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
        reason = state.get("termination_reason") or ""
        if draft is not None:
            supported = bool((draft.get("support_validation") or {}).get("supported", False))
            if draft.get("abstained") or supported or reason in ("", "completed"):
                final = draft
            else:
                # Terminated with a stale unsupported draft (e.g. the verification
                # retry refetched evidence that still graded insufficient): abstain.
                final = _abstain_row(state, reason or "no_evidence")
        else:
            final = _abstain_row(state, reason or "no_evidence")
        if not state.get("termination_reason"):
            state["termination_reason"] = "completed"
        state["final_answer"] = final
        return state

    return finalize
