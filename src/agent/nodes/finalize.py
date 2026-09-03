"""finalize node: emit the final answer (draft, or deterministic abstain)."""

from __future__ import annotations

from typing import Any

from src.agent.config import AgentConfig
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
        "claims": list(state.get("claims") or []),
        "calculations": dict(state.get("calculations") or {}),
    }


def _normalized_abstained_draft(draft: dict[str, Any]) -> dict[str, Any]:
    """Keep the draft for audit while publishing a canonical abstention."""
    final = dict(draft)
    final["abstained"] = True
    final["final_answer"] = FALLBACK_ANSWER
    final["answer"] = FALLBACK_ANSWER
    final["partial_answer"] = False
    final["used_evidence_ids"] = []
    final["citations"] = []
    final["selected_doc_ids"] = []
    final["support_validation"] = {
        **(final.get("support_validation") or {}),
        "supported": False,
        "claim_gate": "abstained",
    }
    return final


def _claim_is_core(claim: dict[str, Any]) -> bool:
    value = claim.get("is_core")
    # Unknown core-ness fails closed; only an explicit false makes a claim
    # optional for finalization.
    return value if isinstance(value, bool) else True


def _citation_index(draft: dict[str, Any], state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for citation in draft.get("citations") or []:
        if isinstance(citation, dict):
            for key in (citation.get("evidence_id"), citation.get("chunk_id")):
                if key:
                    index[str(key)] = dict(citation)
    for row in (state.get("evidence_pool") or {}).values():
        if not isinstance(row, dict):
            continue
        for key in (row.get("evidence_id"), row.get("chunk_id")):
            if key:
                index.setdefault(str(key), dict(row))
    return index


def _apply_claim_gate(draft: dict[str, Any], state: dict[str, Any]) -> dict[str, Any] | None:
    claims = [claim for claim in (state.get("claims") or []) if isinstance(claim, dict)]
    if not claims:
        return draft
    statuses = [str((claim.get("verification") or {}).get("status") or "INSUFFICIENT") for claim in claims]
    entailed = [claim for claim in claims if str((claim.get("verification") or {}).get("status")) == "ENTAILED"]
    core_entailed = [claim for claim in claims if _claim_is_core(claim) and str((claim.get("verification") or {}).get("status")) == "ENTAILED"]
    core_total = sum(1 for claim in claims if _claim_is_core(claim))
    if core_total and not core_entailed:
        state["termination_reason"] = state.get("termination_reason") or "claim_verification_failed"
        return None
    if not entailed:
        state["termination_reason"] = state.get("termination_reason") or "claim_verification_failed"
        return None

    final = dict(draft)
    dependency_limited = bool(state.get("is_multi_hop")) and not bool(
        state.get("deterministic_conclusion_allowed", True)
    )
    all_supported = (
        not dependency_limited
        and len(entailed) == len(claims)
        and all(status == "ENTAILED" for status in statuses)
    )
    # The strict output surface is reconstructed from verified atomic claims
    # even when every extracted claim is entailed.  Keeping the original draft
    # would allow unextracted model text to bypass claim verification.
    original_answer = str(draft.get("final_answer") or draft.get("answer") or "")
    normalized_original = "".join(original_answer.split())
    presented_claims = [
        claim
        for claim in entailed
        if _claim_is_core(claim)
        or "".join(str(claim.get("text") or "").split()) in normalized_original
    ]
    if not presented_claims:
        presented_claims = entailed
    final["final_answer"] = " ".join(
        str(claim.get("text") or "").strip()
        for claim in presented_claims
        if str(claim.get("text") or "").strip()
    )
    final["answer"] = final["final_answer"]
    final["partial_answer"] = not all_supported
    if not all_supported:
        final["partial_answer"] = True
        final["abstain_reason"] = "partial_dependency_coverage" if dependency_limited else "partial_claim_support"
    else:
        final["abstain_reason"] = None

    citation_index = _citation_index(draft, state)
    evidence_ids: list[str] = []
    citations: list[dict[str, Any]] = []
    selected_docs: list[str] = []
    for claim in entailed:
        for evidence_id in claim.get("evidence_ids") or []:
            key = str(evidence_id)
            if key in evidence_ids:
                continue
            evidence_ids.append(key)
            citation = dict(citation_index.get(key) or {"evidence_id": key})
            citation.setdefault("evidence_id", key)
            citations.append(citation)
            if citation.get("doc_id") and citation["doc_id"] not in selected_docs:
                selected_docs.append(str(citation["doc_id"]))
    final["used_evidence_ids"] = evidence_ids
    final["citations"] = citations
    if selected_docs:
        final["selected_doc_ids"] = selected_docs
    final["claims"] = claims
    final["calculations"] = dict(state.get("calculations") or {})
    final["claim_verification_summary"] = dict(state.get("claim_verification_summary") or {})
    if dependency_limited:
        final["dependency_coverage"] = dict(state.get("dependency_coverage") or {})
    final["support_validation"] = {
        **(final.get("support_validation") or {}),
        "supported": bool(core_entailed) and all_supported,
        "claim_gate": "all_entailed" if all_supported else "partial",
    }
    return final


def make_finalize(config: AgentConfig | None = None):
    strict_claim_verification = bool(
        getattr(config, "strict_claim_verification", True) if config is not None else True
    )

    def finalize(state: dict[str, Any]) -> dict[str, Any]:
        state["step_count"] = int(state.get("step_count", 0)) + 1
        draft = state.get("draft_answer")
        reason = state.get("termination_reason") or ""
        if draft is not None:
            supported = bool((draft.get("support_validation") or {}).get("supported", False))
            if draft.get("abstained"):
                final = _normalized_abstained_draft(draft)
            elif state.get("claims"):
                gated = _apply_claim_gate(draft, state)
                if gated is not None:
                    final = gated
                elif not strict_claim_verification and state.get("legacy_support_fallback") and not (
                    int((state.get("claim_verification_summary") or {}).get("contradicted_count", 0))
                ):
                    # Keep the legacy whole-answer result as a compatibility
                    # output, but retain the claim-level audit trail.  The
                    # fallback is intentionally impossible for CONTRADICTED
                    # claims.
                    final = dict(draft)
                    final["claims"] = list(state.get("claims") or [])
                    final["calculations"] = dict(state.get("calculations") or {})
                    final["claim_verification_summary"] = dict(state.get("claim_verification_summary") or {})
                    final["claim_gate"] = "legacy_whole_answer_compat"
                    state["termination_reason"] = "completed"
                else:
                    final = _abstain_row(state, state.get("termination_reason") or "claim_verification_failed")
            elif strict_claim_verification:
                # A supported whole-answer flag is not a substitute for atomic
                # claim extraction and verification.  This also closes the
                # budget-exhaustion path where extraction never ran.
                if not state.get("termination_reason"):
                    state["termination_reason"] = "claim_extraction_missing"
                final = _abstain_row(
                    state,
                    state.get("termination_reason") or "claim_extraction_missing",
                )
            elif (
                not (
                    bool(state.get("is_multi_hop"))
                    and not bool(state.get("deterministic_conclusion_allowed", True))
                )
                and (supported or reason in ("", "completed"))
            ):
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
