"""Claim-level provenance evaluation (checklist v3.0 §P7).

Process metrics over a batch of agent runs, computed from each run's
``state["claims"]`` (produced by the ``extract_claims`` node) and the draft's
whole-answer ``support_validation``:

- **Claim Entailment Yield**       self-marked ENTAILED claims / total claims
- **Critical Claim Support Rate**  supported critical claims / critical claims
  (a claim is critical when it carries a numeric figure, or is the first claim)
- **Derived Claim Validation Accuracy** (v1 weak): the fraction of DERIVED
  claims whose text also appears in the final answer. v1 does not re-run the P4
  financial calculator, so this is a consistency signal, not a calculator
  recheck — documented as N/A for derived validation strictness.

Honest scoping: these are process metrics over the agent's own claims, not a
human-annotated claim gold set.  The legacy ``claim_support_precision`` key is
kept as an input/output adapter, but it is not labelled as accuracy.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

CRITICAL_NUMERIC_PATTERN = re.compile(r"\d")


def _claim_supported(claim: dict[str, Any]) -> bool:
    verification = claim.get("verification") or {}
    if isinstance(verification, dict) and verification.get("status") is not None:
        return str(verification.get("status")) == "ENTAILED"
    return bool(claim.get("supported"))


def _claim_status(claim: Mapping[str, Any]) -> str:
    verification = claim.get("verification")
    value = verification.get("status") if isinstance(verification, Mapping) else None
    if value is None:
        value = claim.get("status") or claim.get("verification_status")
    if value is None and claim.get("supported") is not None:
        value = "ENTAILED" if claim.get("supported") else "INSUFFICIENT"
    return str(value or "").upper()


def _claim_evidence_ids(claim: Mapping[str, Any]) -> set[str]:
    value = claim.get("evidence_ids") or claim.get("mapped_evidence_ids")
    verification = claim.get("verification")
    if value is None and isinstance(verification, Mapping):
        value = verification.get("evidence_ids")
    if not isinstance(value, (list, tuple, set)):
        return set()
    return {str(item) for item in value if str(item)}


def _normalized_claim_text(claim: Mapping[str, Any]) -> str:
    return " ".join(str(claim.get("text") or "").lower().split())


def evaluate_claim_verification(
    predicted_claims: Sequence[Mapping[str, Any]] | None,
    gold_claims: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare reviewed claim status and evidence through one gold seam.

    Matching prefers ``claim_id`` and falls back to exact normalized text so a
    reviewed benchmark does not need to predict runtime-generated hash IDs.
    Extra ENTAILED predictions count as false positives; non-asserted extras do
    not change the gold denominator.
    """
    predictions = [claim for claim in (predicted_claims or []) if isinstance(claim, Mapping)]
    by_id = {str(claim.get("claim_id")): index for index, claim in enumerate(predictions) if claim.get("claim_id")}
    by_text: dict[str, list[int]] = {}
    for index, claim in enumerate(predictions):
        by_text.setdefault(_normalized_claim_text(claim), []).append(index)

    used: set[int] = set()
    comparisons: list[dict[str, Any]] = []
    for gold in gold_claims:
        match_index = by_id.get(str(gold.get("claim_id") or ""))
        if match_index in used:
            match_index = None
        if match_index is None:
            match_index = next((index for index in by_text.get(_normalized_claim_text(gold), []) if index not in used), None)
        predicted = predictions[match_index] if match_index is not None else {}
        if match_index is not None:
            used.add(match_index)
        expected_status = str(gold.get("status") or "").upper()
        expected_evidence = _claim_evidence_ids(gold)
        status_match = _claim_status(predicted) == expected_status
        evidence_match = _claim_evidence_ids(predicted) == expected_evidence
        comparisons.append(
            {
                "gold_claim_id": str(gold.get("claim_id") or ""),
                "predicted_claim_id": str(predicted.get("claim_id") or "") or None,
                "gold_status": expected_status,
                "predicted_status": _claim_status(predicted) or None,
                "status_match": status_match,
                "evidence_match": evidence_match,
                "verification_match": status_match and evidence_match,
            }
        )

    for index, predicted in enumerate(predictions):
        if index not in used and _claim_status(predicted) == "ENTAILED":
            comparisons.append(
                {
                    "gold_claim_id": None,
                    "predicted_claim_id": str(predicted.get("claim_id") or "") or None,
                    "gold_status": None,
                    "predicted_status": "ENTAILED",
                    "status_match": False,
                    "evidence_match": False,
                    "verification_match": False,
                    "unexpected_entailed_claim": True,
                }
            )

    count = len(comparisons)
    mean = lambda field: (sum(bool(row[field]) for row in comparisons) / count) if count else None
    return {
        "metric_scope": "reviewed_gold",
        "evaluated_claim_count": count,
        "claim_status_accuracy": mean("status_match"),
        "claim_evidence_accuracy": mean("evidence_match"),
        "claim_verification_accuracy": mean("verification_match"),
        "comparisons": comparisons,
    }


def is_critical_claim(claim: dict[str, Any], *, index: int = -1) -> bool:
    """A claim is critical when it is the first claim or carries a numeric figure."""
    if index == 0:
        return True
    text = str(claim.get("text") or "")
    return bool(CRITICAL_NUMERIC_PATTERN.search(text))


def critical_claim_ids(claims: list[dict[str, Any]]) -> set[str]:
    return {c["claim_id"] for idx, c in enumerate(claims) if is_critical_claim(c, index=idx)}


def analyze_claims_from_state(state: dict[str, Any]) -> dict[str, Any]:
    """Compute per-query claim metrics from an agent state dict."""
    claims = list(state.get("claims") or [])
    claim_count = len(claims)
    supported_count = sum(1 for c in claims if _claim_supported(c))
    critical = critical_claim_ids(claims)
    critical_count = len(critical)
    critical_supported = sum(1 for c in claims if c["claim_id"] in critical and _claim_supported(c))

    draft = state.get("draft_answer") or {}
    final_text = str(draft.get("final_answer") or draft.get("answer") or "")
    derived_claims = [c for c in claims if c.get("claim_type") == "DERIVED"]
    # No DERIVED claims -> None (N/A), not False: the metric only speaks when
    # there is a derived figure to validate (v1 weak consistency signal).
    derived_consistent = (
        all(str(c.get("text") or "") in final_text for c in derived_claims)
        if derived_claims
        else None
    )

    return {
        "metric_scope": "process",
        "claim_count": claim_count,
        "supported_count": supported_count,
        "claim_entailment_yield": (supported_count / claim_count) if claim_count else None,
        "claim_support_precision": (supported_count / claim_count) if claim_count else None,
        "critical_count": critical_count,
        "critical_supported": critical_supported,
        "critical_claim_support_rate": (critical_supported / critical_count) if critical_count else None,
        "derived_consistent": derived_consistent,
    }


def _mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present) / len(present)


def build_claim_eval_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-query rows into a claim-eval summary dict."""
    queries = len(rows)
    total_claims = sum(int(r.get("claim_count", 0)) for r in rows)
    entailment_yield = _mean([
        r.get("claim_entailment_yield", r.get("claim_support_precision"))
        for r in rows
    ])
    return {
        "metric_scope": "process",
        "queries": queries,
        "total_claims": total_claims,
        "claim_entailment_yield": entailment_yield,
        "claim_support_precision": entailment_yield,
        "critical_claim_support_rate": _mean([r.get("critical_claim_support_rate") for r in rows]),
        "derived_consistent_rate": _mean([r.get("derived_consistent") for r in rows]),
    }
