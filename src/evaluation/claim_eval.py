"""Claim-level provenance evaluation (checklist v3.0 §P7).

Three metrics over a batch of agent runs, computed from each run's
``state["claims"]`` (produced by the ``extract_claims`` node) and the draft's
whole-answer ``support_validation``:

- **Claim Support Precision**      supported claims / total claims
- **Critical Claim Support Rate**  supported critical claims / critical claims
  (a claim is critical when it carries a numeric figure, or is the first claim)
- **Derived Claim Validation Accuracy** (v1 weak): the fraction of DERIVED
  claims whose text also appears in the final answer. v1 does not re-run the P4
  financial calculator, so this is a consistency signal, not a calculator
  recheck — documented as N/A for derived validation strictness.

Honest scoping: these are process metrics over the agent's own claims, not a
human-annotated claim gold set.
"""

from __future__ import annotations

import re
from typing import Any

CRITICAL_NUMERIC_PATTERN = re.compile(r"\d")


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
    supported_count = sum(1 for c in claims if c.get("supported"))
    critical = critical_claim_ids(claims)
    critical_count = len(critical)
    critical_supported = sum(1 for c in claims if c["claim_id"] in critical and c.get("supported"))

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
        "claim_count": claim_count,
        "supported_count": supported_count,
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
    return {
        "queries": queries,
        "total_claims": total_claims,
        "claim_support_precision": _mean([r.get("claim_support_precision") for r in rows]),
        "critical_claim_support_rate": _mean([r.get("critical_claim_support_rate") for r in rows]),
        "derived_consistent_rate": _mean([r.get("derived_consistent") for r in rows]),
    }
