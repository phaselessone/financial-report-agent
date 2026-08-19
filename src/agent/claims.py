"""Claim-level provenance helpers (checklist v3.0 §P7).

``classify_claim_type`` deterministically labels a claim as EXTRACTED,
DERIVED or SYNTHESIZED from its text and the source documents behind it,
so claim_type is stable across runs rather than whatever the extractor LLM
happens to emit. The LLM's own type (if any) is kept as the fallback under
``_classify_fallback`` and only used when the heuristic yields none.
"""

from __future__ import annotations

from typing import Any

# Verbs/markers that indicate the claim carries a derived/calculated figure
# rather than a directly-extracted one. Mirrors the P4 financial_calculator
# surface (growth_rate, yoy, cagr, gross_margin, net_margin, ratio,
# difference, percentage_point_change) plus common phrasing.
#
# Deliberately excludes broad words like 增长/下降/合计/累计/较 that frequently
# appear in non-numeric synthesis ("增长逻辑") — those would mislabel
# cross-report SYNTHESIZED claims as DERIVED. Only markers that themselves
# denote a calculation are listed.
_DERIVED_MARKERS = (
    "同比",
    "环比",
    "增速",
    "涨幅",
    "跌幅",
    "占比",
    "净额",
    "毛利率",
    "净利率",
    "倍数",
    "较上年",
    "较上期",
    "同比增长",
    "增长率",
    "复合",
    "cagr",
    "yoy",
    "growth",
)


def _is_derived(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _DERIVED_MARKERS)


def _distinct_doc_count(evidence_doc_ids: list[str]) -> int:
    return len({doc_id for doc_id in evidence_doc_ids if doc_id})


def classify_claim_type(text: str, evidence_doc_ids: list[str] | tuple[str, ...] | None) -> str:
    """Classify a single claim's type by local heuristics.

    Priority DERIVED > SYNTHESIZED > EXTRACTED.
    """
    text = text or ""
    ids = list(evidence_doc_ids or [])
    if _is_derived(text):
        return "DERIVED"
    if _distinct_doc_count(ids) >= 2:
        return "SYNTHESIZED"
    return "EXTRACTED"


def _classify_fallback(claim: dict[str, Any]) -> str:
    """Fallback: use the LLM-provided claim_type when it is one of the enum."""
    llm_type = str(claim.get("claim_type") or "").strip().upper()
    if llm_type in {"EXTRACTED", "DERIVED", "SYNTHESIZED"}:
        return llm_type
    return "EXTRACTED"
