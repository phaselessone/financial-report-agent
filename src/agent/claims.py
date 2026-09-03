"""Claim-level provenance helpers (checklist v3.0 §P7).

``classify_claim_type`` deterministically labels a claim as EXTRACTED,
DERIVED or SYNTHESIZED from its text and the source documents behind it,
so claim_type is stable across runs rather than whatever the extractor LLM
happens to emit. The LLM's own type (if any) is kept as the fallback under
``_classify_fallback`` and only used when the heuristic yields none.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import re
from typing import Any


CLAIM_TYPES = frozenset({"EXTRACTED", "DERIVED", "SYNTHESIZED"})


def stable_claim_id(text: str) -> str:
    """Return a content-stable id after extraction has normalized the text."""
    normalized = re.sub(r"\s+", " ", str(text or "").strip())
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
    return f"C{digest}"


@dataclass(frozen=True)
class ClaimRecord:
    """Canonical claim trace persisted by strict agent profiles."""

    claim_id: str
    text: str
    claim_type: str
    is_core: bool
    source_step_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    calculation_id: str | None = None
    parent_claim_ids: tuple[str, ...] = ()
    verification: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("claim text must be non-empty")
        if self.claim_id != stable_claim_id(self.text):
            raise ValueError("claim_id must be the stable id of claim text")
        normalized_type = str(self.claim_type or "").upper()
        if normalized_type not in CLAIM_TYPES:
            raise ValueError(f"invalid claim_type {self.claim_type!r}")
        object.__setattr__(self, "claim_type", normalized_type)

    def to_dict(self) -> dict[str, Any]:
        verification = dict(
            self.verification
            or {
                "status": "INSUFFICIENT",
                "method": "pending",
                "score": 0.0,
                "reasons": ["not_verified_yet"],
            }
        )
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "claim_type": self.claim_type,
            "is_core": self.is_core,
            "source_step_ids": list(self.source_step_ids),
            "evidence_ids": list(self.evidence_ids),
            "calculation_id": self.calculation_id,
            "parent_claim_ids": list(self.parent_claim_ids),
            "verification": verification,
            "supported": str(verification.get("status") or "") == "ENTAILED",
        }


def valid_claim_type(value: Any) -> str | None:
    normalized = str(value or "").strip().upper()
    return normalized if normalized in CLAIM_TYPES else None


def normalize_id_list(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


def claim_structure_errors(records: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    """Validate stable ids and parent references without throwing on traces."""
    by_id: dict[str, Mapping[str, Any]] = {}
    errors: dict[str, list[str]] = {}
    for record in records:
        claim_id = str(record.get("claim_id") or "").strip()
        if not claim_id:
            continue
        errors.setdefault(claim_id, [])
        if claim_id in by_id:
            errors[claim_id].append("duplicate_claim_id")
        by_id[claim_id] = record
        text = str(record.get("text") or "").strip()
        if not text or stable_claim_id(text) != claim_id:
            errors[claim_id].append("unstable_claim_id")

    adjacency: dict[str, tuple[str, ...]] = {}
    for claim_id, record in by_id.items():
        parents = normalize_id_list(record.get("parent_claim_ids"))
        present: list[str] = []
        for parent_id in parents:
            if parent_id not in by_id:
                errors[claim_id].append(f"parent_claim_missing:{parent_id}")
            else:
                present.append(parent_id)
        adjacency[claim_id] = tuple(present)

    visiting: list[str] = []
    state: dict[str, int] = {}

    def visit(claim_id: str) -> None:
        marker = state.get(claim_id, 0)
        if marker == 2:
            return
        if marker == 1:
            start = visiting.index(claim_id)
            for cycle_id in visiting[start:]:
                errors[cycle_id].append("parent_claim_cycle")
            return
        state[claim_id] = 1
        visiting.append(claim_id)
        for parent_id in adjacency.get(claim_id, ()):
            visit(parent_id)
        visiting.pop()
        state[claim_id] = 2

    for claim_id in by_id:
        visit(claim_id)
    return {claim_id: list(dict.fromkeys(items)) for claim_id, items in errors.items()}

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
    "计算结果",
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
