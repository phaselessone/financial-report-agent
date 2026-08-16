"""Citation verification (checklist v3.0 §P4): 'citation verify reuses existing validation'.

``verify_citations`` wraps the existing ``validate_answer_support`` and
``build_citations`` helpers. It adds only id resolvability and citation
well-formedness checks on top of those two existing contracts — no new
validation rules and never any LLM call.
"""

from __future__ import annotations

from typing import Any

from src.generation.citation_builder import build_citations
from src.generation.support_validator import validate_answer_support


def verify_citations(
    *,
    query: str,
    question_type: str,
    final_answer: str,
    evidence_summary: str,
    used_evidence_ids: list[str],
    evidence_rows: list[dict[str, Any]],
    answer_mode: str | None = None,
) -> dict[str, Any]:
    """Verify a synthesized answer's citations reusing existing validation.

    ``evidence_rows`` must be post-answer selection rows carrying ``evidence_id``
    (plus ``doc_id`` / ``chunk_id`` / ``text``/``support_span``). ``valid`` is
    ``True`` only when every used id resolves, every built citation is
    well-formed, and ``validate_answer_support`` reports ``supported``.
    """
    present_ids = {row["evidence_id"] for row in evidence_rows if row.get("evidence_id")}
    missing = list(dict.fromkeys(eid for eid in used_evidence_ids if eid not in present_ids))

    support_validation = validate_answer_support(
        query=query,
        question_type=question_type,
        final_answer=final_answer,
        evidence_summary=evidence_summary,
        used_evidence_ids=used_evidence_ids,
        evidence_rows=evidence_rows,
        answer_mode=answer_mode,
    )

    citations = build_citations(used_evidence_ids=used_evidence_ids, evidence_rows=evidence_rows)
    malformed = [c for c in citations if not (c.get("doc_id") and c.get("chunk_id"))]

    valid = bool(not missing and not malformed and support_validation.get("supported"))
    return {
        "valid": valid,
        "missing_evidence_ids": missing,
        "malformed_citations": malformed,
        "support_validation": support_validation,
        "citations": citations,
    }
