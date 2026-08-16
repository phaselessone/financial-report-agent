"""Local deterministic evidence lookup (checklist v3.0 §P4 Financial Tools).

These helpers resolve an evidence reference (chunk_id or evidence_id) against
retrieval rows without touching the network. They are pure, deterministic, and
shadow no LLM functionality.
"""

from __future__ import annotations

from typing import Any


def lookup_evidence(ref: str, *, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Resolve *ref* against pipeline rows.

    *ref* matches ``row["chunk_id"]`` (primary key, present on every retrieval
    row) or ``row.get("evidence_id")`` (present only after answer generation
    assigned ids). If a chunk_id match and an evidence_id match point at
    different rows, the chunk_id match wins. Duplicate chunk_ids -> first wins.
    """
    if not rows:
        return {"found": False, "row": None}

    chunk_match: dict[str, Any] | None = None
    evidence_match: dict[str, Any] | None = None
    for row in rows:
        if chunk_match is None and row.get("chunk_id") == ref:
            chunk_match = row
        if evidence_match is None and row.get("evidence_id") == ref:
            evidence_match = row
        # chunk_id is authoritative: stop scanning once a chunk match is found
        if chunk_match is not None:
            return {"found": True, "row": chunk_match}

    if evidence_match is not None:
        return {"found": True, "row": evidence_match}
    return {"found": False, "row": None}


def lookup_evidence_in_pool(ref: str, *, pool: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Resolve *ref* against an agent ``evidence_pool`` keyed by chunk_id.

    *ref* may be a pool key (chunk_id) or a value's ``evidence_id`` when present.
    """
    if not pool:
        return {"found": False, "row": None}

    # A chunk_id key is authoritative; check it first.
    if ref in pool:
        return {"found": True, "row": pool[ref]}

    for row in pool.values():
        if row.get("evidence_id") == ref:
            return {"found": True, "row": row}

    return {"found": False, "row": None}


def evidence_rows_for_doc(doc_id: str, *, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return rows where ``row.get("doc_id") == doc_id``, preserving input order.

    Rows missing ``doc_id`` are skipped.
    """
    return [row for row in rows if row.get("doc_id") == doc_id]
