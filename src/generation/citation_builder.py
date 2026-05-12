from __future__ import annotations

from typing import Any

from src.utils.text_utils import normalize_text


def build_citations(
    *,
    used_evidence_ids: list[str],
    evidence_rows: list[dict[str, Any]],
    snippet_chars: int = 180,
) -> list[dict[str, Any]]:
    evidence_by_id = {row["evidence_id"]: row for row in evidence_rows}
    citations: list[dict[str, Any]] = []
    for evidence_id in used_evidence_ids:
        row = evidence_by_id.get(evidence_id)
        if not row:
            continue
        snippet_source = row.get("support_span") or row.get("child_text") or row.get("text", "")
        snippet = normalize_text(snippet_source)[:snippet_chars].rstrip()
        if len(normalize_text(snippet_source)) > snippet_chars:
            snippet += "..."
        citations.append(
            {
                "evidence_id": evidence_id,
                "chunk_id": row["chunk_id"],
                "doc_id": row["doc_id"],
                "file_name": row["file_name"],
                "page_start": row["page_start"],
                "page_end": row["page_end"],
                "section_title": row.get("section_title"),
                "element_type": row.get("element_type"),
                "support_type": row.get("support_type"),
                "bundle_id": row.get("bundle_id"),
                "snippet": snippet,
            }
        )
    return citations
