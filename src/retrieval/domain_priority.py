"""Domain-priority post-processing for retrieval results.

The baseline answer eval applies this after ``RetrievalRuntime.search()`` to
rank rows matching the query's industry domain first. The agentic retrieve node
applies the same post-processing (checklist §P3 gate remediation R2) so both
pipelines rank evidence identically.
"""

from __future__ import annotations

from typing import Any

from src.utils.text_utils import industry_from_file_name

DOMAIN_COMPATIBILITY_GROUPS = {
    "consumer": {"consumer", "liquor"},
    "liquor": {"consumer", "liquor"},
}


def _row_domain_bucket(row: dict[str, Any]) -> str:
    return str(row.get("industry") or industry_from_file_name(str(row.get("file_name", ""))) or "")


def _normalize_domain_bucket(bucket: str) -> str:
    return (bucket or "").strip().lower()


def _is_weak_domain_bucket(bucket: str) -> bool:
    normalized = _normalize_domain_bucket(bucket)
    return not normalized or normalized == "other"


def _domains_compatible(left: str, right: str) -> bool:
    left_normalized = _normalize_domain_bucket(left)
    right_normalized = _normalize_domain_bucket(right)
    if not left_normalized or not right_normalized:
        return False
    if left_normalized == right_normalized:
        return True
    return (
        right_normalized in DOMAIN_COMPATIBILITY_GROUPS.get(left_normalized, {left_normalized})
        or left_normalized in DOMAIN_COMPATIBILITY_GROUPS.get(right_normalized, {right_normalized})
    )


def apply_retrieval_domain_priority(retrieval_result: dict[str, Any], domain_hint: str) -> dict[str, Any]:
    """Move rows matching ``domain_hint`` (exact, then compatible) to the front of each row list."""
    normalized_hint = _normalize_domain_bucket(domain_hint)
    if _is_weak_domain_bucket(normalized_hint):
        return retrieval_result
    prioritized_result = dict(retrieval_result)
    for key in ("dense_rows", "bm25_rows", "hybrid_rows", "rerank_rows"):
        rows = list(retrieval_result.get(key, []))
        if not rows:
            continue
        exact_rows = [row for row in rows if _normalize_domain_bucket(_row_domain_bucket(row)) == normalized_hint]
        compatible_rows = [
            row
            for row in rows
            if row not in exact_rows and _domains_compatible(_row_domain_bucket(row), normalized_hint)
        ]
        other_rows = [row for row in rows if row not in exact_rows and row not in compatible_rows]
        if exact_rows or compatible_rows:
            prioritized_result[key] = exact_rows + compatible_rows + other_rows
    return prioritized_result
