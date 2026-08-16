"""``report_lookup`` tool: doc-id scoped retrieval (checklist §P4).

``search_within_report`` / ``search_within_reports`` delegate directly to
``RetrievalRuntime.search(query, doc_ids=...)``; the doc filtering is
implemented in :mod:`src.retrieval.runtime`.
"""

from __future__ import annotations

from typing import Any


def search_within_report(runtime: Any, query: str, doc_id: str) -> dict[str, Any]:
    return runtime.search(query, doc_ids={doc_id})


def search_within_reports(runtime: Any, query: str, doc_ids: set[str]) -> dict[str, Any]:
    return runtime.search(query, doc_ids=set(doc_ids))
