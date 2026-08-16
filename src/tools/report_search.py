"""``report_search`` tool: adapter around ``RetrievalRuntime.search()``.

Checklist v3.0 §P4: ``report_search = adapter around RetrievalRuntime.search()``.
Reuses ``apply_retrieval_domain_priority`` verbatim — this module adds no
retrieval or rerank logic of its own.
"""

from __future__ import annotations

from typing import Any

from src.retrieval.domain_priority import apply_retrieval_domain_priority


class ReportSearchTool:
    def __init__(self, runtime: Any, domain_hint: str = "") -> None:
        self.runtime = runtime
        self.domain_hint = domain_hint

    def search(self, query: str, *, domain_hint: str | None = None) -> dict[str, Any]:
        result = self.runtime.search(query)
        hint = domain_hint if domain_hint is not None else self.domain_hint
        return apply_retrieval_domain_priority(result, hint)


def top_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    return list(result.get("rerank_rows") or result.get("hybrid_rows") or [])
