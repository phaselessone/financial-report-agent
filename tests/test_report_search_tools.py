"""Financial tool tests: report_search / report_lookup adapters (P4, W-B)."""

from __future__ import annotations

import unittest
from typing import Any

from src.retrieval.domain_priority import apply_retrieval_domain_priority
from src.tools.report_lookup import search_within_report, search_within_reports
from src.tools.report_search import ReportSearchTool, top_rows


def make_row(*, chunk_id: str, doc_id: str, text: str, industry: str = "") -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "file_name": f"{doc_id}.pdf",
        "text": text,
        "industry": industry,
        "score": 3.0,
    }


def make_result(rows: list[dict[str, Any]], *, filtered: bool = False) -> dict[str, Any]:
    return {
        "query_mode": "fact",
        "numeric_query": False,
        "dense_rows": list(rows),
        "bm25_rows": list(rows),
        "hybrid_rows": list(rows),
        "rerank_rows": list(rows),
        "timings": {},
        **({"filtered": True} if filtered else {}),
    }


class FakeRuntime:
    def __init__(self, result: dict[str, Any]) -> None:
        self._result = result
        self.calls: list[tuple[str, Any]] = []

    def search(self, query: str, *, doc_ids: set[str] | None = None) -> dict[str, Any]:
        self.calls.append((query, set(doc_ids) if doc_ids is not None else None))
        return self._result


class ReportSearchToolTests(unittest.TestCase):
    def test_passthrough_without_domain_hint(self) -> None:
        result = make_result([make_row(chunk_id="c1", doc_id="d1", text="a")])
        runtime = FakeRuntime(result)
        tool = ReportSearchTool(runtime)  # type: ignore[arg-type]
        self.assertIs(tool.search("查询"), result)
        self.assertEqual(runtime.calls, [("查询", None)])

    def test_constructor_domain_hint_used_when_arg_none(self) -> None:
        liquor = make_row(chunk_id="c1", doc_id="d1", text="a", industry="liquor")
        semi = make_row(chunk_id="c2", doc_id="d2", text="b", industry="semiconductor")
        result = make_result([liquor, semi])
        runtime = FakeRuntime(result)
        tool = ReportSearchTool(runtime, domain_hint="liquor")  # type: ignore[arg-type]
        out = tool.search("查询")
        # Domain-matching rows moved to the front of every row list.
        expected = apply_retrieval_domain_priority(result, "liquor")
        self.assertEqual(out, expected)
        self.assertEqual([row["chunk_id"] for row in out["rerank_rows"]], ["c1", "c2"])

    def test_arg_domain_hint_overrides_constructor(self) -> None:
        liquor = make_row(chunk_id="c1", doc_id="d1", text="a", industry="liquor")
        semi = make_row(chunk_id="c2", doc_id="d2", text="b", industry="semiconductor")
        result = make_result([liquor, semi])
        runtime = FakeRuntime(result)
        tool = ReportSearchTool(runtime, domain_hint="liquor")  # type: ignore[arg-type]
        out = tool.search("查询", domain_hint="semiconductor")
        self.assertEqual([row["chunk_id"] for row in out["rerank_rows"]], ["c2", "c1"])

    def test_weak_domain_hint_returns_unchanged(self) -> None:
        results = make_result([make_row(chunk_id="c1", doc_id="d1", text="a", industry="semiconductor")])
        runtime = FakeRuntime(results)
        tool = ReportSearchTool(runtime, domain_hint="")  # type: ignore[arg-type]
        self.assertIs(tool.search("查询"), results)


class TopRowsTests(unittest.TestCase):
    def test_top_rows_returns_rerank_rows(self) -> None:
        result = make_result([make_row(chunk_id="c1", doc_id="d1", text="a")])
        self.assertEqual(top_rows(result), result["rerank_rows"])

    def test_top_rows_falls_back_to_hybrid(self) -> None:
        result = make_result([make_row(chunk_id="c1", doc_id="d1", text="a")])
        result["rerank_rows"] = []
        self.assertEqual(top_rows(result), result["hybrid_rows"])

    def test_top_rows_empty(self) -> None:
        result = make_result([])
        result["rerank_rows"] = []
        result["hybrid_rows"] = []
        self.assertEqual(top_rows(result), [])


class ReportLookupTests(unittest.TestCase):
    def test_search_within_report_scopes_to_single_doc(self) -> None:
        result = make_result([])
        runtime = FakeRuntime(result)
        out = search_within_report(runtime, "查询", "doc-x")  # type: ignore[arg-type]
        self.assertIs(out, result)
        self.assertEqual(runtime.calls, [("查询", {"doc-x"})])

    def test_search_within_reports_scopes_to_multi_doc(self) -> None:
        result = make_result([])
        runtime = FakeRuntime(result)
        out = search_within_reports(runtime, "查询", {"doc-a", "doc-b"})  # type: ignore[arg-type]
        self.assertIs(out, result)
        self.assertEqual(runtime.calls, [("查询", {"doc-a", "doc-b"})])


if __name__ == "__main__":
    unittest.main()
