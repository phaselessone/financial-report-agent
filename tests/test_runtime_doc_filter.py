"""Doc-scoped retrieval tests for ``RetrievalRuntime.search`` (P4, W-B).

Constructs a real ``RetrievalRuntime`` with fake retrievers/reranker (no
torch/models) and asserts:

- the default (``doc_ids=None``) path is unchanged: no ``"filtered"`` key and
  the retrievers still receive ``dense_top_k`` / ``bm25_top_k`` (20);
- the filtered path over-fetches, filters before hybrid scoring/reranking, and
  surfaces target-doc rows that would otherwise rank below the default top-20;
- empty / unknown ``doc_ids`` produce empty row lists with ``"filtered": True``;
- the module-level pure filter helpers behave correctly.
"""

from __future__ import annotations

import unittest
from typing import Any

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.runtime import RetrievalRuntime, filter_retrieval_result, filter_rows_by_doc_ids


def make_row(*, chunk_id: str, doc_id: str, text: str, score: float = 3.0) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "file_name": f"{doc_id}.pdf",
        "page_start": 1,
        "page_end": 1,
        "text": text,
        "child_text": text,
        "support_span": text,
        "section_title": "",
        "section_path": "",
        "chunk_type": "text",
        "element_type": "paragraph",
        "score": score,
    }


class FakeEmbedder:
    def encode_query(self, query: str) -> Any:
        return [0.0]


class FakeDenseRetriever:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.received_top_k: list[int] = []

    def search(self, query_or_embedding: Any, *, top_k: int) -> list[dict[str, Any]]:
        self.received_top_k.append(top_k)
        return list(self._rows[:top_k])


class FakeBM25Retriever:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.received_top_k: list[int] = []

    def search(self, query: str, *, top_k: int) -> list[dict[str, Any]]:
        self.received_top_k.append(top_k)
        return list(self._rows[:top_k])


class FakeReranker:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def rerank(self, query: str, candidates: list[dict[str, Any]], *, top_k: int, batch_size: int) -> list[dict[str, Any]]:
        self.calls.append(len(candidates))
        result: list[dict[str, Any]] = []
        for row in candidates[:top_k]:
            payload = dict(row)
            payload["rerank_score"] = row.get("score", 3.0)
            result.append(payload)
        return result


def build_runtime(
    dense_rows: list[dict[str, Any]],
    bm25_rows: list[dict[str, Any]],
) -> tuple[RetrievalRuntime, FakeDenseRetriever, FakeBM25Retriever, FakeReranker]:
    dense = FakeDenseRetriever(dense_rows)
    bm25 = FakeBM25Retriever(bm25_rows)
    reranker = FakeReranker()
    runtime = RetrievalRuntime(
        embedder=FakeEmbedder(),  # type: ignore[arg-type]
        dense_retriever=dense,  # type: ignore[arg-type]
        bm25_retriever=bm25,  # type: ignore[arg-type]
        hybrid_retriever=HybridRetriever(),
        reranker=reranker,  # type: ignore[arg-type]
    )
    return runtime, dense, bm25, reranker


class DefaultPathTests(unittest.TestCase):
    def test_default_path_has_no_filtered_key_and_top_k_20(self) -> None:
        rows = [make_row(chunk_id=f"c{i}", doc_id="d1", text=f"文本 {i}", score=float(30 - i)) for i in range(25)]
        runtime, dense, bm25, _ = build_runtime(rows, rows)
        result = runtime.search("查询")

        self.assertNotIn("filtered", result)
        self.assertEqual(dense.received_top_k, [20])
        self.assertEqual(bm25.received_top_k, [20])


class FilteredPathTests(unittest.TestCase):
    def test_filtered_path_overfetches_and_surfaces_target_doc(self) -> None:
        # Build 25 rows across 3 docs. The target doc's rows carry the *lowest*
        # scores, so with a default top-20 they would be dropped entirely.
        rows: list[dict[str, Any]] = []
        for i in range(10):
            rows.append(make_row(chunk_id=f"a{i}", doc_id="doc_a", text=f"文档A文本 {i}", score=float(100 - i)))
        for i in range(10):
            rows.append(make_row(chunk_id=f"b{i}", doc_id="doc_b", text=f"文档B文本 {i}", score=float(50 - i)))
        for i in range(5):
            rows.append(make_row(chunk_id=f"t{i}", doc_id="target", text=f"目标文本 {i}", score=float(5 - i)))

        runtime, dense, bm25, reranker = build_runtime(rows, rows)
        result = runtime.search("查询", doc_ids={"target"})

        # Over-fetch: min(200, max(8, 20*8)) == 160
        self.assertEqual(dense.received_top_k, [160])
        self.assertEqual(bm25.received_top_k, [160])

        self.assertTrue(result["filtered"])
        target_ids = {"t0", "t1", "t2", "t3", "t4"}
        for key in ("dense_rows", "bm25_rows", "hybrid_rows", "rerank_rows"):
            got = {row["doc_id"] for row in result[key]}
            self.assertEqual(got, {"target"}, key)
            self.assertEqual({row["chunk_id"] for row in result[key]}, target_ids, key)
        # Reranker must have NEVER seen non-target rows (filter happened first).
        for count in reranker.calls:
            self.assertLessEqual(count, 5)

    def test_empty_doc_ids_yields_empty_lists(self) -> None:
        rows = [make_row(chunk_id=f"c{i}", doc_id="d1", text=f"文本 {i}") for i in range(10)]
        runtime, _, _, _ = build_runtime(rows, rows)
        result = runtime.search("查询", doc_ids=set())
        self.assertTrue(result["filtered"])
        for key in ("dense_rows", "bm25_rows", "hybrid_rows", "rerank_rows"):
            self.assertEqual(result[key], [], key)

    def test_unknown_doc_id_yields_empty_lists(self) -> None:
        rows = [make_row(chunk_id=f"c{i}", doc_id="d1", text=f"文本 {i}") for i in range(10)]
        runtime, _, _, _ = build_runtime(rows, rows)
        result = runtime.search("查询", doc_ids={"no_such_doc"})
        self.assertTrue(result["filtered"])
        for key in ("dense_rows", "bm25_rows", "hybrid_rows", "rerank_rows"):
            self.assertEqual(result[key], [], key)


class FilterHelperTests(unittest.TestCase):
    def test_filter_rows_by_doc_ids_preserves_order(self) -> None:
        rows = [
            make_row(chunk_id="c1", doc_id="d1", text="a"),
            make_row(chunk_id="c2", doc_id="d2", text="b"),
            make_row(chunk_id="c3", doc_id="d1", text="c"),
            make_row(chunk_id="c4", doc_id="d3", text="d"),
        ]
        self.assertEqual(
            [row["chunk_id"] for row in filter_rows_by_doc_ids(rows, {"d1"})],
            ["c1", "c3"],
        )

    def test_filter_retrieval_result_preserves_non_row_keys_and_sets_filtered(self) -> None:
        rows = [
            make_row(chunk_id="c1", doc_id="d1", text="a"),
            make_row(chunk_id="c2", doc_id="d2", text="b"),
        ]
        result = {
            "query_mode": "fact",
            "numeric_query": False,
            "dense_rows": rows,
            "bm25_rows": rows,
            "hybrid_rows": rows,
            "rerank_rows": rows,
            "timings": {"retrieval_latency_ms": 1.0, "rerank_latency_ms": 2.0},
            "custom_key": object(),
        }
        filtered = filter_retrieval_result(result, {"d1"})
        self.assertTrue(filtered["filtered"])
        self.assertEqual(filtered["query_mode"], "fact")
        self.assertIs(filtered["numeric_query"], False)
        self.assertEqual(filtered["timings"], result["timings"])
        self.assertIs(filtered["custom_key"], result["custom_key"])
        for key in ("dense_rows", "bm25_rows", "hybrid_rows", "rerank_rows"):
            self.assertEqual([row["chunk_id"] for row in filtered[key]], ["c1"], key)
        # Original dict is untouched.
        self.assertNotIn("filtered", result)
        self.assertEqual([row["chunk_id"] for row in result["rerank_rows"]], ["c1", "c2"])


if __name__ == "__main__":
    unittest.main()
