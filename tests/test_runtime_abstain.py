import unittest

import numpy as np

from src.generation.abstain import decide_abstention

try:
    from src.retrieval.runtime import RetrievalRuntime
except ModuleNotFoundError:
    RetrievalRuntime = None


class StubEmbedder:
    def encode_query(self, query: str) -> np.ndarray:
        return np.asarray([1.0], dtype=np.float32)


class StubRetriever:
    def __init__(self, rows):
        self.rows = rows

    def search(self, *_args, top_k: int = 20, **_kwargs):
        return [dict(row) for row in self.rows[:top_k]]


class StubHybridRetriever:
    def search(self, *, query, dense_rows, bm25_rows, top_k: int = 20):
        del query
        merged = [dict(row) for row in dense_rows + bm25_rows]
        return merged[:top_k]


class CaptureReranker:
    def __init__(self) -> None:
        self.seen_rows = []

    def rerank(self, query: str, rows: list[dict[str, object]], *, top_k: int = 5, batch_size: int = 8):
        del query, batch_size
        self.seen_rows = list(rows)
        return [dict(row, rerank_score=float(index), score=float(index), rank=index + 1) for index, row in enumerate(rows[:top_k])]


@unittest.skipIf(RetrievalRuntime is None, "Retrieval runtime dependencies are not installed locally.")
class RetrievalRuntimeTests(unittest.TestCase):
    def _make_row(self, chunk_id: str, *, section_title: str = "") -> dict[str, object]:
        return {
            "chunk_id": chunk_id,
            "doc_id": f"doc-{chunk_id}",
            "file_name": f"{chunk_id}.pdf",
            "page_start": 1,
            "page_end": 1,
            "text": f"text for {chunk_id}",
            "section_title": section_title,
            "section_path": section_title,
            "score": 1.0,
        }

    def test_rerank_uses_rerank_candidates_k(self) -> None:
        dense_rows = [self._make_row(f"dense-{index}") for index in range(2)]
        bm25_rows = [self._make_row(f"bm25-{index}") for index in range(5)]
        reranker = CaptureReranker()
        runtime = RetrievalRuntime(
            embedder=StubEmbedder(),
            dense_retriever=StubRetriever(dense_rows),
            bm25_retriever=StubRetriever(bm25_rows),
            hybrid_retriever=StubHybridRetriever(),
            reranker=reranker,
            dense_top_k=2,
            bm25_top_k=5,
            rerank_top_k=3,
            rerank_candidates_k=5,
            rerank_batch_size=4,
        )

        runtime.search("测试 query")
        self.assertEqual(len(reranker.seen_rows), 5)

    def test_rerank_filters_noise_sections_when_clean_candidates_exist(self) -> None:
        dense_rows = [
            self._make_row("noise-0", section_title="相关报告汇总"),
            self._make_row("dense-1", section_title="核心观点"),
            self._make_row("dense-2", section_title="行业趋势"),
        ]
        bm25_rows = [
            self._make_row("bm25-1", section_title="正文"),
            self._make_row("bm25-2", section_title="重点数据"),
            self._make_row("bm25-3", section_title="投资建议"),
        ]
        reranker = CaptureReranker()
        runtime = RetrievalRuntime(
            embedder=StubEmbedder(),
            dense_retriever=StubRetriever(dense_rows),
            bm25_retriever=StubRetriever(bm25_rows),
            hybrid_retriever=StubHybridRetriever(),
            reranker=reranker,
            dense_top_k=6,
            bm25_top_k=6,
            rerank_top_k=3,
            rerank_candidates_k=5,
            rerank_batch_size=4,
        )

        runtime.search("测试 query")
        seen_chunk_ids = {row["chunk_id"] for row in reranker.seen_rows}
        self.assertNotIn("noise-0", seen_chunk_ids)


class AbstainTests(unittest.TestCase):
    def test_fallback_reason_forces_abstain(self) -> None:
        abstained, reason = decide_abstention(
            question_type="fact",
            evidence_rows=[{"doc_id": "doc-1"}],
            retrieval_scores={},
            support_validation={"supported": True},
            json_parse_failures=0,
            answer_mode="numeric_fact",
            fallback_reason="unsupported_answer",
        )
        self.assertTrue(abstained)
        self.assertEqual(reason, "unsupported_answer")

    def test_domain_mismatch_forces_abstain(self) -> None:
        abstained, reason = decide_abstention(
            question_type="comparison",
            evidence_rows=[{"doc_id": "doc-1"}, {"doc_id": "doc-2"}],
            retrieval_scores={},
            support_validation={"supported": True, "domain_mismatch": True},
            json_parse_failures=0,
            answer_mode="comparison",
            fallback_reason=None,
        )
        self.assertTrue(abstained)
        self.assertEqual(reason, "domain_mismatch")

    def test_low_title_overlap_blocks_report_lookup(self) -> None:
        abstained, reason = decide_abstention(
            question_type="fact",
            evidence_rows=[{"doc_id": "doc-1"}],
            retrieval_scores={},
            support_validation={"supported": True, "low_title_overlap": True},
            json_parse_failures=0,
            answer_mode="report_lookup",
            fallback_reason=None,
        )
        self.assertTrue(abstained)
        self.assertEqual(reason, "low_title_overlap")

    def test_missing_query_terms_high_forces_abstain(self) -> None:
        abstained, reason = decide_abstention(
            question_type="comparison",
            evidence_rows=[{"doc_id": "doc-1"}, {"doc_id": "doc-2"}],
            retrieval_scores={},
            support_validation={"supported": True, "missing_query_terms_high": True},
            json_parse_failures=0,
            answer_mode="comparison",
            fallback_reason=None,
        )
        self.assertTrue(abstained)
        self.assertEqual(reason, "missing_query_terms_high")


if __name__ == "__main__":
    unittest.main()
