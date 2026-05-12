from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

from src.retrieval.bm25_index import BM25Retriever, build_or_load_bm25_index
from src.retrieval.embedder import DenseEmbedder, build_or_load_embeddings
from src.retrieval.faiss_index import DenseRetriever, build_or_load_faiss_index
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.utils.io import ensure_dir
from src.utils.text_utils import extract_terms, first_sentence, normalize_text

NUMERIC_QUERY_HINTS = (
    "多少",
    "多大",
    "占比",
    "同比",
    "环比",
    "增长",
    "下降",
    "收入",
    "利润",
    "销量",
    "价格",
    "%",
    "亿元",
    "万台",
)
COMPARISON_QUERY_HINTS = ("比较", "对比", "区别", "差异", "相比", "相较", "高于", "低于")
INDUCTIVE_QUERY_HINTS = ("共性", "总结", "归纳", "趋势", "共同点", "整体来看", "主要特征")
NOISY_SECTION_PATTERNS = (
    "相关报告",
    "报告汇总",
    "图表目录",
    "目录",
    "附录",
    "风险提示",
    "免责声明",
    "投资评级说明",
    "评级说明",
)
GENERIC_QUERY_TERMS = {
    "报告",
    "研报",
    "周报",
    "月报",
    "白皮书",
    "哪些",
    "什么",
    "情况",
    "分别",
    "共同",
    "主题",
    "关键",
    "数据",
    "变化",
    "讨论",
    "聚焦",
    "强调",
}


class RetrievalRuntime:
    def __init__(
        self,
        *,
        embedder: DenseEmbedder,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        hybrid_retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        dense_top_k: int = 20,
        bm25_top_k: int = 20,
        rerank_top_k: int = 5,
        rerank_candidates_k: int | None = None,
        rerank_batch_size: int = 8,
    ) -> None:
        self.embedder = embedder
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.dense_top_k = dense_top_k
        self.bm25_top_k = bm25_top_k
        self.rerank_top_k = rerank_top_k
        self.rerank_candidates_k = max(rerank_top_k, rerank_candidates_k or max(dense_top_k, bm25_top_k))
        self.rerank_batch_size = rerank_batch_size

    @staticmethod
    def _is_numeric_query(query: str) -> bool:
        normalized = normalize_text(query)
        return any(keyword in normalized for keyword in NUMERIC_QUERY_HINTS) or any(char.isdigit() for char in normalized)

    @staticmethod
    def _query_mode(query: str) -> str:
        normalized = normalize_text(query)
        if any(keyword in normalized for keyword in COMPARISON_QUERY_HINTS):
            return "comparison"
        if any(keyword in normalized for keyword in INDUCTIVE_QUERY_HINTS):
            return "inductive"
        return "fact"

    @staticmethod
    def _expanded_bm25_query(query: str) -> str:
        terms = [term for term in extract_terms(query, top_k=8) if term not in GENERIC_QUERY_TERMS]
        if not terms:
            return query
        return normalize_text(f"{query}\n" + " ".join(terms))

    @staticmethod
    def _support_span(row: dict[str, Any]) -> str:
        support_span = normalize_text(row.get("support_span", "") or "")
        if support_span:
            return support_span
        child_text = normalize_text(row.get("text", "") or "")
        if not child_text:
            return ""
        return first_sentence(child_text, max_chars=180)

    @classmethod
    def _expand_row(cls, row: dict[str, Any], *, rank: int) -> dict[str, Any]:
        payload = dict(row)
        payload["child_text"] = row.get("text", "")
        parent_text = normalize_text(row.get("parent_text", "") or "")
        if parent_text:
            payload["text"] = parent_text
        payload["section_title"] = payload.get("section_title") or payload.get("section_path")
        payload["support_span"] = cls._support_span(row)
        payload.setdefault("bundle_id", payload.get("parent_chunk_id") or payload.get("chunk_id"))
        payload.setdefault("bundle_rank", 1)
        payload.setdefault("rank", rank)
        return payload

    @classmethod
    def _expand_rows(cls, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [cls._expand_row(row, rank=index) for index, row in enumerate(rows, start=1)]

    @staticmethod
    def _is_noise_row(row: dict[str, Any]) -> bool:
        candidates = [
            normalize_text(row.get("section_title", "") or ""),
            normalize_text(row.get("section_path", "") or ""),
            first_sentence(normalize_text(row.get("support_span") or row.get("text", "") or ""), max_chars=64),
        ]
        normalized_candidates = [candidate for candidate in candidates if candidate]
        if not normalized_candidates:
            return False
        return any(any(pattern in candidate for pattern in NOISY_SECTION_PATTERNS) for candidate in normalized_candidates)

    def _rerank_candidates(self, hybrid_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        non_noise_rows = [row for row in hybrid_rows if not self._is_noise_row(row)]
        if len(non_noise_rows) >= self.rerank_top_k:
            return non_noise_rows[: self.rerank_candidates_k]
        return hybrid_rows[: self.rerank_candidates_k]

    def search(self, query: str) -> dict[str, Any]:
        retrieval_start = perf_counter()
        query_embedding = self.embedder.encode_query(query)
        dense_rows_raw = self.dense_retriever.search(query_embedding, top_k=self.dense_top_k)
        bm25_rows_raw = self.bm25_retriever.search(self._expanded_bm25_query(query), top_k=self.bm25_top_k)
        hybrid_rows_raw = self.hybrid_retriever.search(
            query=query,
            dense_rows=dense_rows_raw,
            bm25_rows=bm25_rows_raw,
            top_k=max(self.dense_top_k, self.bm25_top_k),
        )
        dense_rows = self._expand_rows(dense_rows_raw)
        bm25_rows = self._expand_rows(bm25_rows_raw)
        hybrid_rows = self._expand_rows(hybrid_rows_raw)
        retrieval_latency_ms = round((perf_counter() - retrieval_start) * 1000, 2)

        rerank_start = perf_counter()
        rerank_candidates = self._rerank_candidates(hybrid_rows)
        rerank_rows = self.reranker.rerank(
            query,
            rerank_candidates,
            top_k=self.rerank_top_k,
            batch_size=self.rerank_batch_size,
        )
        rerank_latency_ms = round((perf_counter() - rerank_start) * 1000, 2)
        return {
            "query_mode": self._query_mode(query),
            "numeric_query": self._is_numeric_query(query),
            "dense_rows": dense_rows,
            "bm25_rows": bm25_rows,
            "hybrid_rows": hybrid_rows,
            "rerank_rows": rerank_rows,
            "timings": {
                "retrieval_latency_ms": retrieval_latency_ms,
                "rerank_latency_ms": rerank_latency_ms,
            },
        }


def build_retrieval_runtime(
    *,
    chunks: list[dict[str, Any]],
    output_dir: Path,
    model_cache_dir: Path,
    embedding_model: str,
    reranker_model: str,
    device: str = "cuda",
    embedding_batch_size: int = 16,
    rerank_batch_size: int = 8,
    dense_top_k: int = 20,
    bm25_top_k: int = 20,
    rerank_top_k: int = 5,
    rerank_candidates_k: int | None = None,
    rebuild_indexes: bool = False,
) -> RetrievalRuntime:
    chunk_lookup = {chunk["chunk_id"]: chunk for chunk in chunks}
    dense_output_dir = ensure_dir(output_dir / "indexes" / "dense")
    bm25_output_dir = ensure_dir(output_dir / "indexes" / "bm25")

    embedder = DenseEmbedder(embedding_model, cache_dir=model_cache_dir, device=device)
    embeddings, chunk_ids = build_or_load_embeddings(
        chunks=chunks,
        embedder=embedder,
        output_dir=dense_output_dir,
        rebuild=rebuild_indexes,
        batch_size=embedding_batch_size,
    )
    dense_index = build_or_load_faiss_index(
        embeddings=embeddings,
        chunk_ids=chunk_ids,
        output_dir=dense_output_dir,
        rebuild=rebuild_indexes,
    )
    dense_retriever = DenseRetriever(dense_index, chunk_lookup)

    bm25_index = build_or_load_bm25_index(chunks=chunks, output_dir=bm25_output_dir, rebuild=rebuild_indexes)
    bm25_retriever = BM25Retriever(bm25_index, chunk_lookup)

    hybrid_retriever = HybridRetriever(alpha=0.6, beta=0.4)
    reranker = CrossEncoderReranker(reranker_model, cache_dir=model_cache_dir, device=device)
    return RetrievalRuntime(
        embedder=embedder,
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        dense_top_k=dense_top_k,
        bm25_top_k=bm25_top_k,
        rerank_top_k=rerank_top_k,
        rerank_candidates_k=rerank_candidates_k,
        rerank_batch_size=rerank_batch_size,
    )
