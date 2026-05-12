from __future__ import annotations

from typing import Any

from src.utils.text_utils import extract_terms, normalize_for_match, normalize_text, strip_file_extension

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


def _min_max_normalize(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    scores = [float(row["score"]) for row in rows]
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return {row["chunk_id"]: 1.0 for row in rows}
    return {row["chunk_id"]: (float(row["score"]) - min_score) / (max_score - min_score) for row in rows}


def _query_terms(query: str, *, top_k: int = 8) -> list[str]:
    return [term for term in extract_terms(query, top_k=top_k) if term not in GENERIC_QUERY_TERMS]


def _title_overlap_score(query_terms: list[str], row: dict[str, Any]) -> int:
    title_text = normalize_for_match(
        "\n".join(
            [
                normalize_text(strip_file_extension(str(row.get("file_name", "")))),
                normalize_text(str(row.get("section_title", "") or "")),
                normalize_text(str(row.get("section_path", "") or "")),
            ]
        )
    )
    if not title_text:
        return 0
    return sum(normalize_for_match(term) in title_text for term in query_terms if normalize_for_match(term))


def _noise_penalty(row: dict[str, Any]) -> float:
    candidates = [
        normalize_text(str(row.get("section_title", "") or "")),
        normalize_text(str(row.get("section_path", "") or "")),
        normalize_text(strip_file_extension(str(row.get("file_name", "")))),
    ]
    if any(any(pattern in candidate for pattern in NOISY_SECTION_PATTERNS) for candidate in candidates if candidate):
        return 0.25
    return 0.0


class HybridRetriever:
    def __init__(self, alpha: float = 0.6, beta: float = 0.4) -> None:
        self.alpha = alpha
        self.beta = beta

    def search(
        self,
        *,
        query: str,
        dense_rows: list[dict[str, Any]],
        bm25_rows: list[dict[str, Any]],
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        dense_scores = _min_max_normalize(dense_rows)
        bm25_scores = _min_max_normalize(bm25_rows)
        query_terms = _query_terms(query)
        merged: dict[str, dict[str, Any]] = {}

        for row in dense_rows + bm25_rows:
            merged.setdefault(row["chunk_id"], dict(row))

        for chunk_id, payload in merged.items():
            dense_score = dense_scores.get(chunk_id, 0.0)
            bm25_score = bm25_scores.get(chunk_id, 0.0)
            title_overlap = _title_overlap_score(query_terms, payload)
            appendix_penalty = _noise_penalty(payload)
            payload["dense_score"] = dense_score
            payload["bm25_score"] = bm25_score
            payload["title_overlap_score"] = title_overlap
            payload["appendix_penalty"] = appendix_penalty
            payload["score"] = self.alpha * dense_score + self.beta * bm25_score + 0.12 * title_overlap - appendix_penalty
            payload["method"] = "hybrid"

        results = sorted(merged.values(), key=lambda item: item["score"], reverse=True)[:top_k]
        for rank, row in enumerate(results, start=1):
            row["rank"] = rank
        return results
