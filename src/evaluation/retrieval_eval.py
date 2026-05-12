from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.evaluation.benchmark_assets import build_doc_manifest
from src.utils.io import iter_jsonl, read_jsonl, write_json, write_jsonl
from src.utils.text_utils import keyword_hits


def materialize_eval_set(
    *,
    seed_path: Path,
    chunks_path: Path,
    output_path: Path,
    report_output_path: Path | None = None,
) -> list[dict[str, Any]]:
    chunk_rows = read_jsonl(chunks_path)
    doc_manifest = build_doc_manifest(chunk_rows)
    docs_by_key = {row["doc_key"]: row for row in doc_manifest}
    chunks_by_doc_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in chunk_rows:
        chunks_by_doc_id[row["doc_id"]].append(row)

    materialized: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []

    for row in iter_jsonl(seed_path):
        question_id = row.get("question_id") or row.get("qid")
        if not question_id:
            raise ValueError(f"seed row missing question_id/qid: {row}")
        target_doc_keys = list(row.get("target_doc_keys", []))
        target_docs = [docs_by_key.get(doc_key) for doc_key in target_doc_keys]
        unresolved_keys = [doc_key for doc_key, doc in zip(target_doc_keys, target_docs) if doc is None]
        if unresolved_keys:
            missing_rows.append(
                {
                    "question_id": question_id,
                    "industry": row.get("industry", "other"),
                    "question_type": row.get("question_type", "fact"),
                    "intent": row.get("intent", row.get("question_type", "fact")),
                    "query": row["query"],
                    "missing_doc_keys": unresolved_keys,
                }
            )
            continue

        gold_chunks: list[dict[str, Any]] = []
        for target_doc in target_docs:
            candidates = chunks_by_doc_id.get(target_doc["doc_id"], [])
            page_hints = set(int(page) for page in row.get("page_hints", []))
            keywords = list(row.get("keyword_hints", []))
            scored: list[tuple[float, int, dict[str, Any]]] = []
            for candidate in candidates:
                score = 0.0
                if page_hints and (int(candidate["page_start"]) in page_hints or int(candidate["page_end"]) in page_hints):
                    score += 5.0
                score += keyword_hits(
                    "\n".join(
                        [
                            candidate.get("text", ""),
                            candidate.get("section_title") or "",
                            candidate.get("section_path") or "",
                            candidate.get("file_name", ""),
                        ]
                    ),
                    keywords,
                )
                if candidate.get("chunk_type") == "table_like" or candidate.get("element_type") == "table":
                    score += 0.5
                if candidate.get("element_type") == "figure":
                    score += 0.2
                scored.append((score, int(candidate["page_start"]), candidate))
            scored.sort(key=lambda item: (-item[0], item[1]))
            if scored:
                gold_chunks.append(scored[0][2])
            elif candidates:
                gold_chunks.append(candidates[0])

        gold_doc_ids = sorted({chunk["doc_id"] for chunk in gold_chunks})
        gold_page_nums = sorted({page for chunk in gold_chunks for page in range(int(chunk["page_start"]), int(chunk["page_end"]) + 1)})
        gold_chunk_ids = [chunk["chunk_id"] for chunk in gold_chunks]

        materialized.append(
            {
                "question_id": question_id,
                "question_type": row.get("question_type", "fact"),
                "intent": row.get("intent", row.get("question_type", "fact")),
                "industry": row.get("industry", "other"),
                "query": row["query"],
                "gold_doc_ids": gold_doc_ids,
                "gold_page_nums": gold_page_nums,
                "gold_chunk_ids": gold_chunk_ids,
                "target_doc_keys": target_doc_keys,
                "target_titles": [doc["short_title"] for doc in target_docs if doc is not None],
                "target_files": [doc["file_name"] for doc in target_docs if doc is not None],
                "gold_answer_hint": row.get("gold_answer_hint", ""),
            }
        )

    report = {
        "row_count": len(materialized) + len(missing_rows),
        "resolved_row_count": len(materialized),
        "missing_row_count": len(missing_rows),
        "industry_distribution": dict(sorted(Counter(row.get("industry", "other") for row in materialized).items())),
        "intent_distribution": dict(sorted(Counter(row.get("intent", row.get("question_type", "fact")) for row in materialized).items())),
        "missing_rows": missing_rows,
    }
    if report_output_path is not None:
        write_json(report_output_path, report)
    if missing_rows:
        raise ValueError(f"failed to resolve {len(missing_rows)} eval rows; see {report_output_path or 'manifest report'}")

    write_jsonl(output_path, materialized)
    return materialized


def _first_hit_rank(results: list[dict[str, Any]], gold_chunk_ids: set[str], gold_doc_ids: set[str], k: int) -> int | None:
    for rank, row in enumerate(results[:k], start=1):
        if row["chunk_id"] in gold_chunk_ids or row["doc_id"] in gold_doc_ids:
            return rank
    return None


def _metrics_for_results(results_by_query: list[dict[str, Any]]) -> dict[str, float]:
    total = max(len(results_by_query), 1)
    recall_5 = sum(1 for row in results_by_query if row["hit_rank_at_5"] is not None) / total
    recall_10 = sum(1 for row in results_by_query if row["hit_rank_at_10"] is not None) / total
    mrr_10 = sum((1 / row["hit_rank_at_10"]) if row["hit_rank_at_10"] is not None else 0.0 for row in results_by_query) / total
    return {
        "query_count": len(results_by_query),
        "Recall@5": round(recall_5, 4),
        "Recall@10": round(recall_10, 4),
        "MRR@10": round(mrr_10, 4),
    }


def evaluate_retrieval_methods(
    *,
    eval_rows: list[dict[str, Any]],
    method_results: dict[str, dict[str, list[dict[str, Any]]]],
    results_output_path: Path,
    summary_output_path: Path,
    markdown_output_path: Path,
    badcase_output_path: Path,
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    result_rows: list[dict[str, Any]] = []
    badcases: list[dict[str, Any]] = []

    for method_name, by_query in method_results.items():
        method_eval_rows: list[dict[str, Any]] = []
        for eval_row in eval_rows:
            results = by_query[eval_row["question_id"]]
            gold_chunk_ids = set(eval_row["gold_chunk_ids"])
            gold_doc_ids = set(eval_row["gold_doc_ids"])
            hit_rank_at_5 = _first_hit_rank(results, gold_chunk_ids, gold_doc_ids, 5)
            hit_rank_at_10 = _first_hit_rank(results, gold_chunk_ids, gold_doc_ids, 10)
            method_eval_rows.append(
                {
                    "question_id": eval_row["question_id"],
                    "question_type": eval_row["question_type"],
                    "intent": eval_row.get("intent", eval_row["question_type"]),
                    "industry": eval_row.get("industry", "other"),
                    "query": eval_row["query"],
                    "method": method_name,
                    "hit_rank_at_5": hit_rank_at_5,
                    "hit_rank_at_10": hit_rank_at_10,
                    "top_chunk_ids": [row["chunk_id"] for row in results[:10]],
                    "top_doc_ids": [row["doc_id"] for row in results[:10]],
                }
            )
            if hit_rank_at_10 is None:
                badcases.append(
                    {
                        "question_id": eval_row["question_id"],
                        "question_type": eval_row["question_type"],
                        "intent": eval_row.get("intent", eval_row["question_type"]),
                        "industry": eval_row.get("industry", "other"),
                        "query": eval_row["query"],
                        "method": method_name,
                        "gold_doc_ids": eval_row["gold_doc_ids"],
                        "gold_chunk_ids": eval_row["gold_chunk_ids"],
                        "target_doc_keys": eval_row.get("target_doc_keys", []),
                        "target_titles": eval_row.get("target_titles", []),
                        "top_chunk_ids": [row["chunk_id"] for row in results[:5]],
                        "top_doc_ids": [row["doc_id"] for row in results[:5]],
                    }
                )

        summary[method_name] = _metrics_for_results(method_eval_rows)
        result_rows.extend(method_eval_rows)

    if run_metadata is not None:
        summary["metadata"] = run_metadata
    write_json(summary_output_path, summary)
    write_jsonl(results_output_path, result_rows)
    write_jsonl(badcase_output_path, badcases)
    markdown_output_path.write_text(_to_markdown(summary), encoding="utf-8")
    return summary


def _to_markdown(summary: dict[str, Any]) -> str:
    lines = ["| Method | Query Count | Recall@5 | Recall@10 | MRR@10 |", "|---|---:|---:|---:|---:|"]
    for method_name, metrics in summary.items():
        if not isinstance(metrics, dict) or "query_count" not in metrics:
            continue
        lines.append(
            f"| {method_name} | {metrics['query_count']} | {metrics['Recall@5']:.4f} | {metrics['Recall@10']:.4f} | {metrics['MRR@10']:.4f} |"
        )
    return "\n".join(lines) + "\n"

