from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.evaluation.benchmark_assets import build_doc_manifest, short_title_from_file_name
from src.utils.io import read_jsonl, write_json, write_jsonl
from src.utils.text_utils import extract_numeric_tokens, extract_terms, normalize_for_match, normalize_text


def _normalize_answer_eval_profile(profile: str) -> str:
    return "historical-full-raw" if profile == "historical-full" else profile


def _materialize_answer_eval_seed(
    *,
    seed_path: Path,
    chunks: list[dict[str, Any]],
    chunk_lookup: dict[str, dict[str, Any]],
    docs_by_key: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    resolved_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    seed_rows = read_jsonl(seed_path)
    for row in seed_rows:
        question_id = row.get("question_id")
        if not question_id:
            raise ValueError(f"answer seed row missing question_id: {row}")

        target_doc_keys = list(row.get("target_doc_keys", []))
        target_docs = [docs_by_key.get(doc_key) for doc_key in target_doc_keys]
        missing_doc_keys = [doc_key for doc_key, doc in zip(target_doc_keys, target_docs) if doc is None]
        gold_chunk_ids = list(row.get("gold_chunk_ids", []))
        missing_chunk_ids = [chunk_id for chunk_id in gold_chunk_ids if chunk_id not in chunk_lookup]
        if missing_doc_keys or missing_chunk_ids:
            missing_rows.append(
                {
                    "question_id": question_id,
                    "query": row.get("query", ""),
                    "industry": row.get("industry", "other"),
                    "question_type": row.get("question_type", "fact"),
                    "intent": row.get("intent", row.get("question_type", "fact")),
                    "missing_doc_keys": missing_doc_keys,
                    "missing_chunk_ids": missing_chunk_ids,
                }
            )
            continue

        gold_chunks = [chunk_lookup[chunk_id] for chunk_id in gold_chunk_ids]
        resolved_rows.append(
            {
                "question_id": question_id,
                "query": row["query"],
                "question_type": row.get("question_type", "fact"),
                "intent": row.get("intent", row.get("question_type", "fact")),
                "industry": row.get("industry", "other"),
                "gold_answer": row.get("gold_answer", ""),
                "gold_doc_ids": [doc["doc_id"] for doc in target_docs if doc is not None],
                "gold_page_nums": sorted(
                    {
                        page
                        for chunk in gold_chunks
                        for page in range(int(chunk["page_start"]), int(chunk["page_end"]) + 1)
                    }
                ),
                "gold_chunk_ids": gold_chunk_ids,
                "must_abstain": bool(row.get("must_abstain", False)),
                "notes": row.get("review_notes", "curated_answer_seed"),
                "target_doc_keys": target_doc_keys,
                "target_titles": [doc["short_title"] for doc in target_docs if doc is not None],
                "target_files": [doc["file_name"] for doc in target_docs if doc is not None],
            }
        )
    return resolved_rows, missing_rows


def _row_conflict_key(row: dict[str, Any]) -> tuple[str, str]:
    return (
        normalize_for_match(row.get("query", "")),
        normalize_text(row.get("intent", row.get("question_type", "fact")) or "fact"),
    )


def _row_target_signature(row: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return (
        tuple(row.get("target_doc_keys", [])),
        tuple(row.get("gold_chunk_ids", [])),
    )


def _derive_answer_eval_profile_rows(
    rows: list[dict[str, Any]],
    *,
    benchmark_profile: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalized_profile = _normalize_answer_eval_profile(benchmark_profile or "default")
    if normalized_profile not in {"historical-full-core"}:
        return list(rows), {
            "benchmark_profile": normalized_profile,
            "dropped_row_count": 0,
            "dropped_conflict_group_count": 0,
            "dropped_question_ids": [],
        }

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_row_conflict_key(row)].append(row)

    filtered_rows: list[dict[str, Any]] = []
    dropped_question_ids: list[str] = []
    dropped_group_count = 0
    for group_rows in grouped.values():
        signatures = {_row_target_signature(row) for row in group_rows}
        if len(group_rows) > 1 and len(signatures) > 1:
            dropped_group_count += 1
            dropped_question_ids.extend(row["question_id"] for row in group_rows)
            continue
        filtered_rows.extend(group_rows)

    return filtered_rows, {
        "benchmark_profile": normalized_profile,
        "dropped_row_count": len(dropped_question_ids),
        "dropped_conflict_group_count": dropped_group_count,
        "dropped_question_ids": dropped_question_ids,
    }


def materialize_answer_eval_sets(
    *,
    dev_seed_path: Path,
    full_seed_path: Path | None,
    chunks: list[dict[str, Any]],
    dev_output_path: Path,
    full_output_path: Path,
    report_output_path: Path,
    requested_splits: tuple[str, ...] | None = None,
    benchmark_profile: str = "",
) -> dict[str, list[dict[str, Any]]]:
    chunk_lookup = {chunk["chunk_id"]: chunk for chunk in chunks}
    manifest = build_doc_manifest(chunks)
    docs_by_key = {row["doc_key"]: row for row in manifest}

    active_splits = tuple(requested_splits or ("dev", "full"))
    invalid_splits = [split for split in active_splits if split not in {"dev", "full"}]
    if invalid_splits:
        raise ValueError(f"Unsupported requested splits: {invalid_splits}")

    split_specs: list[tuple[str, Path, Path]] = []
    if "dev" in active_splits:
        split_specs.append(("dev", dev_seed_path, dev_output_path))
    if "full" in active_splits and full_seed_path is not None and full_seed_path.exists():
        split_specs.append(("full", full_seed_path, full_output_path))
    elif "full" in active_splits and full_seed_path is not None and not full_seed_path.exists():
        raise FileNotFoundError(f"Missing requested full answer seed: {full_seed_path}")
    elif "full" in active_splits and full_seed_path is None:
        raise FileNotFoundError("Requested split 'full' but no full_seed_path was provided.")

    materialized_sets: dict[str, list[dict[str, Any]]] = {}
    report_splits: dict[str, dict[str, Any]] = {}
    missing_total = 0
    for split_name, seed_path, output_path in split_specs:
        resolved_rows, missing_rows = _materialize_answer_eval_seed(
            seed_path=seed_path,
            chunks=chunks,
            chunk_lookup=chunk_lookup,
            docs_by_key=docs_by_key,
        )
        profile_for_split = (
            _normalize_answer_eval_profile(benchmark_profile)
            if split_name == "full"
            else "current-dev" if benchmark_profile == "current-dev" else ""
        )
        profile_rows, profile_report = _derive_answer_eval_profile_rows(
            resolved_rows,
            benchmark_profile=profile_for_split,
        )
        materialized_sets[split_name] = profile_rows
        write_jsonl(output_path, profile_rows)
        report_splits[split_name] = {
            "seed_path": str(seed_path),
            "row_count": len(resolved_rows) + len(missing_rows),
            "resolved_row_count": len(resolved_rows),
            "materialized_row_count": len(profile_rows),
            "missing_row_count": len(missing_rows),
            "benchmark_profile": profile_for_split or "default",
            "profile_filter": profile_report,
            "industry_distribution": dict(sorted(Counter(row.get("industry", "other") for row in profile_rows).items())),
            "intent_distribution": dict(sorted(Counter(row.get("intent", row.get("question_type", "fact")) for row in profile_rows).items())),
            "missing_rows": missing_rows,
        }
        missing_total += len(missing_rows)

    report = {
        "requested_splits": list(active_splits),
        "available_splits": sorted(materialized_sets),
        "splits": report_splits,
    }
    write_json(report_output_path, report)
    if missing_total:
        raise ValueError(f"failed to resolve {missing_total} answer eval rows; see {report_output_path}")
    return materialized_sets


def _missing_result_row(question_id: str) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "failed": True,
        "error_type": "MissingResult",
        "error_message": "No result row was recorded for this question.",
        "abstained": False,
        "abstain_reason": None,
        "abstain_gate": "error",
        "final_answer": "",
        "fact_subtype": "",
        "answer_source": "error",
        "matched_numeric_tokens": [],
        "title_resolved_from": "",
        "query_domain_bucket": "",
        "selected_domain_buckets": [],
        "doc_guard_triggered": False,
        "citation_rule_applied": "default",
        "support_filter_applied": "default",
        "citations": [],
        "retrieval_scores": {},
        "generation_latency_ms": 0.0,
        "timings": {},
        "support_validation": {"supported": False},
    }


def _merge_eval_row(
    eval_row: dict[str, Any],
    result: dict[str, Any],
    *,
    answer_semantic_hit: bool,
    citation_doc_hit: bool,
    citation_span_hit: bool,
    support_hit: bool,
    unsupported: bool,
) -> dict[str, Any]:
    timings = result.get("timings", {}) or {}
    return {
        "question_id": eval_row["question_id"],
        "query": eval_row["query"],
        "question_type": eval_row["question_type"],
        "intent": eval_row.get("intent", eval_row["question_type"]),
        "industry": eval_row.get("industry", "other"),
        "must_abstain": bool(eval_row["must_abstain"]),
        "answer_semantic_hit": answer_semantic_hit,
        "citation_doc_hit": citation_doc_hit,
        "citation_span_hit": citation_span_hit,
        "support_hit": support_hit,
        "answer_hit": answer_semantic_hit,
        "citation_hit": citation_doc_hit,
        "unsupported": unsupported,
        "abstained": bool(result.get("abstained", False)),
        "abstain_reason": result.get("abstain_reason"),
        "abstain_gate": result.get("abstain_gate"),
        "final_answer": result.get("final_answer", ""),
        "gold_answer": eval_row.get("gold_answer", ""),
        "fact_subtype": result.get("fact_subtype", ""),
        "answer_source": result.get("answer_source"),
        "fallback_used": bool(result.get("fallback_used", False)),
        "fallback_reason": result.get("fallback_reason"),
        "matched_numeric_tokens": result.get("matched_numeric_tokens", []),
        "title_resolved_from": result.get("title_resolved_from", ""),
        "query_domain_bucket": result.get("query_domain_bucket", ""),
        "selected_domain_buckets": result.get("selected_domain_buckets", []),
        "doc_guard_triggered": result.get("doc_guard_triggered", False),
        "citation_rule_applied": result.get("citation_rule_applied", "default"),
        "support_filter_applied": result.get("support_filter_applied", "default"),
        "citations": result.get("citations", []),
        "gold_chunk_ids": eval_row.get("gold_chunk_ids", []),
        "target_doc_keys": eval_row.get("target_doc_keys", []),
        "target_titles": eval_row.get("target_titles", []),
        "retrieval_scores": result.get("retrieval_scores", {}),
        "generation_latency_ms": result.get("generation_latency_ms", 0.0),
        "retrieval_latency_ms": timings.get("retrieval_latency_ms", 0.0),
        "rerank_latency_ms": timings.get("rerank_latency_ms", 0.0),
        "end_to_end_latency_ms": timings.get("end_to_end_latency_ms", 0.0),
        "failed": bool(result.get("failed", False)),
        "error_type": result.get("error_type", ""),
        "error_message": result.get("error_message", ""),
    }


def evaluate_answer_results(
    *,
    eval_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    chunks_path: Path,
    results_output_path: Path,
    summary_output_path: Path,
    markdown_output_path: Path,
    latency_output_path: Path,
    answer_badcase_output_path: Path,
    abstain_badcase_output_path: Path,
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    write_jsonl(results_output_path, result_rows)
    chunk_lookup = {row["chunk_id"]: row for row in read_jsonl(chunks_path)}
    results_by_id = {row["question_id"]: row for row in result_rows}
    answerable_rows: list[dict[str, Any]] = []
    abstain_rows: list[dict[str, Any]] = []
    answer_badcases: list[dict[str, Any]] = []
    abstain_badcases: list[dict[str, Any]] = []

    for eval_row in eval_rows:
        result = results_by_id.get(eval_row["question_id"]) or _missing_result_row(eval_row["question_id"])
        if result.get("failed"):
            merged = _merge_eval_row(
                eval_row,
                result,
                answer_semantic_hit=False,
                citation_doc_hit=False,
                citation_span_hit=False,
                support_hit=False,
                unsupported=False,
            )
            if eval_row["must_abstain"]:
                abstain_rows.append(merged)
                abstain_badcases.append(merged)
            else:
                answerable_rows.append(merged)
                answer_badcases.append(merged)
            continue

        citation_doc_hit = compute_citation_doc_hit(eval_row, result)
        citation_span_hit = compute_citation_span_hit(eval_row, result, chunk_lookup)
        answer_semantic_hit = compute_answer_semantic_hit(eval_row, result, chunk_lookup)
        support_hit = compute_support_hit(
            eval_row,
            result,
            answer_semantic_hit=answer_semantic_hit,
            citation_doc_hit=citation_doc_hit,
        )
        unsupported = bool(not result.get("abstained", False) and not support_hit)
        merged = _merge_eval_row(
            eval_row,
            result,
            answer_semantic_hit=answer_semantic_hit,
            citation_doc_hit=citation_doc_hit,
            citation_span_hit=citation_span_hit,
            support_hit=support_hit,
            unsupported=unsupported,
        )
        if eval_row["must_abstain"]:
            abstain_rows.append(merged)
            if not result.get("abstained", False):
                abstain_badcases.append(merged)
            continue

        answerable_rows.append(merged)
        if not answer_semantic_hit or not citation_doc_hit or not citation_span_hit or not support_hit:
            answer_badcases.append(merged)

    summary = build_answer_eval_summary(
        answerable_rows=answerable_rows,
        abstain_rows=abstain_rows,
        result_rows=result_rows,
        run_metadata=run_metadata,
    )
    summary["badcase_breakdown"] = build_badcase_breakdown(answer_badcases)
    latency_summary = build_latency_summary(result_rows)

    write_json(summary_output_path, summary)
    markdown_output_path.write_text(to_markdown(summary, latency_summary), encoding="utf-8")
    write_json(latency_output_path, latency_summary)
    write_jsonl(answer_badcase_output_path, answer_badcases)
    write_jsonl(abstain_badcase_output_path, abstain_badcases)
    return summary


def _citation_page_overlap(eval_row: dict[str, Any], citation: dict[str, Any]) -> bool:
    gold_pages = set(eval_row.get("gold_page_nums", []))
    return any(page in gold_pages for page in range(int(citation["page_start"]), int(citation["page_end"]) + 1))


def _matched_doc_ids(
    eval_row: dict[str, Any],
    result_row: dict[str, Any],
    *,
    require_span_overlap: bool,
) -> set[str]:
    if result_row.get("failed") or result_row.get("abstained"):
        return set()
    citations = result_row.get("citations", [])
    if not citations:
        return set()
    gold_doc_ids = set(eval_row.get("gold_doc_ids", []))
    gold_chunk_ids = set(eval_row.get("gold_chunk_ids", []))
    matched_docs: set[str] = set()
    for citation in citations:
        doc_hit = citation["doc_id"] in gold_doc_ids or citation["chunk_id"] in gold_chunk_ids
        if not doc_hit:
            continue
        if require_span_overlap and citation["chunk_id"] not in gold_chunk_ids and not _citation_page_overlap(eval_row, citation):
            continue
        matched_docs.add(citation["doc_id"])
    return matched_docs


def compute_citation_doc_hit(eval_row: dict[str, Any], result_row: dict[str, Any]) -> bool:
    matched_docs = _matched_doc_ids(eval_row, result_row, require_span_overlap=False)
    if eval_row["question_type"] in {"comparison", "inductive"}:
        required = 2 if eval_row["question_type"] == "comparison" else min(2, max(len(set(eval_row.get("gold_doc_ids", []))), 1))
        return len(matched_docs) >= required
    return bool(matched_docs)


def _citation_section_overlap(
    eval_row: dict[str, Any],
    citation: dict[str, Any],
    chunk_lookup: dict[str, dict[str, Any]],
) -> bool:
    gold_chunk_ids = set(eval_row.get("gold_chunk_ids", []))
    gold_sections = {
        normalize_text(chunk_lookup[chunk_id].get("section_title", "") or chunk_lookup[chunk_id].get("section_path", "") or "")
        for chunk_id in gold_chunk_ids
        if chunk_id in chunk_lookup
    }
    gold_sections.discard("")
    citation_section = normalize_text(citation.get("section_title", "") or "")
    if citation_section and citation_section in gold_sections:
        return True

    gold_pages = set(eval_row.get("gold_page_nums", []))
    citation_pages = set(range(int(citation["page_start"]), int(citation["page_end"]) + 1))
    if eval_row.get("intent") == "report_lookup" and citation.get("doc_id") in set(eval_row.get("gold_doc_ids", [])):
        if gold_pages and min(gold_pages) <= 2 and citation_pages and min(citation_pages) <= 2:
            return True
    return False


def compute_citation_span_hit(
    eval_row: dict[str, Any],
    result_row: dict[str, Any],
    chunk_lookup: dict[str, dict[str, Any]],
) -> bool:
    if result_row.get("failed") or result_row.get("abstained"):
        return False
    citations = result_row.get("citations", [])
    if not citations:
        return False
    matched_docs: set[str] = set()
    gold_doc_ids = set(eval_row.get("gold_doc_ids", []))
    gold_chunk_ids = set(eval_row.get("gold_chunk_ids", []))
    for citation in citations:
        doc_hit = citation["doc_id"] in gold_doc_ids or citation["chunk_id"] in gold_chunk_ids
        if not doc_hit:
            continue
        if (
            citation["chunk_id"] in gold_chunk_ids
            or _citation_page_overlap(eval_row, citation)
            or _citation_section_overlap(eval_row, citation, chunk_lookup)
        ):
            matched_docs.add(citation["doc_id"])
    if eval_row["question_type"] in {"comparison", "inductive"}:
        required = 2 if eval_row["question_type"] == "comparison" else min(2, max(len(set(eval_row.get("gold_doc_ids", []))), 1))
        return len(matched_docs) >= required
    return bool(matched_docs)


def compute_citation_hit(eval_row: dict[str, Any], result_row: dict[str, Any]) -> bool:
    return compute_citation_doc_hit(eval_row, result_row)


def compute_support_hit(
    eval_row: dict[str, Any],
    result_row: dict[str, Any],
    *,
    answer_semantic_hit: bool,
    citation_doc_hit: bool,
) -> bool:
    support_validation = result_row.get("support_validation", {}) or {}
    if support_validation.get("domain_mismatch"):
        return False
    if eval_row.get("intent", eval_row["question_type"]) == "report_lookup":
        if answer_semantic_hit and citation_doc_hit:
            return True
    return bool(support_validation.get("supported", False))


def _term_overlap_ratio(left: str, right: str, *, left_top_k: int = 10, right_top_k: int = 12) -> float:
    left_terms = set(merged_terms(left, top_k=left_top_k))
    right_terms = set(merged_terms(right, top_k=right_top_k))
    if not left_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms)


def _gold_text(eval_row: dict[str, Any], chunk_lookup: dict[str, dict[str, Any]]) -> str:
    return normalize_text(
        f"{eval_row.get('gold_answer', '')}\n"
        + "\n".join(
            chunk_lookup[chunk_id]["text"]
            for chunk_id in eval_row.get("gold_chunk_ids", [])
            if chunk_id in chunk_lookup
        )
    )


def _citation_support_text(eval_row: dict[str, Any], result_row: dict[str, Any], chunk_lookup: dict[str, dict[str, Any]]) -> str:
    texts: list[str] = []
    gold_doc_ids = set(eval_row.get("gold_doc_ids", []))
    gold_chunk_ids = set(eval_row.get("gold_chunk_ids", []))
    for citation in result_row.get("citations", []):
        if citation.get("doc_id") not in gold_doc_ids and citation.get("chunk_id") not in gold_chunk_ids:
            continue
        if citation.get("snippet"):
            texts.append(normalize_text(citation["snippet"]))
        chunk = chunk_lookup.get(citation.get("chunk_id", ""))
        if chunk is not None:
            texts.append(normalize_text(chunk.get("text", "")))
    return normalize_text("\n".join(texts))


def _semantic_fact_hit(*, gold_text: str, predicted_text: str) -> bool:
    return _term_overlap_ratio(gold_text, predicted_text, left_top_k=12, right_top_k=12) >= 0.25


def _value_fact_hit(*, gold_text: str, predicted_text: str, support_text: str = "") -> bool:
    gold_numeric_tokens = {normalize_for_match(token) for token in extract_numeric_tokens(gold_text) if normalize_for_match(token)}
    predicted_numeric_tokens = {normalize_for_match(token) for token in extract_numeric_tokens(predicted_text) if normalize_for_match(token)}
    support_numeric_tokens = {normalize_for_match(token) for token in extract_numeric_tokens(support_text) if normalize_for_match(token)}
    if predicted_numeric_tokens and (predicted_numeric_tokens & gold_numeric_tokens or predicted_numeric_tokens & support_numeric_tokens):
        return True
    if not predicted_numeric_tokens:
        return _term_overlap_ratio(gold_text, predicted_text, left_top_k=12, right_top_k=12) >= 0.25
    return _term_overlap_ratio(f"{gold_text}\n{support_text}", predicted_text, left_top_k=12, right_top_k=12) >= 0.35


def compute_answer_semantic_hit(
    eval_row: dict[str, Any],
    result_row: dict[str, Any],
    chunk_lookup: dict[str, dict[str, Any]],
) -> bool:
    if result_row.get("failed") or result_row.get("abstained"):
        return False
    combined_prediction = normalize_text(f"{result_row.get('final_answer', '')}\n{result_row.get('evidence_summary', '')}")
    intent = eval_row.get("intent", eval_row["question_type"])
    if intent == "report_lookup":
        target_titles = eval_row.get("target_titles") or [eval_row.get("gold_answer", "")]
        return title_match(combined_prediction, target_titles)

    gold_text = _gold_text(eval_row, chunk_lookup)
    if intent == "numeric_fact":
        fact_subtype = result_row.get("fact_subtype", "semantic_fact")
        if fact_subtype == "value_fact":
            return _value_fact_hit(
                gold_text=gold_text,
                predicted_text=normalize_text(result_row.get("final_answer", "")),
                support_text=_citation_support_text(eval_row, result_row, chunk_lookup),
            )
        return _semantic_fact_hit(
            gold_text=gold_text,
            predicted_text=combined_prediction,
        )

    overlap_ratio = _term_overlap_ratio(gold_text, combined_prediction)
    threshold = 0.25 if intent == "inductive" else 0.3
    return overlap_ratio >= threshold


def compute_answer_hit(
    eval_row: dict[str, Any],
    result_row: dict[str, Any],
    chunk_lookup: dict[str, dict[str, Any]],
    *,
    citation_hit: bool,
) -> bool:
    del citation_hit
    return compute_answer_semantic_hit(eval_row, result_row, chunk_lookup)


def build_answer_eval_summary(
    *,
    answerable_rows: list[dict[str, Any]],
    abstain_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    answer_denominator = max(len(answerable_rows), 1)
    predicted_abstains = sum(1 for row in answerable_rows if row["abstained"]) + sum(1 for row in abstain_rows if row["abstained"])
    actual_abstains = len(abstain_rows)
    true_positive_abstains = sum(1 for row in abstain_rows if row["abstained"])
    answer_semantic_hit_rate = sum(1 for row in answerable_rows if row["answer_semantic_hit"]) / answer_denominator
    citation_doc_hit_rate = sum(1 for row in answerable_rows if row["citation_doc_hit"]) / answer_denominator
    citation_span_hit_rate = sum(1 for row in answerable_rows if row["citation_span_hit"]) / answer_denominator
    support_hit_rate = sum(1 for row in answerable_rows if row["support_hit"]) / answer_denominator
    unsupported_answer_count = sum(1 for row in answerable_rows if row["unsupported"])
    abstain_precision = 1.0 if predicted_abstains == 0 else true_positive_abstains / predicted_abstains
    abstain_recall = 1.0 if actual_abstains == 0 else true_positive_abstains / actual_abstains
    failed_query_count = sum(1 for row in answerable_rows + abstain_rows if row.get("failed"))
    total_queries = len(answerable_rows) + len(abstain_rows)
    fallback_count = sum(1 for row in result_rows if row.get("fallback_used"))
    summary = {
        "query_count": len(answerable_rows) + len(abstain_rows),
        "answerable_query_count": len(answerable_rows),
        "must_abstain_query_count": len(abstain_rows),
        "industry_distribution": dict(sorted(Counter(row.get("industry", "other") for row in answerable_rows).items())),
        "intent_distribution": dict(sorted(Counter(row.get("intent", row.get("question_type", "fact")) for row in answerable_rows).items())),
        "Answer Semantic Hit Rate": round(answer_semantic_hit_rate, 4),
        "Citation Doc Hit Rate": round(citation_doc_hit_rate, 4),
        "Citation Span Hit Rate": round(citation_span_hit_rate, 4),
        "Support Hit Rate": round(support_hit_rate, 4),
        "Answer Hit Rate": round(answer_semantic_hit_rate, 4),
        "Citation Hit Rate": round(citation_doc_hit_rate, 4),
        "Abstain Precision": round(abstain_precision, 4),
        "Abstain Recall": round(abstain_recall, 4),
        "Unsupported Answer Count": unsupported_answer_count,
        "Fallback Count": fallback_count,
        "Fallback Rate": round(fallback_count / max(total_queries, 1), 4),
        "LLM Answer Count": sum(1 for row in result_rows if row.get("answer_source") == "llm"),
        "Deterministic Answer Count": sum(1 for row in result_rows if row.get("answer_source") == "deterministic"),
        "Failed Query Count": failed_query_count,
    }
    if run_metadata is not None:
        summary["metadata"] = run_metadata
    return summary


def build_badcase_breakdown(answer_badcases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter: Counter[tuple[str, str, str, bool, bool, str, bool, bool, bool, bool, str]] = Counter(
        (
            row.get("intent", row.get("question_type", "fact")),
            row.get("fact_subtype", ""),
            row.get("abstain_reason") or "none",
            bool(row.get("doc_guard_triggered", False)),
            bool(row.get("failed", False)),
            row.get("error_type", ""),
            bool(row.get("answer_semantic_hit", False)),
            bool(row.get("citation_doc_hit", False)),
            bool(row.get("citation_span_hit", False)),
            bool(row.get("support_hit", False)),
            row.get("fallback_reason") or "none",
        )
        for row in answer_badcases
    )
    return [
        {
            "intent": intent,
            "fact_subtype": fact_subtype,
            "abstain_reason": abstain_reason,
            "doc_guard_triggered": doc_guard_triggered,
            "failed": failed,
            "error_type": error_type,
            "answer_semantic_hit": answer_semantic_hit,
            "citation_doc_hit": citation_doc_hit,
            "citation_span_hit": citation_span_hit,
            "support_hit": support_hit,
            "fallback_reason": fallback_reason,
            "count": count,
        }
        for (
            intent,
            fact_subtype,
            abstain_reason,
            doc_guard_triggered,
            failed,
            error_type,
            answer_semantic_hit,
            citation_doc_hit,
            citation_span_hit,
            support_hit,
            fallback_reason,
        ), count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def build_latency_summary(result_rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed_rows = [row for row in result_rows if not row.get("failed")]
    total = max(len(completed_rows), 1)
    return {
        "query_count": len(result_rows),
        "Avg Retrieval Latency": round(sum(row.get("timings", {}).get("retrieval_latency_ms", 0.0) for row in completed_rows) / total, 2),
        "Avg Rerank Latency": round(sum(row.get("timings", {}).get("rerank_latency_ms", 0.0) for row in completed_rows) / total, 2),
        "Avg Generation Latency": round(sum(row.get("generation_latency_ms", 0.0) for row in completed_rows) / total, 2),
        "End-to-End Latency": round(sum(row.get("timings", {}).get("end_to_end_latency_ms", 0.0) for row in completed_rows) / total, 2),
    }


def to_markdown(summary: dict[str, Any], latency_summary: dict[str, Any]) -> str:
    lines = ["| Metric | Value |", "|---|---:|"]
    for key, value in summary.items():
        if key == "metadata" and isinstance(value, dict):
            for meta_key, meta_value in value.items():
                lines.append(f"| metadata.{meta_key} | {meta_value} |")
            continue
        lines.append(f"| {key} | {value} |")
    for key, value in latency_summary.items():
        if key == "query_count":
            continue
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines) + "\n"


def title_match(prediction: str, target_titles: list[str]) -> bool:
    normalized_prediction = normalize_for_match(prediction)
    for title in target_titles:
        cleaned_title = short_title_from_file_name(title)
        normalized_title = normalize_for_match(cleaned_title)
        if normalized_title and normalized_title in normalized_prediction:
            return True
        title_terms = [term for term in extract_terms(cleaned_title, top_k=6) if len(term) >= 2]
        if title_terms and sum(normalize_for_match(term) in normalized_prediction for term in title_terms) >= min(2, len(title_terms)):
            return True
    return False


def merged_terms(text: str, *, top_k: int = 6) -> list[str]:
    return extract_terms(text, top_k=top_k)
