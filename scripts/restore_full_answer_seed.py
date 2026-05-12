from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from src.evaluation.benchmark_assets import build_doc_manifest, short_title_from_file_name
from src.utils.io import read_jsonl, write_json, write_jsonl
from src.utils.text_utils import extract_terms, guess_industry, normalize_for_match, normalize_text, strip_file_extension


REPORT_LOOKUP_HINTS = ("哪份报告", "哪份研报", "哪家券商", "是关于", "聚焦")
SEMICONDUCTOR_TERMS = ("半导体", "晶圆", "芯片", "存储", "电子", "代工", "semicon", "gpu", "token")
AGRICULTURE_TERMS = ("农林牧渔", "生猪", "猪价", "养殖", "种业", "农业", "牧渔")
HEALTHCARE_TERMS = ("who", "新冠", "呼吸道", "风险评估", "疫苗", "病毒", "临床", "肺炎")
TRAILING_DATE_RE = re.compile(r"[-_](?:20\d{2}-\d{2}-\d{2}|20\d{6}|\d{6})$")
CHUNK_ID_SUFFIX_RE = re.compile(r"-(fixed_window|title_aware|table_protected)-([cp]\d{4})$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore historical full answer-eval seed from benchmark artifacts.")
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("artifacts/remote_20260401/outputs_reports/answer_eval_results_full.jsonl"),
    )
    parser.add_argument(
        "--answer-badcases-path",
        type=Path,
        default=Path("artifacts/remote_20260401/outputs_badcases/answer_badcases_full.jsonl"),
    )
    parser.add_argument(
        "--abstain-badcases-path",
        type=Path,
        default=Path("artifacts/remote_20260401/outputs_badcases/abstain_badcases_full.jsonl"),
    )
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=Path("data/chunks/chunks.jsonl"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/eval_set/answer_eval_seed_full_draft.jsonl"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("data/eval_set/answer_eval_seed_full_restore_report.json"),
    )
    return parser.parse_args()


def infer_intent(query: str, question_type: str) -> str:
    normalized = normalize_text(query)
    if question_type == "comparison":
        return "comparison"
    if question_type == "inductive":
        return "inductive"
    if any(hint in normalized for hint in REPORT_LOOKUP_HINTS):
        return "report_lookup"
    return "numeric_fact"


def infer_industry(query: str, candidate_files: list[str]) -> str:
    for file_name in candidate_files:
        industry = guess_industry(file_name)
        if industry != "other":
            return industry

    lowered = normalize_text(query).lower()
    if any(term in lowered for term in SEMICONDUCTOR_TERMS):
        return "semiconductor"
    if any(term in lowered for term in AGRICULTURE_TERMS):
        return "agriculture"
    if any(term in lowered for term in HEALTHCARE_TERMS):
        return "healthcare"
    return "other"


def _unique_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _relaxed_title_variants(text: str) -> list[str]:
    stem = strip_file_extension(Path(text).name)
    raw_variants = {
        stem,
        short_title_from_file_name(text),
    }
    for variant in list(raw_variants):
        raw_variants.add(TRAILING_DATE_RE.sub("", variant).strip("-_ "))
        for separator in ("：", ":"):
            if separator in variant:
                suffix = variant.split(separator, 1)[1].strip()
                raw_variants.add(suffix)
                raw_variants.add(TRAILING_DATE_RE.sub("", suffix).strip("-_ "))
    normalized = []
    for variant in raw_variants:
        token = normalize_for_match(variant)
        if len(token) >= 6 and not token.isdigit():
            normalized.append(token)
    return _unique_keep_order(normalized)


def build_manifest_lookups(
    chunks: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
]:
    manifest = build_doc_manifest(chunks)
    by_doc_id = {row["doc_id"]: row for row in manifest}
    by_file_name = {row["file_name"]: row for row in manifest}
    by_short_title: dict[str, list[dict[str, Any]]] = {}
    by_variant: dict[str, list[dict[str, Any]]] = {}
    chunks_by_doc_id: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        chunks_by_doc_id.setdefault(chunk["doc_id"], []).append(chunk)
    for row in manifest:
        by_short_title.setdefault(normalize_for_match(row["short_title"]), []).append(row)
        for variant in _relaxed_title_variants(row["file_name"]) + _relaxed_title_variants(row["short_title"]):
            by_variant.setdefault(variant, []).append(row)
    return manifest, by_doc_id, by_file_name, by_short_title, by_variant, chunks_by_doc_id


def resolve_manifest_doc(
    *,
    manifest: list[dict[str, Any]],
    doc_id: str | None,
    file_name: str | None,
    by_doc_id: dict[str, dict[str, Any]],
    by_file_name: dict[str, dict[str, Any]],
    by_short_title: dict[str, list[dict[str, Any]]],
    by_variant: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    if doc_id and doc_id in by_doc_id:
        return by_doc_id[doc_id]
    if file_name and file_name in by_file_name:
        return by_file_name[file_name]
    if file_name:
        variants = _relaxed_title_variants(file_name)
        candidate_docs: dict[str, dict[str, Any]] = {}
        for variant in variants:
            for match in by_variant.get(variant, []):
                candidate_docs[match["doc_id"]] = match
        for manifest_doc in manifest:
            manifest_variants = _relaxed_title_variants(manifest_doc["file_name"]) + _relaxed_title_variants(manifest_doc["short_title"])
            for variant in variants:
                if any(variant in manifest_variant or manifest_variant in variant for manifest_variant in manifest_variants):
                    candidate_docs[manifest_doc["doc_id"]] = manifest_doc
                    break
        if candidate_docs:
            ref_terms = [normalize_for_match(term) for term in extract_terms(file_name, top_k=12) if normalize_for_match(term)]

            def _score(doc: dict[str, Any]) -> tuple[int, int]:
                doc_text = normalize_for_match(f"{doc['file_name']}\n{doc['short_title']}")
                overlap = sum(term in doc_text for term in ref_terms)
                strongest_variant = max(
                    (
                        min(len(variant), len(manifest_variant))
                        for variant in variants
                        for manifest_variant in (_relaxed_title_variants(doc["file_name"]) + _relaxed_title_variants(doc["short_title"]))
                        if variant in manifest_variant or manifest_variant in variant
                    ),
                    default=0,
                )
                return overlap, strongest_variant

            return max(candidate_docs.values(), key=_score)
    return None


def candidate_docs_from_result(
    row: dict[str, Any],
    *,
    manifest: list[dict[str, Any]],
    by_doc_id: dict[str, dict[str, Any]],
    by_file_name: dict[str, dict[str, Any]],
    by_short_title: dict[str, list[dict[str, Any]]],
    by_variant: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    raw_refs: list[tuple[str | None, str | None]] = []
    for citation in row.get("citations", []):
        raw_refs.append((citation.get("doc_id"), citation.get("file_name")))
    for evidence in row.get("selected_evidence", []):
        raw_refs.append((evidence.get("doc_id"), evidence.get("file_name")))

    counts: Counter[str] = Counter()
    for doc_id, file_name in raw_refs:
        manifest_doc = resolve_manifest_doc(
            manifest=manifest,
            doc_id=doc_id,
            file_name=file_name,
            by_doc_id=by_doc_id,
            by_file_name=by_file_name,
            by_short_title=by_short_title,
            by_variant=by_variant,
        )
        if manifest_doc is None:
            continue
        counts[manifest_doc["doc_id"]] += 1
        candidates[manifest_doc["doc_id"]] = manifest_doc

    query_terms = extract_terms(f"{row.get('query', '')}\n{row.get('final_answer', '')}", top_k=10)

    def _score(doc: dict[str, Any]) -> tuple[int, int]:
        title_text = normalize_for_match(f"{doc['short_title']}\n{doc['file_name']}")
        overlap = sum(normalize_for_match(term) in title_text for term in query_terms)
        return counts[doc["doc_id"]], overlap

    return sorted(candidates.values(), key=_score, reverse=True)


def candidate_chunk_ids_from_result(
    row: dict[str, Any],
    *,
    chunk_lookup: dict[str, dict[str, Any]],
    target_docs: list[dict[str, Any]],
    intent: str,
    chunks_by_doc_id: dict[str, list[dict[str, Any]]],
    manifest: list[dict[str, Any]],
    by_doc_id: dict[str, dict[str, Any]],
    by_file_name: dict[str, dict[str, Any]],
    by_short_title: dict[str, list[dict[str, Any]]],
    by_variant: dict[str, list[dict[str, Any]]],
) -> list[str]:
    target_doc_ids = {doc["doc_id"] for doc in target_docs}
    candidate_chunk_ids: list[str] = []
    for citation in row.get("citations", []):
        chunk_id = citation.get("chunk_id", "")
        remapped_ids = remap_historical_chunk_ids(
            [chunk_id],
            chunk_lookup=chunk_lookup,
            chunks_by_doc_id=chunks_by_doc_id,
            manifest=manifest,
            by_doc_id=by_doc_id,
            by_file_name=by_file_name,
            by_short_title=by_short_title,
            by_variant=by_variant,
        )
        for remapped_id in remapped_ids:
            chunk = chunk_lookup.get(remapped_id)
            if chunk is not None and chunk["doc_id"] in target_doc_ids:
                candidate_chunk_ids.append(remapped_id)
    if not candidate_chunk_ids:
        if intent == "report_lookup":
            candidate_chunk_ids = [doc["summary_chunk_id"] for doc in target_docs[:1]]
        elif intent == "comparison":
            candidate_chunk_ids = [doc["summary_chunk_id"] for doc in target_docs[:2]]
        elif intent == "inductive":
            candidate_chunk_ids = [doc["summary_chunk_id"] for doc in target_docs[:3]]
        else:
            candidate_chunk_ids = [doc["fact_chunk_id"] for doc in target_docs[:1]]
    return _unique_keep_order(candidate_chunk_ids)


def target_doc_keys_from_gold_chunk_ids(gold_chunk_ids: list[str], *, chunk_lookup: dict[str, dict[str, Any]], by_doc_id: dict[str, dict[str, Any]]) -> list[str]:
    target_doc_keys: list[str] = []
    for chunk_id in gold_chunk_ids:
        chunk = chunk_lookup.get(chunk_id)
        if chunk is None:
            continue
        manifest_doc = by_doc_id.get(chunk["doc_id"])
        if manifest_doc is None:
            continue
        target_doc_keys.append(manifest_doc["doc_key"])
    return _unique_keep_order(target_doc_keys)


def remap_historical_chunk_ids(
    chunk_ids: list[str],
    *,
    chunk_lookup: dict[str, dict[str, Any]],
    chunks_by_doc_id: dict[str, list[dict[str, Any]]],
    manifest: list[dict[str, Any]],
    by_doc_id: dict[str, dict[str, Any]],
    by_file_name: dict[str, dict[str, Any]],
    by_short_title: dict[str, list[dict[str, Any]]],
    by_variant: dict[str, list[dict[str, Any]]],
) -> list[str]:
    remapped: list[str] = []
    for chunk_id in chunk_ids:
        if chunk_id in chunk_lookup:
            remapped.append(chunk_id)
            continue
        match = CHUNK_ID_SUFFIX_RE.search(chunk_id)
        if not match:
            continue
        suffix = f"-{match.group(1)}-{match.group(2)}"
        historical_doc_label = chunk_id[: -len(suffix)]
        manifest_doc = resolve_manifest_doc(
            manifest=manifest,
            doc_id=None,
            file_name=historical_doc_label,
            by_doc_id=by_doc_id,
            by_file_name=by_file_name,
            by_short_title=by_short_title,
            by_variant=by_variant,
        )
        if manifest_doc is None:
            continue
        for candidate in chunks_by_doc_id.get(manifest_doc["doc_id"], []):
            if str(candidate["chunk_id"]).endswith(suffix):
                remapped.append(candidate["chunk_id"])
                break
    return _unique_keep_order(remapped)


def build_inferred_row(
    row: dict[str, Any],
    *,
    manifest: list[dict[str, Any]],
    by_doc_id: dict[str, dict[str, Any]],
    by_file_name: dict[str, dict[str, Any]],
    by_short_title: dict[str, list[dict[str, Any]]],
    by_variant: dict[str, list[dict[str, Any]]],
    chunk_lookup: dict[str, dict[str, Any]],
    chunks_by_doc_id: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    question_id = row["question_id"]
    query = row["query"]
    question_type = row["question_type"]
    intent = infer_intent(query, question_type)
    must_abstain = "_abstain_" in question_id

    target_docs = candidate_docs_from_result(
        row,
        manifest=manifest,
        by_doc_id=by_doc_id,
        by_file_name=by_file_name,
        by_short_title=by_short_title,
        by_variant=by_variant,
    )
    if question_type == "fact":
        target_docs = target_docs[:1]
    elif question_type == "comparison":
        target_docs = target_docs[:2]
    else:
        target_docs = target_docs[:3]

    candidate_files = [doc["file_name"] for doc in target_docs]
    industry = infer_industry(query, candidate_files)
    target_doc_keys = [doc["doc_key"] for doc in target_docs]
    gold_chunk_ids = [] if must_abstain else candidate_chunk_ids_from_result(
        row,
        chunk_lookup=chunk_lookup,
        target_docs=target_docs,
        intent=intent,
        chunks_by_doc_id=chunks_by_doc_id,
        manifest=manifest,
        by_doc_id=by_doc_id,
        by_file_name=by_file_name,
        by_short_title=by_short_title,
        by_variant=by_variant,
    )
    gold_answer = "" if must_abstain else normalize_text(row.get("final_answer", ""))

    draft_row = {
        "question_id": question_id,
        "query": query,
        "question_type": question_type,
        "intent": intent,
        "industry": industry,
        "target_doc_keys": target_doc_keys,
        "gold_answer": gold_answer,
        "gold_chunk_ids": gold_chunk_ids,
        "must_abstain": must_abstain,
        "review_notes": "manual_review_required_from_historical_citations",
        "manual_review_required": True,
        "source_label": "artifact_citation_inferred",
    }
    report_row = {
        "question_id": question_id,
        "source_label": "artifact_citation_inferred",
        "manual_review_required": True,
        "question_type": question_type,
        "intent": intent,
        "industry": industry,
        "target_doc_keys": target_doc_keys,
        "target_titles": [doc["short_title"] for doc in target_docs],
        "candidate_gold_chunk_ids": gold_chunk_ids,
        "candidate_gold_answer": gold_answer,
    }
    return draft_row, report_row


def main() -> int:
    args = parse_args()
    results_rows = read_jsonl(args.results_path)
    answer_badcases = {row["question_id"]: row for row in read_jsonl(args.answer_badcases_path)}
    abstain_badcases = {row["question_id"]: row for row in read_jsonl(args.abstain_badcases_path)}
    chunks = read_jsonl(args.chunks_path)
    chunk_lookup = {row["chunk_id"]: row for row in chunks}
    manifest, by_doc_id, by_file_name, by_short_title, by_variant, chunks_by_doc_id = build_manifest_lookups(chunks)

    if len(results_rows) != 50:
        raise ValueError(f"expected 50 historical full rows, got {len(results_rows)}")

    draft_rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []

    for row in results_rows:
        question_id = row["question_id"]
        if question_id in answer_badcases:
            gold_row = answer_badcases[question_id]
            max_docs = 1 if row["question_type"] == "fact" else (2 if row["question_type"] == "comparison" else 3)
            result_docs = candidate_docs_from_result(
                row,
                manifest=manifest,
                by_doc_id=by_doc_id,
                by_file_name=by_file_name,
                by_short_title=by_short_title,
                by_variant=by_variant,
            )
            gold_chunk_ids = remap_historical_chunk_ids(
                list(gold_row.get("gold_chunk_ids", [])),
                chunk_lookup=chunk_lookup,
                chunks_by_doc_id=chunks_by_doc_id,
                manifest=manifest,
                by_doc_id=by_doc_id,
                by_file_name=by_file_name,
                by_short_title=by_short_title,
                by_variant=by_variant,
            )
            target_doc_keys = target_doc_keys_from_gold_chunk_ids(
                gold_chunk_ids,
                chunk_lookup=chunk_lookup,
                by_doc_id=by_doc_id,
            )
            if not target_doc_keys and result_docs:
                target_doc_keys = [doc["doc_key"] for doc in result_docs[:max_docs]]
            if not gold_chunk_ids and result_docs:
                gold_chunk_ids = candidate_chunk_ids_from_result(
                    row,
                    chunk_lookup=chunk_lookup,
                    target_docs=result_docs[:max_docs],
                    intent=infer_intent(row["query"], row["question_type"]),
                    chunks_by_doc_id=chunks_by_doc_id,
                    manifest=manifest,
                    by_doc_id=by_doc_id,
                    by_file_name=by_file_name,
                    by_short_title=by_short_title,
                    by_variant=by_variant,
                )
            draft_rows.append(
                {
                    "question_id": question_id,
                    "query": row["query"],
                    "question_type": row["question_type"],
                    "intent": infer_intent(row["query"], row["question_type"]),
                    "industry": infer_industry(row["query"], [citation.get("file_name", "") for citation in row.get("citations", [])]),
                    "target_doc_keys": target_doc_keys,
                    "gold_answer": gold_row.get("gold_answer", ""),
                    "gold_chunk_ids": gold_chunk_ids,
                    "must_abstain": bool(gold_row.get("must_abstain", False)),
                    "review_notes": "direct_gold_from_answer_badcases_full",
                    "manual_review_required": False,
                    "source_label": "artifact_badcase_gold",
                }
            )
            report_rows.append(
                {
                    "question_id": question_id,
                    "source_label": "artifact_badcase_gold",
                    "manual_review_required": False,
                    "target_doc_keys": target_doc_keys,
                    "gold_chunk_ids": gold_chunk_ids,
                    "gold_answer": gold_row.get("gold_answer", ""),
                    "must_abstain": bool(gold_row.get("must_abstain", False)),
                }
            )
            continue

        if "_abstain_" in question_id:
            draft_rows.append(
                {
                    "question_id": question_id,
                    "query": row["query"],
                    "question_type": row["question_type"],
                    "intent": infer_intent(row["query"], row["question_type"]),
                    "industry": infer_industry(row["query"], [citation.get("file_name", "") for citation in row.get("citations", [])]),
                    "target_doc_keys": [],
                    "gold_answer": "",
                    "gold_chunk_ids": [],
                    "must_abstain": True,
                    "review_notes": "direct_gold_from_abstain_historical_id",
                    "manual_review_required": False,
                    "source_label": "artifact_abstain_gold",
                }
            )
            report_rows.append(
                {
                    "question_id": question_id,
                    "source_label": "artifact_abstain_gold",
                    "manual_review_required": False,
                    "must_abstain": True,
                    "was_abstain_badcase": question_id in abstain_badcases,
                }
            )
            continue

        draft_row, report_row = build_inferred_row(
            row,
            manifest=manifest,
            by_doc_id=by_doc_id,
            by_file_name=by_file_name,
            by_short_title=by_short_title,
            by_variant=by_variant,
            chunk_lookup=chunk_lookup,
            chunks_by_doc_id=chunks_by_doc_id,
        )
        draft_rows.append(draft_row)
        report_rows.append(report_row)

    write_jsonl(args.output_path, draft_rows)
    write_json(
        args.report_path,
        {
            "row_count": len(draft_rows),
            "question_type_distribution": dict(sorted(Counter(row["question_type"] for row in draft_rows).items())),
            "must_abstain_count": sum(1 for row in draft_rows if row["must_abstain"]),
            "manual_review_required_count": sum(1 for row in draft_rows if row.get("manual_review_required", False)),
            "source_distribution": dict(sorted(Counter(row["source_label"] for row in draft_rows).items())),
            "rows": report_rows,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
