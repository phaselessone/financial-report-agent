from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from src.evaluation.benchmark_assets import build_doc_manifest
from src.utils.io import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize restored full answer-eval seed after manual review.")
    parser.add_argument(
        "--draft-path",
        type=Path,
        default=Path("data/eval_set/answer_eval_seed_full_draft.jsonl"),
    )
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=Path("tmp_stage6_data/chunks/chunks.jsonl"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/eval_set/answer_eval_seed_full.jsonl"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("data/eval_set/answer_eval_seed_full_restore_report.json"),
    )
    return parser.parse_args()


def target_doc_keys_from_chunk_ids(gold_chunk_ids: list[str], *, chunk_lookup: dict[str, dict[str, Any]], docs_by_doc_id: dict[str, dict[str, Any]]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for chunk_id in gold_chunk_ids:
        chunk = chunk_lookup.get(chunk_id)
        if chunk is None:
            continue
        manifest_doc = docs_by_doc_id.get(chunk["doc_id"])
        if manifest_doc is None:
            continue
        doc_key = manifest_doc["doc_key"]
        if doc_key in seen:
            continue
        seen.add(doc_key)
        ordered.append(doc_key)
    return ordered


def main() -> int:
    args = parse_args()
    draft_rows = read_jsonl(args.draft_path)
    chunks = read_jsonl(args.chunks_path)
    chunk_lookup = {row["chunk_id"]: row for row in chunks}
    manifest = build_doc_manifest(chunks)
    docs_by_doc_id = {row["doc_id"]: row for row in manifest}
    docs_by_doc_key = {row["doc_key"]: row for row in manifest}

    final_rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    for row in draft_rows:
        target_doc_keys = list(row.get("target_doc_keys", []))
        gold_chunk_ids = list(row.get("gold_chunk_ids", []))
        if not row.get("must_abstain", False) and not target_doc_keys:
            target_doc_keys = target_doc_keys_from_chunk_ids(
                gold_chunk_ids,
                chunk_lookup=chunk_lookup,
                docs_by_doc_id=docs_by_doc_id,
            )

        gold_answer = row.get("gold_answer", "")
        if row.get("intent") == "report_lookup" and target_doc_keys:
            first_doc = docs_by_doc_key.get(target_doc_keys[0])
            if first_doc is not None:
                gold_answer = first_doc["short_title"]

        finalized = dict(row)
        finalized["target_doc_keys"] = target_doc_keys
        finalized["gold_answer"] = gold_answer
        finalized["manual_review_required"] = False
        if row.get("source_label") == "artifact_citation_inferred":
            finalized["review_notes"] = "manual_review_completed_from_historical_restore_20260405"
        final_rows.append(finalized)

        report_rows.append(
            {
                "question_id": finalized["question_id"],
                "source_label": finalized.get("source_label", ""),
                "manual_review_required": False,
                "question_type": finalized["question_type"],
                "intent": finalized["intent"],
                "industry": finalized["industry"],
                "target_doc_keys": finalized["target_doc_keys"],
                "gold_chunk_ids": finalized["gold_chunk_ids"],
                "must_abstain": bool(finalized["must_abstain"]),
                "review_notes": finalized.get("review_notes", ""),
            }
        )

    write_jsonl(args.output_path, final_rows)
    write_json(
        args.report_path,
        {
            "row_count": len(final_rows),
            "question_type_distribution": dict(sorted(Counter(row["question_type"] for row in final_rows).items())),
            "must_abstain_count": sum(1 for row in final_rows if row["must_abstain"]),
            "manual_review_required_count": 0,
            "source_distribution": dict(sorted(Counter(row.get("source_label", "") for row in final_rows).items())),
            "rows": report_rows,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
