from __future__ import annotations

from src.utils.env import load_env_files

load_env_files()

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

from src.evaluation.structured_eval import (
    build_store_from_facts_jsonl,
    evaluate_rag_row,
    evaluate_structured_row,
    load_structured_seed,
    summarize_rows,
    write_summary_json,
    write_summary_markdown,
)
from src.utils.io import ensure_dir, read_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run structured fact benchmark: Text RAG vs Structured Fact Search.")
    parser.add_argument("--seed-path", type=Path, default=Path("data/eval_set/structured_fact_seed_dev.jsonl"))
    parser.add_argument("--facts-path", type=Path, default=Path("data/structured/facts_dev.jsonl"))
    parser.add_argument("--aliases-path", type=Path, default=Path("data/structured/company_aliases.json"))
    parser.add_argument("--store-db-path", type=Path, default=Path("data/structured/facts.duckdb"))
    parser.add_argument("--store-parquet-path", type=Path, default=Path("data/structured/facts.parquet"))
    parser.add_argument("--chunks-path", type=Path, default=Path("data/chunks/chunks.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model-cache-dir", type=Path, default=Path("models"))
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--generation-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--llm-provider", default="")
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--runtime-profile", default=None, choices=("low_vram", "cpu", "standard_gpu"))
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--reranker-device", default=None)
    parser.add_argument("--legs", choices=("structured", "rag", "compare"), default="compare")
    parser.add_argument("--eval-limit", type=int, default=0)
    return parser.parse_args()


def run_structured_leg(seed_rows, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    aliases = read_json(args.aliases_path)
    store = build_store_from_facts_jsonl(args.facts_path, db_path=args.store_db_path)
    if args.store_parquet_path:
        ensure_dir(args.store_parquet_path.parent)
        store.dump_parquet(args.store_parquet_path)
    rows = [evaluate_structured_row(row, store, aliases) for row in seed_rows]
    summary = summarize_rows(rows)
    summary["routed_rate"] = round(sum(1 for row in rows if row.get("routed")) / max(len(rows), 1), 4)
    store.close()
    return rows, summary


def run_rag_leg(seed_rows, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    from src.generation.answerer import infer_question_type
    from src.generation.provider import build_generation_answerer
    from src.retrieval.domain_priority import apply_retrieval_domain_priority
    from src.retrieval.runtime import build_retrieval_runtime
    from src.utils.io import read_jsonl

    chunks = read_jsonl(args.chunks_path)
    runtime = build_retrieval_runtime(
        chunks=chunks,
        output_dir=args.output_dir.resolve(),
        model_cache_dir=args.model_cache_dir.resolve(),
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
        runtime_profile=args.runtime_profile,
        embedding_device=args.embedding_device,
        reranker_device=args.reranker_device,
    )
    answerer = build_generation_answerer(
        provider=args.llm_provider,
        local_model_name=args.generation_model,
        remote_model_name=args.llm_model,
        cache_dir=args.model_cache_dir.resolve(),
        device="cpu",
    )
    rows: list[dict[str, Any]] = []
    for index, seed_row in enumerate(seed_rows, start=1):
        print(f"[rag {index}/{len(seed_rows)}] {seed_row['qid']} {seed_row['query']}", flush=True)
        start = perf_counter()
        try:
            retrieval_result = runtime.search(seed_row["query"])
            retrieval_result = apply_retrieval_domain_priority(
                retrieval_result,
                str(seed_row.get("industry", "")),
            )
            answer_row = answerer.answer(
                query=seed_row["query"],
                question_type="fact",
                retrieval_result=retrieval_result,
                query_domain_hint=seed_row.get("industry", ""),
            )
            latency_ms = round((perf_counter() - start) * 1000, 2)
        except Exception as exc:
            print(f"[rag] {seed_row['qid']} FAILED: {type(exc).__name__}: {exc}", flush=True)
            answer_row = {
                "final_answer": "",
                "abstained": True,
                "citations": [],
                "selected_doc_ids": [],
                "failed": True,
                "error_message": str(exc),
            }
            latency_ms = round((perf_counter() - start) * 1000, 2)
        evaluated = evaluate_rag_row(seed_row, answer_row)
        evaluated["latency_ms"] = latency_ms
        evaluated["raw_answer"] = str(answer_row.get("final_answer") or "")
        evaluated["citations"] = answer_row.get("citations") or []
        evaluated["selected_doc_ids"] = answer_row.get("selected_doc_ids") or []
        rows.append(evaluated)

    summary = summarize_rows(rows)
    usage = getattr(answerer, "usage_summary", lambda: {})()
    return rows, summary, usage


def main() -> int:
    args = parse_args()
    seed_rows = load_structured_seed(args.seed_path)
    if args.eval_limit > 0:
        seed_rows = seed_rows[: args.eval_limit]
    if not seed_rows:
        print(f"no seed rows in {args.seed_path}", file=sys.stderr)
        return 1

    output_dir = ensure_dir(args.output_dir.resolve())
    report_dir = ensure_dir(output_dir / "reports")

    payload: dict[str, Any] = {"total": len(seed_rows)}
    if args.legs in ("structured", "compare"):
        structured_rows, structured_summary = run_structured_leg(seed_rows, args)
        write_jsonl(output_dir / "structured_fact_results.jsonl", structured_rows)
        payload["structured"] = structured_summary
        print("structured leg:", structured_summary)
    if args.legs in ("rag", "compare"):
        rag_rows, rag_summary, usage = run_rag_leg(seed_rows, args)
        write_jsonl(output_dir / "structured_rag_results.jsonl", rag_rows)
        payload["rag"] = rag_summary
        payload["api_usage"] = usage
        print("rag leg:", rag_summary)
        print("api usage:", usage)

    write_summary_json(report_dir / "structured_fact_eval_summary.json", payload)
    if "rag" in payload and "structured" in payload:
        write_summary_markdown(report_dir / "structured_fact_eval_summary.md", payload)
    print("summary written to", report_dir / "structured_fact_eval_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
