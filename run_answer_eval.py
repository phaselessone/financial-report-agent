from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

from src.evaluation.answer_eval import evaluate_answer_results, materialize_answer_eval_sets
from src.evaluation.benchmark_assets import build_answer_seed_draft, industry_from_file_name, prepare_benchmark_assets, write_answer_seed_draft
from src.evaluation.output_archive import archive_output_bundle, default_artifact_label, resolve_artifact_dir, write_scratch_run_policy
from src.evaluation.run_metadata import build_run_metadata
from src.evaluation.retrieval_eval import materialize_eval_set
from src.utils.io import ensure_dir, read_jsonl

DOMAIN_COMPATIBILITY_GROUPS = {
    "consumer": {"consumer", "liquor"},
    "liquor": {"consumer", "liquor"},
}


def _normalize_limit(value: int) -> int:
    return value if value > 0 else 0


def resolve_eval_limit(args: argparse.Namespace) -> int:
    eval_limit = _normalize_limit(args.eval_limit)
    deprecated_limit = _normalize_limit(args.limit)
    if eval_limit and deprecated_limit:
        print("Ignoring deprecated --limit because --eval-limit is set.", file=sys.stderr)
    return eval_limit or deprecated_limit


def resolve_answer_seed_paths(*, split: str, dev_seed_path: Path, full_seed_path: Path) -> dict[str, Path]:
    if not dev_seed_path.exists():
        raise FileNotFoundError(f"Missing dev answer seed: {dev_seed_path}.")

    seed_paths = {"dev": dev_seed_path}
    if full_seed_path.exists():
        seed_paths["full"] = full_seed_path
    elif split == "full":
        raise FileNotFoundError(
            f"Missing full answer seed: {full_seed_path}. "
            "Provide --answer-seed-full-path or switch to --split dev."
        )
    return seed_paths


def normalize_benchmark_profile(profile: str) -> str:
    return "historical-full-raw" if profile == "historical-full" else profile


def resolve_benchmark_profile(args: argparse.Namespace) -> str:
    if args.benchmark_profile:
        return normalize_benchmark_profile(args.benchmark_profile)
    return "current-dev" if args.split == "dev" else "historical-full-core"


def resolve_requested_materialized_splits(args: argparse.Namespace, profile: str) -> tuple[str, ...]:
    requested = [args.split]
    if args.materialize_full and args.split == "dev":
        requested.append("full")
    if profile in {"historical-full", "historical-full-raw", "historical-full-core"} and "full" not in requested:
        requested.append("full")
    return tuple(dict.fromkeys(requested))


def validate_benchmark_profile(*, profile: str, split: str, chunks_path: Path, corpus_label: str) -> None:
    normalized_chunks_path = str(chunks_path).lower()
    normalized_corpus_label = corpus_label.lower()
    if profile == "current-dev" and split != "dev":
        raise ValueError("benchmark profile 'current-dev' only supports --split dev.")
    if profile in {"historical-full", "historical-full-raw", "historical-full-core"}:
        if split != "full":
            raise ValueError(f"benchmark profile '{profile}' only supports --split full.")
        if "stage6" not in normalized_chunks_path and "historical" not in normalized_corpus_label:
            raise ValueError(
                f"benchmark profile '{profile}' requires the historical stage6 corpus. "
                "Use --chunks-path tmp_stage6_data/chunks/chunks.jsonl and --corpus-label stage6-historical."
            )


def _row_domain_bucket(row: dict[str, Any]) -> str:
    return str(row.get("industry") or industry_from_file_name(str(row.get("file_name", ""))) or "")


def _normalize_domain_bucket(bucket: str) -> str:
    return (bucket or "").strip().lower()


def _is_weak_domain_bucket(bucket: str) -> bool:
    normalized = _normalize_domain_bucket(bucket)
    return not normalized or normalized == "other"


def _domains_compatible(left: str, right: str) -> bool:
    left_normalized = _normalize_domain_bucket(left)
    right_normalized = _normalize_domain_bucket(right)
    if not left_normalized or not right_normalized:
        return False
    if left_normalized == right_normalized:
        return True
    return (
        right_normalized in DOMAIN_COMPATIBILITY_GROUPS.get(left_normalized, {left_normalized})
        or left_normalized in DOMAIN_COMPATIBILITY_GROUPS.get(right_normalized, {right_normalized})
    )


def apply_retrieval_domain_priority(retrieval_result: dict[str, Any], domain_hint: str) -> dict[str, Any]:
    normalized_hint = _normalize_domain_bucket(domain_hint)
    if _is_weak_domain_bucket(normalized_hint):
        return retrieval_result
    prioritized_result = dict(retrieval_result)
    for key in ("dense_rows", "bm25_rows", "hybrid_rows", "rerank_rows"):
        rows = list(retrieval_result.get(key, []))
        if not rows:
            continue
        exact_rows = [row for row in rows if _normalize_domain_bucket(_row_domain_bucket(row)) == normalized_hint]
        compatible_rows = [
            row
            for row in rows
            if row not in exact_rows and _domains_compatible(_row_domain_bucket(row), normalized_hint)
        ]
        other_rows = [row for row in rows if row not in exact_rows and row not in compatible_rows]
        if exact_rows or compatible_rows:
            prioritized_result[key] = exact_rows + compatible_rows + other_rows
    return prioritized_result


def build_failed_result_row(
    *,
    eval_row: dict[str, Any],
    error: Exception,
    end_to_end_latency_ms: float,
    retrieval_result: dict[str, Any] | None = None,
    llm_provider: str = "",
    llm_model: str = "",
) -> dict[str, Any]:
    timings = dict((retrieval_result or {}).get("timings", {}))
    timings.setdefault("retrieval_latency_ms", 0.0)
    timings.setdefault("rerank_latency_ms", 0.0)
    timings["end_to_end_latency_ms"] = end_to_end_latency_ms
    from src.generation.answerer import infer_question_type

    question_type = eval_row.get("question_type") or infer_question_type(eval_row["query"])
    answer_mode = eval_row.get("intent", question_type)
    return {
        "question_id": eval_row["question_id"],
        "query": eval_row["query"],
        "question_type": question_type,
        "query_intent": answer_mode,
        "answer_mode": answer_mode,
        "fact_subtype": "",
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "answer_source": "error",
        "final_answer": "",
        "evidence_summary": "",
        "uncertainty_note": "",
        "used_evidence_ids": [],
        "abstained": False,
        "abstain_reason": None,
        "abstain_gate": "error",
        "confidence_label": "low",
        "citations": [],
        "retrieval_scores": {},
        "support_validation": {"supported": False, "missing_numeric_tokens": [], "missing_keywords": []},
        "matched_numeric_tokens": [],
        "title_resolved_from": "",
        "query_domain_bucket": "",
        "selected_domain_buckets": [],
        "doc_guard_triggered": False,
        "citation_rule_applied": "default",
        "support_filter_applied": "default",
        "selected_evidence": [],
        "doc_candidates": [],
        "selected_doc_ids": [],
        "raw_model_output": "",
        "json_parse_failures": 0,
        "generation_latency_ms": 0.0,
        "failed": True,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "timings": timings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run answer generation, citation binding and answer-level evaluation.")
    parser.add_argument("--input-dir", type=Path, default=Path("pdf"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--chunks-path", type=Path, default=Path("data/chunks/chunks.jsonl"))
    parser.add_argument("--retrieval-seed-path", type=Path, default=Path("data/eval_set/retrieval_eval_seed.jsonl"))
    parser.add_argument("--retrieval-eval-path", type=Path, default=Path("data/eval_set/retrieval_eval.jsonl"))
    parser.add_argument("--answer-seed-dev-path", type=Path, default=Path("data/eval_set/answer_eval_seed_dev.jsonl"))
    parser.add_argument("--answer-seed-full-path", type=Path, default=Path("data/eval_set/answer_eval_seed_full.jsonl"))
    parser.add_argument("--answer-seed-draft-dev-path", type=Path, default=Path("data/eval_set/answer_eval_seed_draft_dev.jsonl"))
    parser.add_argument("--answer-eval-dev-path", type=Path, default=Path("data/eval_set/answer_eval_dev.jsonl"))
    parser.add_argument("--answer-eval-full-path", type=Path, default=Path("data/eval_set/answer_eval_full.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model-cache-dir", type=Path, default=Path("models"))
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--generation-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--llm-provider", default="")
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--rerank-batch-size", type=int, default=8)
    parser.add_argument("--dense-top-k", type=int, default=20)
    parser.add_argument("--bm25-top-k", type=int, default=20)
    parser.add_argument("--rerank-top-k", type=int, default=5)
    parser.add_argument("--rerank-candidates-k", type=int, default=0)
    parser.add_argument("--split", choices=("dev", "full"), default="dev")
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--ingest-limit", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="Deprecated alias for --eval-limit.")
    parser.add_argument("--benchmark-profile", choices=("current-dev", "historical-full", "historical-full-raw", "historical-full-core", "custom"), default="")
    parser.add_argument("--materialize-full", action="store_true")
    parser.add_argument("--prepare-benchmark", action="store_true")
    parser.add_argument("--skip-benchmark-prepare", action="store_true")
    parser.add_argument("--benchmark-label-dev", default="current/dev")
    parser.add_argument("--benchmark-label-full", default="stress/historical")
    parser.add_argument("--corpus-label", default="current")
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--artifact-label", default="")
    parser.add_argument("--skip-artifact-archive", action="store_true")
    parser.add_argument("--rebuild-indexes", action="store_true")
    parser.add_argument("--rebuild-chunks", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from src.ingest.pipeline import run_ingest_pipeline
    from src.generation.answerer import infer_question_type
    from src.generation.provider import build_generation_answerer
    from src.retrieval.runtime import build_retrieval_runtime

    eval_limit = resolve_eval_limit(args)
    ingest_limit = _normalize_limit(args.ingest_limit)
    chunks_path = args.chunks_path.resolve()
    benchmark_profile = resolve_benchmark_profile(args)
    validate_benchmark_profile(
        profile=benchmark_profile,
        split=args.split,
        chunks_path=chunks_path,
        corpus_label=args.corpus_label,
    )
    requested_materialized_splits = resolve_requested_materialized_splits(args, benchmark_profile)
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    badcase_dir = ensure_dir(output_dir / "badcases")
    report_dir = ensure_dir(output_dir / "reports")
    artifact_dir = None
    if not args.skip_artifact_archive:
        artifact_label = args.artifact_label or default_artifact_label(split=args.split, benchmark_profile=benchmark_profile)
        artifact_dir = resolve_artifact_dir(artifact_root=args.artifact_root.resolve(), artifact_label=artifact_label)
    source_manifest_path = report_dir / "source_manifest.json"

    if args.rebuild_chunks or not chunks_path.exists():
        run_ingest_pipeline(
            input_dir=args.input_dir.resolve(),
            output_dir=data_dir,
            badcase_dir=badcase_dir,
            limit=ingest_limit,
        )

    chunks = read_jsonl(chunks_path)
    eval_dir = ensure_dir(args.retrieval_seed_path.resolve().parent)
    should_prepare_benchmark = args.prepare_benchmark and not args.skip_benchmark_prepare
    if should_prepare_benchmark:
        manifest, retrieval_seed_rows = prepare_benchmark_assets(
            chunks=chunks,
            manifest_output_path=eval_dir / "doc_manifest.jsonl",
            seed_output_path=args.retrieval_seed_path.resolve(),
        )
        materialize_eval_set(
            seed_path=args.retrieval_seed_path.resolve(),
            chunks_path=chunks_path,
            output_path=args.retrieval_eval_path.resolve(),
            report_output_path=report_dir / "eval_manifest_report.json",
        )

        draft_rows = build_answer_seed_draft(
            retrieval_seed_rows=retrieval_seed_rows,
            chunks=chunks,
            manifest=manifest,
        )
        write_answer_seed_draft(draft_rows, args.answer_seed_draft_dev_path.resolve())
    seed_paths = resolve_answer_seed_paths(
        split=args.split,
        dev_seed_path=args.answer_seed_dev_path.resolve(),
        full_seed_path=args.answer_seed_full_path.resolve(),
    )
    answer_eval_sets = materialize_answer_eval_sets(
        dev_seed_path=seed_paths["dev"],
        full_seed_path=seed_paths.get("full"),
        chunks=chunks,
        dev_output_path=args.answer_eval_dev_path.resolve(),
        full_output_path=args.answer_eval_full_path.resolve(),
        report_output_path=report_dir / "answer_eval_manifest_report.json",
        requested_splits=requested_materialized_splits,
        benchmark_profile=benchmark_profile,
    )
    if args.split not in answer_eval_sets:
        raise FileNotFoundError(f"Requested split '{args.split}' is unavailable.")

    eval_rows = list(answer_eval_sets[args.split])
    if eval_limit > 0:
        eval_rows = eval_rows[:eval_limit]

    benchmark_source_path = seed_paths[args.split]
    if args.split == "dev":
        benchmark_label = args.benchmark_label_dev
    elif benchmark_profile == "historical-full-core":
        benchmark_label = "stress/historical-core"
    else:
        benchmark_label = args.benchmark_label_full

    runtime = build_retrieval_runtime(
        chunks=chunks,
        output_dir=output_dir,
        model_cache_dir=args.model_cache_dir.resolve(),
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
        device=args.device,
        embedding_batch_size=args.embedding_batch_size,
        rerank_batch_size=args.rerank_batch_size,
        dense_top_k=args.dense_top_k,
        bm25_top_k=args.bm25_top_k,
        rerank_top_k=args.rerank_top_k,
        rerank_candidates_k=_normalize_limit(args.rerank_candidates_k) or None,
        rebuild_indexes=args.rebuild_indexes,
    )
    answerer = build_generation_answerer(
        provider=args.llm_provider,
        local_model_name=args.generation_model,
        remote_model_name=args.llm_model,
        cache_dir=args.model_cache_dir.resolve(),
        device=args.device,
    )

    result_rows: list[dict[str, object]] = []
    for index, eval_row in enumerate(eval_rows, start=1):
        print(f"[{index}/{len(eval_rows)}] Answering {eval_row['question_id']}")
        end_to_end_start = perf_counter()
        retrieval_result: dict[str, Any] | None = None
        try:
            retrieval_result = runtime.search(eval_row["query"])
            retrieval_result = apply_retrieval_domain_priority(
                retrieval_result,
                str(eval_row.get("industry", "")),
            )
            answer_row = answerer.answer(
                query=eval_row["query"],
                question_type=eval_row.get("question_type") or infer_question_type(eval_row["query"]),
                retrieval_result=retrieval_result,
                query_domain_hint=eval_row.get("industry", ""),
            )
            answer_row["question_id"] = eval_row["question_id"]
            answer_row["timings"] = {
                **retrieval_result["timings"],
                "end_to_end_latency_ms": round((perf_counter() - end_to_end_start) * 1000, 2),
            }
            result_rows.append(answer_row)
        except Exception as exc:
            result_rows.append(
                build_failed_result_row(
                    eval_row=eval_row,
                    error=exc,
                    retrieval_result=retrieval_result,
                    end_to_end_latency_ms=round((perf_counter() - end_to_end_start) * 1000, 2),
                    llm_provider=getattr(answerer, "llm_provider", ""),
                    llm_model=getattr(answerer, "llm_model", ""),
                )
            )

    run_metadata = build_run_metadata(
        chunks_path=chunks_path,
        eval_rows=eval_rows,
        benchmark_source_path=benchmark_source_path,
        benchmark_label=benchmark_label,
        benchmark_profile=benchmark_profile,
        corpus_label=args.corpus_label,
        split=args.split,
        cwd=Path.cwd(),
        source_manifest_path=source_manifest_path,
        extra_fields={
            "outputs_mode": "scratch",
            "scratch_output_dir": str(output_dir),
            "artifact_dir": str(artifact_dir) if artifact_dir is not None else "",
        },
    )
    summary = evaluate_answer_results(
        eval_rows=eval_rows,
        result_rows=result_rows,
        chunks_path=chunks_path,
        results_output_path=report_dir / f"answer_eval_results_{args.split}.jsonl",
        summary_output_path=report_dir / f"answer_eval_summary_{args.split}.json",
        markdown_output_path=report_dir / f"answer_eval_summary_{args.split}.md",
        latency_output_path=report_dir / f"latency_summary_{args.split}.json",
        answer_badcase_output_path=badcase_dir / f"answer_badcases_{args.split}.jsonl",
        abstain_badcase_output_path=badcase_dir / f"abstain_badcases_{args.split}.jsonl",
        run_metadata=run_metadata,
    )

    ensure_dir(report_dir)
    (report_dir / "answer_eval_results.jsonl").write_text(
        (report_dir / f"answer_eval_results_{args.split}.jsonl").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (report_dir / "answer_eval_summary.json").write_text(
        (report_dir / f"answer_eval_summary_{args.split}.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (report_dir / "answer_eval_summary.md").write_text(
        (report_dir / f"answer_eval_summary_{args.split}.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (report_dir / "latency_summary.json").write_text(
        (report_dir / f"latency_summary_{args.split}.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (badcase_dir / "answer_badcases.jsonl").write_text(
        (badcase_dir / f"answer_badcases_{args.split}.jsonl").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (badcase_dir / "abstain_badcases.jsonl").write_text(
        (badcase_dir / f"abstain_badcases_{args.split}.jsonl").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    if artifact_dir is not None:
        archive_output_bundle(output_dir=output_dir, artifact_dir=artifact_dir)
    write_scratch_run_policy(
        output_dir=output_dir,
        artifact_dir=artifact_dir,
        split=args.split,
        benchmark_profile=benchmark_profile,
        summary_path=report_dir / f"answer_eval_summary_{args.split}.json",
        source_manifest_path=source_manifest_path,
    )

    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
