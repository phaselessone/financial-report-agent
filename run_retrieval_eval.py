from __future__ import annotations

from src.utils.env import load_env_files

load_env_files()

import argparse
from pathlib import Path

from src.evaluation.output_archive import archive_output_bundle, default_artifact_label, resolve_artifact_dir, write_scratch_run_policy
from src.evaluation.benchmark_assets import prepare_benchmark_assets
from src.evaluation.run_metadata import build_run_metadata
from src.evaluation.retrieval_eval import evaluate_retrieval_methods, materialize_eval_set
from src.utils.io import ensure_dir, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build retrieval indexes and run retrieval evaluation.")
    parser.add_argument("--chunks-path", type=Path, default=Path("data/chunks/chunks.jsonl"))
    parser.add_argument("--eval-seed-path", type=Path, default=Path("data/eval_set/retrieval_eval_seed.jsonl"))
    parser.add_argument("--eval-path", type=Path, default=Path("data/eval_set/retrieval_eval.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model-cache-dir", type=Path, default=Path("models"))
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--rerank-batch-size", type=int, default=8)
    parser.add_argument("--dense-top-k", type=int, default=20)
    parser.add_argument("--bm25-top-k", type=int, default=20)
    parser.add_argument("--rerank-top-k", type=int, default=5)
    parser.add_argument("--prepare-benchmark", action="store_true")
    parser.add_argument("--benchmark-label", default="current/retrieval")
    parser.add_argument("--corpus-label", default="current")
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--artifact-label", default="")
    parser.add_argument("--skip-artifact-archive", action="store_true")
    parser.add_argument("--rebuild-indexes", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from src.retrieval.runtime import build_retrieval_runtime

    artifact_dir = None
    if not args.skip_artifact_archive:
        artifact_label = args.artifact_label or default_artifact_label(split="retrieval", benchmark_profile=args.corpus_label)
        artifact_dir = resolve_artifact_dir(artifact_root=args.artifact_root.resolve(), artifact_label=artifact_label)
    chunks = read_jsonl(args.chunks_path.resolve())
    eval_dir = ensure_dir(args.eval_seed_path.resolve().parent)
    if args.prepare_benchmark:
        prepare_benchmark_assets(
            chunks=chunks,
            manifest_output_path=eval_dir / "doc_manifest.jsonl",
            seed_output_path=args.eval_seed_path.resolve(),
        )
    elif not args.eval_seed_path.resolve().exists():
        raise FileNotFoundError(
            f"Missing retrieval eval seed: {args.eval_seed_path.resolve()}. "
            "Run run_prepare_benchmark.py or rerun this command with --prepare-benchmark."
        )

    report_dir = ensure_dir(args.output_dir.resolve() / "reports")
    badcase_dir = ensure_dir(args.output_dir.resolve() / "badcases")
    source_manifest_path = report_dir / "source_manifest.json"
    eval_rows = materialize_eval_set(
        seed_path=args.eval_seed_path.resolve(),
        chunks_path=args.chunks_path.resolve(),
        output_path=args.eval_path.resolve(),
        report_output_path=report_dir / "eval_manifest_report.json",
    )

    runtime = build_retrieval_runtime(
        chunks=chunks,
        output_dir=args.output_dir.resolve(),
        model_cache_dir=args.model_cache_dir.resolve(),
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
        device=args.device,
        embedding_batch_size=args.embedding_batch_size,
        rerank_batch_size=args.rerank_batch_size,
        dense_top_k=args.dense_top_k,
        bm25_top_k=args.bm25_top_k,
        rerank_top_k=args.rerank_top_k,
        rebuild_indexes=args.rebuild_indexes,
    )

    method_results: dict[str, dict[str, list[dict[str, object]]]] = {
        "dense": {},
        "bm25": {},
        "hybrid": {},
        "hybrid_rerank": {},
    }

    for eval_row in eval_rows:
        search_result = runtime.search(eval_row["query"])
        method_results["dense"][eval_row["question_id"]] = search_result["dense_rows"]
        method_results["bm25"][eval_row["question_id"]] = search_result["bm25_rows"]
        method_results["hybrid"][eval_row["question_id"]] = search_result["hybrid_rows"]
        method_results["hybrid_rerank"][eval_row["question_id"]] = search_result["rerank_rows"]

    run_metadata = build_run_metadata(
        chunks_path=args.chunks_path.resolve(),
        eval_rows=eval_rows,
        benchmark_source_path=args.eval_seed_path.resolve(),
        benchmark_label=args.benchmark_label,
        benchmark_profile="custom" if args.prepare_benchmark else "current-dev",
        corpus_label=args.corpus_label,
        split="retrieval",
        cwd=Path.cwd(),
        source_manifest_path=source_manifest_path,
        extra_fields={
            "outputs_mode": "scratch",
            "scratch_output_dir": str(args.output_dir.resolve()),
            "artifact_dir": str(artifact_dir) if artifact_dir is not None else "",
        },
    )
    summary = evaluate_retrieval_methods(
        eval_rows=eval_rows,
        method_results=method_results,
        results_output_path=report_dir / "retrieval_eval_results.jsonl",
        summary_output_path=report_dir / "retrieval_eval_summary.json",
        markdown_output_path=report_dir / "retrieval_eval_summary.md",
        badcase_output_path=badcase_dir / "retrieval_badcases.jsonl",
        run_metadata=run_metadata,
    )
    if artifact_dir is not None:
        archive_output_bundle(output_dir=args.output_dir.resolve(), artifact_dir=artifact_dir)
    write_scratch_run_policy(
        output_dir=args.output_dir.resolve(),
        artifact_dir=artifact_dir,
        split="retrieval",
        benchmark_profile=args.corpus_label,
        summary_path=report_dir / "retrieval_eval_summary.json",
        source_manifest_path=source_manifest_path,
    )

    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
