from __future__ import annotations

from src.utils.env import load_env_files

load_env_files()

import argparse
from pathlib import Path

from src.evaluation.benchmark_assets import build_answer_seed_draft, prepare_benchmark_assets, write_answer_seed_draft
from src.evaluation.retrieval_eval import materialize_eval_set
from src.utils.io import ensure_dir, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare benchmark assets without running retrieval or answer evaluation.")
    parser.add_argument("--chunks-path", type=Path, default=Path("data/chunks/chunks.jsonl"))
    parser.add_argument("--retrieval-seed-path", type=Path, default=Path("data/eval_set/retrieval_eval_seed.jsonl"))
    parser.add_argument("--retrieval-eval-path", type=Path, default=Path("data/eval_set/retrieval_eval.jsonl"))
    parser.add_argument("--answer-seed-draft-dev-path", type=Path, default=Path("data/eval_set/answer_eval_seed_draft_dev.jsonl"))
    parser.add_argument("--answer-seed-current-full-path", type=Path, default=Path("data/eval_set/answer_eval_seed_current_full.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    chunks = read_jsonl(args.chunks_path.resolve())
    eval_dir = ensure_dir(args.retrieval_seed_path.resolve().parent)
    report_dir = ensure_dir(args.output_dir.resolve() / "reports")
    manifest, retrieval_seed_rows = prepare_benchmark_assets(
        chunks=chunks,
        manifest_output_path=eval_dir / "doc_manifest.jsonl",
        seed_output_path=args.retrieval_seed_path.resolve(),
    )
    eval_rows = materialize_eval_set(
        seed_path=args.retrieval_seed_path.resolve(),
        chunks_path=args.chunks_path.resolve(),
        output_path=args.retrieval_eval_path.resolve(),
        report_output_path=report_dir / "eval_manifest_report.json",
    )
    draft_rows = build_answer_seed_draft(
        retrieval_seed_rows=retrieval_seed_rows,
        chunks=chunks,
        manifest=manifest,
        scope="dev",
    )
    write_answer_seed_draft(draft_rows, args.answer_seed_draft_dev_path.resolve())
    current_full_rows = build_answer_seed_draft(
        retrieval_seed_rows=retrieval_seed_rows,
        chunks=chunks,
        manifest=manifest,
        scope="current-full",
    )
    write_answer_seed_draft(current_full_rows, args.answer_seed_current_full_path.resolve())
    print(
        {
            "retrieval_seed_rows": len(retrieval_seed_rows),
            "retrieval_eval_rows": len(eval_rows),
            "answer_seed_draft_dev_rows": len(draft_rows),
            "answer_seed_current_full_rows": len(current_full_rows),
            "retrieval_seed_path": str(args.retrieval_seed_path.resolve()),
            "retrieval_eval_path": str(args.retrieval_eval_path.resolve()),
            "answer_seed_draft_dev_path": str(args.answer_seed_draft_dev_path.resolve()),
            "answer_seed_current_full_path": str(args.answer_seed_current_full_path.resolve()),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
