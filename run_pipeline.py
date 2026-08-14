from __future__ import annotations

from src.utils.env import load_env_files

load_env_files()

import argparse
from pathlib import Path

from src.ingest.pipeline import run_ingest_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PDF ingest and chunk pipeline.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("pdf"),
        help="Directory containing PDF files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Base output directory for parsed pages and chunks.",
    )
    parser.add_argument(
        "--badcase-dir",
        type=Path,
        default=Path("outputs/badcases"),
        help="Directory to store badcase records.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of PDFs to process. 0 means all PDFs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_ingest_pipeline(
        input_dir=args.input_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        badcase_dir=args.badcase_dir.resolve(),
        limit=args.limit,
    )
    print(result["totals"])
    print(result["strategy_report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
