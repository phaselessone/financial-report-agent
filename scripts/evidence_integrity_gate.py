"""CLI for the strict agent trace evidence-integrity gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

# Direct script execution places ``scripts/`` first on sys.path. Add the
# repository root so the same CLI works from a clean subprocess as from pytest.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.evidence_integrity import (
    validate_bundle_evidence_integrity,
    validate_evidence_integrity,
)
from src.evaluation.eval_bundle import read_eval_bundle, read_eval_bundle_reference


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate claim, citation, evidence, and calculation provenance."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--trace-path", type=Path, help="Legacy/raw trace JSONL path.")
    source.add_argument(
        "--bundle-path", type=Path, help="Strict eval bundle directory or bundle-reference JSON."
    )
    parser.add_argument("--chunks-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    return parser


def _load_bundle(path: Path):
    return read_eval_bundle(path) if path.is_dir() else read_eval_bundle_reference(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.bundle_path is not None:
        try:
            bundle = _load_bundle(args.bundle_path)
            report = validate_bundle_evidence_integrity(bundle, args.chunks_path)
        except (OSError, ValueError) as exc:
            issue = {
                "line": 0,
                "code": "BUNDLE_LOAD_ERROR",
                "error_code": "BUNDLE_LOAD_ERROR",
                "message": str(exc),
            }
            report = {
                "status": "BLOCKED",
                "issues": [issue],
                "errors": [issue],
                "summary": {"trace_rows": 0, "issue_count": 1, "error_count": 1},
                "bundle": {"path": str(args.bundle_path)},
                "chunks_path": str(args.chunks_path),
            }
    else:
        report = validate_evidence_integrity(args.trace_path, args.chunks_path)
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    if report["status"] != "READY":
        print("EVIDENCE_INTEGRITY_BLOCKED: trace provenance errors found", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
