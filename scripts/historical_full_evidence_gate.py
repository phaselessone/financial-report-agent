"""Fail-closed content audit for a historical-full candidate corpus."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.phase0_gate import (  # noqa: E402
    DEFAULT_FULL_ATTESTATION,
    DEFAULT_FULL_RESULTS,
    DEFAULT_FULL_SEED,
    GateError,
    check_full_seed_assets,
)
from src.evaluation.historical_evidence_audit import (  # noqa: E402
    HistoricalEvidenceAuditError,
    audit_historical_evidence_files,
    validate_stage6_corpus_attestation,
)


DEFAULT_CHUNKS = Path("tmp_stage6_data/chunks/chunks.jsonl")
DEFAULT_CORPUS_ATTESTATION = Path(
    "benchmarks/full/historical-stage6-corpus.attestation.json"
)
DEFAULT_OUTPUT = Path("outputs/phase0/historical_full_evidence_audit.json")


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that historical full gold chunk IDs resolve to the same "
            "document, page, and text in a hash-pinned candidate corpus."
        )
    )
    parser.add_argument("--seed-path", type=Path, default=DEFAULT_FULL_SEED)
    parser.add_argument("--results-path", type=Path, default=DEFAULT_FULL_RESULTS)
    parser.add_argument("--attestation-path", type=Path, default=DEFAULT_FULL_ATTESTATION)
    parser.add_argument("--chunks-path", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument(
        "--corpus-attestation-path",
        type=Path,
        default=DEFAULT_CORPUS_ATTESTATION,
        help="Reviewed sidecar binding Stage 6 provenance, SHA-256, and benchmark assets.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--expected-chunks-sha256",
        default=os.environ.get("STAGE6_CHUNKS_SHA256"),
        help=(
            "Optional diagnostic pin; a formal READY result still requires "
            "the reviewed corpus attestation sidecar."
        ),
    )
    return parser


def run(args: argparse.Namespace) -> dict:
    seed_path = _repo_path(args.seed_path)
    results_path = _repo_path(args.results_path)
    attestation_path = _repo_path(args.attestation_path)
    chunks_path = _repo_path(args.chunks_path)
    corpus_attestation_path = _repo_path(args.corpus_attestation_path)
    output_path = _repo_path(args.output)

    attestation_error: str | None = None
    attestation_validated = False
    try:
        full_assets = check_full_seed_assets(
            seed_path,
            results_path,
            attestation_path=attestation_path,
        )
        attestation_validated = full_assets.get("review_contract_source") == "attestation"
    except GateError as exc:
        attestation_error = str(exc)

    corpus_attestation: dict | None = None
    corpus_attestation_error: str | None = None
    expected_chunks_sha256 = args.expected_chunks_sha256
    try:
        corpus_attestation = validate_stage6_corpus_attestation(
            corpus_attestation_path,
            repo_root=REPO_ROOT,
            candidate_chunks_path=chunks_path,
            seed_path=seed_path,
            historical_results_path=results_path,
        )
        attested_chunks_sha256 = str(corpus_attestation["asset_sha256"])
        if expected_chunks_sha256 and str(expected_chunks_sha256).strip().lower() != attested_chunks_sha256:
            corpus_attestation_error = (
                "explicit expected chunks SHA-256 disagrees with the reviewed corpus attestation: "
                f"{str(expected_chunks_sha256).strip().lower()} != {attested_chunks_sha256}"
            )
        expected_chunks_sha256 = attested_chunks_sha256
    except HistoricalEvidenceAuditError as exc:
        corpus_attestation_error = str(exc)

    report = audit_historical_evidence_files(
        seed_path=seed_path,
        historical_results_path=results_path,
        candidate_chunks_path=chunks_path,
        expected_corpus_sha256=expected_chunks_sha256,
        reviewed_attestation_validated=attestation_validated,
        expected_question_count=50,
    )
    report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["attestation"] = {
        "path": str(attestation_path),
        "validated": attestation_validated,
        "error": attestation_error,
    }
    report["corpus_attestation"] = {
        "path": str(corpus_attestation_path),
        "validated": corpus_attestation is not None and corpus_attestation_error is None,
        "error": corpus_attestation_error,
        "record": corpus_attestation,
        "diagnostic_explicit_hash_used": (
            corpus_attestation is None and bool(args.expected_chunks_sha256)
        ),
    }
    if attestation_error:
        report["status"] = "BLOCKED"
        report["proof_level"] = "INCOMPLETE"
        report.setdefault("blocking_reasons", []).append(
            f"full benchmark attestation validation failed: {attestation_error}"
        )
    if corpus_attestation_error:
        report["status"] = "BLOCKED"
        report["proof_level"] = "INCOMPLETE"
        report.setdefault("blocking_reasons", []).append(
            f"stage6 corpus attestation validation failed: {corpus_attestation_error}"
        )
    report["output_path"] = str(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = run(args)
    except (HistoricalEvidenceAuditError, OSError, ValueError) as exc:
        print(f"HISTORICAL_FULL_EVIDENCE_GATE_FAILED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report.get("status") != "READY":
        print(
            "HISTORICAL_FULL_EVIDENCE_BLOCKED: candidate corpus is missing, unpinned, or content-incompatible",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
