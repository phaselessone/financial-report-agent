from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

from scripts import historical_full_evidence_gate
from src.evaluation.historical_evidence_audit import (
    CANONICAL_STAGE6_ASSET_PATH,
    EXPECTED_STAGE6_CORPUS_ID,
    sha1_file,
)


def _args(tmp_path: Path, *, expected_chunks_sha256: str | None) -> argparse.Namespace:
    seed_path = tmp_path / "seed.jsonl"
    results_path = tmp_path / "results.jsonl"
    attestation_path = tmp_path / "full.attestation.json"
    chunks_path = tmp_path / "chunks.jsonl"
    corpus_attestation_path = tmp_path / "stage6.attestation.json"
    for path in (seed_path, results_path, attestation_path, chunks_path):
        path.write_text("{}\n", encoding="utf-8")
    return argparse.Namespace(
        seed_path=seed_path,
        results_path=results_path,
        attestation_path=attestation_path,
        chunks_path=chunks_path,
        corpus_attestation_path=corpus_attestation_path,
        output=tmp_path / "audit.json",
        expected_chunks_sha256=expected_chunks_sha256,
    )


def test_explicit_hash_can_only_produce_diagnostic_compatibility(tmp_path: Path) -> None:
    args = _args(tmp_path, expected_chunks_sha256="a" * 64)
    content_report = {
        "schema_version": 1,
        "status": "READY",
        "proof_level": "CONTENT_ANCHORED",
        "blocking_reasons": [],
    }

    with (
        patch.object(
            historical_full_evidence_gate,
            "check_full_seed_assets",
            return_value={"review_contract_source": "attestation"},
        ),
        patch.object(
            historical_full_evidence_gate,
            "validate_stage6_corpus_attestation",
            side_effect=historical_full_evidence_gate.HistoricalEvidenceAuditError(
                "reviewed corpus sidecar is missing"
            ),
        ),
        patch.object(
            historical_full_evidence_gate,
            "audit_historical_evidence_files",
            return_value=content_report,
        ),
    ):
        report = historical_full_evidence_gate.run(args)

    assert report["status"] == "DIAGNOSTIC_CONTENT_COMPATIBLE"
    assert report["proof_level"] == "DIAGNOSTIC_ONLY"
    assert report["formal_release_ready"] is False
    assert report["diagnostic_only"] is True


def test_formal_ready_report_exposes_historical_release_identity(tmp_path: Path) -> None:
    args = _args(tmp_path, expected_chunks_sha256=None)
    chunks_path = tmp_path / CANONICAL_STAGE6_ASSET_PATH
    chunks_path.parent.mkdir(parents=True)
    chunks_path.write_text("{}\n", encoding="utf-8")
    args.chunks_path = chunks_path
    fixture_sha1 = sha1_file(chunks_path)
    content_report = {
        "schema_version": 1,
        "status": "READY",
        "proof_level": "CONTENT_ANCHORED",
        "blocking_reasons": [],
    }
    corpus_record = {
        "corpus_id": EXPECTED_STAGE6_CORPUS_ID,
        "asset_sha1": fixture_sha1,
        "asset_sha256": "b" * 64,
    }

    with (
        patch.object(
            historical_full_evidence_gate,
            "check_full_seed_assets",
            return_value={"review_contract_source": "attestation"},
        ),
        patch.object(
            historical_full_evidence_gate,
            "validate_stage6_corpus_attestation",
            return_value=corpus_record,
        ),
        patch.object(
            historical_full_evidence_gate,
            "audit_historical_evidence_files",
            return_value=content_report,
        ),
        patch.object(historical_full_evidence_gate, "REPO_ROOT", tmp_path),
        patch.object(
            historical_full_evidence_gate,
            "EXPECTED_STAGE6_LEGACY_SHA1",
            fixture_sha1,
        ),
    ):
        report = historical_full_evidence_gate.run(args)

    identity = report["historical_release_identity"]
    assert report["status"] == "READY"
    assert report["formal_release_ready"] is True
    assert report["diagnostic_only"] is False
    assert identity["corpus_id"] == EXPECTED_STAGE6_CORPUS_ID
    assert identity["expected_legacy_sha1"] == fixture_sha1
    assert identity["actual_legacy_sha1"] == fixture_sha1
    assert identity["candidate_path_is_canonical"] is True


def test_formal_gate_stops_before_content_audit_on_legacy_identity_mismatch(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path, expected_chunks_sha256=None)

    with (
        patch.object(
            historical_full_evidence_gate,
            "check_full_seed_assets",
            return_value={"review_contract_source": "attestation"},
        ),
        patch.object(
            historical_full_evidence_gate,
            "validate_stage6_corpus_attestation",
            side_effect=historical_full_evidence_gate.HistoricalEvidenceAuditError(
                "legacy SHA-1 mismatch"
            ),
        ),
        patch.object(
            historical_full_evidence_gate,
            "audit_historical_evidence_files",
        ) as content_audit,
    ):
        report = historical_full_evidence_gate.run(args)

    content_audit.assert_not_called()
    assert report["status"] == "BLOCKED"
    assert report["formal_release_ready"] is False
    assert report["historical_release_identity"]["legacy_sha1_matches"] is False


def test_failed_diagnostic_keeps_incomplete_proof_level(tmp_path: Path) -> None:
    args = _args(tmp_path, expected_chunks_sha256="a" * 64)
    content_report = {
        "schema_version": 1,
        "status": "BLOCKED",
        "proof_level": "INCOMPLETE",
        "blocking_reasons": ["content mismatch"],
    }

    with (
        patch.object(
            historical_full_evidence_gate,
            "check_full_seed_assets",
            return_value={"review_contract_source": "attestation"},
        ),
        patch.object(
            historical_full_evidence_gate,
            "validate_stage6_corpus_attestation",
            side_effect=historical_full_evidence_gate.HistoricalEvidenceAuditError(
                "reviewed corpus sidecar is missing"
            ),
        ),
        patch.object(
            historical_full_evidence_gate,
            "audit_historical_evidence_files",
            return_value=content_report,
        ),
    ):
        report = historical_full_evidence_gate.run(args)

    assert report["status"] == "BLOCKED"
    assert report["proof_level"] == "INCOMPLETE"
    assert report["formal_release_ready"] is False
