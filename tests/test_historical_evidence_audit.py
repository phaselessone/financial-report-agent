from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation.historical_evidence_audit import (
    HistoricalEvidenceAuditError,
    audit_historical_evidence,
    audit_historical_evidence_files,
    sha256_file,
    validate_stage6_corpus_attestation,
)


DOC_ID = "信达证券-电子行业周报-国产替代加速-1520ea65"
CHUNK_ID = f"{DOC_ID}-table_protected-c0004"
FILE_NAME = "信达证券-电子行业周报：国产替代加速.pdf"


def _seed(*, source_label: str = "artifact_citation_inferred") -> list[dict]:
    return [
        {
            "question_id": "fact_01",
            "gold_chunk_ids": [CHUNK_ID],
            "target_doc_keys": ["国产替代加速"],
            "must_abstain": False,
            "manual_review_required": False,
            "source_label": source_label,
        }
    ]


def _results(*, include_reference: bool = True) -> list[dict]:
    citations = []
    if include_reference:
        citations.append(
            {
                "chunk_id": CHUNK_ID,
                "doc_id": DOC_ID,
                "file_name": FILE_NAME,
                "page_start": 2,
                "page_end": 2,
                "snippet": "本周北美重要个股出现分化。SEMICON China 2026 在上海开幕。",
            }
        )
    return [{"question_id": "fact_01", "citations": citations, "selected_evidence": []}]


def _chunks(*, text: str | None = None, doc_id: str = DOC_ID) -> list[dict]:
    return [
        {
            "chunk_id": CHUNK_ID,
            "doc_id": doc_id,
            "file_name": FILE_NAME,
            "page_start": 2,
            "page_end": 2,
            "text": text
            or "章节路径: 行业周报\n本周北美重要个股出现分化。SEMICON China 2026 在上海开幕。",
        }
    ]


def _audit(
    *,
    seed: list[dict] | None = None,
    results: list[dict] | None = None,
    chunks: list[dict] | None = None,
    reviewed_attestation_validated: bool = False,
    actual_hash: str = "a" * 64,
    expected_hash: str | None = "a" * 64,
) -> dict:
    return audit_historical_evidence(
        seed_rows=seed if seed is not None else _seed(),
        historical_result_rows=results if results is not None else _results(),
        candidate_chunks=chunks if chunks is not None else _chunks(),
        candidate_corpus_sha256=actual_hash,
        expected_corpus_sha256=expected_hash,
        reviewed_attestation_validated=reviewed_attestation_validated,
        expected_question_count=None,
    )


def test_content_anchored_candidate_is_ready_only_with_matching_corpus_hash() -> None:
    report = _audit()

    assert report["status"] == "READY"
    assert report["proof_level"] == "CONTENT_ANCHORED"
    assert report["summary"]["verified_content_gold_chunk_count"] == 1
    assert report["chunks"][0]["status"] == "VERIFIED"


def test_exact_chunk_id_with_shifted_text_is_blocked() -> None:
    report = _audit(chunks=_chunks(text="信达证券股份有限公司 北京市西城区"))

    assert report["status"] == "BLOCKED"
    assert report["summary"]["exact_id_present_count"] == 1
    assert report["summary"]["blocked_gold_chunk_count"] == 1
    assert "historical_snippet_mismatch" in report["chunks"][0]["issues"]


def test_page_and_document_identity_are_part_of_the_same_reference_proof() -> None:
    chunks = _chunks(doc_id="wrong-doc")
    chunks[0]["page_start"] = 8
    chunks[0]["page_end"] = 8

    report = _audit(chunks=chunks)

    assert report["status"] == "BLOCKED"
    assert "candidate_doc_id_mismatch" in report["chunks"][0]["issues"]
    assert "historical_page_mismatch" in report["chunks"][0]["issues"]


def test_unanchored_badcase_gold_remains_explicit_and_requires_attestation() -> None:
    seed = _seed(source_label="artifact_badcase_gold")
    results = _results(include_reference=False)

    blocked = _audit(seed=seed, results=results)
    ready = _audit(
        seed=seed,
        results=results,
        reviewed_attestation_validated=True,
    )

    assert blocked["status"] == "BLOCKED"
    assert "review_attestation_not_validated" in blocked["chunks"][0]["issues"]
    assert ready["status"] == "READY"
    assert ready["proof_level"] == "MIXED_CONTENT_AND_REVIEW_ATTESTATION"
    assert ready["chunks"][0]["status"] == "ATTESTED_UNANCHORED"
    assert ready["summary"]["verified_content_gold_chunk_count"] == 0
    assert ready["summary"]["attested_unanchored_gold_chunk_count"] == 1


def test_unanchored_inferred_gold_cannot_borrow_badcase_attestation_policy() -> None:
    report = _audit(
        results=_results(include_reference=False),
        reviewed_attestation_validated=True,
    )

    assert report["status"] == "BLOCKED"
    assert "unanchored_gold_is_not_badcase_review_source" in report["chunks"][0]["issues"]


def test_candidate_corpus_must_be_hash_pinned() -> None:
    missing_pin = _audit(expected_hash=None)
    mismatch = _audit(expected_hash="b" * 64)

    assert missing_pin["status"] == "BLOCKED"
    assert missing_pin["candidate_corpus"]["hash_pinned"] is False
    assert "expected candidate corpus SHA-256 is required" in missing_pin["blocking_reasons"]
    assert mismatch["status"] == "BLOCKED"
    assert "SHA-256 mismatch" in mismatch["blocking_reasons"][0]


def test_missing_exact_gold_chunk_is_not_treated_as_a_materialization_skip() -> None:
    report = _audit(chunks=[])

    assert report["status"] == "BLOCKED"
    assert report["summary"]["exact_id_present_count"] == 0
    assert report["chunks"][0]["issues"] == ["missing_exact_gold_chunk_id"]


def test_seed_document_mapping_conflict_is_rejected() -> None:
    seed = _seed()
    seed[0]["target_doc_keys"] = []

    with pytest.raises(HistoricalEvidenceAuditError, match="gold documents"):
        _audit(seed=seed)


def test_file_wrapper_reports_missing_candidate_without_fabricating_readiness(tmp_path: Path) -> None:
    seed_path = tmp_path / "seed.jsonl"
    results_path = tmp_path / "results.jsonl"
    seed_path.write_text(json.dumps(_seed()[0], ensure_ascii=False) + "\n", encoding="utf-8")
    results_path.write_text(json.dumps(_results()[0], ensure_ascii=False) + "\n", encoding="utf-8")

    report = audit_historical_evidence_files(
        seed_path=seed_path,
        historical_results_path=results_path,
        candidate_chunks_path=tmp_path / "missing-chunks.jsonl",
        expected_corpus_sha256="a" * 64,
        reviewed_attestation_validated=True,
        expected_question_count=None,
    )

    assert report["status"] == "BLOCKED"
    assert report["blocking_reasons"] == [
        f"missing candidate chunks asset: {tmp_path / 'missing-chunks.jsonl'}"
    ]


def test_file_wrapper_records_actual_candidate_hash(tmp_path: Path) -> None:
    seed_path = tmp_path / "seed.jsonl"
    results_path = tmp_path / "results.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    seed_path.write_text(json.dumps(_seed()[0], ensure_ascii=False) + "\n", encoding="utf-8")
    results_path.write_text(json.dumps(_results()[0], ensure_ascii=False) + "\n", encoding="utf-8")
    chunks_path.write_text(json.dumps(_chunks()[0], ensure_ascii=False) + "\n", encoding="utf-8")
    expected_hash = sha256_file(chunks_path)

    report = audit_historical_evidence_files(
        seed_path=seed_path,
        historical_results_path=results_path,
        candidate_chunks_path=chunks_path,
        expected_corpus_sha256=expected_hash,
        reviewed_attestation_validated=False,
        expected_question_count=None,
    )

    assert report["status"] == "READY"
    assert report["inputs"]["candidate_chunks_sha256"] == expected_hash
    assert report["candidate_corpus"]["hash_pinned"] is True


def _write_stage6_attestation(
    tmp_path: Path,
    *,
    seed_path: Path,
    results_path: Path,
    chunks_path: Path,
) -> Path:
    path = tmp_path / "stage6.attestation.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "corpus_id": "stage6-fixture-v1",
                "benchmark_profile": "historical-full-raw",
                "asset": {
                    "path": chunks_path.name,
                    "sha256": sha256_file(chunks_path),
                    "format": "chunks-jsonl",
                    "chunk_count": 1,
                },
                "source": {
                    "origin": "fixture-vault",
                    "owner": "asset-owner-1",
                    "acquired_at": "2026-08-24T00:00:00Z",
                    "source_ref": "vault://stage6/v1",
                },
                "review": {
                    "status": "reviewed",
                    "reviewer_id": "reviewer-1",
                    "review_batch": "batch-1",
                    "reviewed_at": "2026-08-24T01:00:00Z",
                },
                "benchmark_binding": {
                    "seed_sha256": sha256_file(seed_path),
                    "historical_results_sha256": sha256_file(results_path),
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_stage6_attestation_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    seed_path = tmp_path / "seed.jsonl"
    results_path = tmp_path / "results.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    seed_path.write_text(json.dumps(_seed()[0], ensure_ascii=False) + "\n", encoding="utf-8")
    results_path.write_text(json.dumps(_results()[0], ensure_ascii=False) + "\n", encoding="utf-8")
    chunks_path.write_text(json.dumps(_chunks()[0], ensure_ascii=False) + "\n", encoding="utf-8")
    attestation_path = _write_stage6_attestation(
        tmp_path,
        seed_path=seed_path,
        results_path=results_path,
        chunks_path=chunks_path,
    )
    return attestation_path, seed_path, results_path, chunks_path


def test_stage6_attestation_binds_provenance_review_and_all_three_assets(tmp_path: Path) -> None:
    attestation_path, seed_path, results_path, chunks_path = _write_stage6_attestation_fixture(
        tmp_path
    )

    record = validate_stage6_corpus_attestation(
        attestation_path,
        repo_root=tmp_path,
        candidate_chunks_path=chunks_path,
        seed_path=seed_path,
        historical_results_path=results_path,
    )

    assert record["corpus_id"] == "stage6-fixture-v1"
    assert record["asset_sha256"] == sha256_file(chunks_path)
    assert record["source"]["owner"] == "asset-owner-1"
    assert record["review"]["status"] == "reviewed"
    assert record["benchmark_binding"]["seed_sha256"] == sha256_file(seed_path)


def test_stage6_attestation_rejects_unreviewed_release(tmp_path: Path) -> None:
    attestation_path, seed_path, results_path, chunks_path = _write_stage6_attestation_fixture(
        tmp_path
    )
    payload = json.loads(attestation_path.read_text(encoding="utf-8"))
    payload["review"]["status"] = "pending"
    attestation_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(HistoricalEvidenceAuditError, match="review.status"):
        validate_stage6_corpus_attestation(
            attestation_path,
            repo_root=tmp_path,
            candidate_chunks_path=chunks_path,
            seed_path=seed_path,
            historical_results_path=results_path,
        )


def test_stage6_attestation_rejects_benchmark_binding_drift(tmp_path: Path) -> None:
    attestation_path, seed_path, results_path, chunks_path = _write_stage6_attestation_fixture(
        tmp_path
    )
    payload = json.loads(attestation_path.read_text(encoding="utf-8"))
    payload["benchmark_binding"]["seed_sha256"] = "0" * 64
    attestation_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(HistoricalEvidenceAuditError, match="seed SHA-256 mismatch"):
        validate_stage6_corpus_attestation(
            attestation_path,
            repo_root=tmp_path,
            candidate_chunks_path=chunks_path,
            seed_path=seed_path,
            historical_results_path=results_path,
        )
