from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

import scripts.phase0_gate as phase0_gate

from scripts.phase0_gate import (
    GateError,
    SMOKE_GROUPS,
    build_readiness_report,
    check_full_seed_assets,
    collect_baseline,
)


ABSTAIN_IDS = {
    "compare_abstain_01",
    "compare_abstain_02",
    "fact_abstain_01",
    "fact_abstain_02",
    "inductive_abstain_01",
}
CANONICAL_PROFILE = "historical-full-raw"
SOURCE_MANIFEST_SHA256 = "c" * 64


def _write_assets(tmp_path: Path) -> tuple[Path, Path]:
    seed = tmp_path / "seed.jsonl"
    results = tmp_path / "results.jsonl"
    rows = []
    types = ["fact"] * 20 + ["comparison"] * 15 + ["inductive"] * 15
    regular_index = 0
    reserved_ids = list(sorted(ABSTAIN_IDS))
    for index, question_type in enumerate(types):
        if index < len(reserved_ids):
            question_id = reserved_ids[index]
        else:
            regular_index += 1
            question_id = f"q{regular_index:02d}"
        rows.append(
            {
                "question_id": question_id,
                "question_type": question_type,
                "must_abstain": question_id in ABSTAIN_IDS,
                "manual_review_required": False,
                "review_status": "reviewed",
                "reviewer_id": "reviewer-1",
                "review_batch": "full-seed-v1",
                "reviewed_at": "2026-08-24T00:00:00Z",
                "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
                "benchmark_profile": CANONICAL_PROFILE,
            }
        )
    seed.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    results.write_text(
        "".join(
            json.dumps(
                {
                    "question_id": row["question_id"],
                    "review_status": "reviewed",
                    "reviewer_id": "reviewer-1",
                    "review_batch": "full-results-v1",
                    "reviewed_at": "2026-08-24T00:00:00Z",
                    "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
                    "benchmark_profile": CANONICAL_PROFILE,
                }
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    return seed, results


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_attestation(
    tmp_path: Path,
    seed: Path,
    results: Path,
    *,
    review_status: str = "reviewed",
) -> Path:
    provenance = tmp_path / "full_benchmark_provenance.md"
    provenance.write_text(
        "Canonical source: historical results. Manual review completed 2026-04-05.\n",
        encoding="utf-8",
    )
    restoration_report = tmp_path / "answer_eval_seed_full_restore_report.json"
    restoration_report.write_text(
        json.dumps(
            {
                "row_count": 50,
                "manual_review_required_count": 0,
                "must_abstain_count": 5,
                "question_type_distribution": {
                    "comparison": 15,
                    "fact": 20,
                    "inductive": 15,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    attestation = tmp_path / "historical-full-raw.attestation.json"
    attestation.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "benchmark_id": "historical-full-raw-v1",
                "benchmark_profile": CANONICAL_PROFILE,
                "review": {
                    "status": review_status,
                    "attested_by": "asset-custodian-1",
                    "review_batch": "historical-full-restore-20260405",
                    "reviewed_at": "2026-04-05",
                    "manual_review_required_count": 0,
                },
                "source": {
                    "origin_repository": "authorized://financial-report-agent",
                    "canonical_results_artifact": str(results),
                    "provenance_document": {
                        "path": str(provenance),
                        "git_commit": "a" * 40,
                        "normalized_sha256": hashlib.sha256(
                            provenance.read_text(encoding="utf-8").encode("utf-8")
                        ).hexdigest(),
                    },
                },
                "assets": {
                    "seed": {"path": str(seed), "sha256": _hash(seed), "rows": 50},
                    "historical_results": {
                        "path": str(results),
                        "sha256": _hash(results),
                        "rows": 50,
                    },
                },
                "evidence": {
                    "restoration_report": {
                        "path": str(restoration_report),
                        "sha256": _hash(restoration_report),
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return attestation


def _remove_inline_review_metadata(path: Path) -> None:
    review_fields = {
        "review_status",
        "reviewer_id",
        "review_batch",
        "reviewed_at",
        "source_manifest_sha256",
        "benchmark_profile",
    }
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        for field in review_fields:
            row.pop(field, None)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_phase0_baseline_hashes_the_dependency_lockfile(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    for name, content in (
        ("requirements.txt", "example>=1\n"),
        ("requirements-dev.txt", "pytest>=8\n"),
        ("requirements.lock", "example==1.2.3\n"),
    ):
        (repo / name).write_text(content, encoding="utf-8")
    monkeypatch.setattr(phase0_gate, "REPO_ROOT", repo)

    report = phase0_gate.collect_baseline(tmp_path / "baseline.json")

    assert report["configuration_hashes"]["requirements.lock"] == _hash(repo / "requirements.lock")


def test_smoke_groups_cover_required_capabilities() -> None:
    assert set(SMOKE_GROUPS) == {
        "claims",
        "calculator",
        "agent_flow",
        "structured",
        "trajectory",
        "evidence",
        "baseline_regression",
    }
    assert all(path.startswith("tests/") for paths in SMOKE_GROUPS.values() for path in paths)


def test_full_seed_missing_is_a_failure(tmp_path: Path) -> None:
    with pytest.raises(GateError, match="missing external"):
        check_full_seed_assets(tmp_path / "missing-seed.jsonl", tmp_path / "missing-results.jsonl")


def test_full_seed_validates_contract_and_hashes(tmp_path: Path) -> None:
    seed, results = _write_assets(tmp_path)
    seed_hash = hashlib.sha256(seed.read_bytes()).hexdigest()
    results_hash = hashlib.sha256(results.read_bytes()).hexdigest()
    report = check_full_seed_assets(
        seed,
        results,
        expected_seed_sha256=seed_hash,
        expected_results_sha256=results_hash,
    )
    assert report["status"] == "ready"
    assert report["seed"]["rows"] == 50
    assert report["question_type_distribution"] == {"comparison": 15, "fact": 20, "inductive": 15}


def test_full_seed_rejects_hash_or_identity_mismatch(tmp_path: Path) -> None:
    seed, results = _write_assets(tmp_path)
    with pytest.raises(GateError, match="SHA-256 mismatch"):
        check_full_seed_assets(
            seed,
            results,
            expected_seed_sha256="0" * 64,
            expected_results_sha256=_hash(results),
        )
    result_rows = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines()]
    result_rows[0]["question_id"] = "different"
    results.write_text(
        "".join(json.dumps(row) + "\n" for row in result_rows),
        encoding="utf-8",
    )
    with pytest.raises(GateError, match="question_ids"):
        check_full_seed_assets(
            seed,
            results,
            expected_seed_sha256=_hash(seed),
            expected_results_sha256=_hash(results),
        )


def test_full_seed_rejects_duplicate_historical_result_ids(tmp_path: Path) -> None:
    seed, results = _write_assets(tmp_path)
    result_rows = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines()]
    result_rows[1]["question_id"] = result_rows[0]["question_id"]
    results.write_text(
        "".join(json.dumps(row) + "\n" for row in result_rows),
        encoding="utf-8",
    )
    with pytest.raises(GateError, match="historical result question_id"):
        check_full_seed_assets(
            seed,
            results,
            expected_seed_sha256=_hash(seed),
            expected_results_sha256=_hash(results),
        )


def test_full_seed_requires_pinned_hashes_and_positive_review_identity(tmp_path: Path) -> None:
    seed, results = _write_assets(tmp_path)
    with pytest.raises(GateError, match="expected.*SHA-256"):
        check_full_seed_assets(seed, results)

    seed_rows = [json.loads(line) for line in seed.read_text(encoding="utf-8").splitlines()]
    seed_rows[0].pop("review_status")
    seed.write_text("".join(json.dumps(row) + "\n" for row in seed_rows), encoding="utf-8")
    with pytest.raises(GateError, match="review_status"):
        check_full_seed_assets(
            seed,
            results,
            expected_seed_sha256=_hash(seed),
            expected_results_sha256=_hash(results),
        )


def test_full_seed_accepts_immutable_assets_bound_by_review_attestation(tmp_path: Path) -> None:
    seed, results = _write_assets(tmp_path)
    _remove_inline_review_metadata(seed)
    _remove_inline_review_metadata(results)
    attestation = _write_attestation(tmp_path, seed, results)

    report = check_full_seed_assets(seed, results, attestation_path=attestation)

    assert report["status"] == "ready"
    assert report["attestation"]["review_status"] == "reviewed"
    assert report["attestation"]["attested_by"] == "asset-custodian-1"
    assert report["seed"]["sha256"] == _hash(seed)
    assert report["historical_results"]["sha256"] == _hash(results)


def test_full_seed_attestation_is_path_hash_and_review_bound(tmp_path: Path) -> None:
    seed, results = _write_assets(tmp_path)
    _remove_inline_review_metadata(seed)
    _remove_inline_review_metadata(results)
    attestation = _write_attestation(tmp_path, seed, results)

    with pytest.raises(GateError, match="explicit seed SHA-256 disagrees"):
        check_full_seed_assets(
            seed,
            results,
            attestation_path=attestation,
            expected_seed_sha256="0" * 64,
        )

    manifest = json.loads(attestation.read_text(encoding="utf-8"))
    manifest["review"]["status"] = "pending"
    attestation.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    with pytest.raises(GateError, match="review.status"):
        check_full_seed_assets(seed, results, attestation_path=attestation)


def test_full_seed_attestation_rejects_tampered_review_evidence(tmp_path: Path) -> None:
    seed, results = _write_assets(tmp_path)
    _remove_inline_review_metadata(seed)
    _remove_inline_review_metadata(results)
    attestation = _write_attestation(tmp_path, seed, results)
    manifest = json.loads(attestation.read_text(encoding="utf-8"))
    restoration_report = Path(manifest["evidence"]["restoration_report"]["path"])
    restoration_report.write_text("{}\n", encoding="utf-8")

    with pytest.raises(GateError, match="restoration report SHA-256 mismatch"):
        check_full_seed_assets(seed, results, attestation_path=attestation)


def test_full_seed_rejects_noncanonical_profile(tmp_path: Path) -> None:
    seed, results = _write_assets(tmp_path)
    rows = [json.loads(line) for line in seed.read_text(encoding="utf-8").splitlines()]
    rows[0]["benchmark_profile"] = "current-dev"
    seed.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(GateError, match="benchmark_profile"):
        check_full_seed_assets(
            seed,
            results,
            expected_seed_sha256=_hash(seed),
            expected_results_sha256=_hash(results),
        )


def test_baseline_report_is_non_secret_and_reproducible_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("MODEL_OFFLINE", "1")
    monkeypatch.setenv("LLM_API_KEY", "must-not-leak")
    output = tmp_path / "baseline.json"
    report = collect_baseline(output)
    serialized = output.read_text(encoding="utf-8")
    assert report["schema_version"] == 1
    assert report["configuration_environment"] == {
        "LLM_PROVIDER": "fake",
        "MODEL_OFFLINE": "1",
    }
    assert "must-not-leak" not in serialized
    assert report["external_full_seed"]["status"] == "not_checked"
    assert "deterministic_smoke" in report["commands"]
    assert "historical_full_evidence" in report["commands"]


def test_cli_missing_full_seed_returns_nonzero(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "phase0_gate.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "check-full-seed",
            "--seed-path",
            str(tmp_path / "missing-seed"),
            "--results-path",
            str(tmp_path / "missing-results"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "PHASE0_GATE_FAILED" in completed.stderr


def _write_labels(path: Path, *, valid: bool = True, single_class: bool = False) -> Path:
    rows = []
    for index in range(20):
        entailed = single_class or index < 10
        rows.append(
            {
                "sample_id": f"s{index}",
                "claim_id": f"c{index}",
                "evidence_id": f"e{index}",
                "claim_sha256": hashlib.sha256(
                    f"reviewed claim {index}".encode("utf-8")
                ).hexdigest(),
                "evidence_sha256": hashlib.sha256(
                    f"reviewed evidence {index}".encode("utf-8")
                ).hexdigest(),
                "score": 0.95 if entailed else 0.10,
                "label": "ENTAILED" if entailed else "CONTRADICTED",
                "source_ref": f"review://batch-1/{index}",
                "review_status": "reviewed" if valid or index else "pending",
                "reviewer_id": "reviewer-1",
                "reviewed_at": "2026-08-24T00:00:00Z",
                "scorer_kind": "directional_nli",
                "scorer_model": "fixture-nli",
                "scorer_revision": "a" * 40,
                "scorer_config_sha256": "b" * 64,
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_readiness_report_is_blocked_for_missing_assets(tmp_path: Path) -> None:
    output = tmp_path / "readiness.json"
    report = build_readiness_report(
        seed_path=tmp_path / "missing-seed.jsonl",
        results_path=tmp_path / "missing-results.jsonl",
        labels_path=tmp_path / "missing-labels.jsonl",
        output_path=output,
    )
    assert report["status"] == "BLOCKED"
    assert set(report["blocked_assets"]) == {"full_seed", "historical_results", "semantic_labels"}
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "BLOCKED"


def test_readiness_report_requires_reviewed_labels_and_full_assets(tmp_path: Path) -> None:
    seed, results = _write_assets(tmp_path)
    labels = _write_labels(tmp_path / "labels.jsonl")
    report = build_readiness_report(
        seed_path=seed,
        results_path=results,
        labels_path=labels,
        output_path=tmp_path / "readiness.json",
        expected_seed_sha256=_hash(seed),
        expected_results_sha256=_hash(results),
        expected_labels_sha256=_hash(labels),
    )
    assert report["status"] == "READY"
    assert all(item["status"] == "READY" for item in report["assets"].values())
    assert report["assets"]["semantic_labels"]["rows"] == 20
    assert report["semantic_calibration"]["calibrated"] is True


def test_readiness_uses_attestation_hashes_for_immutable_full_assets(tmp_path: Path) -> None:
    seed, results = _write_assets(tmp_path)
    _remove_inline_review_metadata(seed)
    _remove_inline_review_metadata(results)
    attestation = _write_attestation(tmp_path, seed, results)
    labels = _write_labels(tmp_path / "labels.jsonl")

    report = build_readiness_report(
        seed_path=seed,
        results_path=results,
        attestation_path=attestation,
        labels_path=labels,
        output_path=tmp_path / "readiness.json",
        expected_labels_sha256=_hash(labels),
    )

    assert report["status"] == "READY"
    assert report["full_seed_validation"]["review_contract_source"] == "attestation"
    assert report["assets"]["full_seed"]["status"] == "READY"
    assert report["assets"]["historical_results"]["status"] == "READY"


def test_readiness_report_blocks_invalid_labels_even_with_full_assets(tmp_path: Path) -> None:
    seed, results = _write_assets(tmp_path)
    labels = _write_labels(tmp_path / "labels.jsonl", valid=False)
    report = build_readiness_report(
        seed_path=seed,
        results_path=results,
        labels_path=labels,
        output_path=tmp_path / "readiness.json",
        expected_seed_sha256=_hash(seed),
        expected_results_sha256=_hash(results),
        expected_labels_sha256=_hash(labels),
    )
    assert report["status"] == "BLOCKED"
    assert report["assets"]["semantic_labels"]["status"] == "BLOCKED"
    assert "review_status" in report["assets"]["semantic_labels"]["blocking_reason"]


def test_readiness_blocks_unpinned_assets_and_uncalibrated_labels(tmp_path: Path) -> None:
    seed, results = _write_assets(tmp_path)
    labels = _write_labels(tmp_path / "labels.jsonl", single_class=True)

    unpinned = build_readiness_report(
        seed_path=seed,
        results_path=results,
        labels_path=labels,
        output_path=tmp_path / "unpinned.json",
    )
    assert unpinned["status"] == "BLOCKED"
    assert set(unpinned["blocked_assets"]) == {"full_seed", "historical_results", "semantic_labels"}

    uncalibrated = build_readiness_report(
        seed_path=seed,
        results_path=results,
        labels_path=labels,
        output_path=tmp_path / "uncalibrated.json",
        expected_seed_sha256=_hash(seed),
        expected_results_sha256=_hash(results),
        expected_labels_sha256=_hash(labels),
    )
    assert uncalibrated["status"] == "BLOCKED"
    assert uncalibrated["semantic_calibration"]["calibrated"] is False
    assert "semantic_calibration" in uncalibrated["blocking_reasons"]


def test_readiness_cli_returns_blocked_and_writes_report(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "phase0_gate.py"
    output = tmp_path / "readiness.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "readiness",
            "--seed-path",
            str(tmp_path / "missing-seed"),
            "--results-path",
            str(tmp_path / "missing-results"),
            "--labels-path",
            str(tmp_path / "missing-labels"),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "PHASE0_READINESS_BLOCKED" in completed.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "BLOCKED"
