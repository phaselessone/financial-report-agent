from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

import src.evaluation.eval_bundle as eval_bundle_module
from src.evaluation.eval_bundle import (
    EvalBundleIntegrityError,
    build_corpus_asset_manifest,
    read_eval_bundle,
    read_eval_bundle_reference,
    require_comparable_eval_bundles,
    write_eval_bundle,
    write_eval_bundle_reference,
)
from src.evaluation.run_identity import (
    IncompatibleRunIdentityError,
    RunIdentity,
    canonical_sha256,
    hash_case_ids,
)


def make_identity(**overrides: object) -> RunIdentity:
    values: dict[str, object] = {
        "benchmark_profile": "contract-v1",
        "benchmark_version": "1.0",
        "benchmark_hash": "benchmark-sha256",
        "case_ids_hash": hash_case_ids(["case-01", "case-02"]),
        "case_count": 2,
        "corpus_hash": "corpus-sha256",
        "model": "model-x",
        "provider": "provider-x",
        "model_revision": "rev-1",
        "temperature": 0.0,
        "prompt_version": "prompt-v2",
        "budgets": {"max_steps": 12, "max_llm_calls": 6},
        "feature_flags": {
            "runtime": {"strict_trace": True},
            "treatments": {"pipeline": "baseline"},
        },
        "git_commit": "abc123",
        "source_manifest_hash": "source-sha256",
    }
    values.update(overrides)
    return RunIdentity(**values)


def make_agentic_identity(**overrides: object) -> RunIdentity:
    values: dict[str, object] = {
        "feature_flags": {
            "runtime": {"strict_trace": True},
            "treatments": {"pipeline": "agentic"},
        }
    }
    values.update(overrides)
    return make_identity(**values)


def make_case_rows() -> list[dict[str, object]]:
    return [
        {"case_id": "case-01", "answer": "10%", "score": 1.0},
        {"case_id": "case-02", "answer": "20%", "score": 0.0},
    ]


def test_run_identity_has_canonical_serialization_and_stable_run_id() -> None:
    first = make_identity(
        budgets={"max_steps": 12, "max_llm_calls": 6},
        feature_flags={
            "runtime": {"strict_trace": True, "calculator": True},
            "treatments": {"pipeline": "baseline"},
        },
    )
    second = make_identity(
        budgets={"max_llm_calls": 6, "max_steps": 12},
        feature_flags={
            "treatments": {"pipeline": "baseline"},
            "runtime": {"calculator": True, "strict_trace": True},
        },
    )

    assert first.canonical_json() == second.canonical_json()
    assert first.identity_hash == second.identity_hash
    assert first.identity_hash == "28c766e0fbd086b174f1a4ed3ddb426939331eec681e6e676ec05bbd644be6b7"
    assert first.run_id == second.run_id
    assert first.run_id == "run-" + first.identity_hash[:20]
    assert json.loads(first.canonical_json()) == first.to_dict()
    assert set(first.to_dict()) == {
        "benchmark_profile",
        "benchmark_version",
        "benchmark_hash",
        "case_ids_hash",
        "case_count",
        "corpus_hash",
        "model",
        "provider",
        "model_revision",
        "temperature",
        "prompt_version",
        "budgets",
        "feature_flags",
        "git_commit",
        "source_manifest_hash",
    }


def test_run_identity_comparison_allows_only_explicit_treatments_to_differ() -> None:
    baseline = make_identity()
    agentic = make_identity(
        feature_flags={
            "runtime": {"strict_trace": True},
            "treatments": {"pipeline": "agentic", "claim_verification": True},
        }
    )

    baseline.assert_comparable(agentic)

    changed_runtime = make_identity(
        feature_flags={
            "runtime": {"strict_trace": False},
            "treatments": {"pipeline": "agentic"},
        }
    )
    with pytest.raises(IncompatibleRunIdentityError, match="feature_flags"):
        baseline.assert_comparable(changed_runtime)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("case_count", 3),
        ("case_ids_hash", "different-cases"),
        ("benchmark_hash", "different-benchmark"),
        ("corpus_hash", "different-corpus"),
        ("model", "different-model"),
        ("provider", "different-provider"),
        ("model_revision", "different-revision"),
        ("temperature", 0.2),
        ("prompt_version", "different-prompt"),
        ("budgets", {"max_steps": 99, "max_llm_calls": 6}),
        ("git_commit", "different-commit"),
        ("source_manifest_hash", "different-source"),
    ],
)
def test_run_identity_rejects_comparison_identity_mismatch(field: str, value: object) -> None:
    baseline = make_identity()
    changed = replace(baseline, **{field: value})

    with pytest.raises(IncompatibleRunIdentityError) as exc_info:
        baseline.assert_comparable(changed)

    assert field in exc_info.value.mismatched_fields


def test_hash_case_ids_is_order_independent_but_rejects_duplicates() -> None:
    assert hash_case_ids(["case-01", "case-02"]) == hash_case_ids(["case-02", "case-01"])
    with pytest.raises(ValueError, match="duplicate case id"):
        hash_case_ids(["case-01", "case-01"])


def test_eval_bundle_round_trip_uses_run_id_directory_and_contract_files(tmp_path: Path) -> None:
    identity = make_identity()
    bundle = write_eval_bundle(
        tmp_path,
        identity=identity,
        config={"profile": "baseline-rag", "seed": 7},
        metrics={"Answer Accuracy": 0.5},
        per_case=make_case_rows(),
        trajectories=[{"case_id": "case-01", "events": []}],
        metadata={"mode": "baseline", "review_status": "synthetic_not_human_reviewed"},
    )

    assert bundle.path == tmp_path / identity.run_id
    assert {path.name for path in bundle.path.iterdir()} == {
        "config.json",
        "metrics.json",
        "per_case.jsonl",
        "trajectories.jsonl",
        "metadata.json",
    }
    loaded = read_eval_bundle(bundle.path)
    assert loaded.identity == identity
    assert loaded.config["seed"] == 7
    assert loaded.metrics["Answer Accuracy"] == 0.5
    assert [row["case_id"] for row in loaded.per_case] == ["case-01", "case-02"]
    assert loaded.metadata["run_id"] == identity.run_id


def test_eval_bundle_serialization_failure_never_publishes_final_run_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = make_identity()

    def fail_jsonl_write(_path: Path, _rows: object) -> None:
        raise OSError("simulated serialization failure")

    monkeypatch.setattr(eval_bundle_module, "write_jsonl", fail_jsonl_write)

    with pytest.raises(OSError, match="simulated serialization failure"):
        write_eval_bundle(
            tmp_path,
            identity=identity,
            config={},
            metrics={},
            per_case=make_case_rows(),
            trajectories=[],
        )

    assert not (tmp_path / identity.run_id).exists()


def test_explicit_corpus_asset_manifest_rejects_changed_asset_before_publish(
    tmp_path: Path,
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    facts_path = tmp_path / "facts.jsonl"
    aliases_path = tmp_path / "aliases.json"
    chunks_path.write_text('{"chunk_id":"c1"}\n', encoding="utf-8")
    facts_path.write_text('{"fact_id":"f1"}\n', encoding="utf-8")
    aliases_path.write_text('{"Company":["Company"]}', encoding="utf-8")
    manifest = build_corpus_asset_manifest(
        chunks_path=chunks_path,
        facts_path=facts_path,
        aliases_path=aliases_path,
    )
    identity = make_identity(corpus_hash=manifest["corpus_hash"])
    facts_path.write_text('{"fact_id":"changed"}\n', encoding="utf-8")

    with pytest.raises(EvalBundleIntegrityError, match="structured_facts"):
        write_eval_bundle(
            tmp_path / "eval",
            identity=identity,
            config={},
            metrics={},
            per_case=make_case_rows(),
            trajectories=[],
            corpus_asset_manifest=manifest,
        )

    assert not (tmp_path / "eval" / identity.run_id).exists()


def test_eval_bundle_reference_round_trip_validates_expected_run_id(tmp_path: Path) -> None:
    bundle = write_eval_bundle(
        tmp_path / "eval",
        identity=make_identity(),
        config={},
        metrics={},
        per_case=make_case_rows(),
        trajectories=[{"case_id": "case-01"}, {"case_id": "case-02"}],
    )
    reference_path = tmp_path / "reports" / "baseline_bundle.json"
    write_eval_bundle_reference(reference_path, bundle)
    assert read_eval_bundle_reference(reference_path).path == bundle.path

    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    reference["run_id"] = "run-tampered"
    reference_path.write_text(json.dumps(reference), encoding="utf-8")
    with pytest.raises(EvalBundleIntegrityError, match="reference run_id"):
        read_eval_bundle_reference(reference_path)


def test_eval_bundle_write_rejects_case_count_and_hash_mismatch(tmp_path: Path) -> None:
    with pytest.raises(EvalBundleIntegrityError, match="case_count"):
        write_eval_bundle(
            tmp_path,
            identity=make_identity(case_count=3),
            config={},
            metrics={},
            per_case=make_case_rows(),
            trajectories=[],
        )

    with pytest.raises(EvalBundleIntegrityError, match="case_ids_hash"):
        write_eval_bundle(
            tmp_path,
            identity=make_identity(case_ids_hash="wrong"),
            config={},
            metrics={},
            per_case=make_case_rows(),
            trajectories=[],
        )


def test_agentic_eval_bundle_rejects_missing_trajectory_case(tmp_path: Path) -> None:
    with pytest.raises(EvalBundleIntegrityError, match="trajectory case_count mismatch"):
        write_eval_bundle(
            tmp_path,
            identity=make_agentic_identity(),
            config={},
            metrics={},
            per_case=make_case_rows(),
            trajectories=[{"case_id": "case-01"}],
            metadata={"mode": "agentic"},
        )


def test_agentic_eval_bundle_rejects_duplicate_trajectory_case_id(tmp_path: Path) -> None:
    with pytest.raises(EvalBundleIntegrityError, match="duplicate trajectory case id"):
        write_eval_bundle(
            tmp_path,
            identity=make_agentic_identity(),
            config={},
            metrics={},
            per_case=make_case_rows(),
            trajectories=[{"case_id": "case-01"}, {"case_id": "case-01"}],
            metadata={"mode": "agentic"},
        )


def test_agentic_eval_bundle_rejects_trajectory_case_set_mismatch(tmp_path: Path) -> None:
    with pytest.raises(EvalBundleIntegrityError, match="trajectory case_ids_hash mismatch"):
        write_eval_bundle(
            tmp_path,
            identity=make_agentic_identity(),
            config={},
            metrics={},
            per_case=make_case_rows(),
            trajectories=[{"case_id": "case-01"}, {"case_id": "case-03"}],
            metadata={"mode": "agentic"},
        )


def test_eval_bundle_rejects_config_benchmark_profile_mismatch(tmp_path: Path) -> None:
    with pytest.raises(EvalBundleIntegrityError, match="config benchmark_profile mismatch"):
        write_eval_bundle(
            tmp_path,
            identity=make_identity(),
            config={"benchmark_profile": "different-profile"},
            metrics={},
            per_case=make_case_rows(),
            trajectories=[],
            metadata={"mode": "baseline"},
        )


def test_eval_bundle_rejects_config_treatments_mismatch(tmp_path: Path) -> None:
    with pytest.raises(EvalBundleIntegrityError, match="config treatments mismatch"):
        write_eval_bundle(
            tmp_path,
            identity=make_identity(),
            config={"treatments": {"pipeline": "agentic"}},
            metrics={},
            per_case=make_case_rows(),
            trajectories=[],
            metadata={"mode": "baseline"},
        )


def test_eval_bundle_rejects_metadata_mode_treatment_mismatch(tmp_path: Path) -> None:
    with pytest.raises(EvalBundleIntegrityError, match="metadata mode mismatch"):
        write_eval_bundle(
            tmp_path,
            identity=make_identity(),
            config={},
            metrics={},
            per_case=make_case_rows(),
            trajectories=[{"case_id": "case-01"}, {"case_id": "case-02"}],
            metadata={"mode": "agentic"},
        )


def test_eval_bundle_read_rejects_hashed_trajectory_case_set_mismatch(tmp_path: Path) -> None:
    bundle = write_eval_bundle(
        tmp_path,
        identity=make_agentic_identity(),
        config={},
        metrics={},
        per_case=make_case_rows(),
        trajectories=[{"case_id": "case-01"}, {"case_id": "case-02"}],
        metadata={"mode": "agentic"},
    )
    trajectory_rows = [{"case_id": "case-01"}, {"case_id": "case-03"}]
    (bundle.path / "trajectories.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in trajectory_rows),
        encoding="utf-8",
    )
    metadata_path = bundle.path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["artifact_hashes"]["trajectories_sha256"] = canonical_sha256(trajectory_rows)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(EvalBundleIntegrityError, match="trajectory case_ids_hash mismatch"):
        read_eval_bundle(bundle.path)


def test_eval_bundle_read_rejects_tampered_identity(tmp_path: Path) -> None:
    bundle = write_eval_bundle(
        tmp_path,
        identity=make_identity(),
        config={},
        metrics={},
        per_case=make_case_rows(),
        trajectories=[],
    )
    metadata_path = bundle.path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["run_identity"]["corpus_hash"] = "tampered"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(EvalBundleIntegrityError, match="run_id"):
        read_eval_bundle(bundle.path)


@pytest.mark.parametrize(
    ("mismatch", "expected_field"),
    [
        ("case_count", "case_count"),
        ("corpus_hash", "corpus_hash"),
        ("model", "model"),
    ],
)
def test_bundle_group_rejects_identity_mismatch_before_aggregation(
    tmp_path: Path,
    mismatch: str,
    expected_field: str,
) -> None:
    candidate_rows = make_case_rows()
    candidate_trajectories = [
        {"case_id": "case-01"},
        {"case_id": "case-02"},
    ]
    identity_overrides: dict[str, object] = {}
    if mismatch == "case_count":
        candidate_rows.append({"case_id": "case-03", "answer": "30%", "score": 1.0})
        candidate_trajectories.append({"case_id": "case-03"})
        identity_overrides.update(
            case_count=3,
            case_ids_hash=hash_case_ids(["case-01", "case-02", "case-03"]),
        )
    elif mismatch == "corpus_hash":
        identity_overrides["corpus_hash"] = "different-corpus"
    elif mismatch == "model":
        identity_overrides["model"] = "different-model"

    baseline = write_eval_bundle(
        tmp_path / "baseline",
        identity=make_identity(),
        config={},
        metrics={},
        per_case=make_case_rows(),
        trajectories=[],
    )
    agentic = write_eval_bundle(
        tmp_path / "agentic",
        identity=make_agentic_identity(**identity_overrides),
        config={},
        metrics={},
        per_case=candidate_rows,
        trajectories=candidate_trajectories,
    )

    with pytest.raises(IncompatibleRunIdentityError, match=expected_field):
        require_comparable_eval_bundles([baseline, agentic])


def test_bundle_group_allows_treatment_config_only_and_rejects_runtime_config_mismatch(tmp_path: Path) -> None:
    baseline = write_eval_bundle(
        tmp_path / "baseline",
        identity=make_identity(),
        config={"runtime": {"rerank_top_k": 5}, "treatments": {"pipeline": "baseline"}},
        metrics={},
        per_case=make_case_rows(),
        trajectories=[],
    )
    candidate = write_eval_bundle(
        tmp_path / "agentic",
        identity=make_identity(
            feature_flags={
                "runtime": {"strict_trace": True},
                "treatments": {"pipeline": "candidate"},
            },
        ),
        config={"runtime": {"rerank_top_k": 5}, "treatments": {"pipeline": "candidate"}},
        metrics={},
        per_case=make_case_rows(),
        trajectories=[],
    )
    require_comparable_eval_bundles([baseline, candidate])

    changed_runtime = write_eval_bundle(
        tmp_path / "changed-runtime",
        identity=make_identity(
            feature_flags={
                "runtime": {"strict_trace": True},
                "treatments": {"pipeline": "candidate"},
            },
        ),
        config={"runtime": {"rerank_top_k": 10}, "treatments": {"pipeline": "candidate"}},
        metrics={},
        per_case=make_case_rows(),
        trajectories=[],
    )
    with pytest.raises(EvalBundleIntegrityError, match="config identity mismatch"):
        require_comparable_eval_bundles([baseline, changed_runtime])


def test_bundle_comparison_rejects_agentic_bundle_without_ready_integrity_verdict(tmp_path: Path) -> None:
    baseline = write_eval_bundle(
        tmp_path / "baseline",
        identity=make_identity(),
        config={},
        metrics={},
        per_case=make_case_rows(),
        trajectories=[],
        metadata={"mode": "baseline"},
    )
    agentic = write_eval_bundle(
        tmp_path / "agentic",
        identity=make_agentic_identity(),
        config={},
        metrics={},
        per_case=make_case_rows(),
        trajectories=[{"case_id": "case-01"}, {"case_id": "case-02"}],
        metadata={"mode": "agentic"},
    )

    with pytest.raises(EvalBundleIntegrityError, match="READY integrity verdict"):
        require_comparable_eval_bundles([baseline, agentic])


def test_bundle_comparison_cannot_bypass_integrity_verdict_by_omitting_mode(tmp_path: Path) -> None:
    baseline = write_eval_bundle(
        tmp_path / "baseline",
        identity=make_identity(),
        config={},
        metrics={},
        per_case=make_case_rows(),
        trajectories=[],
    )
    agentic = write_eval_bundle(
        tmp_path / "agentic",
        identity=make_agentic_identity(),
        config={},
        metrics={},
        per_case=make_case_rows(),
        trajectories=[{"case_id": "case-01"}, {"case_id": "case-02"}],
    )

    with pytest.raises(EvalBundleIntegrityError, match="READY integrity verdict"):
        require_comparable_eval_bundles([baseline, agentic])
