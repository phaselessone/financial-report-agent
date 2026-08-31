"""Public command contract for the executable four-profile matrix."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

import run_profile_ablation as run_profile_ablation_module
from run_profile_ablation import (
    _build_profile_executor,
    _corpus_hash,
    _load_benchmark,
    _load_optional_semantic_scorer,
    _materialize_contract_oracle_chunks,
    _require_structured_context,
    _resolve_structured_context_paths,
    _validate_model_revision_for_benchmark,
    _validate_semantic_cli_request,
    build_base_identity,
    build_parser,
)
from src.agent.directional_nli import DirectionalNLIScorer, directional_nli_identity
from src.agent.semantic_policy import CalibratedSemanticScorer
from src.evaluation.eval_bundle import build_corpus_asset_manifest
from src.evaluation.hard_case_benchmark import load_cases
from src.evaluation.run_identity import file_sha256, hash_benchmark_rows, hash_case_ids
from tests.test_hard_case_benchmark import (
    _reviewed_asset_inputs,
    _reviewed_case,
    _reviewed_manifest,
)


def test_cli_help_exposes_reviewed_inputs_and_reproducibility_controls() -> None:
    completed = subprocess.run(
        [sys.executable, "run_profile_ablation.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 0, completed.stderr
    assert "--cases-path" in completed.stdout
    assert "--reviewed-manifest-path" in completed.stdout
    assert "--reviewed-source-root" in completed.stdout
    assert "--chunks-path" in completed.stdout
    assert "--contract-oracle" in completed.stdout
    assert "--offline-contract" in completed.stdout
    assert "--semantic-calibration-report" in completed.stdout
    assert "--semantic-scorer-config" in completed.stdout
    assert "--semantic-labels-path" in completed.stdout
    assert "--readiness-report" in completed.stdout
    assert "--embedding-model" in completed.stdout
    assert "--facts-path" in completed.stdout
    assert "--llm-model-revision" in completed.stdout
    assert "--overwrite" in completed.stdout


def _write_reviewed_cli_assets(tmp_path: Path) -> tuple[Path, Path, Path, dict]:
    cases = [_reviewed_case()]
    manifest, source_root, _ = _reviewed_asset_inputs(tmp_path, cases)
    cases_path = tmp_path / "reviewed-cases.jsonl"
    manifest_path = tmp_path / "reviewed-manifest.json"
    cases_path.write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return cases_path, manifest_path, source_root, manifest


def test_reviewed_cli_requires_source_root_and_release_attestation(tmp_path: Path) -> None:
    cases_path, manifest_path, source_root, manifest = _write_reviewed_cli_assets(tmp_path)
    args_without_root = build_parser().parse_args(
        [
            "--cases-path",
            str(cases_path),
            "--reviewed-manifest-path",
            str(manifest_path),
        ]
    )
    with pytest.raises(ValueError, match="require --reviewed-source-root"):
        _load_benchmark(args_without_root)

    manifest.pop("release_attestation")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args_without_attestation = build_parser().parse_args(
        [
            "--cases-path",
            str(cases_path),
            "--reviewed-manifest-path",
            str(manifest_path),
            "--reviewed-source-root",
            str(source_root),
        ]
    )
    with pytest.raises(ValueError, match="requires release_attestation"):
        _load_benchmark(args_without_attestation)


def test_reviewed_cli_returns_runtime_verified_attested_manifest(tmp_path: Path) -> None:
    cases_path, manifest_path, source_root, _ = _write_reviewed_cli_assets(tmp_path)
    args = build_parser().parse_args(
        [
            "--cases-path",
            str(cases_path),
            "--reviewed-manifest-path",
            str(manifest_path),
            "--reviewed-source-root",
            str(source_root),
        ]
    )

    cases, manifest = _load_benchmark(args)

    assert len(cases) == 1
    assert manifest is not None
    assert manifest.source_files_verified is True
    assert manifest.release_attestation_validated is True
    assert manifest["source_verification"]["status"] == "VERIFIED"


def test_base_identity_captures_one_context_and_leaves_treatments_empty() -> None:
    cases = load_cases()
    args = build_parser().parse_args(
        [
            "--allow-synthetic-contract",
            "--llm-provider",
            "deepseek",
            "--llm-model",
            "fixture-model",
            "--llm-model-revision",
            "fixture-revision",
            "--temperature",
            "0",
            "--prompt-version",
            "profile-ablation-test-v1",
            "--max-steps",
            "17",
        ]
    )

    identity = build_base_identity(
        cases=cases,
        args=args,
        provider="deepseek",
        model="fixture-model",
        benchmark_version="contract-v1",
        corpus_hash="corpus-sha",
        git_commit="git-sha",
        source_manifest_hash="manifest-sha",
    )

    assert identity.benchmark_hash == hash_benchmark_rows(cases)
    assert identity.case_ids_hash == hash_case_ids(case["case_id"] for case in cases)
    assert identity.case_count == len(cases)
    assert identity.budgets["max_steps"] == 17
    assert identity.provider == "deepseek"
    assert identity.model == "fixture-model"
    assert identity.model_revision == "fixture-revision"
    assert dict(identity.feature_flags["treatments"]) == {}
    assert identity.feature_flags["runtime"]["profile_execution_seam"] == "profile-spec-v1"
    assert identity.feature_flags["runtime"]["reviewed_manifest_hash"] is None
    assert identity.feature_flags["runtime"]["reviewed_target_count"] is None
    assert identity.feature_flags["runtime"]["retrieval_source"] == "shared-corpus"
    assert identity.feature_flags["runtime"]["semantic_activation"] == {"enabled": False}


def test_corpus_hash_is_bound_to_shared_chunks_not_benchmark_gold(tmp_path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    facts_path = tmp_path / "facts.jsonl"
    aliases_path = tmp_path / "aliases.json"
    chunks_path.write_text(
        '{"chunk_id":"chunk-1","doc_id":"report","page_start":1,"text":"corpus"}\n',
        encoding="utf-8",
    )
    facts_path.write_text('{"fact_id":"fact-1"}\n', encoding="utf-8")
    aliases_path.write_text('{"Company":["Company"]}', encoding="utf-8")

    first = _corpus_hash(
        chunks_path=chunks_path,
        facts_path=facts_path,
        aliases_path=aliases_path,
    )
    chunks_path.write_text(
        '{"chunk_id":"chunk-2","doc_id":"report","page_start":2,"text":"changed"}\n',
        encoding="utf-8",
    )
    second = _corpus_hash(
        chunks_path=chunks_path,
        facts_path=facts_path,
        aliases_path=aliases_path,
    )

    assert first != second


def test_offline_contract_clean_checkout_materializes_manifest_bound_structured_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clean_checkout_root = tmp_path / "clean-checkout"
    clean_checkout_root.mkdir()
    monkeypatch.chdir(clean_checkout_root)
    assert not Path("data").exists()

    output_root = clean_checkout_root / "outputs" / "profile-contract"
    output_root.mkdir(parents=True)
    args = build_parser().parse_args(
        [
            "--allow-synthetic-contract",
            "--contract-oracle",
            "--offline-contract",
            "--output-root",
            str(output_root),
        ]
    )
    cases = load_cases(Path(__file__).resolve().parents[1] / "benchmarks/hard_cases/cases.jsonl")

    facts_path, aliases_path = _resolve_structured_context_paths(args, output_root)
    fact_store, aliases = _require_structured_context(facts_path, aliases_path)
    try:
        assert fact_store.count() == 1
        assert aliases == {"合约示例制造": ["合约示例制造"]}
    finally:
        fact_store.close()

    chunks_path = _materialize_contract_oracle_chunks(
        cases,
        output_root / "contract_oracle_chunks.jsonl",
        include_offline_structured_fixture=True,
    )
    manifest = build_corpus_asset_manifest(
        chunks_path=chunks_path,
        facts_path=facts_path,
        aliases_path=aliases_path,
    )
    assert Path(manifest["paths"]["structured_facts"]) == facts_path.resolve()
    assert Path(manifest["paths"]["company_aliases"]) == aliases_path.resolve()
    assert facts_path.parent == aliases_path.parent == output_root
    assert not any("data" in path.relative_to(clean_checkout_root).parts for path in (facts_path, aliases_path))

    identity = build_base_identity(
        cases=cases,
        args=args,
        provider="offline-deterministic",
        model="synthetic-contract-evidence-echo",
        benchmark_version="contract-v1",
        corpus_hash=manifest["corpus_hash"],
        git_commit="unavailable",
        source_manifest_hash="source-manifest-test",
        corpus_asset_manifest=manifest,
    )
    assert identity.corpus_hash == manifest["corpus_hash"]
    assert (
        identity.feature_flags["runtime"]["corpus_asset_manifest_hash"]
        == manifest["manifest_hash"]
    )
    assert (
        identity.feature_flags["runtime"]["structured_context_source"]
        == "materialized-offline-contract-fixture"
    )


def test_reviewed_base_identity_requires_explicit_corpus_asset_manifest() -> None:
    cases = [_reviewed_case(case_id="REV-IDENTITY-ASSET-01", category="simple_factual")]
    args = build_parser().parse_args(
        [
            "--llm-model",
            "fixture-model",
            "--llm-model-revision",
            "fixture-revision",
            "--benchmark-profile",
            "reviewed-profile",
        ]
    )

    with pytest.raises(ValueError, match="corpus asset manifest"):
        build_base_identity(
            cases=cases,
            args=args,
            provider="deepseek",
            model="fixture-model",
            benchmark_version="reviewed-v1",
            corpus_hash="caller-supplied-hash",
            git_commit="git-sha",
            source_manifest_hash="manifest-sha",
            reviewed_manifest=_reviewed_manifest(cases, category_quotas={"simple_factual": 1}),
        )


@pytest.mark.parametrize("revision", ["", "unversioned", "default", "latest", "unknown"])
def test_reviewed_profile_rejects_unversioned_model_revision(revision: str) -> None:
    with pytest.raises(ValueError, match="versioned --llm-model-revision"):
        _validate_model_revision_for_benchmark(revision, benchmark_kind="reviewed")


def test_synthetic_contract_may_use_non_publishable_unversioned_revision() -> None:
    assert (
        _validate_model_revision_for_benchmark("unversioned", benchmark_kind="synthetic")
        == "unversioned"
    )


def test_profile_executor_wires_claim_judge_from_the_identity_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = build_parser().parse_args(["--claim-llm-budget", "3", "--claim-max-tokens", "77"])
    judge = object()
    observed: dict[str, object] = {}

    def fake_build_judge(llm, *, budget, max_tokens):
        observed["judge_args"] = (llm, budget, max_tokens)
        return judge

    def fake_build_executor(**kwargs):
        observed["executor_args"] = kwargs
        return "executor"

    monkeypatch.setattr(run_profile_ablation_module, "build_claim_llm_judge", fake_build_judge)
    monkeypatch.setattr(run_profile_ablation_module, "build_profile_executor", fake_build_executor)
    llm = object()

    result = _build_profile_executor(
        args=args,
        answerer=object(),
        llm=llm,
        fact_store=object(),
        aliases={"Company": ["Company"]},
        retrieval_runtime=object(),
    )

    assert result == "executor"
    assert observed["judge_args"] == (llm, 3, 77)
    executor_args = observed["executor_args"]
    assert executor_args["llm_judge"] is judge
    assert executor_args["base_config"].claim_llm_budget == 3
    assert executor_args["base_config"].claim_max_tokens == 77


def test_semantic_cli_request_requires_a_complete_reviewed_pair() -> None:
    args = build_parser().parse_args(["--semantic-calibration-report", "calibration.json"])
    with pytest.raises(ValueError, match="requires --semantic-calibration-report"):
        _validate_semantic_cli_request(args, benchmark_kind="reviewed")

    args = build_parser().parse_args(
        [
            "--semantic-calibration-report",
            "calibration.json",
            "--semantic-scorer-config",
            "scorer.json",
            "--semantic-labels-path",
            "labels.jsonl",
            "--readiness-report",
            "readiness.json",
        ]
    )
    with pytest.raises(ValueError, match="restricted to reviewed"):
        _validate_semantic_cli_request(args, benchmark_kind="synthetic")


def test_semantic_cli_activates_exact_local_scorer_and_binds_hashes(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer_config = {
        "kind": "directional_nli",
        "model": "fixture-nli",
        "revision": "fixture-v1",
        "entailment_label_id": 2,
        "max_length": 256,
        "device": "cpu",
        "local_files_only": True,
        "trust_remote_code": False,
    }
    identity = directional_nli_identity(scorer_config)
    report = {
        "calibrated": True,
        "precision_constraint_satisfied": True,
        "positive_support_constraint_satisfied": True,
        "coverage_constraint_satisfied": True,
        "threshold": 0.9,
        "metrics": {"precision": 0.98, "tp": 49, "fp": 1},
        "calibration_config": {
            "min_precision": 0.95,
            "min_predicted_positives": 5,
            "min_true_positives": 5,
        },
        "label_contract": {
            "review_status": "reviewed",
            "content_hash_binding": "sha256",
        },
        "labels_sha256": "a" * 64,
        "scorer_kind": "directional_nli",
        "scorer_identity": identity,
        "calibrated_statuses": ["ENTAILED"],
        "contradiction_state_change_allowed": False,
    }
    report_path = tmp_path / "calibration.json"
    config_path = tmp_path / "scorer.json"
    config_path.write_text(json.dumps(scorer_config), encoding="utf-8")
    labels_path = tmp_path / "labels.jsonl"
    label = {
        "sample_id": "sample-1",
        "claim_id": "claim-1",
        "evidence_id": "evidence-1",
        "claim_sha256": hashlib.sha256(b"reviewed claim 1").hexdigest(),
        "evidence_sha256": hashlib.sha256(b"reviewed evidence 1").hexdigest(),
        "score": 0.97,
        "label": "ENTAILED",
        "source_ref": "review-vault://sample-1",
        "review_status": "reviewed",
        "reviewer_id": "reviewer-1",
        "reviewed_at": "2026-08-24T00:00:00Z",
        "scorer_kind": identity["kind"],
        "scorer_model": identity["model"],
        "scorer_revision": identity["revision"],
        "scorer_config_sha256": identity["config_sha256"],
    }
    labels_path.write_text(json.dumps(label) + "\n", encoding="utf-8")
    labels_sha256 = file_sha256(labels_path)
    report["labels_sha256"] = labels_sha256
    report_path.write_text(json.dumps(report), encoding="utf-8")
    readiness_path = tmp_path / "readiness.json"
    readiness_path.write_text(
        json.dumps(
            {
                "status": "READY",
                "readiness_status": "READY",
                "blocked": False,
                "assets": {
                    "semantic_labels": {
                        "status": "READY",
                        "path": str(labels_path.resolve()),
                        "sha256": labels_sha256,
                    }
                },
                "semantic_calibration": report,
            }
        ),
        encoding="utf-8",
    )
    raw = DirectionalNLIScorer(scorer_config, lambda _premise, _hypothesis: 0.97)
    import src.agent.directional_nli as directional_nli_module

    monkeypatch.setattr(
        directional_nli_module,
        "build_local_directional_nli_scorer",
        lambda _config: raw,
    )
    args = build_parser().parse_args(
        [
            "--semantic-calibration-report",
            str(report_path),
            "--semantic-scorer-config",
            str(config_path),
            "--semantic-labels-path",
            str(labels_path),
            "--readiness-report",
            str(readiness_path),
        ]
    )

    scorer, metadata = _load_optional_semantic_scorer(args)

    assert isinstance(scorer, CalibratedSemanticScorer)
    assert metadata["enabled"] is True
    assert metadata["profiles"] == ["agentic-rag", "full-agent"]
    assert metadata["scorer_identity"] == identity
    assert metadata["labels_sha256"] == labels_sha256
    assert metadata["semantic_labels_path"] == str(labels_path.resolve())
    assert metadata["readiness_report_sha256"] == file_sha256(readiness_path)
    assert metadata["local_files_only"] is True
    assert metadata["trust_remote_code"] is False


def test_semantic_cli_rejects_blocked_readiness_before_activation(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer_config = {
        "kind": "directional_nli",
        "model": "fixture-nli",
        "revision": "fixture-v1",
        "entailment_label_id": 2,
        "local_files_only": True,
        "trust_remote_code": False,
    }
    identity = directional_nli_identity(scorer_config)
    config_path = tmp_path / "scorer.json"
    labels_path = tmp_path / "labels.jsonl"
    report_path = tmp_path / "calibration.json"
    readiness_path = tmp_path / "readiness.json"
    config_path.write_text(json.dumps(scorer_config), encoding="utf-8")
    labels_path.write_text(
        json.dumps(
            {
                "sample_id": "sample-1",
                "claim_id": "claim-1",
                "evidence_id": "evidence-1",
                "claim_sha256": hashlib.sha256(b"reviewed claim 1").hexdigest(),
                "evidence_sha256": hashlib.sha256(b"reviewed evidence 1").hexdigest(),
                "score": 0.97,
                "label": "ENTAILED",
                "source_ref": "review-vault://sample-1",
                "review_status": "reviewed",
                "reviewer_id": "reviewer-1",
                "reviewed_at": "2026-08-24T00:00:00Z",
                "scorer_kind": identity["kind"],
                "scorer_model": identity["model"],
                "scorer_revision": identity["revision"],
                "scorer_config_sha256": identity["config_sha256"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    labels_sha256 = file_sha256(labels_path)
    report = {
        "calibrated": True,
        "precision_constraint_satisfied": True,
        "positive_support_constraint_satisfied": True,
        "coverage_constraint_satisfied": True,
        "threshold": 0.9,
        "metrics": {"precision": 0.98, "tp": 49, "fp": 1},
        "calibration_config": {
            "min_precision": 0.95,
            "min_predicted_positives": 5,
            "min_true_positives": 5,
        },
        "label_contract": {
            "review_status": "reviewed",
            "content_hash_binding": "sha256",
        },
        "labels_sha256": labels_sha256,
        "scorer_kind": "directional_nli",
        "scorer_identity": identity,
        "calibrated_statuses": ["ENTAILED"],
        "contradiction_state_change_allowed": False,
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    readiness_path.write_text(
        json.dumps(
            {
                "status": "BLOCKED",
                "readiness_status": "BLOCKED",
                "blocked": True,
            }
        ),
        encoding="utf-8",
    )
    raw = DirectionalNLIScorer(scorer_config, lambda _premise, _hypothesis: 0.97)
    import src.agent.directional_nli as directional_nli_module

    monkeypatch.setattr(
        directional_nli_module,
        "build_local_directional_nli_scorer",
        lambda _config: raw,
    )
    args = build_parser().parse_args(
        [
            "--semantic-calibration-report",
            str(report_path),
            "--semantic-scorer-config",
            str(config_path),
            "--semantic-labels-path",
            str(labels_path),
            "--readiness-report",
            str(readiness_path),
        ]
    )

    with pytest.raises(ValueError, match="top-level READY"):
        _load_optional_semantic_scorer(args)


def test_semantic_cli_rejects_calibration_hash_not_matching_actual_labels(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer_config = {
        "kind": "directional_nli",
        "model": "fixture-nli",
        "revision": "fixture-v1",
        "entailment_label_id": 2,
        "local_files_only": True,
        "trust_remote_code": False,
    }
    identity = directional_nli_identity(scorer_config)
    config_path = tmp_path / "scorer.json"
    labels_path = tmp_path / "labels.jsonl"
    report_path = tmp_path / "calibration.json"
    readiness_path = tmp_path / "readiness.json"
    config_path.write_text(json.dumps(scorer_config), encoding="utf-8")
    labels_path.write_text(
        json.dumps(
            {
                "sample_id": "sample-1",
                "claim_id": "claim-1",
                "evidence_id": "evidence-1",
                "claim_sha256": hashlib.sha256(b"reviewed claim 1").hexdigest(),
                "evidence_sha256": hashlib.sha256(b"reviewed evidence 1").hexdigest(),
                "score": 0.97,
                "label": "ENTAILED",
                "source_ref": "review-vault://sample-1",
                "review_status": "reviewed",
                "reviewer_id": "reviewer-1",
                "reviewed_at": "2026-08-24T00:00:00Z",
                "scorer_kind": identity["kind"],
                "scorer_model": identity["model"],
                "scorer_revision": identity["revision"],
                "scorer_config_sha256": identity["config_sha256"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    labels_sha256 = file_sha256(labels_path)
    report = {
        "calibrated": True,
        "precision_constraint_satisfied": True,
        "positive_support_constraint_satisfied": True,
        "coverage_constraint_satisfied": True,
        "threshold": 0.9,
        "metrics": {"precision": 0.98, "tp": 49, "fp": 1},
        "calibration_config": {
            "min_precision": 0.95,
            "min_predicted_positives": 5,
            "min_true_positives": 5,
        },
        "label_contract": {
            "review_status": "reviewed",
            "content_hash_binding": "sha256",
        },
        "labels_sha256": "0" * 64,
        "scorer_kind": "directional_nli",
        "scorer_identity": identity,
        "calibrated_statuses": ["ENTAILED"],
        "contradiction_state_change_allowed": False,
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    readiness_path.write_text(
        json.dumps(
            {
                "status": "READY",
                "readiness_status": "READY",
                "blocked": False,
                "assets": {
                    "semantic_labels": {
                        "status": "READY",
                        "path": str(labels_path.resolve()),
                        "sha256": labels_sha256,
                    }
                },
                "semantic_calibration": {**report, "labels_sha256": labels_sha256},
            }
        ),
        encoding="utf-8",
    )
    raw = DirectionalNLIScorer(scorer_config, lambda _premise, _hypothesis: 0.97)
    import src.agent.directional_nli as directional_nli_module

    monkeypatch.setattr(
        directional_nli_module,
        "build_local_directional_nli_scorer",
        lambda _config: raw,
    )
    args = build_parser().parse_args(
        [
            "--semantic-calibration-report",
            str(report_path),
            "--semantic-scorer-config",
            str(config_path),
            "--semantic-labels-path",
            str(labels_path),
            "--readiness-report",
            str(readiness_path),
        ]
    )

    with pytest.raises(ValueError, match="calibration report SHA-256"):
        _load_optional_semantic_scorer(args)
