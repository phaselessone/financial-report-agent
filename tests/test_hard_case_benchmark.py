from __future__ import annotations

import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from src.evaluation.hard_case_benchmark import (
    CATEGORIES,
    DEFAULT_REVIEWED_CASES_PATH,
    DEFAULT_REVIEWED_MANIFEST_PATH,
    PROFILE_TREATMENTS,
    PROFILES,
    REVIEWED_MANIFEST_SCHEMA_VERSION,
    SYNTHETIC_CONTRACT_VERSION,
    ValidatedReviewedManifest,
    canonical_reviewed_case_ids_hash,
    canonical_reviewed_cases_hash,
    evaluate_case,
    evaluate_predictions,
    load_cases,
    load_reviewed_cases,
    normalize_profile,
    validate_case,
    validate_cases,
    validate_reviewed_manifest,
    write_run_bundle,
)
from src.evaluation.profile_runtime import PROFILE_IDS, PROFILE_SPECS
from src.evaluation.run_identity import hash_benchmark_rows, hash_case_ids


def _reviewed_case(**overrides):
    row = {
        "case_id": "REV-001",
        "category": "derived_calculation",
        "question": "2025 年营收比 2024 年增加多少？",
        "evidence": [
            {
                "evidence_id": "e-2025",
                "text": "2025 年营收为 100 亿元。",
                "source": "annual-report-2025",
                "source_id": "annual-report-set",
            },
            {
                "evidence_id": "e-2024",
                "text": "2024 年营收为 80 亿元。",
                "source": "annual-report-2024",
                "source_id": "annual-report-set",
            },
        ],
        "gold_answer": "增加 20 亿元。",
        "gold_claims": [
            {
                "claim_id": "g-1",
                "text": "营收增加 20 亿元",
                "status": "ENTAILED",
                "evidence_ids": ["e-2025", "e-2024"],
            }
        ],
        "gold_calculations": [
            {
                "calculation_id": "gold-k1",
                "status": "SUCCESS",
                "operation": "subtract",
                "formula": "current - prior",
                "inputs": [
                    {"name": "current", "value": "100", "unit": "亿元", "evidence_ids": ["e-2025"]},
                    {"name": "prior", "value": "80", "unit": "亿元", "evidence_ids": ["e-2024"]},
                ],
                "evidence_ids": ["e-2025", "e-2024"],
                "result": {"value": "20", "unit": "亿元"},
            }
        ],
        "required_tools": ["report_search", "calculator"],
        "forbidden_tools": ["web_search"],
        "must_abstain": False,
        "partial_answer_gold": {"allowed": True, "required_claim_ids": ["g-1"]},
        "synthetic": False,
        "review_status": "reviewed",
        "reviewer": "reviewer@example.com",
        "review_time": "2026-08-23T10:00:00+08:00",
        "source_provenance": [
            {
                "source_id": "annual-report-set",
                "uri": "review://annual-reports/2024-2025",
                "sha256": "b" * 64,
            }
        ],
        "benchmark_version": "reviewed-v1",
        "benchmark_hash": "a" * 64,
    }
    row.update(overrides)
    return row


def _reviewed_manifest(cases, **overrides):
    manifest = {
        "schema_version": REVIEWED_MANIFEST_SCHEMA_VERSION,
        "benchmark_version": "reviewed-v1",
        "review_status": "reviewed",
        "case_count": len(cases),
        "cases_hash": canonical_reviewed_cases_hash(cases),
        "case_ids_hash": canonical_reviewed_case_ids_hash(cases),
        "source_provenance": [
            {
                "source_id": "annual-report-set",
                "uri": "review://annual-reports/2024-2025",
                "sha256": "b" * 64,
            }
        ],
        "category_quotas": {"derived_calculation": len(cases)},
        "source_quotas": {"annual-report-set": len(cases)},
    }
    manifest.update(overrides)
    return manifest


def _complete_reviewed_cases():
    return [
        _reviewed_case(case_id=f"REV-{index:03d}", category=category)
        for index, category in enumerate(CATEGORIES, start=1)
    ]


def _complete_reviewed_manifest(cases, **overrides):
    manifest = _reviewed_manifest(
        cases,
        category_quotas={category: 1 for category in CATEGORIES},
        source_quotas={"annual-report-set": len(cases)},
    )
    manifest.update(overrides)
    return manifest


def _release_attestation() -> dict[str, str]:
    return {
        "status": "reviewed",
        "reviewer_id": "reviewer@example.com",
        "review_batch": "reviewed-hard-cases-v1",
        "reviewed_at": "2026-08-23T10:00:00+08:00",
    }


def _reviewed_asset_inputs(
    tmp_path: Path,
    cases: list[dict],
) -> tuple[dict, Path, Path]:
    source_root = tmp_path / "reviewed-sources"
    source_root.mkdir(parents=True)
    source_path = source_root / "annual-report-set.txt"
    source_path.write_text("reviewed annual report fixture\n", encoding="utf-8")
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    case_provenance = {
        "source_id": "annual-report-set",
        "uri": "review://annual-reports/2024-2025",
        "sha256": source_sha256,
    }
    for case in cases:
        case["source_provenance"] = [dict(case_provenance)]
    manifest = _complete_reviewed_manifest(
        cases,
        source_provenance=[
            {
                **case_provenance,
                "path": source_path.relative_to(source_root).as_posix(),
            }
        ],
        release_attestation=_release_attestation(),
    )
    return manifest, source_root, source_path


def _verified_reviewed_manifest(
    tmp_path: Path,
    cases: list[dict],
) -> tuple[ValidatedReviewedManifest, Path, Path]:
    manifest, source_root, source_path = _reviewed_asset_inputs(tmp_path, cases)
    validated = validate_reviewed_manifest(
        manifest,
        cases,
        source_root=source_root,
        verify_source_files=True,
        require_release_attestation=True,
    )
    return validated, source_root, source_path


def _all_metric_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _all_metric_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_metric_keys(item)


def test_exact_counts_and_labels() -> None:
    cases = load_cases()
    assert len(cases) == 100
    assert {category: sum(row["category"] == category for row in cases) for category in CATEGORIES} == {category: 10 for category in CATEGORIES}
    assert all(row["synthetic"] is True and row["review_status"] == "synthetic_not_human_reviewed" for row in cases)
    assert all(row["benchmark_version"] == SYNTHETIC_CONTRACT_VERSION for row in cases)


def test_synthetic_contract_manifest_pins_bytes_identity_and_balanced_quotas() -> None:
    hard_case_dir = Path(__file__).resolve().parents[1] / "benchmarks" / "hard_cases"
    manifest = json.loads(
        (hard_case_dir / "contract-v1.manifest.json").read_text(encoding="utf-8")
    )
    cases_path = hard_case_dir / manifest["cases_file"]
    cases = load_cases(cases_path)

    assert manifest["schema_version"] == "synthetic-hard-cases-manifest-v1"
    assert manifest["benchmark_kind"] == "synthetic_contract"
    assert manifest["benchmark_version"] == SYNTHETIC_CONTRACT_VERSION
    assert manifest["review_status"] == "synthetic_not_human_reviewed"
    assert hashlib.sha256(cases_path.read_bytes()).hexdigest() == manifest["cases_file_sha256"]
    assert hash_benchmark_rows(cases) == manifest["canonical_benchmark_hash"]
    assert hash_case_ids(case["case_id"] for case in cases) == manifest["case_ids_hash"]
    assert len(cases) == manifest["case_count"] == 100
    assert dict(Counter(case["category"] for case in cases)) == manifest["category_counts"]


def test_reviewed_benchmark_has_a_separate_validated_contract() -> None:
    rows = validate_cases([_reviewed_case()])

    assert rows[0]["benchmark_kind"] == "reviewed"
    assert rows[0]["benchmark_version"] == "reviewed-v1"


def test_reviewed_rows_do_not_need_a_self_reported_benchmark_hash() -> None:
    case = _reviewed_case()
    case.pop("benchmark_hash")

    rows = validate_cases([case])
    validated = validate_reviewed_manifest(_reviewed_manifest(rows), rows)

    assert "benchmark_hash" not in rows[0]
    assert validated["benchmark_hash"] == canonical_reviewed_cases_hash(rows)


def test_reviewed_manifest_recomputes_canonical_hashes_and_coverage() -> None:
    assert canonical_reviewed_cases_hash(
        [
            {"case_id": "B", "question": "two", "benchmark_hash": "f" * 64},
            {"case_id": "A", "question": "one", "benchmark_hash": "0" * 64},
        ]
    ) == "6c5c336f557d64c3203c5d32d2c96c0b4af64ca0d3f8da4d4a84befca0e35d62"
    assert canonical_reviewed_case_ids_hash(
        [{"case_id": "B"}, {"case_id": "A"}]
    ) == "b64e3448a83a5b86466465080361c1a7e1157a27ddccd4b68069cb18caffb74a"

    cases = [_reviewed_case()]
    validated = validate_reviewed_manifest(_reviewed_manifest(cases), cases)

    assert validated["cases_hash"] == canonical_reviewed_cases_hash(cases)
    assert validated["case_ids_hash"] == canonical_reviewed_case_ids_hash(cases)
    assert validated["coverage"]["category_counts"]["derived_calculation"] == 1
    assert validated["coverage"]["source_case_counts"] == {"annual-report-set": 1}
    assert validated["coverage"]["category_quotas_met"] is True
    assert validated["coverage"]["source_quotas_met"] is True


def test_reviewed_manifest_rejects_self_reported_hash_or_unprovenanced_source() -> None:
    case = _reviewed_case()
    with pytest.raises(ValueError, match="cases_hash"):
        validate_reviewed_manifest(
            _reviewed_manifest([case], cases_hash="a" * 64),
            [case],
        )

    bad_source = _reviewed_case(
        evidence=[
            {
                "evidence_id": "e-2025",
                "text": "2025 年营收为 100 亿元。",
                "source": "annual-report-2025",
                "source_id": "missing-source",
            }
        ],
        gold_claims=[
            {
                "claim_id": "g-1",
                "text": "营收增加 20 亿元",
                "status": "INSUFFICIENT",
                "evidence_ids": [],
            }
        ],
        gold_calculations=[],
    )
    with pytest.raises(ValueError, match="source_id"):
        validate_reviewed_manifest(_reviewed_manifest([bad_source]), [bad_source])

    mismatched_provenance = _reviewed_case(
        source_provenance=[
            {
                "source_id": "annual-report-set",
                "uri": "review://annual-reports/2024-2025",
                "sha256": "c" * 64,
            }
        ]
    )
    with pytest.raises(ValueError, match="provenance"):
        validate_reviewed_manifest(
            _reviewed_manifest([mismatched_provenance]),
            [mismatched_provenance],
        )


def test_load_reviewed_cases_uses_explicit_manifest_and_has_stable_defaults(tmp_path: Path) -> None:
    case = _reviewed_case()
    cases_path = tmp_path / "reviewed.jsonl"
    manifest_path = tmp_path / "manifest.json"
    cases_path.write_text(json.dumps(case, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(_reviewed_manifest([case]), ensure_ascii=False),
        encoding="utf-8",
    )

    rows, manifest = load_reviewed_cases(cases_path, manifest_path)

    assert DEFAULT_REVIEWED_CASES_PATH.name == "reviewed_cases.jsonl"
    assert DEFAULT_REVIEWED_MANIFEST_PATH.name == "reviewed_manifest.json"
    assert rows == validate_cases([case])
    assert manifest["cases_hash"] == canonical_reviewed_cases_hash(rows)


def test_reviewed_source_verification_rejects_path_escape(tmp_path: Path) -> None:
    cases = [_reviewed_case()]
    manifest, source_root, source_path = _reviewed_asset_inputs(tmp_path, cases)
    outside_path = tmp_path / "outside-source.txt"
    source_path.replace(outside_path)
    outside_sha256 = hashlib.sha256(outside_path.read_bytes()).hexdigest()
    cases[0]["source_provenance"][0]["sha256"] = outside_sha256
    manifest["source_provenance"][0].update(
        {"path": "../outside-source.txt", "sha256": outside_sha256}
    )
    manifest["cases_hash"] = canonical_reviewed_cases_hash(cases)

    with pytest.raises(ValueError, match="escapes source_root"):
        validate_reviewed_manifest(
            manifest,
            cases,
            source_root=source_root,
            verify_source_files=True,
            require_release_attestation=True,
        )


def test_reviewed_source_verification_rejects_missing_file(tmp_path: Path) -> None:
    cases = [_reviewed_case()]
    manifest, source_root, source_path = _reviewed_asset_inputs(tmp_path, cases)
    source_path.unlink()

    with pytest.raises(ValueError, match="file does not exist"):
        validate_reviewed_manifest(
            manifest,
            cases,
            source_root=source_root,
            verify_source_files=True,
            require_release_attestation=True,
        )


def test_reviewed_source_verification_rejects_sha256_drift(tmp_path: Path) -> None:
    cases = [_reviewed_case()]
    manifest, source_root, source_path = _reviewed_asset_inputs(tmp_path, cases)
    source_path.write_text("tampered after review\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_reviewed_manifest(
            manifest,
            cases,
            source_root=source_root,
            verify_source_files=True,
            require_release_attestation=True,
        )


def test_reviewed_manifest_rejects_self_reported_source_verification(tmp_path: Path) -> None:
    cases = [_reviewed_case()]
    manifest, source_root, source_path = _reviewed_asset_inputs(tmp_path, cases)
    manifest["source_verification"] = {
        "status": "VERIFIED",
        "mode": "recomputed-local-file-sha256",
        "sources": [
            {
                "source_id": "annual-report-set",
                "path": source_path.relative_to(source_root).as_posix(),
                "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            }
        ],
    }

    with pytest.raises(ValueError, match="must not be self-reported"):
        validate_reviewed_manifest(
            manifest,
            cases,
            source_root=source_root,
            verify_source_files=True,
            require_release_attestation=True,
        )


def test_reviewed_release_attestation_is_required_for_formal_validation(
    tmp_path: Path,
) -> None:
    cases = [_reviewed_case()]
    manifest, source_root, _ = _reviewed_asset_inputs(tmp_path, cases)
    manifest.pop("release_attestation")

    with pytest.raises(ValueError, match="requires release_attestation"):
        validate_reviewed_manifest(
            manifest,
            cases,
            source_root=source_root,
            verify_source_files=True,
            require_release_attestation=True,
        )


@pytest.mark.parametrize(
    "reviewed_at",
    ["2026-08-23", "2026-08-23T10:00:00", "2026-08-23 10:00:00+08:00"],
)
def test_reviewed_release_attestation_requires_rfc3339_timezone(
    tmp_path: Path,
    reviewed_at: str,
) -> None:
    cases = [_reviewed_case()]
    manifest, source_root, _ = _reviewed_asset_inputs(tmp_path, cases)
    manifest["release_attestation"]["reviewed_at"] = reviewed_at

    with pytest.raises(ValueError, match="RFC-3339 timestamp with timezone"):
        validate_reviewed_manifest(
            manifest,
            cases,
            source_root=source_root,
            verify_source_files=True,
            require_release_attestation=True,
        )


def test_validated_reviewed_manifest_rejects_post_validation_mutation(
    tmp_path: Path,
) -> None:
    cases = _complete_reviewed_cases()
    validated, _, _ = _verified_reviewed_manifest(tmp_path, cases)
    validated.pop("release_attestation")

    with pytest.raises(ValueError, match="mutated after validation"):
        evaluate_predictions(
            cases,
            {case["case_id"]: {"answer": case["gold_answer"]} for case in cases},
            reviewed_target_count=len(cases),
            reviewed_manifest=validated,
        )


def test_reviewed_schemas_pin_manifest_hashes_and_status_aware_calculations() -> None:
    root = Path(__file__).resolve().parents[1] / "benchmarks" / "hard_cases"
    case_schema = json.loads((root / "reviewed-v1.schema.json").read_text(encoding="utf-8"))
    manifest_schema = json.loads(
        (root / "reviewed-manifest.schema.json").read_text(encoding="utf-8")
    )

    assert "benchmark_hash" not in case_schema["required"]
    calculation_schema = case_schema["properties"]["gold_calculations"]["items"]
    assert calculation_schema["required"] == ["calculation_id", "status", "operation"]
    assert calculation_schema["allOf"][0]["then"]["required"] == ["inputs", "result"]
    assert {"cases_hash", "case_ids_hash", "category_quotas", "source_quotas"}.issubset(
        manifest_schema["required"]
    )


def test_reviewed_gold_calculation_requirements_depend_on_status() -> None:
    validate_case(
        _reviewed_case(
            gold_calculations=[
                {
                    "calculation_id": "gold-k1",
                    "status": "INSUFFICIENT",
                    "operation": "subtract",
                    "reason": "prior period value is not reviewed",
                }
            ]
        )
    )

    success_without_result = _reviewed_case(
        gold_calculations=[
            {
                "calculation_id": "gold-k1",
                "status": "SUCCESS",
                "operation": "subtract",
                "inputs": [],
            }
        ]
    )
    with pytest.raises(ValueError, match="requires a result"):
        validate_case(success_without_result)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"reviewer": ""}, "reviewer"),
        ({"review_time": "2026-08-23"}, "review_time"),
        ({"source_provenance": []}, "source_provenance"),
        ({"benchmark_hash": "not-a-sha256"}, "benchmark_hash"),
        ({"gold_claims": []}, "gold_claims"),
        ({"forbidden_tools": ["calculator"]}, "required_tools and forbidden_tools"),
        ({"partial_answer_gold": {}}, "partial_answer_gold"),
    ],
)
def test_reviewed_contract_fails_closed_when_review_evidence_is_incomplete(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        validate_case(_reviewed_case(**overrides))


def test_validation_rejects_duplicate_or_bad_category() -> None:
    cases = load_cases()
    with pytest.raises(ValueError):
        validate_cases(cases + [cases[0]], require_exact_100=False)
    bad = dict(cases[0], category="unknown")
    with pytest.raises(ValueError):
        validate_cases([bad], require_exact_100=False)


def test_profiles_are_canonical_ids_with_executable_feature_treatments() -> None:
    assert PROFILES == (
        "baseline-rag",
        "agentic-rag",
        "structured-agent",
        "full-agent",
    )
    expected_features = {
        "retrieval",
        "structured_facts",
        "controlled_tools",
        "deterministic_calculation",
        "multi_hop_reasoning",
        "claim_verification",
    }
    assert set(PROFILE_TREATMENTS) == set(PROFILES)
    assert all(set(spec) == expected_features for spec in PROFILE_TREATMENTS.values())
    assert PROFILE_TREATMENTS["baseline-rag"] == {
        "retrieval": True,
        "structured_facts": False,
        "controlled_tools": False,
        "deterministic_calculation": False,
        "multi_hop_reasoning": False,
        "claim_verification": False,
    }
    assert PROFILE_TREATMENTS["agentic-rag"]["controlled_tools"] is True
    assert PROFILE_TREATMENTS["agentic-rag"]["structured_facts"] is False
    assert PROFILE_TREATMENTS["structured-agent"]["structured_facts"] is True
    assert PROFILE_TREATMENTS["structured-agent"]["controlled_tools"] is False
    assert PROFILE_TREATMENTS["structured-agent"]["claim_verification"] is False
    assert all(PROFILE_TREATMENTS["full-agent"].values())
    assert normalize_profile("Structured+Tool+Claim Verification") == "full-agent"
    assert PROFILES == PROFILE_IDS
    assert PROFILE_TREATMENTS == {
        profile_id: dict(spec.capabilities.to_dict())
        for profile_id, spec in PROFILE_SPECS.items()
    }


def test_metrics_and_abstention() -> None:
    cases = load_cases()
    predictions = {row["case_id"]: {"answer": row["gold_answer"], "abstained": row["must_abstain"], "cited_evidence_ids": row["required_evidence_ids"], "trace_events": ["rewrite"]} for row in cases}
    metrics, per_case = evaluate_predictions(cases, predictions)
    assert metrics["total"] == 100
    assert metrics["answer_contract_match_rate"] == 1.0
    assert metrics["abstention_contract_match_rate"] == 1.0
    assert metrics["performance_claim_allowed"] is False
    assert metrics["performance_claim_status"] == "SYNTHETIC_CONTRACT_ONLY"
    assert len(per_case) == 100


def test_reviewed_answer_contract_accepts_gold_answer_and_allowed_variants() -> None:
    case = _reviewed_case(allowed_variants=["营收同比增加二十亿元。"])

    variant = evaluate_case(case, {"answer": "营收同比增加二十亿元。"})
    canonical = evaluate_case(case, {"answer": case["gold_answer"]})

    assert variant["answer_contract_match"] == 1.0
    assert canonical["answer_contract_match"] == 1.0


def test_metrics_include_quality_cost_and_tool_contract() -> None:
    cases = load_cases()
    row = next(case for case in cases if case["category"] == "derived_calculation")
    prediction = {
        "case_id": row["case_id"],
        "answer": row["gold_answer"],
        "abstained": False,
        "cited_evidence_ids": row["required_evidence_ids"],
        "tool_calls": [{"tool_name": "calculator"}],
        "calculations": {"K1": {"status": "SUCCESS", "verified": True}},
        "claims": [{"verification": {"status": "ENTAILED"}}],
        # Legacy self-reported values must not be trusted by the evaluator.
        "calculation_accuracy": 0.0,
        "claim_support_accuracy": 0.0,
        "llm_call_count": 1,
        "total_tokens": 12,
        "end_to_end_latency_ms": 4.5,
    }
    metrics, per_case = evaluate_predictions([row], {row["case_id"]: prediction})
    assert metrics["Claim Entailment Yield"] == 1.0
    assert metrics["Calculation Execution Success Rate"] == 1.0
    assert metrics["process_metrics"]["Claim Entailment Yield"] == {
        "numerator": 1,
        "denominator": 1,
        "micro": 1.0,
        "macro": 1.0,
    }
    assert "gold_metrics" not in metrics
    assert metrics["Tool Selection Recall"] == 1.0
    assert metrics["Avg Tokens"] == 12.0
    assert per_case[0]["calculation_execution_success_rate"] == 1.0
    assert not any("accuracy" in key.lower() for key in _all_metric_keys(metrics))


def test_process_metrics_use_only_emitted_runtime_records_and_report_counts() -> None:
    row = next(case for case in load_cases() if case["category"] == "derived_calculation")
    prediction = {
        "claims": [
            {"verification": {"status": "ENTAILED"}},
            {"verification": {"status": "INSUFFICIENT"}},
        ],
        "calculations": [
            {"calculation_id": "actual-1", "status": "SUCCESS"},
            {"calculation_id": "actual-2", "status": "FAILED"},
        ],
        "cited_evidence_ids": ["not-in-gold"],
        "tool_calls": [{"tool_name": "not-required"}],
    }

    metrics, _ = evaluate_predictions([row], {row["case_id"]: prediction})

    assert metrics["process_metrics"]["Claim Entailment Yield"] == {
        "numerator": 1,
        "denominator": 2,
        "micro": 0.5,
        "macro": 0.5,
    }
    assert metrics["process_metrics"]["Calculation Execution Success Rate"] == {
        "numerator": 1,
        "denominator": 2,
        "micro": 0.5,
        "macro": 0.5,
    }
    assert "Evidence Recall" not in metrics["process_metrics"]
    assert "Citation Precision" not in metrics["process_metrics"]
    assert "Tool Selection Recall" not in metrics["process_metrics"]
    assert "gold_metrics" not in metrics


def test_claim_entailment_prefers_explicit_verification_status_over_legacy_supported() -> None:
    row = next(case for case in load_cases() if case["category"] == "derived_calculation")
    prediction = {
        "claims": [
            {
                "verification": {"status": "CONTRADICTED"},
                "supported": True,
            },
            {
                "status": "INSUFFICIENT",
                "supported": True,
            },
            {"supported": True},
        ]
    }

    metrics, _ = evaluate_predictions([row], {row["case_id"]: prediction})

    assert metrics["process_metrics"]["Claim Entailment Yield"] == {
        "numerator": 1,
        "denominator": 3,
        "micro": 0.3333,
        "macro": 0.3333,
    }


def test_reviewed_gold_reports_independent_claim_and_calculation_accuracy() -> None:
    case = _reviewed_case()
    prediction = {
        "answer": case["gold_answer"],
        "abstained": False,
        "cited_evidence_ids": ["e-2025", "e-2024"],
        "tool_calls": [{"tool_name": "report_search"}, {"tool_name": "calculator"}],
        "claims": [
            {
                "claim_id": "g-1",
                "text": "营收增加 20 亿元",
                "evidence_ids": ["e-2025"],
                "verification": {"status": "ENTAILED"},
            }
        ],
        "calculations": [
            {
                "calculation_id": "gold-k1",
                "status": "SUCCESS",
                "operation": "subtract",
                "formula": "prior - current",
                "inputs": [
                    {"name": "current", "value": "100.0", "unit": "亿元", "evidence_id": "e-2025"},
                    {"name": "prior", "value": "80", "unit": "亿元", "evidence_id": "e-2024"},
                ],
                "result": {"value": "20.0", "unit": "亿元"},
            }
        ],
    }

    metrics, per_case = evaluate_predictions(
        [case],
        {case["case_id"]: prediction},
        reviewed_target_count=1,
    )

    assert metrics["benchmark_kind"] == "reviewed"
    assert metrics["Claim Status Accuracy"] == 1.0
    assert metrics["Claim Evidence Accuracy"] == 0.0
    assert metrics["Claim Verification Accuracy"] == 0.0
    assert metrics["Calculation Status Accuracy"] == 1.0
    assert metrics["Calculation Evidence Accuracy"] == 1.0
    assert metrics["Calculation Formula Accuracy"] == 0.0
    assert metrics["Calculation Inputs Accuracy"] == 1.0
    assert metrics["Calculation Result Accuracy"] == 1.0
    assert metrics["Calculation Record Accuracy"] == 0.0
    assert metrics["process_metrics"]["Calculation Execution Success Rate"]["micro"] == 1.0
    assert metrics["gold_metrics"]["Claim Verification Accuracy"]["micro"] == 0.0
    assert metrics["performance_claim_allowed"] is False
    assert metrics["performance_claim_status"] == "COVERAGE_ONLY"
    assert metrics["reviewed_coverage"]["target_met"] is True
    assert per_case[0]["calculation_comparison"][0]["formula_match"] is False


def test_reviewed_gold_metrics_report_micro_macro_and_explicit_counts() -> None:
    first = _reviewed_case(case_id="REV-001", gold_calculations=[])
    second = _reviewed_case(
        case_id="REV-002",
        gold_calculations=[],
        gold_claims=[
            {
                "claim_id": "g-1",
                "text": "营收增加 20 亿元",
                "status": "ENTAILED",
                "evidence_ids": ["e-2025", "e-2024"],
            },
            {
                "claim_id": "g-2",
                "text": "没有足够证据支持利润结论",
                "status": "INSUFFICIENT",
                "evidence_ids": [],
            },
        ],
        partial_answer_gold={"allowed": True, "required_claim_ids": ["g-1"]},
    )
    predictions = {
        "REV-001": {
            "claims": [
                {
                    "claim_id": "g-1",
                    "text": "营收增加 20 亿元",
                    "evidence_ids": ["e-2025", "e-2024"],
                    "verification": {"status": "ENTAILED"},
                }
            ]
        },
        "REV-002": {
            "claims": [
                {
                    "claim_id": "g-1",
                    "text": "营收增加 20 亿元",
                    "evidence_ids": ["e-2025", "e-2024"],
                    "verification": {"status": "ENTAILED"},
                }
            ]
        },
    }

    metrics, _ = evaluate_predictions([first, second], predictions)

    assert metrics["gold_metrics"]["Claim Status Accuracy"] == {
        "numerator": 2,
        "denominator": 3,
        "micro": 0.6667,
        "macro": 0.75,
    }
    assert metrics["Claim Status Accuracy"] == 0.6667


def test_non_success_gold_calculation_only_scores_applicable_fields() -> None:
    case = _reviewed_case(
        gold_claims=[],
        must_abstain=True,
        partial_answer_gold={"allowed": False, "required_claim_ids": []},
        gold_calculations=[
            {
                "calculation_id": "gold-k1",
                "status": "INSUFFICIENT",
                "operation": "subtract",
                "reason": "prior period is missing",
            }
        ],
    )
    prediction = {
        "abstained": True,
        "calculations": [
            {
                "calculation_id": "gold-k1",
                "status": "INSUFFICIENT",
                "operation": "subtract",
            }
        ],
    }

    metrics, rows = evaluate_predictions([case], {case["case_id"]: prediction})

    assert metrics["gold_metrics"]["Calculation Status Accuracy"]["micro"] == 1.0
    assert metrics["gold_metrics"]["Calculation Result Accuracy"] == {
        "numerator": 0,
        "denominator": 0,
        "micro": None,
        "macro": None,
    }
    assert rows[0]["calculation_comparison"][0]["result_match"] is None


def test_extra_predicted_calculation_reduces_every_calculation_gold_dimension() -> None:
    case = _reviewed_case(gold_claims=[])
    correct = deepcopy(case["gold_calculations"][0])
    extra = {
        "calculation_id": "predicted-extra",
        "status": "SUCCESS",
        "operation": "ratio",
        "inputs": [
            {"name": "numerator", "value": "1", "unit": "元"},
            {"name": "denominator", "value": "2", "unit": "元"},
        ],
        "result": {"value": "0.5", "unit": "ratio"},
    }

    metrics, rows = evaluate_predictions(
        [case],
        {case["case_id"]: {"calculations": [correct, extra]}},
    )

    for metric_name in (
        "Calculation Status Accuracy",
        "Calculation Operation Accuracy",
        "Calculation Evidence Accuracy",
        "Calculation Formula Accuracy",
        "Calculation Inputs Accuracy",
        "Calculation Result Accuracy",
        "Calculation Record Accuracy",
    ):
        assert metrics["gold_metrics"][metric_name] == {
            "numerator": 1,
            "denominator": 2,
            "micro": 0.5,
            "macro": 0.5,
        }
    unmatched = rows[0]["calculation_comparison"][1]
    assert unmatched["gold_calculation_id"] is None
    assert unmatched["predicted_calculation_id"] == "predicted-extra"
    assert {
        key: unmatched[key]
        for key in (
            "status_match",
            "operation_match",
            "evidence_match",
            "formula_match",
            "inputs_match",
            "result_match",
            "record_match",
        )
    } == {
        "status_match": False,
        "operation_match": False,
        "evidence_match": False,
        "formula_match": False,
        "inputs_match": False,
        "result_match": False,
        "record_match": False,
    }


def test_duplicate_predicted_calculation_is_counted_as_an_extra_record() -> None:
    case = _reviewed_case(gold_claims=[])
    correct = deepcopy(case["gold_calculations"][0])

    metrics, rows = evaluate_predictions(
        [case],
        {case["case_id"]: {"calculations": [correct, deepcopy(correct)]}},
    )

    result_metric = metrics["gold_metrics"]["Calculation Result Accuracy"]
    record_metric = metrics["gold_metrics"]["Calculation Record Accuracy"]
    assert (result_metric["numerator"], result_metric["denominator"]) == (1, 2)
    assert (record_metric["numerator"], record_metric["denominator"]) == (1, 2)
    assert sum(
        comparison["unmatched_prediction"]
        for comparison in rows[0]["calculation_comparison"]
    ) == 1


def test_unmatched_prediction_counts_both_missing_gold_and_false_positive() -> None:
    case = _reviewed_case(gold_claims=[])
    unmatched_prediction = {
        "calculation_id": "wrong-id",
        "status": "SUCCESS",
        "operation": "ratio",
        "inputs": [],
        "result": {"value": "999", "unit": "ratio"},
    }

    metrics, rows = evaluate_predictions(
        [case],
        {case["case_id"]: {"calculations": [unmatched_prediction]}},
    )

    assert metrics["gold_metrics"]["Calculation Result Accuracy"] == {
        "numerator": 0,
        "denominator": 2,
        "micro": 0.0,
        "macro": 0.0,
    }
    assert metrics["gold_metrics"]["Calculation Record Accuracy"] == {
        "numerator": 0,
        "denominator": 2,
        "micro": 0.0,
        "macro": 0.0,
    }
    comparisons = rows[0]["calculation_comparison"]
    assert comparisons[0]["predicted_calculation_id"] is None
    assert comparisons[0]["record_match"] is False
    assert comparisons[1]["gold_calculation_id"] is None
    assert comparisons[1]["unmatched_prediction"] is True


def test_reviewed_metrics_remain_coverage_only_until_declared_target_is_met() -> None:
    case = _reviewed_case()

    metrics, _ = evaluate_predictions(
        [case],
        {case["case_id"]: {"answer": case["gold_answer"]}},
        reviewed_target_count=10,
    )

    assert metrics["reviewed_coverage"] == {
        "reviewed_case_count": 1,
        "target_case_count": 10,
        "coverage_rate": 0.1,
        "target_met": False,
    }
    assert metrics["performance_claim_allowed"] is False
    assert metrics["performance_claim_status"] == "COVERAGE_ONLY"


def test_legacy_structural_manifest_cannot_pass_reviewed_publish_gate() -> None:
    cases = _complete_reviewed_cases()
    manifest = _complete_reviewed_manifest(cases)
    predictions = {
        case["case_id"]: {"answer": case["gold_answer"]}
        for case in cases
    }

    metrics, _ = evaluate_predictions(
        cases,
        predictions,
        reviewed_target_count=len(cases),
        reviewed_manifest=manifest,
    )

    assert metrics["performance_claim_allowed"] is False
    assert metrics["publish_gate"]["blocking_reasons"][:2] == [
        "reviewed_source_files_not_verified",
        "reviewed_release_attestation_required",
    ]


def test_reviewed_publish_gate_requires_manifest_target_category_source_and_prediction_coverage(
    tmp_path: Path,
) -> None:
    cases = _complete_reviewed_cases()
    manifest, _, _ = _verified_reviewed_manifest(tmp_path, cases)
    predictions = {
        case["case_id"]: {"answer": case["gold_answer"]}
        for case in cases
    }

    metrics, _ = evaluate_predictions(
        cases,
        predictions,
        reviewed_target_count=len(cases),
        reviewed_manifest=manifest,
    )

    assert metrics["performance_claim_allowed"] is True
    assert metrics["performance_claim_status"] == "REVIEWED_TARGET_MET"
    assert metrics["publish_gate"] == {
        "status": "READY",
        "allowed": True,
        "checks": {
            "reviewed_manifest_valid": True,
            "reviewed_source_files_verified": True,
            "reviewed_release_attestation_validated": True,
            "reviewed_target_declared": True,
            "reviewed_target_met": True,
            "reviewed_claim_gold_denominator_nonzero": True,
            "reviewed_calculation_gold_denominator_nonzero": True,
            "all_categories_declared": True,
            "category_quotas_met": True,
            "source_quotas_declared": True,
            "source_quotas_met": True,
            "prediction_coverage_met": True,
        },
        "blocking_reasons": [],
    }
    assert metrics["reviewed_manifest_coverage"]["category_counts"] == {
        category: 1 for category in CATEGORIES
    }

    incomplete_predictions = dict(predictions)
    incomplete_predictions.pop(cases[-1]["case_id"])
    blocked, _ = evaluate_predictions(
        cases,
        incomplete_predictions,
        reviewed_target_count=len(cases),
        reviewed_manifest=manifest,
    )
    assert blocked["performance_claim_allowed"] is False
    assert blocked["publish_gate"]["blocking_reasons"] == [
        "prediction_coverage_incomplete"
    ]


def test_reviewed_publish_gate_requires_nonzero_claim_and_calculation_gold_denominators(
    tmp_path: Path,
) -> None:
    without_calculations = [
        _reviewed_case(
            case_id=f"REV-{index:03d}",
            category=category,
            gold_calculations=[],
        )
        for index, category in enumerate(CATEGORIES, start=1)
    ]
    calculation_manifest, _, _ = _verified_reviewed_manifest(
        tmp_path / "calculation",
        without_calculations,
    )
    calculation_blocked, _ = evaluate_predictions(
        without_calculations,
        {
            case["case_id"]: {"answer": case["gold_answer"]}
            for case in without_calculations
        },
        reviewed_target_count=len(without_calculations),
        reviewed_manifest=calculation_manifest,
    )

    assert calculation_blocked["performance_claim_allowed"] is False
    assert calculation_blocked["publish_gate"]["checks"][
        "reviewed_claim_gold_denominator_nonzero"
    ] is True
    assert calculation_blocked["publish_gate"]["checks"][
        "reviewed_calculation_gold_denominator_nonzero"
    ] is False
    assert calculation_blocked["publish_gate"]["blocking_reasons"] == [
        "reviewed_calculation_gold_denominator_zero"
    ]

    without_claims = [
        _reviewed_case(
            case_id=f"REV-{index:03d}",
            category=category,
            gold_claims=[],
            must_abstain=True,
            partial_answer_gold={"allowed": False, "required_claim_ids": []},
        )
        for index, category in enumerate(CATEGORIES, start=1)
    ]
    claim_manifest, _, _ = _verified_reviewed_manifest(
        tmp_path / "claim",
        without_claims,
    )
    claim_blocked, _ = evaluate_predictions(
        without_claims,
        {
            case["case_id"]: {"abstained": True}
            for case in without_claims
        },
        reviewed_target_count=len(without_claims),
        reviewed_manifest=claim_manifest,
    )

    assert claim_blocked["performance_claim_allowed"] is False
    assert claim_blocked["publish_gate"]["checks"][
        "reviewed_claim_gold_denominator_nonzero"
    ] is False
    assert claim_blocked["publish_gate"]["checks"][
        "reviewed_calculation_gold_denominator_nonzero"
    ] is True
    assert claim_blocked["publish_gate"]["blocking_reasons"] == [
        "reviewed_claim_gold_denominator_zero"
    ]


def test_reviewed_tool_and_partial_gold_contracts_are_scored() -> None:
    case = _reviewed_case()
    prediction = {
        "answer": "营收增加 20 亿元",
        "tool_calls": [
            {"tool_name": "report_search"},
            {"tool_name": "calculator"},
            {"tool_name": "web_search"},
        ],
        "claims": [
            {
                "claim_id": "g-1",
                "text": "营收增加 20 亿元",
                "evidence_ids": ["e-2025", "e-2024"],
                "verification": {"status": "ENTAILED"},
            }
        ],
    }

    row = evaluate_case(case, prediction)

    assert row["tool_selection_recall"] == 1.0
    assert row["forbidden_tool_contract_satisfied"] == 0.0
    assert row["partial_answer_gold_score"] == 1.0


def test_unnecessary_tool_calls_treat_an_empty_gold_contract_as_all_unnecessary() -> None:
    case = _reviewed_case(required_tools=[])

    row = evaluate_case(
        case,
        {
            "tool_calls": [
                {"tool_name": "report_search"},
                {"tool_name": "calculator"},
            ]
        },
    )

    assert row["unnecessary_tool_call_numerator"] == 2
    assert row["unnecessary_tool_call_denominator"] == 2
    assert row["unnecessary_tool_call_rate"] == 1.0


def test_unnecessary_tool_calls_count_each_non_required_invocation() -> None:
    case = _reviewed_case(required_tools=["report_search"])

    row = evaluate_case(
        case,
        {
            "tool_calls": [
                {"tool_name": "report_search"},
                {"tool_name": "report_search"},
                {"tool_name": "web_search"},
                {"tool_name": "web_search"},
            ]
        },
    )

    # required_tools specifies allowed tool types, not the number of calls.
    # Both non-required invocations count; repeated required calls do not.
    assert row["unnecessary_tool_call_numerator"] == 2
    assert row["unnecessary_tool_call_denominator"] == 4
    assert row["unnecessary_tool_call_rate"] == 0.5


def test_unnecessary_tool_call_aggregate_uses_call_level_micro_rate() -> None:
    empty_contract = _reviewed_case(
        case_id="REV-EMPTY-TOOLS",
        required_tools=[],
    )
    typed_contract = _reviewed_case(
        case_id="REV-TYPED-TOOLS",
        required_tools=["report_search"],
    )
    predictions = {
        empty_contract["case_id"]: {
            "tool_calls": [{"tool_name": "calculator"}],
        },
        typed_contract["case_id"]: {
            "tool_calls": [
                {"tool_name": "report_search"},
                {"tool_name": "report_search"},
                {"tool_name": "web_search"},
            ],
        },
    }

    metrics, _ = evaluate_predictions(
        [empty_contract, typed_contract],
        predictions,
    )

    detail = metrics["unnecessary_tool_call_rate_detail"]
    assert detail["numerator"] == 2
    assert detail["denominator"] == 4
    assert detail["micro"] == 0.5
    assert detail["macro"] == pytest.approx((1.0 + 0.3333) / 2, abs=0.0001)
    assert metrics["Unnecessary Tool Call Rate"] == detail["micro"]
    category_detail = metrics["category_metrics"]["derived_calculation"][
        "contract_metrics"
    ]["Unnecessary Tool Call Rate"]
    assert category_detail == detail


def test_reviewed_evaluation_materializes_gold_causal_failure_attribution() -> None:
    case = _reviewed_case(
        gold_claims=[
            {
                "claim_id": "g-1",
                "text": "营收增加 20 亿元",
                "status": "INSUFFICIENT",
                "evidence_ids": [],
            }
        ]
    )
    prediction = {
        "claims": [
            {
                "claim_id": "g-1",
                "text": "营收增加 20 亿元",
                "evidence_ids": [],
                "verification": {"status": "ENTAILED"},
            }
        ],
        "abstained": False,
    }

    metrics, rows = evaluate_predictions([case], {case["case_id"]: prediction})

    assert rows[0]["failure_attribution"]["root_cause"] == "verification_false_positive"
    assert metrics["failure_summary"]["root_cause_counts"] == {"verification_false_positive": 1}


def test_bundle_contains_contract_files(tmp_path: Path) -> None:
    cases = load_cases()
    bundle = write_run_bundle(tmp_path, cases=cases, profile=PROFILES[-1], seed=7)
    assert bundle["config"]["config_hash"]
    assert bundle["config"]["feature_treatment"] == PROFILE_TREATMENTS["full-agent"]
    for name in ("config.json", "metrics.json", "per_case.jsonl", "trajectories.jsonl", "metadata.json"):
        assert (tmp_path / name).exists()
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["seed"] == 7
    assert metadata["synthetic"] is True


def test_reviewed_bundle_uses_recomputed_benchmark_identity(tmp_path: Path) -> None:
    case = _reviewed_case()

    bundle = write_run_bundle(tmp_path, cases=[case], profile="full-agent")

    assert bundle["metadata"]["synthetic"] is False
    assert bundle["metadata"]["review_status"] == "reviewed"
    assert bundle["metadata"]["benchmark_version"] == "reviewed-v1"
    assert bundle["metadata"]["benchmark_hash"] == canonical_reviewed_cases_hash([case])
    assert bundle["metadata"]["benchmark_hash"] != case["benchmark_hash"]
