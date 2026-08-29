"""Execute the four canonical profiles under one strict comparison context."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from src.evaluation.eval_bundle import (
    EvalBundle,
    EvalBundleIntegrityError,
    require_comparable_eval_bundles,
    validate_corpus_asset_manifest,
    write_validated_eval_bundle,
)
from src.evaluation.failure_attribution import attribute_failure
from src.evaluation.hard_case_benchmark import (
    canonical_reviewed_cases_hash,
    evaluate_predictions,
    validate_cases,
    validate_reviewed_manifest,
)
from src.evaluation.profile_runtime import PROFILE_IDS, PROFILE_SPECS, ProfileSpec
from src.evaluation.run_identity import (
    RunIdentity,
    canonical_sha256,
    hash_benchmark_rows,
    hash_case_ids,
)


ProfileExecutor = Callable[
    [ProfileSpec, Mapping[str, Any]],
    Mapping[str, Any],
]


class TreatmentIntegrityError(ValueError):
    """Raised when execution evidence disagrees with the declared treatment."""


@dataclass(frozen=True)
class ProfileMatrixResult:
    bundles: Mapping[str, EvalBundle]
    summary: Mapping[str, Any]


def _profile_identity(base: RunIdentity, spec: ProfileSpec) -> RunIdentity:
    value = base.to_dict()
    flags = dict(value["feature_flags"])
    flags["treatments"] = spec.effective_treatments
    value["feature_flags"] = flags
    return RunIdentity.from_dict(value)


def _assert_base_identity(cases: Sequence[Mapping[str, Any]], identity: RunIdentity) -> None:
    expected_count = len(cases)
    expected_case_ids = hash_case_ids(str(case["case_id"]) for case in cases)
    reviewed = bool(cases) and all(case.get("synthetic") is False for case in cases)
    expected_benchmark = (
        canonical_reviewed_cases_hash(cases) if reviewed else hash_benchmark_rows(cases)
    )
    mismatches: list[str] = []
    if identity.case_count != expected_count:
        mismatches.append("case_count")
    if identity.case_ids_hash != expected_case_ids:
        mismatches.append("case_ids_hash")
    if identity.benchmark_hash != expected_benchmark:
        mismatches.append("benchmark_hash")
    if dict(identity.feature_flags.get("treatments") or {}):
        mismatches.append("feature_flags.treatments (base must be empty)")
    runtime_flags = identity.feature_flags.get("runtime")
    retrieval_source = (
        str(runtime_flags.get("retrieval_source") or "").strip()
        if isinstance(runtime_flags, Mapping)
        else ""
    )
    allowed_sources = {"shared-corpus", "synthetic-contract-oracle"}
    if retrieval_source not in allowed_sources:
        mismatches.append("feature_flags.runtime.retrieval_source")
    elif reviewed and retrieval_source != "shared-corpus":
        mismatches.append("reviewed benchmark requires retrieval_source=shared-corpus")
    if mismatches:
        raise ValueError("base RunIdentity does not match benchmark rows: " + ", ".join(mismatches))


def _require_effective_treatments(
    actual: Any,
    expected: Mapping[str, Any],
    *,
    location: str,
) -> None:
    if not isinstance(actual, Mapping) or canonical_sha256(actual) != canonical_sha256(expected):
        raise TreatmentIntegrityError(
            f"{location} effective_treatments do not exactly match RunIdentity treatments"
        )


def _require_chunks_asset(chunks_path: Path) -> Path:
    source = Path(chunks_path)
    if not source.is_file():
        raise FileNotFoundError(f"chunks asset does not exist: {source}")
    row_count = 0
    seen_ids: set[str] = set()
    try:
        with source.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"chunks asset contains invalid JSON at line {line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(row, Mapping):
                    raise ValueError(f"chunks asset row {line_number} must be a JSON object")
                chunk_id = str(
                    row.get("chunk_id") or row.get("evidence_id") or row.get("id") or ""
                ).strip()
                if not chunk_id:
                    raise ValueError(f"chunks asset row {line_number} requires a chunk id")
                document = str(
                    row.get("doc_id") or row.get("document_id") or row.get("document") or ""
                ).strip()
                page = row.get("page") or row.get("page_num") or row.get("page_number")
                if page is None:
                    page = row.get("page_start") or row.get("page_end")
                if not document or page in (None, ""):
                    raise ValueError(
                        f"chunks asset row {line_number} requires document and page metadata"
                    )
                if chunk_id in seen_ids:
                    raise ValueError(
                        f"chunks asset contains duplicate chunk id at line {line_number}: {chunk_id}"
                    )
                seen_ids.add(chunk_id)
                row_count += 1
    except OSError as exc:
        raise ValueError(f"chunks asset cannot be read: {source}: {exc}") from exc
    if row_count == 0:
        raise ValueError(f"chunks asset must contain at least one row: {source}")
    return source


def _trajectory_row(
    spec: ProfileSpec,
    case_id: str,
    prediction: Mapping[str, Any],
    identity: RunIdentity,
) -> dict[str, Any]:
    raw = prediction.get("trace_events")
    if not isinstance(raw, list):
        raw = prediction.get("trajectory_events")
    if not isinstance(raw, list) or not raw:
        raise TreatmentIntegrityError(
            f"profile={spec.profile_id} case_id={case_id} requires non-empty trace events"
        )
    events: list[dict[str, Any]] = []
    for index, event in enumerate(raw):
        if not isinstance(event, Mapping):
            raise TreatmentIntegrityError(
                f"profile={spec.profile_id} case_id={case_id} trace[{index}] must be a mapping"
            )
        _require_effective_treatments(
            event.get("effective_treatments"),
            spec.effective_treatments,
            location=f"profile={spec.profile_id} case_id={case_id} trace[{index}]",
        )
        events.append(dict(event))
    row = dict(prediction)
    row.pop("trace_events", None)
    row["trajectory_events"] = events
    row["profile"] = spec.profile_id
    row["case_id"] = case_id
    row["effective_treatments"] = spec.effective_treatments
    row["run_id"] = identity.run_id
    row["run_identity"] = identity.to_dict()
    row.setdefault("claims", [])
    row.setdefault("citations", [])
    row.setdefault("calculations", {})
    row.setdefault("used_evidence_ids", [])
    row.setdefault("tool_calls", [])
    row.setdefault("dependency_coverage", {})
    row.setdefault("abstained", False)
    row.setdefault("failure_attribution", attribute_failure(row))
    return row


def _failed_profile_prediction(
    spec: ProfileSpec,
    *,
    case_id: str,
    error: Exception,
    latency_ms: float,
) -> dict[str, Any]:
    """Materialize a case-scoped execution failure without aborting the matrix.

    Benchmark/configuration, corpus, identity and treatment checks remain outside
    this seam and still fail hard.  Only an exception raised while executing one
    treatment/case becomes a scored, auditable failed prediction.
    """

    event = {
        "event_id": f"{spec.profile_id}-{case_id}-executor-failed",
        "event_type": "materialization",
        "node": "profile_executor",
        "action": "materialize_case_failure",
        "step": 0,
        "status": "FAILED",
        "latency_ms": round(float(latency_ms), 2),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "dependencies": [],
        "recovery_of": [],
        "lineage_status": "resolved",
        "budget_usage": {
            "steps": 0,
            "llm_calls": 0,
            "tool_calls": 0,
            "tokens": 0,
            "retrievals": 0,
        },
        "effective_treatments": spec.effective_treatments,
    }
    prediction: dict[str, Any] = {
        "case_id": case_id,
        "answer": "",
        "final_answer": "",
        "evidence_summary": "",
        "abstained": True,
        "abstain_reason": "runtime_exception",
        "failed": True,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "termination_reason": "runtime_exception",
        "cited_evidence_ids": [],
        "used_evidence_ids": [],
        "citations": [],
        "selected_doc_ids": [],
        "claims": [],
        "calculations": {},
        "tool_calls": [],
        "tool_call_count": 0,
        "llm_call_count": 0,
        "total_tokens": 0,
        "reasoning_plan": {},
        "reasoning_step_results": [],
        "reasoning_conclusion": {},
        "dependency_coverage": {
            "decision": "failed",
            "coverage_ratio": 0.0,
        },
        "trace_events": [event],
        "end_to_end_latency_ms": round(float(latency_ms), 2),
        "effective_treatments": spec.effective_treatments,
    }
    prediction["failure_attribution"] = attribute_failure(prediction)
    return prediction


def run_profile_matrix(
    *,
    cases: Sequence[Mapping[str, Any]],
    base_identity: RunIdentity,
    executor: ProfileExecutor,
    output_root: Path,
    chunks_path: Path,
    corpus_asset_manifest: Mapping[str, Any] | None = None,
    reviewed_manifest: Mapping[str, Any] | None = None,
    reviewed_target_count: int | None = None,
    overwrite: bool = False,
) -> ProfileMatrixResult:
    """Run every case under every canonical profile and write strict bundles.

    The executor is the single evaluation seam used by all treatments.  It is
    passed the canonical :class:`ProfileSpec`; every prediction and trace event
    must echo its exact ``effective_treatments``.  Any mismatch aborts before a
    result can be labelled with a treatment that was not actually applied.
    Reviewed runs additionally bind the validated manifest hash and explicit
    release target into the base identity before any profile executes.
    """

    if not callable(executor):
        raise TypeError("executor must be callable")
    if reviewed_target_count is not None and (
        isinstance(reviewed_target_count, bool)
        or int(reviewed_target_count) != reviewed_target_count
        or int(reviewed_target_count) <= 0
    ):
        raise ValueError("reviewed_target_count must be a positive integer when provided")
    rows = validate_cases(cases)
    _assert_base_identity(rows, base_identity)
    chunks_path = _require_chunks_asset(chunks_path)
    reviewed = bool(rows) and all(case.get("synthetic") is False for case in rows)
    validated_corpus_manifest: dict[str, Any] | None = None
    if corpus_asset_manifest is not None:
        validated_corpus_manifest = validate_corpus_asset_manifest(
            base_identity, corpus_asset_manifest, verify_files=True
        )
        declared_chunks = Path(validated_corpus_manifest["paths"]["chunks"]).resolve()
        if declared_chunks != chunks_path.resolve():
            raise EvalBundleIntegrityError(
                "corpus asset manifest chunks path does not match profile matrix chunks_path"
            )
        runtime_flags = base_identity.feature_flags.get("runtime")
        declared_manifest_hash = (
            runtime_flags.get("corpus_asset_manifest_hash")
            if isinstance(runtime_flags, Mapping)
            else None
        )
        if declared_manifest_hash != validated_corpus_manifest["manifest_hash"]:
            raise EvalBundleIntegrityError(
                "RunIdentity corpus_asset_manifest_hash does not match corpus asset manifest"
            )
    elif reviewed:
        raise EvalBundleIntegrityError(
            "reviewed profile matrix requires an explicit corpus asset manifest"
        )
    if reviewed and reviewed_manifest is None:
        raise ValueError("reviewed profile matrix requires a validated benchmark manifest")
    if not reviewed and reviewed_manifest is not None:
        raise ValueError("synthetic profile matrix must not receive a reviewed manifest")
    validated_manifest = (
        validate_reviewed_manifest(reviewed_manifest, rows)
        if reviewed_manifest is not None
        else None
    )
    if validated_manifest is not None:
        runtime_flags = base_identity.feature_flags.get("runtime")
        declared_manifest_hash = (
            runtime_flags.get("reviewed_manifest_hash")
            if isinstance(runtime_flags, Mapping)
            else None
        )
        expected_manifest_hash = canonical_sha256(validated_manifest)
        if declared_manifest_hash != expected_manifest_hash:
            raise ValueError(
                "base RunIdentity reviewed_manifest_hash does not match validated manifest"
            )
    runtime_flags = base_identity.feature_flags.get("runtime")
    declared_target = (
        runtime_flags.get("reviewed_target_count") if isinstance(runtime_flags, Mapping) else None
    )
    if declared_target != reviewed_target_count:
        raise ValueError("base RunIdentity reviewed_target_count does not match matrix request")
    retrieval_source = str(
        runtime_flags.get("retrieval_source") if isinstance(runtime_flags, Mapping) else ""
    )
    eval_root = Path(output_root) / "eval"
    bundles: dict[str, EvalBundle] = {}
    profile_metrics: dict[str, Mapping[str, Any]] = {}

    for profile in PROFILE_IDS:
        spec = PROFILE_SPECS[profile]
        treatments = spec.effective_treatments
        identity = _profile_identity(base_identity, spec)
        _require_effective_treatments(
            identity.feature_flags.get("treatments"),
            treatments,
            location=f"profile={profile} RunIdentity",
        )
        predictions: dict[str, dict[str, Any]] = {}
        trajectories: list[dict[str, Any]] = []
        for case in rows:
            case_id = str(case["case_id"])
            started = perf_counter()
            try:
                prediction = dict(executor(spec, case))
            except (TreatmentIntegrityError, EvalBundleIntegrityError):
                raise
            except Exception as exc:
                prediction = _failed_profile_prediction(
                    spec,
                    case_id=case_id,
                    error=exc,
                    latency_ms=(perf_counter() - started) * 1000,
                )
            _require_effective_treatments(
                prediction.get("effective_treatments"),
                treatments,
                location=f"profile={profile} case_id={case_id} prediction",
            )
            returned_case_id = prediction.get("case_id")
            if returned_case_id is not None and str(returned_case_id) != case_id:
                raise ValueError(
                    f"profile={profile} executor returned case_id={returned_case_id!r}, "
                    f"expected {case_id!r}"
                )
            prediction["case_id"] = case_id
            prediction.setdefault("end_to_end_latency_ms", (perf_counter() - started) * 1000)
            predictions[case_id] = prediction
            trajectories.append(_trajectory_row(spec, case_id, prediction, identity))

        evaluation_options: dict[str, Any] = {
            "reviewed_target_count": reviewed_target_count,
        }
        if validated_manifest is not None:
            evaluation_options["reviewed_manifest"] = validated_manifest
        metrics, per_case = evaluate_predictions(rows, predictions, **evaluation_options)
        per_case_with_treatments = [
            {**dict(row), "effective_treatments": treatments} for row in per_case
        ]
        comparison_context_hash = canonical_sha256(identity.comparison_identity())
        bundle = write_validated_eval_bundle(
            eval_root,
            identity=identity,
            config={
                "comparison_context_hash": comparison_context_hash,
                "reviewed_manifest_hash": (
                    canonical_sha256(validated_manifest) if validated_manifest is not None else None
                ),
                "reviewed_target_count": reviewed_target_count,
                "retrieval_source": retrieval_source,
                "treatments": treatments,
            },
            metrics=metrics,
            per_case=per_case_with_treatments,
            trajectories=trajectories,
            chunks_path=Path(chunks_path),
            corpus_asset_manifest=validated_corpus_manifest,
            metadata={
                "profile": profile,
                "mode": "agentic" if spec.enters_graph else "baseline",
                "evidence_scope": retrieval_source,
                "benchmark_kind": metrics["benchmark_kind"],
                "performance_claim_status": metrics["performance_claim_status"],
            },
            overwrite=overwrite,
        )
        bundles[profile] = bundle
        profile_metrics[profile] = metrics

    require_comparable_eval_bundles(list(bundles.values()))
    claim_allowed = all(
        bool(metrics.get("performance_claim_allowed")) for metrics in profile_metrics.values()
    )
    benchmark_kind = str(next(iter(profile_metrics.values()))["benchmark_kind"])
    if benchmark_kind == "synthetic":
        status = "SYNTHETIC_CONTRACT_ONLY"
    elif claim_allowed:
        status = "READY_FOR_REVIEWED_COMPARISON"
    else:
        status = "COVERAGE_ONLY"
    return ProfileMatrixResult(
        bundles=bundles,
        summary={
            "profiles": list(PROFILE_IDS),
            "run_ids": {profile: bundle.identity.run_id for profile, bundle in bundles.items()},
            "bundle_paths": {profile: str(bundle.path) for profile, bundle in bundles.items()},
            "evidence_integrity": {
                profile: (
                    dict(bundle.metadata["integrity_verdict"])
                    if isinstance(bundle.metadata.get("integrity_verdict"), Mapping)
                    else {
                        "status": "NOT_APPLICABLE",
                        "reason": "baseline profile does not produce verified claims",
                    }
                )
                for profile, bundle in bundles.items()
            },
            "comparison_context_hash": canonical_sha256(base_identity.comparison_identity()),
            "benchmark_kind": benchmark_kind,
            "reviewed_manifest_hash": (
                canonical_sha256(validated_manifest) if validated_manifest is not None else None
            ),
            "reviewed_target_count": reviewed_target_count,
            "retrieval_source": retrieval_source,
            "corpus_asset_manifest_hash": (
                validated_corpus_manifest["manifest_hash"]
                if validated_corpus_manifest is not None
                else None
            ),
            "performance_claim_allowed": claim_allowed,
            "status": status,
            "profile_metrics": profile_metrics,
        },
    )


__all__ = [
    "ProfileExecutor",
    "ProfileMatrixResult",
    "TreatmentIntegrityError",
    "run_profile_matrix",
]
