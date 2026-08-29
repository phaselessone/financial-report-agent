"""Reproducible Phase 0 gates for the financial-report agent.

This module intentionally depends only on the standard library.  It provides
the same entry point on Windows and POSIX and never loads models, calls a
provider, or reads secret-bearing environment files.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_OUTPUT_DIR = Path("outputs/phase0")
DEFAULT_FULL_SEED = Path("data/eval_set/answer_eval_seed_full.jsonl")
DEFAULT_FULL_RESULTS = Path(
    "artifacts/remote_20260401/outputs_reports/answer_eval_results_full.jsonl"
)
DEFAULT_FULL_ATTESTATION = Path("benchmarks/full/historical-full-raw.attestation.json")
DEFAULT_SEMANTIC_LABELS = Path("data/eval_set/semantic_verification_labels.jsonl")

# Each group is deterministic, offline, and maps directly to a Phase 0 smoke
# capability. Keep node ids explicit so adding an unrelated test cannot silently
# expand the gate.
SMOKE_GROUPS: dict[str, tuple[str, ...]] = {
    "claims": (
        "tests/test_claim_types.py",
        "tests/test_extract_claims.py",
        "tests/test_claim_flow.py",
        "tests/test_claim_eval.py",
    ),
    "calculator": ("tests/test_financial_calculator.py",),
    "agent_flow": ("tests/test_agent_flow.py", "tests/test_agent_budget_termination.py"),
    "structured": ("tests/test_structured_schema.py", "tests/test_structured_eval.py"),
    "trajectory": ("tests/test_trajectory_eval.py",),
    "evidence": ("tests/test_evidence_integrity.py", "tests/test_semantic_calibration.py"),
    "baseline_regression": ("tests/test_baseline_regression.py",),
}
SMOKE_COMMAND = [
    sys.executable,
    "-m",
    "pytest",
    "-q",
    *[path for paths in SMOKE_GROUPS.values() for path in paths],
]

SAFE_CONFIG_ENV_KEYS = (
    "LLM_PROVIDER",
    "LLM_MODEL",
    "RUNTIME_PROFILE",
    "EMBEDDING_DEVICE",
    "RERANKER_DEVICE",
    "EMBEDDING_BATCH_SIZE",
    "RERANKER_BATCH_SIZE",
    "MODEL_OFFLINE",
)
MODEL_DEFAULTS = {
    "embedding_model": "BAAI/bge-m3",
    "reranker_model": "BAAI/bge-reranker-v2-m3",
    "llm_provider": "deepseek",
    "llm_model": "deepseek-chat",
    "runtime_profile": "low_vram",
}


class GateError(RuntimeError):
    """A user-actionable Phase 0 gate failure."""


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_text_sha256(path: Path) -> str:
    """Hash text after universal-newline decoding for cross-platform evidence."""

    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise GateError(f"{path}:{line_number}: expected a JSON object")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"cannot read JSONL asset {path}: {exc}") from exc
    return rows


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GateError(f"{label} {path} must contain a JSON object")
    return value


def _normalized_path(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


def _attested_path(value: Any, *, label: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise GateError(f"full benchmark attestation {label} must be non-empty")
    return _repo_path(Path(text))


def _attested_sha256(value: Any, *, label: str) -> str:
    text = str(value or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise GateError(f"full benchmark attestation {label} must be a 64-character SHA-256")
    return text


def _load_full_benchmark_attestation(
    attestation_path: Path,
    *,
    seed_path: Path,
    results_path: Path,
) -> dict[str, Any]:
    """Validate one review attestation that binds immutable benchmark assets.

    Historical JSONL files stay byte-for-byte immutable. Review identity and
    expected hashes live in this sidecar, which also binds the restoration
    report used to establish the 50-row reviewed contract.
    """

    resolved = _repo_path(attestation_path)
    if not resolved.is_file():
        raise GateError(f"missing external full-benchmark attestation: {resolved}")
    payload = _read_json_object(resolved, label="full benchmark attestation")
    if payload.get("schema_version") != 1:
        raise GateError("full benchmark attestation schema_version must equal 1")
    if payload.get("benchmark_profile") != "historical-full-raw":
        raise GateError(
            "full benchmark attestation benchmark_profile must equal 'historical-full-raw'"
        )
    if not str(payload.get("benchmark_id") or "").strip():
        raise GateError("full benchmark attestation benchmark_id must be non-empty")

    review = payload.get("review")
    if not isinstance(review, dict):
        raise GateError("full benchmark attestation review must be an object")
    if review.get("status") != "reviewed":
        raise GateError("full benchmark attestation review.status must equal 'reviewed'")
    for field in ("attested_by", "review_batch", "reviewed_at"):
        if not str(review.get(field) or "").strip():
            raise GateError(f"full benchmark attestation review.{field} must be non-empty")
    if review.get("manual_review_required_count") != 0:
        raise GateError("full benchmark attestation must report zero rows requiring manual review")

    assets = payload.get("assets")
    if not isinstance(assets, dict):
        raise GateError("full benchmark attestation assets must be an object")
    expected_paths = {"seed": seed_path, "historical_results": results_path}
    expected_hashes: dict[str, str] = {}
    for asset_name, actual_path in expected_paths.items():
        asset = assets.get(asset_name)
        if not isinstance(asset, dict):
            raise GateError(f"full benchmark attestation assets.{asset_name} must be an object")
        attested_asset_path = _attested_path(asset.get("path"), label=f"assets.{asset_name}.path")
        if _normalized_path(attested_asset_path) != _normalized_path(actual_path):
            raise GateError(
                f"full benchmark attestation {asset_name} path mismatch: "
                f"expected {attested_asset_path}, got {actual_path}"
            )
        if asset.get("rows") != 50:
            raise GateError(f"full benchmark attestation assets.{asset_name}.rows must equal 50")
        expected_hashes[asset_name] = _attested_sha256(
            asset.get("sha256"),
            label=f"assets.{asset_name}.sha256",
        )

    source = payload.get("source")
    if not isinstance(source, dict):
        raise GateError("full benchmark attestation source must be an object")
    if not str(source.get("origin_repository") or "").strip():
        raise GateError("full benchmark attestation source.origin_repository must be non-empty")
    canonical_results = _attested_path(
        source.get("canonical_results_artifact"),
        label="source.canonical_results_artifact",
    )
    if _normalized_path(canonical_results) != _normalized_path(results_path):
        raise GateError(
            "full benchmark attestation canonical results path does not match results_path"
        )
    provenance = source.get("provenance_document")
    if not isinstance(provenance, dict):
        raise GateError("full benchmark attestation source.provenance_document must be an object")
    provenance_path = _attested_path(
        provenance.get("path"),
        label="source.provenance_document.path",
    )
    if not provenance_path.is_file():
        raise GateError(f"full benchmark provenance document is missing: {provenance_path}")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", str(provenance.get("git_commit") or "").strip()):
        raise GateError(
            "full benchmark attestation provenance git_commit must be a 40-character hex id"
        )
    expected_provenance_hash = _attested_sha256(
        provenance.get("normalized_sha256"),
        label="source.provenance_document.normalized_sha256",
    )
    actual_provenance_hash = _normalized_text_sha256(provenance_path)
    if actual_provenance_hash != expected_provenance_hash:
        raise GateError(
            "provenance document normalized SHA-256 mismatch: "
            f"expected {expected_provenance_hash}, got {actual_provenance_hash}"
        )

    evidence = payload.get("evidence")
    if not isinstance(evidence, dict):
        raise GateError("full benchmark attestation evidence must be an object")
    restoration = evidence.get("restoration_report")
    if not isinstance(restoration, dict):
        raise GateError("full benchmark attestation evidence.restoration_report must be an object")
    restoration_path = _attested_path(
        restoration.get("path"),
        label="evidence.restoration_report.path",
    )
    if not restoration_path.is_file():
        raise GateError(f"full benchmark restoration report is missing: {restoration_path}")
    expected_restoration_hash = _attested_sha256(
        restoration.get("sha256"),
        label="evidence.restoration_report.sha256",
    )
    actual_restoration_hash = _sha256(restoration_path)
    if actual_restoration_hash != expected_restoration_hash:
        raise GateError(
            "restoration report SHA-256 mismatch: "
            f"expected {expected_restoration_hash}, got {actual_restoration_hash}"
        )
    restoration_report = _read_json_object(
        restoration_path, label="full benchmark restoration report"
    )
    if restoration_report.get("row_count") != 50:
        raise GateError("full benchmark restoration report row_count must equal 50")
    if restoration_report.get("manual_review_required_count") != 0:
        raise GateError(
            "full benchmark restoration report must have zero manual-review-required rows"
        )
    if restoration_report.get("must_abstain_count") != 5:
        raise GateError("full benchmark restoration report must_abstain_count must equal 5")
    if restoration_report.get("question_type_distribution") != {
        "comparison": 15,
        "fact": 20,
        "inductive": 15,
    }:
        raise GateError("full benchmark restoration report question-type distribution is invalid")

    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "benchmark_id": str(payload["benchmark_id"]),
        "benchmark_profile": "historical-full-raw",
        "review_status": "reviewed",
        "attested_by": str(review["attested_by"]),
        "review_batch": str(review["review_batch"]),
        "reviewed_at": str(review["reviewed_at"]),
        "seed_sha256": expected_hashes["seed"],
        "results_sha256": expected_hashes["historical_results"],
        "restoration_report_path": str(restoration_path),
        "restoration_report_sha256": actual_restoration_hash,
        "provenance_document_path": str(provenance_path),
        "provenance_document_normalized_sha256": actual_provenance_hash,
        "provenance_git_commit": str(provenance["git_commit"]),
    }


def check_full_seed_assets(
    seed_path: Path,
    results_path: Path,
    *,
    expected_seed_sha256: str | None = None,
    expected_results_sha256: str | None = None,
    attestation_path: Path | None = None,
) -> dict[str, Any]:
    """Validate the private full benchmark without weakening its tests.

    Both assets are required because ``test_full_seed_restore`` defines identity
    against the historical result question ids. A missing asset is a gate failure,
    never a skip or pass.
    """

    seed_path = _repo_path(seed_path)
    results_path = _repo_path(results_path)
    missing = [str(path) for path in (seed_path, results_path) if not path.is_file()]
    if missing:
        raise GateError(
            "missing external full-benchmark asset(s): "
            + ", ".join(missing)
            + ". Restore the reviewed 50-row seed and its historical results; "
            "data/eval_set/answer_eval_seed_current_full.jsonl is not a substitute."
        )

    attestation: dict[str, Any] | None = None
    if attestation_path is not None:
        attestation = _load_full_benchmark_attestation(
            attestation_path,
            seed_path=seed_path,
            results_path=results_path,
        )
        attested_hashes = {
            "seed": attestation["seed_sha256"],
            "results": attestation["results_sha256"],
        }
        for label, explicit in (
            ("seed", expected_seed_sha256),
            ("results", expected_results_sha256),
        ):
            if explicit is None:
                continue
            explicit_hash = _attested_sha256(explicit, label=f"explicit {label} SHA-256")
            if explicit_hash != attested_hashes[label]:
                raise GateError(
                    f"explicit {label} SHA-256 disagrees with the review attestation: "
                    f"{explicit_hash} != {attested_hashes[label]}"
                )
        expected_seed_sha256 = attested_hashes["seed"]
        expected_results_sha256 = attested_hashes["results"]
    else:
        for label, expected in (
            ("seed", expected_seed_sha256),
            ("results", expected_results_sha256),
        ):
            if not expected or not re.fullmatch(r"[0-9a-fA-F]{64}", str(expected).strip()):
                raise GateError(
                    f"expected {label} SHA-256 is required and must be a 64-character hex digest"
                )

    seed_hash = _sha256(seed_path)
    results_hash = _sha256(results_path)
    for label, actual, expected in (
        ("seed", seed_hash, expected_seed_sha256),
        ("results", results_hash, expected_results_sha256),
    ):
        if expected and actual.lower() != expected.strip().lower():
            raise GateError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")

    seed_rows = _read_jsonl(seed_path)
    result_rows = _read_jsonl(results_path)
    if len(seed_rows) != 50:
        raise GateError(f"full seed must contain exactly 50 rows, got {len(seed_rows)}")
    seed_ids = [str(row.get("question_id", "")) for row in seed_rows]
    result_ids = [str(row.get("question_id", "")) for row in result_rows]
    if any(not value for value in seed_ids) or len(seed_ids) != len(set(seed_ids)):
        raise GateError("full seed question_id values must be non-empty and unique")
    if len(result_rows) != 50:
        raise GateError(f"historical results must contain exactly 50 rows, got {len(result_rows)}")
    if any(not value for value in result_ids) or len(result_ids) != len(set(result_ids)):
        raise GateError("historical result question_id values must be non-empty and unique")
    if set(seed_ids) != set(result_ids):
        raise GateError("full seed question_ids do not match the historical results asset")

    distribution = Counter(str(row.get("question_type", "")) for row in seed_rows)
    expected_distribution = Counter({"fact": 20, "comparison": 15, "inductive": 15})
    if distribution != expected_distribution:
        raise GateError(f"unexpected question_type distribution: {dict(distribution)}")
    abstain_ids = sorted(str(row["question_id"]) for row in seed_rows if row.get("must_abstain"))
    expected_abstain_ids = [
        "compare_abstain_01",
        "compare_abstain_02",
        "fact_abstain_01",
        "fact_abstain_02",
        "inductive_abstain_01",
    ]
    if abstain_ids != expected_abstain_ids:
        raise GateError(f"unexpected must_abstain ids: {abstain_ids}")
    if any(bool(row.get("manual_review_required", False)) for row in seed_rows):
        raise GateError("full seed contains rows that still require manual review")
    if attestation is None:
        review_fields = (
            "review_status",
            "reviewer_id",
            "review_batch",
            "reviewed_at",
            "source_manifest_sha256",
            "benchmark_profile",
        )
        for asset_name, rows in (("full seed", seed_rows), ("historical results", result_rows)):
            for index, row in enumerate(rows, start=1):
                missing_review = [
                    field for field in review_fields if not str(row.get(field) or "").strip()
                ]
                if missing_review:
                    raise GateError(
                        f"{asset_name} row {index} missing review fields: {', '.join(missing_review)}"
                    )
                if row.get("review_status") != "reviewed":
                    raise GateError(f"{asset_name} row {index} review_status must equal 'reviewed'")
                if row.get("benchmark_profile") != "historical-full-raw":
                    raise GateError(
                        f"{asset_name} row {index} benchmark_profile must equal 'historical-full-raw'"
                    )
                manifest_hash = str(row.get("source_manifest_sha256") or "").strip()
                if not re.fullmatch(r"[0-9a-fA-F]{64}", manifest_hash):
                    raise GateError(f"{asset_name} row {index} source_manifest_sha256 is invalid")
        seed_manifest_hashes = {str(row["source_manifest_sha256"]).lower() for row in seed_rows}
        result_manifest_hashes = {str(row["source_manifest_sha256"]).lower() for row in result_rows}
        if len(seed_manifest_hashes) != 1 or seed_manifest_hashes != result_manifest_hashes:
            raise GateError("seed/results source_manifest_sha256 identities do not match")

    return {
        "status": "ready",
        "seed": {"path": str(seed_path), "sha256": seed_hash, "rows": len(seed_rows)},
        "historical_results": {
            "path": str(results_path),
            "sha256": results_hash,
            "rows": len(result_rows),
        },
        "question_type_distribution": dict(sorted(distribution.items())),
        "must_abstain_ids": abstain_ids,
        "review_contract_source": "attestation" if attestation is not None else "inline",
        "attestation": attestation,
    }


def _readiness_asset(
    path: Path,
    *,
    kind: str,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Return a serializable asset record without hiding read/parse failures."""

    resolved = _repo_path(path)
    record: dict[str, Any] = {
        "path": str(resolved),
        "sha256": None,
        "rows": None,
        "row_count": None,
        "line_count": None,
        "availability": "missing",
        "status": "BLOCKED",
        "blocking_reason": "",
    }
    if not resolved.is_file():
        record["blocking_reason"] = "missing asset"
        return record
    try:
        record["sha256"] = _sha256(resolved)
        record["availability"] = "present"
        if not expected_sha256 or not re.fullmatch(
            r"[0-9a-fA-F]{64}", str(expected_sha256).strip()
        ):
            record["blocking_reason"] = f"expected {kind} SHA-256 is required"
            return record
        if record["sha256"].lower() != str(expected_sha256).strip().lower():
            record["blocking_reason"] = (
                f"{kind} SHA-256 mismatch: expected {expected_sha256}, got {record['sha256']}"
            )
            return record
        rows = _read_jsonl(resolved)
        record["rows"] = len(rows)
        record["row_count"] = len(rows)
        record["line_count"] = len(rows)
        if kind == "semantic_labels":
            if not rows:
                record["blocking_reason"] = "semantic-label asset contains no labels"
            else:
                # Keep direct ``python scripts/phase0_gate.py`` execution
                # dependency-light; the repository import is needed only for
                # the semantic contract branch.
                from src.evaluation.semantic_calibration import validate_semantic_labels

                validate_semantic_labels(rows)
                record["status"] = "READY"
        else:
            record["status"] = "READY"
    except (GateError, ValueError) as exc:
        record["blocking_reason"] = str(exc)
    except OSError as exc:
        record["blocking_reason"] = f"cannot read asset: {exc}"
    return record


def build_readiness_report(
    *,
    seed_path: Path = DEFAULT_FULL_SEED,
    results_path: Path = DEFAULT_FULL_RESULTS,
    attestation_path: Path | None = None,
    labels_path: Path = DEFAULT_SEMANTIC_LABELS,
    output_path: Path = DEFAULT_OUTPUT_DIR / "readiness.json",
    expected_seed_sha256: str | None = None,
    expected_results_sha256: str | None = None,
    expected_labels_sha256: str | None = None,
) -> dict[str, Any]:
    """Build the strict three-asset benchmark readiness report.

    Full-seed validation is intentionally all-or-nothing.  A current/dev seed
    is never consulted as a fallback, and the report is written even when the
    result is blocked so an audit trail exists for missing external assets.
    """

    assets = {
        "full_seed": _readiness_asset(
            seed_path,
            kind="full_seed",
            expected_sha256=expected_seed_sha256,
        ),
        "historical_results": _readiness_asset(
            results_path,
            kind="historical_results",
            expected_sha256=expected_results_sha256,
        ),
        "semantic_labels": _readiness_asset(
            labels_path,
            kind="semantic_labels",
            expected_sha256=expected_labels_sha256,
        ),
    }
    full_validation: dict[str, Any] | None = None
    full_reason = ""
    try:
        full_validation = check_full_seed_assets(
            seed_path,
            results_path,
            expected_seed_sha256=expected_seed_sha256,
            expected_results_sha256=expected_results_sha256,
            attestation_path=attestation_path,
        )
    except GateError as exc:
        full_reason = str(exc)

    if full_validation is not None:
        assets["full_seed"].update(
            {
                "status": "READY",
                "sha256": full_validation["seed"]["sha256"],
                "rows": full_validation["seed"]["rows"],
                "row_count": full_validation["seed"]["rows"],
                "line_count": full_validation["seed"]["rows"],
                "blocking_reason": "",
            }
        )
        assets["historical_results"].update(
            {
                "status": "READY",
                "sha256": full_validation["historical_results"]["sha256"],
                "rows": full_validation["historical_results"]["rows"],
                "row_count": full_validation["historical_results"]["rows"],
                "line_count": full_validation["historical_results"]["rows"],
                "blocking_reason": "",
            }
        )
    else:
        for key in ("full_seed", "historical_results"):
            assets[key]["status"] = "BLOCKED"
            if not assets[key]["blocking_reason"]:
                assets[key]["blocking_reason"] = full_reason or "full benchmark validation failed"

    semantic_calibration: dict[str, Any] = {
        "threshold": 0.85,
        "sample_count": 0,
        "calibrated": False,
        "reason": "reviewed semantic-label asset is unavailable",
    }
    if assets["semantic_labels"]["status"] == "READY":
        try:
            from src.evaluation.semantic_calibration import calibrate_reviewed_semantic_labels

            semantic_calibration = calibrate_reviewed_semantic_labels(_repo_path(labels_path))
        except (OSError, ValueError) as exc:
            semantic_calibration["reason"] = str(exc)
    blocked = [key for key, item in assets.items() if item["status"] != "READY"]
    blocking_reasons = {key: assets[key]["blocking_reason"] for key in blocked}
    calibration_ready = (
        semantic_calibration.get("calibrated") is True
        and semantic_calibration.get("precision_constraint_satisfied") is True
        and semantic_calibration.get("coverage_constraint_satisfied") is True
    )
    if assets["semantic_labels"]["status"] == "READY" and not calibration_ready:
        blocking_reasons["semantic_calibration"] = str(
            semantic_calibration.get("reason")
            or ", ".join(semantic_calibration.get("coverage_blocking_reasons") or [])
            or "semantic calibration is not precision/coverage qualified"
        )
    overall_blocked = bool(blocked or not calibration_ready)
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "BLOCKED" if overall_blocked else "READY",
        "readiness_status": "BLOCKED" if overall_blocked else "READY",
        "blocked": overall_blocked,
        "profile": "historical-full-raw",
        "assets": assets,
        "blocked_assets": blocked,
        "blocking_reasons": blocking_reasons,
        "full_seed_validation": full_validation,
        "semantic_calibration": semantic_calibration,
        "substitution_policy": {
            "current_seed_allowed": False,
            "current_seed_path": str(
                _repo_path(Path("data/eval_set/answer_eval_seed_current_full.jsonl"))
            ),
        },
        "output_path": str(_repo_path(output_path)),
    }
    destination = _repo_path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _git(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _dependency_versions() -> dict[str, str]:
    names: set[str] = set()
    for file_name in ("requirements.txt", "requirements-dev.txt"):
        path = REPO_ROOT / file_name
        if not path.is_file():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line or line.startswith(("-", "http:", "https:", "git+")):
                continue
            name = line.split(";", 1)[0]
            for marker in ("[", "<", ">", "=", "!", "~"):
                name = name.split(marker, 1)[0]
            if name.strip():
                names.add(name.strip())
    versions: dict[str, str] = {}
    for name in sorted(names, key=str.lower):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _hash_configs() -> dict[str, str]:
    paths = [
        REPO_ROOT / "requirements.txt",
        REPO_ROOT / "requirements-dev.txt",
        REPO_ROOT / "requirements.lock",
    ]
    paths.extend(
        sorted((REPO_ROOT / "configs").rglob("*")) if (REPO_ROOT / "configs").exists() else []
    )
    return {str(path.relative_to(REPO_ROOT)): _sha256(path) for path in paths if path.is_file()}


def collect_baseline(output_path: Path, *, include_full_seed: bool = False) -> dict[str, Any]:
    env_config = {key: os.environ[key] for key in SAFE_CONFIG_ENV_KEYS if os.environ.get(key)}
    model_config = dict(MODEL_DEFAULTS)
    model_config.update(
        {
            "llm_provider": os.environ.get("LLM_PROVIDER")
            or os.environ.get("DEEPSEEK_PROVIDER")
            or MODEL_DEFAULTS["llm_provider"],
            "llm_model": os.environ.get("LLM_MODEL")
            or os.environ.get("DEEPSEEK_MODEL")
            or MODEL_DEFAULTS["llm_model"],
            "runtime_profile": os.environ.get("RUNTIME_PROFILE")
            or MODEL_DEFAULTS["runtime_profile"],
        }
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": {
            "root": str(REPO_ROOT),
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")),
        },
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "dependencies": _dependency_versions(),
        "configuration_hashes": _hash_configs(),
        "model_configuration": model_config,
        "configuration_environment": env_config,
        "commands": {
            "deterministic_smoke": shlex.join(SMOKE_COMMAND),
            "readiness": shlex.join([sys.executable, "scripts/phase0_gate.py", "readiness"]),
            "historical_full_evidence": shlex.join(
                [
                    sys.executable,
                    "scripts/historical_full_evidence_gate.py",
                    "--expected-chunks-sha256",
                    "<authorized-stage6-chunks-sha256>",
                ]
            ),
            "evidence_integrity": shlex.join(
                [
                    sys.executable,
                    "scripts/evidence_integrity_gate.py",
                    "--trace-path",
                    "<trace.jsonl>",
                    "--chunks-path",
                    "<chunks.jsonl>",
                ]
            ),
            "unit_without_private_full_seed": shlex.join(
                [sys.executable, "-m", "pytest", "-q", "--ignore=tests/test_full_seed_restore.py"]
            ),
            "full_suite_with_external_assets": shlex.join([sys.executable, "-m", "pytest", "-q"]),
        },
        "output_paths": {
            "baseline_report": str(_repo_path(output_path)),
            "generated_root": str(REPO_ROOT / DEFAULT_OUTPUT_DIR),
        },
        "smoke_groups": {name: list(paths) for name, paths in SMOKE_GROUPS.items()},
        "external_full_seed": {
            "status": "not_checked",
            "check_command": shlex.join(
                [sys.executable, "scripts/phase0_gate.py", "check-full-seed"]
            ),
        },
    }
    if include_full_seed:
        report["external_full_seed"] = check_full_seed_assets(
            DEFAULT_FULL_SEED,
            DEFAULT_FULL_RESULTS,
            expected_seed_sha256=os.environ.get("FULL_SEED_SHA256"),
            expected_results_sha256=os.environ.get("FULL_RESULTS_SHA256"),
            attestation_path=DEFAULT_FULL_ATTESTATION,
        )

    destination = _repo_path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def run_smoke(extra_pytest_args: Sequence[str] = ()) -> int:
    command = [*SMOKE_COMMAND, *extra_pytest_args]
    print("Phase 0 deterministic smoke groups:")
    for name, paths in SMOKE_GROUPS.items():
        print(f"  {name}: {', '.join(paths)}")
    print("command=" + shlex.join(command))
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 0 reproducibility and asset gates.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="Run the deterministic offline P0 smoke suite.")
    smoke.add_argument(
        "pytest_args", nargs=argparse.REMAINDER, help="Extra arguments forwarded to pytest."
    )

    baseline = subparsers.add_parser(
        "baseline", help="Write a reproducibility report under ignored outputs/."
    )
    baseline.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "baseline.json")
    baseline.add_argument(
        "--include-full-seed",
        action="store_true",
        help="Require and record the private full assets.",
    )

    assets = subparsers.add_parser(
        "check-full-seed", help="Strictly validate the external full benchmark assets."
    )
    assets.add_argument("--seed-path", type=Path, default=DEFAULT_FULL_SEED)
    assets.add_argument("--results-path", type=Path, default=DEFAULT_FULL_RESULTS)
    assets.add_argument("--attestation-path", type=Path, default=DEFAULT_FULL_ATTESTATION)
    assets.add_argument("--expected-seed-sha256", default=os.environ.get("FULL_SEED_SHA256"))
    assets.add_argument("--expected-results-sha256", default=os.environ.get("FULL_RESULTS_SHA256"))

    readiness = subparsers.add_parser(
        "readiness",
        help="Write the strict full benchmark and reviewed semantic-label readiness report.",
    )
    readiness.add_argument("--seed-path", type=Path, default=DEFAULT_FULL_SEED)
    readiness.add_argument("--results-path", type=Path, default=DEFAULT_FULL_RESULTS)
    readiness.add_argument("--attestation-path", type=Path, default=DEFAULT_FULL_ATTESTATION)
    readiness.add_argument("--labels-path", type=Path, default=DEFAULT_SEMANTIC_LABELS)
    readiness.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "readiness.json")
    readiness.add_argument("--expected-seed-sha256", default=os.environ.get("FULL_SEED_SHA256"))
    readiness.add_argument(
        "--expected-results-sha256", default=os.environ.get("FULL_RESULTS_SHA256")
    )
    readiness.add_argument(
        "--expected-labels-sha256", default=os.environ.get("SEMANTIC_LABELS_SHA256")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "smoke":
            forwarded = list(args.pytest_args)
            if forwarded[:1] == ["--"]:
                forwarded = forwarded[1:]
            return run_smoke(forwarded)
        if args.command == "baseline":
            report = collect_baseline(args.output, include_full_seed=args.include_full_seed)
            print(f"baseline_report={report['output_paths']['baseline_report']}")
            return 0
        if args.command == "readiness":
            report = build_readiness_report(
                seed_path=args.seed_path,
                results_path=args.results_path,
                attestation_path=args.attestation_path,
                labels_path=args.labels_path,
                output_path=args.output,
                expected_seed_sha256=args.expected_seed_sha256,
                expected_results_sha256=args.expected_results_sha256,
                expected_labels_sha256=args.expected_labels_sha256,
            )
            print(json.dumps(report, ensure_ascii=False, indent=2))
            if report["status"] != "READY":
                print(
                    "PHASE0_READINESS_BLOCKED: required assets are missing or invalid",
                    file=sys.stderr,
                )
                return 2
            return 0
        report = check_full_seed_assets(
            args.seed_path,
            args.results_path,
            expected_seed_sha256=args.expected_seed_sha256,
            expected_results_sha256=args.expected_results_sha256,
            attestation_path=args.attestation_path,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except GateError as exc:
        print(f"PHASE0_GATE_FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
