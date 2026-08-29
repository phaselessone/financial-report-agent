"""Read, write and validate strict evaluation run bundles."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

from src.evaluation.run_identity import RunIdentity, canonical_sha256, file_sha256, hash_case_ids
from src.utils.io import ensure_dir, read_json, read_jsonl, write_json, write_jsonl


EVAL_BUNDLE_SCHEMA_VERSION = "eval-bundle-v1"
CORPUS_ASSET_MANIFEST_SCHEMA_VERSION = "corpus-assets-v1"
CORPUS_ASSET_NAMES = ("chunks", "structured_facts", "company_aliases")
EVAL_BUNDLE_FILES = (
    "config.json",
    "metrics.json",
    "per_case.jsonl",
    "trajectories.jsonl",
    "metadata.json",
)
_RESERVED_METADATA_FIELDS = {
    "schema_version",
    "run_id",
    "identity_hash",
    "run_identity",
    "artifact_hashes",
    "comparison_config_hash",
    "integrity_verdict",
}


class EvalBundleIntegrityError(ValueError):
    pass


@dataclass(frozen=True)
class EvalBundle:
    path: Path
    identity: RunIdentity
    config: dict[str, Any]
    metrics: dict[str, Any]
    per_case: list[dict[str, Any]]
    trajectories: list[dict[str, Any]]
    metadata: dict[str, Any]


def _corpus_manifest_contract(
    asset_hashes: Mapping[str, str],
) -> dict[str, Any]:
    hashes = {name: str(asset_hashes.get(name) or "").strip() for name in CORPUS_ASSET_NAMES}
    missing = [name for name, digest in hashes.items() if not digest]
    if missing:
        raise EvalBundleIntegrityError(
            "corpus asset manifest is missing hashes: " + ", ".join(missing)
        )
    corpus_hash = canonical_sha256(hashes)
    contract = {
        "schema_version": CORPUS_ASSET_MANIFEST_SCHEMA_VERSION,
        "assets": hashes,
        "corpus_hash": corpus_hash,
    }
    return {**contract, "manifest_hash": canonical_sha256(contract)}


def build_corpus_asset_manifest(
    *,
    chunks_path: Path,
    facts_path: Path,
    aliases_path: Path,
) -> dict[str, Any]:
    """Build the explicit three-asset contract used by formal eval runners."""

    paths = {
        "chunks": Path(chunks_path).resolve(),
        "structured_facts": Path(facts_path).resolve(),
        "company_aliases": Path(aliases_path).resolve(),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "corpus assets do not exist: " + ", ".join(f"{name}={paths[name]}" for name in missing)
        )
    contract = _corpus_manifest_contract({name: file_sha256(path) for name, path in paths.items()})
    return {
        **contract,
        "paths": {name: str(path) for name, path in paths.items()},
    }


def validate_corpus_asset_manifest(
    identity: RunIdentity,
    manifest: Mapping[str, Any],
    *,
    verify_files: bool = True,
) -> dict[str, Any]:
    """Validate manifest structure, identity binding and optionally source bytes."""

    if not isinstance(manifest, Mapping):
        raise EvalBundleIntegrityError("corpus asset manifest must be an object")
    if manifest.get("schema_version") != CORPUS_ASSET_MANIFEST_SCHEMA_VERSION:
        raise EvalBundleIntegrityError("unsupported corpus asset manifest schema_version")
    raw_assets = manifest.get("assets")
    if not isinstance(raw_assets, Mapping) or set(raw_assets) != set(CORPUS_ASSET_NAMES):
        raise EvalBundleIntegrityError(
            "corpus asset manifest must declare exactly: " + ", ".join(CORPUS_ASSET_NAMES)
        )
    contract = _corpus_manifest_contract(
        {name: str(raw_assets.get(name) or "") for name in CORPUS_ASSET_NAMES}
    )
    if manifest.get("corpus_hash") != contract["corpus_hash"]:
        raise EvalBundleIntegrityError("corpus asset manifest corpus_hash is invalid")
    if manifest.get("manifest_hash") != contract["manifest_hash"]:
        raise EvalBundleIntegrityError("corpus asset manifest manifest_hash is invalid")
    if identity.corpus_hash != contract["corpus_hash"]:
        raise EvalBundleIntegrityError(
            "RunIdentity corpus_hash does not match corpus asset manifest"
        )

    raw_paths = manifest.get("paths")
    if not isinstance(raw_paths, Mapping) or set(raw_paths) != set(CORPUS_ASSET_NAMES):
        raise EvalBundleIntegrityError(
            "corpus asset manifest paths must declare exactly: " + ", ".join(CORPUS_ASSET_NAMES)
        )
    paths = {name: str(raw_paths.get(name) or "").strip() for name in CORPUS_ASSET_NAMES}
    if any(not value for value in paths.values()):
        raise EvalBundleIntegrityError("corpus asset manifest paths must be non-empty")
    if verify_files:
        for name in CORPUS_ASSET_NAMES:
            path = Path(paths[name])
            if not path.is_file():
                raise EvalBundleIntegrityError(f"corpus asset is missing: {name}={path}")
            if file_sha256(path) != contract["assets"][name]:
                raise EvalBundleIntegrityError(f"corpus asset hash mismatch: {name}")
    return {**contract, "paths": paths}


def _metadata_with_corpus_manifest(
    identity: RunIdentity,
    metadata: Mapping[str, Any] | None,
    corpus_asset_manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    value = dict(metadata or {})
    if corpus_asset_manifest is None:
        return value
    if "corpus_asset_manifest" in value:
        raise EvalBundleIntegrityError(
            "metadata corpus_asset_manifest must be supplied through the explicit writer parameter"
        )
    value["corpus_asset_manifest"] = validate_corpus_asset_manifest(
        identity, corpus_asset_manifest, verify_files=True
    )
    return value


def _row_case_id(row: Mapping[str, Any]) -> str:
    case_id = str(row.get("case_id") or row.get("question_id") or "").strip()
    if not case_id:
        raise EvalBundleIntegrityError("per_case row requires case_id or question_id")
    return case_id


def _validate_per_case(identity: RunIdentity, per_case: Sequence[Mapping[str, Any]]) -> None:
    if len(per_case) != identity.case_count:
        raise EvalBundleIntegrityError(
            f"case_count mismatch: identity={identity.case_count}, per_case={len(per_case)}"
        )
    try:
        actual_hash = hash_case_ids(_row_case_id(row) for row in per_case)
    except ValueError as exc:
        raise EvalBundleIntegrityError(str(exc)) from exc
    if actual_hash != identity.case_ids_hash:
        raise EvalBundleIntegrityError(
            f"case_ids_hash mismatch: identity={identity.case_ids_hash}, per_case={actual_hash}"
        )


def _trajectory_case_ids(trajectories: Sequence[Mapping[str, Any]]) -> list[str]:
    case_ids: list[str] = []
    for row in trajectories:
        case_id = str(row.get("case_id") or row.get("question_id") or "").strip()
        if not case_id:
            raise EvalBundleIntegrityError("trajectory row requires case_id or question_id")
        if case_id in case_ids:
            raise EvalBundleIntegrityError(f"duplicate trajectory case id: {case_id}")
        case_ids.append(case_id)
    return case_ids


def _validate_trajectories(
    identity: RunIdentity,
    trajectories: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> None:
    if not _is_agentic_bundle(identity, metadata):
        return
    if len(trajectories) != identity.case_count:
        raise EvalBundleIntegrityError(
            f"trajectory case_count mismatch: identity={identity.case_count}, trajectories={len(trajectories)}"
        )
    trajectory_hash = hash_case_ids(_trajectory_case_ids(trajectories))
    if trajectory_hash != identity.case_ids_hash:
        raise EvalBundleIntegrityError(
            "trajectory case_ids_hash mismatch: "
            f"identity={identity.case_ids_hash}, trajectories={trajectory_hash}"
        )


def _is_agentic_bundle(identity: RunIdentity, metadata: Mapping[str, Any]) -> bool:
    mode = str(metadata.get("mode") or "").strip().lower()
    return mode == "agentic" or _pipeline_treatment(identity) == "agentic"


def _pipeline_treatment(identity: RunIdentity) -> str:
    flags = identity.to_dict()["feature_flags"]
    treatments = flags.get("treatments", {}) if isinstance(flags, Mapping) else {}
    return str(treatments.get("pipeline") or "").strip().lower()


def _validate_mode_treatment(identity: RunIdentity, metadata: Mapping[str, Any]) -> None:
    mode = str(metadata.get("mode") or "").strip().lower()
    pipeline = _pipeline_treatment(identity)
    if mode in {"baseline", "agentic"} and pipeline in {"baseline", "agentic"} and mode != pipeline:
        raise EvalBundleIntegrityError(
            f"metadata mode mismatch with run identity treatment: mode={mode}, pipeline={pipeline}"
        )


def _validate_bundle_config(identity: RunIdentity, config: Mapping[str, Any]) -> None:
    if (
        "benchmark_profile" in config
        and config.get("benchmark_profile") != identity.benchmark_profile
    ):
        raise EvalBundleIntegrityError(
            "config benchmark_profile mismatch: "
            f"identity={identity.benchmark_profile}, config={config.get('benchmark_profile')}"
        )
    identity_treatments = identity.to_dict()["feature_flags"].get("treatments", {})
    if "treatments" in config and config.get("treatments") != identity_treatments:
        raise EvalBundleIntegrityError("config treatments mismatch with run identity")


def _comparison_config(config: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(config)
    value.pop("treatments", None)
    return value


def _artifact_hashes(
    *,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    per_case: Sequence[Mapping[str, Any]],
    trajectories: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    return {
        "config_sha256": canonical_sha256(config),
        "metrics_sha256": canonical_sha256(metrics),
        "per_case_sha256": canonical_sha256(per_case),
        "trajectories_sha256": canonical_sha256(trajectories),
    }


def _prepare_bundle_payload(
    *,
    identity: RunIdentity,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    per_case: Iterable[Mapping[str, Any]],
    trajectories: Iterable[Mapping[str, Any]],
    metadata: Mapping[str, Any] | None,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    config_row = dict(config)
    metric_row = dict(metrics)
    case_rows = [dict(row) for row in per_case]
    trajectory_rows = [dict(row) for row in trajectories]
    extra_metadata = dict(metadata or {})
    _validate_per_case(identity, case_rows)
    _validate_mode_treatment(identity, extra_metadata)
    _validate_trajectories(identity, trajectory_rows, extra_metadata)
    _validate_bundle_config(identity, config_row)

    reserved = sorted(_RESERVED_METADATA_FIELDS.intersection(extra_metadata))
    if reserved:
        raise EvalBundleIntegrityError(f"metadata uses reserved fields: {', '.join(reserved)}")
    metadata_row = {
        **extra_metadata,
        "schema_version": EVAL_BUNDLE_SCHEMA_VERSION,
        "run_id": identity.run_id,
        "identity_hash": identity.identity_hash,
        "run_identity": identity.to_dict(),
        "artifact_hashes": _artifact_hashes(
            config=config_row,
            metrics=metric_row,
            per_case=case_rows,
            trajectories=trajectory_rows,
        ),
        "comparison_config_hash": canonical_sha256(_comparison_config(config_row)),
    }
    return config_row, metric_row, case_rows, trajectory_rows, metadata_row


def _write_bundle_files(
    bundle_path: Path,
    *,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    per_case: Sequence[Mapping[str, Any]],
    trajectories: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> None:
    ensure_dir(bundle_path)
    write_json(bundle_path / "config.json", config)
    write_json(bundle_path / "metrics.json", metrics)
    write_jsonl(bundle_path / "per_case.jsonl", per_case)
    write_jsonl(bundle_path / "trajectories.jsonl", trajectories)
    write_json(bundle_path / "metadata.json", metadata)


def _stage_eval_bundle(
    eval_root: Path,
    *,
    identity: RunIdentity,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    per_case: Sequence[Mapping[str, Any]],
    trajectories: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    overwrite: bool,
) -> tuple[EvalBundle, Path, Path]:
    root = Path(eval_root)
    bundle_path = root / identity.run_id
    if bundle_path.exists() and not overwrite:
        raise FileExistsError(f"evaluation bundle already exists: {bundle_path}")
    ensure_dir(root)
    staging_root = root / f".{identity.run_id}.staging-{uuid4().hex}"
    staging_path = staging_root / identity.run_id
    try:
        _write_bundle_files(
            staging_path,
            config=config,
            metrics=metrics,
            per_case=per_case,
            trajectories=trajectories,
            metadata=metadata,
        )
        staged = read_eval_bundle(staging_path)
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    return staged, staging_root, bundle_path


def _commit_staged_bundle(
    staged: EvalBundle,
    *,
    staging_root: Path,
    bundle_path: Path,
    overwrite: bool,
) -> EvalBundle:
    backup_path: Path | None = None
    try:
        if bundle_path.exists():
            if not overwrite:
                raise FileExistsError(f"evaluation bundle already exists: {bundle_path}")
            backup_path = bundle_path.with_name(f".{bundle_path.name}.backup-{uuid4().hex}")
            os.replace(bundle_path, backup_path)
        try:
            os.replace(staged.path, bundle_path)
        except BaseException:
            if backup_path is not None and backup_path.exists() and not bundle_path.exists():
                os.replace(backup_path, bundle_path)
            raise
        if backup_path is not None:
            shutil.rmtree(backup_path, ignore_errors=True)
        return read_eval_bundle(bundle_path)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def write_eval_bundle(
    eval_root: Path,
    *,
    identity: RunIdentity,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    per_case: Iterable[Mapping[str, Any]],
    trajectories: Iterable[Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
    corpus_asset_manifest: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> EvalBundle:
    """Atomically publish ``outputs/eval/<run_id>`` after a staged round trip."""

    config_row, metric_row, case_rows, trajectory_rows, metadata_row = _prepare_bundle_payload(
        identity=identity,
        config=config,
        metrics=metrics,
        per_case=per_case,
        trajectories=trajectories,
        metadata=_metadata_with_corpus_manifest(identity, metadata, corpus_asset_manifest),
    )
    staged, staging_root, bundle_path = _stage_eval_bundle(
        eval_root,
        identity=identity,
        config=config_row,
        metrics=metric_row,
        per_case=case_rows,
        trajectories=trajectory_rows,
        metadata=metadata_row,
        overwrite=overwrite,
    )
    return _commit_staged_bundle(
        staged,
        staging_root=staging_root,
        bundle_path=bundle_path,
        overwrite=overwrite,
    )


def read_eval_bundle(path: Path) -> EvalBundle:
    bundle_path = Path(path)
    missing = [name for name in EVAL_BUNDLE_FILES if not (bundle_path / name).is_file()]
    if missing:
        raise EvalBundleIntegrityError(f"evaluation bundle missing files: {', '.join(missing)}")

    config = read_json(bundle_path / "config.json")
    metrics = read_json(bundle_path / "metrics.json")
    per_case = read_jsonl(bundle_path / "per_case.jsonl")
    trajectories = read_jsonl(bundle_path / "trajectories.jsonl")
    metadata = read_json(bundle_path / "metadata.json")
    if (
        not isinstance(config, dict)
        or not isinstance(metrics, dict)
        or not isinstance(metadata, dict)
    ):
        raise EvalBundleIntegrityError("config, metrics and metadata must be JSON objects")
    if metadata.get("schema_version") != EVAL_BUNDLE_SCHEMA_VERSION:
        raise EvalBundleIntegrityError("unsupported or missing eval bundle schema_version")
    try:
        identity = RunIdentity.from_dict(metadata.get("run_identity") or {})
    except (TypeError, ValueError) as exc:
        raise EvalBundleIntegrityError(f"invalid run_identity: {exc}") from exc
    if metadata.get("identity_hash") != identity.identity_hash:
        raise EvalBundleIntegrityError("identity_hash does not match run_identity")
    if metadata.get("run_id") != identity.run_id or bundle_path.name != identity.run_id:
        raise EvalBundleIntegrityError("run_id does not match run_identity or bundle directory")
    corpus_asset_manifest = metadata.get("corpus_asset_manifest")
    if corpus_asset_manifest is not None:
        validate_corpus_asset_manifest(identity, corpus_asset_manifest, verify_files=False)

    _validate_per_case(identity, per_case)
    _validate_mode_treatment(identity, metadata)
    _validate_trajectories(identity, trajectories, metadata)
    _validate_bundle_config(identity, config)
    actual_hashes = _artifact_hashes(
        config=config,
        metrics=metrics,
        per_case=per_case,
        trajectories=trajectories,
    )
    if metadata.get("artifact_hashes") != actual_hashes:
        raise EvalBundleIntegrityError("artifact hashes do not match bundle contents")
    comparison_config_hash = canonical_sha256(_comparison_config(config))
    if metadata.get("comparison_config_hash") != comparison_config_hash:
        raise EvalBundleIntegrityError("comparison config hash does not match config.json")

    return EvalBundle(
        path=bundle_path,
        identity=identity,
        config=config,
        metrics=metrics,
        per_case=per_case,
        trajectories=trajectories,
        metadata=metadata,
    )


def write_eval_bundle_reference(path: Path, bundle: EvalBundle) -> Path:
    reference_path = Path(path)
    bundle_path = os.path.relpath(bundle.path.resolve(), start=reference_path.parent.resolve())
    write_json(
        reference_path,
        {
            "schema_version": EVAL_BUNDLE_SCHEMA_VERSION,
            "run_id": bundle.identity.run_id,
            "bundle_path": bundle_path,
        },
    )
    return reference_path


def read_eval_bundle_reference(path: Path) -> EvalBundle:
    reference_path = Path(path)
    reference = read_json(reference_path)
    if not isinstance(reference, dict):
        raise EvalBundleIntegrityError("eval bundle reference must be a JSON object")
    raw_bundle_path = str(reference.get("bundle_path") or "").strip()
    if not raw_bundle_path:
        raise EvalBundleIntegrityError("eval bundle reference is missing bundle_path")
    bundle_path = Path(raw_bundle_path)
    if not bundle_path.is_absolute():
        bundle_path = reference_path.parent / bundle_path
    bundle_path = bundle_path.resolve()
    bundle = read_eval_bundle(bundle_path)
    if reference.get("run_id") != bundle.identity.run_id:
        raise EvalBundleIntegrityError("reference run_id does not match evaluation bundle")
    return bundle


def require_ready_eval_bundle_integrity(bundle: EvalBundle) -> None:
    """Reject agentic bundles that do not carry a successful integrity verdict."""

    if not _is_agentic_bundle(bundle.identity, bundle.metadata):
        return
    verdict = bundle.metadata.get("integrity_verdict")
    if not isinstance(verdict, Mapping) or verdict.get("status") != "READY":
        raise EvalBundleIntegrityError(
            "agentic evaluation bundle requires a READY integrity verdict"
        )
    expected = {
        "run_id": bundle.identity.run_id,
        "identity_hash": bundle.identity.identity_hash,
        "case_ids_hash": bundle.identity.case_ids_hash,
        "artifact_hashes": bundle.metadata.get("artifact_hashes"),
    }
    mismatches = [key for key, value in expected.items() if verdict.get(key) != value]
    if mismatches:
        raise EvalBundleIntegrityError(
            "integrity verdict does not match evaluation bundle: " + ", ".join(sorted(mismatches))
        )


def write_eval_bundle_integrity_verdict(
    bundle: EvalBundle,
    report: Mapping[str, Any],
) -> EvalBundle:
    """Persist a READY integrity result bound to this bundle's hashed artifacts."""

    if report.get("status") != "READY":
        raise EvalBundleIntegrityError("cannot record a non-READY integrity verdict")
    report_bundle = report.get("bundle")
    if not isinstance(report_bundle, Mapping):
        raise EvalBundleIntegrityError("integrity report is missing bundle identity")
    if (
        report_bundle.get("run_id") != bundle.identity.run_id
        or report_bundle.get("identity_hash") != bundle.identity.identity_hash
    ):
        raise EvalBundleIntegrityError("integrity report does not match evaluation bundle")

    metadata = dict(bundle.metadata)
    metadata["integrity_verdict"] = {
        "status": "READY",
        "run_id": bundle.identity.run_id,
        "identity_hash": bundle.identity.identity_hash,
        "case_ids_hash": bundle.identity.case_ids_hash,
        "artifact_hashes": dict(metadata.get("artifact_hashes") or {}),
        "summary": dict(report.get("summary") or {}),
    }
    write_json(bundle.path / "metadata.json", metadata)
    return read_eval_bundle(bundle.path)


def write_validated_eval_bundle(
    eval_root: Path,
    *,
    identity: RunIdentity,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    per_case: Iterable[Mapping[str, Any]],
    trajectories: Iterable[Mapping[str, Any]],
    chunks_path: Path,
    metadata: Mapping[str, Any],
    corpus_asset_manifest: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> EvalBundle:
    """Write a strict bundle and persist a READY verdict for agentic runs.

    Evidence validation is intentionally part of the write seam: callers
    cannot obtain an apparently-complete agentic bundle without validating the
    exact trajectories, RunIdentity and corpus chunk asset that it contains.
    """

    config_row, metric_row, case_rows, trajectory_rows, metadata_row = _prepare_bundle_payload(
        identity=identity,
        config=config,
        metrics=metrics,
        per_case=per_case,
        trajectories=trajectories,
        metadata=_metadata_with_corpus_manifest(identity, metadata, corpus_asset_manifest),
    )
    bundle, staging_root, bundle_path = _stage_eval_bundle(
        eval_root,
        identity=identity,
        config=config_row,
        metrics=metric_row,
        per_case=case_rows,
        trajectories=trajectory_rows,
        metadata=metadata_row,
        overwrite=overwrite,
    )
    try:
        if _is_agentic_bundle(bundle.identity, bundle.metadata):
            from src.evaluation.evidence_integrity import validate_bundle_evidence_integrity

            report = validate_bundle_evidence_integrity(bundle, chunks_path)
            if report.get("status") != "READY":
                issue_count = int((report.get("summary") or {}).get("issue_count", 0) or 0)
                raise EvalBundleIntegrityError(
                    f"agentic bundle evidence integrity BLOCKED with {issue_count} issue(s)"
                )
            bundle = write_eval_bundle_integrity_verdict(bundle, report)
        return _commit_staged_bundle(
            bundle,
            staging_root=staging_root,
            bundle_path=bundle_path,
            overwrite=overwrite,
        )
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise


def require_comparable_eval_bundles(bundles: Sequence[EvalBundle]) -> None:
    """Fail before a comparison or aggregate consumes incompatible bundles."""

    if len(bundles) < 2:
        raise ValueError("at least two eval bundles are required")
    reference = bundles[0]
    reference_config_hash = reference.metadata.get("comparison_config_hash")
    for bundle in bundles[1:]:
        reference.identity.assert_comparable(bundle.identity)
        if bundle.metadata.get("comparison_config_hash") != reference_config_hash:
            raise EvalBundleIntegrityError("config identity mismatch between evaluation bundles")
    for bundle in bundles:
        require_ready_eval_bundle_integrity(bundle)


__all__ = [
    "CORPUS_ASSET_MANIFEST_SCHEMA_VERSION",
    "CORPUS_ASSET_NAMES",
    "EVAL_BUNDLE_FILES",
    "EVAL_BUNDLE_SCHEMA_VERSION",
    "EvalBundle",
    "EvalBundleIntegrityError",
    "build_corpus_asset_manifest",
    "read_eval_bundle",
    "read_eval_bundle_reference",
    "require_comparable_eval_bundles",
    "require_ready_eval_bundle_integrity",
    "validate_corpus_asset_manifest",
    "write_eval_bundle",
    "write_eval_bundle_integrity_verdict",
    "write_eval_bundle_reference",
    "write_validated_eval_bundle",
]
