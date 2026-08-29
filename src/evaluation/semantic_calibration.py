"""Reviewed-label validation and deterministic semantic threshold calibration.

The semantic scorer is optional at runtime. This module only calibrates a
threshold when labels carry an auditable review contract; it never turns an
unreviewed or synthetic fixture into a quality claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


DEFAULT_SEMANTIC_THRESHOLD = 0.85
DEFAULT_SEMANTIC_THRESHOLDS = tuple(round(index / 100, 2) for index in range(50, 100, 5))
DEFAULT_MIN_SEMANTIC_LABELS = 20
DEFAULT_MIN_ENTAILED_LABELS = 5
DEFAULT_MIN_NON_ENTAILED_LABELS = 5
REQUIRED_SEMANTIC_LABEL_FIELDS = frozenset(
    {
        "sample_id",
        "claim_id",
        "evidence_id",
        "score",
        "label",
        "source_ref",
        "review_status",
        "reviewer_id",
        "reviewed_at",
        "scorer_kind",
        "scorer_model",
        "scorer_revision",
        "scorer_config_sha256",
    }
)
SEMANTIC_LABELS = frozenset({"ENTAILED", "CONTRADICTED", "INSUFFICIENT"})


class SemanticLabelValidationError(ValueError):
    """Raised when a semantic-label asset is not a reviewed contract fixture."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(str(error) for error in errors)
        super().__init__("; ".join(self.errors) or "invalid semantic labels")


SemanticLabelError = SemanticLabelValidationError
SemanticLabelSchemaError = SemanticLabelValidationError


def _is_synthetic(item: Mapping[str, Any]) -> bool:
    for key in ("synthetic", "is_synthetic"):
        value = item.get(key)
        if value is True or (isinstance(value, str) and value.strip().lower() in {"1", "true", "yes"}):
            return True
    for key in (
        "sample_type",
        "fixture_type",
        "dataset",
        "source_type",
        "provenance_type",
        "sample_id",
        "claim_id",
        "evidence_id",
        "source_ref",
    ):
        value = str(item.get(key) or "").strip().lower()
        if "synthetic" in value:
            return True
    return False


def _finite_score(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) and 0.0 <= score <= 1.0 else None


def validate_semantic_labels(samples: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate and normalize reviewed semantic labels."""

    errors: list[str] = []
    normalized: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    scorer_identities: set[tuple[str, str, str, str]] = set()
    try:
        raw_samples = list(samples)
    except TypeError as exc:
        raise SemanticLabelValidationError([f"labels must be iterable: {exc}"]) from exc

    for index, raw in enumerate(raw_samples, start=1):
        prefix = f"row {index}"
        if not isinstance(raw, Mapping):
            errors.append(f"{prefix}: expected an object")
            continue
        item = dict(raw)
        missing = sorted(REQUIRED_SEMANTIC_LABEL_FIELDS.difference(item))
        if missing:
            errors.append(f"{prefix}: missing required fields: {', '.join(missing)}")
            continue
        if _is_synthetic(item):
            errors.append(f"{prefix}: synthetic samples are not eligible")
        sample_id = str(item.get("sample_id") or "").strip()
        claim_id = str(item.get("claim_id") or "").strip()
        evidence_id = str(item.get("evidence_id") or "").strip()
        source_ref = str(item.get("source_ref") or "").strip()
        reviewer_id = str(item.get("reviewer_id") or "").strip()
        reviewed_at = str(item.get("reviewed_at") or "").strip()
        scorer_kind = str(item.get("scorer_kind") or "").strip()
        scorer_model = str(item.get("scorer_model") or "").strip()
        scorer_revision = str(item.get("scorer_revision") or "").strip()
        scorer_config_sha256 = str(item.get("scorer_config_sha256") or "").strip().lower()
        if not sample_id:
            errors.append(f"{prefix}: sample_id must be non-empty")
        elif sample_id in seen_sample_ids:
            errors.append(f"{prefix}: duplicate sample_id {sample_id!r}")
        else:
            seen_sample_ids.add(sample_id)
        if not claim_id:
            errors.append(f"{prefix}: claim_id must be non-empty")
        if not evidence_id:
            errors.append(f"{prefix}: evidence_id must be non-empty")
        if not source_ref:
            errors.append(f"{prefix}: source_ref must be non-empty")
        if not reviewer_id:
            errors.append(f"{prefix}: reviewer_id must be non-empty")
        if not reviewed_at:
            errors.append(f"{prefix}: reviewed_at must be non-empty")
        if scorer_kind != "directional_nli":
            errors.append(f"{prefix}: scorer_kind must equal 'directional_nli'")
        if not scorer_model:
            errors.append(f"{prefix}: scorer_model must be non-empty")
        if not scorer_revision:
            errors.append(f"{prefix}: scorer_revision must be non-empty")
        if not re.fullmatch(r"[0-9a-f]{64}", scorer_config_sha256):
            errors.append(f"{prefix}: scorer_config_sha256 must be a SHA-256 hex digest")
        scorer_identities.add((scorer_kind, scorer_model, scorer_revision, scorer_config_sha256))
        if item.get("label") not in SEMANTIC_LABELS:
            errors.append(f"{prefix}: label must be one of {sorted(SEMANTIC_LABELS)}")
        score = _finite_score(item.get("score"))
        if score is None:
            errors.append(f"{prefix}: score must be a finite number in [0, 1]")
        if item.get("review_status") != "reviewed":
            errors.append(f"{prefix}: review_status must equal 'reviewed'")
        item.update(
            {
                "sample_id": sample_id,
                "claim_id": claim_id,
                "evidence_id": evidence_id,
                "source_ref": source_ref,
                "reviewer_id": reviewer_id,
                "reviewed_at": reviewed_at,
                "scorer_kind": scorer_kind,
                "scorer_model": scorer_model,
                "scorer_revision": scorer_revision,
                "scorer_config_sha256": scorer_config_sha256,
            }
        )
        if score is not None:
            item["score"] = score
        normalized.append(item)

    if len(scorer_identities) > 1:
        errors.append("semantic labels must share one scorer identity")
    if errors:
        raise SemanticLabelValidationError(errors)
    return normalized


def load_reviewed_semantic_labels(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL semantic-label asset and enforce the reviewed contract."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"semantic-label asset does not exist: {source}")
    rows: list[dict[str, Any]] = []
    try:
        with source.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SemanticLabelValidationError([f"line {line_number}: invalid JSON: {exc.msg}"]) from exc
                if not isinstance(value, Mapping):
                    raise SemanticLabelValidationError([f"line {line_number}: expected a JSON object"])
                rows.append(dict(value))
    except OSError as exc:
        raise SemanticLabelValidationError([f"cannot read {source}: {exc}"]) from exc
    return validate_semantic_labels(rows)


load_semantic_labels = load_reviewed_semantic_labels
load_reviewed_labels = load_reviewed_semantic_labels
validate_reviewed_semantic_labels = validate_semantic_labels
validate_reviewed_labels = validate_semantic_labels
validate_label_rows = validate_semantic_labels


def _label(item: Mapping[str, Any]) -> int:
    raw = item.get("label", item.get("entailed", item.get("supported", False)))
    if isinstance(raw, str):
        return int(raw.strip().upper() in {"1", "TRUE", "YES", "ENTAILED", "SUPPORTED", "POSITIVE"})
    return int(bool(raw))


def threshold_metrics(samples: Iterable[Mapping[str, Any]], threshold: float) -> dict[str, float | int]:
    """Compute confusion counts and precision/recall/F1 at a score threshold."""

    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    tp = fp = tn = fn = 0
    for sample in samples:
        score = float(sample.get("score", 0.0) or 0.0)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("sample score must be a finite number in [0, 1]")
        pred = score >= threshold
        actual = bool(_label(sample))
        if pred and actual:
            tp += 1
        elif pred:
            fp += 1
        elif actual:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "threshold": float(threshold),
        "count": tp + fp + tn + fn,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2,
    }


def calibrate_semantic_threshold(
    samples: Sequence[Mapping[str, Any]],
    *,
    thresholds: Sequence[float] | None = None,
    objective: str = "f1",
    min_precision: float | None = 0.95,
    default: float = DEFAULT_SEMANTIC_THRESHOLD,
) -> dict[str, Any]:
    """Choose a precision-first threshold from semantic-score samples."""

    if not 0.0 <= float(default) <= 1.0:
        raise ValueError("default must be between 0 and 1")
    if not samples:
        return {
            "threshold": float(default),
            "objective": objective,
            "sample_count": 0,
            "calibrated": False,
            "metrics": threshold_metrics([], default),
            "candidates": [],
            "precision_constraint_satisfied": False if min_precision is not None else None,
        }
    if thresholds is None:
        thresholds = DEFAULT_SEMANTIC_THRESHOLDS
    candidates = [float(value) for value in thresholds]
    if not candidates or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in candidates):
        raise ValueError("thresholds must contain finite values between 0 and 1")
    key = objective.lower().strip()
    if key not in {"f1", "precision", "recall", "balanced_accuracy"}:
        raise ValueError("objective must be f1, precision, recall, or balanced_accuracy")
    if min_precision is not None and not 0.0 <= float(min_precision) <= 1.0:
        raise ValueError("min_precision must be between 0 and 1")
    scored = [threshold_metrics(samples, value) for value in candidates]
    eligible = [item for item in scored if min_precision is None or float(item["precision"]) >= float(min_precision)]
    if min_precision is not None and not eligible:
        return {
            "threshold": float(default),
            "objective": key,
            "sample_count": len(samples),
            "calibrated": False,
            "metrics": threshold_metrics(samples, default),
            "candidates": scored,
            "precision_constraint_satisfied": False,
        }
    pool = eligible or scored
    best = max(pool, key=lambda item: (float(item[key]), float(item["threshold"])))
    return {
        "threshold": float(best["threshold"]),
        "objective": key,
        "sample_count": len(samples),
        "calibrated": True,
        "metrics": best,
        "candidates": scored,
        "precision_constraint_satisfied": bool(eligible) if min_precision is not None else None,
    }


def calibrate_reviewed_semantic_labels(
    labels: Sequence[Mapping[str, Any]] | str | Path,
    *,
    min_samples: int = DEFAULT_MIN_SEMANTIC_LABELS,
    min_entailed: int = DEFAULT_MIN_ENTAILED_LABELS,
    min_non_entailed: int = DEFAULT_MIN_NON_ENTAILED_LABELS,
    **kwargs: Any,
) -> dict[str, Any]:
    """Validate reviewed labels and return a precision-first calibration report."""

    source_path = Path(labels) if isinstance(labels, (str, Path)) else None
    rows = load_reviewed_semantic_labels(source_path) if source_path is not None else validate_semantic_labels(labels)
    report = calibrate_semantic_threshold(rows, **kwargs)
    entailed_count = sum(1 for row in rows if row["label"] == "ENTAILED")
    non_entailed_count = len(rows) - entailed_count
    coverage_blocking_reasons: list[str] = []
    if len(rows) < int(min_samples):
        coverage_blocking_reasons.append("minimum_sample_count")
    if entailed_count < int(min_entailed):
        coverage_blocking_reasons.append("entailed_class_coverage")
    if non_entailed_count < int(min_non_entailed):
        coverage_blocking_reasons.append("non_entailed_class_coverage")
    coverage_satisfied = not coverage_blocking_reasons
    report["coverage_constraint_satisfied"] = coverage_satisfied
    report["coverage_blocking_reasons"] = coverage_blocking_reasons
    report["label_coverage"] = {
        "sample_count": len(rows),
        "entailed_count": entailed_count,
        "non_entailed_count": non_entailed_count,
        "minimum_sample_count": int(min_samples),
        "minimum_entailed_count": int(min_entailed),
        "minimum_non_entailed_count": int(min_non_entailed),
    }
    if not coverage_satisfied:
        report["calibrated"] = False
    if source_path is not None:
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        report["dataset_path"] = str(source_path)
        report["dataset_sha256"] = digest
        report["labels_path"] = str(source_path)
        report["labels_sha256"] = digest
    report["calibration_config"] = {
        "candidate_thresholds": list(kwargs.get("thresholds") or DEFAULT_SEMANTIC_THRESHOLDS),
        "objective": str(kwargs.get("objective", "f1")),
        "min_precision": kwargs.get("min_precision", 0.95),
        "default": kwargs.get("default", DEFAULT_SEMANTIC_THRESHOLD),
        "minimum_sample_count": int(min_samples),
        "minimum_entailed_count": int(min_entailed),
        "minimum_non_entailed_count": int(min_non_entailed),
    }
    if rows:
        first = rows[0]
        report["scorer_kind"] = first["scorer_kind"]
        report["scorer_identity"] = {
            "kind": first["scorer_kind"],
            "model": first["scorer_model"],
            "revision": first["scorer_revision"],
            "config_sha256": first["scorer_config_sha256"],
        }
    report["label_contract"] = {
        "required_fields": sorted(REQUIRED_SEMANTIC_LABEL_FIELDS),
        "allowed_labels": sorted(SEMANTIC_LABELS),
        "review_status": "reviewed",
        "score_semantics": "entailed_confidence",
    }
    # The single scalar score is calibrated as ENTAILED-vs-rest confidence.
    # It does not prove precision for a separate CONTRADICTED decision, so the
    # runtime activation wrapper may only change state to ENTAILED.
    report["score_semantics"] = "entailed_confidence"
    report["calibrated_statuses"] = ["ENTAILED"] if report.get("calibrated") else []
    report["contradiction_state_change_allowed"] = False
    return report


__all__ = [
    "DEFAULT_SEMANTIC_THRESHOLD",
    "DEFAULT_SEMANTIC_THRESHOLDS",
    "DEFAULT_MIN_SEMANTIC_LABELS",
    "DEFAULT_MIN_ENTAILED_LABELS",
    "DEFAULT_MIN_NON_ENTAILED_LABELS",
    "REQUIRED_SEMANTIC_LABEL_FIELDS",
    "SEMANTIC_LABELS",
    "SemanticLabelError",
    "SemanticLabelSchemaError",
    "SemanticLabelValidationError",
    "calibrate_reviewed_semantic_labels",
    "calibrate_semantic_threshold",
    "load_reviewed_semantic_labels",
    "load_reviewed_labels",
    "load_semantic_labels",
    "threshold_metrics",
    "validate_semantic_labels",
    "validate_reviewed_semantic_labels",
    "validate_reviewed_labels",
    "validate_label_rows",
]
