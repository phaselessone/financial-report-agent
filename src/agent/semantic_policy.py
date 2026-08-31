"""Fail-closed activation policy for optional semantic claim validation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import math
import re
from typing import Any


_MUTABLE_REVISIONS = frozenset(
    {
        "current",
        "default",
        "dev",
        "development",
        "head",
        "latest",
        "main",
        "master",
        "release",
        "stable",
        "unversioned",
    }
)


class SemanticActivationError(ValueError):
    """Raised when a scorer lacks an auditable reviewed calibration contract."""


def _immutable_revision(value: str) -> bool:
    normalized = value.strip().casefold()
    return bool(
        normalized
        and normalized not in _MUTABLE_REVISIONS
        and not normalized.startswith("refs/heads/")
        and not normalized.startswith("heads/")
    )


def _non_negative_count(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SemanticActivationError(
            f"semantic calibration {field_name} must be a non-negative integer"
        )
    return value


@dataclass(frozen=True)
class CalibratedSemanticScorer:
    """Directional scorer plus the threshold proven by reviewed labels."""

    threshold: float
    labels_sha256: str
    scorer_kind: str
    scorer_model: str
    scorer_revision: str
    scorer_config_sha256: str
    calibrated_statuses: tuple[str, ...]
    _scorer: Callable[[str, str], Mapping[str, Any] | float] = field(repr=False)

    def __call__(self, claim_text: str, evidence_text: str) -> Mapping[str, Any] | float:
        result = self._scorer(claim_text, evidence_text)
        if isinstance(result, Mapping):
            status = str(result.get("status") or "").strip().upper()
            if status in {"ENTAILED", "CONTRADICTED"} and status not in self.calibrated_statuses:
                # The current reviewed contract calibrates an entailment
                # confidence score only.  A high-confidence contradiction has
                # different class semantics and must remain INSUFFICIENT until
                # a separate contradiction-precision contract exists.
                return {
                    "status": "INSUFFICIENT",
                    "score": 0.0,
                    "reason": "semantic_status_not_calibrated",
                }
        return result


def activate_semantic_scorer(
    scorer: Callable[[str, str], Mapping[str, Any] | float],
    calibration_report: Mapping[str, Any],
    *,
    scorer_identity: Mapping[str, Any],
) -> CalibratedSemanticScorer:
    """Activate only reviewed, precision-qualified directional NLI scorers."""
    if not callable(scorer):
        raise SemanticActivationError("semantic scorer must be callable")
    if not isinstance(calibration_report, Mapping):
        raise SemanticActivationError("calibration report must be a mapping")
    if calibration_report.get("calibrated") is not True:
        raise SemanticActivationError("semantic calibration is not complete")
    if calibration_report.get("precision_constraint_satisfied") is not True:
        raise SemanticActivationError("semantic precision constraint is not satisfied")
    if calibration_report.get("positive_support_constraint_satisfied") is not True:
        raise SemanticActivationError(
            "semantic predicted-positive/true-positive support constraint is not satisfied"
        )
    if calibration_report.get("coverage_constraint_satisfied") is not True:
        raise SemanticActivationError("semantic label coverage constraint is not satisfied")
    contract = calibration_report.get("label_contract")
    if (
        not isinstance(contract, Mapping)
        or contract.get("review_status") != "reviewed"
        or contract.get("content_hash_binding") != "sha256"
    ):
        raise SemanticActivationError("semantic labels are not reviewed")
    scorer_kind = str(calibration_report.get("scorer_kind") or "").strip()
    if scorer_kind != "directional_nli":
        raise SemanticActivationError("only directional_nli scorers may change claim status")
    calibrated_identity = calibration_report.get("scorer_identity")
    if not isinstance(calibrated_identity, Mapping) or not isinstance(scorer_identity, Mapping):
        raise SemanticActivationError("semantic scorer identity is missing")
    identity_fields = ("kind", "model", "revision", "config_sha256")
    expected_identity = {
        field: str(calibrated_identity.get(field) or "").strip() for field in identity_fields
    }
    runtime_identity = {
        field: str(scorer_identity.get(field) or "").strip() for field in identity_fields
    }
    if any(not value for value in expected_identity.values()):
        raise SemanticActivationError("calibrated semantic scorer identity is incomplete")
    if not _immutable_revision(expected_identity["revision"]):
        raise SemanticActivationError("calibrated semantic scorer revision is not immutable")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_identity["config_sha256"]):
        raise SemanticActivationError("calibrated semantic scorer config hash is invalid")
    if expected_identity != runtime_identity or expected_identity["kind"] != scorer_kind:
        raise SemanticActivationError("runtime semantic scorer identity does not match calibration")
    labels_sha256 = (
        str(
            calibration_report.get("labels_sha256")
            or calibration_report.get("dataset_sha256")
            or ""
        )
        .strip()
        .lower()
    )
    if not re.fullmatch(r"[0-9a-f]{64}", labels_sha256):
        raise SemanticActivationError("reviewed semantic label hash is missing or invalid")
    metrics = calibration_report.get("metrics")
    config = calibration_report.get("calibration_config")
    if not isinstance(metrics, Mapping) or not isinstance(config, Mapping):
        raise SemanticActivationError("calibration threshold/precision is malformed")
    try:
        threshold = float(calibration_report.get("threshold"))
        precision = float(metrics.get("precision"))
        minimum = float(config.get("min_precision"))
    except (TypeError, ValueError) as exc:
        raise SemanticActivationError("calibration threshold/precision is malformed") from exc
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise SemanticActivationError("calibration threshold is outside [0, 1]")
    if not math.isfinite(precision) or not math.isfinite(minimum) or precision < minimum:
        raise SemanticActivationError("reported semantic precision is below the required minimum")
    true_positives = _non_negative_count(metrics.get("tp"), field_name="metrics.tp")
    false_positives = _non_negative_count(metrics.get("fp"), field_name="metrics.fp")
    minimum_predicted = _non_negative_count(
        config.get("min_predicted_positives"),
        field_name="calibration_config.min_predicted_positives",
    )
    minimum_true = _non_negative_count(
        config.get("min_true_positives"),
        field_name="calibration_config.min_true_positives",
    )
    predicted_positives = true_positives + false_positives
    derived_precision = true_positives / predicted_positives if predicted_positives else 0.0
    if not math.isclose(precision, derived_precision, rel_tol=0.0, abs_tol=1e-12):
        raise SemanticActivationError("reported semantic precision does not match TP/FP counts")
    if predicted_positives < minimum_predicted or true_positives < minimum_true:
        raise SemanticActivationError(
            "reported semantic calibration lacks predicted-positive/true-positive support"
        )
    statuses = calibration_report.get("calibrated_statuses")
    if (
        statuses != ["ENTAILED"]
        or calibration_report.get("contradiction_state_change_allowed") is not False
    ):
        raise SemanticActivationError("semantic calibrated status contract is not fail-closed")
    return CalibratedSemanticScorer(
        threshold=threshold,
        labels_sha256=labels_sha256,
        scorer_kind=scorer_kind,
        scorer_model=expected_identity["model"],
        scorer_revision=expected_identity["revision"],
        scorer_config_sha256=expected_identity["config_sha256"].lower(),
        calibrated_statuses=("ENTAILED",),
        _scorer=scorer,
    )


__all__ = [
    "CalibratedSemanticScorer",
    "SemanticActivationError",
    "activate_semantic_scorer",
]
