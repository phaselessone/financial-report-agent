"""Fail-closed activation policy for optional semantic claim validation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import math
import re
from typing import Any


class SemanticActivationError(ValueError):
    """Raised when a scorer lacks an auditable reviewed calibration contract."""


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
    if calibration_report.get("coverage_constraint_satisfied") is not True:
        raise SemanticActivationError("semantic label coverage constraint is not satisfied")
    contract = calibration_report.get("label_contract")
    if not isinstance(contract, Mapping) or contract.get("review_status") != "reviewed":
        raise SemanticActivationError("semantic labels are not reviewed")
    scorer_kind = str(calibration_report.get("scorer_kind") or "").strip()
    if scorer_kind != "directional_nli":
        raise SemanticActivationError("only directional_nli scorers may change claim status")
    calibrated_identity = calibration_report.get("scorer_identity")
    if not isinstance(calibrated_identity, Mapping) or not isinstance(scorer_identity, Mapping):
        raise SemanticActivationError("semantic scorer identity is missing")
    identity_fields = ("kind", "model", "revision", "config_sha256")
    expected_identity = {field: str(calibrated_identity.get(field) or "").strip() for field in identity_fields}
    runtime_identity = {field: str(scorer_identity.get(field) or "").strip() for field in identity_fields}
    if any(not value for value in expected_identity.values()):
        raise SemanticActivationError("calibrated semantic scorer identity is incomplete")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_identity["config_sha256"]):
        raise SemanticActivationError("calibrated semantic scorer config hash is invalid")
    if expected_identity != runtime_identity or expected_identity["kind"] != scorer_kind:
        raise SemanticActivationError("runtime semantic scorer identity does not match calibration")
    labels_sha256 = str(
        calibration_report.get("labels_sha256")
        or calibration_report.get("dataset_sha256")
        or ""
    ).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", labels_sha256):
        raise SemanticActivationError("reviewed semantic label hash is missing or invalid")
    try:
        threshold = float(calibration_report.get("threshold"))
        metrics = calibration_report.get("metrics")
        precision = float(metrics.get("precision")) if isinstance(metrics, Mapping) else math.nan
        config = calibration_report.get("calibration_config")
        minimum = float(config.get("min_precision")) if isinstance(config, Mapping) else math.nan
    except (TypeError, ValueError) as exc:
        raise SemanticActivationError("calibration threshold/precision is malformed") from exc
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise SemanticActivationError("calibration threshold is outside [0, 1]")
    if not math.isfinite(precision) or not math.isfinite(minimum) or precision < minimum:
        raise SemanticActivationError("reported semantic precision is below the required minimum")
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
