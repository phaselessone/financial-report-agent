from __future__ import annotations

import pytest

from src.agent.semantic_policy import (
    CalibratedSemanticScorer,
    SemanticActivationError,
    activate_semantic_scorer,
)

FULL_COMMIT_SHA = "a" * 40


def _safe_report(**overrides):
    report = {
        "calibrated": True,
        "precision_constraint_satisfied": True,
        "positive_support_constraint_satisfied": True,
        "coverage_constraint_satisfied": True,
        "threshold": 0.9,
        "metrics": {"precision": 1.0, "tp": 10, "fp": 0},
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
        "scorer_identity": {
            "kind": "directional_nli",
            "model": "fixture-nli",
            "revision": FULL_COMMIT_SHA,
            "config_sha256": "b" * 64,
        },
        "calibrated_statuses": ["ENTAILED"],
        "contradiction_state_change_allowed": False,
    }
    report.update(overrides)
    return report


@pytest.mark.parametrize(
    "overrides",
    [
        {"calibrated": False},
        {"precision_constraint_satisfied": False},
        {"positive_support_constraint_satisfied": False},
        {
            "label_contract": {
                "review_status": "synthetic_not_human_reviewed",
                "content_hash_binding": "sha256",
            }
        },
        {"labels_sha256": ""},
        {"scorer_kind": "embedding_similarity"},
        {"metrics": {"precision": 0.9, "tp": 9, "fp": 1}},
        {"metrics": {"precision": 1.0, "tp": 1, "fp": 0}},
        {"label_contract": {"review_status": "reviewed", "content_hash_binding": "none"}},
        {"calibrated_statuses": ["ENTAILED", "CONTRADICTED"]},
    ],
)
def test_semantic_scorer_activation_fails_closed_without_full_reviewed_contract(overrides) -> None:
    with pytest.raises(SemanticActivationError):
        activate_semantic_scorer(
            lambda _claim, _evidence: {},
            _safe_report(**overrides),
            scorer_identity=_safe_report()["scorer_identity"],
        )


def test_reviewed_precision_first_directional_scorer_can_be_activated() -> None:
    calls = []

    def scorer(claim: str, evidence: str):
        calls.append((claim, evidence))
        return {"status": "ENTAILED", "score": 0.96}

    activated = activate_semantic_scorer(
        scorer,
        _safe_report(),
        scorer_identity=_safe_report()["scorer_identity"],
    )

    assert isinstance(activated, CalibratedSemanticScorer)
    assert activated.threshold == 0.9
    assert activated("claim", "evidence") == {"status": "ENTAILED", "score": 0.96}
    assert calls == [("claim", "evidence")]


@pytest.mark.parametrize("revision", ["feature-v1", "A" * 40])
def test_semantic_scorer_activation_requires_full_commit_revision(revision: str) -> None:
    report = _safe_report()
    report["scorer_identity"] = {
        **report["scorer_identity"],
        "revision": revision,
    }

    with pytest.raises(SemanticActivationError, match="full Git commit SHA"):
        activate_semantic_scorer(
            lambda _claim, _evidence: {},
            report,
            scorer_identity=report["scorer_identity"],
        )


def test_semantic_scorer_activation_requires_hub_repo_id() -> None:
    report = _safe_report()
    report["scorer_identity"] = {
        **report["scorer_identity"],
        "model": "./fixture-nli",
    }

    with pytest.raises(SemanticActivationError, match="Hub repo ID"):
        activate_semantic_scorer(
            lambda _claim, _evidence: {},
            report,
            scorer_identity=report["scorer_identity"],
        )


def test_entailment_only_calibration_cannot_activate_semantic_contradiction() -> None:
    activated = activate_semantic_scorer(
        lambda _claim, _evidence: {"status": "CONTRADICTED", "score": 0.99},
        _safe_report(),
        scorer_identity=_safe_report()["scorer_identity"],
    )

    result = activated("claim", "evidence")
    assert result["status"] == "INSUFFICIENT"
    assert result["score"] == 0.0
    assert result["reason"] == "semantic_status_not_calibrated"
    assert activated.calibrated_statuses == ("ENTAILED",)


def test_semantic_scorer_activation_rejects_runtime_identity_mismatch() -> None:
    with pytest.raises(SemanticActivationError, match="identity"):
        activate_semantic_scorer(
            lambda _claim, _evidence: {},
            _safe_report(),
            scorer_identity={
                "kind": "directional_nli",
                "model": "different-model",
                "revision": FULL_COMMIT_SHA,
                "config_sha256": "b" * 64,
            },
        )
