import math
import json

import pytest

from src.evaluation.semantic_calibration import (
    SemanticLabelValidationError,
    calibrate_semantic_threshold,
    calibrate_reviewed_semantic_labels,
    load_reviewed_semantic_labels,
    threshold_metrics,
    validate_semantic_labels,
)


def test_threshold_metrics_and_calibration():
    samples = [
        {"score": 0.95, "label": 1},
        {"score": 0.90, "label": 1},
        {"score": 0.60, "label": 0},
        {"score": 0.20, "label": 0},
    ]
    metrics = threshold_metrics(samples, 0.85)
    assert metrics["tp"] == 2
    assert metrics["fp"] == 0
    result = calibrate_semantic_threshold(samples, thresholds=[0.5, 0.85, 0.95])
    assert result["calibrated"] is True
    assert result["threshold"] == 0.85


def test_empty_calibration_returns_default():
    result = calibrate_semantic_threshold([], default=0.88)
    assert result["threshold"] == 0.88
    assert result["calibrated"] is False


def _reviewed(sample_id: str = "s1", **overrides):
    row = {
        "sample_id": sample_id,
        "claim_id": "c1",
        "evidence_id": "e1",
        "score": 0.9,
        "label": "ENTAILED",
        "source_ref": "review://batch-1/1",
        "review_status": "reviewed",
        "reviewer_id": "reviewer-1",
        "reviewed_at": "2026-08-24T00:00:00Z",
        "scorer_kind": "directional_nli",
        "scorer_model": "fixture-nli",
        "scorer_revision": "fixture-v1",
        "scorer_config_sha256": "b" * 64,
    }
    row.update(overrides)
    return row


def test_reviewed_label_contract_accepts_three_state_rows():
    rows = validate_semantic_labels(
        [
            _reviewed(),
            _reviewed(
                "s2",
                claim_id="c2",
                evidence_id="e2",
                score=0.2,
                label="CONTRADICTED",
            ),
            _reviewed(
                "s3",
                claim_id="c3",
                evidence_id="e3",
                score=0.4,
                label="INSUFFICIENT",
            ),
        ]
    )
    assert [row["label"] for row in rows] == ["ENTAILED", "CONTRADICTED", "INSUFFICIENT"]
    assert rows[0]["score"] == 0.9


@pytest.mark.parametrize(
    "override, message",
    [
        ({"score": math.nan}, "finite"),
        ({"score": 1.1}, r"\[0, 1\]"),
        ({"label": "UNKNOWN"}, "label"),
        ({"review_status": "pending"}, "review_status"),
        ({"source_ref": ""}, "source_ref"),
        ({"synthetic": True}, "synthetic"),
    ],
)
def test_reviewed_label_contract_rejects_invalid_rows(override, message):
    with pytest.raises(SemanticLabelValidationError, match=message):
        validate_semantic_labels([_reviewed(**override)])


def test_reviewed_label_contract_rejects_duplicate_sample_ids():
    with pytest.raises(SemanticLabelValidationError, match="duplicate sample_id"):
        validate_semantic_labels([_reviewed(), _reviewed(claim_id="c2", evidence_id="e2")])


def test_reviewed_label_loader_reads_jsonl(tmp_path):
    path = tmp_path / "labels.jsonl"
    path.write_text(json.dumps(_reviewed()) + "\n", encoding="utf-8")
    assert load_reviewed_semantic_labels(path)[0]["sample_id"] == "s1"


def test_calibration_report_records_label_provenance(tmp_path):
    path = tmp_path / "labels.jsonl"
    rows = [
        _reviewed(f"positive-{index}", claim_id=f"cp{index}", evidence_id=f"ep{index}", score=0.95, label="ENTAILED")
        for index in range(10)
    ] + [
        _reviewed(f"negative-{index}", claim_id=f"cn{index}", evidence_id=f"en{index}", score=0.10, label="CONTRADICTED")
        for index in range(10)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report = calibrate_reviewed_semantic_labels(path)
    assert report["dataset_sha256"]
    assert report["calibration_config"]["min_precision"] == 0.95
    assert report["calibrated"] is True
    assert report["coverage_constraint_satisfied"] is True
    assert report["scorer_kind"] == "directional_nli"
    assert report["scorer_identity"] == {
        "kind": "directional_nli",
        "model": "fixture-nli",
        "revision": "fixture-v1",
        "config_sha256": "b" * 64,
    }
    assert report["calibrated_statuses"] == ["ENTAILED"]
    assert report["contradiction_state_change_allowed"] is False
    assert report["score_semantics"] == "entailed_confidence"


def test_reviewed_calibration_refuses_too_few_or_single_class_labels() -> None:
    too_few = calibrate_reviewed_semantic_labels([_reviewed()])
    assert too_few["calibrated"] is False
    assert too_few["coverage_constraint_satisfied"] is False
    assert "minimum_sample_count" in too_few["coverage_blocking_reasons"]
    assert "non_entailed_class_coverage" in too_few["coverage_blocking_reasons"]


def test_precision_first_calibration_tie_breaks_to_higher_threshold():
    result = calibrate_semantic_threshold(
        [
            _reviewed("s1", score=0.9, label="ENTAILED"),
            _reviewed("s2", score=0.4, label="CONTRADICTED"),
        ],
        thresholds=[0.50, 0.55],
    )
    assert result["calibrated"] is True
    assert result["threshold"] == 0.55


def test_precision_first_calibration_blocks_when_no_candidate_meets_constraint():
    result = calibrate_semantic_threshold(
        [
            _reviewed("s1", score=0.9, label="CONTRADICTED"),
            _reviewed("s2", score=0.8, label="CONTRADICTED"),
        ],
        thresholds=[0.50, 0.55],
        default=0.85,
    )
    assert result["threshold"] == 0.85
    assert result["calibrated"] is False
    assert result["precision_constraint_satisfied"] is False
