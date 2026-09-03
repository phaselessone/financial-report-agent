import pytest

from src.evaluation.hard_case_benchmark import evaluate_case
from src.evaluation.failure_attribution import (
    FailureType,
    attribute_failure,
    build_failure_summary,
    normalize_failure_type,
)


def test_normalize_failure_aliases_are_stable():
    assert normalize_failure_type("max-llm-calls") == FailureType.BUDGET_EXHAUSTED.value
    assert normalize_failure_type("retrieval failed") == FailureType.RETRIEVAL_ERROR.value
    assert normalize_failure_type("not-known") == FailureType.UNKNOWN.value
    assert normalize_failure_type("TOOL_SELECTION_ERROR") == FailureType.TOOL_SELECTION_ERROR.value
    assert normalize_failure_type("structured miss") == FailureType.STRUCTURED_LOOKUP_MISS.value


def test_attribute_first_root_final_and_recovery():
    row = {
        "total_tokens": 12,
        "end_to_end_latency_ms": 44,
        "trajectory_events": [
            {"event_id": "e1", "dependencies": [], "recovery_of": [], "step": 1, "node": "retrieve", "status": "NO_NEW_EVIDENCE", "error_type": "no_evidence"},
            {"event_id": "e2", "dependencies": ["e1"], "recovery_of": ["e1"], "step": 2, "node": "retrieve", "status": "SUCCESS", "latency_ms": 3},
            {"event_id": "e3", "dependencies": ["e2"], "recovery_of": [], "step": 3, "node": "validate", "status": "FAILED", "error_type": "unsupported"},
        ],
    }
    result = attribute_failure(row)
    assert result["first_failure"] == "no_evidence"
    assert result["root_cause"] == "unsupported"
    assert result["final_failure"] == "unsupported"
    assert result["recovered"] is False
    assert result["cost"]["tokens"] == 12


def test_summary_counts_root_causes_and_events():
    result = build_failure_summary([
        {"failed": True, "error_type": "ValueError", "error_message": "x"},
        {"termination_reason": "max_llm_calls", "trajectory_events": []},
    ])
    assert result["failed_query_count"] == 2
    assert result["root_cause_counts"]["unknown"] == 1
    assert result["root_cause_counts"]["budget_exhausted"] == 1


def test_reviewed_gold_creates_verification_and_abstain_false_positive_negative_events():
    verification_fp = attribute_failure(
        {
            "claims": [
                {
                    "claim_id": "g-1",
                    "text": "unsupported conclusion",
                    "evidence_ids": [],
                    "verification": {"status": "ENTAILED"},
                }
            ],
            "abstained": True,
        },
        gold={
            "must_abstain": False,
            "gold_claims": [
                {
                    "claim_id": "g-1",
                    "text": "unsupported conclusion",
                    "status": "INSUFFICIENT",
                    "evidence_ids": [],
                }
            ],
        },
    )
    assert verification_fp["root_cause"] == "verification_false_positive"
    assert verification_fp["root_cause_resolution"] == "multiple_independent"
    assert verification_fp["downstream_symptoms"] == []
    assert {
        item["failure_type"]
        for item in verification_fp["independent_failures"]
    } == {"abstain_false_positive"}
    assert all(
        item["dependencies"] == []
        for item in verification_fp["failure_events"]
        if item["origin"] == "reviewed_gold"
    )

    abstain_fn = attribute_failure(
        {"claims": [], "abstained": False},
        gold={"must_abstain": True, "gold_claims": []},
    )
    assert abstain_fn["root_cause"] == "abstain_false_negative"


def test_reviewed_gold_creates_citation_event_when_claim_status_matches_but_evidence_differs():
    result = attribute_failure(
        {
            "claims": [
                {
                    "claim_id": "predicted-g-1",
                    "text": "Revenue was 100",
                    "evidence_ids": ["wrong-evidence"],
                    "verification": {"status": "ENTAILED"},
                }
            ]
        },
        gold={
            "gold_claims": [
                {
                    "claim_id": "g-1",
                    "text": "Revenue was 100",
                    "status": "ENTAILED",
                    "evidence_ids": ["reviewed-evidence"],
                }
            ]
        },
    )

    reviewed_events = [
        event
        for event in result["failure_events"]
        if event["origin"] == "reviewed_gold"
    ]
    assert len(reviewed_events) == 1
    assert reviewed_events[0]["failure_type"] == FailureType.CITATION_ERROR.value
    assert reviewed_events[0]["outcome_dimensions"] == ["claim.evidence"]
    assert reviewed_events[0]["gold_id"] == "g-1"
    assert reviewed_events[0]["predicted_id"] == "predicted-g-1"
    assert reviewed_events[0]["dependencies"] == []
    assert result["has_failure"] is True


def test_root_is_earliest_causal_failure_and_outcome_errors_are_downstream_symptoms():
    result = attribute_failure(
        {
            "trajectory_events": [
                {"event_id": "e3", "dependencies": ["e1"], "recovery_of": [], "step": 3, "node": "validate", "status": "FAILED", "error_type": "validation_failed"},
                {"event_id": "e1", "dependencies": [], "recovery_of": [], "step": 1, "node": "retrieve", "status": "FAILED", "error_type": "retrieval_miss"},
                {"event_id": "e4", "dependencies": ["e3"], "recovery_of": [], "step": 4, "node": "synthesize", "status": "FAILED", "error_type": "citation_error"},
            ],
            "claims": [
                {
                    "claim_id": "g-1",
                    "text": "Revenue was 100",
                    "evidence_ids": [],
                    "verification": {"status": "INSUFFICIENT"},
                }
            ],
            "abstained": False,
        },
        gold={
            "must_abstain": False,
            "gold_claims": [
                {
                    "claim_id": "g-1",
                    "text": "Revenue was 100",
                    "status": "ENTAILED",
                    "evidence_ids": ["e-1"],
                }
            ],
        },
    )

    assert result["first_failure"] == "retrieval_miss"
    assert result["root_cause"] == "retrieval_miss"
    assert result["root_cause_step"] == 1
    assert result["root_cause_resolution"] == "multiple_independent"
    assert result["downstream_symptoms"] == [
        "validation_failed",
        "citation_error",
    ]
    assert result["final_failure"] == "verification_false_negative"
    assert {
        item["failure_type"]
        for item in result["independent_failures"]
    } == {"verification_false_negative"}


def test_downstream_failures_remain_distinct_from_outcome_symptoms_and_summary_keeps_stage_counts():
    row = {
        "trajectory_events": [
            {"event_id": "e1", "dependencies": [], "recovery_of": [], "step": 1, "status": "FAILED", "error_type": "retrieval_miss", "tokens": 2, "latency_ms": 3},
            {"event_id": "e2", "dependencies": ["e1"], "recovery_of": [], "step": 2, "status": "FAILED", "error_type": "tool_execution_failed", "tokens": 5, "latency_ms": 7},
            {"event_id": "e3", "dependencies": ["e2"], "recovery_of": [], "step": 3, "status": "FAILED", "error_type": "citation_error", "tokens": 1, "latency_ms": 2},
        ]
    }

    attribution = attribute_failure(row)
    summary = build_failure_summary([row])

    assert attribution["downstream_failures"] == ["tool_execution_failed", "citation_error"]
    assert attribution["downstream_symptoms"] == ["citation_error"]
    assert attribution["cost_before_root_cause"] == {"tokens": 2, "latency_ms": 3.0, "retries": 0}
    assert summary["first_failure_counts"] == {"retrieval_miss": 1}
    assert summary["root_cause_counts"] == {"retrieval_miss": 1}
    assert summary["final_failure_counts"] == {"citation_error": 1}
    assert summary["downstream_symptom_counts"] == {"citation_error": 1}
    assert summary["total_cost_before_root_cause"]["tokens"] == 2


def test_root_is_earliest_unrecovered_causal_ancestor() -> None:
    result = attribute_failure(
        {
            "trajectory_events": [
                {
                    "event_id": "e1",
                    "dependencies": [],
                    "recovery_of": [],
                    "step": 1,
                    "node": "retrieve",
                    "status": "FAILED",
                    "error_type": "retrieval_miss",
                    "budget_usage": {"tokens": 2},
                    "latency_ms": 3,
                },
                {
                    "event_id": "e2",
                    "dependencies": ["e1"],
                    "recovery_of": ["e1"],
                    "step": 2,
                    "node": "retrieve",
                    "status": "SUCCESS",
                    "budget_usage": {"tokens": 5},
                    "latency_ms": 7,
                },
                {
                    "event_id": "e3",
                    "dependencies": ["e2"],
                    "recovery_of": [],
                    "step": 3,
                    "node": "tool",
                    "status": "FAILED",
                    "error_type": "tool_execution_failed",
                    "budget_usage": {"tokens": 7},
                    "latency_ms": 11,
                },
                {
                    "event_id": "e4",
                    "dependencies": ["e3"],
                    "recovery_of": [],
                    "step": 4,
                    "node": "finalize",
                    "status": "FAILED",
                    "error_type": "citation_error",
                    "budget_usage": {"tokens": 1},
                    "latency_ms": 2,
                },
            ],
            "budget_usage": {"tokens": 15, "retries": 2},
        }
    )

    assert result["first_failure"] == "retrieval_miss"
    assert result["first_failure_event_id"] == "e1"
    assert result["root_cause"] == "tool_execution_failed"
    assert result["root_cause_event_id"] == "e3"
    assert result["root_cause_resolution"] == "resolved"
    assert result["recovered_failure_event_ids"] == ["e1"]
    assert result["cost_before_first_failure"] == {"tokens": 2, "latency_ms": 3.0, "retries": 0}
    assert result["cost_before_root_cause"] == {"tokens": 14, "latency_ms": 21.0, "retries": 0}
    assert result["cost"] == {"tokens": 15, "latency_ms": 23.0, "retries": 2}


def test_failure_without_explicit_lineage_is_unresolved_instead_of_guessed() -> None:
    result = attribute_failure(
        {
            "trace_events": [
                {"step": 1, "node": "retrieve", "status": "FAILED", "error_type": "retrieval_miss"},
                {"step": 2, "node": "finalize", "status": "FAILED", "error_type": "citation_error"},
            ]
        }
    )

    assert result["first_failure"] == "retrieval_miss"
    assert result["root_cause"] == "unresolved"
    assert result["root_cause_resolution"] == "unresolved"
    assert result["root_cause_candidate"] == "retrieval_miss"


def test_derived_state_failure_does_not_depend_on_the_last_runtime_event() -> None:
    result = attribute_failure(
        {
            "step_count": 2,
            "trajectory_events": [
                {
                    "event_id": "runtime-retrieval",
                    "dependencies": [],
                    "recovery_of": [],
                    "step": 1,
                    "status": "FAILED",
                    "error_type": "retrieval_miss",
                }
            ],
            "support_validation": {"supported": False},
        }
    )

    derived = next(
        event
        for event in result["failure_events"]
        if event["origin"] == "derived_state"
    )
    assert derived["failure_type"] == "validation_failed"
    assert derived["dependencies"] == []
    assert result["root_cause"] == "retrieval_miss"
    assert result["root_cause_resolution"] == "multiple_independent"
    assert result["downstream_failures"] == []
    assert [item["failure_type"] for item in result["independent_failures"]] == [
        "validation_failed"
    ]


def _reviewed_calculation_case() -> dict:
    return {
        "case_id": "REV-CALC-FAILURE-001",
        "category": "derived_calculation",
        "question": "2025 年营收比 2024 年增加多少？",
        "evidence": [
            {
                "evidence_id": "e-2025",
                "text": "2025 年营收为 100 亿元。",
                "source": "annual-report-2025",
            },
            {
                "evidence_id": "e-2024",
                "text": "2024 年营收为 80 亿元。",
                "source": "annual-report-2024",
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
                    {
                        "name": "current",
                        "value": "100",
                        "unit": "亿元",
                        "evidence_ids": ["e-2025"],
                    },
                    {
                        "name": "prior",
                        "value": "80",
                        "unit": "亿元",
                        "evidence_ids": ["e-2024"],
                    },
                ],
                "evidence_ids": ["e-2025", "e-2024"],
                "result": {"value": "20", "unit": "亿元"},
            }
        ],
        "required_tools": ["report_search", "calculator"],
        "forbidden_tools": ["web_search"],
        "must_abstain": False,
        "partial_answer_gold": {
            "allowed": True,
            "required_claim_ids": ["g-1"],
        },
        "synthetic": False,
        "review_status": "reviewed",
    }


def _matching_prediction() -> dict:
    return {
        "answer": "增加 20 亿元。",
        "abstained": False,
        "claims": [
            {
                "claim_id": "g-1",
                "text": "营收增加 20 亿元",
                "evidence_ids": ["e-2025", "e-2024"],
                "verification": {"status": "ENTAILED"},
            }
        ],
        "calculations": {
            "gold-k1": {
                "calculation_id": "gold-k1",
                "status": "SUCCESS",
                "operation": "subtract",
                "formula": "current - prior",
                "inputs": [
                    {
                        "name": "current",
                        "value": "100",
                        "unit": "亿元",
                        "evidence_ids": ["e-2025"],
                    },
                    {
                        "name": "prior",
                        "value": "80",
                        "unit": "亿元",
                        "evidence_ids": ["e-2024"],
                    },
                ],
                "evidence_ids": ["e-2025", "e-2024"],
                "result": {"value": "20", "unit": "亿元"},
            }
        },
        "tool_calls": [
            {"tool_name": "report_search", "status": "SUCCESS"},
            {"tool_name": "calculator", "status": "SUCCESS"},
        ],
        "trajectory_events": [
            {
                "event_id": "runtime-success",
                "dependencies": [],
                "recovery_of": [],
                "lineage_status": "resolved",
                "step": 1,
                "node": "finalize",
                "status": "SUCCESS",
            }
        ],
    }


def test_matching_reviewed_calculation_contract_has_no_gold_failure() -> None:
    row = evaluate_case(_reviewed_calculation_case(), _matching_prediction())

    assert row["failure_attribution"]["has_failure"] is False
    assert row["failure_attribution"]["failure_events"] == []


@pytest.mark.parametrize(
    ("dimension", "mutate"),
    [
        ("calculation.status", lambda record: record.update(status="FAILED")),
        ("calculation.operation", lambda record: record.update(operation="divide")),
        ("calculation.formula", lambda record: record.update(formula="current / prior")),
        (
            "calculation.inputs",
            lambda record: record["inputs"][0].update(value="101"),
        ),
        (
            "calculation.result",
            lambda record: record["result"].update(value="999"),
        ),
    ],
)
def test_reviewed_calculation_mismatch_is_an_auditable_gold_failure(
    dimension, mutate
) -> None:
    prediction = _matching_prediction()
    mutate(prediction["calculations"]["gold-k1"])

    row = evaluate_case(_reviewed_calculation_case(), prediction)

    failures = [
        event
        for event in row["failure_attribution"]["failure_events"]
        if event["failure_type"] == FailureType.CALCULATION_FAILED.value
    ]
    assert len(failures) == 1
    assert failures[0]["origin"] == "reviewed_gold"
    assert failures[0]["outcome_dimensions"] == [dimension]
    assert failures[0]["gold_id"] == "gold-k1"
    assert failures[0]["dependencies"] == []
    assert row["failure_attribution"]["has_failure"] is True


def test_independent_gold_tool_answer_and_partial_outcomes_do_not_invent_causality() -> None:
    prediction = _matching_prediction()
    prediction["answer"] = "错误答案"
    prediction["claims"][0]["verification"]["status"] = "INSUFFICIENT"
    prediction["tool_calls"] = [
        {"tool_name": "report_search", "status": "SUCCESS"},
        {"tool_name": "web_search", "status": "SUCCESS"},
    ]

    row = evaluate_case(_reviewed_calculation_case(), prediction)
    attribution = row["failure_attribution"]
    gold_events = [
        event for event in attribution["failure_events"] if event["origin"] == "reviewed_gold"
    ]
    dimensions = {
        dimension
        for event in gold_events
        for dimension in event["outcome_dimensions"]
    }

    assert {
        "claim.verification_status",
        "tools.required",
        "tools.forbidden",
        "answer",
        "partial_answer",
    }.issubset(dimensions)
    assert all(event["dependencies"] == [] for event in gold_events)
    assert attribution["root_cause_resolution"] == "multiple_independent"
    assert attribution["downstream_failures"] == []
    assert attribution["downstream_symptoms"] == []
    assert len(attribution["independent_failures"]) == len(gold_events) - 1


def test_must_abstain_gold_does_not_create_an_answer_mismatch_side_effect() -> None:
    case = _reviewed_calculation_case()
    case.update(
        gold_answer="",
        gold_claims=[],
        gold_calculations=[],
        required_tools=[],
        must_abstain=True,
        partial_answer_gold={"allowed": False, "required_claim_ids": []},
    )
    prediction = {
        "answer": "不应回答",
        "abstained": False,
        "claims": [],
        "calculations": {},
        "tool_calls": [],
    }

    row = evaluate_case(case, prediction)
    dimensions = {
        dimension
        for event in row["failure_attribution"]["failure_events"]
        for dimension in event["outcome_dimensions"]
    }

    assert "abstention" in dimensions
    assert "answer" not in dimensions
