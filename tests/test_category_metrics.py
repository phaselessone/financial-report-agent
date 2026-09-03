from __future__ import annotations

from src.evaluation.hard_case_benchmark import CATEGORIES, evaluate_predictions


def _reviewed_case(*, case_id: str, category: str, calculation: bool) -> dict:
    gold_calculations = []
    required_tools = ["report_search"]
    if calculation:
        required_tools.append("calculator")
        gold_calculations = [
            {
                "calculation_id": f"{case_id}-calc",
                "status": "SUCCESS",
                "operation": "subtract",
                "formula": "current - prior",
                "inputs": [
                    {
                        "name": "current",
                        "value": "100",
                        "unit": "亿元",
                        "evidence_ids": [f"{case_id}-evidence"],
                    },
                    {
                        "name": "prior",
                        "value": "80",
                        "unit": "亿元",
                        "evidence_ids": [f"{case_id}-evidence"],
                    },
                ],
                "evidence_ids": [f"{case_id}-evidence"],
                "result": {"value": "20", "unit": "亿元"},
            }
        ]
    return {
        "case_id": case_id,
        "category": category,
        "question": "问题",
        "evidence": [
            {
                "evidence_id": f"{case_id}-evidence",
                "text": "证据",
                "source": "report",
            }
        ],
        "gold_answer": "正确答案",
        "gold_claims": [
            {
                "claim_id": f"{case_id}-claim",
                "text": "正确结论",
                "status": "ENTAILED",
                "evidence_ids": [f"{case_id}-evidence"],
            }
        ],
        "gold_calculations": gold_calculations,
        "required_tools": required_tools,
        "forbidden_tools": ["web_search"],
        "must_abstain": False,
        "partial_answer_gold": {
            "allowed": True,
            "required_claim_ids": [f"{case_id}-claim"],
        },
        "synthetic": False,
        "review_status": "reviewed",
        "benchmark_version": "reviewed-v1",
    }


def _prediction(case: dict, *, correct: bool) -> dict:
    evidence_id = case["evidence"][0]["evidence_id"]
    claim_id = case["gold_claims"][0]["claim_id"]
    calculations = {}
    if case["gold_calculations"]:
        gold = case["gold_calculations"][0]
        calculations[gold["calculation_id"]] = {
            **gold,
            "result": {
                "value": "20" if correct else "999",
                "unit": "亿元",
            },
        }
    return {
        "answer": "正确答案" if correct else "错误答案",
        "abstained": False,
        "claims": [
            {
                "claim_id": claim_id,
                "text": "正确结论",
                "evidence_ids": [evidence_id],
                "verification": {
                    "status": "ENTAILED" if correct else "INSUFFICIENT"
                },
            }
        ],
        "calculations": calculations,
        "tool_calls": [
            {"tool_name": "report_search", "status": "SUCCESS"},
            *(
                [{"tool_name": "calculator", "status": "SUCCESS"}]
                if correct and case["gold_calculations"]
                else []
            ),
        ],
        "llm_call_count": 2 if correct else 3,
        "tool_call_count": 2 if correct else 1,
        "total_tokens": 10 if correct else 30,
        "end_to_end_latency_ms": 5 if correct else 15,
        "trajectory_events": [
            {
                "event_id": f"{case['case_id']}-runtime",
                "dependencies": [],
                "recovery_of": [],
                "lineage_status": "resolved",
                "step": 1,
                "node": "finalize",
                "status": "SUCCESS",
                "budget_usage": {
                    "tokens": 7 if correct else 21,
                    "latency_ms": 4 if correct else 12,
                },
            }
        ],
    }


def test_category_metrics_separate_process_gold_failure_cost_and_runtime_usage() -> None:
    failed = _reviewed_case(
        case_id="REV-CATEGORY-CALC",
        category="derived_calculation",
        calculation=True,
    )
    passed = _reviewed_case(
        case_id="REV-CATEGORY-FACT",
        category="simple_factual",
        calculation=False,
    )

    metrics, _ = evaluate_predictions(
        [failed, passed],
        {
            failed["case_id"]: _prediction(failed, correct=False),
            passed["case_id"]: _prediction(passed, correct=True),
        },
    )

    failed_category = metrics["category_metrics"]["derived_calculation"]
    process = failed_category["process_metrics"]
    contract = failed_category["contract_metrics"]
    gold = failed_category["gold_metrics"]
    failure = failed_category["failure_summary"]

    assert process["Avg LLM Calls"] == {
        "numerator": 3.0,
        "denominator": 1,
        "micro": 3.0,
        "macro": 3.0,
    }
    assert process["Avg Tool Calls"]["micro"] == 1.0
    assert process["Avg Tokens"]["micro"] == 30.0
    assert process["Avg Latency"]["micro"] == 15.0
    assert contract["Tool Selection Recall"]["micro"] == 0.5
    assert contract["Abstention Contract Match Rate"]["micro"] == 1.0
    assert contract["Partial Credit"]["micro"] == 0.0
    assert gold["Calculation Result Accuracy"]["micro"] == 0.0
    assert gold["Partial Answer Gold Score"]["micro"] == 0.0
    assert failure["failure_rate"] == 1.0
    assert failure["root_cause_counts"] == {
        "verification_false_negative": 1
    }
    assert failure["final_failure_counts"] == {"synthesis_error": 1}
    assert failure["total_cost_before_first_failure"] == {
        "tokens": 21,
        "latency_ms": 12.0,
        "retries": 0,
    }
    assert failure["average_cost_before_first_failure"] == {
        "tokens": 21.0,
        "latency_ms": 12.0,
        "retries": 0.0,
    }

    passed_category = metrics["category_metrics"]["simple_factual"]
    assert passed_category["failure_summary"]["failed_query_count"] == 0
    assert passed_category["process_metrics"]["Avg Tokens"]["micro"] == 10.0
    assert passed_category["gold_metrics"]["Claim Verification Accuracy"]["micro"] == 1.0


def test_empty_category_reports_explicit_empty_process_and_failure_contracts() -> None:
    case = _reviewed_case(
        case_id="REV-CATEGORY-ONLY",
        category="simple_factual",
        calculation=False,
    )

    metrics, _ = evaluate_predictions(
        [case],
        {case["case_id"]: _prediction(case, correct=True)},
    )

    empty = metrics["category_metrics"]["multi_hop"]
    assert empty["total"] == 0
    assert empty["process_metrics"]["Avg Tokens"] == {
        "numerator": 0,
        "denominator": 0,
        "micro": None,
        "macro": None,
    }
    assert empty["failure_summary"]["query_count"] == 0
    assert empty["failure_summary"]["failure_rate"] == 0.0
    assert empty["gold_metrics"] == {}
    assert set(metrics["category_metrics"]) == set(CATEGORIES)


def test_correct_must_abstain_is_not_a_failed_query_but_keeps_process_audit() -> None:
    case = {
        "case_id": "SYN-MUST-ABSTAIN",
        "category": "must_abstain",
        "question": "现有证据能否支持该结论？",
        "evidence": [],
        "gold_answer": "",
        "required_claims": [],
        "required_evidence": [],
        "required_tools": ["report_search"],
        "forbidden_tools": [],
        "must_abstain": True,
        "partial_credit": True,
        "synthetic": True,
        "review_status": "synthetic_not_human_reviewed",
    }
    prediction = {
        "answer": "信息不足，暂时无法给出可靠答案。",
        "abstained": True,
        "termination_reason": "duplicate_tool_call",
        "tool_calls": [
            {
                "tool_name": "report_search",
                "arguments": {"query": case["question"], "domain_hint": ""},
                "status": "SUCCESS",
            }
        ],
        "trajectory_events": [
            {
                "event_id": "runtime-stop",
                "dependencies": [],
                "recovery_of": [],
                "lineage_status": "resolved",
                "step": 2,
                "node": "execute_step",
                "status": "SKIPPED",
                "error_type": "DUPLICATE_TOOL_CALL",
            }
        ],
    }

    metrics, rows = evaluate_predictions([case], {case["case_id"]: prediction})

    attribution = rows[0]["failure_attribution"]
    assert attribution["expected_abstention"] is True
    assert attribution["has_failure"] is False
    assert attribution["failure_count"] == 0
    assert attribution["observed_process_attribution"]["has_failure"] is True
    assert attribution["observed_process_attribution"]["failure_count"] >= 1

    failure = metrics["failure_summary"]
    assert failure["failed_query_count"] == 0
    assert failure["expected_abstention_count"] == 1
    assert failure["observed_process_failure_query_count"] == 1
