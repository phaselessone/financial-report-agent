"""claim_eval metrics tests (checklist v3.0 §P7).

Three metrics over a batch of agent runs:
  Claim Support Precision      supported claims / total claims
  Critical Claim Support Rate  supported critical claims / critical claims
  Derived Claim Validation Accuracy (v1 weak: derived claims whose text is
        consistent with the draft; flagged N/A rather than a real calculator
        recheck since v1 does not wire the P4 calculator)
"""

from __future__ import annotations

import unittest

from src.evaluation.claim_eval import (
    analyze_claims_from_state,
    build_claim_eval_summary,
    critical_claim_ids,
    evaluate_claim_verification,
    is_critical_claim,
)


def _claim(text, *, supported=True, claim_type="EXTRACTED", claim_id="claim-1"):
    return {
        "claim_id": claim_id,
        "text": text,
        "claim_type": claim_type,
        "evidence_ids": ["c1"],
        "calculation_id": None,
        "parent_claim_ids": [],
        "verification": {
            "status": "ENTAILED" if supported else "INSUFFICIENT",
            "method": "test",
            "score": 1.0 if supported else 0.0,
            "reasons": ["fixture"],
        },
        "supported": supported,
    }


class ClaimEvalMetricsTests(unittest.TestCase):
    def test_reviewed_claim_verification_compares_status_and_evidence_independently(self) -> None:
        gold_claims = [
            {
                "claim_id": "g-1",
                "text": "营收增加 20 亿元",
                "status": "ENTAILED",
                "evidence_ids": ["e-2025", "e-2024"],
            }
        ]
        predicted_claims = [
            {
                "claim_id": "g-1",
                "text": "营收增加 20 亿元",
                "evidence_ids": ["e-2025"],
                "verification": {"status": "ENTAILED"},
            }
        ]

        metrics = evaluate_claim_verification(predicted_claims, gold_claims)

        self.assertEqual(metrics["claim_status_accuracy"], 1.0)
        self.assertEqual(metrics["claim_evidence_accuracy"], 0.0)
        self.assertEqual(metrics["claim_verification_accuracy"], 0.0)
        self.assertEqual(metrics["evaluated_claim_count"], 1)

    def test_is_critical_marked_for_numeric_or_first_claim(self) -> None:
        c_numeric = _claim("营收1234亿元", claim_id="claim-1")
        c_text = _claim("机构看好增长", claim_id="claim-2")
        self.assertTrue(is_critical_claim(c_numeric))
        self.assertTrue(is_critical_claim(c_text, index=0))  # first claim is critical
        self.assertFalse(is_critical_claim(c_text, index=1))  # non-numeric, non-first

    def test_critical_claim_ids(self) -> None:
        claims = [
            _claim("营收1234亿元", claim_id="claim-1"),
            _claim("机构看好增长", claim_id="claim-2"),
            _claim("研发投入80亿", claim_id="claim-3"),
        ]
        self.assertEqual(critical_claim_ids(claims), {"claim-1", "claim-3"})  # numeric ones

    def test_analyze_claims_from_state_metrics(self) -> None:
        state = {
            "claims": [
                _claim("营收1234亿元", supported=True, claim_id="claim-1"),
                _claim("同比增长31%", supported=True, claim_type="DERIVED", claim_id="claim-2"),
                _claim("机构看好增长", supported=False, claim_id="claim-3"),
            ],
            "draft_answer": {
                "final_answer": "营收1234亿元，同比增长31%，机构看好增长。",
                "support_validation": {"supported": True},
            },
        }
        metrics = analyze_claims_from_state(state)
        self.assertEqual(metrics["claim_count"], 3)
        self.assertEqual(metrics["supported_count"], 2)
        self.assertEqual(metrics["claim_support_precision"], 2 / 3)
        self.assertEqual(metrics["claim_entailment_yield"], 2 / 3)
        self.assertEqual(metrics["metric_scope"], "process")
        self.assertFalse(any("accuracy" in key for key in metrics))
        # critical = claim-1 (numeric) and claim-2 (numeric); both supported
        self.assertEqual(metrics["critical_count"], 2)
        self.assertEqual(metrics["critical_supported"], 2)
        self.assertEqual(metrics["critical_claim_support_rate"], 1.0)
        # derived consistency: "同比增长31%" appears in the draft verbatim
        self.assertTrue(metrics["derived_consistent"])

    def test_analyze_no_claims(self) -> None:
        metrics = analyze_claims_from_state({"claims": []})
        self.assertEqual(metrics["claim_count"], 0)
        self.assertIsNone(metrics["claim_support_precision"])  # undefined -> None
        self.assertIsNone(metrics["critical_claim_support_rate"])

    def test_derived_inconsistent_detected(self) -> None:
        state = {
            "claims": [_claim("同比增长31%", supported=True, claim_type="DERIVED", claim_id="claim-1")],
            "draft_answer": {"final_answer": "营收1234亿元。", "support_validation": {"supported": True}},
        }
        metrics = analyze_claims_from_state(state)
        self.assertFalse(metrics["derived_consistent"])

    def test_build_claim_eval_summary(self) -> None:
        rows = [
            {
                "query": "q1",
                "termination_reason": "completed",
                "claim_count": 2,
                "supported_count": 1,
                "claim_support_precision": 0.5,
                "critical_count": 1,
                "critical_supported": 0,
                "critical_claim_support_rate": 0.0,
                "derived_consistent": True,
            },
            {
                "query": "q2",
                "termination_reason": "completed",
                "claim_count": 4,
                "supported_count": 4,
                "claim_support_precision": 1.0,
                "critical_count": 2,
                "critical_supported": 2,
                "critical_claim_support_rate": 1.0,
                "derived_consistent": False,
            },
        ]
        summary = build_claim_eval_summary(rows)
        self.assertEqual(summary["queries"], 2)
        self.assertEqual(summary["claim_support_precision"], 0.75)  # avg
        self.assertEqual(summary["claim_entailment_yield"], 0.75)
        self.assertEqual(summary["metric_scope"], "process")
        self.assertEqual(summary["critical_claim_support_rate"], 0.5)
        self.assertEqual(summary["derived_consistent_rate"], 0.5)
        self.assertEqual(summary["total_claims"], 6)


if __name__ == "__main__":
    unittest.main()
