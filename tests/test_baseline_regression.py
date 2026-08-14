from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from src.generation.answerer import LocalEvidenceAnswerer
from src.utils.io import read_jsonl

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "baseline_regression.jsonl"

SNAPSHOT_KEYS = (
    "question_type",
    "answer_mode",
    "selected_doc_ids",
    "used_evidence_ids",
    "abstained",
    "abstain_reason",
    "citation_count",
    "supported",
    "matched_numeric_tokens",
)


def build_retrieval_result(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rerank_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        ranked = dict(row)
        ranked.setdefault("rank", index)
        ranked.setdefault("score", float(len(rows) - index + 1))
        rerank_rows.append(ranked)
    return {
        "dense_rows": [],
        "bm25_rows": [],
        "hybrid_rows": [],
        "rerank_rows": rerank_rows,
        "timings": {"retrieval_latency_ms": 0.0, "rerank_latency_ms": 0.0},
    }


def _seq_responder(responses: list[str]):
    remaining = list(responses)

    def respond(prompt: str) -> str:
        if remaining:
            return remaining.pop(0)
        return ""

    return respond


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    answerer = LocalEvidenceAnswerer.__new__(LocalEvidenceAnswerer)
    answerer._generate = _seq_responder(list(case.get("provider_responses", [])))
    override = case.get("prepare_evidence_override")
    retrieval_result = build_retrieval_result(case.get("rows", []))
    if override is not None:
        with patch("src.generation.answerer._prepare_evidence", return_value=tuple(override)):
            result = LocalEvidenceAnswerer.answer(
                answerer,
                query=case["query"],
                question_type=case["question_type"],
                retrieval_result=retrieval_result,
                query_domain_hint=case.get("query_domain_hint", ""),
            )
    else:
        result = LocalEvidenceAnswerer.answer(
            answerer,
            query=case["query"],
            question_type=case["question_type"],
            retrieval_result=retrieval_result,
            query_domain_hint=case.get("query_domain_hint", ""),
        )
    return snapshot_result(result)


def snapshot_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_type": result["question_type"],
        "answer_mode": result["answer_mode"],
        "selected_doc_ids": list(result["selected_doc_ids"]),
        "used_evidence_ids": list(result["used_evidence_ids"]),
        "abstained": bool(result["abstained"]),
        "abstain_reason": result["abstain_reason"],
        "citation_count": len(result["citations"]),
        "supported": bool(result["support_validation"].get("supported", False)),
        "matched_numeric_tokens": list(result["matched_numeric_tokens"]),
    }


class BaselineRegressionTests(unittest.TestCase):
    def test_baseline_snapshot_matches_frozen_fixture(self) -> None:
        cases = read_jsonl(FIXTURE_PATH)
        self.assertGreaterEqual(len(cases), 28, "baseline regression fixture must contain at least 28 cases")
        case_ids = [case["case_id"] for case in cases]
        self.assertEqual(len(case_ids), len(set(case_ids)), "case ids must be unique")
        for case in cases:
            with self.subTest(case_id=case["case_id"]):
                observed = run_case(case)
                self.assertEqual(
                    observed,
                    case["expected"],
                    f"baseline drift for {case['case_id']}: {json.dumps(observed, ensure_ascii=False)}",
                )


if __name__ == "__main__":
    unittest.main()
