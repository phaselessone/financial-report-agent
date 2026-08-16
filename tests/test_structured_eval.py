"""Structured benchmark evaluation tests (checklist v3.0 §P5).

Contract: the four benchmark metrics — Exact Numeric Accuracy, Period
Accuracy, Value Type Accuracy, Citation Accuracy — are computed per query on
fixture gold rows for both legs (structured store hit, and RAG answer rows).
No remote API, no model loading.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from src.evaluation.structured_eval import (
    build_store_from_facts_jsonl,
    evaluate_rag_row,
    evaluate_structured_row,
    gold_to_fact,
    summarize_rows,
)
from src.structured.schema import FinancialFact, Period, PeriodType

GOLD_ROW = {
    "qid": "sf001",
    "company": "宁德时代",
    "metric": "revenue",
    "period": {"kind": "FY", "year": 2025},
    "value": "423702000000",
    "unit": "元",
    "value_type": "actual",
    "doc_id": "doc-ningde-1",
    "page": 9,
    "evidence_id": "chunk-x",
    "raw_value": "4237.02亿元",
    "source_span": "2025 年实现营收4237.02 亿元",
    "query": "宁德时代2025年营业收入是多少",
}

ALIASES = {"宁德时代": ["宁德时代", "宁德"]}


def make_fact(**overrides):
    base = dict(
        company="宁德时代",
        metric="revenue",
        period={"kind": "FY", "year": 2025},
        value_type="actual",
        value="423702000000",
        unit="元",
        doc_id="doc-ningde-1",
        page=9,
        evidence_id="chunk-x",
        raw_value="4237.02亿元",
        source_span="2025 年实现营收4237.02 亿元",
    )
    base.update(overrides)
    return FinancialFact.from_dict(base)


class TestGoldToFact(unittest.TestCase):
    def test_round_trip(self) -> None:
        fact = gold_to_fact(GOLD_ROW)
        self.assertEqual(fact.value, Decimal("423702000000"))
        self.assertEqual(fact.period, Period(PeriodType.FY, 2025))
        self.assertEqual(fact.doc_id, "doc-ningde-1")


class TestBuildStore(unittest.TestCase):
    def test_builds_from_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            facts_path = Path(tmp) / "facts.jsonl"
            facts_path.write_text(
                json.dumps({k: v for k, v in make_fact().to_dict().items()}) + "\n", encoding="utf-8"
            )
            store = build_store_from_facts_jsonl(facts_path)
            self.addCleanup(store.close)
            self.assertEqual(store.count(), 1)


class TestEvaluateStructuredRow(unittest.TestCase):
    def setUp(self) -> None:
        from src.structured.fact_store import FactStore

        self.store = FactStore(":memory:")
        self.addCleanup(self.store.close)

    def test_hit_all_correct(self) -> None:
        self.store.upsert([make_fact()])
        result = evaluate_structured_row(GOLD_ROW, self.store, ALIASES)
        self.assertTrue(result["routed"])
        self.assertTrue(result["hit"])
        self.assertTrue(result["numeric_correct"])
        self.assertTrue(result["period_correct"])
        self.assertTrue(result["value_type_correct"])
        self.assertTrue(result["citation_correct"])

    def test_miss(self) -> None:
        result = evaluate_structured_row(GOLD_ROW, self.store, ALIASES)
        self.assertTrue(result["routed"])
        self.assertFalse(result["hit"])
        self.assertFalse(result["numeric_correct"])

    def test_wrong_value(self) -> None:
        self.store.upsert([make_fact(value="423700000000")])
        result = evaluate_structured_row(GOLD_ROW, self.store, ALIASES)
        self.assertTrue(result["hit"])
        self.assertFalse(result["numeric_correct"])
        self.assertTrue(result["period_correct"])

    def test_wrong_period(self) -> None:
        self.store.upsert([make_fact(period={"kind": "FY", "year": 2024})])
        result = evaluate_structured_row(GOLD_ROW, self.store, ALIASES)
        self.assertFalse(result["period_correct"])
        self.assertFalse(result["citation_correct"])  # doc still matches


class TestEvaluateRagRow(unittest.TestCase):
    def test_exact_numeric_match(self) -> None:
        result = evaluate_rag_row(
            GOLD_ROW,
            {"final_answer": "2025年宁德时代实现营业收入4237.02亿元，同比增长17.04%。",
             "citations": [{"doc_id": "doc-ningde-1"}], "selected_doc_ids": []},
        )
        self.assertTrue(result["numeric_correct"])
        self.assertTrue(result["period_correct"])
        self.assertEqual(result["value_type_correct"], True)  # 实现 → actual
        self.assertTrue(result["citation_correct"])

    def test_numeric_mismatch(self) -> None:
        result = evaluate_rag_row(
            GOLD_ROW,
            {"final_answer": "2025年宁德时代营业收入为4237亿元。", "citations": [], "selected_doc_ids": []},
        )
        self.assertFalse(result["numeric_correct"])
        self.assertTrue(result["period_correct"])  # 2025年 stated
        self.assertIsNone(result["value_type_correct"])
        self.assertFalse(result["citation_correct"])

    def test_period_implied_by_numeric_match(self) -> None:
        result = evaluate_rag_row(
            GOLD_ROW,
            {"final_answer": "宁德时代营业收入为4237.02亿元。", "citations": [], "selected_doc_ids": []},
        )
        self.assertTrue(result["numeric_correct"])
        self.assertTrue(result["period_correct"])  # value match implies period

    def test_percent_gold(self) -> None:
        gold = {**GOLD_ROW, "metric": "gross_margin", "value": "25", "unit": "%", "raw_value": "25%"}
        result = evaluate_rag_row(gold, {"final_answer": "毛利率为25%。", "citations": [], "selected_doc_ids": []})
        self.assertTrue(result["numeric_correct"])

    def test_forecast_value_type_mismatch(self) -> None:
        gold = {**GOLD_ROW, "value_type": "forecast"}
        result = evaluate_rag_row(
            gold, {"final_answer": "2025年宁德时代实现营业收入4237.02亿元。", "citations": [], "selected_doc_ids": []}
        )
        self.assertFalse(result["value_type_correct"])

    def test_forecast_value_type_match(self) -> None:
        gold = {**GOLD_ROW, "value_type": "forecast"}
        result = evaluate_rag_row(
            gold,
            {"final_answer": "预计2025年宁德时代营业收入为4237.02亿元。", "citations": [], "selected_doc_ids": []},
        )
        self.assertTrue(result["value_type_correct"])

    def test_selected_doc_ids_citation(self) -> None:
        result = evaluate_rag_row(
            GOLD_ROW,
            {"final_answer": "4237.02亿元", "citations": [], "selected_doc_ids": ["doc-ningde-1"]},
        )
        self.assertTrue(result["citation_correct"])

    def test_abstained_answer(self) -> None:
        result = evaluate_rag_row(
            GOLD_ROW, {"final_answer": "", "abstained": True, "citations": [], "selected_doc_ids": []}
        )
        self.assertTrue(result["abstained"])
        self.assertFalse(result["numeric_correct"])
        self.assertFalse(result["period_correct"])


class TestSummarizeRows(unittest.TestCase):
    def test_perfect_summary(self) -> None:
        rows = [
            {"numeric_correct": True, "period_correct": True, "value_type_correct": True, "citation_correct": True},
            {"numeric_correct": True, "period_correct": True, "value_type_correct": True, "citation_correct": True},
        ]
        summary = summarize_rows(rows)
        self.assertEqual(summary["exact_numeric_accuracy"], 1.0)
        self.assertEqual(summary["citation_accuracy"], 1.0)

    def test_value_type_attempted_excludes_unknown(self) -> None:
        rows = [
            {"numeric_correct": True, "period_correct": True, "value_type_correct": True, "citation_correct": True},
            {"numeric_correct": False, "period_correct": True, "value_type_correct": None, "citation_correct": False},
        ]
        summary = summarize_rows(rows)
        self.assertEqual(summary["value_type_attempted"], 1)
        self.assertEqual(summary["value_type_accuracy"], 1.0)
        self.assertEqual(summary["exact_numeric_accuracy"], 0.5)

    def test_empty_rows(self) -> None:
        summary = summarize_rows([])
        self.assertEqual(summary["total"], 0)
        self.assertEqual(summary["exact_numeric_accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
