"""Fact query / routing tests (checklist v3.0 §P5).

Contract: ``detect_query_signature`` requires all three elements
(company + metric + period) to route a query to structured search; otherwise
None (the query stays on the RAG path). ``route`` answers obvious
company+metric+period queries from the fact store with provenance-bearing
citations, no LLM calls.
"""

from __future__ import annotations

import unittest
from decimal import Decimal

from src.structured.fact_query import FactQuery, detect_query_signature, route
from src.structured.fact_store import FactStore
from src.structured.schema import FinancialFact, Metric, Period, PeriodType

ALIASES = {"贵州茅台": ["贵州茅台", "茅台"], "五粮液": ["五粮液"]}


def make_fact(company="贵州茅台", metric="revenue", year=2025, kind="FY", value="123450000000", **overrides):
    base = dict(
        company=company,
        metric=metric,
        period=Period(kind=kind, year=year),
        value_type="actual",
        value=Decimal(value),
        unit="元",
        doc_id="doc-maotai-1",
        page=3,
        evidence_id="E1",
        raw_value="1234.5亿元",
        source_span="2025年公司实现营业收入1234.5亿元",
    )
    base.update(overrides)
    return FinancialFact(**base)


class TestDetectQuerySignature(unittest.TestCase):
    def test_full_signature(self) -> None:
        signature = detect_query_signature("贵州茅台2025年营业收入是多少", company_aliases=ALIASES)
        self.assertIsNotNone(signature)
        self.assertEqual(signature.company, "贵州茅台")
        self.assertEqual(signature.metric, Metric.REVENUE)
        self.assertEqual(signature.period, Period(PeriodType.FY, 2025))

    def test_quarter_signature(self) -> None:
        signature = detect_query_signature("茅台2025Q1归母净利润", company_aliases=ALIASES)
        self.assertEqual(signature.company, "贵州茅台")
        self.assertEqual(signature.metric, Metric.NET_PROFIT)
        self.assertEqual(signature.period, Period(PeriodType.Q1, 2025))

    def test_h1_signature(self) -> None:
        signature = detect_query_signature("2025年上半年五粮液毛利率", company_aliases=ALIASES)
        self.assertEqual(signature.metric, Metric.GROSS_MARGIN)
        self.assertEqual(signature.period, Period(PeriodType.H1, 2025))

    def test_missing_company_none(self) -> None:
        self.assertIsNone(detect_query_signature("2025年营收是多少", company_aliases=ALIASES))

    def test_missing_metric_none(self) -> None:
        self.assertIsNone(detect_query_signature("贵州茅台2025年业绩如何", company_aliases=ALIASES))

    def test_missing_period_none(self) -> None:
        self.assertIsNone(detect_query_signature("茅台营收", company_aliases=ALIASES))

    def test_missing_period_even_with_anchor_none(self) -> None:
        self.assertIsNone(detect_query_signature("茅台营收", company_aliases=ALIASES, anchor_year=2025))

    def test_out_of_scope_metric_none(self) -> None:
        self.assertIsNone(detect_query_signature("贵州茅台2025年净利率", company_aliases=ALIASES))

    def test_empty_none(self) -> None:
        self.assertIsNone(detect_query_signature("", company_aliases=ALIASES))


class TestRoute(unittest.TestCase):
    def setUp(self) -> None:
        self.store = FactStore(":memory:")
        self.addCleanup(self.store.close)

    def test_route_hit(self) -> None:
        self.store.upsert([make_fact()])
        result = route("贵州茅台2025年营业收入是多少", store=self.store, company_aliases=ALIASES)
        self.assertIsNotNone(result)
        self.assertTrue(result["routed"])
        self.assertEqual(len(result["facts"]), 1)
        self.assertIn("贵州茅台", result["answer"])
        self.assertIn("1234.5亿元", result["answer"])
        citations = result["citations"]
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["doc_id"], "doc-maotai-1")
        self.assertEqual(citations[0]["page"], 3)

    def test_route_store_miss_still_routed(self) -> None:
        result = route("贵州茅台2025年营业收入是多少", store=self.store, company_aliases=ALIASES)
        self.assertIsNotNone(result)
        self.assertTrue(result["routed"])
        self.assertEqual(result["facts"], [])
        self.assertEqual(result["citations"], [])

    def test_route_no_signature_returns_none(self) -> None:
        self.store.upsert([make_fact()])
        self.assertIsNone(route("这个行业景气度如何", store=self.store, company_aliases=ALIASES))

    def test_route_returns_fact_query(self) -> None:
        self.store.upsert([make_fact()])
        result = route("茅台2025Q1归母净利润", store=self.store, company_aliases=ALIASES)
        self.assertIsNotNone(result)
        signature: FactQuery = result["fact_query"]
        self.assertEqual(signature.metric, Metric.NET_PROFIT)
        self.assertEqual(signature.period, Period(PeriodType.Q1, 2025))
        self.assertEqual(result["facts"], [])

    def test_margin_answer_format(self) -> None:
        margin = make_fact(
            company="五粮液",
            metric="gross_margin",
            year=2025,
            kind="H1",
            value="25",
            unit="%",
            raw_value="25%",
            source_span="2025年上半年毛利率25%",
        )
        self.store.upsert([margin])
        result = route("2025年上半年五粮液毛利率", store=self.store, company_aliases=ALIASES)
        self.assertIsNotNone(result)
        self.assertIn("25%", result["answer"])


if __name__ == "__main__":
    unittest.main()
