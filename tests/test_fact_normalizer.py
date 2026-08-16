"""Fact normalizer tests (checklist v3.0 §P5).

Contract: ``match_metric`` maps metric keywords to V1 metrics with guards
against lookalike rates (净利率 must not match 净利润); ``normalize_company_name``
resolves company mentions through an alias map, longest alias wins.
"""

from __future__ import annotations

import unittest

from src.structured.fact_normalizer import match_metric, normalize_company_name
from src.structured.schema import Metric

ALIASES = {"贵州茅台": ["贵州茅台", "茅台"], "五粮液": ["五粮液"], "宁德时代": ["宁德时代", "宁德"]}


class TestMatchMetric(unittest.TestCase):
    def test_revenue_full(self) -> None:
        self.assertEqual(match_metric("营业收入"), Metric.REVENUE)

    def test_revenue_short(self) -> None:
        self.assertEqual(match_metric("营收"), Metric.REVENUE)

    def test_revenue_total(self) -> None:
        self.assertEqual(match_metric("营业总收入"), Metric.REVENUE)

    def test_net_profit(self) -> None:
        self.assertEqual(match_metric("净利润"), Metric.NET_PROFIT)

    def test_net_profit_attributable(self) -> None:
        self.assertEqual(match_metric("归母净利润"), Metric.NET_PROFIT)

    def test_net_profit_ex_nonrecurring(self) -> None:
        self.assertEqual(match_metric("扣非净利润"), Metric.NET_PROFIT)

    def test_gross_margin(self) -> None:
        self.assertEqual(match_metric("毛利率"), Metric.GROSS_MARGIN)

    def test_gross_margin_composite(self) -> None:
        self.assertEqual(match_metric("综合毛利率"), Metric.GROSS_MARGIN)

    def test_rd_expense(self) -> None:
        self.assertEqual(match_metric("研发费用"), Metric.RD_EXPENSE)

    def test_rd_investment(self) -> None:
        self.assertEqual(match_metric("研发投入"), Metric.RD_EXPENSE)

    def test_operating_cash_flow_full(self) -> None:
        self.assertEqual(match_metric("经营活动现金流量净额"), Metric.OPERATING_CASH_FLOW)

    def test_operating_cash_flow_short(self) -> None:
        self.assertEqual(match_metric("经营性现金流"), Metric.OPERATING_CASH_FLOW)

    def test_net_margin_rate_not_net_profit(self) -> None:
        self.assertIsNone(match_metric("净利率"))

    def test_net_profit_rate_not_net_profit(self) -> None:
        self.assertIsNone(match_metric("净利润率"))

    def test_gross_profit_not_in_v1(self) -> None:
        self.assertIsNone(match_metric("毛利"))

    def test_multiple_metrics_ambiguous(self) -> None:
        self.assertIsNone(match_metric("营收和净利润"))

    def test_empty(self) -> None:
        self.assertIsNone(match_metric(""))

    def test_growth_rate_of_revenue_still_revenue(self) -> None:
        self.assertEqual(match_metric("营收增速"), Metric.REVENUE)


class TestNormalizeCompanyName(unittest.TestCase):
    def test_exact_canonical(self) -> None:
        self.assertEqual(normalize_company_name("贵州茅台2025年营收", ALIASES), "贵州茅台")

    def test_short_alias(self) -> None:
        self.assertEqual(normalize_company_name("茅台2025年净利润", ALIASES), "贵州茅台")

    def test_second_company(self) -> None:
        self.assertEqual(normalize_company_name("五粮液净利率", ALIASES), "五粮液")

    def test_longest_alias_wins(self) -> None:
        self.assertEqual(normalize_company_name("宁德时代2025Q1", ALIASES), "宁德时代")

    def test_no_match(self) -> None:
        self.assertIsNone(normalize_company_name("比亚迪2025年营收", ALIASES))

    def test_empty_text(self) -> None:
        self.assertIsNone(normalize_company_name("", ALIASES))

    def test_not_str(self) -> None:
        self.assertIsNone(normalize_company_name(None, ALIASES))


if __name__ == "__main__":
    unittest.main()
