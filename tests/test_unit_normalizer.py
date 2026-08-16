"""Unit/value normalizer tests (checklist v3.0 §P5).

Contract: ``normalize_fact_value`` reuses the P4 calculator's numeric parsing
and normalizes fact values to base units — yuan for currency metrics, the
percent number for margin metrics. Dimension mismatches (percent for a
currency metric, currency for a margin, bare currency numbers) return None.
"""

from __future__ import annotations

import unittest
from decimal import Decimal

from src.structured.schema import Metric
from src.structured.unit_normalizer import normalize_fact_value


class TestNormalizeFactValue(unittest.TestCase):
    def test_currency_yi(self) -> None:
        value, unit = normalize_fact_value("1234.5亿元", Metric.REVENUE)
        self.assertEqual(value, Decimal("123450000000"))
        self.assertEqual(unit, "元")

    def test_currency_wan(self) -> None:
        value, unit = normalize_fact_value("5,678万元", Metric.NET_PROFIT)
        self.assertEqual(value, Decimal("56780000"))
        self.assertEqual(unit, "元")

    def test_currency_qian(self) -> None:
        value, unit = normalize_fact_value("9千元", Metric.RD_EXPENSE)
        self.assertEqual(value, Decimal("9000"))
        self.assertEqual(unit, "元")

    def test_currency_operating_cash_flow(self) -> None:
        value, unit = normalize_fact_value("2.05亿元", Metric.OPERATING_CASH_FLOW)
        self.assertEqual(value, Decimal("205000000"))
        self.assertEqual(unit, "元")

    def test_margin_percent(self) -> None:
        value, unit = normalize_fact_value("17.4%", Metric.GROSS_MARGIN)
        self.assertEqual(value, Decimal("17.4"))
        self.assertEqual(unit, "%")

    def test_margin_bare_number_accepted(self) -> None:
        value, unit = normalize_fact_value("21.0", Metric.GROSS_MARGIN)
        self.assertEqual(value, Decimal("21.0"))
        self.assertEqual(unit, "%")

    def test_bare_currency_rejected(self) -> None:
        self.assertIsNone(normalize_fact_value("12.23", Metric.NET_PROFIT))

    def test_percent_for_currency_rejected(self) -> None:
        self.assertIsNone(normalize_fact_value("12.5%", Metric.REVENUE))

    def test_currency_for_margin_rejected(self) -> None:
        self.assertIsNone(normalize_fact_value("30.96亿元", Metric.GROSS_MARGIN))

    def test_multiple_numbers_rejected(self) -> None:
        self.assertIsNone(normalize_fact_value("10亿元和20亿元", Metric.REVENUE))

    def test_empty_rejected(self) -> None:
        self.assertIsNone(normalize_fact_value("", Metric.REVENUE))

    def test_garbage_rejected(self) -> None:
        self.assertIsNone(normalize_fact_value("同比大幅增长", Metric.NET_PROFIT))

    def test_fullwidth_digits(self) -> None:
        value, unit = normalize_fact_value("１２３．４亿", Metric.REVENUE)
        self.assertEqual(value, Decimal("12340000000"))
        self.assertEqual(unit, "元")

    def test_negative_value(self) -> None:
        value, unit = normalize_fact_value("-3.2亿元", Metric.NET_PROFIT)
        self.assertEqual(value, Decimal("-320000000"))
        self.assertEqual(unit, "元")

    def test_unknown_metric_rejected(self) -> None:
        self.assertIsNone(normalize_fact_value("12.3亿元", "ebitda"))

    def test_usd_rejected(self) -> None:
        self.assertIsNone(normalize_fact_value("5.67亿美元", Metric.REVENUE))

    def test_hkd_rejected(self) -> None:
        self.assertIsNone(normalize_fact_value("224.46亿港元", Metric.REVENUE))

    def test_rmb_still_accepted(self) -> None:
        value, unit = normalize_fact_value("224.46亿元", Metric.REVENUE)
        self.assertEqual(value, Decimal("22446000000"))
        self.assertEqual(unit, "元")


if __name__ == "__main__":
    unittest.main()
