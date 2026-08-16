"""Period normalizer tests (checklist v3.0 §P5).

Contract: ``normalize_period`` maps raw Chinese/English period mentions to a
canonical ``Period(kind, year)``. V1 kinds: FY / Q1 / H1 / Q3. Ambiguous or
unresolvable input (no year, no anchor, multi-year ranges) returns None.
"""

from __future__ import annotations

import unittest

from src.structured.period_normalizer import normalize_period
from src.structured.schema import Period, PeriodType


class TestNormalizePeriod(unittest.TestCase):
    def test_full_year(self) -> None:
        self.assertEqual(normalize_period("2025年"), Period(PeriodType.FY, 2025))

    def test_bare_year(self) -> None:
        self.assertEqual(normalize_period("2024"), Period(PeriodType.FY, 2024))

    def test_year_with_degree_suffix(self) -> None:
        self.assertEqual(normalize_period("2025年度"), Period(PeriodType.FY, 2025))

    def test_fy_prefix(self) -> None:
        self.assertEqual(normalize_period("FY2025"), Period(PeriodType.FY, 2025))

    def test_whole_year(self) -> None:
        self.assertEqual(normalize_period("2025年全年"), Period(PeriodType.FY, 2025))

    def test_q1_cn(self) -> None:
        self.assertEqual(normalize_period("2025年一季度"), Period(PeriodType.Q1, 2025))

    def test_q1_short(self) -> None:
        self.assertEqual(normalize_period("2025Q1"), Period(PeriodType.Q1, 2025))

    def test_q1_lower(self) -> None:
        self.assertEqual(normalize_period("2025q1"), Period(PeriodType.Q1, 2025))

    def test_q3_cn(self) -> None:
        self.assertEqual(normalize_period("2025年第三季度"), Period(PeriodType.Q3, 2025))

    def test_q3_first_three(self) -> None:
        self.assertEqual(normalize_period("2025年前三季度"), Period(PeriodType.Q3, 2025))

    def test_h1_cn(self) -> None:
        self.assertEqual(normalize_period("2025年上半年"), Period(PeriodType.H1, 2025))

    def test_h1_short(self) -> None:
        self.assertEqual(normalize_period("2025H1"), Period(PeriodType.H1, 2025))

    def test_month_range_h1(self) -> None:
        self.assertEqual(normalize_period("2025年1-6月"), Period(PeriodType.H1, 2025))

    def test_month_range_q1(self) -> None:
        self.assertEqual(normalize_period("2025年1-3月"), Period(PeriodType.Q1, 2025))

    def test_month_range_q3(self) -> None:
        self.assertEqual(normalize_period("2025年7-9月"), Period(PeriodType.Q3, 2025))

    def test_two_digit_year_q1(self) -> None:
        self.assertEqual(normalize_period("25Q1"), Period(PeriodType.Q1, 2025))

    def test_two_digit_year_with_cn(self) -> None:
        self.assertEqual(normalize_period("25年"), Period(PeriodType.FY, 2025))

    def test_two_digit_year_with_cn_in_sentence(self) -> None:
        self.assertEqual(normalize_period("三花智控：25 年营收 310.1 亿"), Period(PeriodType.FY, 2025))

    def test_relative_q1_uses_anchor(self) -> None:
        self.assertEqual(normalize_period("一季度", anchor_year=2024), Period(PeriodType.Q1, 2024))

    def test_relative_h1_uses_anchor(self) -> None:
        self.assertEqual(normalize_period("上半年", anchor_year=2024), Period(PeriodType.H1, 2024))

    def test_relative_whole_year_uses_anchor(self) -> None:
        self.assertEqual(normalize_period("全年", anchor_year=2024), Period(PeriodType.FY, 2024))

    def test_relative_q3_uses_anchor(self) -> None:
        self.assertEqual(normalize_period("前三季度", anchor_year=2024), Period(PeriodType.Q3, 2024))

    def test_relative_without_anchor_returns_none(self) -> None:
        self.assertIsNone(normalize_period("一季度"))
        self.assertIsNone(normalize_period("上半年"))
        self.assertIsNone(normalize_period("全年"))

    def test_year_range_returns_none(self) -> None:
        self.assertIsNone(normalize_period("2024-2026年"))
        self.assertIsNone(normalize_period("2024/2025年"))

    def test_garbage_returns_none(self) -> None:
        self.assertIsNone(normalize_period("同比增长12%"))
        self.assertIsNone(normalize_period("展望未来"))
        self.assertIsNone(normalize_period(""))

    def test_year_only_in_range_returns_none(self) -> None:
        self.assertIsNone(normalize_period("2023年至2025年"))

    def test_q1_full_cn_formal(self) -> None:
        self.assertEqual(normalize_period("2025年第一季度"), Period(PeriodType.Q1, 2025))

    def test_space_separated_year_quarter(self) -> None:
        self.assertEqual(normalize_period("2025 年 Q1"), Period(PeriodType.Q1, 2025))


if __name__ == "__main__":
    unittest.main()
