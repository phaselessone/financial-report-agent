"""Value type classifier tests (checklist v3.0 §P5).

Contract: ``classify_value_type`` labels a fact's source span as
actual / forecast / unknown using deterministic keyword rules. Forecast
markers win over actual markers (e.g. "预计...同比增长" is a forecast).
"""

from __future__ import annotations

import unittest

from src.structured.schema import ValueType
from src.structured.value_type import classify_value_type


class TestClassifyValueType(unittest.TestCase):
    def test_actual_implied_by_achieved(self) -> None:
        self.assertEqual(classify_value_type("2025年实现营业收入1234.5亿元"), ValueType.ACTUAL)

    def test_actual_yoy(self) -> None:
        self.assertEqual(classify_value_type("净利润12.23亿元，同比增长23.2%"), ValueType.ACTUAL)

    def test_actual_recorded(self) -> None:
        self.assertEqual(classify_value_type("公司录得营收23.2亿元"), ValueType.ACTUAL)

    def test_forecast_expected(self) -> None:
        self.assertEqual(classify_value_type("预计2025年实现营收约23.2亿元"), ValueType.FORECAST)

    def test_forecast_guidance(self) -> None:
        self.assertEqual(classify_value_type("全年指引营收50亿元"), ValueType.FORECAST)

    def test_forecast_wins_over_yoy(self) -> None:
        self.assertEqual(classify_value_type("预计2025年营收同比增长16%"), ValueType.FORECAST)

    def test_forecast_will(self) -> None:
        self.assertEqual(classify_value_type("2025年将实现净利润翻倍"), ValueType.FORECAST)

    def test_unknown_when_no_marker(self) -> None:
        self.assertEqual(classify_value_type("营业收入为1234.5亿元"), ValueType.UNKNOWN)

    def test_unknown_empty(self) -> None:
        self.assertEqual(classify_value_type(""), ValueType.UNKNOWN)

    def test_estimate_year_suffix(self) -> None:
        self.assertEqual(classify_value_type("2025E 净利润 870.9 亿元"), ValueType.FORECAST)

    def test_two_digit_estimate_year_suffix(self) -> None:
        self.assertEqual(classify_value_type("25E 营收"), ValueType.FORECAST)

    def test_unknown_not_str(self) -> None:
        self.assertEqual(classify_value_type(None), ValueType.UNKNOWN)

    def test_adjusted_marker_takes_precedence(self) -> None:
        self.assertEqual(classify_value_type("经调整净利润同比增长12%"), ValueType.ADJUSTED)


if __name__ == "__main__":
    unittest.main()
