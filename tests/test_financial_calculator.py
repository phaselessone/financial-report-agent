"""Financial calculator tests (checklist v3.0 §P4).

TDD contract for src.tools.financial_calculator: local deterministic
Decimal arithmetic, unit normalization, provenance passthrough, and a
source-level guard against dynamic code execution.
"""

from __future__ import annotations

import unittest
from decimal import Decimal
from pathlib import Path

from src.tools.financial_calculator import (
    CalculationError,
    CalculationInput,
    cagr,
    difference,
    gross_margin,
    growth_rate,
    input_from_row,
    net_margin,
    parse_numeric_value,
    percentage_point_change,
    ratio,
    yoy,
)

MODULE_SOURCE = (Path(__file__).resolve().parents[1] / "src" / "tools" / "financial_calculator.py").read_text(
    encoding="utf-8"
)


class TestParseNumericValue(unittest.TestCase):
    def test_currency_yi(self) -> None:
        value, scale, unit = parse_numeric_value("123.4亿元")
        self.assertEqual(value, Decimal("12340000000"))
        self.assertEqual(scale, 10**8)
        self.assertEqual(unit, "亿")

    def test_currency_wan_with_comma(self) -> None:
        value, scale, unit = parse_numeric_value("5,678万元")
        self.assertEqual(value, Decimal("56780000"))
        self.assertEqual(scale, 10**4)
        self.assertEqual(unit, "万")

    def test_currency_qian(self) -> None:
        value, scale, unit = parse_numeric_value("9千元")
        self.assertEqual(value, Decimal("9000"))
        self.assertEqual(scale, 10**3)
        self.assertEqual(unit, "千")

    def test_percent(self) -> None:
        value, scale, unit = parse_numeric_value("12.5%")
        self.assertEqual(value, Decimal("12.5"))
        self.assertEqual(scale, 1)
        self.assertEqual(unit, "%")

    def test_bare_number(self) -> None:
        value, scale, unit = parse_numeric_value("42")
        self.assertEqual(value, Decimal("42"))
        self.assertEqual(scale, 1)
        self.assertIsNone(unit)

    def test_fullwidth_digits_and_dot(self) -> None:
        value, scale, unit = parse_numeric_value("１２３．４亿")
        self.assertEqual(value, Decimal("12340000000"))
        self.assertEqual(scale, 10**8)
        self.assertEqual(unit, "亿")

    def test_negative_value(self) -> None:
        value, scale, unit = parse_numeric_value("-3.2%")
        self.assertEqual(value, Decimal("-3.2"))
        self.assertEqual(unit, "%")

    def test_empty_text_raises(self) -> None:
        with self.assertRaises(CalculationError):
            parse_numeric_value("")

    def test_non_numeric_raises(self) -> None:
        with self.assertRaises(CalculationError):
            parse_numeric_value("同比增长")

    def test_ambiguous_multi_candidate_raises(self) -> None:
        with self.assertRaises(CalculationError):
            parse_numeric_value("收入12.3亿元,利润4.5亿元")

    def test_unknown_unit_raises(self) -> None:
        with self.assertRaises(CalculationError):
            parse_numeric_value("12.3美元")


class TestCalculationInput(unittest.TestCase):
    def test_from_raw_carries_provenance(self) -> None:
        item = CalculationInput.from_raw("123.4亿元", evidence_id="ev1", chunk_id="ch1", doc_id="d1")
        self.assertEqual(item.value, Decimal("12340000000"))
        self.assertEqual(item.evidence_id, "ev1")
        self.assertEqual(item.chunk_id, "ch1")
        self.assertEqual(item.doc_id, "d1")

    def test_input_from_row_prefers_support_span(self) -> None:
        row = {
            "chunk_id": "ch1",
            "doc_id": "d1",
            "evidence_id": "ev1",
            "support_span": "营收123.4亿元",
            "text": "无关 9",
        }
        item = input_from_row(row)
        self.assertEqual(item.value, Decimal("12340000000"))
        self.assertEqual(item.raw_text, "营收123.4亿元")
        self.assertEqual(item.evidence_id, "ev1")

    def test_input_from_row_falls_back_to_text(self) -> None:
        row = {"chunk_id": "ch1", "text": "利润45.6亿元"}
        item = input_from_row(row)
        self.assertEqual(item.value, Decimal("4560000000"))
        self.assertIsNone(item.evidence_id)


class TestRatioFamilyOps(unittest.TestCase):
    def test_growth_rate(self) -> None:
        result = growth_rate(100, 115)
        self.assertEqual(result.operation, "growth_rate")
        self.assertEqual(result.value, Decimal("0.1500"))
        self.assertEqual(result.formatted, "15.0000%")
        self.assertEqual(result.formula, "(115-100)/100")

    def test_growth_rate_negative(self) -> None:
        result = growth_rate(100, 85)
        self.assertEqual(result.value, Decimal("-0.1500"))
        self.assertEqual(result.formatted, "-15.0000%")

    def test_yoy_alias(self) -> None:
        result = yoy(80, 96)
        self.assertEqual(result.operation, "yoy")
        self.assertEqual(result.value, Decimal("0.2000"))

    def test_cagr(self) -> None:
        result = cagr(100, 144, 2)
        self.assertEqual(result.value, Decimal("0.2000"))
        self.assertEqual(result.formatted, "20.0000%")

    def test_cagr_multi_year(self) -> None:
        result = cagr(100, 125, 3)
        self.assertAlmostEqual(float(result.value), float((Decimal("125") / Decimal("100")) ** (Decimal(1) / Decimal(3)) - 1), places=4)

    def test_gross_margin(self) -> None:
        result = gross_margin(40, 100)
        self.assertEqual(result.value, Decimal("0.4000"))
        self.assertEqual(result.formatted, "40.0000%")

    def test_net_margin(self) -> None:
        result = net_margin(15, 100)
        self.assertEqual(result.value, Decimal("0.1500"))

    def test_ratio(self) -> None:
        result = ratio(3, 8)
        self.assertEqual(result.value, Decimal("0.3750"))
        self.assertEqual(result.formatted, "37.5000%")

    def test_precision_override(self) -> None:
        result = growth_rate(100, 115, precision=2)
        self.assertEqual(result.value, Decimal("0.15"))
        self.assertEqual(result.formatted, "15.00%")

    def test_round_half_up(self) -> None:
        self.assertEqual(ratio(2, 3, precision=2).value, Decimal("0.67"))
        self.assertEqual(ratio(1, 3, precision=2).value, Decimal("0.33"))

    def test_string_and_float_inputs_coerced(self) -> None:
        self.assertEqual(growth_rate("100", 115).value, Decimal("0.1500"))
        self.assertEqual(ratio(3.0, 8).value, Decimal("0.3750"))

    def test_growth_rate_percent_pair(self) -> None:
        result = growth_rate("12.5%", "13.0%")
        self.assertEqual(result.value, Decimal("0.0400"))
        self.assertEqual(result.formatted, "4.0000%")

    def test_growth_rate_dimension_mismatch_raises(self) -> None:
        with self.assertRaises(CalculationError):
            growth_rate("10", "12.5%")
        with self.assertRaises(CalculationError):
            growth_rate("12.5%", "10亿元")

    def test_ratio_currency_pair(self) -> None:
        self.assertEqual(ratio("2亿元", "4亿元").value, Decimal("0.5000"))

    def test_cagr_currency_pair(self) -> None:
        self.assertEqual(cagr("100亿元", "121亿元", 2).value, Decimal("0.1000"))

    def test_margin_dimension_mismatch_raises(self) -> None:
        with self.assertRaises(CalculationError):
            gross_margin("10%", "100亿元")
        with self.assertRaises(CalculationError):
            net_margin("10亿元", "12.5%")

    def test_provenance_passthrough_in_order(self) -> None:
        left = CalculationInput.from_raw("100亿元", evidence_id="ev1", chunk_id="ch1", doc_id="d1")
        right = CalculationInput.from_raw("115亿元", evidence_id="ev2", chunk_id="ch2", doc_id="d2")
        result = growth_rate(left, right)
        self.assertEqual(len(result.provenance), 2)
        self.assertEqual(result.provenance[0]["evidence_id"], "ev1")
        self.assertEqual(result.provenance[1]["evidence_id"], "ev2")
        self.assertEqual(result.provenance[0]["raw_text"], "100亿元")

    def test_deterministic_repeat(self) -> None:
        first = ratio(1, 7)
        second = ratio(1, 7)
        self.assertEqual(str(first.value), str(second.value))
        self.assertEqual(first.formatted, second.formatted)

    def test_division_by_zero_raises(self) -> None:
        with self.assertRaises(CalculationError):
            ratio(1, 0)

    def test_growth_rate_zero_start_raises(self) -> None:
        with self.assertRaises(CalculationError):
            growth_rate(0, 5)

    def test_margin_zero_revenue_raises(self) -> None:
        with self.assertRaises(CalculationError):
            gross_margin(10, 0)

    def test_margin_negative_revenue_raises(self) -> None:
        with self.assertRaises(CalculationError):
            net_margin(10, -5)

    def test_cagr_non_positive_start_raises(self) -> None:
        with self.assertRaises(CalculationError):
            cagr(0, 100, 2)

    def test_cagr_non_positive_end_raises(self) -> None:
        with self.assertRaises(CalculationError):
            cagr(100, 0, 2)

    def test_cagr_non_positive_years_raises(self) -> None:
        with self.assertRaises(CalculationError):
            cagr(100, 144, 0)
        with self.assertRaises(CalculationError):
            cagr(100, 144, -1)

    def test_nan_input_raises(self) -> None:
        with self.assertRaises(CalculationError):
            ratio(Decimal("NaN"), 2)

    def test_inf_input_raises(self) -> None:
        with self.assertRaises(CalculationError):
            growth_rate(Decimal("Infinity"), 2)


class TestDifferenceOps(unittest.TestCase):
    def test_currency_difference_normalizes(self) -> None:
        result = difference("123.4亿元", "0.4亿元")
        self.assertEqual(result.value, Decimal("12300000000.0000"))
        self.assertEqual(result.formatted, "12300000000.0000元")
        self.assertEqual(result.unit, "元")

    def test_mixed_currency_scales(self) -> None:
        result = difference("2亿元", "3000万元")
        self.assertEqual(result.value, Decimal("170000000"))
        self.assertEqual(result.unit, "元")

    def test_percent_difference(self) -> None:
        result = difference("13.0%", "12.5%")
        self.assertEqual(result.value, Decimal("0.5000"))
        self.assertEqual(result.unit, "%")

    def test_bare_difference(self) -> None:
        result = difference("10", "4")
        self.assertEqual(result.value, Decimal("6.0000"))
        self.assertIsNone(result.unit)

    def test_unit_dimension_mismatch_raises(self) -> None:
        with self.assertRaises(CalculationError):
            difference("12.5%", "100亿元")

    def test_percentage_point_change(self) -> None:
        result = percentage_point_change("12.5%", "13.0%")
        self.assertEqual(result.operation, "percentage_point_change")
        self.assertEqual(result.value, Decimal("0.5000"))
        self.assertEqual(result.formatted, "0.5000pp")
        self.assertEqual(result.unit, "pp")

    def test_percentage_point_change_requires_percent(self) -> None:
        with self.assertRaises(CalculationError):
            percentage_point_change("12.5", "13.0%")
        with self.assertRaises(CalculationError):
            percentage_point_change("12.5%", "13.0")

    def test_difference_provenance(self) -> None:
        left = CalculationInput.from_raw("100亿元", doc_id="d1")
        right = CalculationInput.from_raw("80亿元", doc_id="d2")
        result = difference(left, right)
        self.assertEqual([item["doc_id"] for item in result.provenance], ["d1", "d2"])


class TestSourceGuard(unittest.TestCase):
    def test_no_dynamic_execution_or_llm(self) -> None:
        forbidden = ("eval(", "exec(", "subprocess", "importlib", "httpx", "src.llm")
        for token in forbidden:
            self.assertNotIn(token, MODULE_SOURCE, f"financial_calculator.py must not contain {token!r}")


if __name__ == "__main__":
    unittest.main()
