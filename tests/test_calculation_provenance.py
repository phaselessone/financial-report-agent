"""Focused contract tests for calculation provenance records."""

from __future__ import annotations

import unittest
from decimal import Decimal

from src.agent.calculation_provenance import (
    CalculationOperand,
    CalculationStatus,
    execute_calculation,
)


def operand(
    name: str,
    value,
    *,
    unit: str | None = None,
    evidence_id: str | None = None,
    fact_id: str | None = None,
    period: int | str | None = None,
) -> CalculationOperand:
    return CalculationOperand(
        name=name,
        value=value,
        unit=unit,
        evidence_id=evidence_id,
        fact_id=fact_id,
        period=period,
    )


class TestCalculationProvenance(unittest.TestCase):
    def test_yoy_success_is_verified_and_ordered(self) -> None:
        record = execute_calculation(
            "yoy",
            [
                operand("prior", 100, unit="亿元", evidence_id="E-prior"),
                operand("current", 120, unit="亿元", fact_id="F-current"),
            ],
            claimed_result="20%",
        )

        self.assertEqual(record.status, CalculationStatus.SUCCESS)
        self.assertTrue(record.verified)
        self.assertEqual(record.tool, "financial_calculator")
        self.assertEqual([item.name for item in record.inputs], ["prior", "current"])
        self.assertEqual([item.value for item in record.inputs], ["10000000000", "12000000000"])
        self.assertEqual(record.result.formatted, "20.0000%")
        self.assertEqual(record.formula, "(12000000000-10000000000)/10000000000")
        self.assertTrue(record.claim_result_matches)

    def test_negative_yoy(self) -> None:
        record = execute_calculation(
            "yoy",
            [
                operand("prior", "100亿元", evidence_id="E1"),
                operand("current", "85亿元", evidence_id="E2"),
            ],
            claimed_result="-15%",
        )
        self.assertEqual(record.status, CalculationStatus.SUCCESS)
        self.assertEqual(record.result.value, "-0.1500")
        self.assertEqual(record.result.formatted, "-15.0000%")

    def test_qoq_is_whitelisted_yoy_formula_alias(self) -> None:
        record = execute_calculation(
            "qoq",
            [
                {"value": 80, "unit": "元", "evidence_id": "E1"},
                {"value": 100, "unit": "元", "evidence_id": "E2"},
            ],
            claimed_result="25%",
        )
        self.assertEqual(record.status, CalculationStatus.SUCCESS)
        self.assertEqual(record.operation, "qoq")
        self.assertEqual([item.name for item in record.inputs], ["prior", "current"])
        self.assertEqual(record.result.value, "0.2500")

    def test_zero_denominator_returns_observable_failure(self) -> None:
        record = execute_calculation(
            "ratio",
            [
                operand("numerator", 1, evidence_id="E1"),
                operand("denominator", 0, evidence_id="E2"),
            ],
        )
        self.assertEqual(record.status, CalculationStatus.DIVIDE_BY_ZERO)
        self.assertFalse(record.verified)
        self.assertIsNone(record.result)
        self.assertEqual(record.error_type, "CalculationError")
        self.assertIn("non-zero", record.error_message)
        self.assertTrue(record.calculation_id.startswith("CALC-"))

    def test_mixed_currency_scales_are_normalized_to_yuan(self) -> None:
        record = execute_calculation(
            "difference",
            [
                operand("left", 2, unit="亿", evidence_id="E1"),
                operand("right", 3000, unit="万", evidence_id="E2"),
            ],
            claimed_result="1.7亿元",
        )
        self.assertEqual(record.status, CalculationStatus.SUCCESS)
        self.assertEqual(record.result.value, "170000000.0000")
        self.assertEqual(record.result.unit, "元")
        self.assertEqual(record.result.formatted, "170000000.0000元")

    def test_cagr_derives_years_from_periods(self) -> None:
        record = execute_calculation(
            "cagr",
            [
                operand("start", 100, fact_id="F1", period="FY2021"),
                operand("end", 144, fact_id="F2", period="FY2023"),
            ],
            claimed_result="20%",
        )
        self.assertEqual(record.status, CalculationStatus.SUCCESS)
        self.assertEqual(record.years, "2")
        self.assertEqual(record.result.value, "0.2000")

    def test_cagr_rejects_period_span_disagreement(self) -> None:
        record = execute_calculation(
            "cagr",
            [
                operand("start", 100, evidence_id="E1", period=2021),
                operand("end", 144, evidence_id="E2", period=2023),
            ],
            years=3,
        )
        self.assertEqual(record.status, CalculationStatus.INVALID_PERIOD)
        self.assertFalse(record.verified)
        self.assertIn("period span", record.error_message)

    def test_cagr_two_and_three_year_periods_are_distinct(self) -> None:
        two_year = execute_calculation(
            "cagr",
            [operand("start", 100, fact_id="F1"), operand("end", 144, fact_id="F2")],
            years=2,
        )
        three_year = execute_calculation(
            "cagr",
            [operand("start", 100, fact_id="F1"), operand("end", 144, fact_id="F2")],
            years=3,
        )
        self.assertEqual(two_year.status, CalculationStatus.SUCCESS)
        self.assertEqual(three_year.status, CalculationStatus.SUCCESS)
        self.assertNotEqual(two_year.result.value, three_year.result.value)
        self.assertNotEqual(two_year.calculation_id, three_year.calculation_id)

    def test_percentage_point_result_is_not_percent_result(self) -> None:
        pp = execute_calculation(
            "percentage_point_change",
            [
                operand("prior_pct", 12.5, unit="%", evidence_id="E1"),
                operand("current_pct", 13, unit="%", evidence_id="E2"),
            ],
            claimed_result="0.5个百分点",
        )
        wrong_unit = execute_calculation(
            "percentage_point_change",
            [
                operand("prior_pct", 12.5, unit="%", evidence_id="E1"),
                operand("current_pct", 13, unit="%", evidence_id="E2"),
            ],
            claimed_result="0.5%",
        )
        relative_percent = execute_calculation(
            "yoy",
            [
                operand("prior", 12.5, unit="%", evidence_id="E1"),
                operand("current", 13, unit="%", evidence_id="E2"),
            ],
            claimed_result="4%",
        )
        self.assertEqual(pp.status, CalculationStatus.SUCCESS)
        self.assertEqual(pp.result.unit, "pp")
        self.assertEqual(wrong_unit.status, CalculationStatus.RESULT_MISMATCH)
        self.assertEqual(relative_percent.status, CalculationStatus.SUCCESS)
        self.assertEqual(relative_percent.result.formatted, "4.0000%")

    def test_claimed_result_mismatch_preserves_calculation(self) -> None:
        record = execute_calculation(
            "gross_margin",
            [
                operand("gross_profit", 40, unit="亿元", evidence_id="E1"),
                operand("revenue", 100, unit="亿元", evidence_id="E2"),
            ],
            claimed_result="45%",
        )
        self.assertEqual(record.status, CalculationStatus.RESULT_MISMATCH)
        self.assertFalse(record.verified)
        self.assertFalse(record.claim_result_matches)
        self.assertEqual(record.result.formatted, "40.0000%")

    def test_missing_provenance_never_produces_verified_record(self) -> None:
        record = execute_calculation(
            "net_margin",
            [
                operand("net_profit", 15, unit="亿元", evidence_id="E1"),
                operand("revenue", 100, unit="亿元"),
            ],
            claimed_result="15%",
        )
        self.assertEqual(record.status, CalculationStatus.MISSING_PROVENANCE)
        self.assertFalse(record.verified)
        self.assertFalse(record.provenance_complete)
        self.assertEqual(record.result.value, "0.1500")
        self.assertIn("revenue", record.error_message)

    def test_fact_id_alone_is_valid_provenance(self) -> None:
        record = execute_calculation(
            "ratio",
            [operand("numerator", 3, fact_id="F1"), operand("denominator", 8, fact_id="F2")],
        )
        self.assertEqual(record.status, CalculationStatus.SUCCESS)
        self.assertTrue(record.provenance_complete)
        self.assertTrue(record.verified)

    def test_upstream_calculation_lineage_is_persisted_without_impersonating_fact_ids(self) -> None:
        record = execute_calculation(
            "difference",
            [
                {
                    "name": "left",
                    "value": "0.20",
                    "upstream_calculation_id": "CALC-UPSTREAM-LEFT",
                    "source_step_id": "calculate_left",
                    "upstream_evidence_ids": ["E1", "E2"],
                },
                {
                    "name": "right",
                    "value": "0.10",
                    "upstream_calculation_id": "CALC-UPSTREAM-RIGHT",
                    "source_step_id": "calculate_right",
                    "upstream_evidence_ids": ["E3", "E4"],
                },
            ],
        )

        payload = record.to_dict()
        self.assertEqual(record.status, CalculationStatus.SUCCESS)
        self.assertEqual(payload["input_calculation_ids"], ["CALC-UPSTREAM-LEFT", "CALC-UPSTREAM-RIGHT"])
        self.assertEqual(payload["source_step_ids"], ["calculate_left", "calculate_right"])
        self.assertEqual(payload["evidence_ids"], ["E1", "E2", "E3", "E4"])
        self.assertTrue(all(item["fact_id"] is None for item in payload["inputs"]))

    def test_unit_mismatch_returns_record(self) -> None:
        record = execute_calculation(
            "difference",
            [
                operand("left", 12.5, unit="%", evidence_id="E1"),
                operand("right", 100, unit="亿元", evidence_id="E2"),
            ],
        )
        self.assertEqual(record.status, CalculationStatus.UNIT_MISMATCH)
        self.assertFalse(record.verified)
        self.assertIsNotNone(record.error_message)

    def test_unknown_operation_is_not_dynamically_dispatched(self) -> None:
        record = execute_calculation(
            "python_eval",
            [operand("left", 1, evidence_id="E1"), operand("right", 2, evidence_id="E2")],
        )
        self.assertEqual(record.status, CalculationStatus.UNSUPPORTED_OPERATION)
        self.assertFalse(record.verified)

    def test_malformed_input_returns_record_instead_of_raising(self) -> None:
        record = execute_calculation(
            "yoy",
            [{"name": "prior", "value": "not-a-number", "evidence_id": "E1"}, 7],
        )
        self.assertEqual(record.status, CalculationStatus.INVALID_INPUT)
        self.assertFalse(record.verified)
        self.assertTrue(record.calculation_id.startswith("CALC-"))

    def test_non_iterable_inputs_return_record_instead_of_raising(self) -> None:
        record = execute_calculation("yoy", None)  # type: ignore[arg-type]
        self.assertEqual(record.status, CalculationStatus.INVALID_INPUT)
        self.assertFalse(record.verified)
        self.assertEqual(record.inputs[0].value, "None")
        self.assertTrue(record.calculation_id.startswith("CALC-"))

    def test_calculation_id_is_content_stable(self) -> None:
        inputs = [
            operand("prior", Decimal("100"), evidence_id="E1"),
            operand("current", Decimal("120"), evidence_id="E2"),
        ]
        first = execute_calculation("yoy", inputs, precision=2, claimed_result="20%")
        second = execute_calculation("yoy", inputs, precision=2, claimed_result="20%")
        changed = execute_calculation("yoy", inputs, precision=4, claimed_result="20%")
        self.assertEqual(first.calculation_id, second.calculation_id)
        self.assertNotEqual(first.calculation_id, changed.calculation_id)
        self.assertEqual(first.to_dict(), second.to_dict())


if __name__ == "__main__":
    unittest.main()
