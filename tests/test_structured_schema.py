"""Structured financial fact schema tests (checklist v3.0 §P5).

Contract: :class:`src.structured.schema.FinancialFact` enforces mandatory
provenance at construction time; a fact without doc_id/page/evidence_id/
raw_value/source_span can never become an official fact.
"""

from __future__ import annotations

import unittest
from datetime import date
from decimal import Decimal

from src.structured.schema import (
    FinancialFact,
    Metric,
    Period,
    PeriodBasis,
    PeriodType,
    PROVENANCE_FIELDS,
    ValueType,
)

GOOD_FACT = dict(
    company="贵州茅台",
    metric=Metric.REVENUE,
    period=Period(kind=PeriodType.FY, year=2025),
    value_type=ValueType.ACTUAL,
    value=Decimal("1234.5"),
    unit="元",
    doc_id="doc-1",
    page=3,
    evidence_id="E1",
    raw_value="1234.5亿元",
    source_span="2025年实现营业收入1234.5亿元",
)


class TestPeriod(unittest.TestCase):
    def test_period_construction(self) -> None:
        period = Period(kind="FY", year=2025)
        self.assertEqual(period.kind, PeriodType.FY)
        self.assertEqual(period.year, 2025)

    def test_period_rejects_bad_kind(self) -> None:
        with self.assertRaises(ValueError):
            Period(kind="H2", year=2025)

    def test_period_rejects_bad_year(self) -> None:
        with self.assertRaises(ValueError):
            Period(kind="Q1", year=1800)

    def test_period_to_dict(self) -> None:
        self.assertEqual(Period(kind="Q3", year=2025).to_dict(), {"kind": "Q3", "year": 2025})

    def test_period_from_dict(self) -> None:
        self.assertEqual(Period.from_dict({"kind": "H1", "year": 2024}), Period(kind=PeriodType.H1, year=2024))

    def test_period_label(self) -> None:
        self.assertEqual(Period(kind="FY", year=2025).label, "FY2025")


class TestFinancialFact(unittest.TestCase):
    def test_constructs_with_full_provenance(self) -> None:
        fact = FinancialFact(**GOOD_FACT)
        self.assertEqual(fact.metric, Metric.REVENUE)
        self.assertEqual(fact.value, Decimal("1234.5"))

    def test_accepts_str_enums(self) -> None:
        fact = FinancialFact(**{**GOOD_FACT, "metric": "net_profit", "value_type": "forecast"})
        self.assertEqual(fact.metric, Metric.NET_PROFIT)
        self.assertEqual(fact.value_type, ValueType.FORECAST)

    def test_accepts_adjusted_value_type(self) -> None:
        fact = FinancialFact(**{**GOOD_FACT, "value_type": "adjusted"})
        self.assertEqual(fact.value_type, ValueType.ADJUSTED)

    def test_accepts_period_dict(self) -> None:
        fact = FinancialFact(**{**GOOD_FACT, "period": {"kind": "H1", "year": 2024}})
        self.assertEqual(fact.period, Period(kind=PeriodType.H1, year=2024))

    def test_missing_doc_id_raises(self) -> None:
        with self.assertRaises(ValueError):
            FinancialFact(**{**GOOD_FACT, "doc_id": ""})

    def test_missing_page_raises(self) -> None:
        with self.assertRaises(ValueError):
            FinancialFact(**{**GOOD_FACT, "page": None})

    def test_missing_evidence_id_raises(self) -> None:
        with self.assertRaises(ValueError):
            FinancialFact(**{**GOOD_FACT, "evidence_id": ""})

    def test_missing_raw_value_raises(self) -> None:
        with self.assertRaises(ValueError):
            FinancialFact(**{**GOOD_FACT, "raw_value": "  "})

    def test_missing_source_span_raises(self) -> None:
        with self.assertRaises(ValueError):
            FinancialFact(**{**GOOD_FACT, "source_span": ""})

    def test_non_finite_value_raises(self) -> None:
        with self.assertRaises(ValueError):
            FinancialFact(**{**GOOD_FACT, "value": Decimal("NaN")})

    def test_bad_unit_raises(self) -> None:
        with self.assertRaises(ValueError):
            FinancialFact(**{**GOOD_FACT, "unit": "亿"})

    def test_percent_unit_ok_for_margin(self) -> None:
        fact = FinancialFact(
            **{**GOOD_FACT, "metric": "gross_margin", "value": Decimal("17.4"), "unit": "%"}
        )
        self.assertEqual(fact.unit, "%")

    def test_negative_page_raises(self) -> None:
        with self.assertRaises(ValueError):
            FinancialFact(**{**GOOD_FACT, "page": -1})

    def test_fact_id_deterministic(self) -> None:
        first = FinancialFact(**GOOD_FACT)
        second = FinancialFact(**GOOD_FACT)
        self.assertEqual(first.fact_id, second.fact_id)
        self.assertTrue(first.fact_id.startswith("F"))

    def test_fact_id_differs_by_doc(self) -> None:
        first = FinancialFact(**GOOD_FACT)
        other = FinancialFact(**{**GOOD_FACT, "doc_id": "doc-2"})
        self.assertNotEqual(first.fact_id, other.fact_id)

    def test_to_dict_round_trip(self) -> None:
        fact = FinancialFact(**GOOD_FACT)
        restored = FinancialFact.from_dict(fact.to_dict())
        self.assertEqual(restored, fact)

    def test_v2_coordinates_round_trip(self) -> None:
        fact = FinancialFact(
            **{
                **GOOD_FACT,
                "accounting_scope": "consolidated",
                "period_basis": "standalone",
                "source_date": "2026-03-28",
                "revision_status": "restated",
            }
        )
        self.assertEqual(fact.source_date, date(2026, 3, 28))
        restored = FinancialFact.from_dict(fact.to_dict())
        self.assertEqual(restored, fact)
        self.assertEqual(restored.accounting_scope, "consolidated")
        self.assertEqual(restored.period_basis, PeriodBasis.STANDALONE)
        self.assertEqual(restored.revision_status, "restated")

    def test_legacy_json_without_v2_coordinates_still_loads(self) -> None:
        legacy = FinancialFact(**GOOD_FACT).to_dict()
        legacy.pop("accounting_scope", None)
        legacy.pop("period_basis", None)
        legacy.pop("source_date", None)
        legacy.pop("revision_status", None)
        restored = FinancialFact.from_dict(legacy)
        self.assertIsNone(restored.accounting_scope)
        self.assertIsNone(restored.period_basis)
        self.assertIsNone(restored.source_date)
        self.assertIsNone(restored.revision_status)

    def test_to_dict_has_provenance_fields(self) -> None:
        data = FinancialFact(**GOOD_FACT).to_dict()
        for field in PROVENANCE_FIELDS:
            self.assertIn(field, data)

    def test_missing_provenance_fields_helper(self) -> None:
        from src.structured.schema import missing_provenance_fields

        self.assertEqual(missing_provenance_fields(dict(doc_id="d", page=1)), ["evidence_id", "raw_value", "source_span"])
        self.assertEqual(missing_provenance_fields(dict(GOOD_FACT)), [])


if __name__ == "__main__":
    unittest.main()
