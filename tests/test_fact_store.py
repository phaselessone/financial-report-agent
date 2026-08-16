"""Fact store tests (checklist v3.0 §P5).

Contract: DuckDB-backed store with mandatory-provenance columns (NOT NULL),
idempotent upsert, exact (company, metric, period) queries, and Parquet
export. Only :class:`FinancialFact` instances with complete provenance can
be persisted.
"""

from __future__ import annotations

import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from src.structured.fact_store import FactStore
from src.structured.schema import FinancialFact, Metric, Period, PeriodType, ValueType


def make_fact(company="贵州茅台", metric="revenue", year=2025, kind="FY", value="123450000000", **overrides):
    base = dict(
        company=company,
        metric=metric,
        period=Period(kind=kind, year=year),
        value_type="actual",
        value=Decimal(value),
        unit="元",
        doc_id=f"doc-{company}-{year}",
        page=3,
        evidence_id="E1",
        raw_value="1234.5亿元",
        source_span="2025年公司实现营业收入1234.5亿元",
    )
    base.update(overrides)
    return FinancialFact(**base)


class TestFactStore(unittest.TestCase):
    def test_memory_upsert_and_count(self) -> None:
        store = FactStore(":memory:")
        self.addCleanup(store.close)
        inserted = store.upsert([make_fact()])
        self.assertEqual(inserted, 1)
        self.assertEqual(store.count(), 1)

    def test_upsert_idempotent(self) -> None:
        store = FactStore(":memory:")
        self.addCleanup(store.close)
        fact = make_fact()
        self.assertEqual(store.upsert([fact]), 1)
        self.assertEqual(store.upsert([fact]), 0)
        self.assertEqual(store.count(), 1)

    def test_upsert_rejects_non_fact(self) -> None:
        store = FactStore(":memory:")
        self.addCleanup(store.close)
        with self.assertRaises(ValueError):
            store.upsert([{"company": "x"}])  # type: ignore[arg-type]

    def test_exact_query_hit(self) -> None:
        store = FactStore(":memory:")
        self.addCleanup(store.close)
        store.upsert([make_fact(), make_fact(metric="net_profit", value="136550000000")])
        facts = store.query(company="贵州茅台", metric="revenue", period=Period(PeriodType.FY, 2025))
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].value, Decimal("123450000000"))

    def test_exact_query_miss_wrong_year(self) -> None:
        store = FactStore(":memory:")
        self.addCleanup(store.close)
        store.upsert([make_fact()])
        self.assertEqual(store.query(company="贵州茅台", metric="revenue", period=Period(PeriodType.FY, 2024)), [])

    def test_query_without_period(self) -> None:
        store = FactStore(":memory:")
        self.addCleanup(store.close)
        store.upsert([make_fact(year=2024), make_fact(year=2025)])
        facts = store.query(company="贵州茅台", metric="revenue")
        self.assertEqual({fact.period.year for fact in facts}, {2024, 2025})

    def test_query_wrong_metric(self) -> None:
        store = FactStore(":memory:")
        self.addCleanup(store.close)
        store.upsert([make_fact()])
        self.assertEqual(store.query(company="贵州茅台", metric="gross_margin"), [])

    def test_all_facts_round_trip(self) -> None:
        store = FactStore(":memory:")
        self.addCleanup(store.close)
        margin = make_fact(metric="gross_margin", value="25", unit="%", raw_value="25%", source_span="毛利率25%")
        store.upsert([make_fact(), margin])
        facts = store.all_facts()
        self.assertEqual(len(facts), 2)
        by_id = {fact.fact_id: fact for fact in facts}
        restored = by_id[margin.fact_id]
        self.assertEqual(restored.unit, "%")
        self.assertEqual(restored.value, Decimal("25"))
        self.assertEqual(restored.metric, Metric.GROSS_MARGIN)

    def test_decimal_exactness_preserved(self) -> None:
        store = FactStore(":memory:")
        self.addCleanup(store.close)
        fact = make_fact(value="123450000000.1234")
        store.upsert([fact])
        restored = store.all_facts()[0]
        self.assertEqual(restored.value, Decimal("123450000000.1234"))

    def test_provenance_columns_not_null(self) -> None:
        store = FactStore(":memory:")
        self.addCleanup(store.close)
        import duckdb

        with self.assertRaises(Exception):
            store._conn.execute(
                "INSERT INTO facts (fact_id, company, metric, period_kind, year, value_type, value, unit, "
                "doc_id, page, evidence_id, raw_value, source_span) "
                "VALUES ('F1','x','revenue','FY',2025,'actual','1','元','d',1,NULL,'1','span')"
            )

    def test_parquet_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FactStore(":memory:")
            fact = make_fact()
            store.upsert([fact])
            parquet_path = Path(tmp) / "facts.parquet"
            store.dump_parquet(parquet_path)
            store.close()

            second = FactStore(":memory:")
            self.addCleanup(second.close)
            second.load_parquet(parquet_path)
            self.assertEqual(second.count(), 1)
            restored = second.all_facts()[0]
            self.assertEqual(restored, fact)

    def test_file_backed_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "facts.duckdb"
            store = FactStore(db_path)
            store.upsert([make_fact()])
            store.close()

            reopened = FactStore(db_path)
            self.addCleanup(reopened.close)
            self.assertEqual(reopened.count(), 1)

    def test_period_values_preserved(self) -> None:
        store = FactStore(":memory:")
        self.addCleanup(store.close)
        q3 = make_fact(metric="net_profit", year=2024, kind="Q3", value="900000000")
        store.upsert([q3])
        facts = store.query(company="贵州茅台", metric="net_profit", period=Period(PeriodType.Q3, 2024))
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].period.kind, PeriodType.Q3)
        self.assertEqual(facts[0].value_type, ValueType.ACTUAL)

    def test_empty_store(self) -> None:
        store = FactStore(":memory:")
        self.addCleanup(store.close)
        self.assertEqual(store.count(), 0)
        self.assertEqual(store.all_facts(), [])


if __name__ == "__main__":
    unittest.main()
