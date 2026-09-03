"""Fact store tests (checklist v3.0 §P5).

Contract: DuckDB-backed store with mandatory-provenance columns (NOT NULL),
idempotent upsert, exact (company, metric, period) queries, and Parquet
export. Only :class:`FinancialFact` instances with complete provenance can
be persisted.
"""

from __future__ import annotations

import tempfile
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path

import duckdb

from src.structured.fact_store import FactStore
from src.structured.schema import FinancialFact, Metric, Period, PeriodBasis, PeriodType, ValueType


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

    def test_v2_coordinates_round_trip(self) -> None:
        store = FactStore(":memory:")
        self.addCleanup(store.close)
        fact = make_fact(
            accounting_scope="consolidated",
            period_basis="standalone",
            source_date="2026-03-28",
            revision_status="restated",
        )
        store.upsert([fact])
        restored = store.all_facts()[0]
        self.assertEqual(restored.accounting_scope, "consolidated")
        self.assertEqual(restored.period_basis, PeriodBasis.STANDALONE)
        self.assertEqual(restored.source_date, date(2026, 3, 28))
        self.assertEqual(restored.revision_status, "restated")

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

    def test_loads_legacy_parquet_without_v2_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parquet_path = Path(tmp) / "legacy.parquet"
            conn = duckdb.connect(":memory:")
            conn.execute(
                """
                CREATE TABLE legacy AS SELECT
                    'Flegacy'::VARCHAR AS fact_id,
                    '贵州茅台'::VARCHAR AS company,
                    'revenue'::VARCHAR AS metric,
                    'FY'::VARCHAR AS period_kind,
                    2025::INTEGER AS year,
                    'actual'::VARCHAR AS value_type,
                    '123450000000'::VARCHAR AS value,
                    '元'::VARCHAR AS unit,
                    'doc-legacy'::VARCHAR AS doc_id,
                    3::INTEGER AS page,
                    'E-legacy'::VARCHAR AS evidence_id,
                    '1234.5亿元'::VARCHAR AS raw_value,
                    '2025年公司实现营业收入1234.5亿元'::VARCHAR AS source_span
                """
            )
            conn.execute("COPY legacy TO ? (FORMAT PARQUET)", [str(parquet_path)])
            conn.close()

            store = FactStore(":memory:")
            self.addCleanup(store.close)
            self.assertEqual(store.load_parquet(parquet_path), 1)
            restored = store.all_facts()[0]
            self.assertIsNone(restored.accounting_scope)
            self.assertIsNone(restored.period_basis)
            self.assertIsNone(restored.source_date)
            self.assertIsNone(restored.revision_status)

    def test_query_many_accepts_period_basis_enum(self) -> None:
        store = FactStore(":memory:")
        self.addCleanup(store.close)
        store.upsert([
            make_fact(period=Period("Q2", 2025), period_basis="standalone", evidence_id="E-single"),
            make_fact(period=Period("Q2", 2025), period_basis="cumulative", evidence_id="E-ytd"),
        ])
        facts = store.query_many(period_bases=[PeriodBasis.STANDALONE])
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].period_basis, PeriodBasis.STANDALONE)

    def test_file_backed_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "facts.duckdb"
            store = FactStore(db_path)
            store.upsert([make_fact()])
            store.close()

            reopened = FactStore(db_path)
            self.addCleanup(reopened.close)
            self.assertEqual(reopened.count(), 1)

    def test_reopens_and_migrates_legacy_duckdb(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "legacy.duckdb"
            conn = duckdb.connect(str(db_path))
            conn.execute(
                """
                CREATE TABLE facts (
                    fact_id VARCHAR PRIMARY KEY, company VARCHAR NOT NULL,
                    metric VARCHAR NOT NULL, period_kind VARCHAR NOT NULL,
                    year INTEGER NOT NULL, value_type VARCHAR NOT NULL,
                    value VARCHAR NOT NULL, unit VARCHAR NOT NULL,
                    doc_id VARCHAR NOT NULL, page INTEGER NOT NULL,
                    evidence_id VARCHAR NOT NULL, raw_value VARCHAR NOT NULL,
                    source_span VARCHAR NOT NULL
                )
                """
            )
            conn.execute(
                "INSERT INTO facts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    "Flegacy",
                    "贵州茅台",
                    "revenue",
                    "FY",
                    2025,
                    "actual",
                    "123450000000",
                    "元",
                    "doc-legacy",
                    3,
                    "E-legacy",
                    "1234.5亿元",
                    "2025年公司实现营业收入1234.5亿元",
                ],
            )
            conn.close()

            store = FactStore(db_path)
            self.addCleanup(store.close)
            restored = store.all_facts()[0]
            self.assertEqual(restored.fact_id, "Flegacy")
            self.assertIsNone(restored.accounting_scope)
            self.assertIsNone(restored.period_basis)
            self.assertIsNone(restored.source_date)
            self.assertIsNone(restored.revision_status)

    def test_query_many_filters_before_materializing_rows(self) -> None:
        store = FactStore(":memory:")
        self.addCleanup(store.close)
        store.upsert([make_fact()])
        store._conn.execute(
            "INSERT INTO facts (fact_id, company, metric, period_kind, year, value_type, value, unit, "
            "doc_id, page, evidence_id, raw_value, source_span) "
            "VALUES ('Fbad','过滤范围外','revenue','FY',2025,'actual','not-a-number','元','d',1,'Ebad','x','span')"
        )
        facts = store.query_many(companies=["贵州茅台"], metrics=[Metric.REVENUE])
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].company, "贵州茅台")

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
