"""DuckDB fact store (checklist v3.0 §P5).

Official facts live in DuckDB with mandatory-provenance columns (NOT NULL):
a fact without doc_id / page / evidence_id / raw_value / source_span can
never be persisted. Upserts are idempotent on the deterministic fact_id;
queries are exact on (company, metric, period kind, year). Parquet export
provides the portable snapshot format.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb

from src.structured.schema import (
    FinancialFact,
    Metric,
    Period,
    PeriodType,
    ValueType,
    missing_provenance_fields,
)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS facts (
    fact_id VARCHAR PRIMARY KEY,
    company VARCHAR NOT NULL,
    metric VARCHAR NOT NULL,
    period_kind VARCHAR NOT NULL,
    year INTEGER NOT NULL,
    value_type VARCHAR NOT NULL,
    value VARCHAR NOT NULL,
    unit VARCHAR NOT NULL,
    doc_id VARCHAR NOT NULL,
    page INTEGER NOT NULL,
    evidence_id VARCHAR NOT NULL,
    raw_value VARCHAR NOT NULL,
    source_span VARCHAR NOT NULL
)
"""

_COLUMNS = (
    "fact_id",
    "company",
    "metric",
    "period_kind",
    "year",
    "value_type",
    "value",
    "unit",
    "doc_id",
    "page",
    "evidence_id",
    "raw_value",
    "source_span",
)


def _fact_to_row(fact: FinancialFact) -> tuple[Any, ...]:
    return (
        fact.fact_id,
        fact.company,
        fact.metric.value,
        fact.period.kind.value,
        fact.period.year,
        fact.value_type.value,
        str(fact.value),
        fact.unit,
        fact.doc_id,
        fact.page,
        fact.evidence_id,
        fact.raw_value,
        fact.source_span,
    )


def _row_to_fact(row: Sequence[Any]) -> FinancialFact:
    values = dict(zip(_COLUMNS, row))
    return FinancialFact(
        company=values["company"],
        metric=Metric(values["metric"]),
        period=Period(kind=PeriodType(values["period_kind"]), year=values["year"]),
        value_type=ValueType(values["value_type"]),
        value=Decimal(values["value"]),
        unit=values["unit"],
        doc_id=values["doc_id"],
        page=values["page"],
        evidence_id=values["evidence_id"],
        raw_value=values["raw_value"],
        source_span=values["source_span"],
        fact_id=values["fact_id"],
    )


class FactStore:
    """DuckDB-backed store for official financial facts."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._conn = duckdb.connect(self.db_path)
        self._conn.execute(_SCHEMA_SQL)

    def close(self) -> None:
        self._conn.close()

    def upsert(self, facts: Iterable[FinancialFact]) -> int:
        """Insert facts, skipping duplicates; returns the inserted count.

        Only :class:`FinancialFact` instances with complete provenance can be
        persisted — anything else raises ValueError (defense in depth on top
        of the schema's constructor checks).
        """
        rows: list[tuple[Any, ...]] = []
        for fact in facts:
            if not isinstance(fact, FinancialFact):
                raise ValueError(f"FactStore only persists FinancialFact instances, got {type(fact).__name__}")
            if missing_provenance_fields(fact.to_provenance()):
                raise ValueError(f"fact {fact.fact_id} missing mandatory provenance")
            rows.append(_fact_to_row(fact))
        if not rows:
            return 0
        placeholders = ", ".join(["?"] * len(_COLUMNS))
        before = self.count()
        self._conn.executemany(
            f"INSERT OR IGNORE INTO facts ({', '.join(_COLUMNS)}) VALUES ({placeholders})", rows
        )
        return self.count() - before

    def query(
        self,
        *,
        company: str | None = None,
        metric: Metric | str | None = None,
        period: Period | None = None,
    ) -> list[FinancialFact]:
        """Query facts with optional filters; company/metric match exactly."""
        conditions: list[str] = []
        params: list[Any] = []
        if company is not None:
            conditions.append("company = ?")
            params.append(company)
        if metric is not None:
            conditions.append("metric = ?")
            params.append(metric.value if isinstance(metric, Metric) else Metric(metric).value)
        if period is not None:
            conditions.append("period_kind = ? AND year = ?")
            params.extend([period.kind.value, period.year])
        sql = f"SELECT {', '.join(_COLUMNS)} FROM facts"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_fact(row) for row in rows]

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0])

    def all_facts(self) -> list[FinancialFact]:
        rows = self._conn.execute(f"SELECT {', '.join(_COLUMNS)} FROM facts").fetchall()
        return [_row_to_fact(row) for row in rows]

    def dump_parquet(self, path: str | Path) -> None:
        """Export the facts table to a Parquet file."""
        self._conn.execute(f"COPY (SELECT {', '.join(_COLUMNS)} FROM facts) TO ? (FORMAT PARQUET)", [str(path)])

    def load_parquet(self, path: str | Path) -> int:
        """Import facts from a Parquet snapshot; returns inserted count."""
        before = self.count()
        self._conn.execute(
            f"INSERT OR IGNORE INTO facts ({', '.join(_COLUMNS)}) "
            f"SELECT {', '.join(_COLUMNS)} FROM read_parquet(?)",
            [str(path)],
        )
        return self.count() - before
