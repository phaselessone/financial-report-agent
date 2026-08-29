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
    PeriodBasis,
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
    source_span VARCHAR NOT NULL,
    accounting_scope VARCHAR,
    period_basis VARCHAR,
    source_date DATE,
    revision_status VARCHAR
)
"""

_V2_COLUMN_DEFINITIONS: dict[str, str] = {
    "accounting_scope": "VARCHAR",
    "period_basis": "VARCHAR",
    "source_date": "DATE",
    "revision_status": "VARCHAR",
}

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
    "accounting_scope",
    "period_basis",
    "source_date",
    "revision_status",
)

_LEGACY_COLUMNS = _COLUMNS[:13]


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
        fact.accounting_scope,
        fact.period_basis.value if fact.period_basis is not None else None,
        fact.source_date,
        fact.revision_status,
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
        accounting_scope=values["accounting_scope"],
        period_basis=values["period_basis"],
        source_date=values["source_date"],
        revision_status=values["revision_status"],
        fact_id=values["fact_id"],
    )


class FactStore:
    """DuckDB-backed store for official financial facts."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._conn = duckdb.connect(self.db_path)
        self._conn.execute(_SCHEMA_SQL)
        self._migrate_legacy_schema()

    def _migrate_legacy_schema(self) -> None:
        existing = {str(row[1]) for row in self._conn.execute("PRAGMA table_info('facts')").fetchall()}
        for column, sql_type in _V2_COLUMN_DEFINITIONS.items():
            if column not in existing:
                self._conn.execute(f"ALTER TABLE facts ADD COLUMN {column} {sql_type}")

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
        companies: Sequence[str] | None = None,
        metrics: Sequence[Metric | str] | None = None,
        periods: Sequence[Period] | None = None,
        value_type: ValueType | str | None = None,
        period_basis: PeriodBasis | str | None = None,
        accounting_scope: str | None = None,
        revision_status: str | None = None,
        order_by: str = "period",
        descending: bool = False,
    ) -> list[FinancialFact]:
        """Query facts with optional filters; company/metric match exactly."""
        conditions: list[str] = []
        params: list[Any] = []
        if companies:
            placeholders = ", ".join(["?"] * len(companies))
            conditions.append(f"company IN ({placeholders})")
            params.extend(list(companies))
        if company is not None:
            conditions.append("company = ?")
            params.append(company)
        if metrics:
            placeholders = ", ".join(["?"] * len(metrics))
            conditions.append(f"metric IN ({placeholders})")
            params.extend([metric.value if isinstance(metric, Metric) else Metric(metric).value for metric in metrics])
        if metric is not None:
            conditions.append("metric = ?")
            params.append(metric.value if isinstance(metric, Metric) else Metric(metric).value)
        if periods:
            placeholders = ", ".join(["(?, ?)"] * len(periods))
            conditions.append(f"(period_kind, year) IN ({placeholders})")
            for item in periods:
                params.extend([item.kind.value, item.year])
        if period is not None:
            conditions.append("period_kind = ? AND year = ?")
            params.extend([period.kind.value, period.year])
        if value_type is not None:
            conditions.append("value_type = ?")
            params.append(value_type.value if isinstance(value_type, ValueType) else ValueType(value_type).value)
        if period_basis is not None:
            conditions.append("period_basis = ?")
            params.append(period_basis.value if isinstance(period_basis, PeriodBasis) else PeriodBasis(period_basis).value)
        if accounting_scope is not None:
            conditions.append("accounting_scope = ?")
            params.append(accounting_scope)
        if revision_status is not None:
            conditions.append("revision_status = ?")
            params.append(revision_status)
        sql = f"SELECT {', '.join(_COLUMNS)} FROM facts"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        if order_by not in {"company", "metric", "period", "value", "page"}:
            raise ValueError(f"unsupported order_by {order_by!r}")
        direction = "DESC" if descending else "ASC"
        if order_by == "period":
            sql += f" ORDER BY year {direction}, period_kind {direction}, company {direction}, metric {direction}"
        elif order_by == "company":
            sql += f" ORDER BY company {direction}, year {direction}, period_kind {direction}, metric {direction}"
        elif order_by == "metric":
            sql += f" ORDER BY metric {direction}, company {direction}, year {direction}, period_kind {direction}"
        elif order_by == "value":
            sql += f" ORDER BY CAST(value AS DOUBLE) {direction}, company {direction}, metric {direction}"
        else:
            sql += f" ORDER BY page {direction}, company {direction}, metric {direction}"
        rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_fact(row) for row in rows]

    def query_many(
        self,
        *,
        companies: Sequence[str] | None = None,
        metrics: Sequence[Metric | str] | None = None,
        periods: Sequence[Period | tuple[Period, Period]] | None = None,
        value_types: Sequence[ValueType | str] | None = None,
        period_bases: Sequence[PeriodBasis | str] | None = None,
        accounting_scopes: Sequence[str] | None = None,
        revision_statuses: Sequence[str] | None = None,
        order_by: str = "period",
        descending: bool = False,
        limit: int | None = None,
    ) -> list[FinancialFact]:
        conditions: list[str] = []
        params: list[Any] = []
        if companies:
            normalized_companies = tuple(dict.fromkeys(company for company in companies if company))
            if normalized_companies:
                placeholders = ", ".join("?" for _ in normalized_companies)
                conditions.append(f"company IN ({placeholders})")
                params.extend(normalized_companies)
        if metrics:
            normalized_metrics = tuple(dict.fromkeys(Metric(metric).value for metric in metrics))
            placeholders = ", ".join("?" for _ in normalized_metrics)
            conditions.append(f"metric IN ({placeholders})")
            params.extend(normalized_metrics)
        if periods:
            period_conditions: list[str] = []
            for selector in periods:
                if isinstance(selector, Period):
                    period_conditions.append("(period_kind = ? AND year = ?)")
                    params.extend([selector.kind.value, selector.year])
                    continue
                if not isinstance(selector, tuple) or len(selector) != 2:
                    raise ValueError(f"invalid period selector {selector!r}")
                start, end = selector
                if not isinstance(start, Period) or not isinstance(end, Period) or start.kind != end.kind:
                    raise ValueError(f"period range must use one period kind: {selector!r}")
                if start.year > end.year:
                    raise ValueError(f"period range start must not exceed end: {selector!r}")
                period_conditions.append("(period_kind = ? AND year BETWEEN ? AND ?)")
                params.extend([start.kind.value, start.year, end.year])
            conditions.append("(" + " OR ".join(period_conditions) + ")")
        if value_types:
            normalized_value_types = tuple(dict.fromkeys(ValueType(value_type).value for value_type in value_types))
            placeholders = ", ".join("?" for _ in normalized_value_types)
            conditions.append(f"value_type IN ({placeholders})")
            params.extend(normalized_value_types)
        if period_bases:
            normalized_period_bases = tuple(
                dict.fromkeys(
                    basis.value if isinstance(basis, PeriodBasis) else PeriodBasis(basis).value
                    for basis in period_bases
                )
            )
            placeholders = ", ".join("?" for _ in normalized_period_bases)
            conditions.append(f"period_basis IN ({placeholders})")
            params.extend(normalized_period_bases)
        if accounting_scopes:
            normalized_scopes = tuple(dict.fromkeys(scope.strip() for scope in accounting_scopes if scope.strip()))
            if normalized_scopes:
                placeholders = ", ".join("?" for _ in normalized_scopes)
                conditions.append(f"accounting_scope IN ({placeholders})")
                params.extend(normalized_scopes)
        if revision_statuses:
            normalized_revisions = tuple(
                dict.fromkeys(status.strip() for status in revision_statuses if status.strip())
            )
            if normalized_revisions:
                placeholders = ", ".join("?" for _ in normalized_revisions)
                conditions.append(f"revision_status IN ({placeholders})")
                params.extend(normalized_revisions)
        order_expressions = {
            "company": "company, year, period_kind, metric",
            "metric": "metric, company, year, period_kind",
            "period": "year, period_kind, company, metric",
            "value": "CAST(value AS DOUBLE), company, metric",
            "page": "page, company, metric",
        }
        if order_by not in order_expressions:
            raise ValueError(f"unsupported order_by {order_by!r}")
        direction = "DESC" if descending else "ASC"
        sql = f"SELECT {', '.join(_COLUMNS)} FROM facts"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY " + ", ".join(
            f"{expression.strip()} {direction}" for expression in order_expressions[order_by].split(",")
        )
        if limit is not None:
            if not isinstance(limit, int):
                raise ValueError("limit must be an int or None")
            sql += " LIMIT ?"
            params.append(max(limit, 0))
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
        parquet_path = str(path)
        cursor = self._conn.execute("SELECT * FROM read_parquet(?) LIMIT 0", [parquet_path])
        source_columns = {str(item[0]) for item in cursor.description}
        missing_required = set(_LEGACY_COLUMNS) - source_columns
        if missing_required:
            raise ValueError(f"Parquet snapshot missing required columns: {sorted(missing_required)}")
        select_columns = [
            column if column in source_columns else f"CAST(NULL AS {_V2_COLUMN_DEFINITIONS[column]}) AS {column}"
            for column in _COLUMNS
        ]
        before = self.count()
        self._conn.execute(
            f"INSERT OR IGNORE INTO facts ({', '.join(_COLUMNS)}) "
            f"SELECT {', '.join(select_columns)} FROM read_parquet(?)",
            [parquet_path],
        )
        return self.count() - before
