"""Structured financial fact schema (checklist v3.0 §P5).

:class:`FinancialFact` is the canonical unit of the structured fact layer.
Mandatory provenance (doc_id / page / evidence_id / raw_value / source_span)
is enforced at construction time: a fact without complete provenance can
never become an official fact.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

PROVENANCE_FIELDS: tuple[str, ...] = ("doc_id", "page", "evidence_id", "raw_value", "source_span")

class Metric(str, Enum):
    REVENUE = "revenue"
    NET_PROFIT = "net_profit"
    GROSS_MARGIN = "gross_margin"
    RD_EXPENSE = "rd_expense"
    OPERATING_CASH_FLOW = "operating_cash_flow"
    OPERATING_PROFIT = "operating_profit"
    GROSS_PROFIT = "gross_profit"
    EPS = "eps"
    NET_MARGIN = "net_margin"
    ROE = "roe"
    ROA = "roa"
    TOTAL_ASSETS = "total_assets"
    TOTAL_LIABILITIES = "total_liabilities"
    INVENTORY = "inventory"
    ACCOUNTS_RECEIVABLE = "accounts_receivable"
    INTEREST_BEARING_DEBT = "interest_bearing_debt"
    CAPEX = "capex"
    FREE_CASH_FLOW = "free_cash_flow"


class PeriodType(str, Enum):
    FY = "FY"
    Q1 = "Q1"
    Q2 = "Q2"
    H1 = "H1"
    Q3 = "Q3"


class ValueType(str, Enum):
    ACTUAL = "actual"
    FORECAST = "forecast"
    ADJUSTED = "adjusted"
    UNKNOWN = "unknown"


class PeriodBasis(str, Enum):
    STANDALONE = "standalone"
    CUMULATIVE = "cumulative"


@dataclass(frozen=True)
class Period:
    kind: PeriodType
    year: int

    def __post_init__(self) -> None:
        if not isinstance(self.kind, PeriodType):
            try:
                object.__setattr__(self, "kind", PeriodType(self.kind))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid period kind {self.kind!r}") from exc
        if not isinstance(self.year, int) or not 1990 <= self.year <= 2100:
            raise ValueError(f"invalid period year {self.year!r}")

    @property
    def label(self) -> str:
        return f"{self.kind.value}{self.year}"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "year": self.year}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Period":
        return cls(kind=data["kind"], year=int(data["year"]))


def _coerce_period(value: Period | dict[str, Any]) -> Period:
    if isinstance(value, Period):
        return value
    if isinstance(value, dict):
        return Period.from_dict(value)
    raise ValueError(f"invalid period {value!r}")


def _coerce_optional_date(value: date | str | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return date.fromisoformat(value.strip())
        except ValueError as exc:
            raise ValueError(f"invalid source date {value!r}") from exc
    raise ValueError(f"invalid source date {value!r}")


def _coerce_optional_label(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string or None")
    cleaned = value.strip()
    return cleaned or None


def _coerce_optional_period_basis(value: PeriodBasis | str | None) -> PeriodBasis | None:
    if value is None:
        return None
    if isinstance(value, PeriodBasis):
        return value
    try:
        return PeriodBasis(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid period basis {value!r}") from exc


@dataclass(frozen=True)
class FinancialFact:
    company: str
    metric: Metric
    period: Period
    value_type: ValueType
    value: Decimal
    unit: str
    doc_id: str
    page: int
    evidence_id: str
    raw_value: str
    source_span: str
    accounting_scope: str | None = None
    period_basis: PeriodBasis | None = None
    source_date: date | None = None
    revision_status: str | None = None
    fact_id: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.metric, Metric):
            try:
                object.__setattr__(self, "metric", Metric(self.metric))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid metric {self.metric!r}") from exc
        if not isinstance(self.value_type, ValueType):
            try:
                object.__setattr__(self, "value_type", ValueType(self.value_type))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid value type {self.value_type!r}") from exc
        object.__setattr__(self, "period", _coerce_period(self.period))
        object.__setattr__(
            self,
            "accounting_scope",
            _coerce_optional_label(self.accounting_scope, field_name="accounting_scope"),
        )
        object.__setattr__(self, "period_basis", _coerce_optional_period_basis(self.period_basis))
        object.__setattr__(self, "source_date", _coerce_optional_date(self.source_date))
        object.__setattr__(
            self,
            "revision_status",
            _coerce_optional_label(self.revision_status, field_name="revision_status"),
        )
        if not isinstance(self.value, Decimal):
            try:
                object.__setattr__(self, "value", Decimal(str(self.value)))
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"invalid fact value {self.value!r}") from exc
        if not self.value.is_finite():
            raise ValueError(f"non-finite fact value {self.value!r}")
        from src.structured.metric_registry import metric_allowed_units

        allowed_units = metric_allowed_units(self.metric)
        if self.unit not in allowed_units:
            raise ValueError(f"fact unit must be one of {sorted(allowed_units)}, got {self.unit!r}")
        if not isinstance(self.page, int) or self.page < 0:
            raise ValueError(f"fact page must be a non-negative int, got {self.page!r}")
        missing = missing_provenance_fields(self.to_provenance())
        if missing:
            raise ValueError(f"fact missing mandatory provenance: {', '.join(missing)}")
        if not self.company or not self.company.strip():
            raise ValueError("fact company must be non-empty")
        if not self.fact_id:
            object.__setattr__(self, "fact_id", self.make_fact_id())

    def to_provenance(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in PROVENANCE_FIELDS}

    def make_fact_id(self) -> str:
        key = (
            f"{self.company}|{self.metric.value}|{self.period.label}|"
            f"{self.value_type.value}|{self.value}|{self.unit}|"
            f"{self.doc_id}|{self.page}|{self.evidence_id}"
        )
        coordinates = (
            self.accounting_scope,
            self.period_basis.value if self.period_basis is not None else None,
            self.source_date.isoformat() if self.source_date is not None else None,
            self.revision_status,
        )
        # Preserve deterministic IDs for legacy facts while preventing two v2
        # coordinates from collapsing onto the same identity.
        if any(item is not None for item in coordinates):
            key += "|" + "|".join(item or "" for item in coordinates)
        return "F" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "company": self.company,
            "metric": self.metric.value,
            "period": self.period.to_dict(),
            "value_type": self.value_type.value,
            "value": str(self.value),
            "unit": self.unit,
            "doc_id": self.doc_id,
            "page": self.page,
            "evidence_id": self.evidence_id,
            "raw_value": self.raw_value,
            "source_span": self.source_span,
            "accounting_scope": self.accounting_scope,
            "period_basis": self.period_basis.value if self.period_basis is not None else None,
            "source_date": self.source_date.isoformat() if self.source_date is not None else None,
            "revision_status": self.revision_status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FinancialFact":
        return cls(
            company=data["company"],
            metric=data["metric"],
            period=_coerce_period(data["period"]),
            value_type=data["value_type"],
            value=Decimal(data["value"]),
            unit=data["unit"],
            doc_id=data["doc_id"],
            page=data["page"],
            evidence_id=data["evidence_id"],
            raw_value=data["raw_value"],
            source_span=data["source_span"],
            accounting_scope=data.get("accounting_scope"),
            period_basis=data.get("period_basis"),
            source_date=data.get("source_date"),
            revision_status=data.get("revision_status"),
            fact_id=data.get("fact_id", ""),
        )


def missing_provenance_fields(provenance: dict[str, Any]) -> list[str]:
    """Return the provenance fields missing or empty in ``provenance``."""
    missing: list[str] = []
    for field in PROVENANCE_FIELDS:
        value = provenance.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(field)
    return missing
