"""Structured financial fact schema (checklist v3.0 §P5).

:class:`FinancialFact` is the canonical unit of the structured fact layer.
Mandatory provenance (doc_id / page / evidence_id / raw_value / source_span)
is enforced at construction time: a fact without complete provenance can
never become an official fact.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any

PROVENANCE_FIELDS: tuple[str, ...] = ("doc_id", "page", "evidence_id", "raw_value", "source_span")

_UNITS = frozenset({"元", "%"})


class Metric(str, Enum):
    REVENUE = "revenue"
    NET_PROFIT = "net_profit"
    GROSS_MARGIN = "gross_margin"
    RD_EXPENSE = "rd_expense"
    OPERATING_CASH_FLOW = "operating_cash_flow"


class PeriodType(str, Enum):
    FY = "FY"
    Q1 = "Q1"
    H1 = "H1"
    Q3 = "Q3"


class ValueType(str, Enum):
    ACTUAL = "actual"
    FORECAST = "forecast"
    UNKNOWN = "unknown"


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
        if not isinstance(self.value, Decimal):
            try:
                object.__setattr__(self, "value", Decimal(str(self.value)))
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"invalid fact value {self.value!r}") from exc
        if not self.value.is_finite():
            raise ValueError(f"non-finite fact value {self.value!r}")
        if self.unit not in _UNITS:
            raise ValueError(f"fact unit must be one of {sorted(_UNITS)}, got {self.unit!r}")
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
