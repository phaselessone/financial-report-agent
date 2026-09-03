"""Resolve calculation inputs without guessing among financial values.

The public seam is :class:`CalculationOperandResolver`: callers provide one
fully-coordinated requirement plus the facts/evidence already in the run.  The
module returns either one provenance-carrying operand, an ambiguity, or a
missing result; it never silently selects the first number in a row.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
import re
from typing import Any, Literal

from src.agent.calculation_provenance import CalculationOperand


@dataclass(frozen=True)
class OperandRequirement:
    name: str
    entity: str
    metric: str
    period: str
    value_type: str = "actual"
    unit: str | None = None
    period_basis: str | None = None
    accounting_scope: str | None = None
    revision_status: str | None = None


@dataclass(frozen=True)
class ResolvedOperand:
    operand: CalculationOperand
    source: Literal["structured_fact", "evidence", "retrieved_evidence"]


@dataclass(frozen=True)
class AmbiguousOperand:
    candidate_ids: tuple[str, ...]
    reason: str = "ambiguous_operand"


@dataclass(frozen=True)
class MissingOperand:
    reason: str = "missing_operand"
    retrieval_attempted: bool = False


OperandResolution = ResolvedOperand | AmbiguousOperand | MissingOperand
OperandRetriever = Callable[[OperandRequirement], Sequence[Mapping[str, Any]]]

_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "revenue": ("营业收入", "营收", "revenue"),
    "net_profit": ("净利润", "net profit"),
    "gross_profit": ("毛利润", "毛利", "gross profit"),
    "operating_profit": ("营业利润", "operating profit"),
    "gross_margin": ("毛利率", "gross margin"),
    "net_margin": ("净利率", "net margin"),
    "operating_cash_flow": ("经营活动现金流量净额", "经营现金流", "operating cash flow"),
    "rd_expense": ("研发费用", "研发投入", "r&d"),
    "eps": ("每股收益", "eps"),
    "roe": ("净资产收益率", "roe"),
    "roa": ("总资产收益率", "roa"),
}
_NUMBER_RE = re.compile(r"[+-]?\d[\d,]*(?:\.\d+)?")
_FORECAST_RE = re.compile(r"预计|预测|预期|一致预期|指引|forecast|estimate|guidance", re.IGNORECASE)


class _AmbiguousNumericCandidate(ValueError):
    """One evidence row exposes more than one coordinate-compatible value."""


class CalculationOperandResolver:
    """Resolve one calculation requirement against the current run context."""

    def __init__(self, retrieve: OperandRetriever | None = None) -> None:
        self._retrieve = retrieve

    def resolve(
        self,
        requirement: OperandRequirement,
        structured_facts: Sequence[Any],
        evidence_rows: Sequence[Mapping[str, Any]],
    ) -> OperandResolution:
        matches = [fact for fact in structured_facts if _matches_fact(fact, requirement)]
        if matches:
            if len(matches) > 1:
                return AmbiguousOperand(tuple(_fact_id(fact) for fact in matches))
            return _resolved(matches[0], requirement, source="structured_fact")

        evidence_matches: list[Mapping[str, Any]] = []
        ambiguous_evidence: list[str] = []
        for row in evidence_rows:
            try:
                if _matches_evidence(row, requirement):
                    evidence_matches.append(row)
            except _AmbiguousNumericCandidate:
                ambiguous_evidence.append(_candidate_id(row))
        if ambiguous_evidence:
            return AmbiguousOperand(
                tuple(dict.fromkeys(ambiguous_evidence)),
                reason="multiple_numeric_candidates",
            )
        if len(evidence_matches) > 1:
            return AmbiguousOperand(tuple(_candidate_id(row) for row in evidence_matches))
        if evidence_matches:
            return _resolved(evidence_matches[0], requirement, source="evidence")
        if self._retrieve is None:
            return MissingOperand()
        try:
            retrieved_rows = list(self._retrieve(requirement))
        except Exception:  # noqa: BLE001 - external retrieval fails closed
            return MissingOperand(reason="operand_retrieval_failed", retrieval_attempted=True)
        retrieved_matches: list[Mapping[str, Any]] = []
        ambiguous_retrieved: list[str] = []
        for row in retrieved_rows:
            try:
                if _matches_evidence(row, requirement):
                    retrieved_matches.append(row)
            except _AmbiguousNumericCandidate:
                ambiguous_retrieved.append(_candidate_id(row))
        if ambiguous_retrieved:
            return AmbiguousOperand(
                tuple(dict.fromkeys(ambiguous_retrieved)),
                reason="multiple_numeric_candidates",
            )
        if len(retrieved_matches) > 1:
            return AmbiguousOperand(tuple(_candidate_id(row) for row in retrieved_matches))
        if not retrieved_matches:
            return MissingOperand(retrieval_attempted=True)
        return _resolved(retrieved_matches[0], requirement, source="retrieved_evidence")


def _resolved(
    candidate: Any,
    requirement: OperandRequirement,
    *,
    source: Literal["structured_fact", "evidence", "retrieved_evidence"],
) -> ResolvedOperand:
        value, unit, raw_value = _candidate_numeric(candidate, requirement)
        return ResolvedOperand(
            operand=CalculationOperand(
                name=requirement.name,
                value=value,
                unit=unit,
                evidence_id=(
                    _optional_text(_field(candidate, "evidence_id"))
                    or _optional_text(_field(candidate, "chunk_id"))
                ),
                fact_id=_optional_text(_field(candidate, "fact_id")),
                chunk_id=_optional_text(_field(candidate, "chunk_id")),
                doc_id=_optional_text(_field(candidate, "doc_id")),
                raw_text=raw_value,
                period=_period_label(_field(candidate, "period")),
            ),
            source=source,
        )


def _field(item: Any, name: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(name)
    return getattr(item, name, None)


def _enum_text(value: Any) -> str:
    return str(getattr(value, "value", value) or "").strip()


def _period_label(value: Any) -> str:
    if isinstance(value, Mapping):
        return f"{_enum_text(value.get('kind'))}{value.get('year')}"
    label = getattr(value, "label", None)
    return str(label if label is not None else value or "").strip()


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _fact_id(fact: Any) -> str:
    return _optional_text(_field(fact, "fact_id")) or "unknown"


def _candidate_id(candidate: Any) -> str:
    for name in ("fact_id", "evidence_id", "chunk_id"):
        value = _optional_text(_field(candidate, name))
        if value:
            return value
    return "unknown"


def _matches_fact(fact: Any, requirement: OperandRequirement) -> bool:
    if str(_field(fact, "company") or "").strip() != requirement.entity.strip():
        return False
    if _enum_text(_field(fact, "metric")) != requirement.metric.strip():
        return False
    if _period_label(_field(fact, "period")).upper() != requirement.period.strip().upper():
        return False
    if _enum_text(_field(fact, "value_type")).lower() != requirement.value_type.strip().lower():
        return False
    if requirement.unit is not None and str(_field(fact, "unit") or "").strip() != requirement.unit.strip():
        return False
    if requirement.accounting_scope is not None:
        if str(_field(fact, "accounting_scope") or "").strip() != requirement.accounting_scope.strip():
            return False
    if requirement.period_basis is not None:
        if _enum_text(_field(fact, "period_basis")) != requirement.period_basis.strip():
            return False
    if requirement.revision_status is not None:
        if str(_field(fact, "revision_status") or "").strip() != requirement.revision_status.strip():
            return False
    value = _field(fact, "value")
    try:
        return Decimal(str(value)).is_finite()
    except Exception:  # noqa: BLE001 - malformed candidate must simply not match
        return False


def _matches_evidence(row: Mapping[str, Any], requirement: OperandRequirement) -> bool:
    text = _candidate_text(row)
    entity = _optional_text(_field(row, "entity")) or _optional_text(_field(row, "company"))
    if entity is None and requirement.entity.strip() in text:
        entity = requirement.entity.strip()
    if entity != requirement.entity.strip():
        return False
    metric = _enum_text(_field(row, "metric"))
    if not metric and _metric_in_text(requirement.metric, text):
        metric = requirement.metric.strip()
    if metric != requirement.metric.strip():
        return False
    period = _period_label(_field(row, "period")).upper()
    if not period and _period_in_text(requirement.period, text):
        period = requirement.period.strip().upper()
    if period != requirement.period.strip().upper():
        return False
    explicit_value_type = _enum_text(_field(row, "value_type")).lower()
    value_type = explicit_value_type or ("forecast" if _FORECAST_RE.search(text) else "actual")
    if value_type != requirement.value_type.strip().lower():
        return False
    if requirement.accounting_scope is not None:
        scope = str(_field(row, "accounting_scope") or "").strip()
        if not scope and requirement.accounting_scope in text:
            scope = requirement.accounting_scope
        if scope != requirement.accounting_scope.strip():
            return False
    if requirement.period_basis is not None:
        period_basis = _enum_text(_field(row, "period_basis"))
        if period_basis != requirement.period_basis.strip():
            return False
    if requirement.revision_status is not None:
        revision_status = str(_field(row, "revision_status") or "").strip()
        if revision_status != requirement.revision_status.strip():
            return False
    try:
        value, unit, _raw = _candidate_numeric(row, requirement)
        if requirement.unit is not None and unit != requirement.unit.strip():
            return False
        return value.is_finite()
    except _AmbiguousNumericCandidate:
        raise
    except Exception:  # noqa: BLE001
        return False


def _candidate_text(candidate: Any) -> str:
    parts = []
    for name in ("support_span", "source_span", "text", "child_text", "raw_text"):
        value = _optional_text(_field(candidate, name))
        if value and value not in parts:
            parts.append(value)
    return "\n".join(parts)


def _metric_in_text(metric: str, text: str) -> bool:
    lowered = text.lower()
    aliases = _METRIC_ALIASES.get(metric.strip(), (metric.strip(),))
    return any(alias.lower() in lowered for alias in aliases if alias)


def _period_in_text(period: str, text: str) -> bool:
    normalized = period.strip().upper()
    year_match = re.search(r"(?:19|20)\d{2}", normalized)
    if not year_match or year_match.group(0) not in text:
        return False
    if "Q" in normalized:
        quarter = normalized.split("Q", 1)[1][:1]
        markers = {"1": ("Q1", "一季度", "第一季度"), "2": ("Q2", "二季度", "第二季度"), "3": ("Q3", "三季度", "第三季度"), "4": ("Q4", "四季度", "第四季度")}
        return any(marker in text.upper() for marker in markers.get(quarter, ()))
    if "H1" in normalized:
        return "H1" in text.upper() or "上半年" in text
    if "H2" in normalized:
        return "H2" in text.upper() or "下半年" in text
    return True


def _candidate_numeric(candidate: Any, requirement: OperandRequirement) -> tuple[Decimal, str | None, str]:
    explicit_value = _field(candidate, "value")
    explicit_unit = _optional_text(_field(candidate, "unit")) or requirement.unit
    raw_value = _optional_text(_field(candidate, "raw_value"))
    if explicit_value not in (None, ""):
        value = Decimal(str(explicit_value).replace(",", ""))
        return value, explicit_unit, raw_value or f"{explicit_value}{explicit_unit or ''}"
    if raw_value:
        match = _NUMBER_RE.search(raw_value)
        if match:
            return Decimal(match.group(0).replace(",", "")), explicit_unit, raw_value

    text = _candidate_text(candidate)
    year_match = re.search(r"(?:19|20)\d{2}", requirement.period)
    segment = text
    if year_match:
        target = year_match.group(0)
        starts = [match.start() for match in re.finditer(re.escape(target), text)]
        if starts:
            segment = text[starts[0] :]
            segment = re.split(r"[；;。\n]", segment, maxsplit=1)[0]
    aliases = _METRIC_ALIASES.get(requirement.metric, (requirement.metric,))
    alias_pattern = "|".join(re.escape(alias) for alias in sorted(aliases, key=len, reverse=True) if alias)
    unit_pattern = re.escape(requirement.unit) if requirement.unit else r"[^\d\s，,；;。]+"
    pattern = re.compile(
        rf"(?:{alias_pattern})[^；;。\n]{{0,40}}?({_NUMBER_RE.pattern})\s*({unit_pattern})",
        re.IGNORECASE,
    )
    matches = list(pattern.finditer(segment))
    if not matches:
        raise ValueError("candidate has no coordinate-anchored numeric value")
    distinct = list(
        dict.fromkeys(
            (
                Decimal(match.group(1).replace(",", "")),
                match.group(2),
            )
            for match in matches
        )
    )
    if len(distinct) > 1:
        raise _AmbiguousNumericCandidate("candidate has multiple coordinate-anchored numeric values")
    match = matches[0]
    raw = f"{match.group(1)}{match.group(2)}"
    return Decimal(match.group(1).replace(",", "")), match.group(2), raw


__all__ = [
    "AmbiguousOperand",
    "CalculationOperandResolver",
    "MissingOperand",
    "OperandRequirement",
    "OperandResolution",
    "OperandRetriever",
    "ResolvedOperand",
]
