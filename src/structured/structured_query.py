from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Iterable, Literal

from src.structured.fact_normalizer import normalize_company_name
from src.structured.metric_registry import (
    extract_metrics,
    metric_allowed_units,
    metric_label,
    metrics_compatible,
)
from src.structured.period_normalizer import normalize_period
from src.structured.schema import FinancialFact, Metric, Period, PeriodBasis, PeriodType, ValueType
from src.structured.fact_store import FactStore
from src.utils.text_utils import normalize_text

StructuredOperation = Literal["lookup", "compare", "trend", "calculate"]
CalculationKind = Literal["yoy", "qoq", "ratio", "change"]
PeriodSelection = Period | tuple[Period, Period]


@dataclass(frozen=True)
class StructuredQuery:
    entities: tuple[str, ...]
    metrics: tuple[Metric, ...]
    periods: tuple[PeriodSelection, ...]
    operation: StructuredOperation
    value_type: ValueType | None = ValueType.ACTUAL
    period_basis: PeriodBasis | None = None
    calculation_kind: CalculationKind | None = None
    accounting_scope: str | None = None
    revision_status: str | None = None


@dataclass(frozen=True)
class OperandRequirement:
    requirement_id: str
    name: str
    entity: str
    metric: Metric
    period: Period
    value_type: ValueType | None
    unit: str
    period_basis: PeriodBasis | None = None
    accounting_scope: str | None = None
    revision_status: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "requirement_id": self.requirement_id,
            "name": self.name,
            "entity": self.entity,
            "metric": self.metric.value,
            "period": self.period.to_dict(),
            "value_type": self.value_type.value if self.value_type is not None else None,
            "unit": self.unit,
            "period_basis": self.period_basis.value if self.period_basis is not None else None,
            "accounting_scope": self.accounting_scope,
            "revision_status": self.revision_status,
        }


@dataclass(frozen=True)
class StructuredQueryResult:
    query: StructuredQuery
    facts: tuple[FinancialFact, ...]
    status: str
    reason: str | None = None
    operand_requirements: tuple[OperandRequirement, ...] = ()


def extract_companies(text: str | object, aliases: Mapping[str, Sequence[str]]) -> tuple[str, ...]:
    if not isinstance(text, str) or not text.strip():
        return ()
    found: list[tuple[int, str]] = []
    for canonical, alias_list in aliases.items():
        longest = 0
        for alias in alias_list:
            if alias and alias in text:
                longest = max(longest, len(alias))
        if longest:
            found.append((longest, canonical))
    return tuple(company for _length, company in sorted(found, key=lambda item: (-item[0], item[1])))


def extract_periods(text: str | object, *, anchor_year: int | None = None) -> tuple[PeriodSelection, ...]:
    if not isinstance(text, str) or not text.strip():
        return ()
    normalized = normalize_text(text)
    if any(marker in normalized for marker in ("近三年", "过去三年", "最近三年")) and anchor_year is not None:
        return ((Period("FY", anchor_year - 2), Period("FY", anchor_year)),)
    markers = ("FY", "Q1", "Q2", "H1", "Q3", "季度", "全年", "年", "半年", "同比", "较", "相比")
    periods: list[PeriodSelection] = []
    fragments = [normalized]
    for splitter in ("到", "至", "-", "和", "及", "、", "相比", "较"):
        next_fragments: list[str] = []
        for fragment in fragments:
            next_fragments.extend(part for part in fragment.split(splitter) if part)
        fragments = next_fragments
    for fragment in fragments:
        if not any(marker in fragment for marker in markers):
            continue
        period = normalize_period(fragment, anchor_year=anchor_year)
        if period is not None and period not in periods:
            periods.append(period)
    if not periods:
        parsed = normalize_period(normalized, anchor_year=anchor_year)
        if parsed is not None:
            periods.append(parsed)
    return tuple(periods)


def _normalize_entities(entities: Iterable[str]) -> tuple[str, ...]:
    cleaned = tuple(entity.strip() for entity in entities if isinstance(entity, str) and entity.strip())
    return cleaned


def _normalize_metrics(metrics: Iterable[Metric | str]) -> tuple[Metric, ...]:
    # Lookup/compare/trend queries may intentionally mix dimensions (for
    # example revenue + net margin).  Dimension compatibility is a property of
    # an arithmetic operation, not of selecting several indicators.  The
    # calculate operation applies the stricter check in build_structured_query.
    return tuple(Metric(metric) for metric in metrics)


def _normalize_periods(
    periods: Iterable[Period | tuple[Period, Period] | dict[str, int] | str],
    *,
    anchor_year: int | None = None,
) -> tuple[PeriodSelection, ...]:
    resolved: list[PeriodSelection] = []
    for period in periods:
        if isinstance(period, Period):
            resolved.append(period)
            continue
        if isinstance(period, tuple) and len(period) == 2 and all(isinstance(item, Period) for item in period):
            resolved.append((period[0], period[1]))
            continue
        if isinstance(period, dict):
            resolved.append(Period.from_dict(period))
            continue
        parsed = normalize_period(period, anchor_year=anchor_year)
        if parsed is None:
            normalized = normalize_text(period)
            range_match = re.search(r"(20\d{2})\s*[-~至到]\s*(20\d{2})\s*年?", normalized)
            if range_match:
                start_year = int(range_match.group(1))
                end_year = int(range_match.group(2))
                resolved.append((Period("FY", start_year), Period("FY", end_year)))
                continue
            if any(marker in normalized for marker in ("近三年", "过去三年", "最近三年")):
                if anchor_year is None:
                    raise ValueError(f"relative period {period!r} requires anchor_year")
                resolved.append((Period("FY", anchor_year - 2), Period("FY", anchor_year)))
                continue
            raise ValueError(f"unresolvable period {period!r}")
        resolved.append(parsed)
    return tuple(resolved)


def build_structured_query(
    *,
    entities: Iterable[str],
    metrics: Iterable[Metric | str],
    periods: Iterable[Period | tuple[Period, Period] | dict[str, int] | str],
    operation: StructuredOperation,
    anchor_year: int | None = None,
    value_type: ValueType | str | None = ValueType.ACTUAL,
    period_basis: PeriodBasis | str | None = None,
    calculation_kind: CalculationKind | None = None,
    accounting_scope: str | None = None,
    revision_status: str | None = None,
) -> StructuredQuery:
    if operation not in {"lookup", "compare", "trend", "calculate"}:
        raise ValueError(f"invalid structured operation {operation!r}")
    normalized_entities = _normalize_entities(entities)
    normalized_metrics = _normalize_metrics(metrics)
    normalized_periods = _normalize_periods(periods, anchor_year=anchor_year)
    if not normalized_entities:
        raise ValueError("structured query requires at least one entity")
    if not normalized_metrics:
        raise ValueError("structured query requires at least one metric")
    if not normalized_periods:
        raise ValueError("structured query requires at least one period")
    if operation == "calculate" and not metrics_compatible(normalized_metrics):
        raise ValueError("calculation metrics are not dimension-compatible")
    if calculation_kind not in {None, "yoy", "qoq", "ratio", "change"}:
        raise ValueError(f"invalid calculation kind {calculation_kind!r}")
    if operation != "calculate" and calculation_kind is not None:
        raise ValueError("calculation_kind is only valid for calculate queries")
    if value_type is None:
        normalized_value_type = None
    else:
        normalized_value_type = value_type if isinstance(value_type, ValueType) else ValueType(value_type)
    if period_basis is None:
        normalized_period_basis = None
    else:
        normalized_period_basis = period_basis if isinstance(period_basis, PeriodBasis) else PeriodBasis(period_basis)
    normalized_scope = accounting_scope.strip() if isinstance(accounting_scope, str) and accounting_scope.strip() else None
    normalized_revision = revision_status.strip() if isinstance(revision_status, str) and revision_status.strip() else None
    return StructuredQuery(
        entities=normalized_entities,
        metrics=normalized_metrics,
        periods=normalized_periods,
        operation=operation,
        value_type=normalized_value_type,
        period_basis=normalized_period_basis,
        calculation_kind=calculation_kind,
        accounting_scope=normalized_scope,
        revision_status=normalized_revision,
    )


def _expand_calculation_periods(
    periods: tuple[PeriodSelection, ...],
    calculation_kind: CalculationKind | None,
) -> tuple[PeriodSelection, ...]:
    if calculation_kind is None or len(periods) != 1 or not isinstance(periods[0], Period):
        return periods
    current = periods[0]
    if calculation_kind == "yoy":
        return (Period(current.kind, current.year - 1), current)
    if calculation_kind == "qoq":
        previous_by_kind = {
            "Q2": Period("Q1", current.year),
            "Q3": Period("Q2", current.year),
        }
        previous = previous_by_kind.get(current.kind.value)
        if previous is not None:
            return (previous, current)
    return periods


def parse_structured_query(
    text: str,
    *,
    company_aliases: Mapping[str, Sequence[str]],
    anchor_year: int | None = None,
) -> StructuredQuery | None:
    if not isinstance(text, str) or not text.strip():
        return None
    entities = extract_companies(text, company_aliases)
    if not entities:
        fallback = normalize_company_name(text, company_aliases)
        if fallback is not None:
            entities = (fallback,)
    metrics = extract_metrics(text)
    periods = extract_periods(text, anchor_year=anchor_year)
    if not entities or not metrics or not periods:
        return None
    normalized = normalize_text(text)
    if any(marker in normalized for marker in ("趋势", "变化", "走势")):
        operation: StructuredOperation = "trend"
    elif any(marker in normalized for marker in ("比较", "对比", "高于", "低于")):
        operation = "compare"
    elif any(marker in normalized for marker in ("增速", "同比", "环比", "增长率", "占比", "比例")):
        operation = "calculate"
    else:
        operation = "lookup"
    calculation_kind: CalculationKind | None = None
    if operation == "calculate":
        if "环比" in normalized:
            calculation_kind = "qoq"
        elif any(marker in normalized for marker in ("同比", "增速", "增长率")):
            calculation_kind = "yoy"
        elif any(marker in normalized for marker in ("占比", "比例")):
            calculation_kind = "ratio"
        else:
            calculation_kind = "change"
        periods = _expand_calculation_periods(periods, calculation_kind)
    if any(marker in normalized.lower() for marker in ("经调整", "调整后", "non-gaap", "adjusted")):
        value_type = ValueType.ADJUSTED
    elif any(marker in normalized for marker in ("预计", "预期", "预测", "指引", "目标")):
        value_type = ValueType.FORECAST
    else:
        value_type = ValueType.ACTUAL
    period_basis = (
        PeriodBasis.STANDALONE
        if any(marker in normalized for marker in ("单季", "单季度"))
        else None
    )
    if any(marker in normalized for marker in ("合并口径", "合并报表")):
        accounting_scope = "consolidated"
    elif any(marker in normalized for marker in ("母公司口径", "母公司报表")):
        accounting_scope = "parent"
    else:
        accounting_scope = None
    if any(marker in normalized for marker in ("重述", "追溯调整")):
        revision_status = "restated"
    elif any(marker in normalized for marker in ("原始披露", "初始披露")):
        revision_status = "original"
    else:
        revision_status = None
    try:
        return build_structured_query(
            entities=entities,
            metrics=metrics,
            periods=periods,
            operation=operation,
            anchor_year=anchor_year,
            value_type=value_type,
            period_basis=period_basis,
            calculation_kind=calculation_kind,
            accounting_scope=accounting_scope,
            revision_status=revision_status,
        )
    except ValueError:
        return None


def _match_period(fact_period: Period, selector: PeriodSelection) -> bool:
    if isinstance(selector, Period):
        return fact_period == selector
    start, end = selector
    if start.kind != end.kind or fact_period.kind != start.kind:
        return False
    return start.year <= fact_period.year <= end.year


def _match_fact(fact: FinancialFact, *, query: StructuredQuery) -> bool:
    if fact.company not in query.entities:
        return False
    if fact.metric not in query.metrics:
        return False
    if not any(_match_period(fact.period, selector) for selector in query.periods):
        return False
    if query.value_type is not None and fact.value_type is not query.value_type:
        return False
    if query.period_basis is not None and fact.period_basis is not query.period_basis:
        return False
    if query.accounting_scope is not None and fact.accounting_scope != query.accounting_scope:
        return False
    if query.revision_status is not None and fact.revision_status != query.revision_status:
        return False
    if fact.unit not in metric_allowed_units(fact.metric):
        return False
    return True


def _supports_trend_periods(periods: tuple[PeriodSelection, ...]) -> bool:
    if len(periods) >= 2:
        return True
    if len(periods) == 1 and isinstance(periods[0], tuple):
        return True
    return False


def _expand_period_selector(selector: PeriodSelection) -> tuple[Period, ...]:
    if isinstance(selector, Period):
        return (selector,)
    start, end = selector
    if start.kind != end.kind or start.year > end.year:
        return ()
    return tuple(Period(start.kind, year) for year in range(start.year, end.year + 1))


def _build_operand_requirements(query: StructuredQuery) -> tuple[OperandRequirement, ...]:
    periods = tuple(period for selector in query.periods for period in _expand_period_selector(selector))
    requirements: list[OperandRequirement] = []
    for entity in query.entities:
        for metric in query.metrics:
            for period_index, period in enumerate(periods):
                if query.calculation_kind in {"yoy", "qoq"} and len(periods) == 2:
                    name = "prior" if period_index == 0 else "current"
                elif len(query.metrics) > 1 and len(periods) == 1:
                    name = metric.value
                else:
                    name = f"operand_{len(requirements) + 1}"
                identity = "|".join(
                    (
                        entity,
                        metric.value,
                        period.label,
                        query.value_type.value if query.value_type is not None else "any",
                        query.period_basis.value if query.period_basis is not None else "",
                        query.accounting_scope or "",
                        query.revision_status or "",
                    )
                )
                requirements.append(
                    OperandRequirement(
                        requirement_id="OR" + hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12],
                        name=name,
                        entity=entity,
                        metric=metric,
                        period=period,
                        value_type=query.value_type,
                        unit=metric_allowed_units(metric)[0],
                        period_basis=query.period_basis,
                        accounting_scope=query.accounting_scope,
                        revision_status=query.revision_status,
                    )
                )
    return tuple(requirements)


def execute_structured_query(store: FactStore, query: StructuredQuery) -> StructuredQueryResult:
    if any(
        period.kind is PeriodType.Q2
        for selector in query.periods
        for period in _expand_period_selector(selector)
    ) and query.period_basis is None:
        return StructuredQueryResult(query=query, facts=(), status="fail_closed", reason="ambiguous_period_basis")
    if query.operation == "calculate":
        requirements = _build_operand_requirements(query)
        if len(requirements) < 2:
            return StructuredQueryResult(
                query=query,
                facts=(),
                status="fail_closed",
                reason="insufficient_operand_requirements",
                operand_requirements=requirements,
            )
        return StructuredQueryResult(
            query=query,
            facts=(),
            status="ok",
            operand_requirements=requirements,
        )
    value_types = None if query.value_type is None else [query.value_type]
    period_bases = None if query.period_basis is None else [query.period_basis.value]
    accounting_scopes = None if query.accounting_scope is None else [query.accounting_scope]
    revision_statuses = None if query.revision_status is None else [query.revision_status]
    filtered = tuple(
        fact
        for fact in store.query_many(
            companies=query.entities,
            metrics=query.metrics,
            periods=query.periods,
            value_types=value_types,
            period_bases=period_bases,
            accounting_scopes=accounting_scopes,
            revision_statuses=revision_statuses,
            order_by="period",
        )
        if _match_fact(fact, query=query)
    )
    if not filtered:
        return StructuredQueryResult(query=query, facts=(), status="miss", reason="no matching fact")
    if query.operation == "compare" and len(query.entities) < 2:
        return StructuredQueryResult(query=query, facts=filtered, status="fail_closed", reason="compare requires at least two entities")
    if query.operation == "trend" and not _supports_trend_periods(query.periods):
        return StructuredQueryResult(query=query, facts=filtered, status="fail_closed", reason="trend requires a period range")
    return StructuredQueryResult(query=query, facts=filtered, status="ok")


def summarize_structured_result(result: StructuredQueryResult) -> dict[str, object]:
    return {
        "operation": result.query.operation,
        "status": result.status,
        "reason": result.reason,
        "count": len(result.facts),
        "operand_requirement_count": len(result.operand_requirements),
        "metrics": [metric_label(metric) for metric in result.query.metrics],
    }
