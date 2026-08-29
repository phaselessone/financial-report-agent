from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.structured.schema import Metric, ValueType
from src.utils.text_utils import normalize_for_match


@dataclass(frozen=True)
class MetricSpec:
    canonical_name: str
    aliases: tuple[str, ...]
    dimension: str
    allowed_units: tuple[str, ...]
    value_types: tuple[ValueType, ...]


_SUPPORTED_VALUE_TYPES = (ValueType.ACTUAL, ValueType.FORECAST, ValueType.ADJUSTED)

# Every enum member is bound at the declaration site.  Do not couple registry
# correctness to the iteration order of Metric and a separate tuple.
_METRIC_TO_SPEC: dict[Metric, MetricSpec] = {
    Metric.REVENUE: MetricSpec("营业收入", ("营业收入", "营业总收入", "营收", "收入", "sales revenue"), "currency", ("元",), _SUPPORTED_VALUE_TYPES),
    Metric.NET_PROFIT: MetricSpec("净利润", ("归母净利润", "扣非净利润", "净利润", "net profit"), "currency", ("元",), _SUPPORTED_VALUE_TYPES),
    Metric.GROSS_MARGIN: MetricSpec("毛利率", ("综合毛利率", "毛利率", "gross margin"), "percentage", ("%",), _SUPPORTED_VALUE_TYPES),
    Metric.RD_EXPENSE: MetricSpec("研发费用", ("研发费用", "研发投入", "研发支出", "r&d expense"), "currency", ("元",), _SUPPORTED_VALUE_TYPES),
    Metric.OPERATING_CASH_FLOW: MetricSpec(
        "经营性现金流",
        ("经营活动产生的现金流量净额", "经营活动现金流量净额", "经营性现金流量净额", "经营现金流净额", "经营性现金流", "operating cash flow"),
        "currency",
        ("元",),
        _SUPPORTED_VALUE_TYPES,
    ),
    Metric.OPERATING_PROFIT: MetricSpec("营业利润", ("营业利润", "operating profit"), "currency", ("元",), _SUPPORTED_VALUE_TYPES),
    Metric.GROSS_PROFIT: MetricSpec("毛利", ("毛利", "gross profit"), "currency", ("元",), _SUPPORTED_VALUE_TYPES),
    Metric.EPS: MetricSpec("每股收益", ("每股收益", "每股盈利", "eps"), "per_share", ("元/股", "元每股"), _SUPPORTED_VALUE_TYPES),
    Metric.NET_MARGIN: MetricSpec("净利率", ("净利润率", "净利率", "sales net margin"), "percentage", ("%",), _SUPPORTED_VALUE_TYPES),
    Metric.ROE: MetricSpec("ROE", ("roe", "净资产收益率"), "percentage", ("%",), _SUPPORTED_VALUE_TYPES),
    Metric.ROA: MetricSpec("ROA", ("roa", "总资产收益率"), "percentage", ("%",), _SUPPORTED_VALUE_TYPES),
    Metric.TOTAL_ASSETS: MetricSpec("总资产", ("总资产", "资产总计", "total assets"), "currency", ("元",), _SUPPORTED_VALUE_TYPES),
    Metric.TOTAL_LIABILITIES: MetricSpec("总负债", ("总负债", "负债总计", "total liabilities"), "currency", ("元",), _SUPPORTED_VALUE_TYPES),
    Metric.INVENTORY: MetricSpec("存货", ("存货", "inventory"), "currency", ("元",), _SUPPORTED_VALUE_TYPES),
    Metric.ACCOUNTS_RECEIVABLE: MetricSpec("应收账款", ("应收账款", "accounts receivable"), "currency", ("元",), _SUPPORTED_VALUE_TYPES),
    Metric.INTEREST_BEARING_DEBT: MetricSpec("有息负债", ("有息负债", "带息负债", "interest-bearing debt", "interest bearing debt"), "currency", ("元",), _SUPPORTED_VALUE_TYPES),
    Metric.CAPEX: MetricSpec("资本开支", ("资本开支", "capex", "资本性支出"), "currency", ("元",), _SUPPORTED_VALUE_TYPES),
    Metric.FREE_CASH_FLOW: MetricSpec("自由现金流", ("自由现金流", "fcf", "free cash flow"), "currency", ("元",), _SUPPORTED_VALUE_TYPES),
}

if set(_METRIC_TO_SPEC) != set(Metric):
    missing = set(Metric) - set(_METRIC_TO_SPEC)
    extra = set(_METRIC_TO_SPEC) - set(Metric)
    raise RuntimeError(f"metric registry mismatch: missing={missing}, extra={extra}")


def all_metric_specs() -> tuple[MetricSpec, ...]:
    return tuple(_METRIC_TO_SPEC[metric] for metric in Metric)


def metric_spec(metric: Metric | str) -> MetricSpec:
    return _METRIC_TO_SPEC[Metric(metric)]


def metric_label(metric: Metric | str) -> str:
    return metric_spec(metric).canonical_name


def metric_aliases(metric: Metric | str) -> tuple[str, ...]:
    return metric_spec(metric).aliases


def metric_dimension(metric: Metric | str) -> str:
    return metric_spec(metric).dimension


def metric_allowed_units(metric: Metric | str) -> tuple[str, ...]:
    return metric_spec(metric).allowed_units


def metric_value_types(metric: Metric | str) -> tuple[ValueType, ...]:
    return metric_spec(metric).value_types


def _normalized_aliases() -> tuple[tuple[int, int, Metric, str], ...]:
    items: list[tuple[int, int, Metric, str]] = []
    for order, (metric, spec) in enumerate(_METRIC_TO_SPEC.items()):
        for alias in spec.aliases:
            normalized = normalize_for_match(alias).lower()
            if normalized:
                items.append((len(normalized), order, metric, normalized))
    return tuple(sorted(items, key=lambda item: (-item[0], item[1], item[3])))


_ALIAS_INDEX = _normalized_aliases()


def extract_metrics(text: str | object) -> tuple[Metric, ...]:
    if not isinstance(text, str) or not text.strip():
        return ()
    normalized = normalize_for_match(text).lower()
    if not normalized:
        return ()
    matches: list[tuple[int, int, Metric, str]] = []
    for alias_length, order, metric, alias in _ALIAS_INDEX:
        if alias in normalized:
            matches.append((alias_length, order, metric, alias))
    if not matches:
        return ()
    ordered: list[Metric] = []
    seen: set[Metric] = set()
    # Prefer the longest alias at a given textual position.  This prevents the
    # nested alias ``毛利`` from making ``毛利率`` look like two metrics while
    # still allowing genuine multi-metric queries such as ``毛利和营收``.
    dominated: set[tuple[Metric, str]] = set()
    for length, _order, metric, alias in matches:
        for other_length, _other_order, other_metric, other_alias in matches:
            if metric != other_metric and length < other_length and alias in other_alias:
                dominated.add((metric, alias))
    for _alias_length, _order, metric, alias in sorted(matches, key=lambda item: (-item[0], item[1], item[2].value)):
        if (metric, alias) in dominated:
            continue
        if metric not in seen:
            seen.add(metric)
            ordered.append(metric)
    return tuple(ordered)


def match_metric(text: str | object) -> Metric | None:
    metrics = extract_metrics(text)
    return metrics[0] if len(metrics) == 1 else None


def metrics_compatible(metrics: Iterable[Metric | str]) -> bool:
    resolved = [Metric(metric) for metric in metrics]
    if len(resolved) <= 1:
        return True
    dimensions = {metric_dimension(metric) for metric in resolved}
    if len(dimensions) != 1:
        return False
    allowed_units = set(metric_allowed_units(resolved[0]))
    for metric in resolved[1:]:
        allowed_units &= set(metric_allowed_units(metric))
    return bool(allowed_units)
