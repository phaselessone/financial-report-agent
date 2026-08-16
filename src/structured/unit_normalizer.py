"""Fact value/unit normalizer (checklist v3.0 §P5).

Reuses the P4 calculator's :func:`parse_numeric_value` and enforces
metric-dimension consistency: currency metrics (revenue / net_profit /
rd_expense / operating_cash_flow) must carry a currency unit and normalize
to base yuan; the margin metric keeps the percent number. Anything else
(percent for currency, currency for margin, bare currency numbers, unknown
metrics) returns None. Pure local function, no LLM.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Any

from src.structured.schema import Metric
from src.tools.financial_calculator import CalculationError, parse_numeric_value

_CURRENCY_METRICS = frozenset(
    {
        Metric.REVENUE,
        Metric.NET_PROFIT,
        Metric.RD_EXPENSE,
        Metric.OPERATING_CASH_FLOW,
    }
)
_CURRENCY_UNITS = frozenset({"亿", "万", "千"})

_FOREIGN_CURRENCY_RE = re.compile(r"(美元|港元|港币|澳门元|欧元|日元|韩元|新台币)")


def is_margin_metric(metric: Metric | str) -> bool:
    try:
        return Metric(metric) is Metric.GROSS_MARGIN
    except ValueError:
        return False


def is_currency_metric(metric: Metric | str) -> bool:
    try:
        return Metric(metric) in _CURRENCY_METRICS
    except ValueError:
        return False


def normalize_fact_value(raw_value: str, metric: Metric | str) -> tuple[Decimal, str] | None:
    """Normalize ``raw_value`` for ``metric`` into (value, unit).

    Returns None on any dimension mismatch or unparseable input.
    """
    try:
        metric_value = Metric(metric)
    except (ValueError, TypeError):
        return None
    if metric_value in _CURRENCY_METRICS and _FOREIGN_CURRENCY_RE.search(raw_value):
        return None
    try:
        value, _scale, unit_label = parse_numeric_value(raw_value)
    except (CalculationError, TypeError):
        return None
    if metric_value in _CURRENCY_METRICS:
        if unit_label not in _CURRENCY_UNITS:
            return None
        return value, "元"
    if metric_value is Metric.GROSS_MARGIN:
        if unit_label not in (None, "%"):
            return None
        return value, "%"
    return None
