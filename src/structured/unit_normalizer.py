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

from src.structured.metric_registry import metric_dimension
from src.structured.schema import Metric
from src.tools.financial_calculator import CalculationError, parse_numeric_value

_CURRENCY_UNITS = frozenset({"亿", "万", "千"})
_PER_SHARE_RE = re.compile(r"^\s*([+-]?\d[\d,]*(?:\.\d+)?)\s*(?:元\s*(?:/|／)\s*股|元每股)\s*$")

_FOREIGN_CURRENCY_RE = re.compile(r"(美元|港元|港币|澳门元|欧元|日元|韩元|新台币)")


def is_margin_metric(metric: Metric | str) -> bool:
    try:
        return metric_dimension(Metric(metric)) == "percentage"
    except ValueError:
        return False


def is_currency_metric(metric: Metric | str) -> bool:
    try:
        return metric_dimension(Metric(metric)) == "currency"
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
    if metric_dimension(metric_value) == "currency" and _FOREIGN_CURRENCY_RE.search(raw_value):
        return None
    if metric_dimension(metric_value) == "per_share":
        match = _PER_SHARE_RE.match(str(raw_value or ""))
        if not match:
            return None
        try:
            value = Decimal(match.group(1).replace(",", ""))
        except Exception:
            return None
        if not value.is_finite():
            return None
        return value, "元/股"
    try:
        value, _scale, unit_label = parse_numeric_value(raw_value)
    except (CalculationError, TypeError):
        return None
    if metric_dimension(metric_value) == "currency":
        if unit_label not in _CURRENCY_UNITS:
            return None
        return value, "元"
    if metric_dimension(metric_value) == "percentage":
        if unit_label not in (None, "%"):
            return None
        return value, "%"
    return None
