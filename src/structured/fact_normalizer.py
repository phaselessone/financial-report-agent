"""Fact-level normalizers (checklist v3.0 §P5).

``match_metric`` maps Chinese metric keywords to V1 metrics, guarded against
lookalike rate terms (净利率 / 净利润率 must not match 净利润) and multi-metric
ambiguity. ``normalize_company_name`` resolves company mentions through a
caller-supplied alias map (canonical name -> alias substrings), longest alias
wins. Both are pure local functions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from src.structured.schema import Metric

_METRIC_KEYWORDS: tuple[tuple[Metric, tuple[str, ...]], ...] = (
    (
        Metric.OPERATING_CASH_FLOW,
        (
            "经营活动产生的现金流量净额",
            "经营活动现金流量净额",
            "经营性现金流量净额",
            "经营现金流净额",
            "经营性现金流",
        ),
    ),
    (Metric.NET_PROFIT, ("归母净利润", "扣非净利润", "净利润")),
    (Metric.REVENUE, ("营业总收入", "营业收入", "营收")),
    (Metric.RD_EXPENSE, ("研发费用", "研发投入", "研发支出")),
    (Metric.GROSS_MARGIN, ("综合毛利率", "毛利率")),
)

_NET_PROFIT_RATE_GUARDS = ("净利率", "净利润率", "利润率")


def match_metric(text: Any) -> Metric | None:
    """Return the V1 metric named in ``text`` or None.

    Multi-metric mentions are ambiguous and return None; rate lookalikes
    (净利率 / 净利润率) never match net_profit.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    if any(guard in text for guard in _NET_PROFIT_RATE_GUARDS):
        return None
    matched: list[Metric] = []
    for metric, keywords in _METRIC_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            matched.append(metric)
    if len(matched) != 1:
        return None
    return matched[0]


def normalize_company_name(text: Any, aliases: Mapping[str, Sequence[str]]) -> str | None:
    """Resolve the company mentioned in ``text`` through an alias map.

    ``aliases`` maps canonical names to alias substrings; the longest matching
    alias wins. Returns the canonical name or None.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    best: tuple[int, str] | None = None
    for canonical, alias_list in aliases.items():
        for alias in alias_list:
            if alias and alias in text:
                if best is None or len(alias) > best[0]:
                    best = (len(alias), canonical)
    return best[1] if best else None
