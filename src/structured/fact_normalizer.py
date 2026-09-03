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

from src.structured.metric_registry import extract_metrics
from src.structured.schema import Metric

_NET_PROFIT_RATE_GUARDS = ("净利率", "净利润率")


def match_metric(text: Any) -> Metric | None:
    """Return the V1 metric named in ``text`` or None.

    Multi-metric mentions are ambiguous and return None; rate lookalikes
    (净利率 / 净利润率) never match net_profit.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    matched = extract_metrics(text)
    # Rate aliases are now first-class M6 metrics.  The guard only prevents a
    # future/ambiguous net-profit fallback from swallowing them; it must not
    # make ``净利率`` itself unsupported.
    if any(guard in text for guard in _NET_PROFIT_RATE_GUARDS):
        matched = tuple(metric for metric in matched if metric is not Metric.NET_PROFIT)
    return matched[0] if len(matched) == 1 else None


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
