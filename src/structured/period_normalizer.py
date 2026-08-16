"""Period normalizer (checklist v3.0 §P5).

Maps raw period mentions ("2025年一季度", "25Q1", "2025H1", "2025年全年",
"FY2025") to a canonical :class:`~src.structured.schema.Period`. V1 kinds:
FY / Q1 / H1 / Q3. Returns None when the period is ambiguous, out of V1
scope (e.g. Q2 / Q4), a multi-year range, or unresolvable without an anchor
year. Pure local function, no LLM.
"""

from __future__ import annotations

import re

from src.structured.schema import Period, PeriodType
from src.utils.text_utils import normalize_text

_YEAR_RE = re.compile(r"(20\d{2})")
_TWO_DIGIT_YEAR_RE = re.compile(r"(?<!\d)(\d{2})(?=[qQhH]\d)")
_TWO_DIGIT_YEAR_CN_RE = re.compile(r"(?<!\d)(\d{2})\s*年")

_Q1_MARKERS = ("q1", "一季度", "第一季度", "一季报", "1-3月")
_H1_MARKERS = ("h1", "上半年", "半年报", "1-6月")
_Q3_MARKERS = ("q3", "三季度", "第三季度", "前三季度", "三季报", "7-9月")
_FY_MARKERS = ("年度", "财年", "fy", "全年", "年报")

_OUT_OF_SCOPE_MARKERS = (
    "q2",
    "二季度",
    "第二季度",
    "q4",
    "四季度",
    "第四季度",
    "1-9月",
    "1-12月",
)


def _extract_years(compact: str) -> list[int]:
    years = [int(token) for token in _YEAR_RE.findall(compact)]
    for match in _TWO_DIGIT_YEAR_RE.finditer(compact):
        candidate = int(match.group(1))
        if not any(year % 100 == candidate for year in years):
            years.append(2000 + candidate)
    for match in _TWO_DIGIT_YEAR_CN_RE.finditer(compact):
        candidate = int(match.group(1))
        if not any(year % 100 == candidate for year in years):
            years.append(2000 + candidate)
    return sorted(set(years))


def normalize_period(raw: str, *, anchor_year: int | None = None) -> Period | None:
    """Return the canonical :class:`Period` for ``raw`` or None if unresolvable.

    ``anchor_year`` resolves relative mentions ("一季度", "上半年", "全年").
    Multi-year mentions and out-of-V1 kinds (Q2 / Q4) return None.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    compact = re.sub(r"\s+", "", normalize_text(raw)).lower()
    if any(marker in compact for marker in _OUT_OF_SCOPE_MARKERS):
        return None

    years = _extract_years(compact)
    if len(years) > 1:
        return None

    year = years[0] if years else anchor_year
    if year is None:
        return None

    kind: PeriodType | None = None
    if any(marker in compact for marker in _Q1_MARKERS):
        kind = PeriodType.Q1
    elif any(marker in compact for marker in _H1_MARKERS):
        kind = PeriodType.H1
    elif any(marker in compact for marker in _Q3_MARKERS):
        kind = PeriodType.Q3
    elif any(marker in compact for marker in _FY_MARKERS):
        kind = PeriodType.FY
    elif years:
        kind = PeriodType.FY
    if kind is None:
        return None
    try:
        return Period(kind=kind, year=year)
    except ValueError:
        return None
