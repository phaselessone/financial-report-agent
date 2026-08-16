"""Value type classifier (checklist v3.0 §P5).

Deterministic keyword rules label a fact's source span as actual / forecast
/ unknown. Forecast markers are checked first so forward-looking phrasing
wins over trailing past-tense markers ("预计...同比增长" is a forecast).
Pure local function, no LLM.
"""

from __future__ import annotations

import re
from typing import Any

from src.structured.schema import ValueType

_ESTIMATE_YEAR_RE = re.compile(r"(?<!\d)(?:20)?\d{2}\s*[eE](?!\d)")

_FORECAST_MARKERS = (
    "预计",
    "预期",
    "预测",
    "测算",
    "指引",
    "有望",
    "或达",
    "将实现",
    "将达",
    "计划",
    "展望",
    "目标营收",
)

_ACTUAL_MARKERS = (
    "实现",
    "录得",
    "达到",
    "完成",
    "同比增长",
    "同比下滑",
    "同比减少",
    "同比增加",
    "同比",
    "环比",
    "累计",
)


def classify_value_type(source_text: Any) -> ValueType:
    """Classify ``source_text`` into a :class:`ValueType`.

    Unknown when the span carries no recognizable marker.
    """
    if not isinstance(source_text, str) or not source_text.strip():
        return ValueType.UNKNOWN
    if _ESTIMATE_YEAR_RE.search(source_text):
        return ValueType.FORECAST
    if any(marker in source_text for marker in _FORECAST_MARKERS):
        return ValueType.FORECAST
    if any(marker in source_text for marker in _ACTUAL_MARKERS):
        return ValueType.ACTUAL
    return ValueType.UNKNOWN
