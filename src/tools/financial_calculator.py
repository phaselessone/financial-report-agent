"""Local deterministic financial calculator (checklist v3.0 §P4).

Pure ``decimal.Decimal`` arithmetic with unit normalization and evidence
provenance passthrough. This module contains no LLM calls and no dynamic
code execution of any kind (checklist §P4 Forbidden).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any

from src.utils.text_utils import normalize_text

# (token, label, scale) — longest tokens first so "亿元" wins over "亿".
_UNIT_TOKENS: tuple[tuple[str, str, int], ...] = (
    ("亿元", "亿", 10**8),
    ("万元", "万", 10**4),
    ("千元", "千", 10**3),
    ("亿", "亿", 10**8),
    ("万", "万", 10**4),
    ("千", "千", 10**3),
    ("%", "%", 1),
)
_NUMBER_RE = re.compile(r"[+-]?\d[\d,]*(?:\.\d+)?")
_UNIT_SUFFIX_RE = re.compile(r"[A-Za-z\u4e00-\u9fff%]+")
_FULLWIDTH_TABLE = str.maketrans("０１２３４５６７８９．，", "0123456789.,")


class CalculationError(Exception):
    """Raised when a calculation input is missing, ambiguous, or invalid."""


@dataclass(frozen=True)
class CalculationInput:
    """One operand with its provenance.

    ``value`` is already normalized to the base unit (currency values are in
    yuan; percent values keep the percent number, e.g. "12.5%" -> 12.5).
    """

    value: Decimal
    raw_text: str
    unit: str | None
    scale: int
    evidence_id: str | None = None
    chunk_id: str | None = None
    doc_id: str | None = None

    @classmethod
    def from_raw(
        cls,
        raw_text: str,
        *,
        evidence_id: str | None = None,
        chunk_id: str | None = None,
        doc_id: str | None = None,
    ) -> "CalculationInput":
        value, scale, unit = parse_numeric_value(raw_text)
        return cls(
            value=value,
            raw_text=raw_text,
            unit=unit,
            scale=scale,
            evidence_id=evidence_id,
            chunk_id=chunk_id,
            doc_id=doc_id,
        )


@dataclass(frozen=True)
class CalculationResult:
    """Deterministic calculation result.

    ``value`` is the quantized result; for ratio-family operations it is the
    FRACTION (e.g. 0.15) with ``formatted`` the percent string ("15.0000%").
    ``provenance`` keeps one dict per input in order, with the same keys:
    evidence_id / chunk_id / doc_id / raw_text (None where absent).
    """

    operation: str
    inputs: tuple[CalculationInput, ...]
    value: Decimal
    formatted: str
    unit: str | None
    formula: str
    provenance: tuple[dict[str, Any], ...]


def parse_numeric_value(raw_text: str) -> tuple[Decimal, int, str | None]:
    """Parse ``raw_text`` into (value_in_base_unit, scale, unit_label).

    Accepts currency suffixes (亿/万/千, with optional 元), percent ("%"), and
    bare numbers; tolerates commas, spaces, and full-width digits/punctuation.
    Raises :class:`CalculationError` when the text has no number, more than one
    numeric candidate, a non-finite value, or an unrecognized unit suffix.
    """
    if not isinstance(raw_text, str):
        raise CalculationError(f"expected str raw_text, got {type(raw_text).__name__}")
    normalized = normalize_text(raw_text).translate(_FULLWIDTH_TABLE)
    candidates = _NUMBER_RE.findall(normalized)
    if not candidates:
        raise CalculationError(f"no numeric value found in {raw_text!r}")
    if len(candidates) > 1:
        raise CalculationError(f"ambiguous input: multiple numeric values in {raw_text!r}")
    candidate = candidates[0]
    cleaned = candidate.replace(",", "")
    try:
        value = Decimal(cleaned)
    except InvalidOperation as exc:
        raise CalculationError(f"unparseable numeric value {candidate!r} in {raw_text!r}") from exc
    if not value.is_finite():
        raise CalculationError(f"non-finite numeric value in {raw_text!r}")

    suffix_match = _UNIT_SUFFIX_RE.match(normalized, normalized.index(candidate) + len(candidate))
    suffix = suffix_match.group(0) if suffix_match else ""
    if not suffix:
        return value, 1, None
    token = next((item for item in _UNIT_TOKENS if suffix.startswith(item[0])), None)
    if token is None:
        raise CalculationError(f"unknown unit suffix {suffix!r} in {raw_text!r}")
    _token_text, label, scale = token
    return value * scale, scale, label


def input_from_row(row: dict[str, Any], *, prefer: str = "support_span") -> CalculationInput:
    """Build a :class:`CalculationInput` from a retrieval/evidence row.

    Reads the numeric value from ``prefer`` (default support_span), falling
    back to child_text and then text, and carries the row's evidence_id,
    chunk_id, and doc_id (each may be absent).
    """
    raw_text = str(row.get(prefer) or row.get("child_text") or row.get("text", ""))
    return CalculationInput.from_raw(
        raw_text,
        evidence_id=row.get("evidence_id"),
        chunk_id=row.get("chunk_id"),
        doc_id=row.get("doc_id"),
    )


def _quantize(value: Decimal, precision: int) -> Decimal:
    if not isinstance(precision, int) or precision < 0:
        raise CalculationError(f"precision must be a non-negative int, got {precision!r}")
    return value.quantize(Decimal(1).scaleb(-precision), rounding=ROUND_HALF_UP)


def _as_input(item: CalculationInput | str | int | float | Decimal) -> CalculationInput:
    if isinstance(item, CalculationInput):
        return item
    if isinstance(item, str):
        return CalculationInput.from_raw(item)
    if isinstance(item, (int, float, Decimal)):
        return CalculationInput(value=Decimal(str(item)), raw_text=str(item), unit=None, scale=1)
    raise TypeError(f"unsupported input type {type(item).__name__}")


def _checked(*items: CalculationInput | str | int | float | Decimal) -> tuple[CalculationInput, ...]:
    inputs = tuple(_as_input(item) for item in items)
    for item in inputs:
        if not item.value.is_finite():
            raise CalculationError(f"non-finite input value in {item.raw_text!r}")
    return inputs


def _provenance(inputs: tuple[CalculationInput, ...]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "evidence_id": item.evidence_id,
            "chunk_id": item.chunk_id,
            "doc_id": item.doc_id,
            "raw_text": item.raw_text,
        }
        for item in inputs
    )


def _require_compatible_dimension(left: CalculationInput, right: CalculationInput, op_name: str) -> None:
    """Reject operations mixing percent values with currency/bare values."""
    if (left.unit == "%") != (right.unit == "%"):
        raise CalculationError(f"{op_name} requires matching unit dimensions (percent vs currency)")


def _ratio_result(
    *,
    operation: str,
    inputs: tuple[CalculationInput, ...],
    fraction: Decimal,
    precision: int,
    formula: str,
) -> CalculationResult:
    value = _quantize(fraction, precision)
    percent = _quantize(fraction * 100, precision)
    return CalculationResult(
        operation=operation,
        inputs=inputs,
        value=value,
        formatted=f"{percent}%",
        unit=None,
        formula=formula,
        provenance=_provenance(inputs),
    )


def growth_rate(start, end, *, precision: int = 4) -> CalculationResult:
    """(end - start) / start as a fraction; ``formatted`` is the percent string.

    Both operands must share a unit dimension (percent vs percent, or
    currency/bare values).
    """
    left, right = _checked(start, end)
    _require_compatible_dimension(left, right, "growth_rate")
    if left.value == 0:
        raise CalculationError("growth_rate start must be non-zero")
    fraction = (right.value - left.value) / left.value
    return _ratio_result(
        operation="growth_rate",
        inputs=(left, right),
        fraction=fraction,
        precision=precision,
        formula=f"({right.value}-{left.value})/{left.value}",
    )


def yoy(prior, current, *, precision: int = 4) -> CalculationResult:
    """Year-over-year growth: (current - prior) / prior (semantic alias).

    Both operands must share a unit dimension (percent vs percent, or
    currency/bare values).
    """
    left, right = _checked(prior, current)
    _require_compatible_dimension(left, right, "yoy")
    if left.value == 0:
        raise CalculationError("yoy prior value must be non-zero")
    fraction = (right.value - left.value) / left.value
    return _ratio_result(
        operation="yoy",
        inputs=(left, right),
        fraction=fraction,
        precision=precision,
        formula=f"({right.value}-{left.value})/{left.value}",
    )


def cagr(start, end, years, *, precision: int = 4) -> CalculationResult:
    """Compound annual growth rate: (end/start)^(1/years) - 1.

    Requires start > 0, end > 0, and years > 0, and matching unit dimensions
    (percent vs percent, or currency/bare values). Uses Decimal power, so the
    result is exact under the active decimal context.
    """
    left, right = _checked(start, end)
    _require_compatible_dimension(left, right, "cagr")
    if not isinstance(years, (int, float, Decimal)):
        raise TypeError(f"years must be a number, got {type(years).__name__}")
    years_value = Decimal(str(years))
    if not years_value.is_finite() or years_value <= 0:
        raise CalculationError("cagr years must be positive")
    if left.value <= 0:
        raise CalculationError("cagr start must be positive")
    if right.value <= 0:
        raise CalculationError("cagr end must be positive")
    fraction = (right.value / left.value) ** (Decimal(1) / years_value) - 1
    return _ratio_result(
        operation="cagr",
        inputs=(left, right),
        fraction=fraction,
        precision=precision,
        formula=f"({right.value}/{left.value})^(1/{years_value})-1",
    )


def gross_margin(gross_profit, revenue, *, precision: int = 4) -> CalculationResult:
    """gross_profit / revenue as a fraction; revenue must be positive.

    Both operands must share a unit dimension (percent vs percent, or
    currency/bare values).
    """
    profit, revenue_value = _checked(gross_profit, revenue)
    _require_compatible_dimension(profit, revenue_value, "gross_margin")
    if revenue_value.value <= 0:
        raise CalculationError("gross_margin revenue must be positive")
    fraction = profit.value / revenue_value.value
    return _ratio_result(
        operation="gross_margin",
        inputs=(profit, revenue_value),
        fraction=fraction,
        precision=precision,
        formula=f"{profit.value}/{revenue_value.value}",
    )


def net_margin(net_profit, revenue, *, precision: int = 4) -> CalculationResult:
    """net_profit / revenue as a fraction; revenue must be positive.

    Both operands must share a unit dimension (percent vs percent, or
    currency/bare values).
    """
    profit, revenue_value = _checked(net_profit, revenue)
    _require_compatible_dimension(profit, revenue_value, "net_margin")
    if revenue_value.value <= 0:
        raise CalculationError("net_margin revenue must be positive")
    fraction = profit.value / revenue_value.value
    return _ratio_result(
        operation="net_margin",
        inputs=(profit, revenue_value),
        fraction=fraction,
        precision=precision,
        formula=f"{profit.value}/{revenue_value.value}",
    )


def ratio(numerator, denominator, *, precision: int = 4) -> CalculationResult:
    """numerator / denominator as a fraction; denominator must be non-zero.

    Both operands must share a unit dimension (percent vs percent, or
    currency/bare values).
    """
    top, bottom = _checked(numerator, denominator)
    _require_compatible_dimension(top, bottom, "ratio")
    if bottom.value == 0:
        raise CalculationError("ratio denominator must be non-zero")
    fraction = top.value / bottom.value
    return _ratio_result(
        operation="ratio",
        inputs=(top, bottom),
        fraction=fraction,
        precision=precision,
        formula=f"{top.value}/{bottom.value}",
    )


def difference(left, right, *, precision: int = 4) -> CalculationResult:
    """left - right in base units.

    Both operands must share a unit dimension: percent with percent, currency
    with currency (any of 亿/万/千 labels, already normalized to yuan), or
    bare with bare/currency. Result unit: "%" for percents, "元" for currency
    inputs carrying a label, None for bare numbers.
    """
    left_input, right_input = _checked(left, right)
    _require_compatible_dimension(left_input, right_input, "difference")
    value = _quantize(left_input.value - right_input.value, precision)
    if left_input.unit == "%":
        unit: str | None = "%"
    elif left_input.unit or right_input.unit:
        unit = "元"
    else:
        unit = None
    formatted = f"{value}{unit}" if unit else str(value)
    return CalculationResult(
        operation="difference",
        inputs=(left_input, right_input),
        value=value,
        formatted=formatted,
        unit=unit,
        formula=f"{left_input.value}-{right_input.value}",
        provenance=_provenance((left_input, right_input)),
    )


def percentage_point_change(prior_pct, current_pct, *, precision: int = 4) -> CalculationResult:
    """current_pct - prior_pct in percentage points; both inputs must be percents."""
    prior, current = _checked(prior_pct, current_pct)
    if prior.unit != "%" or current.unit != "%":
        raise CalculationError("percentage_point_change requires percent inputs")
    value = _quantize(current.value - prior.value, precision)
    return CalculationResult(
        operation="percentage_point_change",
        inputs=(prior, current),
        value=value,
        formatted=f"{value}pp",
        unit="pp",
        formula=f"{current.value}-{prior.value}",
        provenance=_provenance((prior, current)),
    )
