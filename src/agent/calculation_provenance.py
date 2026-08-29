"""Auditable calculation records for derived financial claims.

This module is the seam between agent reasoning and the deterministic
financial calculator. Callers use one interface, execute_calculation, and
always receive an immutable CalculationRecord: calculator errors, missing
provenance, and claim/result mismatches are data rather than hidden exceptions.

Operand values are expressed in their declared unit. Currency units are
normalized to yuan before the existing Decimal calculator is called. A
successful calculation is only marked verified when every operand references
an evidence_id or fact_id and, when supplied, the claimed result agrees with
the calculated result.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from enum import Enum
from typing import Any, Mapping, Sequence

from src.tools.financial_calculator import (
    CalculationError,
    CalculationInput,
    CalculationResult,
    cagr,
    difference,
    gross_margin,
    growth_rate,
    net_margin,
    parse_numeric_value,
    percentage_point_change,
    ratio,
    yoy,
)


class CalculationStatus(str, Enum):
    """Normalized outcome of an attempted calculation."""

    SUCCESS = "SUCCESS"
    INVALID_INPUT = "INVALID_INPUT"
    DIVIDE_BY_ZERO = "DIVIDE_BY_ZERO"
    UNIT_MISMATCH = "UNIT_MISMATCH"
    INVALID_PERIOD = "INVALID_PERIOD"
    MISSING_PROVENANCE = "MISSING_PROVENANCE"
    RESULT_MISMATCH = "RESULT_MISMATCH"
    UNSUPPORTED_OPERATION = "UNSUPPORTED_OPERATION"
    INTERNAL_ERROR = "INTERNAL_ERROR"


@dataclass(frozen=True)
class CalculationOperand:
    """One ordered input supplied to execute_calculation.

    value is expressed in unit (for example value=2, unit="亿"). Either an
    evidence_id or fact_id is required for the resulting record to be verified.
    period is optional except when it is used to derive or validate CAGR years.
    """

    name: str
    value: Decimal | int | float | str
    unit: str | None = None
    evidence_id: str | None = None
    fact_id: str | None = None
    chunk_id: str | None = None
    doc_id: str | None = None
    raw_text: str | None = None
    period: int | str | None = None
    upstream_calculation_id: str | None = None
    source_step_id: str | None = None
    upstream_evidence_ids: Sequence[str] = ()


@dataclass(frozen=True)
class CalculationInputRecord:
    """Canonical, ordered operand persisted in a calculation trace."""

    name: str
    value: str
    unit: str | None
    raw_text: str
    scale: int | None
    evidence_id: str | None
    fact_id: str | None
    chunk_id: str | None
    doc_id: str | None
    period: str | None
    upstream_calculation_id: str | None
    source_step_id: str | None
    upstream_evidence_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "raw_text": self.raw_text,
            "scale": self.scale,
            "evidence_id": self.evidence_id,
            "fact_id": self.fact_id,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "period": self.period,
            "upstream_calculation_id": self.upstream_calculation_id,
            "source_step_id": self.source_step_id,
            "upstream_evidence_ids": list(self.upstream_evidence_ids),
        }


@dataclass(frozen=True)
class CalculationOutput:
    value: str
    formatted: str
    unit: str | None

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "formatted": self.formatted, "unit": self.unit}


@dataclass(frozen=True)
class RoundingRecord:
    precision: int
    mode: str = "ROUND_HALF_UP"

    def to_dict(self) -> dict[str, Any]:
        return {"precision": self.precision, "mode": self.mode}


@dataclass(frozen=True)
class CalculationRecord:
    """Stable, JSON-serializable trace for one attempted calculation."""

    calculation_id: str
    tool: str
    operation: str
    formula: str | None
    inputs: tuple[CalculationInputRecord, ...]
    result: CalculationOutput | None
    rounding: RoundingRecord
    status: CalculationStatus
    verified: bool
    provenance_complete: bool
    claimed_result: str | None = None
    claim_result_matches: bool | None = None
    years: str | None = None
    error_type: str | None = None
    error_message: str | None = None

    @property
    def input_calculation_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                item.upstream_calculation_id
                for item in self.inputs
                if item.upstream_calculation_id
            )
        )

    @property
    def source_step_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(item.source_step_id for item in self.inputs if item.source_step_id)
        )

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        values: list[str] = []
        for item in self.inputs:
            if item.evidence_id and item.evidence_id not in values:
                values.append(item.evidence_id)
            for evidence_id in item.upstream_evidence_ids:
                if evidence_id and evidence_id not in values:
                    values.append(evidence_id)
        return tuple(values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "calculation_id": self.calculation_id,
            "tool": self.tool,
            "operation": self.operation,
            "formula": self.formula,
            "inputs": [item.to_dict() for item in self.inputs],
            "input_calculation_ids": list(self.input_calculation_ids),
            "source_step_ids": list(self.source_step_ids),
            "evidence_ids": list(self.evidence_ids),
            "result": self.result.to_dict() if self.result else None,
            "rounding": self.rounding.to_dict(),
            "status": self.status.value,
            "verified": self.verified,
            "provenance_complete": self.provenance_complete,
            "claimed_result": self.claimed_result,
            "claim_result_matches": self.claim_result_matches,
            "years": self.years,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


_TOOL_NAME = "financial_calculator"
_CURRENCY_SCALES: dict[str, int] = {
    "元": 1,
    "亿": 10**8,
    "亿元": 10**8,
    "万": 10**4,
    "万元": 10**4,
    "千": 10**3,
    "千元": 10**3,
}
_POSITIONAL_NAMES: dict[str, tuple[str, str]] = {
    "growth_rate": ("start", "end"),
    "yoy": ("prior", "current"),
    "qoq": ("prior", "current"),
    "cagr": ("start", "end"),
    "gross_margin": ("gross_profit", "revenue"),
    "net_margin": ("net_profit", "revenue"),
    "ratio": ("numerator", "denominator"),
    "difference": ("left", "right"),
    "percentage_point_change": ("prior_pct", "current_pct"),
}
_RATIO_OPERATIONS = frozenset({"growth_rate", "yoy", "qoq", "cagr", "gross_margin", "net_margin", "ratio"})
_FULLWIDTH_TABLE = str.maketrans("０１２３４５６７８９．，－＋％", "0123456789.,-+%")
_NUMBER_RE = re.compile(r"[+-]?\d[\d,]*(?:\.\d+)?")
_PP_RE = re.compile(r"(?:个百分点|百分点|\bpp\b)", re.IGNORECASE)
_PERIOD_RE = re.compile(r"^(?:(FY|Q1|H1|Q3)[- ]?)?((?:19|20)\d{2}|2100)(?:年)?$", re.IGNORECASE)


def execute_calculation(
    operation: str,
    inputs: Sequence[CalculationOperand | Mapping[str, Any]],
    *,
    precision: int = 4,
    years: int | float | Decimal | None = None,
    claimed_result: str | int | float | Decimal | None = None,
) -> CalculationRecord:
    """Execute one whitelisted operation and always return a trace record.

    qoq is a semantic alias for the calculator's yoy formula. CAGR years may
    be supplied explicitly, derived from two operand periods, or checked
    against those periods when both are present. Unsupported operations and
    malformed inputs never reach dynamic dispatch.
    """

    operation_name = str(operation or "").strip().lower()
    rounding = RoundingRecord(precision=precision if isinstance(precision, int) else -1)
    claimed_text = None if claimed_result is None else str(claimed_result)

    try:
        raw_operands = tuple(_coerce_operand(item, index, operation_name) for index, item in enumerate(inputs))
    except Exception as exc:  # input snapshots still make the failure observable
        snapshots = _snapshot_inputs(inputs, operation_name)
        return _record(
            operation=operation_name,
            inputs=snapshots,
            rounding=rounding,
            status=CalculationStatus.INVALID_INPUT,
            claimed_result=claimed_text,
            error=exc,
        )

    snapshots = tuple(_snapshot_operand(item) for item in raw_operands)
    if operation_name not in _POSITIONAL_NAMES:
        return _record(
            operation=operation_name,
            inputs=snapshots,
            rounding=rounding,
            status=CalculationStatus.UNSUPPORTED_OPERATION,
            claimed_result=claimed_text,
            error=ValueError(f"unsupported calculation operation {operation_name!r}"),
        )
    if not isinstance(precision, int) or isinstance(precision, bool) or precision < 0:
        return _record(
            operation=operation_name,
            inputs=snapshots,
            rounding=rounding,
            status=CalculationStatus.INVALID_INPUT,
            claimed_result=claimed_text,
            error=ValueError(f"precision must be a non-negative int, got {precision!r}"),
        )
    if len(raw_operands) != 2:
        return _record(
            operation=operation_name,
            inputs=snapshots,
            rounding=rounding,
            status=CalculationStatus.INVALID_INPUT,
            claimed_result=claimed_text,
            error=ValueError(f"{operation_name} requires exactly 2 ordered inputs"),
        )

    try:
        calculator_inputs, input_records = _normalize_operands(raw_operands)
    except Exception as exc:
        status = CalculationStatus.UNIT_MISMATCH if _is_unit_error(str(exc)) else CalculationStatus.INVALID_INPUT
        return _record(
            operation=operation_name,
            inputs=snapshots,
            rounding=rounding,
            status=status,
            claimed_result=claimed_text,
            error=exc,
        )

    resolved_years: Decimal | None = None
    if operation_name == "cagr":
        try:
            resolved_years = _resolve_cagr_years(raw_operands, years)
        except Exception as exc:
            return _record(
                operation=operation_name,
                inputs=input_records,
                rounding=rounding,
                status=CalculationStatus.INVALID_PERIOD,
                claimed_result=claimed_text,
                years=None if years is None else str(years),
                error=exc,
            )
    elif years is not None:
        return _record(
            operation=operation_name,
            inputs=input_records,
            rounding=rounding,
            status=CalculationStatus.INVALID_INPUT,
            claimed_result=claimed_text,
            years=str(years),
            error=ValueError("years is only valid for cagr"),
        )

    try:
        result = _run_calculator(operation_name, calculator_inputs, precision, resolved_years)
    except (CalculationError, TypeError, ValueError, ArithmeticError) as exc:
        return _record(
            operation=operation_name,
            inputs=input_records,
            rounding=rounding,
            status=_classify_calculator_error(operation_name, calculator_inputs, exc),
            claimed_result=claimed_text,
            years=None if resolved_years is None else str(resolved_years),
            error=exc,
        )
    except Exception as exc:  # fail closed while retaining diagnostics
        return _record(
            operation=operation_name,
            inputs=input_records,
            rounding=rounding,
            status=CalculationStatus.INTERNAL_ERROR,
            claimed_result=claimed_text,
            years=None if resolved_years is None else str(resolved_years),
            error=exc,
        )

    provenance_complete = all(_has_provenance(item) for item in input_records)
    output = CalculationOutput(value=str(result.value), formatted=result.formatted, unit=result.unit)
    claim_matches: bool | None = None
    if claimed_result is not None:
        try:
            claim_matches = _claimed_result_matches(operation_name, result, claimed_text or "")
        except (CalculationError, InvalidOperation, ValueError):
            claim_matches = False

    if not provenance_complete:
        missing = ", ".join(item.name for item in input_records if not _has_provenance(item))
        return _record(
            operation=operation_name,
            inputs=input_records,
            rounding=rounding,
            status=CalculationStatus.MISSING_PROVENANCE,
            formula=result.formula,
            result=output,
            claimed_result=claimed_text,
            claim_result_matches=claim_matches,
            years=None if resolved_years is None else str(resolved_years),
            error=ValueError(f"inputs missing evidence_id or fact_id: {missing}"),
        )
    if claim_matches is False:
        return _record(
            operation=operation_name,
            inputs=input_records,
            rounding=rounding,
            status=CalculationStatus.RESULT_MISMATCH,
            formula=result.formula,
            result=output,
            claimed_result=claimed_text,
            claim_result_matches=False,
            years=None if resolved_years is None else str(resolved_years),
            error=ValueError("claimed result does not match calculated result"),
        )
    return _record(
        operation=operation_name,
        inputs=input_records,
        rounding=rounding,
        status=CalculationStatus.SUCCESS,
        formula=result.formula,
        result=output,
        claimed_result=claimed_text,
        claim_result_matches=claim_matches,
        years=None if resolved_years is None else str(resolved_years),
    )


def _coerce_operand(
    item: CalculationOperand | Mapping[str, Any], index: int, operation: str
) -> CalculationOperand:
    if isinstance(item, CalculationOperand):
        return item
    if not isinstance(item, Mapping):
        raise TypeError(f"input {index} must be CalculationOperand or mapping")
    data = dict(item)
    if "name" not in data:
        names = _POSITIONAL_NAMES.get(operation, ("left", "right"))
        data["name"] = names[index] if index < len(names) else f"input_{index + 1}"
    try:
        return CalculationOperand(**data)
    except TypeError as exc:
        raise ValueError(f"invalid input {index}: {exc}") from exc


def _normalize_operands(
    operands: tuple[CalculationOperand, ...]
) -> tuple[tuple[CalculationInput, ...], tuple[CalculationInputRecord, ...]]:
    calculator_inputs: list[CalculationInput] = []
    records: list[CalculationInputRecord] = []
    for operand in operands:
        if not isinstance(operand.name, str) or not operand.name.strip():
            raise ValueError("calculation input name must be non-empty")
        calculator_input = _normalize_operand_value(operand)
        calculator_inputs.append(calculator_input)
        records.append(
            CalculationInputRecord(
                name=operand.name.strip(),
                value=str(calculator_input.value),
                unit=_canonical_record_unit(calculator_input.unit),
                raw_text=calculator_input.raw_text,
                scale=calculator_input.scale,
                evidence_id=_clean_id(operand.evidence_id),
                fact_id=_clean_id(operand.fact_id),
                chunk_id=_clean_id(operand.chunk_id),
                doc_id=_clean_id(operand.doc_id),
                period=None if operand.period is None else str(operand.period),
                upstream_calculation_id=_clean_id(operand.upstream_calculation_id),
                source_step_id=_clean_id(operand.source_step_id),
                upstream_evidence_ids=_clean_ids(operand.upstream_evidence_ids),
            )
        )
    return tuple(calculator_inputs), tuple(records)


def _normalize_operand_value(operand: CalculationOperand) -> CalculationInput:
    unit = None if operand.unit is None else str(operand.unit).strip()
    if unit == "":
        unit = None
    raw_text = operand.raw_text or _render_raw_value(operand.value, unit)

    if unit is None and isinstance(operand.value, str):
        value, scale, parsed_unit = parse_numeric_value(operand.value)
        return CalculationInput(
            value=value,
            raw_text=raw_text,
            unit=parsed_unit,
            scale=scale,
            evidence_id=_clean_id(operand.evidence_id),
            chunk_id=_clean_id(operand.chunk_id),
            doc_id=_clean_id(operand.doc_id),
        )

    numeric = _as_decimal(operand.value)
    if unit in _CURRENCY_SCALES:
        scale = _CURRENCY_SCALES[unit]
        normalized = numeric * scale
        calculator_unit: str | None = "元"
    elif unit == "%":
        scale = 1
        normalized = numeric
        calculator_unit = "%"
    elif unit is None:
        scale = 1
        normalized = numeric
        calculator_unit = None
    else:
        raise CalculationError(f"unknown input unit {unit!r}")
    if not normalized.is_finite():
        raise CalculationError(f"non-finite input value in {raw_text!r}")
    return CalculationInput(
        value=normalized,
        raw_text=raw_text,
        unit=calculator_unit,
        scale=scale,
        evidence_id=_clean_id(operand.evidence_id),
        chunk_id=_clean_id(operand.chunk_id),
        doc_id=_clean_id(operand.doc_id),
    )


def _run_calculator(
    operation: str,
    inputs: tuple[CalculationInput, ...],
    precision: int,
    years: Decimal | None,
) -> CalculationResult:
    left, right = inputs
    if operation == "growth_rate":
        return growth_rate(left, right, precision=precision)
    if operation in {"yoy", "qoq"}:
        return yoy(left, right, precision=precision)
    if operation == "cagr":
        assert years is not None
        return cagr(left, right, years, precision=precision)
    if operation == "gross_margin":
        return gross_margin(left, right, precision=precision)
    if operation == "net_margin":
        return net_margin(left, right, precision=precision)
    if operation == "ratio":
        return ratio(left, right, precision=precision)
    if operation == "difference":
        return difference(left, right, precision=precision)
    if operation == "percentage_point_change":
        return percentage_point_change(left, right, precision=precision)
    raise AssertionError(f"unreachable operation {operation!r}")


def _resolve_cagr_years(
    operands: tuple[CalculationOperand, ...], years: int | float | Decimal | None
) -> Decimal:
    start_period, end_period = operands[0].period, operands[1].period
    if (start_period is None) != (end_period is None):
        raise ValueError("cagr requires both operand periods when either period is supplied")

    derived: Decimal | None = None
    if start_period is not None and end_period is not None:
        start_kind, start_year = _parse_period(start_period)
        end_kind, end_year = _parse_period(end_period)
        if start_kind != end_kind:
            raise ValueError(f"cagr period kinds must match: {start_kind!r} != {end_kind!r}")
        if end_year <= start_year:
            raise ValueError("cagr end period must be later than start period")
        derived = Decimal(end_year - start_year)

    supplied = None if years is None else _as_decimal(years)
    if supplied is not None and (not supplied.is_finite() or supplied <= 0):
        raise ValueError("cagr years must be positive")
    if supplied is None and derived is None:
        raise ValueError("cagr requires years or two operand periods")
    if supplied is not None and derived is not None and supplied != derived:
        raise ValueError(f"cagr years {supplied} does not match period span {derived}")
    return supplied if supplied is not None else derived  # type: ignore[return-value]


def _parse_period(value: int | str) -> tuple[str, int]:
    if isinstance(value, bool):
        raise ValueError(f"invalid financial period {value!r}")
    text = str(value).strip().upper()
    match = _PERIOD_RE.fullmatch(text)
    if not match:
        raise ValueError(f"invalid financial period {value!r}")
    return (match.group(1) or "FY", int(match.group(2)))


def _claimed_result_matches(operation: str, result: CalculationResult, claimed_text: str) -> bool:
    claimed_value, claimed_unit, decimal_places = _parse_claimed_result(claimed_text)
    if operation in _RATIO_OPERATIONS:
        if claimed_unit == "%":
            comparable = result.value * 100
        elif claimed_unit is None:
            comparable = result.value
        else:
            return False
    elif result.unit == "pp":
        if claimed_unit not in {None, "pp"}:
            return False
        comparable = result.value
    elif result.unit == "%":
        if claimed_unit not in {None, "%"}:
            return False
        comparable = result.value
    elif result.unit == "元":
        if claimed_unit not in {None, "元"}:
            return False
        comparable = result.value
    else:
        if claimed_unit is not None:
            return False
        comparable = result.value
    quantum = Decimal(1).scaleb(-decimal_places)
    return comparable.quantize(quantum, rounding=ROUND_HALF_UP) == claimed_value.quantize(
        quantum, rounding=ROUND_HALF_UP
    )


def _parse_claimed_result(text: str) -> tuple[Decimal, str | None, int]:
    normalized = text.translate(_FULLWIDTH_TABLE)
    candidates = _NUMBER_RE.findall(normalized)
    if len(candidates) != 1:
        raise CalculationError(f"claimed result must contain exactly one numeric value: {text!r}")
    token = candidates[0]
    decimal_value = Decimal(token.replace(",", ""))
    decimal_places = len(token.rsplit(".", 1)[1]) if "." in token else 0
    if _PP_RE.search(normalized):
        return decimal_value, "pp", decimal_places
    value, _scale, parsed_unit = parse_numeric_value(normalized)
    if parsed_unit in _CURRENCY_SCALES:
        return value, "元", decimal_places
    return value, parsed_unit, decimal_places


def _classify_calculator_error(
    operation: str, inputs: tuple[CalculationInput, ...], exc: Exception
) -> CalculationStatus:
    message = str(exc).lower()
    if _is_unit_error(message):
        return CalculationStatus.UNIT_MISMATCH
    if "years" in message or "period" in message:
        return CalculationStatus.INVALID_PERIOD
    denominator_index = {
        "growth_rate": 0,
        "yoy": 0,
        "qoq": 0,
        "cagr": 0,
        "gross_margin": 1,
        "net_margin": 1,
        "ratio": 1,
    }.get(operation)
    if denominator_index is not None and inputs[denominator_index].value == 0:
        return CalculationStatus.DIVIDE_BY_ZERO
    if "non-zero" in message or "division" in message:
        return CalculationStatus.DIVIDE_BY_ZERO
    return CalculationStatus.INVALID_INPUT


def _is_unit_error(message: str) -> bool:
    lowered = message.lower()
    return "unit" in lowered or "percent" in lowered or "percentage" in lowered


def _record(
    *,
    operation: str,
    inputs: tuple[CalculationInputRecord, ...],
    rounding: RoundingRecord,
    status: CalculationStatus,
    formula: str | None = None,
    result: CalculationOutput | None = None,
    claimed_result: str | None = None,
    claim_result_matches: bool | None = None,
    years: str | None = None,
    error: Exception | None = None,
) -> CalculationRecord:
    provenance_complete = bool(inputs) and all(_has_provenance(item) for item in inputs)
    verified = status is CalculationStatus.SUCCESS and provenance_complete and claim_result_matches is not False
    content = {
        "schema_version": 1,
        "tool": _TOOL_NAME,
        "operation": operation,
        "formula": formula,
        "inputs": [item.to_dict() for item in inputs],
        "input_calculation_ids": list(
            dict.fromkeys(
                item.upstream_calculation_id
                for item in inputs
                if item.upstream_calculation_id
            )
        ),
        "source_step_ids": list(
            dict.fromkeys(item.source_step_id for item in inputs if item.source_step_id)
        ),
        "evidence_ids": list(
            dict.fromkeys(
                evidence_id
                for item in inputs
                for evidence_id in (
                    *((item.evidence_id,) if item.evidence_id else ()),
                    *item.upstream_evidence_ids,
                )
                if evidence_id
            )
        ),
        "result": result.to_dict() if result else None,
        "rounding": rounding.to_dict(),
        "status": status.value,
        "verified": verified,
        "provenance_complete": provenance_complete,
        "claimed_result": claimed_result,
        "claim_result_matches": claim_result_matches,
        "years": years,
        "error_type": type(error).__name__ if error else None,
        "error_message": str(error) if error else None,
    }
    canonical = json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    calculation_id = "CALC-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return CalculationRecord(
        calculation_id=calculation_id,
        tool=_TOOL_NAME,
        operation=operation,
        formula=formula,
        inputs=inputs,
        result=result,
        rounding=rounding,
        status=status,
        verified=verified,
        provenance_complete=provenance_complete,
        claimed_result=claimed_result,
        claim_result_matches=claim_result_matches,
        years=years,
        error_type=type(error).__name__ if error else None,
        error_message=str(error) if error else None,
    )


def _snapshot_inputs(inputs: Any, operation: str) -> tuple[CalculationInputRecord, ...]:
    try:
        iterable = tuple(inputs)
    except TypeError:
        iterable = (inputs,)
    snapshots: list[CalculationInputRecord] = []
    for index, item in enumerate(iterable):
        try:
            snapshots.append(_snapshot_operand(_coerce_operand(item, index, operation)))
        except Exception:
            snapshots.append(
                CalculationInputRecord(
                    name=f"input_{index + 1}",
                    value=_stable_text(item),
                    unit=None,
                    raw_text=_stable_text(item),
                    scale=None,
                    evidence_id=None,
                    fact_id=None,
                    chunk_id=None,
                    doc_id=None,
                    period=None,
                    upstream_calculation_id=None,
                    source_step_id=None,
                    upstream_evidence_ids=(),
                )
            )
    return tuple(snapshots)


def _snapshot_operand(item: CalculationOperand) -> CalculationInputRecord:
    return CalculationInputRecord(
        name=str(item.name),
        value=_stable_text(item.value),
        unit=None if item.unit is None else str(item.unit),
        raw_text=item.raw_text or _render_raw_value(item.value, item.unit),
        scale=None,
        evidence_id=_clean_id(item.evidence_id),
        fact_id=_clean_id(item.fact_id),
        chunk_id=_clean_id(item.chunk_id),
        doc_id=_clean_id(item.doc_id),
        period=None if item.period is None else str(item.period),
        upstream_calculation_id=_clean_id(item.upstream_calculation_id),
        source_step_id=_clean_id(item.source_step_id),
        upstream_evidence_ids=_clean_ids(item.upstream_evidence_ids),
    )


def _has_provenance(item: CalculationInputRecord) -> bool:
    return bool(
        item.evidence_id
        or item.fact_id
        or (
            item.upstream_calculation_id
            and item.source_step_id
            and item.upstream_evidence_ids
        )
    )


def _clean_id(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _clean_ids(values: Sequence[str] | None) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return ()
    return tuple(
        dict.fromkeys(str(value).strip() for value in values if str(value).strip())
    )


def _canonical_record_unit(unit: str | None) -> str | None:
    if unit in _CURRENCY_SCALES or unit == "元":
        return "元"
    return unit


def _render_raw_value(value: Any, unit: Any) -> str:
    return f"{_stable_text(value)}{'' if unit is None else unit}"


def _stable_text(value: Any) -> str:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Mapping):
        return json.dumps(dict(value), ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return str(value)


def _as_decimal(value: Any) -> Decimal:
    if isinstance(value, bool):
        raise CalculationError(f"boolean is not a financial numeric value: {value!r}")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise CalculationError(f"invalid numeric value {value!r}") from exc
    if not result.is_finite():
        raise CalculationError(f"non-finite numeric value {value!r}")
    return result


__all__ = [
    "CalculationInputRecord",
    "CalculationOperand",
    "CalculationOutput",
    "CalculationRecord",
    "CalculationStatus",
    "RoundingRecord",
    "execute_calculation",
]
