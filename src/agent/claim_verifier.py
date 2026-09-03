"""Claim-specific provenance mapping and three-state verification.

The module deliberately exposes one small interface, :func:`verify_claim`.
It does not retrieve evidence and has no provider dependency: callers pass the
current evidence pool and may inject a semantic scorer and/or a structured LLM
judge.  Deterministic checks always run first.

For numeric claims, ENTAILED is fail-closed.  Entity, metric, period, value,
unit dimension and actual/forecast type must all be identifiable and match the
same evidence fact.  Currency units are compared after Decimal normalization;
percent and percentage-point units remain different dimensions.

The optional LLM judge receives one mapping and must return a mapping (or a JSON
object string) with exactly the following useful fields::

    {
        "status": "ENTAILED|CONTRADICTED|INSUFFICIENT",
        "evidence_ids": ["E1"],
        "score": 0.0,
        "reasons": ["short reason"]
    }

Malformed judge output fails closed as INSUFFICIENT.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Literal, TypedDict
from enum import Enum


VerificationStatus = Literal["ENTAILED", "CONTRADICTED", "INSUFFICIENT"]
class VerificationMethod(str, Enum):
    """Stable serialized names for the three validator layers plus controls."""

    DETERMINISTIC = "deterministic"
    SEMANTIC = "semantic"
    LLM = "llm"
    CALCULATION = "calculation"
    BUDGET = "budget"


VERIFICATION_METHODS = frozenset(item.value for item in VerificationMethod)
SemanticScorer = Callable[[str, str], float | Mapping[str, Any]]
LLMJudge = Callable[[Mapping[str, Any]], str | Mapping[str, Any]]
CalculationLookup = Mapping[str, Mapping[str, Any]] | Callable[[str], Mapping[str, Any] | None]

ENTAILED: VerificationStatus = "ENTAILED"
CONTRADICTED: VerificationStatus = "CONTRADICTED"
INSUFFICIENT: VerificationStatus = "INSUFFICIENT"
_STATUSES = {ENTAILED, CONTRADICTED, INSUFFICIENT}


class VerificationDetails(TypedDict):
    status: VerificationStatus
    method: str
    score: float
    reasons: list[str]


class ClaimVerificationResult(TypedDict):
    evidence_ids: list[str]
    verification: VerificationDetails


_FULLWIDTH = str.maketrans("０１２３４５６７８９．，％", "0123456789.,%")
_NON_WORD_RE = re.compile(r"[^0-9a-zA-Z\u4e00-\u9fff%]+")
_NUMBER_RE = r"[+-]?\d[\d,]*(?:\.\d+)?"
_UNIT_PATTERN = (
    r"percentage\s+points?|pct\s+points?|个百分点|百万元|千万元|亿元|万元|千元|"
    r"元/股|元／股|人民币元|元|pp|pct|亿|万|千|%|倍|吨|台|辆|家|股"
)
_NUMBER_WITH_UNIT_RE = re.compile(
    rf"(?P<number>{_NUMBER_RE})\s*(?P<unit>{_UNIT_PATTERN})",
    flags=re.IGNORECASE,
)
_PLAIN_NUMBER_RE = re.compile(_NUMBER_RE)

_UNIT_INFO: dict[str, tuple[str, Decimal]] = {
    "人民币元": ("currency", Decimal(1)),
    "元": ("currency", Decimal(1)),
    "千元": ("currency", Decimal(10) ** 3),
    "万元": ("currency", Decimal(10) ** 4),
    "百万元": ("currency", Decimal(10) ** 6),
    "千万元": ("currency", Decimal(10) ** 7),
    "亿元": ("currency", Decimal(10) ** 8),
    "千": ("currency", Decimal(10) ** 3),
    "万": ("currency", Decimal(10) ** 4),
    "亿": ("currency", Decimal(10) ** 8),
    "%": ("percent", Decimal(1)),
    "pp": ("percentage_point", Decimal(1)),
    "pct": ("percentage_point", Decimal(1)),
    "百分点": ("percentage_point", Decimal(1)),
    "个百分点": ("percentage_point", Decimal(1)),
    "percentage point": ("percentage_point", Decimal(1)),
    "percentage points": ("percentage_point", Decimal(1)),
    "pct point": ("percentage_point", Decimal(1)),
    "pct points": ("percentage_point", Decimal(1)),
    "倍": ("multiple", Decimal(1)),
    "元/股": ("currency_per_share", Decimal(1)),
    "吨": ("unit:吨", Decimal(1)),
    "台": ("unit:台", Decimal(1)),
    "辆": ("unit:辆", Decimal(1)),
    "家": ("unit:家", Decimal(1)),
    "股": ("unit:股", Decimal(1)),
}

_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "revenue": ("营业收入", "营收", "revenue"),
    "net_profit_parent": ("归母净利润", "归属母公司净利润", "attributable net profit"),
    "net_profit": ("净利润", "net profit"),
    "gross_profit": ("毛利润", "毛利", "gross profit"),
    "operating_profit": ("营业利润", "经营利润", "operating profit"),
    "gross_margin": ("销售毛利率", "毛利率", "gross margin"),
    "net_margin": ("销售净利率", "净利率", "net margin"),
    "operating_cash_flow": ("经营活动现金流量净额", "经营性现金流", "经营现金流", "operating cash flow"),
    "research_and_development": ("研发费用", "研发投入", "research and development", "r&d"),
    "eps": ("每股收益", "eps"),
    "roe": ("净资产收益率", "roe"),
    "roa": ("总资产收益率", "roa"),
    "yoy_growth": ("同比增长率", "同比增速", "同比增长", "同比", "yoy"),
    "qoq_growth": ("环比增长率", "环比增速", "环比增长", "环比", "qoq"),
    "cagr": ("复合年增长率", "年复合增长率", "cagr"),
}

_FORECAST_RE = re.compile(
    r"预计|预测|预期|一致预期|指引|估算|forecast|estimate|estimated|guidance|consensus|\b[12]0\d{2}e\b",
    flags=re.IGNORECASE,
)
_ADJUSTED_RE = re.compile(
    r"经调整|调整后|non[-\s]?gaap|adjusted",
    flags=re.IGNORECASE,
)
_NEGATIONS = ("没有", "并非", "不是", "未能", "未", "不", "无", "否认", "not", "no")
_GENERIC_TOKENS = {"公司", "报告", "显示", "指出", "认为", "the", "and", "report"}
_ENTITY_GENERIC = {
    "年度", "年", "季度", "上半年", "下半年", "实际", "预计", "预测", "营收", "收入",
    "营业收入", "净利润", "利润", "毛利", "毛利率", "净利率", "同比", "环比", "增长",
    "增长率", "增速", "金额", "约为", "为", "达到", "实现", "机构", "市场",
    "元", "万元", "亿元", "千元", "百万元", "千万元", "百分点", "个百分点",
}


@dataclass(frozen=True)
class _NumericValue:
    value: Decimal
    unit: str
    dimension: str


@dataclass(frozen=True)
class _Profile:
    entities: frozenset[str]
    metric: str | None
    period: str | None
    numeric: _NumericValue | None
    value_type: str | None
    accounting_scope: str | None
    revision_status: str | None


def verify_claim(
    claim: Mapping[str, Any],
    evidence_rows: Sequence[Mapping[str, Any]],
    *,
    calculation_lookup: CalculationLookup | None = None,
    semantic_scorer: SemanticScorer | None = None,
    llm_judge: LLMJudge | None = None,
    semantic_threshold: float = 0.85,
    llm_budget: int = 1,
) -> ClaimVerificationResult:
    """Map evidence and verify one atomic claim.

    The function performs no I/O. ``llm_budget=0`` disables the injected judge
    for this call.  A numeric claim can never become ENTAILED through semantic
    or LLM judgment when its six deterministic constraints are incomplete.
    """
    if not isinstance(claim, Mapping):
        raise TypeError("claim must be a mapping")
    if not 0.0 <= semantic_threshold <= 1.0:
        raise ValueError("semantic_threshold must be between 0 and 1")
    if not isinstance(llm_budget, int) or llm_budget < 0:
        raise ValueError("llm_budget must be a non-negative int")

    text = str(claim.get("text") or "").strip()
    rows = [row for row in evidence_rows if isinstance(row, Mapping) and _evidence_id(row)]
    if not text:
        return _result(INSUFFICIENT, "deterministic", 0.0, ["empty_claim_text"], [])
    calculation_id = str(claim.get("calculation_id") or "").strip()
    if str(claim.get("claim_type") or "").strip().upper() == "DERIVED":
        if not calculation_id:
            return _result(
                INSUFFICIENT,
                "calculation",
                0.0,
                ["derived_claim_missing_calculation"],
                [],
            )
        if calculation_lookup is None:
            return _result(
                INSUFFICIENT,
                "calculation",
                0.0,
                ["calculation_not_found"],
                [],
            )
    if not rows:
        return _result(INSUFFICIENT, "deterministic", 0.0, ["no_evidence"], [])

    entity_catalog = _entity_catalog(rows)
    claim_profile = _profile(claim, text, entity_catalog=entity_catalog, default_actual=True)
    numeric_missing = _missing_numeric_constraints(claim_profile) if claim_profile.numeric else []

    if calculation_id and calculation_lookup is not None:
        calculation_result = _verify_calculation(
            calculation_id, claim_profile, calculation_lookup, rows, claim_text=text
        )
        if calculation_result is not None:
            return calculation_result

    if claim_profile.numeric is not None and not numeric_missing:
        return _verify_complete_numeric(claim_profile, rows, entity_catalog)

    candidates = _qualitative_candidates(text, rows)
    if not candidates:
        reasons = ["no_claim_specific_evidence"]
        if numeric_missing:
            reasons.append("numeric_constraints_incomplete:" + ",".join(numeric_missing))
        return _result(INSUFFICIENT, "deterministic", 0.0, reasons, [])

    deterministic = _verify_qualitative_deterministically(claim, text, candidates)
    if deterministic is not None:
        if deterministic["verification"]["status"] == ENTAILED and numeric_missing:
            return _result(
                INSUFFICIENT,
                "deterministic",
                0.0,
                deterministic["verification"]["reasons"]
                + [
                    "numeric_constraints_incomplete:" + ",".join(numeric_missing),
                    "numeric_entailment_blocked_by_incomplete_constraints",
                ],
                deterministic["evidence_ids"],
            )
        return deterministic

    semantic_result: ClaimVerificationResult | None = None
    semantic_score = 0.0
    semantic_reasons: list[str] = []
    if semantic_scorer is not None:
        semantic_result, semantic_score, semantic_reasons = _verify_semantically(
            text, candidates, semantic_scorer, semantic_threshold
        )
        if semantic_result is not None:
            if semantic_result["verification"]["status"] != ENTAILED or not numeric_missing:
                return semantic_result
            semantic_reasons.append("numeric_entailment_blocked_by_incomplete_constraints")

    if llm_judge is None:
        reasons = semantic_reasons or ["deterministic_undecidable", "no_llm_judge"]
        if numeric_missing:
            reasons.append("numeric_constraints_incomplete:" + ",".join(numeric_missing))
        method = "semantic" if semantic_scorer is not None else "deterministic"
        return _result(INSUFFICIENT, method, semantic_score, reasons, [_evidence_id(row) for row in candidates])

    if llm_budget == 0:
        reasons = ["llm_budget_exhausted"]
        if numeric_missing:
            reasons.append("numeric_constraints_incomplete:" + ",".join(numeric_missing))
        return _result(INSUFFICIENT, "budget", semantic_score, reasons, [_evidence_id(row) for row in candidates])

    judged = _verify_with_llm(text, claim, candidates, llm_judge)
    if judged["verification"]["status"] == ENTAILED and numeric_missing:
        return _result(
            INSUFFICIENT,
            "llm",
            judged["verification"]["score"],
            judged["verification"]["reasons"]
            + ["numeric_entailment_blocked_by_incomplete_constraints"],
            judged["evidence_ids"],
        )
    return judged


def _result(
    status: VerificationStatus, method: str, score: float, reasons: Sequence[str], evidence_ids: Sequence[str]
) -> ClaimVerificationResult:
    # Accept enum members and legacy aliases while always emitting a JSON-safe
    # string.  This keeps old serialized traces readable and prevents Enum
    # objects leaking into JSON encoders.
    method_value = method.value if isinstance(method, VerificationMethod) else str(method)
    aliases = {"calc": VerificationMethod.CALCULATION.value, "judge": VerificationMethod.LLM.value}
    method_value = aliases.get(method_value, method_value)
    if method_value not in VERIFICATION_METHODS:
        method_value = VerificationMethod.DETERMINISTIC.value
    return {
        "evidence_ids": list(dict.fromkeys(str(item) for item in evidence_ids if item)),
        "verification": {
            "status": status,
            "method": method_value,
            "score": max(0.0, min(1.0, float(score))),
            "reasons": list(dict.fromkeys(str(reason) for reason in reasons if reason)),
        },
    }


def _compact(value: Any) -> str:
    return _NON_WORD_RE.sub("", str(value or "").translate(_FULLWIDTH).lower())


def _nested_sources(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    sources = [payload]
    for key in ("constraints", "fact", "metadata"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            sources.append(value)
    return sources


def _first(payload: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for source in _nested_sources(payload):
        for key in keys:
            if key in source and source[key] not in (None, "", []):
                return source[key]
    return None


def _string_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        values: list[str] = []
        for key in ("name", "entity", "company", "value", "aliases"):
            if key in value:
                values.extend(_string_values(value[key]))
        return values
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = []
        for item in value:
            values.extend(_string_values(item))
        return values
    text = str(value).strip()
    return [text] if text else []


def _entities(payload: Mapping[str, Any], text: str, entity_catalog: Sequence[str]) -> frozenset[str]:
    explicit: list[str] = []
    for source in _nested_sources(payload):
        for key in ("entity", "entities", "company", "company_name", "issuer", "entity_name", "aliases"):
            explicit.extend(_string_values(source.get(key)))
    normalized = {_compact(item) for item in explicit if _compact(item)}
    if normalized:
        return frozenset(normalized)
    compact_text = _compact(text)
    inferred = {_compact(item) for item in entity_catalog if _compact(item) and _compact(item) in compact_text}
    if not inferred:
        # Retrieval rows in legacy fixtures often have no structured ``entity``
        # field.  Infer only the non-numeric, non-metric CJK spans that remain
        # after removing known financial vocabulary.  This is deliberately
        # conservative: a bare year/metric claim still has no entity and stays
        # fail-closed, while ``宁德时代2025年营收...`` can be matched against
        # the same textual issuer in its evidence row.
        spans = re.findall(r"[\u4e00-\u9fff]{2,12}", str(text or ""))
        for span in spans:
            normalized_span = span
            for token in sorted(_ENTITY_GENERIC, key=len, reverse=True):
                normalized_span = normalized_span.replace(token, "")
            normalized_span = _compact(normalized_span)
            if normalized_span and len(normalized_span) >= 2:
                inferred.add(normalized_span)
    return frozenset(inferred)


def _entity_catalog(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    values: list[str] = []
    for row in rows:
        for source in _nested_sources(row):
            for key in ("entity", "entities", "company", "company_name", "issuer", "entity_name", "aliases"):
                values.extend(_string_values(source.get(key)))
    return list(dict.fromkeys(value for value in values if value))


def _canonical_metric(value: Any) -> str | None:
    compact = _compact(value)
    if not compact:
        return None
    canonical_compact = { _compact(key): key for key in _METRIC_ALIASES }
    if compact in canonical_compact:
        return canonical_compact[compact]
    aliases = sorted(
        ((alias, canonical) for canonical, items in _METRIC_ALIASES.items() for alias in items),
        key=lambda item: len(_compact(item[0])),
        reverse=True,
    )
    for alias, canonical in aliases:
        if _compact(alias) and _compact(alias) in compact:
            return canonical
    return None


def _canonical_period(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).translate(_FULLWIDTH).strip().upper()
    quarter = re.search(
        r"(20\d{2}).*?(?:Q([1-4])|第?([一二三四1-4])季度)",
        text,
        flags=re.IGNORECASE,
    )
    if not quarter:
        quarter = re.search(r"Q([1-4])\s*(20\d{2})", text, flags=re.IGNORECASE)
        if quarter:
            return f"{quarter.group(2)}Q{quarter.group(1)}"
    if quarter:
        raw_quarter = quarter.group(2) or quarter.group(3)
        q = {"一": "1", "二": "2", "三": "3", "四": "4"}.get(raw_quarter, raw_quarter)
        return f"{quarter.group(1)}Q{q}"
    half = re.search(r"(20\d{2}).*?(?:H([12])|上半年|下半年)", text, flags=re.IGNORECASE)
    if half:
        h = half.group(2) or ("1" if "上半年" in text else "2")
        return f"{half.group(1)}H{h}"
    year = re.search(r"(?:FY\s*)?(20\d{2})", text, flags=re.IGNORECASE)
    return year.group(1) if year else None


def _canonical_value_type(value: Any, text: str, *, default_actual: bool) -> str | None:
    compact = _compact(value)
    if compact:
        if any(token in compact for token in ("adjusted", "nongaap", "经调整", "调整后")):
            return "adjusted"
        if any(token in compact for token in ("forecast", "estimate", "estimated", "consensus", "guidance", "预测", "预计", "预期", "一致预期")):
            return "forecast"
        if any(token in compact for token in ("actual", "reported", "实际", "已实现", "报告值")):
            return "actual"
    normalized_text = text.translate(_FULLWIDTH)
    if _ADJUSTED_RE.search(normalized_text):
        return "adjusted"
    if _FORECAST_RE.search(normalized_text):
        return "forecast"
    return "actual" if default_actual else None


def _canonical_conflict_dimension(value: Any) -> str | None:
    normalized = _compact(value)
    return normalized or None


def _canonical_unit(raw: Any) -> str | None:
    value = str(raw or "").translate(_FULLWIDTH).strip().lower().replace("％", "%").replace("／", "/")
    value = re.sub(r"\s+", " ", value)
    aliases = {
        "百分点": "个百分点",
        "percentage points": "percentage points",
        "percentage point": "percentage point",
        "pct points": "pct points",
        "pct point": "pct point",
    }
    value = aliases.get(value, value)
    return value if value in _UNIT_INFO else None


def _decimal(value: Any) -> Decimal | None:
    try:
        result = Decimal(str(value).translate(_FULLWIDTH).replace(",", ""))
    except (InvalidOperation, ValueError):
        return None
    return result if result.is_finite() else None


def _numeric_from_value(value: Any, unit: Any = None) -> _NumericValue | None:
    explicit_unit = _canonical_unit(unit)
    if isinstance(value, Mapping):
        return _numeric_from_value(
            value.get("formatted", value.get("value", value.get("result"))),
            value.get("unit", unit),
        )
    if isinstance(value, (int, float, Decimal)):
        number = _decimal(value)
        if number is None or explicit_unit is None:
            return None
        dimension, scale = _UNIT_INFO[explicit_unit]
        return _NumericValue(number * scale, explicit_unit, dimension)
    text = str(value or "").translate(_FULLWIDTH).strip()
    match = _NUMBER_WITH_UNIT_RE.search(text)
    if match:
        parsed_unit = _canonical_unit(match.group("unit"))
        number = _decimal(match.group("number"))
        if parsed_unit is None or number is None:
            return None
        dimension, scale = _UNIT_INFO[parsed_unit]
        return _NumericValue(number * scale, parsed_unit, dimension)
    if explicit_unit is not None:
        match = _PLAIN_NUMBER_RE.fullmatch(text.replace(" " , ""))
        number = _decimal(match.group(0)) if match else None
        if number is not None:
            dimension, scale = _UNIT_INFO[explicit_unit]
            return _NumericValue(number * scale, explicit_unit, dimension)
    return None


def _numeric(
    payload: Mapping[str, Any],
    text: str,
    *,
    preferred: _NumericValue | None = None,
) -> _NumericValue | None:
    unit = _first(payload, ("unit", "value_unit"))
    explicit = _first(payload, ("value", "raw_value", "claim_value"))
    if explicit is not None:
        parsed = _numeric_from_value(explicit, unit)
        if parsed is not None:
            return parsed
    matches = list(_NUMBER_WITH_UNIT_RE.finditer(text.translate(_FULLWIDTH)))
    if not matches:
        return None
    candidates = [_numeric_from_value(match.group(0)) for match in matches]
    candidates = [candidate for candidate in candidates if candidate is not None]
    if not candidates:
        return None
    if preferred is not None:
        # Evidence sentences commonly contain both the reported value and a
        # change value (for example, ``100 亿元，同比增长 20%``).  Select the
        # candidate that agrees with the claim's dimension/value instead of
        # assuming the final token is the reported fact.
        same_dimension = [item for item in candidates if item.dimension == preferred.dimension]
        exact = [item for item in same_dimension if _values_equal(item, preferred)]
        if exact:
            return exact[0]
        if same_dimension:
            return same_dimension[0]
    return candidates[0]


def _profile(
    payload: Mapping[str, Any],
    text: str,
    *,
    entity_catalog: Sequence[str],
    default_actual: bool,
    numeric_hint: _NumericValue | None = None,
) -> _Profile:
    numeric = _numeric(payload, text, preferred=numeric_hint)
    metric_value = _first(payload, ("metric", "metric_name", "financial_metric"))
    period_value = _first(payload, ("period", "fiscal_period", "report_period", "year"))
    type_value = _first(payload, ("value_type", "estimate_type", "actual_forecast"))
    return _Profile(
        entities=_entities(payload, text, entity_catalog),
        metric=_canonical_metric(metric_value) or _canonical_metric(text),
        period=_canonical_period(period_value) or _canonical_period(text),
        numeric=numeric,
        value_type=_canonical_value_type(type_value, text, default_actual=default_actual and numeric is not None),
        accounting_scope=_canonical_conflict_dimension(
            _first(payload, ("accounting_scope", "scope"))
        ),
        revision_status=_canonical_conflict_dimension(
            _first(payload, ("revision_status", "revision"))
        ),
    )


def _missing_numeric_constraints(profile: _Profile) -> list[str]:
    missing: list[str] = []
    if not profile.entities:
        missing.append("entity")
    if profile.metric is None:
        missing.append("metric")
    if profile.period is None:
        missing.append("period")
    if profile.numeric is None:
        missing.append("value")
    elif not profile.numeric.unit:
        missing.append("unit")
    if profile.value_type is None:
        missing.append("value_type")
    return missing


def _evidence_text(row: Mapping[str, Any]) -> str:
    parts = [
        row.get("support_span"), row.get("source_span"), row.get("child_text"),
        row.get("text"), row.get("raw_text"), row.get("section_title"), row.get("file_name"),
    ]
    return "\n".join(str(part) for part in parts if part)


def _evidence_id(row: Mapping[str, Any]) -> str:
    return str(row.get("evidence_id") or row.get("chunk_id") or row.get("fact_id") or "").strip()


def _evidence_ids(row: Mapping[str, Any]) -> set[str]:
    """Return every stable identifier by which an evidence row is addressable.

    Structured fact rows usually carry both a ``fact_id`` and a synthetic
    ``chunk_id``/``evidence_id``.  Calculations are allowed to reference the
    fact identifier directly, so validation must consider all aliases while
    keeping the canonical evidence id for ordinary claim citations.
    """
    return {
        str(row.get(key) or "").strip()
        for key in ("evidence_id", "chunk_id", "fact_id")
        if str(row.get(key) or "").strip()
    }


def _values_equal(left: _NumericValue, right: _NumericValue) -> bool:
    if left.dimension != right.dimension:
        return False
    tolerance = max(Decimal("1e-9"), abs(left.value) * Decimal("1e-9"))
    return abs(left.value - right.value) <= tolerance


def _same_base_coordinates(claim: _Profile, row: _Profile) -> bool:
    assert claim.numeric is not None and row.numeric is not None
    return (
        bool(claim.entities & row.entities)
        and claim.metric == row.metric
        and claim.period == row.period
        and claim.numeric.dimension == row.numeric.dimension
        and claim.value_type == row.value_type
    )


def _selected_conflict_dimensions_match(claim: _Profile, row: _Profile) -> bool:
    """Return whether a row remains eligible under explicit claim dimensions."""
    return (
        (claim.accounting_scope is None or claim.accounting_scope == row.accounting_scope)
        and (claim.revision_status is None or claim.revision_status == row.revision_status)
    )


def _complete_conflict_coordinates_match(claim: _Profile, row: _Profile) -> bool:
    """Require scope and revision on both sides before declaring conflict."""
    return (
        _same_base_coordinates(claim, row)
        and claim.accounting_scope is not None
        and row.accounting_scope is not None
        and claim.accounting_scope == row.accounting_scope
        and claim.revision_status is not None
        and row.revision_status is not None
        and claim.revision_status == row.revision_status
    )


def _numeric_relevance(claim: _Profile, row: _Profile) -> int:
    score = 0
    score += int(bool(claim.entities & row.entities))
    score += int(claim.metric is not None and claim.metric == row.metric)
    score += int(claim.period is not None and claim.period == row.period)
    score += int(claim.value_type is not None and claim.value_type == row.value_type)
    if claim.numeric is not None and row.numeric is not None:
        score += int(claim.numeric.dimension == row.numeric.dimension)
        score += int(_values_equal(claim.numeric, row.numeric))
    return score


def _verify_complete_numeric(
    claim: _Profile, rows: Sequence[Mapping[str, Any]], entity_catalog: Sequence[str]
) -> ClaimVerificationResult:
    supports: list[str] = []
    conflicts: list[str] = []
    unresolved_conflicts: list[str] = []
    relevant: list[tuple[int, str]] = []
    for row in rows:
        row_profile = _profile(
            row,
            _evidence_text(row),
            entity_catalog=entity_catalog,
            default_actual=True,
            numeric_hint=claim.numeric,
        )
        evidence_id = _evidence_id(row)
        relevance = _numeric_relevance(claim, row_profile)
        if relevance >= 2:
            relevant.append((relevance, evidence_id))
        if row_profile.numeric is None or _missing_numeric_constraints(row_profile):
            continue
        if _same_base_coordinates(claim, row_profile) and _selected_conflict_dimensions_match(
            claim, row_profile
        ):
            if _values_equal(claim.numeric, row_profile.numeric):
                supports.append(evidence_id)
            elif _complete_conflict_coordinates_match(claim, row_profile):
                conflicts.append(evidence_id)
            else:
                unresolved_conflicts.append(evidence_id)
    if conflicts:
        reasons = ["numeric_value_conflict"]
        if supports:
            reasons.append("support_and_conflict")
        return _result(CONTRADICTED, "deterministic", 0.0, reasons, supports + conflicts)
    if unresolved_conflicts:
        return _result(
            INSUFFICIENT,
            "deterministic",
            0.0,
            ["numeric_conflict_dimensions_incomplete"],
            supports + unresolved_conflicts,
        )
    if supports:
        return _result(ENTAILED, "deterministic", 1.0, ["all_numeric_constraints_match"], supports)
    relevant_ids = [item[1] for item in sorted(relevant, key=lambda item: item[0], reverse=True)]
    return _result(
        INSUFFICIENT,
        "deterministic",
        0.0,
        ["no_evidence_matches_all_numeric_constraints"],
        relevant_ids,
    )


def _resolve_calculation(lookup: CalculationLookup, calculation_id: str) -> Mapping[str, Any] | None:
    try:
        record = lookup(calculation_id) if callable(lookup) else lookup.get(calculation_id)
    except Exception:
        return None
    return record if isinstance(record, Mapping) else None


def _calculation_provenance(record: Mapping[str, Any]) -> tuple[list[str], bool]:
    direct = record.get("evidence_ids")
    if isinstance(direct, Sequence) and not isinstance(direct, (str, bytes)) and direct:
        ids = [str(item) for item in direct if item]
        return ids, len(ids) == len(direct)
    inputs = record.get("inputs") or record.get("provenance")
    if not isinstance(inputs, Sequence) or isinstance(inputs, (str, bytes)) or not inputs:
        return [], False
    ids: list[str] = []
    complete = True
    for item in inputs:
        if not isinstance(item, Mapping):
            complete = False
            continue
        evidence_id = str(item.get("evidence_id") or item.get("fact_id") or "").strip()
        if not evidence_id:
            complete = False
        else:
            ids.append(evidence_id)
    return list(dict.fromkeys(ids)), complete


def _calculation_numeric(record: Mapping[str, Any]) -> _NumericValue | None:
    for key in ("formatted", "result", "value"):
        if key in record:
            parsed = _numeric_from_value(record[key], record.get("unit"))
            if parsed is not None:
                return parsed
    return None


def _verify_calculation(
    calculation_id: str,
    claim: _Profile,
    lookup: CalculationLookup,
    rows: Sequence[Mapping[str, Any]],
    *,
    claim_text: str = "",
) -> ClaimVerificationResult | None:
    record = _resolve_calculation(lookup, calculation_id)
    if record is None:
        return _result(INSUFFICIENT, "calculation", 0.0, ["calculation_not_found"], [])
    status = str(record.get("status") or "success").strip().lower()
    if status not in {"success", "ok", "complete", "completed"}:
        return _result(INSUFFICIENT, "calculation", 0.0, ["calculation_not_successful"], [])
    ids, complete = _calculation_provenance(record)
    known_ids = set().union(*(_evidence_ids(row) for row in rows)) if rows else set()
    if not complete or not ids or any(item not in known_ids for item in ids):
        return _result(INSUFFICIENT, "calculation", 0.0, ["calculation_provenance_incomplete"], [item for item in ids if item in known_ids])
    comparison = record.get("comparison")
    if isinstance(comparison, Mapping) and claim.numeric is None:
        expected = _compact(str(comparison.get("answer") or ""))
        actual = _compact(claim_text)
        if expected and actual == expected:
            return _result(
                ENTAILED,
                "calculation",
                1.0,
                ["comparison_conclusion_and_provenance_match"],
                ids,
            )
        winner = _compact(str(comparison.get("winner") or ""))
        loser = _compact(str(comparison.get("loser") or ""))
        comparison_markers = ("更快", "更高", "高于", "领先", "较高", "相同", "持平")
        if loser and loser in actual and winner not in actual and any(
            marker in actual for marker in comparison_markers
        ):
            return _result(
                CONTRADICTED,
                "calculation",
                0.0,
                ["claim_contradicts_comparison_result"],
                ids,
            )
        return _result(
            INSUFFICIENT,
            "calculation",
            0.0,
            ["comparison_claim_does_not_match_calculation"],
            ids,
        )
    calculated = _calculation_numeric(record)
    if claim.numeric is None or calculated is None:
        return _result(INSUFFICIENT, "calculation", 0.0, ["calculation_result_unparseable"], ids)
    if not _values_equal(claim.numeric, calculated):
        return _result(CONTRADICTED, "calculation", 0.0, ["claim_value_differs_from_calculation"], ids)
    return _result(ENTAILED, "calculation", 1.0, ["calculation_result_and_provenance_match"], ids)


def _tokens(text: str) -> set[str]:
    normalized = str(text or "").translate(_FULLWIDTH).lower()
    tokens: set[str] = set()
    for part in re.findall(r"[a-z]{2,}|[\u4e00-\u9fff]+|\d+(?:\.\d+)?%?", normalized):
        if part in _GENERIC_TOKENS:
            continue
        if re.fullmatch(r"[\u4e00-\u9fff]+", part) and len(part) > 2:
            tokens.update(part[index:index + 2] for index in range(len(part) - 1))
        else:
            tokens.add(part)
    return tokens


def _overlap(claim_text: str, evidence_text: str) -> float:
    claim_tokens = _tokens(claim_text)
    evidence_tokens = _tokens(evidence_text)
    return len(claim_tokens & evidence_tokens) / max(len(claim_tokens), 1)


def _qualitative_candidates(
    text: str, rows: Sequence[Mapping[str, Any]], *, limit: int = 8
) -> list[Mapping[str, Any]]:
    scored = [(_overlap(text, _evidence_text(row)), index, row) for index, row in enumerate(rows)]
    selected = [item for item in sorted(scored, key=lambda item: (-item[0], item[1])) if item[0] > 0]
    return [item[2] for item in selected[:limit]]


def _has_negation(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _NEGATIONS)


def _without_negation(text: str) -> str:
    lowered = text.lower()
    for marker in sorted(_NEGATIONS, key=len, reverse=True):
        lowered = lowered.replace(marker, "")
    return _compact(lowered)


def _qualitative_conflict_coordinates(
    claim: Mapping[str, Any],
    claim_text: str,
    row: Mapping[str, Any],
    evidence_text: str,
    *,
    entity_catalog: Sequence[str],
) -> str:
    claim_profile = _profile(
        claim,
        claim_text,
        entity_catalog=entity_catalog,
        default_actual=True,
    )
    evidence_profile = _profile(
        row,
        evidence_text,
        entity_catalog=entity_catalog,
        default_actual=True,
    )
    claim_coordinates = (
        claim_profile.entities,
        claim_profile.metric,
        claim_profile.period,
        claim_profile.value_type,
        claim_profile.accounting_scope,
        claim_profile.revision_status,
    )
    evidence_coordinates = (
        evidence_profile.entities,
        evidence_profile.metric,
        evidence_profile.period,
        evidence_profile.value_type,
        evidence_profile.accounting_scope,
        evidence_profile.revision_status,
    )
    if (
        not claim_profile.entities
        or not evidence_profile.entities
        or any(value is None for value in claim_coordinates[1:])
        or any(value is None for value in evidence_coordinates[1:])
    ):
        return "incomplete"
    return "match" if claim_coordinates == evidence_coordinates else "different"


def _verify_qualitative_deterministically(
    claim: Mapping[str, Any],
    text: str,
    candidates: Sequence[Mapping[str, Any]],
) -> ClaimVerificationResult | None:
    claim_norm = _compact(text)
    supports: list[str] = []
    conflicts: list[str] = []
    unresolved_conflicts: list[str] = []
    entity_catalog = _entity_catalog(candidates)
    for row in candidates:
        evidence_text = _evidence_text(row)
        evidence_norm = _compact(evidence_text)
        same_without_negation = _without_negation(text) in _without_negation(evidence_text)
        opposite_polarity = _has_negation(text) != _has_negation(evidence_text)
        if same_without_negation and opposite_polarity and len(_without_negation(text)) >= 4:
            coordinate_state = _qualitative_conflict_coordinates(
                claim,
                text,
                row,
                evidence_text,
                entity_catalog=entity_catalog,
            )
            if coordinate_state == "match":
                conflicts.append(_evidence_id(row))
            elif coordinate_state == "incomplete":
                unresolved_conflicts.append(_evidence_id(row))
        elif claim_norm and claim_norm in evidence_norm:
            supports.append(_evidence_id(row))
    if conflicts:
        reasons = ["qualitative_polarity_conflict"]
        if supports:
            reasons.append("support_and_conflict")
        return _result(CONTRADICTED, "deterministic", 0.0, reasons, supports + conflicts)
    if unresolved_conflicts:
        return _result(
            INSUFFICIENT,
            "deterministic",
            0.0,
            ["qualitative_conflict_dimensions_incomplete"],
            supports + unresolved_conflicts,
        )
    if supports:
        return _result(ENTAILED, "deterministic", 1.0, ["claim_text_contained_in_evidence"], supports)
    return None


def _coerce_semantic(output: Any) -> tuple[VerificationStatus | None, float]:
    if isinstance(output, Mapping):
        raw_status = str(output.get("status") or "").strip().upper()
        status = raw_status if raw_status in _STATUSES else None
        raw_score = output.get("score", 0.0)
    else:
        status = None
        raw_score = output
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        return None, 0.0
    if not 0.0 <= score <= 1.0:
        return None, 0.0
    return status, score


def _verify_semantically(
    text: str, candidates: Sequence[Mapping[str, Any]], scorer: SemanticScorer, threshold: float
) -> tuple[ClaimVerificationResult | None, float, list[str]]:
    entailed: list[tuple[float, str]] = []
    contradicted: list[tuple[float, str]] = []
    nondirectional_high: list[tuple[float, str]] = []
    errors = 0
    best = 0.0
    for row in candidates:
        try:
            output = scorer(text, _evidence_text(row))
        except Exception:
            errors += 1
            continue
        status, score = _coerce_semantic(output)
        best = max(best, score)
        if status == CONTRADICTED and score >= threshold:
            contradicted.append((score, _evidence_id(row)))
        elif status == ENTAILED and score >= threshold:
            entailed.append((score, _evidence_id(row)))
        elif status is None and score >= threshold:
            nondirectional_high.append((score, _evidence_id(row)))
    if contradicted:
        return _result(CONTRADICTED, "semantic", max(item[0] for item in contradicted), ["semantic_contradiction"], [item[1] for item in contradicted]), best, []
    if entailed:
        return _result(ENTAILED, "semantic", max(item[0] for item in entailed), ["semantic_entailment"], [item[1] for item in entailed]), best, []
    reasons = ["semantic_undecidable"]
    if nondirectional_high:
        reasons.append("semantic_similarity_not_directional")
    if errors:
        reasons.append("semantic_scorer_error")
    return None, best, reasons


def _judge_payload(
    text: str, claim: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    return {
        "claim": {
            "claim_id": str(claim.get("claim_id") or ""),
            "text": text,
            "claim_type": str(claim.get("claim_type") or "EXTRACTED"),
        },
        "evidence": [
            {"evidence_id": _evidence_id(row), "text": _evidence_text(row)} for row in candidates
        ],
        "allowed_statuses": [ENTAILED, CONTRADICTED, INSUFFICIENT],
        "required_output_fields": ["status", "evidence_ids", "score", "reasons"],
    }


def _parse_judge_output(output: Any, allowed_ids: set[str]) -> tuple[VerificationStatus, list[str], float, list[str]] | None:
    if hasattr(output, "content"):
        output = output.content
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(output, Mapping):
        return None
    if not all(key in output for key in ("status", "evidence_ids", "score", "reasons")):
        return None
    raw_status = str(output["status"]).strip().upper()
    if raw_status not in _STATUSES:
        return None
    raw_ids = output["evidence_ids"]
    raw_reasons = output["reasons"]
    if not isinstance(raw_ids, list) or not all(isinstance(item, str) for item in raw_ids):
        return None
    if any(item not in allowed_ids for item in raw_ids):
        return None
    if raw_status in {ENTAILED, CONTRADICTED} and not raw_ids:
        return None
    if not isinstance(raw_reasons, list) or not raw_reasons or not all(isinstance(item, str) and item.strip() for item in raw_reasons):
        return None
    try:
        score = float(output["score"])
    except (TypeError, ValueError):
        return None
    if not 0.0 <= score <= 1.0:
        return None
    return raw_status, list(dict.fromkeys(raw_ids)), score, [item.strip() for item in raw_reasons]


def _verify_with_llm(
    text: str, claim: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]], judge: LLMJudge
) -> ClaimVerificationResult:
    payload = _judge_payload(text, claim, candidates)
    try:
        output = judge(payload)
    except Exception:
        return _result(INSUFFICIENT, "llm", 0.0, ["llm_judge_error"], [_evidence_id(row) for row in candidates])
    parsed = _parse_judge_output(output, {_evidence_id(row) for row in candidates})
    if parsed is None:
        return _result(INSUFFICIENT, "llm", 0.0, ["llm_judge_malformed"], [_evidence_id(row) for row in candidates])
    status, evidence_ids, score, reasons = parsed
    return _result(status, "llm", score, reasons, evidence_ids)


__all__ = [
    "CONTRADICTED",
    "ENTAILED",
    "INSUFFICIENT",
    "ClaimVerificationResult",
    "VerificationDetails",
    "VerificationStatus",
    "VerificationMethod",
    "VERIFICATION_METHODS",
    "verify_claim",
]
