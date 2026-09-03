"""Controlled, deterministic tool planning and execution.

This module is the seam between the LangGraph nodes and the existing financial
tool implementations.  Callers learn two small interfaces:

``plan_next_tool`` chooses one whitelisted action without executing it, and
``execute_tool_plan`` executes exactly that action while returning a normalized,
observable result.  No model-produced code, shell command, or dynamic import is
ever accepted.
"""

from __future__ import annotations

import hashlib
import json
import re
from time import perf_counter
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from src.agent.calculation_operand_resolver import (
    CalculationOperandResolver,
    OperandRequirement,
    ResolvedOperand,
)
from src.agent.calculation_provenance import execute_calculation
from src.structured.agent_bridge import financial_fact_to_row
from src.structured.fact_normalizer import normalize_company_name
from src.structured.fact_query import detect_query_signature, route
from src.tools.citation_verify import verify_citations
from src.tools.evidence_search import lookup_evidence_in_pool
from src.tools.financial_calculator import parse_numeric_value
from src.tools.report_search import ReportSearchTool, top_rows

TOOL_ACTIONS = frozenset(
    {
        "structured_lookup",
        "report_search",
        "evidence_search",
        "calculator",
        "citation_verify",
        "finish",
    }
)
TOOL_STATUSES = frozenset({"SUCCESS", "FAILED", "SKIPPED"})

_CALC_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("percentage_point_change", ("百分点", "percentage point", "pp change", "pp变化")),
    ("cagr", ("cagr", "复合年增长率", "年复合增长率", "复合增长")),
    ("qoq", ("qoq", "环比", "环比增长", "环比增速")),
    ("yoy", ("yoy", "同比", "同比增长", "同比增速")),
    ("difference", ("difference", "差额", "差值", "相差", "相差多少")),
    ("ratio", ("ratio", "占比", "比例", "比率", "倍数", "倍")),
    ("growth_rate", ("growth rate", "增长率", "增速", "涨幅", "跌幅")),
)
_METRIC_LABELS: dict[str, tuple[str, ...]] = {
    "revenue": ("营业收入", "营业总收入", "营收", "revenue"),
    "net_profit": ("净利润", "归母净利润", "归属母公司净利润", "net profit"),
    "gross_profit": ("毛利润", "毛利", "gross profit"),
    "gross_margin": ("毛利率", "gross margin"),
    "net_margin": ("净利率", "净利润率", "net margin"),
    "rd_expense": ("研发费用", "研发投入", "研发支出", "r&d"),
    "operating_cash_flow": ("经营性现金流", "经营现金流", "经营活动现金流量净额"),
}
_PERIOD_KIND_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Q1", ("q1", "一季度", "第一季度", "一季报")),
    ("H1", ("h1", "上半年", "半年报")),
    ("Q3", ("q3", "三季度", "第三季度", "前三季度", "三季报")),
)
_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
_NUM_WITH_UNIT_RE = re.compile(r"[+-]?\d[\d,]*(?:\.\d+)?\s*(?:亿元?|万元?|千元?|元|%|个百分点|pp)?", re.IGNORECASE)


def _row_text(row: Mapping[str, Any]) -> str:
    return " ".join(
        str(row.get(key) or "") for key in ("support_span", "child_text", "text")
    )


def _calculation_fact_from_row(
    row: Mapping[str, Any],
    *,
    operand: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Extract one conservative calculator operand from a report row.

    Report search rows do not necessarily contain a pre-normalized
    ``FinancialFact``.  We only promote a row when the requested company,
    metric, and period markers are present and exactly one numeric token can
    be parsed.  The raw token is retained so the Decimal calculator performs
    the unit conversion itself; the row's evidence/chunk provenance is copied
    through unchanged.
    """
    operand = operand if isinstance(operand, Mapping) else {}
    text = _row_text(row)
    if not text.strip():
        return None
    company = str(operand.get("company") or "").strip()
    if company and company not in text:
        return None
    metric = str(operand.get("metric") or "").strip()
    labels = _METRIC_LABELS.get(metric, (metric,))
    if metric and not any(label and label.lower() in text.lower() for label in labels):
        return None
    period = operand.get("period")
    if isinstance(period, Mapping):
        year = period.get("year")
        kind = str(period.get("kind") or "FY")
        if year is not None and str(year) not in text:
            return None
        if kind != "FY" and kind.lower() not in text.lower():
            # Chinese quarter/half-year markers are accepted as equivalents.
            kind_markers = {
                "Q1": ("一季度", "第一季度", "q1"),
                "H1": ("上半年", "半年报", "h1"),
                "Q3": ("三季度", "第三季度", "前三季度", "q3"),
            }
            if not any(marker in text.lower() for marker in kind_markers.get(kind, ())):
                return None
    tokens: list[str] = []
    for match in _NUM_WITH_UNIT_RE.finditer(text):
        token = match.group(0).strip().replace(" ", "")
        numeric = token.rstrip("亿元万元千元元%个百分点pp")
        if not numeric or numeric in {"+", "-"}:
            continue
        if re.fullmatch(r"(?:19|20)\d{2}", numeric) and token == numeric:
            continue
        tokens.append(token)
    if len(tokens) != 1:
        return None
    token = tokens[0]
    try:
        parse_numeric_value(token)
    except Exception:
        return None
    evidence_id = str(row.get("evidence_id") or row.get("chunk_id") or "").strip() or None
    return {
        "name": str(operand.get("name") or ""),
        "value": token,
        "unit": None,
        "evidence_id": evidence_id,
        "chunk_id": row.get("chunk_id"),
        "doc_id": row.get("doc_id"),
        "raw_value": token,
        "source_span": text,
        "period": period,
    }


def _period_label(period: Any) -> str:
    if isinstance(period, Mapping):
        return f"{period.get('kind', 'FY')}{period.get('year', '')}"
    return str(period or "").strip()


def _resolve_calculation_fact_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    operand: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Resolve one requested operand across all report rows, failing closed."""
    company = str(operand.get("company") or operand.get("entity") or "").strip()
    metric = str(operand.get("metric") or "").strip()
    period = _period_label(operand.get("period"))
    if not company or not metric or not period:
        return None
    requirement = OperandRequirement(
        name=str(operand.get("calculation_operand") or operand.get("name") or "").strip(),
        entity=company,
        metric=metric,
        period=period,
        value_type=str(operand.get("value_type") or "actual"),
        unit=str(operand.get("unit") or "").strip() or None,
        accounting_scope=str(operand.get("accounting_scope") or "").strip() or None,
    )
    resolution = CalculationOperandResolver().resolve(
        requirement,
        structured_facts=[],
        evidence_rows=rows,
    )
    if not isinstance(resolution, ResolvedOperand):
        return None
    item = resolution.operand
    return {
        "name": item.name,
        "value": str(item.value),
        "unit": item.unit,
        "evidence_id": item.evidence_id,
        "fact_id": item.fact_id,
        "chunk_id": item.chunk_id,
        "doc_id": item.doc_id,
        "raw_value": item.raw_text,
        "period": operand.get("period"),
    }


def _calculation_operation(query: str) -> str | None:
    lowered = query.lower()
    for operation, markers in _CALC_MARKERS:
        if any(marker.lower() in lowered for marker in markers):
            return operation
    # A margin query without an explicit arithmetic marker is a fact lookup,
    # not a gross_profit / revenue calculation. Formula terms opt in.
    if any(token in query for token in ("毛利/", "毛利除以", "毛利润/", "毛利润除以")):
        return "gross_margin"
    if any(token in query for token in ("净利润/", "净利除以", "净利润除以")):
        return "net_margin"
    return None


def _metric_candidates(query: str) -> list[str]:
    candidates: list[tuple[int, str]] = []
    for metric, labels in _METRIC_LABELS.items():
        for label in labels:
            position = query.lower().find(label.lower())
            if position >= 0:
                candidates.append((position, metric))
                break
    return [metric for _position, metric in sorted(candidates)]


def _period_kind(query: str) -> str:
    lowered = query.lower()
    for kind, markers in _PERIOD_KIND_MARKERS:
        if any(marker.lower() in lowered for marker in markers):
            return kind
    return "FY"


def _number_inputs(query: str) -> list[str]:
    values: list[str] = []
    for match in _NUM_WITH_UNIT_RE.finditer(query):
        token = match.group(0).strip().replace(" ", "")
        numeric = token.rstrip("亿元万元千元元%个百分点pp")
        if numeric in {"", "+", "-"}:
            continue
        # Bare report years are periods, not operands.
        if token == numeric and re.fullmatch(r"(?:19|20)\d{2}", numeric):
            continue
        values.append(token)
    return values


def _period_spec(company: str | None, metric: str, year: int, kind: str) -> dict[str, Any]:
    labels = _METRIC_LABELS.get(metric, (metric,))
    label = labels[0]
    company_text = company or ""
    period_label = f"{kind}{year}" if kind != "FY" else str(year)
    return {
        "name": "",
        "company": company,
        "metric": metric,
        "period": {"kind": kind, "year": year},
        "query": f"{company_text}{period_label}{label}",
    }


def detect_calculation_intent(
    query: str, *, company_aliases: Mapping[str, Sequence[str]] | None = None
) -> dict[str, Any] | None:
    """Infer a bounded calculation request from an explicit financial question.

    The result is metadata only; no retrieval or arithmetic occurs here. It is
    deliberately conservative: a query must contain a recognized operation
    marker and enough metric/period/value information to name two operands.
    """
    if not isinstance(query, str) or not query.strip():
        return None
    operation = _calculation_operation(query)
    if operation is None:
        return None
    aliases = company_aliases or {}
    company = normalize_company_name(query, aliases) if aliases else None
    years = [int(item) for item in _YEAR_RE.findall(query)]
    years = list(dict.fromkeys(years))
    kind = _period_kind(query)
    metrics = _metric_candidates(query)
    values = _number_inputs(query)
    operands: list[dict[str, Any]] = []

    if operation in {"yoy", "qoq", "growth_rate", "cagr"}:
        metric = next((item for item in metrics if item not in {"gross_profit", "gross_margin", "net_margin"}), None)
        if metric is None and metrics:
            metric = metrics[0]
        if len(years) >= 2 and metric:
            periods = [years[0], years[-1]]
        elif len(years) == 1 and metric and operation in {"yoy", "qoq"}:
            periods = [years[0] - 1, years[0]]
        else:
            periods = []
        if metric and len(periods) == 2:
            names = ("prior", "current") if operation in {"yoy", "qoq"} else ("start", "end")
            for name, year in zip(names, periods):
                spec = _period_spec(company, metric, year, kind)
                spec["name"] = name
                operands.append(spec)
        elif len(values) >= 2:
            names = ("prior", "current") if operation in {"yoy", "qoq"} else ("start", "end")
            operands = [{"name": name, "value": value} for name, value in zip(names, values[:2])]
    elif operation in {"ratio", "difference"}:
        if len(metrics) >= 2:
            period = years[-1] if years else None
            if period is not None:
                names = ("numerator", "denominator") if operation == "ratio" else ("left", "right")
                for name, metric in zip(names, metrics[:2]):
                    spec = _period_spec(company, metric, period, kind)
                    spec["name"] = name
                    operands.append(spec)
        elif len(years) >= 2 and metrics:
            names = ("numerator", "denominator") if operation == "ratio" else ("left", "right")
            for name, year in zip(names, (years[0], years[-1])):
                spec = _period_spec(company, metrics[0], year, kind)
                spec["name"] = name
                operands.append(spec)
        elif len(years) == 1 and metrics and operation == "difference":
            for name, year in zip(("left", "right"), (years[0] - 1, years[0])):
                spec = _period_spec(company, metrics[0], year, kind)
                spec["name"] = name
                operands.append(spec)
        elif len(values) >= 2:
            names = ("numerator", "denominator") if operation == "ratio" else ("left", "right")
            operands = [{"name": name, "value": value} for name, value in zip(names, values[:2])]
    elif operation == "percentage_point_change":
        if len(values) >= 2 and any("%" in value for value in values):
            operands = [
                {"name": "prior_pct", "value": values[0], "unit": "%"},
                {"name": "current_pct", "value": values[1], "unit": "%"},
            ]
        elif len(years) >= 2 and metrics:
            for name, year in zip(("prior_pct", "current_pct"), (years[0], years[-1])):
                spec = _period_spec(company, metrics[0], year, kind)
                spec["name"] = name
                operands.append(spec)
    elif operation in {"gross_margin", "net_margin"}:
        # Formula-style margin questions explicitly name the two source facts.
        source_metrics = ("gross_profit", "revenue") if operation == "gross_margin" else ("net_profit", "revenue")
        period = years[-1] if years else None
        if period is not None:
            for name, metric in zip(source_metrics, source_metrics):
                spec = _period_spec(company, metric, period, kind)
                spec["name"] = name
                operands.append(spec)

    if len(operands) != 2:
        return None
    if operation == "cagr":
        request_years = years[-1] - years[0] if len(years) >= 2 else None
    else:
        request_years = None
    return {
        "operation": operation,
        "query": query,
        "operands": operands,
        "years": request_years,
        "precision": 4,
        "fact_first": any("metric" in operand for operand in operands),
    }


def _jsonable(value: Any) -> Any:
    """Return a deterministic JSON-safe representation for ids and logs."""
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_jsonable(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True)) if isinstance(value, (set, frozenset)) else items
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def tool_call_key(action: str, arguments: dict[str, Any]) -> str:
    payload = json.dumps(
        {"action": action, "arguments": _jsonable(arguments)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_tool_plan(action: str, arguments: dict[str, Any] | None = None, *, reason: str = "") -> dict[str, Any]:
    if action not in TOOL_ACTIONS:
        raise ValueError(f"tool action is not whitelisted: {action!r}")
    safe_arguments = _jsonable(arguments or {})
    if not isinstance(safe_arguments, dict):
        raise ValueError("tool arguments must be a mapping")
    return {
        "next_action": action,
        "arguments": safe_arguments,
        "reason": str(reason or ""),
        "call_key": tool_call_key(action, safe_arguments),
    }


def plan_next_tool(
    *,
    query: str,
    domain_hint: str = "",
    fact_store: Any = None,
    company_aliases: dict[str, list[str]] | None = None,
    attempted_call_keys: set[str] | None = None,
    calculation_intent: dict[str, Any] | None = None,
    calculation_facts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Choose the shortest deterministic path for a single-hop query.

    A complete company+metric+period signature tries the structured store once.
    A miss falls back to report search.  Every other query searches reports
    directly.  Repeating the exact same action and arguments returns ``finish``
    instead of executing an unproductive loop.
    """
    attempted = set(attempted_call_keys or set())
    aliases = company_aliases or {}
    if calculation_intent is None:
        calculation_intent = detect_calculation_intent(query, company_aliases=aliases)

    if calculation_intent:
        calc_plan = _plan_calculation(
            intent=calculation_intent,
            facts=calculation_facts or {},
            attempted_call_keys=attempted,
            fact_store=fact_store,
        )
        if calc_plan is not None:
            return calc_plan

    if fact_store is not None:
        signature = detect_query_signature(query, company_aliases=aliases)
        if signature is not None:
            arguments = {
                "query": query,
                "company": signature.company,
                "metric": signature.metric.value,
                "period": signature.period.to_dict(),
            }
            plan = make_tool_plan("structured_lookup", arguments, reason="complete_structured_signature")
            if plan["call_key"] not in attempted:
                return plan

    search_plan = make_tool_plan(
        "report_search",
        {"query": query, "domain_hint": domain_hint},
        reason="structured_unavailable_or_missed",
    )
    if search_plan["call_key"] not in attempted:
        return search_plan
    return make_tool_plan("finish", {}, reason="duplicate_tool_call")


def _fact_to_calculation_input(name: str, fact: Any) -> dict[str, Any] | None:
    """Convert a FinancialFact/to_dict row into the calculator operand shape."""
    if hasattr(fact, "to_dict") and callable(fact.to_dict):
        fact = fact.to_dict()
    if not isinstance(fact, Mapping):
        return None
    value = fact.get("value")
    if value is None:
        value = fact.get("raw_value")
    if value is None:
        value = fact.get("support_span") or fact.get("text")
    if value is None:
        return None
    period = fact.get("period")
    if isinstance(period, Mapping):
        period = f"{period.get('kind', 'FY')}{period.get('year', '')}"
    return {
        "name": name,
        "value": value,
        "unit": fact.get("unit"),
        "evidence_id": fact.get("evidence_id"),
        "fact_id": fact.get("fact_id"),
        "doc_id": fact.get("doc_id"),
        "raw_text": fact.get("raw_value") or fact.get("source_span"),
        "period": period,
    }


def _calculation_inputs(
    intent: Mapping[str, Any], facts: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inputs: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for operand in intent.get("operands") or []:
        if not isinstance(operand, Mapping):
            continue
        name = str(operand.get("name") or "")
        fact = facts.get(name)
        if isinstance(fact, (list, tuple)):
            fact = fact[0] if fact else None
        converted = _fact_to_calculation_input(name, fact)
        if converted is None and "value" in operand:
            converted = {
                "name": name,
                "value": operand.get("value"),
                "unit": operand.get("unit"),
                "evidence_id": operand.get("evidence_id"),
                "fact_id": operand.get("fact_id"),
                "period": operand.get("period"),
            }
        if converted is None:
            missing.append(dict(operand))
        else:
            inputs.append(converted)
    return inputs, missing


def _plan_calculation(
    *,
    intent: Mapping[str, Any],
    facts: Mapping[str, Any],
    attempted_call_keys: set[str],
    fact_store: Any = None,
) -> dict[str, Any] | None:
    operation = str(intent.get("operation") or "").strip().lower()
    inputs, missing = _calculation_inputs(intent, facts)
    if missing:
        operand = missing[0]
        query = str(operand.get("query") or "").strip()
        if not query:
            return make_tool_plan("finish", {}, reason="calculation_missing_operand_query")
        # Keep the route explicit even when the optional store is absent.  The
        # executor records a successful zero-row structured miss, after which
        # the planner can take the report-search fallback without hiding the
        # intended fact-first policy.
        structured = make_tool_plan(
            "structured_lookup",
            {
                "query": query,
                "calculation_operand": str(operand.get("name") or ""),
                "calculation_operation": operation,
                "company": operand.get("company"),
                "metric": operand.get("metric"),
                "period": operand.get("period"),
            },
            reason="calculation_fact_first",
        )
        if structured["call_key"] not in attempted_call_keys:
            return structured
        report = make_tool_plan(
            "report_search",
            {
                "query": query,
                "domain_hint": "",
                "calculation_operand": str(operand.get("name") or ""),
                "calculation_operation": operation,
                "company": operand.get("company"),
                "metric": operand.get("metric"),
                "period": operand.get("period"),
            },
            reason="calculation_structured_miss_fallback",
        )
        if report["call_key"] not in attempted_call_keys:
            return report
        return make_tool_plan("finish", {}, reason="calculation_fact_unavailable")

    calculator_arguments = {
        "operation": operation,
        "inputs": inputs,
        "precision": int(intent.get("precision", 4) or 4),
    }
    if intent.get("years") is not None:
        calculator_arguments["years"] = intent.get("years")
    if intent.get("claimed_result") is not None:
        calculator_arguments["claimed_result"] = intent.get("claimed_result")
    calculator = make_tool_plan("calculator", calculator_arguments, reason="calculation_facts_ready")
    if calculator["call_key"] not in attempted_call_keys:
        return calculator
    return make_tool_plan("finish", {}, reason="duplicate_tool_call")


def _summary(action: str, payload: dict[str, Any]) -> str:
    if action in {"structured_lookup", "report_search"}:
        return f"rows={len(payload.get('rows') or [])}"
    if action == "evidence_search":
        return "found" if payload.get("found") else "not_found"
    if action == "citation_verify":
        return "valid" if payload.get("valid") else "invalid"
    if action == "calculator":
        record = payload.get("calculation") or {}
        return f"calculation_status={record.get('status', 'UNKNOWN')}"
    return str(payload.get("reason") or action)[:200]


def execute_tool_plan(
    plan: dict[str, Any],
    *,
    runtime: Any = None,
    fact_store: Any = None,
    company_aliases: dict[str, list[str]] | None = None,
    evidence_pool: dict[str, dict[str, Any]] | None = None,
    calculation_executor: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute one validated plan and return a normalized observation."""
    started = perf_counter()
    action = str(plan.get("next_action") or "")
    arguments = plan.get("arguments") or {}
    if action not in TOOL_ACTIONS:
        return {
            "status": "FAILED",
            "payload": {},
            "result_summary": "non_whitelisted_tool",
            "error_type": "TOOL_SELECTION_ERROR",
            "latency_ms": round((perf_counter() - started) * 1000, 2),
        }
    if not isinstance(arguments, dict):
        return {
            "status": "FAILED",
            "payload": {},
            "result_summary": "malformed_arguments",
            "error_type": "INVALID_ARGUMENTS",
            "latency_ms": round((perf_counter() - started) * 1000, 2),
        }

    try:
        if action == "finish":
            payload = {"reason": plan.get("reason") or "finished"}
            status = "SKIPPED"
        elif action == "structured_lookup":
            if fact_store is None:
                routed = None
                facts = []
            else:
                routed = route(
                    str(arguments.get("query") or ""),
                    store=fact_store,
                    company_aliases=company_aliases or {},
                )
                facts = list((routed or {}).get("facts") or [])
            payload = {
                "routed": bool(routed),
                "facts": [fact.to_dict() for fact in facts],
                "rows": [financial_fact_to_row(fact) for fact in facts],
                "answer": (routed or {}).get("answer", ""),
                "citations": (routed or {}).get("citations", []),
            }
            status = "SUCCESS"
        elif action == "report_search":
            if runtime is None:
                raise ValueError("retrieval runtime is unavailable")
            result = ReportSearchTool(runtime, str(arguments.get("domain_hint") or "")).search(
                str(arguments.get("query") or "")
            )
            rows = top_rows(result)
            operand_fact = None
            if arguments.get("calculation_operand"):
                operand_fact = _resolve_calculation_fact_from_rows(
                    [row for row in rows if isinstance(row, Mapping)],
                    operand=arguments,
                )
            payload = {
                "result": result,
                "rows": rows,
                "calculation_facts": [operand_fact] if operand_fact else [],
            }
            status = "SUCCESS"
        elif action == "evidence_search":
            payload = lookup_evidence_in_pool(
                str(arguments.get("ref") or ""), pool=evidence_pool or {}
            )
            status = "SUCCESS"
        elif action == "citation_verify":
            rows = list((evidence_pool or {}).values())
            payload = verify_citations(
                query=str(arguments.get("query") or ""),
                question_type=str(arguments.get("question_type") or "fact"),
                final_answer=str(arguments.get("final_answer") or ""),
                evidence_summary=str(arguments.get("evidence_summary") or ""),
                used_evidence_ids=list(arguments.get("used_evidence_ids") or []),
                evidence_rows=rows,
                answer_mode=arguments.get("answer_mode"),
            )
            status = "SUCCESS"
        else:  # calculator
            executor = calculation_executor or execute_calculation
            calculation = executor(**arguments)
            payload = {"calculation": _jsonable(calculation)}
            status = "SUCCESS"
        error_type = None
    except Exception as exc:  # execution failures are data, never hidden
        payload = {}
        status = "FAILED"
        error_type = type(exc).__name__
        result_summary = str(exc)[:200]
    else:
        result_summary = _summary(action, payload)

    return {
        "status": status,
        "payload": payload,
        "result_summary": result_summary,
        "error_type": error_type,
        "latency_ms": round((perf_counter() - started) * 1000, 2),
    }
