"""Structured fact query and routing (checklist v3.0 §P5).

Queries with an obvious company + metric + period signature route to the
structured fact store; everything else returns None and stays on the RAG
path. Routing and answer building are local deterministic functions — no
LLM calls on the structured path.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from src.structured.fact_normalizer import match_metric, normalize_company_name
from src.structured.period_normalizer import normalize_period
from src.structured.schema import FinancialFact, Metric, Period
from src.structured.fact_store import FactStore

_METRIC_LABELS: dict[Metric, str] = {
    Metric.REVENUE: "营业收入",
    Metric.NET_PROFIT: "净利润",
    Metric.GROSS_MARGIN: "毛利率",
    Metric.RD_EXPENSE: "研发费用",
    Metric.OPERATING_CASH_FLOW: "经营性现金流",
}


@dataclass(frozen=True)
class FactQuery:
    company: str
    metric: Metric
    period: Period


def detect_query_signature(
    query: str,
    *,
    company_aliases: dict[str, list[str]],
    anchor_year: int | None = None,
) -> FactQuery | None:
    """Return a :class:`FactQuery` when the query names all three elements."""
    if not isinstance(query, str) or not query.strip():
        return None
    company = normalize_company_name(query, company_aliases)
    if company is None:
        return None
    metric = match_metric(query)
    if metric is None:
        return None
    period = normalize_period(query, anchor_year=anchor_year)
    if period is None:
        return None
    return FactQuery(company=company, metric=metric, period=period)


def _format_value(value: Decimal, unit: str) -> str:
    if unit == "%":
        return f"{value}%"
    if abs(value) >= Decimal("1e8"):
        return f"{format((value / Decimal('1e8')).normalize(), 'f')}亿元"
    if abs(value) >= Decimal("1e4"):
        return f"{format((value / Decimal('1e4')).normalize(), 'f')}万元"
    return f"{value}元"


def build_structured_answer(signature: FactQuery, facts: list[FinancialFact]) -> dict[str, Any]:
    """Build a deterministic structured answer with provenance citations."""
    if not facts:
        return {
            "answer": f"结构化事实库中未找到 {signature.company} {signature.period.label} {_METRIC_LABELS[signature.metric]} 的记录。",
            "citations": [],
        }
    lines: list[str] = []
    citations: list[dict[str, Any]] = []
    for fact in facts:
        lines.append(
            f"{fact.period.label} {fact.company} {_METRIC_LABELS[fact.metric]}({fact.value_type.value}): "
            f"{_format_value(fact.value, fact.unit)}"
        )
        citations.append(
            {
                "evidence_id": fact.evidence_id,
                "doc_id": fact.doc_id,
                "page": fact.page,
                "snippet": fact.source_span,
            }
        )
    return {"answer": "；".join(lines) + "。", "citations": citations}


def route(
    query: str,
    *,
    store: FactStore,
    company_aliases: dict[str, list[str]],
    anchor_year: int | None = None,
) -> dict[str, Any] | None:
    """Route a query to structured search; None means the query stays on RAG."""
    signature = detect_query_signature(query, company_aliases=company_aliases, anchor_year=anchor_year)
    if signature is None:
        return None
    facts = store.query(company=signature.company, metric=signature.metric, period=signature.period)
    answer = build_structured_answer(signature, facts)
    return {
        "routed": True,
        "fact_query": signature,
        "facts": facts,
        "answer": answer["answer"],
        "citations": answer["citations"],
    }
