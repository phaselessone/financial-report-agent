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

from src.structured.fact_store import FactStore
from src.structured.metric_registry import metric_label
from src.structured.period_normalizer import normalize_period
from src.structured.schema import FinancialFact, Metric, Period, PeriodBasis, ValueType
from src.structured.structured_query import (
    StructuredOperation,
    StructuredQuery,
    build_structured_query,
    execute_structured_query,
    extract_companies,
    extract_periods,
    parse_structured_query,
)

# All metrics in the central registry are eligible for a deterministic
# company+metric+period lookup.  The old name is retained as a compatibility
# alias for callers that imported it indirectly.
_LEGACY_ROUTE_METRICS = set(Metric)


@dataclass(frozen=True)
class FactQuery:
    company: str
    metric: Metric
    period: Period
    value_type: ValueType = ValueType.ACTUAL
    period_basis: PeriodBasis | None = None
    accounting_scope: str | None = None
    revision_status: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metric", Metric(self.metric))
        if not isinstance(self.period, Period):
            raise ValueError("FactQuery period must be a Period")
        object.__setattr__(self, "value_type", ValueType(self.value_type))
        if self.period_basis is not None:
            object.__setattr__(self, "period_basis", PeriodBasis(self.period_basis))
        for field_name in ("accounting_scope", "revision_status"):
            value = getattr(self, field_name)
            object.__setattr__(self, field_name, value.strip() if isinstance(value, str) and value.strip() else None)

    def to_structured_query(self) -> StructuredQuery:
        """Adapt the legacy single-fact shape to the canonical interface."""
        return build_structured_query(
            entities=[self.company],
            metrics=[self.metric],
            periods=[self.period],
            operation="lookup",
            value_type=self.value_type,
            period_basis=self.period_basis,
            accounting_scope=self.accounting_scope,
            revision_status=self.revision_status,
        )

    @classmethod
    def from_structured_query(cls, query: StructuredQuery) -> "FactQuery":
        if (
            query.operation != "lookup"
            or len(query.entities) != 1
            or len(query.metrics) != 1
            or len(query.periods) != 1
            or not isinstance(query.periods[0], Period)
            or query.value_type is None
        ):
            raise ValueError("structured query cannot be represented by legacy FactQuery")
        return cls(
            company=query.entities[0],
            metric=query.metrics[0],
            period=query.periods[0],
            value_type=query.value_type,
            period_basis=query.period_basis,
            accounting_scope=query.accounting_scope,
            revision_status=query.revision_status,
        )


def detect_query_signature(
    query: str,
    *,
    company_aliases: dict[str, list[str]],
    anchor_year: int | None = None,
) -> FactQuery | None:
    structured = parse_structured_query(query, company_aliases=company_aliases, anchor_year=anchor_year)
    if (
        structured is None
        or structured.operation != "lookup"
        or len(structured.entities) != 1
        or len(structured.metrics) != 1
        or len(structured.periods) != 1
        or not isinstance(structured.periods[0], Period)
        or structured.metrics[0] not in _LEGACY_ROUTE_METRICS
    ):
        return None
    return FactQuery.from_structured_query(structured)


def _format_value(value: Decimal, unit: str) -> str:
    if unit in {"元/股", "元每股"}:
        return f"{value}元/股"
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
            "answer": f"结构化事实库中未找到 {signature.company} {signature.period.label} {metric_label(signature.metric)} 的记录。",
            "citations": [],
        }
    lines: list[str] = []
    citations: list[dict[str, Any]] = []
    for fact in facts:
        lines.append(
            f"{fact.period.label} {fact.company} {metric_label(fact.metric)}({fact.value_type.value}): "
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
    structured_query = signature.to_structured_query()
    structured_result = execute_structured_query(store, structured_query)
    facts = list(structured_result.facts)
    answer = build_structured_answer(signature, facts)
    return {
        "routed": True,
        "fact_query": signature,
        "structured_query": structured_query,
        "structured_status": structured_result.status,
        "structured_reason": structured_result.reason,
        "facts": facts,
        "answer": answer["answer"],
        "citations": answer["citations"],
    }
