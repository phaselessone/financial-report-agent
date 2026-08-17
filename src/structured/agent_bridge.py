"""Bridge structured facts into the agent evidence pipeline (checklist v3.0 §P6).

Turns a :class:`FinancialFact` into a high-confidence "evidence row" that the
existing ``_prepare_evidence`` / ``build_citations`` / ``validate_answer_support``
pipeline consumes first-class, so multi-hop synthesis can mix structured facts
(company + metric + period lookups) with RAG rows without changing the answerer.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

from src.structured.schema import FinancialFact, Metric

_METRIC_LABELS: dict[Metric, str] = {
    Metric.REVENUE: "营业收入",
    Metric.NET_PROFIT: "净利润",
    Metric.GROSS_MARGIN: "毛利率",
    Metric.RD_EXPENSE: "研发费用",
    Metric.OPERATING_CASH_FLOW: "经营性现金流",
}


def format_fact_value(value: Decimal, unit: str) -> str:
    if unit == "%":
        return f"{value}%"
    if abs(value) >= Decimal("1e8"):
        return f"{format((value / Decimal('1e8')).normalize(), 'f')}亿元"
    if abs(value) >= Decimal("1e4"):
        return f"{format((value / Decimal('1e4')).normalize(), 'f')}万元"
    return f"{value}元"


def format_fact_line(fact: FinancialFact) -> str:
    """Self-contained one-line rendering of a fact (company + period + metric + value)."""
    label = _METRIC_LABELS[fact.metric]
    return f"{fact.company} {fact.period.label} {label}({fact.value_type.value}) = {format_fact_value(fact.value, fact.unit)}"


def financial_fact_to_row(fact: FinancialFact) -> dict[str, Any]:
    """Map a structured fact to an evidence row compatible with the answerer.

    ``support_span`` carries the verbatim source span (citation fidelity); the
    ``text``/``child_text`` carry a self-contained line so the LLM sees company,
    period, metric and value together. ``score`` is set above the reranker
    ceiling so structured facts rank first when mixed with RAG rows.
    """
    line = f"{format_fact_line(fact)}；原文：{fact.source_span}"
    return {
        "chunk_id": f"structured:{fact.fact_id}",
        "doc_id": fact.doc_id,
        "evidence_id": fact.evidence_id,
        "file_name": f"{fact.doc_id}.pdf",
        "page_start": fact.page,
        "page_end": fact.page,
        "section_title": "",
        "section_path": "",
        "chunk_type": "structured_fact",
        "element_type": "paragraph",
        "support_span": line,
        "text": line,
        "child_text": line,
        "score": 5.0,
        "rerank_score": 5.0,
    }


def load_structured_context(facts_path: str | Path, aliases_path: str | Path) -> tuple[Any, dict[str, list[str]]] | tuple[None, None]:
    """Load the fact store and company aliases for agent consumption.

    Returns ``(None, None)`` when either file is missing so the agent can run
    without a structured layer (pure-RAG multi-hop). Both files must be present
    to enable structured routing.
    """
    facts_path = Path(facts_path)
    aliases_path = Path(aliases_path)
    if not facts_path.exists() or not aliases_path.exists():
        return None, None
    from src.evaluation.structured_eval import build_store_from_facts_jsonl
    from src.utils.io import read_json

    store = build_store_from_facts_jsonl(facts_path)
    aliases = read_json(aliases_path)
    return store, aliases
