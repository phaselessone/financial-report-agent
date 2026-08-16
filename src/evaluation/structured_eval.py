"""Structured-fact benchmark evaluation (checklist v3.0 §P5).

Compares two pipelines per numeric benchmark query:

* Structured Fact Search: local deterministic routing -> fact store -> exact
  value with provenance. Zero LLM calls.
* Text RAG: the existing retrieval + answerer pipeline (remote LLM).

Metrics per query: Exact Numeric Accuracy, Period Accuracy, Value Type
Accuracy, Citation Accuracy. Gold rows come from
``data/eval_set/structured_fact_seed_dev.jsonl``.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from src.structured.fact_store import FactStore
from src.structured.fact_query import route
from src.structured.schema import FinancialFact, Metric, Period, PeriodType, ValueType
from src.structured.unit_normalizer import normalize_fact_value
from src.structured.value_type import classify_value_type
from src.utils.io import ensure_dir, read_jsonl, write_jsonl

NUM_TOKEN_RE = re.compile(r"[+-]?\d[\d,]*(?:\.\d+)?\s*(?:亿元|万元|千元|亿|万|%)")


def load_structured_seed(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def build_store_from_facts_jsonl(path: Path, *, db_path: str | Path = ":memory:") -> FactStore:
    store = FactStore(db_path)
    facts = [FinancialFact.from_dict(json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    store.upsert(facts)
    return store


def gold_to_fact(row: dict[str, Any]) -> FinancialFact:
    return FinancialFact(
        company=row["company"],
        metric=row["metric"],
        period=Period.from_dict(row["period"]),
        value_type=row["value_type"],
        value=Decimal(row["value"]),
        unit=row["unit"],
        doc_id=row["doc_id"],
        page=row["page"],
        evidence_id=row["evidence_id"],
        raw_value=row["raw_value"],
        source_span=row["source_span"],
    )


def _gold_value(row: dict[str, Any]) -> Decimal:
    return Decimal(row["value"])


def evaluate_structured_row(row: dict[str, Any], store: FactStore, aliases: dict[str, list[str]]) -> dict[str, Any]:
    result = route(row["query"], store=store, company_aliases=aliases)
    if result is None or not result["facts"]:
        return {
            "qid": row["qid"],
            "routed": bool(result),
            "hit": False,
            "numeric_correct": False,
            "period_correct": False,
            "value_type_correct": False,
            "citation_correct": False,
            "answer": result["answer"] if result else "",
        }
    facts = result["facts"]
    gold_period = Period.from_dict(row["period"])
    gold_value = _gold_value(row)
    citations = result["citations"]
    return {
        "qid": row["qid"],
        "routed": True,
        "hit": True,
        "numeric_correct": any(fact.value == gold_value for fact in facts),
        "period_correct": any(fact.period == gold_period for fact in facts),
        "value_type_correct": any(fact.value_type.value == row["value_type"] for fact in facts),
        "citation_correct": any(citation["doc_id"] == row["doc_id"] for citation in citations),
        "answer": result["answer"],
    }


def _answer_numeric_values(answer: str) -> list[Decimal]:
    values: list[Decimal] = []
    for token in NUM_TOKEN_RE.findall(answer):
        try:
            from src.tools.financial_calculator import CalculationError, parse_numeric_value

            value, _scale, _unit = parse_numeric_value(token)
        except CalculationError:
            continue
        values.append(value)
    return values


def evaluate_rag_row(
    row: dict[str, Any],
    answer_row: dict[str, Any],
) -> dict[str, Any]:
    answer = str(answer_row.get("final_answer") or "")
    gold_value = _gold_value(row)
    gold_period = Period.from_dict(row["period"])

    numeric_values = _answer_numeric_values(answer)
    numeric_correct = any(value == gold_value for value in numeric_values)

    year = str(gold_period.year)
    # The gold value is period-specific: an exact numeric match implies the
    # answer sourced the queried period. Otherwise the answer must state the
    # queried year explicitly.
    period_correct = numeric_correct or (f"{year}年" in answer) or (year in answer)

    predicted_value_type = classify_value_type(answer)
    value_type_correct: bool | None
    if predicted_value_type is ValueType.UNKNOWN:
        value_type_correct = None
    else:
        value_type_correct = predicted_value_type.value == row["value_type"]

    citations = answer_row.get("citations") or []
    selected_doc_ids = answer_row.get("selected_doc_ids") or []
    citation_doc_ids = {citation.get("doc_id") for citation in citations if isinstance(citation, dict)}
    citation_correct = row["doc_id"] in citation_doc_ids or row["doc_id"] in set(selected_doc_ids)

    return {
        "qid": row["qid"],
        "abstained": bool(answer_row.get("abstained")),
        "numeric_correct": numeric_correct,
        "period_correct": period_correct,
        "value_type_correct": value_type_correct,
        "citation_correct": citation_correct,
        "answer": answer[:300],
    }


def summarize_rows(rows: list[dict[str, Any]], *, value_type_key: str = "value_type_correct") -> dict[str, Any]:
    total = len(rows)
    numeric = sum(1 for row in rows if row["numeric_correct"])
    period = sum(1 for row in rows if row["period_correct"])
    citation = sum(1 for row in rows if row["citation_correct"])
    attempted = [row for row in rows if row["value_type_correct"] is not None]
    value_type_correct = sum(1 for row in attempted if row["value_type_correct"])
    return {
        "total": total,
        "exact_numeric_accuracy": round(numeric / total, 4) if total else 0.0,
        "period_accuracy": round(period / total, 4) if total else 0.0,
        "value_type_accuracy": round(value_type_correct / len(attempted), 4) if attempted else 0.0,
        "value_type_attempted": len(attempted),
        "citation_accuracy": round(citation / total, 4) if total else 0.0,
    }


def write_summary_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_summary_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Structured Fact Benchmark — Text RAG vs Structured Fact Search",
        "",
        "| Metric | Text RAG | Structured |",
        "|---|---|---|",
    ]
    rag = payload["rag"]
    structured = payload["structured"]
    for key, label in (
        ("exact_numeric_accuracy", "Exact Numeric Accuracy"),
        ("period_accuracy", "Period Accuracy"),
        ("value_type_accuracy", "Value Type Accuracy"),
        ("citation_accuracy", "Citation Accuracy"),
    ):
        lines.append(f"| {label} | {rag[key]:.2%} | {structured[key]:.2%} |")
    lines.append("")
    lines.append(f"- rows: {payload['total']} (structured routed: {payload.get('routed_rate', 0):.2%})")
    lines.append(f"- value_type_attempted (RAG): {rag.get('value_type_attempted', 0)}")
    lines.append(f"- API usage (RAG leg): {json.dumps(payload.get('api_usage', {}), ensure_ascii=False)}")
    path.write_text("\n".join(lines), encoding="utf-8")
