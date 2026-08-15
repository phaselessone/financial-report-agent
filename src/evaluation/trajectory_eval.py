"""Agent trajectory evaluation (checklist v3.0 §P3 per-query logs + process/system metrics).

The agent graph already accumulates counters, per-call LLM logs
(``state["llm_calls_log"]``), rewritten queries and per-round retrieval history;
this module flattens a final agent state into one structured per-query trace row
and aggregates process/system metrics. All functions are deterministic and
consume only plain dicts, so they are unit-testable without a live runtime.
"""

from __future__ import annotations

from math import ceil, floor
from typing import Any


def percentile(values: list[float] | tuple[float, ...], p: float) -> float | None:
    """Linear-interpolation percentile over finite, non-None values.

    Mirrors ``numpy.percentile(..., method="linear")`` for the common case.
    Empty input returns None; single values return themselves.
    """
    data = sorted(float(value) for value in values if value is not None)
    if not data:
        return None
    if len(data) == 1:
        return data[0]
    rank = (len(data) - 1) * (p / 100.0)
    lower = floor(rank)
    upper = ceil(rank)
    if lower == upper:
        return data[int(rank)]
    return data[lower] * (upper - rank) + data[upper] * (rank - lower)


def _final_answer_fields(final_answer: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(final_answer, dict):
        final_answer = {}
    support_validation = final_answer.get("support_validation") or {}
    return {
        "final_answer": final_answer.get("final_answer", ""),
        "evidence_summary": final_answer.get("evidence_summary", ""),
        "abstained": bool(final_answer.get("abstained", False)),
        "abstain_reason": final_answer.get("abstain_reason"),
        "citations": final_answer.get("citations", []),
        "used_evidence_ids": final_answer.get("used_evidence_ids", []),
        "selected_doc_ids": final_answer.get("selected_doc_ids", []),
        "support_validation": support_validation,
        "matched_numeric_tokens": final_answer.get("matched_numeric_tokens", []),
        "fact_subtype": final_answer.get("fact_subtype", ""),
    }


def build_agent_trace_row(
    state: dict[str, Any],
    eval_row: dict[str, Any],
    *,
    end_to_end_latency_ms: float,
) -> dict[str, Any]:
    """Flatten a final agent state into the checklist §P3 per-query log row.

    The row also carries the final-answer fields expected by the answer-eval hit
    functions, so quality metrics can reuse them unchanged.
    """
    calls_log = list(state.get("llm_calls_log") or [])
    return {
        "question_id": eval_row["question_id"],
        "query": eval_row["query"],
        "question_type": eval_row.get("question_type", "fact"),
        "intent": eval_row.get("intent", eval_row.get("question_type", "fact")),
        "industry": eval_row.get("industry", "other"),
        "category": eval_row.get("category", ""),
        "step_count": int(state.get("step_count", 0)),
        "retrieval_count": int(state.get("retrieval_count", 0)),
        "rewrite_count": int(state.get("rewrite_count", 0)),
        "generation_count": int(state.get("generation_count", 0)),
        "llm_call_count": int(state.get("llm_call_count", 0)),
        "prompt_tokens": int(state.get("prompt_tokens", 0)),
        "completion_tokens": int(state.get("completion_tokens", 0)),
        "total_tokens": int(state.get("total_tokens", 0)),
        "api_latency_ms_total": round(sum(float(call.get("latency_ms", 0.0) or 0.0) for call in calls_log), 2),
        "api_retries_total": int(sum(int(call.get("retries", 0) or 0) for call in calls_log)),
        "llm_calls_log": calls_log,
        "termination_reason": state.get("termination_reason", ""),
        "rewritten_queries": list(state.get("rewritten_queries") or []),
        "retrieval_history": list(state.get("retrieval_history") or []),
        "no_improvement": bool(state.get("no_improvement", False)),
        **_final_answer_fields(state.get("final_answer")),
        "end_to_end_latency_ms": round(float(end_to_end_latency_ms), 2),
        "failed": False,
        "error_type": "",
        "error_message": "",
    }


def build_failed_agent_trace_row(
    *,
    eval_row: dict[str, Any],
    error: Exception,
    end_to_end_latency_ms: float,
) -> dict[str, Any]:
    """Trace row for a query whose agent run raised (batch must keep going)."""
    return {
        "question_id": eval_row["question_id"],
        "query": eval_row["query"],
        "question_type": eval_row.get("question_type", "fact"),
        "intent": eval_row.get("intent", eval_row.get("question_type", "fact")),
        "industry": eval_row.get("industry", "other"),
        "category": eval_row.get("category", ""),
        "step_count": 0,
        "retrieval_count": 0,
        "rewrite_count": 0,
        "generation_count": 0,
        "llm_call_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "api_latency_ms_total": 0.0,
        "api_retries_total": 0,
        "llm_calls_log": [],
        "termination_reason": "",
        "rewritten_queries": [],
        "retrieval_history": [],
        "no_improvement": False,
        "final_answer": "",
        "evidence_summary": "",
        "abstained": False,
        "abstain_reason": None,
        "citations": [],
        "used_evidence_ids": [],
        "selected_doc_ids": [],
        "support_validation": {},
        "matched_numeric_tokens": [],
        "fact_subtype": "",
        "end_to_end_latency_ms": round(float(end_to_end_latency_ms), 2),
        "failed": True,
        "error_type": type(error).__name__,
        "error_message": str(error),
    }


def _completed_rows(trace_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in trace_rows if not row.get("failed")]


def _average(rows: list[dict[str, Any]], key: str) -> float:
    return round(sum(float(row.get(key, 0.0) or 0.0) for row in rows) / max(len(rows), 1), 2)


def build_agent_process_summary(trace_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Process + System metrics (checklist §P3): average steps/calls and latency percentiles."""
    completed = _completed_rows(trace_rows)
    end_to_end = [float(row["end_to_end_latency_ms"]) for row in completed if row.get("end_to_end_latency_ms") is not None]
    return {
        "query_count": len(trace_rows),
        "completed_query_count": len(completed),
        "failed_query_count": len(trace_rows) - len(completed),
        "Average Steps": _average(completed, "step_count"),
        "Average Retrieval Calls": _average(completed, "retrieval_count"),
        "Average Rewrite Calls": _average(completed, "rewrite_count"),
        "Average Generation Calls": _average(completed, "generation_count"),
        "Average LLM Calls": _average(completed, "llm_call_count"),
        "System P50 End-to-End Latency": percentile(end_to_end, 50),
        "System P95 End-to-End Latency": percentile(end_to_end, 95),
    }
