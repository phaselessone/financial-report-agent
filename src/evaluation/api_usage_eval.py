"""API usage / cost evaluation (checklist v3.0 §P3 API metrics).

Aggregates per-call LLM records (from agent trace rows or per-query provider
response lists) into token-per-query, retry-rate and latency-percentile metrics.
"""

from __future__ import annotations

from typing import Any

from src.evaluation.trajectory_eval import percentile


def _field(call: Any, name: str, default: Any = None) -> Any:
    """Read a per-call field from either a dict (trace log) or an object (LLMResponse)."""
    if isinstance(call, dict):
        return call.get(name, default)
    return getattr(call, name, default)


def _aggregate(calls_by_query: list[list[Any]]) -> dict[str, Any]:
    query_count = len(calls_by_query)
    prompt_per_query: list[float] = []
    completion_per_query: list[float] = []
    total_per_query: list[float] = []
    latencies: list[float] = []
    total_calls = 0
    retries_total = 0
    retried_calls = 0
    for calls in calls_by_query:
        prompt = completion = total = 0.0
        for call in calls:
            call_prompt = float(_field(call, "prompt_tokens", 0) or 0)
            call_completion = float(_field(call, "completion_tokens", 0) or 0)
            call_total = float(_field(call, "total_tokens", 0) or 0) or call_prompt + call_completion
            prompt += call_prompt
            completion += call_completion
            total += call_total
            latency = _field(call, "latency_ms")
            if latency is not None:
                latencies.append(float(latency))
            retries = int(_field(call, "retries", 0) or 0)
            retries_total += retries
            if retries > 0:
                retried_calls += 1
            total_calls += 1
        prompt_per_query.append(prompt)
        completion_per_query.append(completion)
        total_per_query.append(total)
    return {
        "query_count": query_count,
        "llm_calls": total_calls,
        "prompt_per_query": prompt_per_query,
        "completion_per_query": completion_per_query,
        "total_per_query": total_per_query,
        "latencies": latencies,
        "retries_total": retries_total,
        "retried_calls": retried_calls,
    }


def _mean(values: list[float]) -> float:
    return round(sum(values) / max(len(values), 1), 2)


def _summary_from_aggregate(agg: dict[str, Any]) -> dict[str, Any]:
    return {
        "query_count": agg["query_count"],
        "llm_calls": agg["llm_calls"],
        "Prompt Tokens / Query": _mean(agg["prompt_per_query"]),
        "Completion Tokens / Query": _mean(agg["completion_per_query"]),
        "Total Tokens / Query": _mean(agg["total_per_query"]),
        "Total Prompt Tokens": int(sum(agg["prompt_per_query"])),
        "Total Completion Tokens": int(sum(agg["completion_per_query"])),
        "Total Tokens": int(sum(agg["total_per_query"])),
        "API Retry Rate": round(agg["retries_total"] / max(agg["llm_calls"], 1), 4),
        "Retried Call Ratio": round(agg["retried_calls"] / max(agg["llm_calls"], 1), 4),
        "P50 API Latency": percentile(agg["latencies"], 50),
        "P95 API Latency": percentile(agg["latencies"], 95),
        "Average LLM Calls / Query": round(agg["llm_calls"] / max(agg["query_count"], 1), 2),
    }


def build_api_usage_summary(trace_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """API cost metrics from agent trace rows (per-call ``llm_calls_log``)."""
    calls_by_query = [list(row.get("llm_calls_log") or []) for row in trace_rows]
    return _summary_from_aggregate(_aggregate(calls_by_query))


def build_api_usage_summary_from_calls(calls_by_query: list[list[Any]]) -> dict[str, Any]:
    """API cost metrics from per-query provider response lists (baseline side)."""
    return _summary_from_aggregate(_aggregate(calls_by_query))
