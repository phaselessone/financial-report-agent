"""Token usage parsing and aggregation for LLM responses."""

from __future__ import annotations

from typing import Any

from src.llm.types import LLMResponse


def parse_usage(payload: dict[str, Any]) -> tuple[int, int, int]:
    """Extract ``(prompt_tokens, completion_tokens, total_tokens)`` from a payload.

    Missing or malformed fields normalize to 0; this never raises.
    """
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return 0, 0, 0

    def as_int(key: str) -> int:
        value = usage.get(key)
        try:
            return int(value) if value is not None else 0
        except (TypeError, ValueError):
            return 0

    prompt_tokens = as_int("prompt_tokens")
    completion_tokens = as_int("completion_tokens")
    total_tokens = as_int("total_tokens")
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens
    return prompt_tokens, completion_tokens, total_tokens


def summarize_usage(responses: list[LLMResponse]) -> dict[str, float | int]:
    """Aggregate per-call LLM responses into an API-usage summary."""
    if not responses:
        return {
            "llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "total_latency_ms": 0.0,
            "total_retries": 0,
        }
    return {
        "llm_calls": len(responses),
        "prompt_tokens": sum(response.prompt_tokens for response in responses),
        "completion_tokens": sum(response.completion_tokens for response in responses),
        "total_tokens": sum(response.usage_total_tokens() for response in responses),
        "total_latency_ms": round(sum(response.latency_ms for response in responses), 2),
        "total_retries": sum(response.retries for response in responses),
    }
