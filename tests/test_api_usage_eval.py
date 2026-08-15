"""API usage eval tests (checklist v3.0 §P3 API cost metrics)."""

from __future__ import annotations

import unittest

from src.evaluation.api_usage_eval import build_api_usage_summary, build_api_usage_summary_from_calls
from src.llm.types import LLMResponse


def make_trace_row(calls_log: list[dict]) -> dict:
    return {
        "question_id": "q1",
        "query": "q",
        "llm_call_count": len(calls_log),
        "prompt_tokens": sum(call["prompt_tokens"] for call in calls_log),
        "completion_tokens": sum(call["completion_tokens"] for call in calls_log),
        "total_tokens": sum(call["total_tokens"] for call in calls_log),
        "llm_calls_log": calls_log,
        "failed": False,
    }


def make_call(*, prompt: int, completion: int, latency_ms: float = 0.0, retries: int = 0) -> dict:
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "latency_ms": latency_ms,
        "retries": retries,
    }


def make_response(*, prompt: int, completion: int, latency_ms: float = 0.0, retries: int = 0) -> LLMResponse:
    return LLMResponse(
        content="{}",
        provider="fake",
        model="fake",
        prompt_tokens=prompt,
        completion_tokens=completion,
        latency_ms=latency_ms,
        retries=retries,
    )


class ApiUsageSummaryTests(unittest.TestCase):
    def test_metrics_from_trace_rows(self) -> None:
        rows = [
            make_trace_row(
                [
                    make_call(prompt=100, completion=50, latency_ms=100.0, retries=1),
                    make_call(prompt=200, completion=100, latency_ms=200.0, retries=0),
                ]
            ),
            make_trace_row([make_call(prompt=300, completion=150, latency_ms=300.0, retries=2)]),
        ]
        summary = build_api_usage_summary(rows)
        self.assertEqual(summary["query_count"], 2)
        self.assertEqual(summary["llm_calls"], 3)
        self.assertEqual(summary["Prompt Tokens / Query"], 300.0)
        self.assertEqual(summary["Completion Tokens / Query"], 150.0)
        self.assertEqual(summary["Total Tokens / Query"], 450.0)
        self.assertEqual(summary["Total Prompt Tokens"], 600)
        self.assertEqual(summary["Total Completion Tokens"], 300)
        self.assertEqual(summary["Total Tokens"], 900)
        # 3 retries over 3 calls -> rate 1.0; 2 of 3 calls retried -> ratio 2/3
        self.assertEqual(summary["API Retry Rate"], 1.0)
        self.assertAlmostEqual(summary["Retried Call Ratio"], 2 / 3, places=4)
        self.assertEqual(summary["P50 API Latency"], 200.0)
        self.assertAlmostEqual(summary["P95 API Latency"], 290.0)
        self.assertAlmostEqual(summary["Average LLM Calls / Query"], 1.5)

    def test_metrics_from_per_query_response_lists_match_trace_path(self) -> None:
        calls_by_query = [
            [
                make_response(prompt=100, completion=50, latency_ms=100.0, retries=1),
                make_response(prompt=200, completion=100, latency_ms=200.0, retries=0),
            ],
            [make_response(prompt=300, completion=150, latency_ms=300.0, retries=2)],
        ]
        summary = build_api_usage_summary_from_calls(calls_by_query)
        self.assertEqual(summary["llm_calls"], 3)
        self.assertEqual(summary["Total Tokens / Query"], 450.0)
        self.assertEqual(summary["API Retry Rate"], 1.0)
        self.assertEqual(summary["P50 API Latency"], 200.0)

    def test_empty_input_is_safe(self) -> None:
        summary = build_api_usage_summary([])
        self.assertEqual(summary["query_count"], 0)
        self.assertEqual(summary["llm_calls"], 0)
        self.assertEqual(summary["Prompt Tokens / Query"], 0.0)
        self.assertEqual(summary["API Retry Rate"], 0.0)
        self.assertIsNone(summary["P50 API Latency"])
        self.assertIsNone(summary["P95 API Latency"])


if __name__ == "__main__":
    unittest.main()
