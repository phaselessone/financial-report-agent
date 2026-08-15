"""Trajectory eval tests (checklist v3.0 §P3 per-query logs + process/system metrics)."""

from __future__ import annotations

import unittest

from src.agent.config import AgentConfig
from src.agent.graph import run_agentic_rag
from src.evaluation.trajectory_eval import (
    build_agent_process_summary,
    build_agent_trace_row,
    build_failed_agent_trace_row,
    percentile,
)
from src.generation.routing import FALLBACK_ANSWER
from tests.test_agent_flow import FakeAnswerer, FakeLLM, FakeRuntime, make_result, make_row, rewrite_response, supported_draft

TRACE_FIELDS = (
    "question_id",
    "query",
    "question_type",
    "intent",
    "industry",
    "category",
    "step_count",
    "retrieval_count",
    "rewrite_count",
    "generation_count",
    "llm_call_count",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "api_latency_ms_total",
    "api_retries_total",
    "llm_calls_log",
    "termination_reason",
    "rewritten_queries",
    "retrieval_history",
    "final_answer",
    "evidence_summary",
    "abstained",
    "abstain_reason",
    "citations",
    "used_evidence_ids",
    "support_validation",
    "matched_numeric_tokens",
    "end_to_end_latency_ms",
    "failed",
    "error_type",
    "error_message",
)


def eval_row(**overrides) -> dict:
    row = {
        "question_id": "semiconductor_ms_comp_01",
        "query": "比较A公司和B公司的策略差异",
        "question_type": "comparison",
        "intent": "comparison",
        "industry": "semiconductor",
        "category": "agent_multi_source",
    }
    row.update(overrides)
    return row


def make_trace_row(*, counters: dict | None = None, latency_ms: float = 0.0, failed: bool = False) -> dict:
    row = {
        "question_id": "q1",
        "query": "q",
        "question_type": "fact",
        "intent": "fact",
        "industry": "semiconductor",
        "category": "agent_numeric_missing",
        "step_count": 4,
        "retrieval_count": 1,
        "rewrite_count": 0,
        "generation_count": 1,
        "llm_call_count": 1,
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "llm_calls_log": [],
        "termination_reason": "completed",
        "rewritten_queries": [],
        "retrieval_history": [],
        "final_answer": "答。",
        "evidence_summary": "",
        "abstained": False,
        "abstain_reason": None,
        "citations": [],
        "used_evidence_ids": [],
        "support_validation": {"supported": True},
        "matched_numeric_tokens": [],
        "end_to_end_latency_ms": 120.0,
        "failed": failed,
        "error_type": "",
        "error_message": "",
    }
    row.update(counters or {})
    row["end_to_end_latency_ms"] = latency_ms
    return row


class AgentTraceRowTests(unittest.TestCase):
    def test_trace_row_carries_checklist_fields_and_matches_state(self) -> None:
        runtime = FakeRuntime([make_result([make_row(chunk_id="c1", doc_id="d1", text="半导体行业景气度持续回升。")])])
        llm = FakeLLM([])
        answerer = FakeAnswerer([supported_draft("半导体行业景气度持续回升。")])
        state = run_agentic_rag(
            runtime=runtime,
            answerer=answerer,
            llm=llm,
            config=AgentConfig(),
            query="半导体行业景气度如何？",
        )
        row = build_agent_trace_row(state, eval_row(query="半导体行业景气度如何？"), end_to_end_latency_ms=42.0)
        for field in TRACE_FIELDS:
            self.assertIn(field, row)
        self.assertEqual(row["step_count"], state["step_count"])
        self.assertEqual(row["retrieval_count"], state["retrieval_count"])
        self.assertEqual(row["llm_call_count"], state["llm_call_count"])
        self.assertEqual(row["prompt_tokens"], state["prompt_tokens"])
        self.assertEqual(row["termination_reason"], "completed")
        self.assertEqual(row["end_to_end_latency_ms"], 42.0)
        self.assertEqual(row["final_answer"], "半导体行业景气度持续回升。")
        self.assertEqual(row["retrieval_history"], state["retrieval_history"])
        self.assertEqual(row["llm_calls_log"], state.get("llm_calls_log", []))

    def test_trace_row_sums_per_call_api_latency_and_retries(self) -> None:
        state = {
            "query": "q",
            "step_count": 3,
            "retrieval_count": 1,
            "rewrite_count": 1,
            "generation_count": 0,
            "llm_call_count": 2,
            "prompt_tokens": 30,
            "completion_tokens": 20,
            "total_tokens": 50,
            "llm_calls_log": [
                {"latency_ms": 100.0, "retries": 1},
                {"latency_ms": 250.0, "retries": 2},
            ],
            "termination_reason": "completed",
            "rewritten_queries": [{"rewritten_query": "q2", "reason": "no_evidence"}],
            "retrieval_history": [{"round": 1, "query": "q", "new_chunk_ids": ["c1"]}],
            "final_answer": supported_draft("ok"),
        }
        row = build_agent_trace_row(state, eval_row(), end_to_end_latency_ms=0.0)
        self.assertEqual(row["api_latency_ms_total"], 350.0)
        self.assertEqual(row["api_retries_total"], 3)

    def test_trace_row_handles_abstain_terminal_without_draft(self) -> None:
        # max_llm_calls=0: synthesize terminates before generating, finalize abstains.
        runtime = FakeRuntime([])
        llm = FakeLLM([])
        answerer = FakeAnswerer([])
        state = run_agentic_rag(
            runtime=runtime,
            answerer=answerer,
            llm=llm,
            config=AgentConfig(max_llm_calls=0),
            query="半导体行业景气度如何？",
        )
        row = build_agent_trace_row(state, eval_row(), end_to_end_latency_ms=0.0)
        self.assertTrue(row["abstained"])
        self.assertEqual(row["abstain_reason"], "max_llm_calls")
        self.assertEqual(row["final_answer"], FALLBACK_ANSWER)

    def test_failed_trace_row_shape(self) -> None:
        row = build_failed_agent_trace_row(
            eval_row=eval_row(category="agent_recovery"),
            error=ValueError("boom"),
            end_to_end_latency_ms=9.0,
        )
        self.assertTrue(row["failed"])
        self.assertEqual(row["error_type"], "ValueError")
        self.assertEqual(row["error_message"], "boom")
        self.assertEqual(row["category"], "agent_recovery")
        self.assertEqual(row["llm_call_count"], 0)
        self.assertEqual(row["end_to_end_latency_ms"], 9.0)
        self.assertIn("final_answer", row)


class PercentileTests(unittest.TestCase):
    def test_empty_values_return_none(self) -> None:
        self.assertIsNone(percentile([], 50))

    def test_single_value_returns_itself(self) -> None:
        self.assertEqual(percentile([7.0], 95), 7.0)

    def test_linear_interpolation(self) -> None:
        self.assertAlmostEqual(percentile([1.0, 2.0, 3.0, 4.0], 50), 2.5)
        self.assertAlmostEqual(percentile([1.0, 2.0, 3.0, 4.0], 95), 3.85)

    def test_unsorted_input(self) -> None:
        self.assertEqual(percentile([4.0, 1.0, 3.0, 2.0], 50), 2.5)

    def test_none_values_are_ignored(self) -> None:
        self.assertEqual(percentile([None, 1.0, 2.0, 3.0, 4.0], 50), 2.5)


class ProcessSummaryTests(unittest.TestCase):
    def test_averages_and_system_percentiles(self) -> None:
        rows = [
            make_trace_row(
                counters={"step_count": 4, "retrieval_count": 1, "rewrite_count": 0, "generation_count": 1, "llm_call_count": 1},
                latency_ms=100.0,
            ),
            make_trace_row(
                counters={"step_count": 8, "retrieval_count": 2, "rewrite_count": 1, "generation_count": 1, "llm_call_count": 2},
                latency_ms=300.0,
            ),
            make_trace_row(counters={"step_count": 2, "retrieval_count": 1, "rewrite_count": 0, "generation_count": 1, "llm_call_count": 1}, latency_ms=200.0),
        ]
        summary = build_agent_process_summary(rows)
        self.assertEqual(summary["query_count"], 3)
        self.assertEqual(summary["completed_query_count"], 3)
        self.assertAlmostEqual(summary["Average Steps"], 14 / 3, places=2)
        self.assertAlmostEqual(summary["Average Retrieval Calls"], 4 / 3, places=2)
        self.assertAlmostEqual(summary["Average Rewrite Calls"], 1 / 3, places=2)
        self.assertAlmostEqual(summary["Average LLM Calls"], 4 / 3, places=2)
        self.assertEqual(summary["System P50 End-to-End Latency"], 200.0)
        self.assertAlmostEqual(summary["System P95 End-to-End Latency"], 290.0)

    def test_failed_rows_excluded_from_averages(self) -> None:
        rows = [
            make_trace_row(counters={"step_count": 4, "retrieval_count": 1, "rewrite_count": 0, "generation_count": 1, "llm_call_count": 1}),
            make_trace_row(counters={"step_count": 9, "retrieval_count": 9, "rewrite_count": 9, "generation_count": 9, "llm_call_count": 9}, failed=True),
        ]
        summary = build_agent_process_summary(rows)
        self.assertEqual(summary["query_count"], 2)
        self.assertEqual(summary["completed_query_count"], 1)
        self.assertEqual(summary["Average Steps"], 4)
        self.assertEqual(summary["Average LLM Calls"], 1)


if __name__ == "__main__":
    unittest.main()
