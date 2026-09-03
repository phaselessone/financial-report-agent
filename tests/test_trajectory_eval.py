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
from src.evaluation.run_identity import RunIdentity, hash_case_ids
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
    "tool_calls",
    "tool_call_count",
    "calculations",
    "claims",
    "claim_verification_summary",
    "trajectory_events",
    "dependency_coverage",
    "failure_attribution",
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
    @staticmethod
    def _run_identity() -> RunIdentity:
        return RunIdentity(
            benchmark_profile="contract-v1",
            benchmark_version="1.0",
            benchmark_hash="benchmark-sha256",
            case_ids_hash=hash_case_ids(["semiconductor_ms_comp_01"]),
            case_count=1,
            corpus_hash="corpus-sha256",
            model="model-x",
            provider="provider-x",
            model_revision="rev-1",
            temperature=0.0,
            prompt_version="prompt-v2",
            budgets={"max_steps": 12},
            feature_flags={"runtime": {}, "treatments": {"pipeline": "agentic"}},
            git_commit="abc123",
            source_manifest_hash="source-sha256",
        )

    def test_trace_row_carries_run_identity(self) -> None:
        identity = self._run_identity()
        row = build_agent_trace_row(
            {"termination_reason": "completed"},
            eval_row(),
            end_to_end_latency_ms=1.0,
            run_identity=identity,
        )

        self.assertEqual(row["run_identity"], identity.to_dict())
        self.assertEqual(row["run_id"], identity.run_id)

    def test_strict_trace_materialization_requires_run_identity(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires run_identity"):
            build_agent_trace_row(
                {"termination_reason": "completed"},
                eval_row(),
                end_to_end_latency_ms=1.0,
                strict=True,
            )
        with self.assertRaisesRegex(ValueError, "requires run_identity"):
            build_failed_agent_trace_row(
                eval_row=eval_row(),
                error=RuntimeError("boom"),
                end_to_end_latency_ms=1.0,
                strict=True,
            )

        identity = self._run_identity()
        row = build_agent_trace_row(
            {"termination_reason": "completed"},
            eval_row(),
            end_to_end_latency_ms=1.0,
            run_identity=identity,
            strict=True,
        )
        self.assertEqual(row["run_id"], identity.run_id)

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

    def test_trace_row_carries_dependency_coverage(self) -> None:
        coverage = {
            "required": ["q1", "q2"],
            "completed": ["q1"],
            "missing": ["q2"],
            "coverage_ratio": 0.5,
            "decision": "partial",
        }
        row = build_agent_trace_row(
            {"termination_reason": "completed", "dependency_coverage": coverage},
            eval_row(),
            end_to_end_latency_ms=1.0,
        )
        self.assertEqual(row["dependency_coverage"], coverage)

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
        self.assertEqual(row["termination_reason"], "runtime_exception")
        self.assertEqual(len(row["trajectory_events"]), 1)
        self.assertEqual(row["trajectory_events"][0]["status"], "FAILED")
        self.assertEqual(row["trajectory_events"][0]["event_type"], "materialization")
        self.assertTrue(row["failure_attribution"]["has_failure"])

    def test_failed_trace_row_preserves_exception_partial_state(self) -> None:
        error = RuntimeError("boom")
        error.agent_partial_state = {
            "step_count": 3,
            "tool_call_count": 2,
            "total_tokens": 41,
            "termination_reason": "node_exception",
            "tool_calls": [{"tool_name": "calculator", "status": "SUCCESS"}],
            "calculations": {"growth": {"status": "SUCCESS", "value": "20"}},
            "claims": [{"claim_id": "c1", "text": "增长 20"}],
            "trajectory_events": [
                {
                    "event_id": "e1",
                    "dependencies": [],
                    "recovery_of": [],
                    "node": "execute_step",
                    "status": "FAILED",
                    "error_type": "tool_execution_failed",
                    "budget_usage": {"tokens": 41},
                }
            ],
        }

        row = build_failed_agent_trace_row(
            eval_row=eval_row(category="agent_recovery"),
            error=error,
            end_to_end_latency_ms=9.0,
        )

        self.assertTrue(row["failed"])
        self.assertEqual(row["step_count"], 3)
        self.assertEqual(row["tool_call_count"], 2)
        self.assertEqual(row["total_tokens"], 41)
        self.assertEqual(row["tool_calls"], error.agent_partial_state["tool_calls"])
        self.assertEqual(row["calculations"], error.agent_partial_state["calculations"])
        self.assertEqual(row["claims"], error.agent_partial_state["claims"])
        self.assertEqual(row["trajectory_events"], error.agent_partial_state["trajectory_events"])
        self.assertEqual(row["failure_attribution"]["root_cause"], "tool_execution_failed")

    def test_materialized_trace_rows_include_failure_attribution(self) -> None:
        success = build_agent_trace_row(
            {
                "termination_reason": "completed",
                "trajectory_events": [
                    {"event_id": "e1", "dependencies": [], "recovery_of": [], "step": 1, "node": "retrieve", "status": "FAILED", "error_type": "retrieval_miss"},
                    {"event_id": "e2", "dependencies": ["e1"], "recovery_of": ["e1"], "step": 2, "node": "retrieve", "status": "SUCCESS"},
                ],
            },
            eval_row(),
            end_to_end_latency_ms=4.0,
        )
        failed = build_failed_agent_trace_row(
            eval_row=eval_row(),
            error=RuntimeError("provider unavailable"),
            end_to_end_latency_ms=5.0,
        )

        self.assertIsNone(success["failure_attribution"]["root_cause"])
        self.assertEqual(success["failure_attribution"]["root_cause_resolution"], "recovered")
        self.assertTrue(success["failure_attribution"]["recovered"])
        self.assertTrue(failed["failure_attribution"]["has_failure"])


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
