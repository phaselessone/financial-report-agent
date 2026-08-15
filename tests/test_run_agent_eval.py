"""run_agent_eval CLI helper tests (checklist v3.0 §P3 entry point)."""

from __future__ import annotations

import argparse
import unittest
from pathlib import Path

try:
    import pymupdf  # noqa: F401
except ModuleNotFoundError:
    HAS_FITZ = False
    run_agent_eval = None
else:
    HAS_FITZ = True
    from run_agent_eval import (
        build_agent_config,
        resolve_baseline_results_path,
        resolve_report_paths,
        run_agentic_eval_rows,
        run_baseline_rows,
    )

from src.agent.config import AgentConfig
from tests.test_agent_flow import FakeAnswerer, FakeLLM, FakeRuntime, make_result, make_row, supported_draft


def make_eval_row(question_id: str = "q1") -> dict:
    return {
        "question_id": question_id,
        "query": "半导体行业景气度如何？",
        "question_type": "fact",
        "intent": "numeric_fact",
        "industry": "semiconductor",
        "category": "agent_numeric_missing",
        "must_recover": True,
    }


@unittest.skipUnless(HAS_FITZ, "fitz is required for run_agent_eval imports")
class RunAgentEvalArgTests(unittest.TestCase):
    def test_build_agent_config_maps_budgets(self) -> None:
        args = argparse.Namespace(
            max_steps=3,
            max_retrieval_rounds=2,
            max_query_rewrites=1,
            max_generation_attempts=2,
            max_llm_calls=4,
            max_total_tokens=500,
        )
        config = build_agent_config(args)
        self.assertIsInstance(config, AgentConfig)
        self.assertEqual(config.max_steps, 3)
        self.assertEqual(config.max_retrieval_rounds, 2)
        self.assertEqual(config.max_query_rewrites, 1)
        self.assertEqual(config.max_generation_attempts, 2)
        self.assertEqual(config.max_llm_calls, 4)
        self.assertEqual(config.max_total_tokens, 500)

    def test_resolve_baseline_results_path_defaults_to_reports_dir(self) -> None:
        args = argparse.Namespace(baseline_results_path=None)
        output_dir = Path("outputs")
        self.assertEqual(
            resolve_baseline_results_path(args, output_dir),
            output_dir / "reports" / "agent_baseline_results_dev.jsonl",
        )

    def test_resolve_baseline_results_path_prefers_explicit_path(self) -> None:
        args = argparse.Namespace(baseline_results_path=Path("artifacts/custom.jsonl"))
        self.assertEqual(resolve_baseline_results_path(args, Path("outputs")), Path("artifacts/custom.jsonl"))

    def test_resolve_report_paths_covers_all_artifacts(self) -> None:
        paths = resolve_report_paths(Path("outputs"))
        for key in (
            "agent_eval_results",
            "agent_traces",
            "agent_eval_summary_json",
            "agent_eval_summary_md",
            "api_usage_summary_json",
            "api_usage_summary_md",
            "baseline_results",
            "baseline_summary",
            "comparison_json",
            "comparison_md",
            "gate_verdict",
            "manifest_report",
        ):
            self.assertIn(key, paths)
        self.assertTrue(all(path.parent == Path("outputs") / "reports" for path in paths.values()))


@unittest.skipUnless(HAS_FITZ, "fitz is required for run_agent_eval imports")
class RunAgentEvalLoopTests(unittest.TestCase):
    def test_run_agentic_eval_rows_produces_trace_rows(self) -> None:
        runtime = FakeRuntime([make_result([make_row(chunk_id="c1", doc_id="d1", text="半导体行业景气度持续回升。")])])
        llm = FakeLLM([])
        answerer = FakeAnswerer([supported_draft("半导体行业景气度持续回升。")])
        trace_rows = run_agentic_eval_rows(
            eval_rows=[make_eval_row()],
            runtime=runtime,
            answerer=answerer,
            llm=llm,
            config=AgentConfig(),
        )
        self.assertEqual(len(trace_rows), 1)
        row = trace_rows[0]
        self.assertEqual(row["question_id"], "q1")
        self.assertFalse(row["failed"])
        self.assertEqual(row["termination_reason"], "completed")

    def test_run_agentic_eval_rows_captures_failures_per_row(self) -> None:
        class BoomRuntime:
            def search(self, query: str):
                raise RuntimeError("index broken")

        llm = FakeLLM([])
        answerer = FakeAnswerer([])
        trace_rows = run_agentic_eval_rows(
            eval_rows=[make_eval_row(), make_eval_row("q2")],
            runtime=BoomRuntime(),
            answerer=answerer,
            llm=llm,
            config=AgentConfig(),
        )
        self.assertEqual(len(trace_rows), 2)
        self.assertTrue(all(row["failed"] for row in trace_rows))
        self.assertEqual(trace_rows[0]["error_type"], "RuntimeError")

    def test_run_baseline_rows_records_answers_and_per_query_calls(self) -> None:
        runtime = FakeRuntime([make_result([make_row(chunk_id="c1", doc_id="d1", text="半导体行业景气度持续回升。")])])
        answerer = FakeAnswerer([supported_draft("半导体行业景气度持续回升。")], llm_calls_per_answer=2)
        result_rows, calls_by_query = run_baseline_rows(
            eval_rows=[make_eval_row()],
            runtime=runtime,
            answerer=answerer,
        )
        self.assertEqual(len(result_rows), 1)
        self.assertEqual(result_rows[0]["question_id"], "q1")
        self.assertEqual(len(calls_by_query), 1)
        self.assertEqual(len(calls_by_query[0]), 2)

    def test_baseline_row_feeds_agent_trace_quality_hits(self) -> None:
        runtime = FakeRuntime([make_result([make_row(chunk_id="c1", doc_id="d1", text="半导体行业景气度持续回升。")])])
        answerer = FakeAnswerer([supported_draft("半导体行业景气度持续回升。")], llm_calls_per_answer=1)
        result_rows, _ = run_baseline_rows(
            eval_rows=[make_eval_row()],
            runtime=runtime,
            answerer=answerer,
        )
        # The answer row carries the fields the quality hit functions need.
        row = result_rows[0]
        for key in ("final_answer", "abstained", "citations", "support_validation"):
            self.assertIn(key, row)


if __name__ == "__main__":
    unittest.main()
