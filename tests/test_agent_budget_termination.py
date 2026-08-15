"""Agent budget and termination tests (checklist v3.0 §P2 hard budgets)."""

from __future__ import annotations

import unittest

from src.agent.config import AgentConfig
from src.agent.graph import run_agentic_rag
from src.llm.types import LLMResponse
from tests.test_agent_flow import (
    FakeAnswerer,
    FakeLLM,
    FakeRuntime,
    abstained_draft,
    make_result,
    make_row,
    rewrite_response,
    supported_draft,
)


class BudgetAndTerminationTests(unittest.TestCase):
    def test_max_steps_terminates(self) -> None:
        runtime = FakeRuntime([])  # always empty, would keep rewriting
        llm = FakeLLM([rewrite_response("q2", "no_evidence") for _ in range(10)])
        answerer = FakeAnswerer([abstained_draft("no_evidence")])
        state = run_agentic_rag(
            runtime=runtime,
            answerer=answerer,
            llm=llm,
            config=AgentConfig(max_steps=3),
            query="半导体行业景气度如何？",
        )
        self.assertEqual(state["termination_reason"], "max_steps")
        # one guard-trip past the cap, then passthrough nodes and finalize
        self.assertLessEqual(state["step_count"], 5)
        self.assertIsNotNone(state["final_answer"])

    def test_max_llm_calls_terminates(self) -> None:
        runtime = FakeRuntime([])  # always empty, would keep rewriting
        llm = FakeLLM([rewrite_response("q2", "no_evidence") for _ in range(10)])
        answerer = FakeAnswerer([abstained_draft("no_evidence")])
        state = run_agentic_rag(
            runtime=runtime,
            answerer=answerer,
            llm=llm,
            config=AgentConfig(max_llm_calls=1),
            query="半导体行业景气度如何？",
        )
        self.assertEqual(state["termination_reason"], "max_llm_calls")
        self.assertEqual(state["llm_call_count"], 1)
        self.assertEqual(len(llm.calls), 1)

    def test_no_improvement_stops_searching_after_one_rewrite(self) -> None:
        same_rows = [make_row(chunk_id="c1", doc_id="d1", text="A公司强调AI算力需求。")]
        runtime = FakeRuntime([make_result(same_rows), make_result(same_rows)])
        llm = FakeLLM([rewrite_response("A公司 策略", "source_diversity_missing")])
        answerer = FakeAnswerer([supported_draft("A公司强调AI算力需求。")])
        state = run_agentic_rag(
            runtime=runtime,
            answerer=answerer,
            llm=llm,
            config=AgentConfig(),
            query="比较A公司和B公司的策略差异",
        )
        self.assertTrue(state["no_improvement"])
        self.assertEqual(state["retrieval_count"], 2)
        self.assertEqual(state["rewrite_count"], 1)
        self.assertEqual(len(llm.calls), 1)  # no further rewrites
        self.assertEqual(state["termination_reason"], "completed")

    def test_token_accounting_accumulates_rewrite_and_generation(self) -> None:
        runtime = FakeRuntime(
            [
                make_result([]),
                make_result([make_row(chunk_id="c1", doc_id="d1", text="半导体行业景气度持续回升。")]),
            ]
        )
        llm = FakeLLM([rewrite_response("半导体 景气", "no_evidence", prompt=10, completion=5)])
        answerer = FakeAnswerer([supported_draft("半导体行业景气度持续回升。")], llm_calls_per_answer=2)
        state = run_agentic_rag(
            runtime=runtime,
            answerer=answerer,
            llm=llm,
            config=AgentConfig(),
            query="半导体行业景气度如何？",
        )
        # 1 rewrite (10+5) + 2 answerer calls (20+10 each)
        self.assertEqual(state["llm_call_count"], 3)
        self.assertEqual(state["prompt_tokens"], 10 + 2 * 20)
        self.assertEqual(state["completion_tokens"], 5 + 2 * 10)
        self.assertEqual(state["total_tokens"], 15 + 2 * 30)

    def test_max_retrieval_rounds_caps_search(self) -> None:
        # Round 1: empty. Round 2: single source (still insufficient for comparison).
        # Round 3 would be attempted via a second rewrite, but the cap stops it first.
        runtime = FakeRuntime(
            [
                make_result([]),
                make_result([make_row(chunk_id="c1", doc_id="d1", text="A公司强调AI算力需求。")]),
            ]
        )
        llm = FakeLLM(
            [
                rewrite_response("A公司 策略", "no_evidence"),
                rewrite_response("A公司 B公司 策略 对比", "source_diversity_missing"),
            ]
        )
        answerer = FakeAnswerer([abstained_draft("max_retrieval_rounds")])
        state = run_agentic_rag(
            runtime=runtime,
            answerer=answerer,
            llm=llm,
            config=AgentConfig(max_retrieval_rounds=2),
            query="比较A公司和B公司的策略差异",
        )
        self.assertEqual(state["retrieval_count"], 2)
        self.assertEqual(state["rewrite_count"], 2)
        self.assertEqual(state["termination_reason"], "max_retrieval_rounds")

    def test_max_generation_attempts_caps_regeneration(self) -> None:
        runtime = FakeRuntime([make_result([make_row(chunk_id="c1", doc_id="d1", text="半导体行业景气度持续回升。")])])
        llm = FakeLLM([])
        unsupported = supported_draft("unsupported")
        unsupported["support_validation"] = {"supported": False}
        answerer = FakeAnswerer([unsupported, unsupported])
        state = run_agentic_rag(
            runtime=runtime,
            answerer=answerer,
            llm=llm,
            config=AgentConfig(max_generation_attempts=2),
            query="半导体行业景气度如何？",
        )
        self.assertEqual(state["generation_count"], 2)
        self.assertEqual(state["termination_reason"], "max_generation_attempts")


if __name__ == "__main__":
    unittest.main()
