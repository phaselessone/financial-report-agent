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
    llm_call_kinds,
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
        # P7: exactly one rewrite call; the fixed claim-extraction call is ignored.
        self.assertEqual(len([k for k in llm_call_kinds(llm) if k == "rewrite"]), 1)
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
        # 1 rewrite (10+5) + 2 answerer calls (20+10 each) + extract_claims.
        # The extract_claims call exhausts the FakeLLM's single rewrite response,
        # so it books 0 tokens but still counts as one LLM call (P7).
        self.assertEqual(state["llm_call_count"], 4)
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
            config=AgentConfig(max_retrieval_rounds=2, max_failed_rewrite_rounds=5),
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
            config=AgentConfig(max_generation_attempts=2, max_retrieval_rounds=1),
            query="半导体行业景气度如何？",
        )
        self.assertEqual(state["generation_count"], 2)
        self.assertEqual(state["termination_reason"], "max_generation_attempts")

    def test_abstains_when_rewritten_evidence_still_insufficient(self) -> None:
        # Round 1: single source (insufficient for comparison). Round 2: a NEW
        # chunk from the same doc — still single-source, so the post-rewrite
        # grade fails again and the agent must abstain instead of burning steps.
        runtime = FakeRuntime(
            [
                make_result([make_row(chunk_id="c1", doc_id="d1", text="A公司强调AI算力需求。")]),
                make_result([make_row(chunk_id="c2", doc_id="d1", text="A公司聚焦国产替代。")]),
            ]
        )
        llm = FakeLLM([rewrite_response("A公司 B公司 策略", "source_diversity_missing") for _ in range(10)])
        answerer = FakeAnswerer([])
        state = run_agentic_rag(
            runtime=runtime,
            answerer=answerer,
            llm=llm,
            config=AgentConfig(),
            query="比较A公司和B公司的策略差异",
        )
        self.assertEqual(state["termination_reason"], "abstain_evidence_insufficient")
        self.assertEqual(state["retrieval_count"], 2)
        self.assertEqual(state["rewrite_count"], 1)
        self.assertEqual(state["generation_count"], 0)
        self.assertTrue(state["final_answer"]["abstained"])
        self.assertEqual(state["final_answer"]["abstain_reason"], "abstain_evidence_insufficient")
        self.assertEqual(len(llm.calls), 1)

    def test_insufficient_rewrite_rounds_counter_resets_on_success(self) -> None:
        # A failing post-rewrite grade then a successful round must not leak strikes.
        runtime = FakeRuntime(
            [
                make_result([make_row(chunk_id="c1", doc_id="d1", text="A公司强调AI算力需求。")]),
                make_result([make_row(chunk_id="c2", doc_id="d1", text="A公司聚焦国产替代。")]),
                make_result(
                    [
                        make_row(chunk_id="c3", doc_id="d1", text="A公司强调AI算力需求。"),
                        make_row(chunk_id="c4", doc_id="d2", text="B公司聚焦存储涨价。"),
                    ]
                ),
            ]
        )
        llm = FakeLLM(
            [
                rewrite_response("A公司 国产替代", "source_diversity_missing"),
                rewrite_response("A公司 B公司 策略 对比", "source_diversity_missing"),
            ]
        )
        answerer = FakeAnswerer([supported_draft("A公司强调AI算力，B公司聚焦存储。")])
        state = run_agentic_rag(
            runtime=runtime,
            answerer=answerer,
            llm=llm,
            config=AgentConfig(max_failed_rewrite_rounds=2, max_steps=12),
            query="比较A公司和B公司的策略差异",
        )
        # first post-rewrite grade failed (strike 1 <= 2), second rewrite recovered
        self.assertEqual(state["termination_reason"], "completed")
        self.assertEqual(state["rewrite_count"], 2)
        self.assertEqual(state["failed_rewrite_rounds"], 0)

    def test_llm_calls_log_records_each_call_with_node_metadata(self) -> None:
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
        log = state["llm_calls_log"]
        # P7: extract_claims adds one entry after synthesis.
        self.assertEqual(len(log), 4)
        self.assertEqual(
            [entry["node"] for entry in log],
            ["rewrite_query", "synthesize", "synthesize", "extract_claims"],
        )
        for entry in log:
            for key in (
                "step_count",
                "node",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "latency_ms",
                "retries",
                "request_id",
                "finish_reason",
            ):
                self.assertIn(key, entry)
        self.assertEqual(sum(entry["prompt_tokens"] for entry in log), state["prompt_tokens"])
        self.assertEqual(sum(entry["completion_tokens"] for entry in log), state["completion_tokens"])
        self.assertEqual(sum(entry["total_tokens"] for entry in log), state["total_tokens"])
        self.assertEqual(log[0]["prompt_tokens"], 10)
        self.assertEqual(log[0]["completion_tokens"], 5)

    def test_max_total_tokens_blocks_second_rewrite_before_call(self) -> None:
        # Both rounds surface a single source: still insufficient for comparison,
        # so a second rewrite would be attempted. The token budget consumed by the
        # first rewrite (20 tokens vs budget 10) must block it before any call.
        runtime = FakeRuntime(
            [
                make_result([make_row(chunk_id="c1", doc_id="d1", text="A公司强调AI算力需求。")]),
                make_result([make_row(chunk_id="c2", doc_id="d1", text="A公司聚焦国产替代。")]),
            ]
        )
        llm = FakeLLM([rewrite_response("A公司 B公司 策略", "source_diversity_missing", prompt=10, completion=10) for _ in range(10)])
        answerer = FakeAnswerer([abstained_draft("max_total_tokens")])
        state = run_agentic_rag(
            runtime=runtime,
            answerer=answerer,
            llm=llm,
            config=AgentConfig(max_total_tokens=10, max_failed_rewrite_rounds=5),
            query="比较A公司和B公司的策略差异",
        )
        self.assertEqual(state["termination_reason"], "max_total_tokens")
        self.assertEqual(state["llm_call_count"], 1)
        self.assertEqual(len(llm.calls), 1)

    def test_max_total_tokens_stops_synthesize_after_budget_consumed(self) -> None:
        # Empty corpus: the rewrite consumes the budget (20 tokens vs budget 15),
        # then synthesize must terminate without generating.
        runtime = FakeRuntime([])
        llm = FakeLLM([rewrite_response("q2", "no_evidence", prompt=10, completion=10) for _ in range(10)])
        answerer = FakeAnswerer([abstained_draft("no_evidence")])
        state = run_agentic_rag(
            runtime=runtime,
            answerer=answerer,
            llm=llm,
            config=AgentConfig(max_total_tokens=15),
            query="半导体行业景气度如何？",
        )
        self.assertEqual(state["termination_reason"], "max_total_tokens")
        self.assertEqual(state["llm_call_count"], 1)
        self.assertEqual(state["generation_count"], 0)


if __name__ == "__main__":
    unittest.main()
