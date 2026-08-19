"""End-to-end multi-hop agent flow tests (checklist v3.0 §P6).

The multi-hop trigger is deliberately narrow (fact-type numeric+opinion mix), so
comparison/inductive queries keep their single-hop grade/rewrite recovery
(covered by test_agent_flow / test_agent_budget_termination). These tests cover
the new multi-hop path only.
"""

from __future__ import annotations

import json
import unittest
from decimal import Decimal

from src.agent.config import AgentConfig
from src.agent.graph import run_agentic_rag
from src.llm.types import LLMResponse
from src.structured.fact_store import FactStore
from src.structured.schema import FinancialFact, Period
from tests.test_agent_flow import (
    FakeAnswerer,
    FakeLLM,
    FakeRuntime,
    abstained_draft,
    make_result,
    make_row,
    supported_draft,
)


def decompose_response(sub_questions) -> LLMResponse:
    return LLMResponse(
        content=json.dumps({"sub_questions": sub_questions}, ensure_ascii=False),
        provider="fake",
        model="fake",
        prompt_tokens=10,
        completion_tokens=5,
    )


MULTI_HOP_QUERY = "宁德时代2025年营业收入是多少？市场对其增长逻辑怎么看？"


class MultiHopFlowTests(unittest.TestCase):
    def test_numeric_opinion_mix_decomposes_then_synthesizes(self) -> None:
        runtime = FakeRuntime(
            [
                make_result([make_row(chunk_id="c1", doc_id="d1", text="宁德时代2025年营收1234亿元。")]),
                make_result([make_row(chunk_id="c2", doc_id="d2", text="机构看好宁德时代增长逻辑。")]),
            ]
        )
        llm = FakeLLM(
            [
                decompose_response(
                    [
                        {"id": "q1", "query": "宁德时代2025年营业收入是多少"},
                        {"id": "q2", "query": "市场对宁德时代增长逻辑的看法"},
                    ]
                )
            ]
        )
        answerer = FakeAnswerer([supported_draft("宁德时代2025年营收1234亿元，机构看好其增长逻辑。")])
        state = run_agentic_rag(
            runtime=runtime,
            answerer=answerer,
            llm=llm,
            config=AgentConfig(),
            query=MULTI_HOP_QUERY,
        )
        self.assertTrue(state["is_multi_hop"])
        self.assertEqual(len(state["sub_questions"]), 2)
        self.assertEqual(state["retrieval_count"], 2)
        self.assertEqual(len(llm.calls), 1)  # decompose only, no rewrite
        self.assertEqual(state["rewrite_count"], 0)
        self.assertEqual(state["termination_reason"], "completed")
        self.assertEqual(state["final_answer"]["final_answer"], "宁德时代2025年营收1234亿元，机构看好其增长逻辑。")
        # synthesize answered the ORIGINAL query, not the last sub-question
        self.assertEqual(answerer.answer_calls[0]["query"], MULTI_HOP_QUERY)

    def test_fact_single_hop_skips_decompose(self) -> None:
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
        self.assertFalse(state["is_multi_hop"])
        self.assertEqual(len(llm.calls), 0)  # no decompose, no rewrite
        self.assertEqual(state["retrieval_count"], 1)
        self.assertEqual(state["termination_reason"], "completed")

    def test_comparison_stays_single_hop(self) -> None:
        # Guard: comparison must NOT be hijacked onto the multi-hop path; it
        # keeps the single-hop grade/rewrite recovery (P3 gate baseline).
        runtime = FakeRuntime([make_result([make_row(chunk_id="c1", doc_id="d1", text="半导体行业景气度持续回升。")])])
        llm = FakeLLM([])
        answerer = FakeAnswerer([supported_draft("ok")])
        state = run_agentic_rag(
            runtime=runtime,
            answerer=answerer,
            llm=llm,
            config=AgentConfig(),
            query="比较A公司和B公司的策略差异",
        )
        self.assertFalse(state["is_multi_hop"])
        self.assertEqual(state["sub_questions"], [])

    def test_structured_subquestion_routes_to_fact_store(self) -> None:
        store = FactStore(":memory:")
        self.addCleanup(store.close)
        store.upsert(
            [
                FinancialFact(
                    company="贵州茅台",
                    metric="revenue",
                    period=Period("FY", 2025),
                    value_type="actual",
                    value=Decimal("123450000000"),
                    unit="元",
                    doc_id="doc-maotai",
                    page=3,
                    evidence_id="E1",
                    raw_value="1234.5亿元",
                    source_span="2025年营收1234.5亿元",
                )
            ]
        )
        aliases = {"贵州茅台": ["贵州茅台", "茅台"]}
        runtime = FakeRuntime([make_result([make_row(chunk_id="c1", doc_id="d1", text="机构看好白酒。")])])
        llm = FakeLLM(
            [
                decompose_response(
                    [
                        {"id": "q1", "query": "贵州茅台2025年营业收入是多少"},
                        {"id": "q2", "query": "机构对贵州茅台的看法"},
                    ]
                )
            ]
        )
        answerer = FakeAnswerer([supported_draft("贵州茅台营收1234.5亿元，机构看好。")])
        state = run_agentic_rag(
            runtime=runtime,
            answerer=answerer,
            llm=llm,
            config=AgentConfig(),
            query="贵州茅台2025年营业收入是多少？机构怎么看？",
            fact_store=store,
            company_aliases=aliases,
        )
        self.assertEqual(len(state["structured_facts"]), 1)
        self.assertEqual(state["retrieval_count"], 1)  # only the opinion sub-question used RAG
        self.assertEqual(runtime.calls, ["机构对贵州茅台的看法"])
        self.assertEqual(state["termination_reason"], "completed")
        self.assertEqual(state["final_answer"]["final_answer"], "贵州茅台营收1234.5亿元，机构看好。")

    def test_three_metric_stays_single_hop(self) -> None:
        # P6 DoD 三指标比较: multi-metric fact query must stay on the
        # single-hop path; it has no opinion/outlook anchor so
        # is_multi_hop_query returns False. The single-hop path may
        # legitimately call the LLM for rewrite — only the multi-hop
        # routing must be off, which is what we assert.
        runtime = FakeRuntime([make_result([make_row(chunk_id="c1", doc_id="d1", text="贵州茅台营收1234.5亿元。")])])
        llm = FakeLLM([])
        answerer = FakeAnswerer([supported_draft("ok")])
        state = run_agentic_rag(
            runtime=runtime,
            answerer=answerer,
            llm=llm,
            config=AgentConfig(),
            query="贵州茅台2025年营业收入、净利润、毛利率分别是多少？",
        )
        self.assertFalse(state["is_multi_hop"])
        self.assertEqual(state["sub_questions"], [])

    def test_inductive_stays_single_hop(self) -> None:
        # P6 DoD 跨报告归纳: cross-report inductive query must stay on the
        # single-hop path. infer_question_type classifies it as inductive,
        # which is not 'fact', so is_multi_hop_query returns False.
        runtime = FakeRuntime([make_result([make_row(chunk_id="c1", doc_id="d1", text="行业趋势：新能源、半导体。")])])
        llm = FakeLLM([])
        answerer = FakeAnswerer([supported_draft("ok")])
        state = run_agentic_rag(
            runtime=runtime,
            answerer=answerer,
            llm=llm,
            config=AgentConfig(),
            query="近期多家券商研报共同强调的行业趋势是什么？",
        )
        self.assertFalse(state["is_multi_hop"])
        self.assertEqual(state["sub_questions"], [])

    def test_all_rag_subquestions_are_retrieved(self) -> None:
        # Every RAG sub-question gets exactly one retrieval. The multi-hop
        # path is bounded by max_sub_questions, NOT by max_retrieval_rounds —
        # P6 plan §1 reserves that cap for the single-hop rewrite loop, so
        # surplus RAG sub-questions must never be silently dropped
        # (code-review P1). With 5 RAG sub-questions we expect 5 searches.
        config = AgentConfig()  # max_retrieval_rounds=3 default — not applied here
        runtime = FakeRuntime(
            [
                make_result([make_row(chunk_id=f"c{i}", doc_id=f"d{i}", text=f"r{i}")])
                for i in range(1, 6)
            ]
        )
        llm = FakeLLM(
            [
                decompose_response(
                    [
                        {"id": "q1", "query": "宁德时代营收是多少"},
                        {"id": "q2", "query": "宁德时代净利润是多少"},
                        {"id": "q3", "query": "宁德时代毛利率是多少"},
                        {"id": "q4", "query": "宁德时代研发费用是多少"},
                        {"id": "q5", "query": "宁德时代经营性现金流是多少"},
                    ]
                )
            ]
        )
        answerer = FakeAnswerer([supported_draft("ok")])
        state = run_agentic_rag(
            runtime=runtime,
            answerer=answerer,
            llm=llm,
            config=config,
            query=MULTI_HOP_QUERY,
        )
        self.assertGreater(len(state["sub_questions"]), config.max_retrieval_rounds)
        self.assertEqual(state["retrieval_count"], len(state["sub_questions"]))
        self.assertEqual(len(runtime.calls), len(state["sub_questions"]))
        skipped = [
            result for result in state["sub_question_results"] if result.get("source") == "skipped_budget"
        ]
        self.assertEqual(len(skipped), 0)

    def test_multi_hop_abstains_without_rewrite_loop(self) -> None:
        runtime = FakeRuntime([make_result([make_row(chunk_id="c1", doc_id="d1", text="某观点。")])])
        llm = FakeLLM(
            [
                decompose_response(
                    [
                        {"id": "q1", "query": "宁德时代2025年营业收入是多少"},
                        {"id": "q2", "query": "市场对宁德时代增长逻辑的看法"},
                    ]
                )
            ]
        )
        answerer = FakeAnswerer([abstained_draft("no_evidence")])
        state = run_agentic_rag(
            runtime=runtime,
            answerer=answerer,
            llm=llm,
            config=AgentConfig(),
            query=MULTI_HOP_QUERY,
        )
        self.assertTrue(state["final_answer"]["abstained"])
        self.assertEqual(state["rewrite_count"], 0)
        self.assertEqual(state["termination_reason"], "completed")


if __name__ == "__main__":
    unittest.main()
