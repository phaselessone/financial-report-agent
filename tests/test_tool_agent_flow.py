"""LangGraph integration tests for plan/execute/observe (M3)."""

from __future__ import annotations

import unittest
from decimal import Decimal

from src.agent.config import AgentConfig
from src.agent.graph import run_agentic_rag
from src.agent.nodes.observe_tool_result import make_observe_tool_result
from src.agent.tool_orchestration import execute_tool_plan, make_tool_plan
from src.structured.fact_store import FactStore
from src.structured.schema import FinancialFact, Period
from tests.test_agent_flow import FakeAnswerer, FakeLLM, FakeRuntime, make_result, make_row, supported_draft


ALIASES = {"贵州茅台": ["贵州茅台", "茅台"]}


def make_fact() -> FinancialFact:
    return FinancialFact(
        company="贵州茅台", metric="revenue", period=Period("FY", 2025),
        value_type="actual", value=Decimal("123450000000"), unit="元",
        doc_id="doc-maotai", page=3, evidence_id="src-e1", raw_value="1234.5亿元",
        source_span="贵州茅台2025年营业收入1234.5亿元",
    )


def make_revenue_fact(year: int, value: str, evidence_id: str) -> FinancialFact:
    return FinancialFact(
        company="贵州茅台",
        metric="revenue",
        period=Period("FY", year),
        value_type="actual",
        value=Decimal(value),
        unit="元",
        doc_id=f"doc-maotai-{year}",
        page=3,
        evidence_id=evidence_id,
        raw_value=f"{Decimal(value) / Decimal('1e8')}亿元",
        source_span=f"贵州茅台{year}年营业收入",
    )


class ToolAgentFlowTests(unittest.TestCase):
    def test_observe_stores_calculation_record_and_trajectory_candidate(self) -> None:
        config = AgentConfig(max_steps=5)
        plan = make_tool_plan(
            "calculator",
            {
                "operation": "ratio",
                "inputs": [
                    {"name": "numerator", "value": 3, "fact_id": "F1"},
                    {"name": "denominator", "value": 8, "fact_id": "F2"},
                ],
            },
            reason="calculation_facts_ready",
        )
        observation = execute_tool_plan(plan)
        state = {
            "pending_tool_plan": plan,
            "pending_tool_observation": observation,
            "tool_calls": [],
            "tool_call_keys": set(),
            "trajectory_events": [],
            "step_count": 0,
            "evidence_pool": {},
            "seen_chunk_ids": set(),
            "structured_facts": [],
        }
        updated = make_observe_tool_result(config)(state)
        self.assertEqual(updated["tool_outcome"], "calculation_success")
        self.assertEqual(len(updated["calculations"]), 1)
        calculation_id, calculation = next(iter(updated["calculations"].items()))
        self.assertEqual(calculation_id, calculation["calculation_id"])
        self.assertEqual(updated["tool_calls"][0]["calculation_id"], calculation_id)
        self.assertEqual(updated["trajectory_events"][0]["calculation_id"], calculation_id)

    def test_structured_lookup_is_shortest_path(self) -> None:
        store = FactStore(":memory:")
        self.addCleanup(store.close)
        store.upsert([make_fact()])
        runtime = FakeRuntime([])
        state = run_agentic_rag(
            runtime=runtime,
            answerer=FakeAnswerer([supported_draft("贵州茅台2025年营业收入为1234.5亿元。")]),
            llm=FakeLLM([]),
            config=AgentConfig(enable_tool_orchestration=True),
            query="贵州茅台2025年营业收入是多少？",
            fact_store=store,
            company_aliases=ALIASES,
        )
        self.assertEqual(runtime.calls, [])
        self.assertEqual(state["tool_calls"][0]["tool_name"], "structured_lookup")
        self.assertEqual(state["tool_calls"][0]["status"], "SUCCESS")
        self.assertEqual(state["tool_outcome"], "structured_hit")
        self.assertEqual(state["termination_reason"], "completed")
        final = state["final_answer"]
        self.assertFalse(final["abstained"])
        self.assertEqual(final["used_evidence_ids"], ["src-e1"])
        self.assertEqual(len(final["claims"]), 1)
        self.assertEqual(final["claims"][0]["verification"]["status"], "ENTAILED")
        self.assertEqual(final["claims"][0]["evidence_ids"], ["src-e1"])
        self.assertEqual(len(final["citations"]), 1)
        self.assertEqual(final["citations"][0]["doc_id"], "doc-maotai")
        self.assertEqual(final["citations"][0]["page_start"], 3)

    def test_structured_miss_falls_back_to_report_search(self) -> None:
        store = FactStore(":memory:")
        self.addCleanup(store.close)
        runtime = FakeRuntime([make_result([make_row(chunk_id="c1", doc_id="d1", text="贵州茅台2025年营业收入1234.5亿元。")])])
        state = run_agentic_rag(
            runtime=runtime,
            answerer=FakeAnswerer([supported_draft("贵州茅台2025年营业收入1234.5亿元。")]),
            llm=FakeLLM([]),
            config=AgentConfig(enable_tool_orchestration=True),
            query="贵州茅台2025年营业收入是多少？",
            fact_store=store,
            company_aliases=ALIASES,
        )
        self.assertEqual([item["tool_name"] for item in state["tool_calls"][:2]], ["structured_lookup", "report_search"])
        self.assertEqual(runtime.calls, ["贵州茅台2025年营业收入是多少？"])
        self.assertEqual(state["retrieval_count"], 1)

    def test_yoy_fetches_both_facts_then_verifies_derived_claim(self) -> None:
        store = FactStore(":memory:")
        self.addCleanup(store.close)
        store.upsert(
            [
                make_revenue_fact(2024, "10000000000", "E-prior"),
                make_revenue_fact(2025, "12000000000", "E-current"),
            ]
        )
        state = run_agentic_rag(
            runtime=FakeRuntime([]),
            answerer=FakeAnswerer([supported_draft("同比增长20%。")]),
            llm=FakeLLM([]),
            config=AgentConfig(enable_tool_orchestration=True),
            query="贵州茅台2025年营业收入同比增长率是多少？",
            fact_store=store,
            company_aliases=ALIASES,
        )
        self.assertEqual(
            [item["tool_name"] for item in state["tool_calls"]],
            ["structured_lookup", "structured_lookup", "calculator"],
        )
        self.assertEqual(state["termination_reason"], "completed")
        self.assertEqual(len(state["calculations"]), 1)
        claim = state["claims"][0]
        self.assertEqual(claim["claim_type"], "DERIVED")
        self.assertEqual(claim["verification"]["status"], "ENTAILED")
        self.assertTrue(claim["calculation_id"].startswith("CALC-"))
        self.assertEqual(claim["evidence_ids"], ["E-prior", "E-current"])

    def test_report_tool_failure_terminates_with_reason(self) -> None:
        class BrokenRuntime:
            def search(self, query):
                raise RuntimeError("index unavailable")

        state = run_agentic_rag(
            runtime=BrokenRuntime(),
            answerer=FakeAnswerer([]),
            llm=FakeLLM([]),
            config=AgentConfig(enable_tool_orchestration=True),
            query="行业景气度如何？",
        )
        self.assertEqual(state["termination_reason"], "tool_execution_failed")
        self.assertTrue(state["final_answer"]["abstained"])
        self.assertEqual(state["tool_calls"][0]["error_type"], "RuntimeError")

    def test_tool_call_budget_stops_before_execution(self) -> None:
        state = run_agentic_rag(
            runtime=FakeRuntime([]), answerer=FakeAnswerer([]), llm=FakeLLM([]),
            config=AgentConfig(enable_tool_orchestration=True, max_tool_calls=0),
            query="行业景气度如何？",
        )
        self.assertEqual(state["termination_reason"], "max_tool_calls")
        self.assertEqual(state["tool_call_count"], 0)


if __name__ == "__main__":
    unittest.main()
