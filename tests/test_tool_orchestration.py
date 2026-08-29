"""Controlled tool orchestration tests (M3)."""

from __future__ import annotations

import unittest
from decimal import Decimal

from src.agent.tool_orchestration import (
    detect_calculation_intent,
    execute_tool_plan,
    make_tool_plan,
    plan_next_tool,
    tool_call_key,
)
from src.structured.fact_store import FactStore
from src.structured.schema import FinancialFact, Period
from tests.test_agent_flow import FakeRuntime, make_result, make_row


ALIASES = {"贵州茅台": ["贵州茅台", "茅台"]}


def fact() -> FinancialFact:
    return FinancialFact(
        company="贵州茅台",
        metric="revenue",
        period=Period("FY", 2025),
        value_type="actual",
        value=Decimal("123450000000"),
        unit="元",
        doc_id="doc-maotai",
        page=3,
        evidence_id="source-E1",
        raw_value="1234.5亿元",
        source_span="2025年营业收入1234.5亿元",
    )


class ToolPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = FactStore(":memory:")
        self.addCleanup(self.store.close)

    def test_complete_signature_prefers_structured_lookup(self) -> None:
        plan = plan_next_tool(
            query="贵州茅台2025年营业收入是多少？",
            fact_store=self.store,
            company_aliases=ALIASES,
        )
        self.assertEqual(plan["next_action"], "structured_lookup")
        self.assertEqual(plan["arguments"]["metric"], "revenue")

    def test_non_structured_query_uses_report_search(self) -> None:
        plan = plan_next_tool(query="白酒行业景气度如何？", fact_store=self.store, company_aliases=ALIASES)
        self.assertEqual(plan["next_action"], "report_search")

    def test_duplicate_structured_call_falls_through_to_report_search(self) -> None:
        first = plan_next_tool(
            query="贵州茅台2025年营业收入是多少？",
            fact_store=self.store,
            company_aliases=ALIASES,
        )
        second = plan_next_tool(
            query="贵州茅台2025年营业收入是多少？",
            fact_store=self.store,
            company_aliases=ALIASES,
            attempted_call_keys={first["call_key"]},
        )
        self.assertEqual(second["next_action"], "report_search")

    def test_duplicate_report_search_finishes(self) -> None:
        first = plan_next_tool(query="行业趋势")
        second = plan_next_tool(query="行业趋势", attempted_call_keys={first["call_key"]})
        self.assertEqual(second["next_action"], "finish")
        self.assertEqual(second["reason"], "duplicate_tool_call")

    def test_non_whitelisted_action_rejected(self) -> None:
        with self.assertRaises(ValueError):
            make_tool_plan("shell", {"cmd": "echo unsafe"})

    def test_call_key_is_order_independent(self) -> None:
        self.assertEqual(tool_call_key("report_search", {"a": 1, "b": 2}), tool_call_key("report_search", {"b": 2, "a": 1}))

    def test_calculation_intent_is_conservative_and_bounded(self) -> None:
        intent = detect_calculation_intent("Acme 2021 to 2023 revenue CAGR", company_aliases={"Acme": ["Acme"]})
        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertEqual(intent["operation"], "cagr")
        self.assertEqual(intent["years"], 2)
        self.assertEqual([item["name"] for item in intent["operands"]], ["start", "end"])
        self.assertIsNone(detect_calculation_intent("Acme 2023 revenue", company_aliases={"Acme": ["Acme"]}))

    def test_calculation_intent_covers_common_financial_phrasings(self) -> None:
        aliases = {"贵州茅台": ["贵州茅台", "茅台"]}
        yoy_intent = detect_calculation_intent("贵州茅台2025年营业收入同比增长率是多少？", company_aliases=aliases)
        self.assertIsNotNone(yoy_intent)
        assert yoy_intent is not None
        self.assertEqual(yoy_intent["operation"], "yoy")
        self.assertEqual([item["period"]["year"] for item in yoy_intent["operands"]], [2024, 2025])

        ratio_intent = detect_calculation_intent("贵州茅台2025年研发费用占营业收入比例是多少？", company_aliases=aliases)
        self.assertIsNotNone(ratio_intent)
        assert ratio_intent is not None
        self.assertEqual(ratio_intent["operation"], "ratio")
        self.assertEqual({item["metric"] for item in ratio_intent["operands"]}, {"rd_expense", "revenue"})

        pp_intent = detect_calculation_intent("毛利率从20%升至25%变化了多少个百分点？")
        self.assertIsNotNone(pp_intent)
        assert pp_intent is not None
        self.assertEqual(pp_intent["operation"], "percentage_point_change")

        margin_intent = detect_calculation_intent("贵州茅台2025年毛利润除以营业收入是多少？", company_aliases=aliases)
        self.assertIsNotNone(margin_intent)
        assert margin_intent is not None
        self.assertEqual(margin_intent["operation"], "gross_margin")

    def test_calculation_planner_fetches_facts_before_calculator(self) -> None:
        intent = {
            "operation": "yoy",
            "precision": 4,
            "operands": [
                {"name": "prior", "query": "Acme 2024 revenue"},
                {"name": "current", "query": "Acme 2025 revenue"},
            ],
        }
        first = plan_next_tool(query="ignored", calculation_intent=intent)
        self.assertEqual(first["next_action"], "structured_lookup")
        self.assertEqual(first["arguments"]["calculation_operand"], "prior")
        second = plan_next_tool(
            query="ignored",
            calculation_intent=intent,
            attempted_call_keys={first["call_key"]},
            calculation_facts={
                "prior": {
                    "value": "10000000000",
                    "unit": "元",
                    "evidence_id": "E-prior",
                    "fact_id": "F-prior",
                    "raw_value": "100亿元",
                }
            },
        )
        self.assertEqual(second["next_action"], "structured_lookup")
        self.assertEqual(second["arguments"]["calculation_operand"], "current")
        facts = {
            "prior": {
                "value": "10000000000",
                "unit": "元",
                "evidence_id": "E-prior",
                "fact_id": "F-prior",
                "raw_value": "100亿元",
                "period": {"kind": "FY", "year": 2024},
            },
            "current": {
                "value": "12000000000",
                "unit": "元",
                "evidence_id": "E-current",
                "fact_id": "F-current",
                "raw_value": "120亿元",
                "period": {"kind": "FY", "year": 2025},
            },
        }
        calculator = plan_next_tool(query="ignored", calculation_intent=intent, calculation_facts=facts)
        self.assertEqual(calculator["next_action"], "calculator")
        self.assertEqual(calculator["arguments"]["operation"], "yoy")
        self.assertEqual(calculator["arguments"]["inputs"][0]["evidence_id"], "E-prior")


class ToolExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = FactStore(":memory:")
        self.addCleanup(self.store.close)

    def test_structured_hit_returns_rows_with_provenance(self) -> None:
        self.store.upsert([fact()])
        plan = make_tool_plan("structured_lookup", {"query": "茅台2025年营业收入"})
        result = execute_tool_plan(plan, fact_store=self.store, company_aliases=ALIASES)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(len(result["payload"]["rows"]), 1)
        self.assertEqual(result["payload"]["facts"][0]["fact_id"], fact().fact_id)

    def test_structured_miss_is_successful_observation(self) -> None:
        plan = make_tool_plan("structured_lookup", {"query": "茅台2025年营业收入"})
        result = execute_tool_plan(plan, fact_store=self.store, company_aliases=ALIASES)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["result_summary"], "rows=0")

    def test_report_search_reuses_runtime(self) -> None:
        runtime = FakeRuntime([make_result([make_row(chunk_id="c1", doc_id="d1", text="证据")])])
        plan = make_tool_plan("report_search", {"query": "趋势", "domain_hint": ""})
        result = execute_tool_plan(plan, runtime=runtime)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["result_summary"], "rows=1")
        self.assertEqual(runtime.calls, ["趋势"])

    def test_report_search_resolves_requested_operand_from_multi_number_row(self) -> None:
        runtime = FakeRuntime(
            [
                make_result(
                    [
                        make_row(
                            chunk_id="c1",
                            doc_id="d1",
                            text="贵州茅台2024年营业收入100亿元；2025年营业收入120亿元。",
                        )
                    ]
                )
            ]
        )
        plan = make_tool_plan(
            "report_search",
            {
                "query": "贵州茅台2025年营业收入",
                "calculation_operand": "current",
                "company": "贵州茅台",
                "metric": "revenue",
                "period": {"kind": "FY", "year": 2025},
                "unit": "亿元",
            },
        )

        result = execute_tool_plan(plan, runtime=runtime)

        facts = result["payload"]["calculation_facts"]
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["value"], "120")
        self.assertEqual(facts[0]["unit"], "亿元")
        self.assertEqual(facts[0]["evidence_id"], "c1")

    def test_missing_dependency_is_observable_failure(self) -> None:
        result = execute_tool_plan(make_tool_plan("report_search", {"query": "q"}))
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["error_type"], "ValueError")
        self.assertIn("unavailable", result["result_summary"])

    def test_finish_is_skipped_not_failed(self) -> None:
        result = execute_tool_plan(make_tool_plan("finish", {}, reason="done"))
        self.assertEqual(result["status"], "SKIPPED")
        self.assertEqual(result["error_type"], None)

    def test_calculator_uses_default_provenance_adapter_and_serializes_record(self) -> None:
        plan = make_tool_plan(
            "calculator",
            {
                "operation": "yoy",
                "inputs": [
                    {"name": "prior", "value": 100, "unit": "亿元", "evidence_id": "E1"},
                    {"name": "current", "value": 120, "unit": "亿元", "evidence_id": "E2"},
                ],
                "claimed_result": "20%",
            },
        )
        result = execute_tool_plan(plan)
        self.assertEqual(result["status"], "SUCCESS")
        calculation = result["payload"]["calculation"]
        self.assertEqual(calculation["status"], "SUCCESS")
        self.assertTrue(calculation["verified"])
        self.assertTrue(calculation["calculation_id"].startswith("CALC-"))


if __name__ == "__main__":
    unittest.main()
