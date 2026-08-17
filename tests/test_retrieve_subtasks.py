"""retrieve_subtasks node + structured bridge tests (checklist v3.0 §P6)."""

from __future__ import annotations

import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from src.agent.config import AgentConfig
from src.agent.nodes.retrieve_subtasks import make_retrieve_subtasks
from src.agent.state import new_agent_state
from src.structured.agent_bridge import (
    financial_fact_to_row,
    format_fact_line,
    load_structured_context,
)
from src.structured.fact_store import FactStore
from src.structured.schema import FinancialFact, Period
from tests.test_agent_flow import FakeRuntime, make_result, make_row

ALIASES = {"贵州茅台": ["贵州茅台", "茅台"], "五粮液": ["五粮液"]}


def make_fact(company="贵州茅台", metric="revenue", year=2025, value="123450000000", doc_id="doc-maotai-1", **overrides):
    base = dict(
        company=company,
        metric=metric,
        period=Period(kind="FY", year=year),
        value_type="actual",
        value=Decimal(value),
        unit="元",
        doc_id=doc_id,
        page=3,
        evidence_id="E1",
        raw_value="1234.5亿元",
        source_span="2025年公司实现营业收入1234.5亿元",
    )
    base.update(overrides)
    return FinancialFact(**base)


class FinancialFactToRowTests(unittest.TestCase):
    def test_row_contract(self) -> None:
        row = financial_fact_to_row(make_fact())
        self.assertEqual(row["doc_id"], "doc-maotai-1")
        self.assertEqual(row["page_start"], 3)
        self.assertEqual(row["page_end"], 3)
        self.assertEqual(row["element_type"], "paragraph")
        self.assertEqual(row["chunk_type"], "structured_fact")
        self.assertEqual(row["score"], 5.0)
        self.assertEqual(row["rerank_score"], 5.0)
        self.assertTrue(row["chunk_id"].startswith("structured:"))
        self.assertIn("贵州茅台", row["support_span"])
        self.assertIn("1234.5亿元", row["support_span"])

    def test_format_fact_line(self) -> None:
        line = format_fact_line(make_fact())
        self.assertIn("贵州茅台", line)
        self.assertIn("FY2025", line)
        self.assertIn("营业收入", line)
        self.assertIn("1234.5亿元", line)


class RetrieveSubtasksTests(unittest.TestCase):
    def test_structured_hit_records_and_skips_runtime(self) -> None:
        store = FactStore(":memory:")
        self.addCleanup(store.close)
        store.upsert([make_fact()])
        runtime = FakeRuntime([])
        node = make_retrieve_subtasks(runtime, store, ALIASES, AgentConfig())
        state = new_agent_state(query="比较贵州茅台和五粮液的营收")
        state["sub_questions"] = [{"id": "q1", "query": "贵州茅台2025年营业收入是多少", "required_fields": []}]
        node(state)
        self.assertEqual(len(state["structured_facts"]), 1)
        self.assertTrue(state["structured_facts"][0]["routed"])
        self.assertEqual(state["structured_facts"][0]["facts"][0]["company"], "贵州茅台")
        self.assertEqual(runtime.calls, [])
        self.assertEqual(state["retrieval_count"], 0)
        self.assertEqual(state["sub_question_results"][0]["source"], "structured")

    def test_store_miss_falls_back_to_rag(self) -> None:
        store = FactStore(":memory:")
        self.addCleanup(store.close)
        runtime = FakeRuntime([make_result([make_row(chunk_id="c1", doc_id="d1", text="茅台营收。")])])
        node = make_retrieve_subtasks(runtime, store, ALIASES, AgentConfig())
        state = new_agent_state(query="q")
        state["sub_questions"] = [{"id": "q1", "query": "贵州茅台2025年营业收入是多少"}]
        node(state)
        self.assertEqual(state["retrieval_count"], 1)
        self.assertEqual(state["structured_facts"], [])
        self.assertEqual(state["sub_question_results"][0]["source"], "rag")

    def test_no_fact_store_all_rag_and_restores_query(self) -> None:
        runtime = FakeRuntime(
            [
                make_result([make_row(chunk_id="c1", doc_id="d1", text="x")]),
                make_result([make_row(chunk_id="c2", doc_id="d2", text="y")]),
            ]
        )
        node = make_retrieve_subtasks(runtime, None, None, AgentConfig())
        state = new_agent_state(query="比较A和B")
        state["sub_questions"] = [
            {"id": "q1", "query": "A"},
            {"id": "q2", "query": "B"},
        ]
        node(state)
        self.assertEqual(state["retrieval_count"], 2)
        self.assertEqual(len(state["evidence_pool"]), 2)
        self.assertEqual(state["active_query"], "比较A和B")

    def test_mixed_structured_and_rag(self) -> None:
        store = FactStore(":memory:")
        self.addCleanup(store.close)
        store.upsert([make_fact()])
        runtime = FakeRuntime([make_result([make_row(chunk_id="c1", doc_id="d1", text="机构观点。")])])
        node = make_retrieve_subtasks(runtime, store, ALIASES, AgentConfig())
        state = new_agent_state(query="混合")
        state["sub_questions"] = [
            {"id": "q1", "query": "贵州茅台2025年营业收入是多少"},
            {"id": "q2", "query": "机构对白酒怎么看"},
        ]
        node(state)
        self.assertEqual(len(state["structured_facts"]), 1)
        self.assertEqual(state["retrieval_count"], 1)
        self.assertEqual(len(state["sub_question_results"]), 2)
        self.assertIn("c1", state["evidence_pool"])
        self.assertTrue(any(key.startswith("structured:") for key in state["evidence_pool"]))


class LoadStructuredContextTests(unittest.TestCase):
    def test_missing_returns_none(self) -> None:
        store, aliases = load_structured_context("/nonexistent/facts.jsonl", "/nonexistent/aliases.json")
        self.assertIsNone(store)
        self.assertIsNone(aliases)

    def test_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            facts_path = Path(tmp) / "facts.jsonl"
            aliases_path = Path(tmp) / "aliases.json"
            facts_path.write_text(json.dumps(make_fact().to_dict(), ensure_ascii=False) + "\n", encoding="utf-8")
            aliases_path.write_text(json.dumps(ALIASES, ensure_ascii=False), encoding="utf-8")
            store, aliases = load_structured_context(facts_path, aliases_path)
            self.assertIsNotNone(store)
            self.assertEqual(aliases, ALIASES)
            self.assertEqual(store.count(), 1)
            store.close()


if __name__ == "__main__":
    unittest.main()
