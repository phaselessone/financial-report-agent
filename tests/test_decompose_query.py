"""decompose_query node + multi-hop trigger tests (checklist v3.0 §P6)."""

from __future__ import annotations

import json
import unittest

from src.agent.config import AgentConfig
from src.agent.nodes.decompose_query import make_decompose_query, parse_sub_questions
from src.agent.state import new_agent_state
from src.generation.routing import is_multi_hop_query
from src.llm.types import LLMResponse
from tests.test_agent_flow import FakeLLM


def decompose_response(sub_questions, *, prompt: int = 10, completion: int = 5) -> LLMResponse:
    return LLMResponse(
        content=json.dumps({"sub_questions": sub_questions}, ensure_ascii=False),
        provider="fake",
        model="fake",
        prompt_tokens=prompt,
        completion_tokens=completion,
    )


class ParseSubQuestionsTests(unittest.TestCase):
    def test_valid_payload(self) -> None:
        subs = parse_sub_questions(
            {
                "sub_questions": [
                    {"id": "q1", "query": "A公司2025年营收", "required_fields": ["company", "metric"]},
                    {"id": "q2", "query": "B公司2025年营收", "required_fields": []},
                ]
            },
            query="比较A和B",
            max_sub_questions=6,
        )
        self.assertEqual(len(subs), 2)
        self.assertEqual(subs[0]["id"], "q1")
        self.assertEqual(subs[0]["query"], "A公司2025年营收")
        self.assertEqual(subs[0]["required_fields"], ["company", "metric"])
        self.assertEqual(subs[1]["required_fields"], [])

    def test_id_normalized(self) -> None:
        subs = parse_sub_questions({"sub_questions": [{"query": "x"}]}, query="q", max_sub_questions=6)
        self.assertEqual(subs[0]["id"], "q1")

    def test_truncates_beyond_max(self) -> None:
        subs = parse_sub_questions(
            {"sub_questions": [{"query": f"q{i}"} for i in range(8)]},
            query="q",
            max_sub_questions=6,
        )
        self.assertEqual(len(subs), 6)

    def test_drops_empty_and_duplicate(self) -> None:
        subs = parse_sub_questions(
            {
                "sub_questions": [
                    {"query": ""},
                    {"query": "x"},
                    {"query": "x"},
                    {"query": "y"},
                ]
            },
            query="q",
            max_sub_questions=6,
        )
        self.assertEqual([s["query"] for s in subs], ["x", "y"])

    def test_malformed_none(self) -> None:
        self.assertIsNone(parse_sub_questions({}, query="q", max_sub_questions=6))
        self.assertIsNone(parse_sub_questions({"sub_questions": "x"}, query="q", max_sub_questions=6))
        self.assertIsNone(parse_sub_questions({"sub_questions": []}, query="q", max_sub_questions=6))
        self.assertIsNone(parse_sub_questions(None, query="q", max_sub_questions=6))

    def test_required_fields_non_list_coerced(self) -> None:
        subs = parse_sub_questions(
            {"sub_questions": [{"query": "x", "required_fields": "company"}]},
            query="q",
            max_sub_questions=6,
        )
        self.assertEqual(subs[0]["required_fields"], [])


class IsMultiHopQueryTests(unittest.TestCase):
    def test_comparison_stays_single_hop(self) -> None:
        self.assertFalse(is_multi_hop_query("比较A公司和B公司的营收"))

    def test_inductive_stays_single_hop(self) -> None:
        self.assertFalse(is_multi_hop_query("近期研报共同关注哪些主题"))

    def test_plain_fact(self) -> None:
        self.assertFalse(is_multi_hop_query("贵州茅台2025年营业收入是多少"))

    def test_numeric_opinion_mix(self) -> None:
        self.assertTrue(is_multi_hop_query("宁德时代2025年营业收入是多少？市场对其增长逻辑怎么看？"))

    def test_report_lookup(self) -> None:
        self.assertFalse(is_multi_hop_query("找一份半导体行业研报"))


class DecomposeNodeTests(unittest.TestCase):
    def test_decomposes(self) -> None:
        llm = FakeLLM(
            [
                decompose_response(
                    [
                        {"id": "q1", "query": "A公司2025年营收"},
                        {"id": "q2", "query": "B公司2025年营收"},
                    ]
                )
            ]
        )
        state = new_agent_state(query="比较A公司和B公司的营收")
        make_decompose_query(llm, AgentConfig())(state)
        self.assertEqual(len(state["sub_questions"]), 2)
        self.assertEqual(state["llm_call_count"], 1)
        self.assertEqual(len(llm.calls), 1)

    def test_fallback_single_subquestion(self) -> None:
        llm = FakeLLM([LLMResponse(content="{}", provider="fake", model="fake")])
        state = new_agent_state(query="比较A公司和B公司的营收")
        make_decompose_query(llm, AgentConfig())(state)
        self.assertEqual(len(state["sub_questions"]), 1)
        self.assertEqual(state["sub_questions"][0]["query"], "比较A公司和B公司的营收")

    def test_max_llm_calls_terminates(self) -> None:
        llm = FakeLLM([])
        state = new_agent_state(query="比较A公司和B公司的营收")
        make_decompose_query(llm, AgentConfig(max_llm_calls=0))(state)
        self.assertEqual(state["termination_reason"], "max_llm_calls")
        self.assertEqual(len(llm.calls), 0)


if __name__ == "__main__":
    unittest.main()
