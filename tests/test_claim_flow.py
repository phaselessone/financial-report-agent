"""End-to-end claim-level provenance flow tests (checklist v3.0 §P7).

Verifies the graph produces ``state["claims"]`` after synthesis for the two
representative supported shapes: single-hop and multi-hop. The abstained branch
is covered at node level (test_extract_claims.test_abstained_draft_yields_no_claims);
an end-to-end abstain is not asserted here because it depends on the pre-existing
grade/rewrite loop, which is out of P7 scope.
"""

from __future__ import annotations

import json
import unittest

from src.agent.config import AgentConfig
from src.agent.graph import run_agentic_rag
from src.llm.types import LLMResponse
from tests.test_agent_flow import (
    FakeAnswerer,
    FakeLLM,
    FakeRuntime,
    make_result,
    make_row,
)
from tests.test_multi_hop_flow import decompose_response

MULTI_HOP_QUERY = "宁德时代2025年营业收入是多少？市场对其增长逻辑怎么看？"


def claims_response(texts) -> LLMResponse:
    return LLMResponse(
        content=json.dumps({"claims": [{"text": t} for t in texts]}, ensure_ascii=False),
        provider="fake",
        model="fake",
        prompt_tokens=10,
        completion_tokens=5,
    )


def supported_draft_with_evidence(final_answer: str, *, used_evidence_ids, doc_ids) -> dict:
    """A supported draft shaped like the real answer_service output: final_answer
    holds the text, used_evidence_ids and citations carry the evidence."""
    return {
        "query": "",
        "question_type": "fact",
        "answer_mode": "fact",
        "final_answer": final_answer,
        "abstained": False,
        "abstain_reason": None,
        "used_evidence_ids": used_evidence_ids,
        "citations": [
            {"evidence_id": f"E{i}", "chunk_id": cid, "doc_id": doc_ids[i - 1]}
            for i, cid in enumerate(used_evidence_ids, start=1)
        ],
        "support_validation": {"supported": True, "matched_numeric_tokens": []},
        "selected_doc_ids": doc_ids,
    }


class ClaimFlowTests(unittest.TestCase):
    def test_single_hop_supported_answer_emits_claims(self) -> None:
        runtime = FakeRuntime([make_result([make_row(chunk_id="c1", doc_id="d1", text="宁德时代2025年营收1234亿元。")])])
        # Single-hop fact query: the fake LLM is consulted once, by extract_claims.
        llm = FakeLLM([claims_response(["宁德时代2025年营收1234亿元。"])])
        answerer = FakeAnswerer(
            [
                supported_draft_with_evidence(
                    "宁德时代2025年营收1234亿元。",
                    used_evidence_ids=["c1"],
                    doc_ids=["d1"],
                )
            ]
        )
        state = run_agentic_rag(
            runtime=runtime,
            answerer=answerer,
            llm=llm,
            config=AgentConfig(),
            query="宁德时代2025年营业收入是多少？",
            question_type="fact",
        )
        self.assertFalse(state["is_multi_hop"])
        claims = state["claims"]
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0]["claim_type"], "EXTRACTED")
        self.assertTrue(claims[0]["supported"])
        self.assertEqual(claims[0]["evidence_ids"], ["c1"])  # carried from draft
        self.assertEqual(state["termination_reason"], "completed")
        # synthesize's internal answerer call (1) + extract_claims (1)
        self.assertEqual(state["llm_call_count"], 2)

    def test_multi_hop_answer_emits_claims_from_both_legs(self) -> None:
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
                ),
                claims_response(["宁德时代2025年营收1234亿元。", "机构看好其增长逻辑。"]),
            ]
        )
        answerer = FakeAnswerer(
            [
                supported_draft_with_evidence(
                    "宁德时代2025年营收1234亿元，机构看好其增长逻辑。",
                    used_evidence_ids=["c1", "c2"],
                    doc_ids=["d1", "d2"],
                )
            ]
        )
        state = run_agentic_rag(
            runtime=runtime,
            answerer=answerer,
            llm=llm,
            config=AgentConfig(),
            query=MULTI_HOP_QUERY,
        )
        self.assertTrue(state["is_multi_hop"])
        self.assertEqual(len(state["claims"]), 2)
        self.assertTrue(all(c["supported"] for c in state["claims"]))
        # two distinct sources, no derived marker -> both SYNTHESIZED
        self.assertEqual({c["claim_type"] for c in state["claims"]}, {"SYNTHESIZED"})
        # synthesize internal (1) + decompose (1) + extract_claims (1)
        self.assertEqual(state["llm_call_count"], 3)


if __name__ == "__main__":
    unittest.main()
