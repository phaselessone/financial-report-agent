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
from unittest.mock import patch

from src.agent.config import AgentConfig
from src.agent.claims import stable_claim_id
from src.agent.graph import run_agentic_rag
from src.agent.nodes.verify_answer import _verify_claims
from src.agent.semantic_policy import activate_semantic_scorer
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
    def test_report_location_answer_keeps_only_claim_specific_document_and_page(self) -> None:
        claim_text = "《电子行业周报》第7页指出，半导体设备国产替代加速。"
        support = make_row(
            chunk_id="E-support",
            doc_id="electronic-weekly",
            text=claim_text,
        )
        support["page_start"] = 7
        support["page_end"] = 7
        distractor = make_row(
            chunk_id="E-distractor",
            doc_id="unrelated-report",
            text="《另一份报告》第12页讨论半导体行业库存。",
        )
        distractor["page_start"] = 12
        distractor["page_end"] = 12
        draft = {
            "query": "",
            "question_type": "fact",
            "answer_mode": "report_lookup",
            "final_answer": claim_text,
            "abstained": False,
            "abstain_reason": None,
            # The model-level draft deliberately cites both candidates.  The
            # strict claim gate must narrow the published citation to the row
            # that actually entails the atomic conclusion.
            "used_evidence_ids": ["E-support", "E-distractor"],
            "citations": [
                {
                    "evidence_id": "E-support",
                    "chunk_id": "E-support",
                    "doc_id": "electronic-weekly",
                    "page": 7,
                },
                {
                    "evidence_id": "E-distractor",
                    "chunk_id": "E-distractor",
                    "doc_id": "unrelated-report",
                    "page": 12,
                },
            ],
            "support_validation": {"supported": True},
            "selected_doc_ids": ["electronic-weekly", "unrelated-report"],
        }

        state = run_agentic_rag(
            runtime=FakeRuntime([make_result([support, distractor])]),
            answerer=FakeAnswerer([draft]),
            llm=FakeLLM([claims_response([claim_text])]),
            config=AgentConfig(strict_claim_verification=True, max_claim_retrievals=0),
            query="“半导体设备国产替代加速”这一结论在哪份报告哪一页？",
        )

        final = state["final_answer"]
        self.assertFalse(final["abstained"])
        self.assertEqual(final["used_evidence_ids"], ["E-support"])
        self.assertEqual(
            final["citations"],
            [
                {
                    "evidence_id": "E-support",
                    "chunk_id": "E-support",
                    "doc_id": "electronic-weekly",
                    "page": 7,
                }
            ],
        )
        self.assertEqual(len(final["claims"]), 1)
        self.assertEqual(final["claims"][0]["verification"]["status"], "ENTAILED")
        self.assertEqual(final["claims"][0]["evidence_ids"], ["E-support"])

    def test_synthesized_claim_with_dangling_parent_fails_closed(self) -> None:
        text = "甲公司的增长更快。"
        state = {
            "claims": [
                {
                    "claim_id": stable_claim_id(text),
                    "text": text,
                    "claim_type": "SYNTHESIZED",
                    "is_core": True,
                    "parent_claim_ids": ["C-missing"],
                }
            ],
            "evidence_pool": {
                "c1": {"chunk_id": "c1", "text": text, "doc_id": "d1", "page_start": 1}
            },
            "calculations": {},
            "is_multi_hop": False,
        }

        summary = _verify_claims(state, config=AgentConfig(), allow_claim_retrieval=False)

        self.assertEqual(summary["entailed_count"], 0)
        self.assertEqual(summary["insufficient_count"], 1)
        self.assertIn(
            "parent_claim_missing:C-missing",
            state["claims"][0]["verification"]["reasons"],
        )

    def test_synthesized_claim_requires_all_parents_to_be_entailed(self) -> None:
        base_text = "甲公司2025年营收为120亿元。"
        child_text = "甲公司的增长更快。"
        base_id = stable_claim_id(base_text)
        state = {
            "claims": [
                {
                    "claim_id": base_id,
                    "text": base_text,
                    "claim_type": "EXTRACTED",
                    "is_core": False,
                    "parent_claim_ids": [],
                },
                {
                    "claim_id": stable_claim_id(child_text),
                    "text": child_text,
                    "claim_type": "SYNTHESIZED",
                    "is_core": True,
                    "parent_claim_ids": [base_id],
                },
            ],
            "evidence_pool": {
                "c1": {"chunk_id": "c1", "text": child_text, "doc_id": "d1", "page_start": 1}
            },
            "calculations": {},
            "is_multi_hop": False,
        }

        summary = _verify_claims(state, config=AgentConfig(), allow_claim_retrieval=False)

        self.assertEqual(summary["entailed_count"], 0)
        self.assertEqual(summary["insufficient_count"], 2)
        self.assertIn(
            f"parent_claim_not_entailed:{base_id}",
            state["claims"][1]["verification"]["reasons"],
        )

    def test_core_synthesized_compare_claim_without_calculation_cannot_use_llm_judge(self) -> None:
        parent_a_text = "甲公司2025年营业收入同比增长20%。"
        parent_b_text = "乙公司2025年营业收入同比增长5%。"
        parent_a_id = stable_claim_id(parent_a_text)
        parent_b_id = stable_claim_id(parent_b_text)
        core_text = "甲公司的扩张动能领先乙公司。"
        judge_calls: list[dict[str, object]] = []

        def judge(payload):
            judge_calls.append(dict(payload))
            return {
                "status": "ENTAILED",
                "evidence_ids": ["E-A-prior", "E-A-current", "E-B-prior", "E-B-current"],
                "score": 0.99,
                "reasons": ["model inferred the comparison"],
            }

        state = {
            "claims": [
                {
                    "claim_id": parent_a_id,
                    "text": parent_a_text,
                    "claim_type": "DERIVED",
                    "is_core": False,
                    "source_step_ids": ["calculate_a_yoy"],
                    "calculation_id": "calc-a",
                    "parent_claim_ids": [],
                },
                {
                    "claim_id": parent_b_id,
                    "text": parent_b_text,
                    "claim_type": "DERIVED",
                    "is_core": False,
                    "source_step_ids": ["calculate_b_yoy"],
                    "calculation_id": "calc-b",
                    "parent_claim_ids": [],
                },
                {
                    "claim_id": stable_claim_id(core_text),
                    "text": core_text,
                    "claim_type": "SYNTHESIZED",
                    "is_core": True,
                    "source_step_ids": ["compare_growth"],
                    "calculation_id": None,
                    "parent_claim_ids": [parent_a_id, parent_b_id],
                },
            ],
            "reasoning_plan": {
                "plan_id": "comparison-plan",
                "steps": [
                    {"step_id": "calculate_a_yoy", "kind": "CALCULATE", "depends_on": []},
                    {"step_id": "calculate_b_yoy", "kind": "CALCULATE", "depends_on": []},
                    {
                        "step_id": "compare_growth",
                        "kind": "COMPARE",
                        "depends_on": ["calculate_a_yoy", "calculate_b_yoy"],
                    },
                ],
                "answer_requirement_ids": ["compare_growth"],
            },
            "reasoning_step_results": [
                {"step_id": "calculate_a_yoy", "status": "SUCCESS", "calculation_id": "calc-a"},
                {"step_id": "calculate_b_yoy", "status": "SUCCESS", "calculation_id": "calc-b"},
                {"step_id": "compare_growth", "status": "SUCCESS", "calculation_id": "calc-compare"},
            ],
            "evidence_pool": {
                "E-A-prior": {"evidence_id": "E-A-prior", "text": "甲公司2024年营业收入为100元。"},
                "E-A-current": {"evidence_id": "E-A-current", "text": "甲公司2025年营业收入为120元。"},
                "E-B-prior": {"evidence_id": "E-B-prior", "text": "乙公司2024年营业收入为200元。"},
                "E-B-current": {"evidence_id": "E-B-current", "text": "乙公司2025年营业收入为210元。"},
            },
            "calculations": {
                "calc-a": {
                    "calculation_id": "calc-a",
                    "status": "SUCCESS",
                    "verified": True,
                    "provenance_complete": True,
                    "formatted": "20%",
                    "evidence_ids": ["E-A-prior", "E-A-current"],
                },
                "calc-b": {
                    "calculation_id": "calc-b",
                    "status": "SUCCESS",
                    "verified": True,
                    "provenance_complete": True,
                    "formatted": "5%",
                    "evidence_ids": ["E-B-prior", "E-B-current"],
                },
                "calc-compare": {
                    "calculation_id": "calc-compare",
                    "status": "SUCCESS",
                    "verified": True,
                    "provenance_complete": True,
                    "input_calculation_ids": ["calc-a", "calc-b"],
                    "source_step_ids": ["calculate_a_yoy", "calculate_b_yoy"],
                    "evidence_ids": ["E-A-prior", "E-A-current", "E-B-prior", "E-B-current"],
                    "comparison": {"answer": "甲公司增长更快。", "winner": "甲公司", "loser": "乙公司"},
                },
            },
            "is_multi_hop": True,
        }

        summary = _verify_claims(
            state,
            config=AgentConfig(claim_llm_budget=1),
            llm_judge=judge,
            allow_claim_retrieval=False,
        )

        self.assertEqual(summary["entailed_count"], 2)
        self.assertEqual(summary["insufficient_count"], 1)
        self.assertEqual(judge_calls, [])
        self.assertIn(
            "calculation_backed_claim_missing_calculation",
            state["claims"][2]["verification"]["reasons"],
        )

    def test_config_semantic_threshold_is_passed_to_claim_verifier(self) -> None:
        claim_text = "事实"
        state = {
            "claims": [{"claim_id": stable_claim_id(claim_text), "text": claim_text}],
            "evidence_pool": {"c1": {"chunk_id": "c1", "text": "事实", "doc_id": "d1", "page_start": 1}},
            "calculations": {},
            "is_multi_hop": False,
        }
        observed = {}

        def fake_verify(claim, rows, **kwargs):
            observed["threshold"] = kwargs["semantic_threshold"]
            return {
                "evidence_ids": ["c1"],
                "verification": {
                    "status": "ENTAILED",
                    "method": "deterministic",
                    "score": 1.0,
                    "reasons": [],
                },
            }

        with patch("src.agent.nodes.verify_answer.verify_claim", side_effect=fake_verify):
            _verify_claims(state, config=AgentConfig(claim_semantic_threshold=0.42), allow_claim_retrieval=False)
        self.assertEqual(observed["threshold"], 0.42)

    def test_calibrated_semantic_scorer_supplies_its_reviewed_threshold(self) -> None:
        claim_text = "增长质量改善。"
        state = {
            "claims": [{"claim_id": stable_claim_id(claim_text), "text": claim_text}],
            "evidence_pool": {"c1": {"chunk_id": "c1", "text": "质量改善", "doc_id": "d1"}},
            "calculations": {},
            "is_multi_hop": False,
        }
        activated = activate_semantic_scorer(
            lambda _claim, _evidence: {"status": "ENTAILED", "score": 0.96},
            {
                "calibrated": True,
                "precision_constraint_satisfied": True,
                "coverage_constraint_satisfied": True,
                "threshold": 0.9,
                "metrics": {"precision": 0.97},
                "calibration_config": {"min_precision": 0.95},
                "label_contract": {"review_status": "reviewed"},
                "labels_sha256": "a" * 64,
                "scorer_kind": "directional_nli",
                "scorer_identity": {
                    "kind": "directional_nli",
                    "model": "fixture-nli",
                    "revision": "fixture-v1",
                    "config_sha256": "b" * 64,
                },
            },
            scorer_identity={
                "kind": "directional_nli",
                "model": "fixture-nli",
                "revision": "fixture-v1",
                "config_sha256": "b" * 64,
            },
        )
        observed = {}

        def fake_verify(claim, rows, **kwargs):
            observed.update(kwargs)
            return {
                "evidence_ids": ["c1"],
                "verification": {
                    "status": "ENTAILED",
                    "method": "semantic",
                    "score": 0.96,
                    "reasons": [],
                },
            }

        with patch("src.agent.nodes.verify_answer.verify_claim", side_effect=fake_verify):
            _verify_claims(
                state,
                config=AgentConfig(claim_semantic_threshold=0.42),
                semantic_scorer=activated,
                allow_claim_retrieval=False,
            )

        self.assertIs(observed["semantic_scorer"], activated)
        self.assertEqual(observed["semantic_threshold"], 0.9)

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
                claims_response(["宁德时代2025年营收1234亿元。", "机构看好宁德时代增长逻辑。"]),
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
        # Whole-answer citations do not make each independent atomic claim a
        # synthesis. Parent-linked conclusions are explicitly SYNTHESIZED.
        self.assertEqual({c["claim_type"] for c in state["claims"]}, {"EXTRACTED"})
        # synthesize internal (1) + decompose (1) + extract_claims (1)
        self.assertEqual(state["llm_call_count"], 3)


if __name__ == "__main__":
    unittest.main()
