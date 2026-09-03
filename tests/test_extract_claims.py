"""extract_claims node tests (checklist v3.0 §P7).

The node turns a draft answer into stable claim records: one remote LLM call
that splits the answer text into claim candidates plus deterministic local
claim_type. Claim-specific evidence mapping and verification happen later.
"""

from __future__ import annotations

import json
import unittest

from src.agent.config import AgentConfig
from src.agent.nodes.extract_claims import (
    deterministic_atomic_claims,
    make_extract_claims,
    stable_claim_id,
)
from src.llm.types import LLMResponse


def _llm_with(content: str, *, tokens: tuple[int, int] = (10, 5)):
    class _Fake:
        calls = 0

        def generate(self, **kwargs) -> LLMResponse:
            self.calls += 1
            return LLMResponse(
                content=content,
                provider="fake",
                model="fake",
                prompt_tokens=tokens[0],
                completion_tokens=tokens[1],
            )

    return _Fake()


def _claims_state(**overrides) -> dict:
    state = {
        "query": "宁德时代2025年营收和增长如何？",
        "draft_answer": {
            "answer": "宁德时代2025年营收1234亿元，同比增长31%。",
            "support_validation": {"supported": True, "support_filter_applied": "default"},
            "used_evidence_ids": ["c1"],
            "citations": [{"evidence_id": "E1", "chunk_id": "c1", "doc_id": "d1"}],
            "matched_numeric_tokens": ["1234"],
        },
        "step_count": 0,
        "llm_call_count": 0,
        "total_tokens": 0,
        "llm_calls_log": [],
    }
    state.update(overrides)
    return state


class ExtractClaimsTests(unittest.TestCase):
    def test_splits_draft_into_claims_with_local_type_and_supported(self) -> None:
        llm = _llm_with(
            json.dumps(
                {
                    "claims": [
                        {"text": "宁德时代2025年营收1234亿元。"},
                        {"text": "营收同比增长31%。"},
                    ]
                },
                ensure_ascii=False,
            )
        )
        node = make_extract_claims(llm, AgentConfig())
        state = node(_claims_state())
        claims = state["claims"]
        self.assertEqual(len(claims), 2)
        self.assertEqual(len({c["claim_id"] for c in claims}), 2)
        self.assertTrue(all(str(c["claim_id"]).startswith("C") for c in claims))
        self.assertTrue(all(c["evidence_ids"] == [] for c in claims))
        self.assertTrue(all(c["verification"]["status"] == "INSUFFICIENT" for c in claims))
        self.assertTrue(all(c["supported"] is False for c in claims))
        # local heuristic types: 营收...亿元 -> EXTRACTED, 同比增长31% -> DERIVED
        by_text = {c["text"]: c for c in claims}
        self.assertEqual(by_text["宁德时代2025年营收1234亿元。"]["claim_type"], "EXTRACTED")
        self.assertEqual(by_text["营收同比增长31%。"]["claim_type"], "DERIVED")
        self.assertEqual(state["llm_call_count"], 1)

    def test_uses_llm_claim_type_fallback_when_heuristic_unclear(self) -> None:
        # text "两家公司都强调AI驱动的逻辑" (no marker, single doc) -> EXTRACTED;
        # LLM's claim_type is only a fallback for ambiguous cases; with single-doc
        # non-derived text we still expect EXTRACTED locally.
        llm = _llm_with(
            json.dumps({"claims": [{"text": "两家公司都强调AI驱动的逻辑。"}]}, ensure_ascii=False)
        )
        node = make_extract_claims(llm, AgentConfig())
        state = node(_claims_state(draft_answer={**_claims_state()["draft_answer"]}))
        self.assertEqual(state["claims"][0]["claim_type"], "EXTRACTED")

    def test_preserves_explicit_core_type_steps_and_resolves_parent_to_stable_id(self) -> None:
        base_text = "甲公司2025年营收为120亿元。"
        conclusion_text = "甲公司的增长更快。"
        llm = _llm_with(
            json.dumps(
                {
                    "claims": [
                        {
                            "id": "base",
                            "text": base_text,
                            "claim_type": "EXTRACTED",
                            "is_core": False,
                            "source_step_ids": ["lookup-a-2025"],
                        },
                        {
                            "id": "conclusion",
                            "text": conclusion_text,
                            "claim_type": "SYNTHESIZED",
                            "is_core": True,
                            "source_step_ids": ["compare-growth"],
                            "parent_claim_ids": ["base"],
                        },
                    ]
                },
                ensure_ascii=False,
            )
        )

        state = make_extract_claims(llm, AgentConfig())(_claims_state())
        base, conclusion = state["claims"]

        self.assertFalse(base["is_core"])
        self.assertEqual(base["source_step_ids"], ["lookup-a-2025"])
        self.assertEqual(conclusion["claim_type"], "SYNTHESIZED")
        self.assertTrue(conclusion["is_core"])
        self.assertEqual(conclusion["source_step_ids"], ["compare-growth"])
        self.assertEqual(conclusion["parent_claim_ids"], [stable_claim_id(base_text)])

    def test_abstained_draft_yields_no_claims(self) -> None:
        llm = _llm_with(json.dumps({"claims": [{"text": "不应出现"}]}))
        node = make_extract_claims(llm, AgentConfig())
        state = node(
            _claims_state(
                draft_answer={
                    "answer": "",
                    "abstained": True,
                    "abstain_reason": "no_evidence",
                    "support_validation": {"supported": False},
                    "used_evidence_ids": [],
                    "citations": [],
                }
            )
        )
        self.assertEqual(state["claims"], [])
        self.assertEqual(llm.calls, 0)  # no LLM call for an abstained draft
        self.assertEqual(state["llm_call_count"], 0)

    def test_malformed_json_uses_deterministic_atomic_split(self) -> None:
        llm = _llm_with("not json at all")
        node = make_extract_claims(llm, AgentConfig())
        state = node(_claims_state())
        claims = state["claims"]
        self.assertEqual(
            [claim["text"] for claim in claims],
            ["宁德时代2025年营收1234亿元。", "同比增长31%。"],
        )
        self.assertEqual([claim["claim_type"] for claim in claims], ["EXTRACTED", "DERIVED"])
        self.assertTrue(all(claim["supported"] is False for claim in claims))
        self.assertTrue(
            all(claim["verification"]["status"] == "INSUFFICIENT" for claim in claims)
        )
        self.assertEqual(state["claim_extraction_fallback"], "deterministic_atomic_split")

    def test_deterministic_fallback_splits_sentences_but_keeps_coordinate_prefix(self) -> None:
        self.assertEqual(
            deterministic_atomic_claims("甲公司营收100亿元。乙公司净利润20亿元；行业需求回升。"),
            [
                {"text": "甲公司营收100亿元。"},
                {"text": "乙公司净利润20亿元；"},
                {"text": "行业需求回升。"},
            ],
        )
        self.assertEqual(
            deterministic_atomic_claims("截至2025年，甲公司营业收入为100亿元。"),
            [{"text": "截至2025年，甲公司营业收入为100亿元。"}],
        )
        self.assertEqual(
            deterministic_atomic_claims(
                "针对“甲公司2025年营业收入同比增长多少？”，计算结果为-15%。"
            ),
            [
                {
                    "text": "针对“甲公司2025年营业收入同比增长多少？”，计算结果为-15%。"
                }
            ],
        )

    def test_llm_call_counted_and_logged(self) -> None:
        llm = _llm_with(json.dumps({"claims": [{"text": "某事实。"}]}, ensure_ascii=False))
        node = make_extract_claims(llm, AgentConfig())
        state = node(_claims_state())
        self.assertEqual(state["llm_call_count"], 1)
        self.assertEqual(len(state["llm_calls_log"]), 1)
        self.assertEqual(state["llm_calls_log"][0]["node"], "extract_claims")


if __name__ == "__main__":
    unittest.main()
