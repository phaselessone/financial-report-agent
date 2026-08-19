"""extract_claims node tests (checklist v3.0 §P7).

The node turns a draft answer into a claim list: one remote LLM call that
splits the answer text into claim candidates, deterministic local claim_type,
and `supported` reusing the draft's whole-answer support_validation.
"""

from __future__ import annotations

import json
import unittest

from src.agent.config import AgentConfig
from src.agent.nodes.extract_claims import make_extract_claims
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
        self.assertTrue(all(c["supported"] for c in claims))  # reused whole-answer support
        self.assertEqual({c["claim_id"] for c in claims}, {"claim-1", "claim-2"})
        self.assertTrue(all(c["evidence_ids"] == ["c1"] for c in claims))
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

    def test_malformed_json_falls_back_to_whole_answer(self) -> None:
        llm = _llm_with("not json at all")
        node = make_extract_claims(llm, AgentConfig())
        state = node(_claims_state())
        claims = state["claims"]
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0]["text"], "宁德时代2025年营收1234亿元，同比增长31%。")
        # whole-answer fallback; "同比增长31%" carries a derived figure -> DERIVED
        self.assertEqual(claims[0]["claim_type"], "DERIVED")
        self.assertEqual(claims[0]["supported"], True)

    def test_llm_call_counted_and_logged(self) -> None:
        llm = _llm_with(json.dumps({"claims": [{"text": "某事实。"}]}, ensure_ascii=False))
        node = make_extract_claims(llm, AgentConfig())
        state = node(_claims_state())
        self.assertEqual(state["llm_call_count"], 1)
        self.assertEqual(len(state["llm_calls_log"]), 1)
        self.assertEqual(state["llm_calls_log"][0]["node"], "extract_claims")


if __name__ == "__main__":
    unittest.main()
