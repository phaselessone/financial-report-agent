"""Agent flow tests: checklist v3.0 §P2 DoD scenarios (mocked deps, no API)."""

from __future__ import annotations

import unittest

from src.agent.config import AgentConfig
from src.agent.graph import run_agentic_rag
from src.llm.types import LLMResponse


def make_row(*, chunk_id: str, doc_id: str, text: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "file_name": f"{doc_id}.pdf",
        "page_start": 1,
        "page_end": 1,
        "text": text,
        "child_text": text,
        "support_span": text,
        "section_title": "",
        "section_path": "",
        "chunk_type": "text",
        "element_type": "paragraph",
        "score": 3.0,
        "rerank_score": 3.0,
    }


def llm_call_kinds(llm) -> list[str]:
    """Tag each captured LLM call by the node that made it.

    P7 adds a per-draft ``extract_claims`` call between synthesis and
    verification, so existing tests that asserted on absolute ``len(llm.calls)``
    or ``llm.calls[0]`` are now polluted by that fixed extra call. This helper
    lets those assertions stay meaningful by filtering for the node they care
    about (rewrite / decompose) instead of absolute counts or indices.
    """
    kinds: list[str] = []
    for call in llm.calls:
        texts = " ".join(
            str(message.get("content", ""))
            for message in call.get("messages", [])
            if isinstance(message, dict)
        )
        if "query decomposer" in texts:
            kinds.append("decompose")
        elif "claim extractor" in texts:
            kinds.append("extract_claims")
        elif "Current draft answer" in texts or "rewritten" in texts:
            kinds.append("rewrite")
        else:
            kinds.append("other")
    return kinds


def make_result(rows: list[dict]) -> dict:
    return {
        "query_mode": "fact",
        "numeric_query": False,
        "dense_rows": [],
        "bm25_rows": [],
        "hybrid_rows": rows,
        "rerank_rows": rows,
        "timings": {},
    }


class FakeRuntime:
    def __init__(self, results: list[dict]):
        self._results = list(results)
        self.calls: list[str] = []

    def search(self, query: str) -> dict:
        self.calls.append(query)
        if not self._results:
            return make_result([])
        return self._results.pop(0)


class FakeLLM:
    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def generate(self, **kwargs) -> LLMResponse:
        self.calls.append(kwargs)
        if self._responses:
            return self._responses.pop(0)
        return LLMResponse(content="{}", provider="fake", model="fake")


def rewrite_response(rewritten: str, reason: str, *, prompt: int = 10, completion: int = 5) -> LLMResponse:
    import json

    return LLMResponse(
        content=json.dumps({"rewritten_query": rewritten, "reason": reason}, ensure_ascii=False),
        provider="fake",
        model="fake",
        prompt_tokens=prompt,
        completion_tokens=completion,
    )


class FakeAnswerer:
    def __init__(self, drafts: list[dict], *, llm_calls_per_answer: int = 1):
        self._drafts = list(drafts)
        self.llm_calls: list[LLMResponse] = []
        self.answer_calls: list[dict] = []
        self._llm_calls_per_answer = llm_calls_per_answer

    def answer(self, **kwargs) -> dict:
        self.answer_calls.append(kwargs)
        for _ in range(self._llm_calls_per_answer):
            self.llm_calls.append(
                LLMResponse(content="{}", provider="fake", model="fake", prompt_tokens=20, completion_tokens=10)
            )
        if self._drafts:
            return self._drafts.pop(0)
        return {
            "query": kwargs.get("query", ""),
            "final_answer": "答。",
            "abstained": False,
            "support_validation": {"supported": True},
        }


def supported_draft(final_answer: str) -> dict:
    return {
        "query": "",
        "question_type": "fact",
        "answer_mode": "fact",
        "final_answer": final_answer,
        "abstained": False,
        "abstain_reason": None,
        "used_evidence_ids": [],
        "citations": [],
        "support_validation": {"supported": True, "matched_numeric_tokens": []},
        "selected_doc_ids": [],
    }


def abstained_draft(reason: str) -> dict:
    draft = supported_draft("信息不足，暂时无法给出可靠答案。")
    draft["abstained"] = True
    draft["abstain_reason"] = reason
    draft["support_validation"] = {"supported": False}
    return draft


class AgentFlowTests(unittest.TestCase):
    def test_easy_fact_uses_one_retrieval_and_no_llm_rewrite(self) -> None:
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
        self.assertEqual(state["termination_reason"], "completed")
        self.assertEqual(state["retrieval_count"], 1)
        self.assertEqual(state["rewrite_count"], 0)
        self.assertEqual(state["generation_count"], 1)
        # P7: a per-draft claim-extraction call happens, but no rewrite/decompose.
        kinds = llm_call_kinds(llm)
        self.assertNotIn("rewrite", kinds)
        self.assertNotIn("decompose", kinds)
        self.assertEqual(state["final_answer"]["final_answer"], "半导体行业景气度持续回升。")

    def test_failed_retrieval_rewrites_then_recovers(self) -> None:
        runtime = FakeRuntime(
            [
                make_result([]),  # first round: no evidence
                make_result([make_row(chunk_id="c1", doc_id="d1", text="半导体行业景气度持续回升。")]),
            ]
        )
        llm = FakeLLM([rewrite_response("半导体 景气度 回升", "no_evidence")])
        answerer = FakeAnswerer([supported_draft("半导体行业景气度持续回升。")])
        state = run_agentic_rag(
            runtime=runtime,
            answerer=answerer,
            llm=llm,
            config=AgentConfig(),
            query="半导体行业景气度如何？",
        )
        self.assertEqual(state["termination_reason"], "completed")
        self.assertEqual(state["retrieval_count"], 2)
        self.assertEqual(state["rewrite_count"], 1)
        self.assertEqual(state["rewritten_queries"][0]["reason"], "no_evidence")
        self.assertEqual(state["final_answer"]["final_answer"], "半导体行业景气度持续回升。")

    def test_numeric_missing_triggers_rewrite_then_retry(self) -> None:
        runtime = FakeRuntime(
            [
                make_result([make_row(chunk_id="c1", doc_id="d1", text="半导体行业景气度持续回升。")]),
                make_result([make_row(chunk_id="c2", doc_id="d1", text="晶圆代工价格同比上涨20%。")]),
            ]
        )
        llm = FakeLLM([rewrite_response("晶圆代工 价格 上涨", "numeric_missing")])
        answerer = FakeAnswerer([supported_draft("晶圆代工价格同比上涨20%。")])
        state = run_agentic_rag(
            runtime=runtime,
            answerer=answerer,
            llm=llm,
            config=AgentConfig(),
            query="晶圆代工价格是多少？",
        )
        self.assertEqual(state["retrieval_count"], 2)
        self.assertEqual(state["rewrite_count"], 1)
        self.assertEqual(state["rewritten_queries"][0]["reason"], "numeric_missing")
        self.assertEqual(state["termination_reason"], "completed")

    def test_comparison_source_insufficient_retries(self) -> None:
        single_source = make_result(
            [make_row(chunk_id="c1", doc_id="d1", text="A公司强调AI算力需求。"), make_row(chunk_id="c2", doc_id="d1", text="A公司聚焦国产替代。")]
        )
        two_sources = make_result(
            [make_row(chunk_id="c1", doc_id="d1", text="A公司强调AI算力需求。"), make_row(chunk_id="c3", doc_id="d2", text="B公司聚焦存储涨价。")]
        )
        runtime = FakeRuntime([single_source, two_sources])
        llm = FakeLLM([rewrite_response("A公司 B公司 策略 对比", "source_diversity_missing")])
        answerer = FakeAnswerer([supported_draft("A公司强调AI算力，B公司聚焦存储。")])
        state = run_agentic_rag(
            runtime=runtime,
            answerer=answerer,
            llm=llm,
            config=AgentConfig(),
            query="比较A公司和B公司的策略差异",
        )
        self.assertEqual(state["retrieval_count"], 2)
        self.assertEqual(state["rewrite_count"], 1)
        self.assertEqual(state["rewritten_queries"][0]["reason"], "source_diversity_missing")
        self.assertEqual(state["termination_reason"], "completed")

    def test_no_evidence_eventually_abstains(self) -> None:
        runtime = FakeRuntime([])  # always empty
        llm = FakeLLM(
            [
                rewrite_response("半导体 景气", "no_evidence"),
                rewrite_response("半导体 行业 景气 回升", "no_evidence"),
            ]
        )
        answerer = FakeAnswerer([abstained_draft("no_evidence")])
        state = run_agentic_rag(
            runtime=runtime,
            answerer=answerer,
            llm=llm,
            config=AgentConfig(),
            query="半导体行业景气度如何？",
        )
        self.assertTrue(state["final_answer"]["abstained"])
        self.assertEqual(state["retrieval_count"], 2)  # no-improvement stops searching after the rewrite
        self.assertEqual(state["rewrite_count"], 1)
        self.assertTrue(state["no_improvement"])
        self.assertEqual(state["termination_reason"], "completed")

    def test_verification_retry_regenerates_once(self) -> None:
        runtime = FakeRuntime([make_result([make_row(chunk_id="c1", doc_id="d1", text="半导体行业景气度持续回升。")])])
        llm = FakeLLM([])
        unsupported = supported_draft("第一次草稿")
        unsupported["support_validation"] = {"supported": False}
        answerer = FakeAnswerer([unsupported, supported_draft("第二次草稿（修复后）")])
        state = run_agentic_rag(
            runtime=runtime,
            answerer=answerer,
            llm=llm,
            config=AgentConfig(max_retrieval_rounds=1),  # legacy regenerate path when re-retrieval is exhausted
            query="半导体行业景气度如何？",
        )
        self.assertEqual(state["generation_count"], 2)
        self.assertEqual(state["termination_reason"], "completed")
        self.assertEqual(state["final_answer"]["final_answer"], "第二次草稿（修复后）")

    def test_unsupported_draft_retries_with_rewrite_and_fresh_evidence(self) -> None:
        first_evidence = make_result(
            [
                make_row(chunk_id="c1", doc_id="d1", text="A公司强调AI算力需求。"),
                make_row(chunk_id="c3", doc_id="d2", text="B公司聚焦存储涨价。"),
            ]
        )
        second_evidence = make_result(
            [
                make_row(chunk_id="c2", doc_id="d1", text="A公司强调AI算力需求。"),
                make_row(chunk_id="c4", doc_id="d2", text="B公司聚焦存储涨价。"),
            ]
        )
        runtime = FakeRuntime([first_evidence, second_evidence])
        llm = FakeLLM([rewrite_response("A公司 B公司 策略 对比", "source_diversity_missing")])
        unsupported = supported_draft("第一次草稿")
        unsupported["support_validation"] = {"supported": False}
        answerer = FakeAnswerer([unsupported, supported_draft("A公司强调AI算力，B公司聚焦存储。")])
        state = run_agentic_rag(
            runtime=runtime,
            answerer=answerer,
            llm=llm,
            config=AgentConfig(),
            query="比较A公司和B公司的策略差异",
        )
        self.assertEqual(state["termination_reason"], "completed")
        self.assertEqual(state["retrieval_count"], 2)
        self.assertEqual(state["rewrite_count"], 1)
        self.assertEqual(state["generation_count"], 2)
        self.assertEqual(len(state["rewritten_queries"]), 1)
        self.assertEqual(state["final_answer"]["final_answer"], "A公司强调AI算力，B公司聚焦存储。")

    def test_verification_retry_abstains_when_refetched_evidence_still_insufficient(self) -> None:
        # Round 1 grades sufficient (two sources) and synthesizes an unsupported
        # draft. The verification retry rewrites + refetches; the fresh evidence
        # is new but single-source, so the post-rewrite grade fails and the run
        # abstains instead of regenerating.
        first_evidence = make_result(
            [
                make_row(chunk_id="c1", doc_id="d1", text="A公司强调AI算力需求。"),
                make_row(chunk_id="c3", doc_id="d2", text="B公司聚焦存储涨价。"),
            ]
        )
        second_evidence = make_result([make_row(chunk_id="c2", doc_id="d1", text="A公司聚焦国产替代。")])
        runtime = FakeRuntime([first_evidence, second_evidence])
        llm = FakeLLM([rewrite_response("A公司 B公司 策略", "source_diversity_missing")])
        unsupported = supported_draft("草稿")
        unsupported["support_validation"] = {"supported": False}
        answerer = FakeAnswerer([unsupported])
        state = run_agentic_rag(
            runtime=runtime,
            answerer=answerer,
            llm=llm,
            config=AgentConfig(),
            query="比较A公司和B公司的策略差异",
        )
        self.assertEqual(state["termination_reason"], "abstain_evidence_insufficient")
        self.assertTrue(state["final_answer"]["abstained"])
        self.assertEqual(state["retrieval_count"], 2)
        self.assertEqual(state["rewrite_count"], 1)
        self.assertEqual(state["generation_count"], 1)

    def test_low_confidence_supported_draft_retries_with_rewrite(self) -> None:
        # A supported-but-low-confidence draft (grade passed round 1, but the
        # answerer is not confident) must trigger the rewrite + re-retrieve path.
        first_evidence = make_result(
            [
                make_row(chunk_id="c1", doc_id="d1", text="A公司强调AI算力需求。"),
                make_row(chunk_id="c3", doc_id="d2", text="B公司聚焦存储涨价。"),
            ]
        )
        second_evidence = make_result(
            [
                make_row(chunk_id="c2", doc_id="d1", text="A公司强调AI算力需求。"),
                make_row(chunk_id="c4", doc_id="d2", text="B公司聚焦存储涨价。"),
            ]
        )
        runtime = FakeRuntime([first_evidence, second_evidence])
        llm = FakeLLM([rewrite_response("A公司 B公司 策略 对比", "low_confidence_answer")])
        low_confidence = supported_draft("第一次草稿")
        low_confidence["confidence_label"] = "low"
        confident = supported_draft("A公司强调AI算力，B公司聚焦存储。")
        confident["confidence_label"] = "high"
        answerer = FakeAnswerer([low_confidence, confident])
        state = run_agentic_rag(
            runtime=runtime,
            answerer=answerer,
            llm=llm,
            config=AgentConfig(),
            query="比较A公司和B公司的策略差异",
        )
        self.assertEqual(state["termination_reason"], "completed")
        self.assertEqual(state["rewrite_count"], 1)
        self.assertEqual(state["retrieval_count"], 2)
        self.assertEqual(state["generation_count"], 2)
        self.assertEqual(state["final_answer"]["final_answer"], "A公司强调AI算力，B公司聚焦存储。")

    def test_high_confidence_supported_draft_finalizes_without_retry(self) -> None:
        runtime = FakeRuntime([make_result([make_row(chunk_id="c1", doc_id="d1", text="半导体行业景气度持续回升。")])])
        llm = FakeLLM([])
        confident = supported_draft("半导体行业景气度持续回升。")
        confident["confidence_label"] = "high"
        answerer = FakeAnswerer([confident])
        state = run_agentic_rag(
            runtime=runtime,
            answerer=answerer,
            llm=llm,
            config=AgentConfig(),
            query="半导体行业景气度如何？",
        )
        self.assertEqual(state["termination_reason"], "completed")
        self.assertEqual(state["rewrite_count"], 0)
        self.assertEqual(state["retrieval_count"], 1)
        self.assertEqual(state["generation_count"], 1)

    def test_rewrite_prompt_carries_current_draft_answer(self) -> None:
        first_evidence = make_result(
            [
                make_row(chunk_id="c1", doc_id="d1", text="A公司强调AI算力需求。"),
                make_row(chunk_id="c3", doc_id="d2", text="B公司聚焦存储涨价。"),
            ]
        )
        second_evidence = make_result(
            [
                make_row(chunk_id="c2", doc_id="d1", text="A公司强调AI算力需求。"),
                make_row(chunk_id="c4", doc_id="d2", text="B公司聚焦存储涨价。"),
            ]
        )
        runtime = FakeRuntime([first_evidence, second_evidence])
        llm = FakeLLM([rewrite_response("A公司 B公司 策略 对比", "low_confidence_answer")])
        low_confidence = supported_draft("第一次草稿")
        low_confidence["confidence_label"] = "low"
        confident = supported_draft("A公司强调AI算力，B公司聚焦存储。")
        confident["confidence_label"] = "high"
        answerer = FakeAnswerer([low_confidence, confident])
        state = run_agentic_rag(
            runtime=runtime,
            answerer=answerer,
            llm=llm,
            config=AgentConfig(),
            query="比较A公司和B公司的策略差异",
        )
        self.assertEqual(state["termination_reason"], "completed")
        # P7: extract_claims now runs before verify->rewrite, so rewrite is not
        # necessarily calls[0]; find it by node kind and check its user prompt.
        rewrite_idx = [i for i, k in enumerate(llm_call_kinds(llm)) if k == "rewrite"][0]
        user_prompt = llm.calls[rewrite_idx]["messages"][1]["content"]
        self.assertIn("Current draft answer: 第一次草稿", user_prompt)

    def test_resynthesis_uses_pooled_evidence_from_all_rounds(self) -> None:
        first_evidence = make_result(
            [
                make_row(chunk_id="c1", doc_id="d1", text="A公司强调AI算力需求。"),
                make_row(chunk_id="c3", doc_id="d2", text="B公司聚焦存储涨价。"),
            ]
        )
        second_evidence = make_result(
            [
                make_row(chunk_id="c2", doc_id="d1", text="A公司强调AI算力需求。"),
                make_row(chunk_id="c4", doc_id="d2", text="B公司聚焦存储涨价。"),
            ]
        )
        runtime = FakeRuntime([first_evidence, second_evidence])
        llm = FakeLLM([rewrite_response("A公司 B公司 策略 对比", "source_diversity_missing")])
        unsupported = supported_draft("第一次草稿")
        unsupported["support_validation"] = {"supported": False}
        answerer = FakeAnswerer([unsupported, supported_draft("A公司强调AI算力，B公司聚焦存储。")])
        state = run_agentic_rag(
            runtime=runtime,
            answerer=answerer,
            llm=llm,
            config=AgentConfig(),
            query="比较A公司和B公司的策略差异",
        )
        self.assertEqual(state["termination_reason"], "completed")
        self.assertEqual(len(answerer.answer_calls), 2)
        # The first synthesis uses only the current round; the resynthesis must
        # receive evidence pooled across both rounds.
        first_rows = {row["chunk_id"] for row in answerer.answer_calls[0]["retrieval_result"]["rerank_rows"]}
        second_rows = {row["chunk_id"] for row in answerer.answer_calls[1]["retrieval_result"]["rerank_rows"]}
        self.assertEqual(first_rows, {"c1", "c3"})
        self.assertIn("c1", second_rows)
        self.assertIn("c2", second_rows)

    def test_abstained_draft_with_evidence_retries_with_rewrite(self) -> None:
        # The answerer abstained although evidence was seen: retry with a rewrite
        # instead of finalizing (bounded by the rewrite budget).
        first_evidence = make_result(
            [
                make_row(chunk_id="c1", doc_id="d1", text="A公司强调AI算力需求。"),
                make_row(chunk_id="c3", doc_id="d2", text="B公司聚焦存储涨价。"),
            ]
        )
        second_evidence = make_result(
            [
                make_row(chunk_id="c2", doc_id="d1", text="A公司强调AI算力需求。"),
                make_row(chunk_id="c4", doc_id="d2", text="B公司聚焦存储涨价。"),
            ]
        )
        runtime = FakeRuntime([first_evidence, second_evidence])
        llm = FakeLLM([rewrite_response("A公司 B公司 策略 对比", "abstained_despite_evidence")])
        abstained = abstained_draft("insufficient_source_diversity")
        confident = supported_draft("A公司强调AI算力，B公司聚焦存储。")
        confident["confidence_label"] = "high"
        answerer = FakeAnswerer([abstained, confident])
        state = run_agentic_rag(
            runtime=runtime,
            answerer=answerer,
            llm=llm,
            config=AgentConfig(),
            query="比较A公司和B公司的策略差异",
        )
        self.assertEqual(state["termination_reason"], "completed")
        self.assertEqual(state["rewrite_count"], 1)
        self.assertEqual(state["retrieval_count"], 2)
        self.assertEqual(state["generation_count"], 2)
        self.assertFalse(state["final_answer"]["abstained"])
        self.assertEqual(state["final_answer"]["final_answer"], "A公司强调AI算力，B公司聚焦存储。")

    def test_abstained_draft_without_evidence_finalizes_without_retry(self) -> None:
        runtime = FakeRuntime([])  # always empty -> evidence pool stays empty
        llm = FakeLLM(
            [
                rewrite_response("半导体 景气", "no_evidence"),
                rewrite_response("半导体 行业 景气 回升", "no_evidence"),
            ]
        )
        answerer = FakeAnswerer([abstained_draft("no_evidence")])
        state = run_agentic_rag(
            runtime=runtime,
            answerer=answerer,
            llm=llm,
            config=AgentConfig(),
            query="半导体行业景气度如何？",
        )
        self.assertTrue(state["final_answer"]["abstained"])
        self.assertEqual(state["termination_reason"], "completed")
        self.assertEqual(state["rewrite_count"], 1)

    def test_analyze_query_makes_no_llm_call(self) -> None:
        runtime = FakeRuntime([make_result([make_row(chunk_id="c1", doc_id="d1", text="半导体行业景气度持续回升。")])])
        llm = FakeLLM([])
        answerer = FakeAnswerer([supported_draft("ok")])
        state = run_agentic_rag(
            runtime=runtime,
            answerer=answerer,
            llm=llm,
            config=AgentConfig(),
            query="半导体行业现状分析",
        )
        self.assertEqual(len([k for k in llm_call_kinds(llm) if k == "rewrite"]), 0)
        self.assertEqual(state["question_type"], "fact")
        self.assertIn(state["answer_mode"], ("fact", "numeric_fact"))

    def test_retrieve_applies_domain_priority(self) -> None:
        liquor_row = make_row(chunk_id="c1", doc_id="d1", text="白酒行业景气度持续回升。")
        semi_row = make_row(chunk_id="c2", doc_id="d2", text="半导体行业景气度持续回升。")
        liquor_row["file_name"] = "白酒_2026-01-01_AP0001_白酒周报.pdf"
        semi_row["file_name"] = "半导体_2026-01-01_AP0001_半导体周报.pdf"
        runtime = FakeRuntime([make_result([liquor_row, semi_row])])
        llm = FakeLLM([])
        answerer = FakeAnswerer([supported_draft("半导体行业景气度持续回升。")])
        state = run_agentic_rag(
            runtime=runtime,
            answerer=answerer,
            llm=llm,
            config=AgentConfig(),
            query="半导体行业景气度如何？",
            domain_hint="semiconductor",
        )
        rows = state["last_retrieval_result"]["rerank_rows"]
        self.assertEqual([row["chunk_id"] for row in rows], ["c2", "c1"])
        self.assertEqual(state["termination_reason"], "completed")

    def test_rewrite_prompt_carries_domain_evidence_and_missing_reason(self) -> None:
        single_source = make_result([make_row(chunk_id="c1", doc_id="d1", text="A公司强调AI算力需求。")])
        two_sources = make_result(
            [
                make_row(chunk_id="c1", doc_id="d1", text="A公司强调AI算力需求。"),
                make_row(chunk_id="c3", doc_id="d2", text="B公司聚焦存储涨价。"),
            ]
        )
        runtime = FakeRuntime([single_source, two_sources])
        llm = FakeLLM([rewrite_response("A公司 B公司 策略 对比", "source_diversity_missing")])
        answerer = FakeAnswerer([supported_draft("A公司强调AI算力，B公司聚焦存储。")])
        state = run_agentic_rag(
            runtime=runtime,
            answerer=answerer,
            llm=llm,
            config=AgentConfig(),
            query="比较A公司和B公司的策略差异",
            domain_hint="semiconductor",
        )
        self.assertEqual(state["termination_reason"], "completed")
        rewrite_call = llm.calls[0]
        user_prompt = rewrite_call["messages"][1]["content"]
        self.assertIn("Domain hint: semiconductor", user_prompt)
        self.assertIn("Question type: comparison", user_prompt)
        self.assertIn("Missing information: insufficient_source_diversity", user_prompt)
        self.assertIn("A公司强调AI算力需求。", user_prompt)


if __name__ == "__main__":
    unittest.main()
