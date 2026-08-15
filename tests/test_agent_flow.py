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
        self.assertEqual(len(llm.calls), 0)
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
            config=AgentConfig(),
            query="半导体行业景气度如何？",
        )
        self.assertEqual(state["generation_count"], 2)
        self.assertEqual(state["termination_reason"], "completed")
        self.assertEqual(state["final_answer"]["final_answer"], "第二次草稿（修复后）")

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
        self.assertEqual(len(llm.calls), 0)
        self.assertEqual(state["question_type"], "fact")
        self.assertIn(state["answer_mode"], ("fact", "numeric_fact"))


if __name__ == "__main__":
    unittest.main()
