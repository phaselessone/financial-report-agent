"""Actual baseline/agent graph executors behind the four-profile matrix."""

from __future__ import annotations

import pytest

from src.agent.config import AgentConfig
from src.evaluation.hard_case_benchmark import PROFILE_TREATMENTS, load_cases
from src.evaluation.offline_contract_runtime import (
    OfflineContractAnswerer,
    OfflineContractLLM,
)
from src.evaluation.profile_runtime import PROFILE_SPECS, build_profile_executor
from tests.test_agent_flow import FakeAnswerer, FakeLLM, supported_draft


def test_profile_executor_runs_baseline_and_agent_graph_on_same_case_evidence() -> None:
    case = load_cases()[0]
    answerer = FakeAnswerer(
        [supported_draft(case["gold_answer"]), supported_draft(case["gold_answer"])]
    )
    executor = build_profile_executor(
        answerer=answerer,
        llm=FakeLLM([]),
        base_config=AgentConfig(strict_claim_verification=True),
        allow_contract_oracle=True,
    )

    baseline = executor("baseline-rag", PROFILE_TREATMENTS["baseline-rag"], case)
    agentic = executor("agentic-rag", PROFILE_TREATMENTS["agentic-rag"], case)

    assert baseline["applied_treatments"] == PROFILE_TREATMENTS["baseline-rag"]
    assert baseline["trace_events"][0]["node"] == "baseline_rag"
    assert baseline["tool_calls"][0]["tool_name"] == "report_search"
    assert agentic["applied_treatments"] == PROFILE_TREATMENTS["agentic-rag"]
    assert any(event.get("event_type") == "node" for event in agentic["trace_events"])
    assert agentic["tool_calls"]


class RecordingFactStore:
    def __init__(self) -> None:
        self.queries: list[dict] = []

    def query(self, **coordinates):
        self.queries.append(dict(coordinates))
        return []

    def query_many(self, **coordinates):
        self.queries.append(dict(coordinates))
        return []


def test_profile_specs_change_the_observable_execution_path() -> None:
    """The matrix is an execution ablation, not four labels on one prediction."""

    case = {
        "case_id": "PROFILE-001",
        "category": "simple_factual",
        "question": "贵州茅台2025年营业收入是多少？",
        "question_type": "fact",
        "synthetic": True,
        "evidence": [
            {
                "evidence_id": "E-2025",
                "text": "贵州茅台2025年营业收入为100亿元。",
                "source": "annual-report-2025",
            }
        ],
    }
    store = RecordingFactStore()
    answerer = FakeAnswerer(
        [supported_draft("贵州茅台2025年营业收入为100亿元。") for _ in range(4)]
    )
    executor = build_profile_executor(
        answerer=answerer,
        llm=FakeLLM([]),
        base_config=AgentConfig(),
        fact_store=store,
        company_aliases={"贵州茅台": ["贵州茅台", "茅台"]},
        allow_contract_oracle=True,
    )

    baseline = executor(PROFILE_SPECS["baseline-rag"], case)
    assert [event["node"] for event in baseline["trace_events"]] == ["baseline_rag"]
    assert store.queries == []

    agentic = executor(PROFILE_SPECS["agentic-rag"], case)
    assert store.queries == []
    assert "analyze_query" in {event.get("node") for event in agentic["trace_events"]}
    assert "report_search" in {call.get("tool_name") for call in agentic["tool_calls"]}

    structured = executor(PROFILE_SPECS["structured-agent"], case)
    structured_query_count = len(store.queries)
    assert structured_query_count > 0
    assert "report_search" not in {call.get("tool_name") for call in structured["tool_calls"]}
    structured_nodes = {event.get("node") for event in structured["trace_events"]}
    assert {"extract_claims", "verify_answer"}.issubset(structured_nodes)
    assert structured["effective_treatments"]["claim_verification"] is False
    assert structured["effective_treatments"]["strict_claim_provenance"] is True

    full = executor(PROFILE_SPECS["full-agent"], case)
    assert len(store.queries) > structured_query_count
    assert "report_search" in {call.get("tool_name") for call in full["tool_calls"]}
    full_nodes = {event.get("node") for event in full["trace_events"]}
    assert {"extract_claims", "verify_answer"}.issubset(full_nodes)

    for profile, prediction in (
        ("baseline-rag", baseline),
        ("agentic-rag", agentic),
        ("structured-agent", structured),
        ("full-agent", full),
    ):
        expected = PROFILE_SPECS[profile].effective_treatments
        assert prediction["effective_treatments"] == expected
        assert prediction["trace_events"]
        assert all(
            event.get("effective_treatments") == expected
            for event in prediction["trace_events"]
        )


def test_profile_executor_requires_an_explicit_shared_runtime_or_contract_oracle() -> None:
    with pytest.raises(ValueError, match="retrieval runtime"):
        build_profile_executor(
            answerer=FakeAnswerer([]),
            llm=FakeLLM([]),
            base_config=AgentConfig(),
        )


def test_contract_oracle_rejects_reviewed_cases() -> None:
    executor = build_profile_executor(
        answerer=FakeAnswerer([supported_draft("answer")]),
        llm=FakeLLM([]),
        base_config=AgentConfig(),
        allow_contract_oracle=True,
    )
    reviewed_case = {
        "case_id": "REVIEWED-001",
        "question": "What does the report say?",
        "synthetic": False,
        "evidence": [
            {"evidence_id": "gold-only", "text": "gold", "source": "gold-report"}
        ],
    }

    with pytest.raises(ValueError, match="synthetic contract"):
        executor(PROFILE_SPECS["baseline-rag"], reviewed_case)


def test_graph_profile_falls_back_to_category_for_must_abstain_contract() -> None:
    executor = build_profile_executor(
        answerer=OfflineContractAnswerer(),
        llm=OfflineContractLLM(),
        base_config=AgentConfig(),
        allow_contract_oracle=True,
    )
    case = {
        "case_id": "HC-MUST-ABSTAIN-FALLBACK",
        "category": "must_abstain",
        "question": "Can this unsupported conclusion be stated?",
        "synthetic": True,
        "evidence": [
            {
                "evidence_id": "E-UNSUPPORTED",
                "source": "synthetic-report",
                "text": "The available evidence does not establish that conclusion.",
            }
        ],
    }

    prediction = executor(PROFILE_SPECS["agentic-rag"], case)

    assert prediction["abstained"] is True
    assert prediction["claims"] == []
    assert prediction["cited_evidence_ids"] == []


@pytest.mark.parametrize("profile_id", ["agentic-rag", "full-agent"])
def test_graph_must_abstain_stops_before_duplicate_report_search(
    profile_id: str,
) -> None:
    """A failed verification retry must not execute the same search again."""

    case = next(case for case in load_cases() if case["category"] == "must_abstain")
    executor = build_profile_executor(
        answerer=OfflineContractAnswerer(),
        llm=OfflineContractLLM(),
        base_config=AgentConfig(),
        fact_store=RecordingFactStore(),
        allow_contract_oracle=True,
    )

    prediction = executor(PROFILE_SPECS[profile_id], case)

    report_searches = [
        call
        for call in prediction["tool_calls"]
        if call.get("tool_name") == "report_search"
    ]
    assert prediction["abstained"] is True
    assert prediction["termination_reason"] == "duplicate_tool_call"
    assert len(report_searches) == 1
    assert report_searches[0]["arguments"] == {
        "query": case["question"],
        "domain_hint": "",
    }


class RecordingRetrievalRuntime:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def search(self, query: str) -> dict:
        self.calls.append(query)
        row = {
            "chunk_id": "shared-corpus-1",
            "evidence_id": "shared-corpus-1",
            "doc_id": "shared-report",
            "page_start": 1,
            "text": "Shared corpus evidence.",
        }
        return {
            "query_mode": "shared_corpus",
            "numeric_query": False,
            "dense_rows": [row],
            "bm25_rows": [row],
            "hybrid_rows": [row],
            "rerank_rows": [row],
            "timings": {},
        }


def test_reviewed_profile_uses_shared_runtime_not_gold_evidence() -> None:
    runtime = RecordingRetrievalRuntime()
    answerer = FakeAnswerer([supported_draft("answer")])
    executor = build_profile_executor(
        answerer=answerer,
        llm=FakeLLM([]),
        base_config=AgentConfig(),
        retrieval_runtime=runtime,
    )
    reviewed_case = {
        "case_id": "REVIEWED-001",
        "question": "What does the report say?",
        "synthetic": False,
        "evidence": [
            {"evidence_id": "gold-only", "text": "gold", "source": "gold-report"}
        ],
    }

    executor(PROFILE_SPECS["baseline-rag"], reviewed_case)

    assert runtime.calls == [reviewed_case["question"]]
    retrieved = answerer.answer_calls[0]["retrieval_result"]["rerank_rows"]
    assert [row["chunk_id"] for row in retrieved] == ["shared-corpus-1"]
    assert all(row["chunk_id"] != "gold-only" for row in retrieved)
