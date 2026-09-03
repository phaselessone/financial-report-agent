"""Runtime wiring for the precision-gated semantic verification layer."""

from __future__ import annotations

import json

import pytest

from src.agent.config import AgentConfig
from src.agent.graph import build_agent_graph, run_agentic_rag
from src.agent.llm_claim_judge import build_claim_llm_judge
from src.agent.semantic_policy import SemanticActivationError, activate_semantic_scorer
from src.llm.types import LLMResponse
from tests.test_agent_flow import (
    FakeAnswerer,
    FakeLLM,
    FakeRuntime,
    make_result,
    make_row,
    supported_draft,
)


def _calibration_report() -> dict[str, object]:
    return {
        "calibrated": True,
        "precision_constraint_satisfied": True,
        "positive_support_constraint_satisfied": True,
        "coverage_constraint_satisfied": True,
        "threshold": 0.9,
        "metrics": {"precision": 1.0, "tp": 10, "fp": 0},
        "calibration_config": {
            "min_precision": 0.95,
            "min_predicted_positives": 5,
            "min_true_positives": 5,
        },
        "label_contract": {
            "review_status": "reviewed",
            "content_hash_binding": "sha256",
        },
        "labels_sha256": "a" * 64,
        "scorer_kind": "directional_nli",
        "scorer_identity": _scorer_identity(),
        "calibrated_statuses": ["ENTAILED"],
        "contradiction_state_change_allowed": False,
    }


def _scorer_identity() -> dict[str, str]:
    return {
        "kind": "directional_nli",
        "model": "fixture-nli",
        "revision": "a" * 40,
        "config_sha256": "b" * 64,
    }


def test_graph_rejects_unactivated_semantic_scorer() -> None:
    with pytest.raises(SemanticActivationError, match="activated"):
        build_agent_graph(
            FakeRuntime([]),
            FakeAnswerer([]),
            FakeLLM([]),
            AgentConfig(),
            semantic_scorer=lambda _claim, _evidence: {"status": "ENTAILED", "score": 1.0},
        )


def test_activated_directional_scorer_is_used_by_runtime_graph() -> None:
    calls: list[tuple[str, str]] = []

    def directional(claim: str, evidence: str):
        calls.append((claim, evidence))
        return {"status": "ENTAILED", "score": 0.96}

    scorer = activate_semantic_scorer(
        directional,
        _calibration_report(),
        scorer_identity=_scorer_identity(),
    )
    extractor = LLMResponse(
        content=json.dumps(
            {
                "claims": [
                    {
                        "id": "c1",
                        "text": "公司加大研发投入。",
                        "claim_type": "EXTRACTED",
                        "is_core": True,
                        "source_step_ids": ["search_1"],
                        "parent_claim_ids": [],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        provider="fake",
        model="fake",
    )
    state = run_agentic_rag(
        runtime=FakeRuntime(
            [
                make_result(
                    [
                        make_row(
                            chunk_id="E1",
                            doc_id="D1",
                            text="公司持续增加研发资源。",
                        )
                    ]
                )
            ]
        ),
        answerer=FakeAnswerer([supported_draft("公司加大研发投入。")]),
        llm=FakeLLM([extractor]),
        config=AgentConfig(strict_claim_verification=True),
        query="公司的研发投入如何？",
        semantic_scorer=scorer,
    )

    assert calls
    assert state["claims"][0]["verification"]["method"] == "semantic"
    assert state["claims"][0]["verification"]["status"] == "ENTAILED"
    assert state["final_answer"]["abstained"] is False


def test_high_similarity_without_directional_status_abstains_in_runtime_graph() -> None:
    scorer = activate_semantic_scorer(
        lambda _claim, _evidence: 0.99,
        _calibration_report(),
        scorer_identity=_scorer_identity(),
    )
    extractor = LLMResponse(
        content=json.dumps(
            {
                "claims": [
                    {
                        "id": "c1",
                        "text": "人工智能推动增长。",
                        "claim_type": "EXTRACTED",
                        "is_core": True,
                        "source_step_ids": ["search_answer"],
                        "parent_claim_ids": [],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        provider="fake",
        model="fake",
    )

    state = run_agentic_rag(
        runtime=FakeRuntime(
            [
                make_result(
                    [
                        make_row(
                            chunk_id="E1",
                            doc_id="D1",
                            text="AI成为业绩增长动力。",
                        )
                    ]
                )
            ]
        ),
        answerer=FakeAnswerer([supported_draft("人工智能推动增长。")]),
        llm=FakeLLM([extractor]),
        config=AgentConfig(
            strict_claim_verification=True,
            max_claim_retrievals=0,
            claim_llm_budget=0,
        ),
        query="人工智能如何影响增长？",
        semantic_scorer=scorer,
    )

    claim = state["claims"][0]
    assert claim["verification"]["status"] == "INSUFFICIENT"
    assert claim["verification"]["method"] == "semantic"
    assert "semantic_similarity_not_directional" in claim["verification"]["reasons"]
    assert state["final_answer"]["abstained"] is True
    assert state["final_answer"]["used_evidence_ids"] == []
    assert state["final_answer"]["citations"] == []


def test_budgeted_llm_judge_is_wired_only_after_deterministic_is_undecidable() -> None:
    extractor = LLMResponse(
        content=json.dumps(
            {
                "claims": [
                    {
                        "id": "c1",
                        "text": "公司加大研发投入。",
                        "claim_type": "EXTRACTED",
                        "is_core": True,
                        "source_step_ids": ["search_1"],
                        "parent_claim_ids": [],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        provider="fake",
        model="fake",
    )
    judge_calls: list[dict[str, object]] = []

    def judge(payload):
        judge_calls.append(dict(payload))
        return {
            "status": "ENTAILED",
            "evidence_ids": ["E1"],
            "score": 0.95,
            "reasons": ["directional qualitative judgment"],
        }

    state = run_agentic_rag(
        runtime=FakeRuntime(
            [
                make_result(
                    [
                        make_row(
                            chunk_id="E1",
                            doc_id="D1",
                            text="公司持续增加研发资源。",
                        )
                    ]
                )
            ]
        ),
        answerer=FakeAnswerer([supported_draft("公司加大研发投入。")]),
        llm=FakeLLM([extractor]),
        config=AgentConfig(strict_claim_verification=True, claim_llm_budget=1),
        query="公司的研发投入如何？",
        llm_judge=judge,
    )

    assert len(judge_calls) == 1
    assert state["claim_llm_judge_count"] == 1
    assert state["claims"][0]["verification"]["method"] == "llm"
    assert state["final_answer"]["abstained"] is False


def test_production_claim_judge_usage_is_recorded_in_counters_log_and_trajectory() -> None:
    extractor = LLMResponse(
        content=json.dumps(
            {
                "claims": [
                    {
                        "id": "c1",
                        "text": "公司加大研发投入。",
                        "claim_type": "EXTRACTED",
                        "is_core": True,
                        "source_step_ids": ["search_1"],
                        "parent_claim_ids": [],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        provider="fake",
        model="fake",
        prompt_tokens=4,
        completion_tokens=2,
    )
    judge_response = LLMResponse(
        content=json.dumps(
            {
                "status": "ENTAILED",
                "evidence_ids": ["E1"],
                "score": 0.95,
                "reasons": ["directional qualitative judgment"],
            }
        ),
        provider="fake",
        model="fake",
        prompt_tokens=17,
        completion_tokens=9,
        total_tokens=26,
        request_id="req-judge-1",
    )
    llm = FakeLLM([extractor, judge_response])
    judge = build_claim_llm_judge(llm, budget=1, max_tokens=128)

    state = run_agentic_rag(
        runtime=FakeRuntime(
            [make_result([make_row(chunk_id="E1", doc_id="D1", text="公司持续增加研发资源。")])]
        ),
        answerer=FakeAnswerer([supported_draft("公司加大研发投入。")]),
        llm=llm,
        config=AgentConfig(strict_claim_verification=True, claim_llm_budget=1),
        query="公司的研发投入如何？",
        llm_judge=judge,
    )

    assert state["claim_llm_judge_count"] == 1
    assert state["llm_call_count"] == 3
    assert state["prompt_tokens"] == 41
    assert state["completion_tokens"] == 21
    assert state["total_tokens"] == 62
    judge_logs = [entry for entry in state["llm_calls_log"] if entry["node"] == "verify_answer"]
    assert len(judge_logs) == 1
    assert judge_logs[0]["request_id"] == "req-judge-1"
    assert judge_logs[0]["total_tokens"] == 26
    verify_events = [
        event
        for event in state["trajectory_events"]
        if event.get("event_type") == "node" and event.get("node") == "verify_answer"
    ]
    assert len(verify_events) == 1
    assert verify_events[0]["budget_usage"]["llm_calls"] == 1
    assert verify_events[0]["budget_usage"]["tokens"] == 26


def test_failed_production_claim_judge_is_fail_closed_and_counted_as_zero_token_call() -> None:
    extractor = LLMResponse(
        content=json.dumps(
            {
                "claims": [
                    {
                        "id": "c1",
                        "text": "公司加大研发投入。",
                        "claim_type": "EXTRACTED",
                        "is_core": True,
                        "source_step_ids": ["search_1"],
                        "parent_claim_ids": [],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        provider="fake",
        model="fake",
        prompt_tokens=4,
        completion_tokens=2,
    )

    class _FailsOnJudge:
        provider_name = "fake-provider"

        def __init__(self) -> None:
            self.calls = 0

        def generate(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return extractor
            raise RuntimeError("provider unavailable")

    llm = _FailsOnJudge()
    judge = build_claim_llm_judge(llm, budget=1)
    state = run_agentic_rag(
        runtime=FakeRuntime(
            [make_result([make_row(chunk_id="E1", doc_id="D1", text="公司持续增加研发资源。")])]
        ),
        answerer=FakeAnswerer([supported_draft("公司加大研发投入。")]),
        llm=llm,
        config=AgentConfig(
            strict_claim_verification=True,
            claim_llm_budget=1,
            max_claim_retrievals=0,
        ),
        query="公司的研发投入如何？",
        llm_judge=judge,
    )

    assert state["claims"][0]["verification"]["status"] == "INSUFFICIENT"
    assert state["claims"][0]["verification"]["reasons"] == ["llm_judge_error"]
    assert state["claim_llm_judge_count"] == 1
    assert state["llm_call_count"] == 3
    failed_logs = [entry for entry in state["llm_calls_log"] if entry["node"] == "verify_answer"]
    assert len(failed_logs) == 1
    assert failed_logs[0]["status"] == "FAILED"
    assert failed_logs[0]["error_type"] == "RuntimeError"
    assert failed_logs[0]["total_tokens"] == 0
    verify_event = next(
        event
        for event in state["trajectory_events"]
        if event.get("event_type") == "node" and event.get("node") == "verify_answer"
    )
    assert verify_event["budget_usage"]["llm_calls"] == 1
    assert verify_event["budget_usage"]["tokens"] == 0


def test_claim_retrieval_retry_cannot_reuse_one_judge_budget_or_drop_usage() -> None:
    extractor = LLMResponse(
        content=json.dumps(
            {
                "claims": [
                    {
                        "id": "c1",
                        "text": "公司加大研发投入。",
                        "claim_type": "EXTRACTED",
                        "is_core": True,
                        "source_step_ids": ["search_1"],
                        "parent_claim_ids": [],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        provider="fake",
        model="fake",
    )
    insufficient = LLMResponse(
        content=json.dumps(
            {
                "status": "INSUFFICIENT",
                "evidence_ids": [],
                "score": 0.2,
                "reasons": ["evidence remains ambiguous"],
            }
        ),
        provider="fake",
        model="fake",
        prompt_tokens=8,
        completion_tokens=4,
    )
    llm = FakeLLM([extractor, insufficient])
    judge = build_claim_llm_judge(llm, budget=1)
    state = run_agentic_rag(
        runtime=FakeRuntime(
            [
                make_result([make_row(chunk_id="E1", doc_id="D1", text="公司持续增加研发资源。")]),
                make_result([make_row(chunk_id="E2", doc_id="D2", text="研发团队规模有所扩大。")]),
            ]
        ),
        answerer=FakeAnswerer([supported_draft("公司加大研发投入。")]),
        llm=llm,
        config=AgentConfig(
            strict_claim_verification=True,
            claim_llm_budget=1,
            max_claim_retrievals=1,
        ),
        query="公司的研发投入如何？",
        llm_judge=judge,
    )

    provider_judge_calls = [
        call
        for call in llm.calls
        if (call.get("metadata") or {}).get("purpose") == "claim_verification"
    ]
    assert state["claim_retrieval_count"] == 1
    assert state["claim_llm_judge_count"] == 1
    assert len(provider_judge_calls) == 1
    assert len([entry for entry in state["llm_calls_log"] if entry["node"] == "verify_answer"]) == 1
    assert state["claims"][0]["verification"]["method"] == "budget"
    assert state["claims"][0]["verification"]["reasons"] == ["llm_budget_exhausted"]
