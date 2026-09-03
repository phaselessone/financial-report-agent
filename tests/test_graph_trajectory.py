"""Node-level trajectory contract for the unified agent graph."""

from __future__ import annotations

import json

import pytest

from src.agent.config import AgentConfig
from src.agent.graph import run_agentic_rag
from src.agent.trajectory import canonicalize_trajectory_events, trace_graph_node
from tests.test_agent_flow import (
    FakeAnswerer,
    FakeLLM,
    FakeRuntime,
    make_result,
    make_row,
    supported_draft,
)


def test_every_executed_graph_node_emits_one_uniform_node_event() -> None:
    state = run_agentic_rag(
        runtime=FakeRuntime(
            [
                make_result(
                    [
                        make_row(
                            chunk_id="E1",
                            doc_id="D1",
                            text="宁德时代2025年营业收入为100亿元。",
                        )
                    ]
                )
            ]
        ),
        answerer=FakeAnswerer([supported_draft("宁德时代2025年营业收入为100亿元。")]),
        llm=FakeLLM([]),
        config=AgentConfig(strict_claim_verification=False),
        query="宁德时代2025年营业收入是多少？",
    )

    events = [
        event
        for event in state["trajectory_events"]
        if event.get("event_type") == "node"
    ]
    names = [event["node"] for event in events]
    assert names == [
        "analyze_query",
        "build_reasoning_plan",
        "plan_next_step",
        "execute_step",
        "observe_step_result",
        "plan_next_step",
        "dependency_gate",
        "grade_evidence",
        "synthesize",
        "extract_claims",
        "verify_answer",
        "finalize",
    ]
    required_fields = {
        "step",
        "node",
        "action",
        "input_summary",
        "output_summary",
        "status",
        "latency_ms",
        "tokens",
        "error_type",
    }
    assert all(required_fields <= set(event) for event in events)
    assert all(event["status"] == "SUCCESS" for event in events)
    assert all(event["latency_ms"] >= 0 for event in events)
    assert [event["step"] for event in events] == list(range(1, len(events) + 1))
    assert all(isinstance(event["input_summary"], str) for event in events)
    assert all(isinstance(event["output_summary"], str) for event in events)
    assert all(event["tokens"] == event["budget_usage"]["tokens"] for event in events)
    assert len({event["event_id"] for event in events}) == len(events)
    assert events[0]["dependencies"] == []
    assert all(event["dependencies"] == [events[index - 1]["event_id"]] for index, event in enumerate(events) if index)
    assert all(event["recovery_of"] == [] for event in events)
    assert all(
        set(event["budget_usage"])
        == {"steps", "llm_calls", "tool_calls", "tokens", "retrievals"}
        for event in events
    )


def test_node_wrapper_records_exception_before_reraising() -> None:
    state = {"trajectory_events": [], "tool_call_count": 2}

    def broken(current):
        current["tool_call_count"] += 1
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        trace_graph_node("broken", broken)(state)

    event = state["trajectory_events"][-1]
    assert event["event_type"] == "node"
    assert event["node"] == "broken"
    assert event["status"] == "FAILED"
    assert event["error_type"] == "RuntimeError"
    assert event["budget_usage"]["tool_calls"] == 1
    assert event["tokens"] == 0
    assert json.loads(event["input_summary"])["counters"]["tool_calls"] == 2
    assert json.loads(event["output_summary"])["counters"]["tool_calls"] == 3
    assert event["event_id"]
    assert event["dependencies"] == []
    assert event["recovery_of"] == []


def test_node_exception_exposes_partial_state_snapshot_on_the_same_error() -> None:
    state = {"trajectory_events": [], "total_tokens": 7}

    def broken(current):
        current["total_tokens"] = 11
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom") as caught:
        trace_graph_node("broken", broken)(state)

    partial = caught.value.agent_partial_state
    assert partial["total_tokens"] == 11
    assert partial["trajectory_events"][-1]["status"] == "FAILED"
    assert partial["trajectory_events"][-1]["budget_usage"]["tokens"] == 4
    assert partial["trajectory_events"][-1]["tokens"] == 4


def test_node_summaries_record_shape_without_copying_state_text() -> None:
    state = {
        "trajectory_events": [],
        "query": "private query text",
        "claims": [{"text": "private claim text"}],
        "termination_reason": "private termination text",
    }

    result = trace_graph_node("shape", lambda current: current)(state)
    event = result["trajectory_events"][-1]

    assert "private query text" not in event["input_summary"]
    assert "private claim text" not in event["output_summary"]
    assert "private termination text" not in event["output_summary"]
    assert json.loads(event["input_summary"])["collection_sizes"]["claims"] == 1


@pytest.mark.parametrize("alias", ["trajectory_events", "trace_events", "trajectory", "agent_steps", "events"])
def test_canonical_trace_aliases_share_event_identity_and_lineage(alias: str) -> None:
    source = {
        alias: [
            {"id": "e1", "node": "retrieve", "status": "FAILED", "depends_on": []},
            {"id": "e2", "node": "retrieve", "status": "SUCCESS", "depends_on": ["e1"], "recovers": "e1"},
        ]
    }

    events = canonicalize_trajectory_events(source)

    assert events[0]["event_id"] == "e1"
    assert events[0]["dependencies"] == []
    assert events[0]["lineage_status"] == "resolved"
    assert events[1]["dependencies"] == ["e1"]
    assert events[1]["recovery_of"] == ["e1"]


def test_declared_lineage_cannot_make_an_identity_free_event_resolved() -> None:
    events = canonicalize_trajectory_events(
        [{"node": "retrieve", "status": "FAILED", "lineage_status": "resolved"}]
    )

    assert events[0]["event_id"] == "event-0001"
    assert events[0]["lineage_status"] == "unresolved"
