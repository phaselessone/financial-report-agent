"""Materialize a reasoning-step observation into the shared agent state."""

from __future__ import annotations

from typing import Any

from src.agent.reasoning_plan import LegacySubQuestionAdapter, ReasoningStep, StepResult


def observe_step_result(state: dict[str, Any]) -> dict[str, Any]:
    raw_step = state.get("pending_reasoning_step")
    observation = state.get("pending_step_observation") or {}
    if not raw_step:
        state["termination_reason"] = state.get("termination_reason") or "missing_pending_reasoning_step"
        return state
    step = ReasoningStep.from_mapping(raw_step)
    raw_result = observation.get("result")
    if not raw_result:
        raw_result = {
            "step_id": step.step_id,
            "status": "FAILED",
            "error_type": "MISSING_STEP_OBSERVATION",
        }
    result = StepResult.from_mapping(raw_result)
    artifact = dict(observation.get("artifact") or {})

    existing = [
        item
        for item in (state.get("reasoning_step_results") or [])
        if str((item or {}).get("step_id") or "") != result.step_id
    ]
    existing.append(result.to_dict())
    state["reasoning_step_results"] = existing
    state.setdefault("reasoning_step_artifacts", {})[result.step_id] = artifact

    rows = [row for row in (artifact.get("rows") or []) if isinstance(row, dict)]
    pool = state.setdefault("evidence_pool", {})
    seen = state.setdefault("seen_chunk_ids", set())
    new_ids: list[str] = []
    for row in rows:
        row_id = str(row.get("chunk_id") or row.get("evidence_id") or "").strip()
        if not row_id:
            continue
        normalized = dict(row)
        normalized.setdefault("chunk_id", row_id)
        normalized.setdefault("evidence_id", row_id)
        if row_id not in seen:
            new_ids.append(row_id)
        pool[row_id] = normalized
        seen.add(row_id)
    state["last_new_ids"] = len(new_ids)

    if artifact.get("source") == "report_search":
        state["tool_outcome"] = (
            "tool_execution_failed" if result.status.value == "FAILED" else "report_search"
        )
        state["retrieval_count"] = int(state.get("retrieval_count", 0)) + 1
        retrieval_result = dict(artifact.get("retrieval_result") or {})
        state["last_retrieval_result"] = retrieval_result
        state.setdefault("retrieval_history", []).append(
            {
                "round": state["retrieval_count"],
                "query": artifact.get("query", ""),
                "new_chunk_ids": new_ids,
                "row_count": len(rows),
                "top_scores": [
                    round(float(row.get("score") or row.get("rerank_score") or -999.0), 4)
                    for row in rows[:3]
                ],
                "reasoning_step_id": step.step_id,
            }
        )
    elif artifact.get("source") == "structured":
        state["tool_outcome"] = (
            "structured_hit" if result.status.value == "SUCCESS" else "structured_miss"
        )
        state.setdefault("structured_facts", []).extend(
            fact for fact in (artifact.get("facts") or []) if isinstance(fact, dict)
        )
    elif artifact.get("source") == "evidence_pool":
        state["tool_outcome"] = (
            "evidence_pool_hit" if result.status.value == "SUCCESS" else "evidence_pool_ambiguous"
        )
    elif artifact.get("source") == "calculator":
        state["tool_outcome"] = (
            "calculation_success" if result.status.value == "SUCCESS" else "calculation_failed"
        )
        calculation = artifact.get("calculation")
        if isinstance(calculation, dict) and calculation.get("calculation_id"):
            state.setdefault("calculations", {})[str(calculation["calculation_id"])] = dict(calculation)
        conclusion = artifact.get("conclusion")
        if isinstance(conclusion, dict):
            state["reasoning_conclusion"] = dict(conclusion)

    for raw_call in observation.get("tool_calls") or []:
        if not isinstance(raw_call, dict):
            continue
        call = dict(raw_call)
        call.setdefault("tool_call_id", f"tool-call-{len(state.setdefault('tool_calls', [])) + 1}")
        state["tool_calls"].append(call)
        state["tool_call_count"] = int(state.get("tool_call_count", 0)) + 1

    if step.kind.value in {"LOOKUP", "SEARCH"}:
        legacy = LegacySubQuestionAdapter.to_legacy(
            result,
            required_fields=step.arguments.get("required_fields") or (),
            facts=artifact.get("facts") or (),
        ).to_dict()
        legacy.update(
            {
                "query": str(step.arguments.get("query") or ""),
                "source": artifact.get("source") or step.kind.value.lower(),
            }
        )
        old_results = [
            item
            for item in (state.get("sub_question_results") or [])
            if str((item or {}).get("id") or "") != result.step_id
        ]
        old_results.append(legacy)
        state["sub_question_results"] = old_results

    state.setdefault("trajectory_events", []).append(
        {
            "step": int(state.get("step_count", 0)),
            "node": "observe_step_result",
            "action": step.kind.value,
            "reasoning_step_id": step.step_id,
            "status": result.status.value,
            "latency_ms": result.latency_ms,
            "tokens": 0,
            "error_type": result.error_type,
            "fact_ids": list(result.fact_ids),
            "evidence_ids": list(result.evidence_ids),
            "calculation_id": result.calculation_id,
            "budget_usage": dict(result.budget_usage),
        }
    )
    state["pending_reasoning_step"] = None
    state["pending_step_observation"] = None
    return state


def make_observe_step_result(*_args: Any, **_kwargs: Any):
    return observe_step_result


__all__ = ["make_observe_step_result", "observe_step_result"]
