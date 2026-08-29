"""Select the next dependency-ready reasoning step."""

from __future__ import annotations

from typing import Any

from src.agent.reasoning_plan import ReasoningPlan, StepResult


def plan_next_step(state: dict[str, Any]) -> dict[str, Any]:
    plan_value = state.get("reasoning_plan")
    if not plan_value:
        state["termination_reason"] = state.get("termination_reason") or "missing_reasoning_plan"
        return state
    plan = ReasoningPlan.from_mapping(plan_value)
    results = [StepResult.from_mapping(item) for item in (state.get("reasoning_step_results") or [])]
    result_ids = {result.step_id for result in results}

    for step in plan.steps:
        if step.step_id in result_ids:
            continue
        # A step becomes executable when every dependency has a terminal
        # result. execute_step decides whether to run or emit SKIPPED.
        if all(dependency in result_ids for dependency in step.depends_on):
            state["pending_reasoning_step"] = step.to_dict()
            state["reasoning_plan_complete"] = False
            return state

    unfinished = [step.step_id for step in plan.steps if step.step_id not in result_ids]
    if unfinished:
        state["termination_reason"] = state.get("termination_reason") or "reasoning_plan_deadlock"
    state["pending_reasoning_step"] = None
    state["reasoning_plan_complete"] = not unfinished
    return state


def make_plan_next_step(*_args: Any, **_kwargs: Any):
    return plan_next_step


__all__ = ["make_plan_next_step", "plan_next_step"]
