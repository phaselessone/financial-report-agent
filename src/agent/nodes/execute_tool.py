"""Execute one planned financial tool call with hard budgets."""

from __future__ import annotations

from typing import Any

from src.agent.config import AgentConfig
from src.agent.policies import begin_node
from src.agent.tool_orchestration import execute_tool_plan


def make_execute_tool(runtime, fact_store, company_aliases, config: AgentConfig, *, calculation_executor=None):
    def execute_tool(state: dict[str, Any]) -> dict[str, Any]:
        if not begin_node(state, config):
            return state
        if int(state.get("tool_call_count", 0)) >= config.max_tool_calls:
            state["termination_reason"] = "max_tool_calls"
            call_id = f"tool-call-{len(state.setdefault('tool_calls', [])) + 1}"
            plan = state.get("pending_tool_plan") or {}
            budget_record = {
                "tool_call_id": call_id,
                "tool_name": str(plan.get("next_action") or "unknown"),
                "arguments": dict(plan.get("arguments") or {}),
                "reason": "budget_exhausted:max_tool_calls",
                "status": "SKIPPED",
                "result_summary": "budget_exhausted:max_tool_calls",
                "latency_ms": 0.0,
                "error_type": "BUDGET_EXHAUSTED",
            }
            state["tool_calls"].append(budget_record)
            state.setdefault("trajectory_events", []).append(
                {
                    "step": int(state.get("step_count", 0)),
                    "node": "execute_tool",
                    "action": budget_record["tool_name"],
                    "input_summary": budget_record["reason"],
                    "output_summary": budget_record["result_summary"],
                    "status": "SKIPPED",
                    "latency_ms": 0.0,
                    "tokens": 0,
                    "error_type": "BUDGET_EXHAUSTED",
                }
            )
            return state
        plan = state.get("pending_tool_plan") or {"next_action": "finish", "arguments": {}, "reason": "missing_plan"}
        observation = execute_tool_plan(
            plan,
            runtime=runtime,
            fact_store=fact_store,
            company_aliases=company_aliases or {},
            evidence_pool=state.get("evidence_pool") or {},
            calculation_executor=calculation_executor,
        )
        if plan.get("next_action") != "finish":
            state["tool_call_count"] = int(state.get("tool_call_count", 0)) + 1
        state["pending_tool_observation"] = observation
        return state

    return execute_tool
