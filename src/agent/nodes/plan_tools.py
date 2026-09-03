"""Plan the next whitelisted deterministic tool action."""

from __future__ import annotations

from typing import Any

from src.agent.config import AgentConfig
from src.agent.policies import begin_node
from src.agent.tool_orchestration import detect_calculation_intent, plan_next_tool


def make_plan_tools(fact_store, company_aliases, config: AgentConfig):
    def plan_tools(state: dict[str, Any]) -> dict[str, Any]:
        if not begin_node(state, config):
            return state
        query = str(state.get("active_query") or state.get("query") or "")
        calculation_intent = state.get("calculation_intent")
        if calculation_intent is None:
            calculation_intent = detect_calculation_intent(
                query,
                company_aliases=company_aliases or {},
            )
            if calculation_intent is not None:
                state["calculation_intent"] = calculation_intent
        state["pending_tool_plan"] = plan_next_tool(
            query=query,
            domain_hint=str(state.get("query_domain_bucket") or state.get("domain_hint") or ""),
            fact_store=fact_store,
            company_aliases=company_aliases or {},
            attempted_call_keys=set(state.get("tool_call_keys") or set()),
            calculation_intent=calculation_intent,
            calculation_facts=state.get("calculation_facts") or {},
        )
        return state

    return plan_tools
