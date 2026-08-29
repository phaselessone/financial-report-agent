"""Build the stable reasoning plan after deterministic query analysis."""

from __future__ import annotations

from typing import Any

from src.agent.config import AgentConfig
from src.agent.nodes.decompose_query import make_decompose_query
from src.agent.reasoning_plan import build_reasoning_plan


def make_build_reasoning_plan(llm, fact_store, company_aliases, config: AgentConfig):
    decompose = make_decompose_query(llm, config, count_step=False)

    def build_plan(state: dict[str, Any]) -> dict[str, Any]:
        # Planning itself does not consume an execution-budget step.  Only
        # execute_step (or the legacy retrieve/tool adapter) spends max_steps;
        # otherwise inserting the plan seam would reduce existing recovery
        # budgets without doing additional external work.
        if state.get("termination_reason"):
            return state

        is_multi_hop = bool(state.get("is_multi_hop"))
        if is_multi_hop and not state.get("sub_questions"):
            decompose(state)
            if state.get("termination_reason"):
                return state

        # Rewrites remain a one-step recovery plan over the rewritten query.
        # Multi-hop decomposition is fixed for the run and always answers the
        # original question.
        plan_query = (
            str(state.get("query") or "")
            if is_multi_hop
            else str(state.get("active_query") or state.get("query") or "")
        )
        plan = build_reasoning_plan(
            plan_query,
            fact_store_available=fact_store is not None,
            company_aliases=company_aliases or {},
            sub_questions=state.get("sub_questions") if is_multi_hop else None,
        )
        state["reasoning_plan"] = plan.to_dict()
        state["reasoning_step_results"] = []
        state["reasoning_step_artifacts"] = {}
        state["pending_reasoning_step"] = None
        state["pending_step_observation"] = None
        state["reasoning_plan_complete"] = False
        state["reasoning_coverage"] = {}
        state["reasoning_conclusion"] = None
        state["dependency_edges"] = [
            {"source": dependency, "target": step.step_id, "required": step.required}
            for step in plan.steps
            for dependency in step.depends_on
        ]
        state.setdefault("trajectory_events", []).append(
            {
                "step": int(state.get("step_count", 0)),
                "node": "build_reasoning_plan",
                "action": "build_plan",
                "status": "SUCCESS",
                "plan_id": plan.plan_id,
                "planned_steps": len(plan.steps),
                "tokens": 0,
            }
        )
        return state

    return build_plan


__all__ = ["make_build_reasoning_plan"]
