"""Record a tool result and merge new evidence into agent state."""

from __future__ import annotations

from typing import Any

from src.agent.config import AgentConfig
from src.agent.policies import begin_node, new_evidence_ids


def make_observe_tool_result(config: AgentConfig):
    def observe_tool_result(state: dict[str, Any]) -> dict[str, Any]:
        if not begin_node(state, config):
            return state
        plan = state.get("pending_tool_plan") or {}
        observation = state.get("pending_tool_observation") or {}
        action = str(plan.get("next_action") or "finish")
        status = str(observation.get("status") or "FAILED")
        call_key = str(plan.get("call_key") or "")
        if call_key:
            state.setdefault("tool_call_keys", set()).add(call_key)
        call_id = f"tool-call-{len(state.setdefault('tool_calls', [])) + 1}"
        record = {
            "tool_call_id": call_id,
            "tool_name": action,
            "arguments": dict(plan.get("arguments") or {}),
            "reason": str(plan.get("reason") or ""),
            "status": status,
            "result_summary": str(observation.get("result_summary") or ""),
            "latency_ms": float(observation.get("latency_ms") or 0.0),
            "error_type": observation.get("error_type"),
        }
        payload = observation.get("payload") or {}
        calculation = payload.get("calculation") if isinstance(payload, dict) else None
        if isinstance(calculation, dict):
            calculation_id = str(calculation.get("calculation_id") or "").strip()
            if calculation_id:
                state.setdefault("calculations", {})[calculation_id] = dict(calculation)
                record["calculation_id"] = calculation_id
        state["tool_calls"].append(record)
        trajectory = {
            "step": int(state.get("step_count", 0)),
            "node": "observe_tool_result",
            "action": action,
            "input_summary": str(plan.get("reason") or "")[:200],
            "output_summary": record["result_summary"],
            "status": status,
            "latency_ms": record["latency_ms"],
            "tokens": 0,
            "error_type": record["error_type"],
        }
        if isinstance(calculation, dict):
            trajectory["calculation"] = dict(calculation)
            trajectory["calculation_id"] = calculation.get("calculation_id")
        state.setdefault("trajectory_events", []).append(trajectory)

        if status == "FAILED":
            state["tool_outcome"] = "failed"
            state["termination_reason"] = "tool_execution_failed"
            return state
        if action == "finish":
            state["tool_outcome"] = "finish"
            if not state.get("evidence_pool"):
                state["termination_reason"] = "no_tool_progress"
            return state

        if isinstance(calculation, dict):
            calculation_status = str(calculation.get("status") or "UNKNOWN").upper()
            if calculation_status == "SUCCESS":
                state["tool_outcome"] = "calculation_success"
            else:
                state["tool_outcome"] = "calculation_failed"
                state["termination_reason"] = "calculation_failed"
            return state

        rows = [row for row in (payload.get("rows") or []) if isinstance(row, dict)]
        seen = state.setdefault("seen_chunk_ids", set())
        pool = state.setdefault("evidence_pool", {})
        new_ids = new_evidence_ids(seen_chunk_ids=seen, rows=rows)
        for row in rows:
            chunk_id = row.get("chunk_id")
            if chunk_id:
                pool[chunk_id] = row
                seen.add(chunk_id)
        state["last_new_ids"] = len(new_ids)

        if action == "structured_lookup":
            facts = list(payload.get("facts") or [])
            state.setdefault("structured_facts", []).extend(facts)
            # Keep structured rows in the same evidence pool as RAG rows even
            # when this lookup is one operand of a calculation.  Claim-level
            # verification and citation binding must see both operands.
            for row in rows:
                chunk_id = row.get("chunk_id")
                if chunk_id:
                    pool[chunk_id] = row
                    seen.add(chunk_id)
            calculation_operand = str((plan.get("arguments") or {}).get("calculation_operand") or "").strip()
            if calculation_operand and facts:
                state.setdefault("calculation_facts", {})[calculation_operand] = facts[0]
                # A calculation is a two-stage lookup.  Do not mistake the
                # first operand for a complete answer: return to the planner
                # until every requested operand has provenance.
                intent = state.get("calculation_intent") or {}
                required_names = {
                    str(item.get("name") or "")
                    for item in (intent.get("operands") or [])
                    if isinstance(item, dict) and item.get("name")
                }
                have_names = set((state.get("calculation_facts") or {}).keys())
                if required_names and required_names.issubset(have_names):
                    state["tool_outcome"] = "calculation_facts_ready"
                    return state
                state["tool_outcome"] = "calculation_operand_hit"
                return state
            if rows:
                result = {
                    "query_mode": "structured_tool",
                    "numeric_query": True,
                    "dense_rows": [],
                    "bm25_rows": [],
                    "hybrid_rows": rows,
                    "rerank_rows": rows,
                    "timings": {},
                }
                state["last_retrieval_result"] = result
                state["evidence_sufficient"] = True
                state["tool_outcome"] = "structured_hit"
            else:
                state["tool_outcome"] = "structured_miss"
            return state

        if action == "report_search":
            # A report fallback may still satisfy one operand of a pending
            # calculation.  Promote only the explicitly extracted, provenance
            # carrying candidate; ordinary report search remains on the RAG
            # grading path.
            calculation_facts = [
                item for item in (payload.get("calculation_facts") or []) if isinstance(item, dict)
            ]
            if calculation_facts:
                operand_name = str((plan.get("arguments") or {}).get("calculation_operand") or "").strip()
                if operand_name:
                    state.setdefault("calculation_facts", {})[operand_name] = calculation_facts[0]
                    intent = state.get("calculation_intent") or {}
                    required_names = {
                        str(item.get("name") or "")
                        for item in (intent.get("operands") or [])
                        if isinstance(item, dict) and item.get("name")
                    }
                    if required_names and required_names.issubset(set((state.get("calculation_facts") or {}).keys())):
                        state["tool_outcome"] = "calculation_facts_ready"
                        return state
            result = payload.get("result") or {"rerank_rows": rows, "hybrid_rows": rows}
            state["last_retrieval_result"] = result
            state["retrieval_count"] = int(state.get("retrieval_count", 0)) + 1
            state.setdefault("retrieval_history", []).append(
                {
                    "round": state["retrieval_count"],
                    "query": (plan.get("arguments") or {}).get("query", state.get("active_query", "")),
                    "new_chunk_ids": new_ids,
                    "row_count": len(rows),
                    "top_scores": [round(float(row.get("score") or row.get("rerank_score") or -999.0), 4) for row in rows[:3]],
                    "tool_call_id": call_id,
                }
            )
            if not new_ids:
                state["consecutive_empty_tool_results"] = int(state.get("consecutive_empty_tool_results", 0)) + 1
            else:
                state["consecutive_empty_tool_results"] = 0
            if int(state.get("consecutive_empty_tool_results", 0)) >= config.max_empty_tool_results:
                state["no_improvement"] = True
            state["tool_outcome"] = "report_search"
            return state

        state["tool_outcome"] = action
        return state

    return observe_tool_result
