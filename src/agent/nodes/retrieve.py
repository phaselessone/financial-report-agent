"""retrieve node: one RetrievalRuntime.search round, pool merge, dedup bookkeeping."""

from __future__ import annotations

from typing import Any

from src.agent.config import AgentConfig
from src.agent.policies import begin_node, new_evidence_ids
from src.retrieval.domain_priority import apply_retrieval_domain_priority


def make_retrieve(runtime, config: AgentConfig):
    def retrieve(state: dict[str, Any]) -> dict[str, Any]:
        if not begin_node(state, config):
            return state
        if int(state.get("retrieval_count", 0)) >= config.max_retrieval_rounds:
            state["termination_reason"] = "max_retrieval_rounds"
            return state

        result = runtime.search(state["active_query"])
        # Align with the baseline eval loop: prefer evidence matching the query
        # domain before grading (checklist §P3 gate remediation R2).
        domain_hint = state.get("query_domain_bucket", "") or state.get("domain_hint", "")
        result = apply_retrieval_domain_priority(result, domain_hint)
        state["retrieval_count"] = int(state.get("retrieval_count", 0)) + 1

        rows = list(result.get("rerank_rows") or result.get("hybrid_rows") or [])
        rows = [row for row in rows if isinstance(row, dict)]
        seen = state.setdefault("seen_chunk_ids", set())
        pool = state.setdefault("evidence_pool", {})
        new_ids = new_evidence_ids(seen_chunk_ids=seen, rows=rows)
        for row in rows:
            chunk_id = row.get("chunk_id")
            if chunk_id:
                pool[chunk_id] = row
                seen.add(chunk_id)

        state["last_new_ids"] = len(new_ids)
        state["last_retrieval_result"] = result
        state.setdefault("retrieval_history", []).append(
            {
                "round": state["retrieval_count"],
                "query": state["active_query"],
                "new_chunk_ids": new_ids,
                "row_count": len(rows),
                "top_scores": [round(float(row.get("score", -999.0)), 4) for row in rows[:3]],
            }
        )
        return state

    return retrieve
