"""grade_evidence node: deterministic grading (no API) + no-improvement stop."""

from __future__ import annotations

from typing import Any

from src.agent.config import AgentConfig
from src.agent.policies import begin_node, grade_evidence_deterministic


def make_grade_evidence(config: AgentConfig):
    def grade_evidence(state: dict[str, Any]) -> dict[str, Any]:
        if not begin_node(state, config):
            return state

        # No-improvement stop: a rewrite followed by a retrieval that surfaced
        # zero new chunks stops further searching (checklist §P2).
        if int(state.get("rewrite_count", 0)) > 0 and int(state.get("last_new_ids", 0)) == 0:
            state["no_improvement"] = True
            state["missing_information"] = "rewrite_no_new_evidence"

        retrieval_result = state.get("last_retrieval_result") or {
            "rerank_rows": [],
            "hybrid_rows": [],
        }
        sufficient, reason = grade_evidence_deterministic(
            query=state["active_query"],
            question_type=state.get("question_type", ""),
            answer_mode=state.get("answer_mode", ""),
            fact_subtype=state.get("fact_subtype", ""),
            query_domain_buckets=list(state.get("query_domain_buckets") or []),
            query_domain_bucket=state.get("query_domain_bucket", ""),
            retrieval_result=retrieval_result,
        )
        state["evidence_sufficient"] = sufficient
        if sufficient:
            state["missing_information"] = ""
        elif not state.get("no_improvement"):
            state["missing_information"] = reason
        return state

    return grade_evidence
