"""analyze_query node: deterministic routing only, no API call (checklist §P2)."""

from __future__ import annotations

from typing import Any

from src.agent.config import AgentConfig
from src.agent.policies import begin_node
from src.generation.routing import (
    infer_answer_mode,
    infer_fact_subtype,
    infer_question_type,
    is_multi_hop_query,
    is_numeric_or_table_query,
    resolve_query_domain_buckets,
)


def make_analyze_query(config: AgentConfig):
    def analyze_query(state: dict[str, Any]) -> dict[str, Any]:
        if not begin_node(state, config):
            return state
        query = state.get("active_query") or state["query"]
        question_type = state.get("question_type") or infer_question_type(query)
        answer_mode = infer_answer_mode(query, question_type)
        fact_subtype = infer_fact_subtype(query, answer_mode)
        domain_buckets = resolve_query_domain_buckets(query, domain_hint=state.get("domain_hint", ""))
        state.update(
            {
                "question_type": question_type,
                "answer_mode": answer_mode,
                "fact_subtype": fact_subtype,
                "numeric_query": is_numeric_or_table_query(query),
                "query_domain_buckets": domain_buckets,
                "query_domain_bucket": domain_buckets[0] if domain_buckets else "",
                "is_multi_hop": is_multi_hop_query(query, question_type=question_type, answer_mode=answer_mode, fact_subtype=fact_subtype),
            }
        )
        return state

    return analyze_query
