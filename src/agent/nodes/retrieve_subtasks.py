"""retrieve_subtasks node: serial per-sub-question retrieval (checklist v3.0 §P6).

For each decomposed sub-question, tries the structured fact store first (zero
LLM, zero retrieval round) and falls back to one RAG retrieval. Evidence from
every sub-question accumulates into ``evidence_pool``; the final
``last_retrieval_result`` is the merged, de-duplicated evidence fed to the
existing synthesize node.
"""

from __future__ import annotations

from typing import Any

from src.agent.config import AgentConfig
from src.agent.policies import begin_node, new_evidence_ids
from src.retrieval.domain_priority import apply_retrieval_domain_priority
from src.structured.agent_bridge import financial_fact_to_row
from src.structured.fact_query import route


def _fact_query_to_dict(fact_query: Any) -> dict[str, Any]:
    return {
        "company": fact_query.company,
        "metric": fact_query.metric.value,
        "period": fact_query.period.to_dict(),
    }


def make_retrieve_subtasks(runtime, fact_store, company_aliases, config: AgentConfig):
    def retrieve_subtasks(state: dict[str, Any]) -> dict[str, Any]:
        if not begin_node(state, config):
            return state

        subs = list(state.get("sub_questions") or [])
        domain_hint = state.get("query_domain_bucket", "") or state.get("domain_hint", "")
        merged_rows: list[dict[str, Any]] = []
        seen = state.setdefault("seen_chunk_ids", set())
        pool = state.setdefault("evidence_pool", {})

        for index, sub in enumerate(subs, start=1):
            if not isinstance(sub, dict):
                continue
            query = str(sub.get("query") or "").strip()
            if not query:
                continue
            sub_id = str(sub.get("id") or f"q{index}")
            state["active_query"] = query

            structured_result = None
            if fact_store is not None:
                structured_result = route(query, store=fact_store, company_aliases=company_aliases or {})

            if structured_result is not None and structured_result.get("facts"):
                # Structured hit: zero LLM, zero retrieval round.
                fact_rows = [financial_fact_to_row(fact) for fact in structured_result["facts"]]
                for row in fact_rows:
                    chunk_id = row["chunk_id"]
                    pool[chunk_id] = row
                    seen.add(chunk_id)
                    merged_rows.append(row)
                state.setdefault("structured_facts", []).append(
                    {
                        "sub_question_id": sub_id,
                        "query": query,
                        "routed": True,
                        "fact_query": _fact_query_to_dict(structured_result["fact_query"]),
                        "facts": [fact.to_dict() for fact in structured_result["facts"]],
                        "answer": structured_result["answer"],
                        "citations": structured_result["citations"],
                    }
                )
                state.setdefault("sub_question_results", []).append(
                    {
                        "id": sub_id,
                        "query": query,
                        "source": "structured",
                        "fact_count": len(structured_result["facts"]),
                    }
                )
            else:
                # RAG fallback: exactly one retrieval round, no rewrite.
                # Bounded by max_sub_questions (one search per sub-question),
                # NOT by max_retrieval_rounds — per P6 plan §1 that budget
                # constrains only the single-hop rewrite loop. Skipping RAG
                # sub-questions here would silently drop part of a decomposed
                # query (code-review P1).
                result = runtime.search(query)
                result = apply_retrieval_domain_priority(result, domain_hint)
                state["retrieval_count"] = int(state.get("retrieval_count", 0)) + 1

                rows = list(result.get("rerank_rows") or result.get("hybrid_rows") or [])
                rows = [row for row in rows if isinstance(row, dict)]
                new_ids = new_evidence_ids(seen_chunk_ids=seen, rows=rows)
                for row in rows:
                    chunk_id = row.get("chunk_id")
                    if chunk_id:
                        pool[chunk_id] = row
                        seen.add(chunk_id)
                        merged_rows.append(row)
                state.setdefault("sub_question_results", []).append(
                    {
                        "id": sub_id,
                        "query": query,
                        "source": "rag",
                        "new_chunk_ids": new_ids,
                        "row_count": len(rows),
                    }
                )

        # Synthesize must answer the ORIGINAL question, not the last sub-question.
        state["active_query"] = state["query"]

        deduped: dict[str, dict[str, Any]] = {}
        for row in merged_rows:
            deduped[row["chunk_id"]] = row
        merged = sorted(
            deduped.values(),
            key=lambda row: float(row.get("rerank_score") or row.get("score") or 0.0),
            reverse=True,
        )
        state["last_retrieval_result"] = {
            "query_mode": "multi_hop",
            "numeric_query": False,
            "dense_rows": [],
            "bm25_rows": [],
            "hybrid_rows": merged,
            "rerank_rows": merged,
            "timings": {},
        }
        return state

    return retrieve_subtasks
