"""rewrite_query node: remote LLM rewrite with structured JSON output."""

from __future__ import annotations

from typing import Any

from src.agent.config import AgentConfig
from src.agent.policies import begin_node, record_llm_response, token_budget_exceeded
from src.generation.payload_parser import parse_model_json

_REWRITE_SYSTEM_PROMPT = (
    "You are a financial research query rewriter. "
    "Rewrite the query so it can retrieve the evidence that is MISSING from the "
    "chunks already seen (listed under Top evidence). Prefer report-specific "
    "terms (companies, products, metrics, periods) that distinguish the target "
    "documents; drop terms that led to the insufficient evidence. "
    "Output one JSON object only: "
    '{"rewritten_query": "...", "reason": "..."}. '
    "Reason must be one of: no_evidence, domain_mismatch, source_diversity_missing, numeric_missing."
)


def _top_evidence_lines(state: dict[str, Any], *, limit: int = 3) -> list[str]:
    result = state.get("last_retrieval_result") or {}
    rows = result.get("rerank_rows") or result.get("hybrid_rows") or []
    lines: list[str] = []
    for row in rows[:limit]:
        if not isinstance(row, dict):
            continue
        title = row.get("file_name", "")
        snippet = (row.get("support_span") or row.get("child_text") or row.get("text") or "")[:160]
        score = row.get("score")
        lines.append(f"- [{title} score={score}] {snippet}")
    return lines


def make_rewrite_query(llm, config: AgentConfig):
    def rewrite_query(state: dict[str, Any]) -> dict[str, Any]:
        if not begin_node(state, config):
            return state
        if int(state.get("rewrite_count", 0)) >= config.max_query_rewrites:
            state["termination_reason"] = "max_query_rewrites"
            return state
        if int(state.get("llm_call_count", 0)) >= config.max_llm_calls:
            state["termination_reason"] = "max_llm_calls"
            return state
        if token_budget_exceeded(state, config):
            return state

        evidence_lines = _top_evidence_lines(state)
        evidence_block = (
            "Top evidence already seen (may be insufficient):\n" + "\n".join(evidence_lines)
            if evidence_lines
            else ""
        )
        draft_answer = state.get("draft_answer") or {}
        current_answer = str(draft_answer.get("final_answer", "") or "")[:200]
        answer_block = f"Current draft answer: {current_answer}\n" if current_answer else ""
        user_prompt = (
            f"Original query: {state['query']}\n"
            f"Current query: {state['active_query']}\n"
            f"Question type: {state.get('question_type', '')}\n"
            f"Domain hint: {state.get('domain_hint', '') or state.get('query_domain_bucket', '')}\n"
            f"Missing information: {state.get('missing_information', '')}\n"
            f"Evidence chunks seen so far: {len(state.get('evidence_pool', {}))}\n"
            f"{answer_block}"
            f"{evidence_block}"
        )
        response = llm.generate(
            messages=[
                {"role": "system", "content": _REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=config.rewrite_temperature,
            max_tokens=config.rewrite_max_tokens,
        )
        record_llm_response(state, response, node="rewrite_query")

        payload = parse_model_json(response.content) or {}
        rewritten = str(payload.get("rewritten_query") or "").strip() or state["active_query"]
        reason = str(payload.get("reason") or state.get("missing_information") or "")
        state["active_query"] = rewritten
        state.setdefault("rewritten_queries", []).append(
            {"rewritten_query": rewritten, "reason": reason}
        )
        state["rewrite_count"] = int(state.get("rewrite_count", 0)) + 1
        return state

    return rewrite_query
