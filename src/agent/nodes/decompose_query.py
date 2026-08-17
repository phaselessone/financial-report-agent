"""decompose_query node: remote LLM query decomposition (checklist v3.0 §P6).

A multi-hop query is decomposed into independent sub-questions, each answerable
by one fact lookup or one document search. The node follows the same LLM-call
convention as :mod:`src.agent.nodes.rewrite_query`: budget guards first, one
``llm.generate`` call, ``record_llm_response``, then ``parse_model_json``.

Decomposition is deterministic-triggered upstream (``is_multi_hop_query``); a
malformed or empty decomposition degrades to a single sub-question that echoes
the original query, so the multi-hop path stays correct under LLM failure.
"""

from __future__ import annotations

from typing import Any

from src.agent.config import AgentConfig
from src.agent.policies import begin_node, record_llm_response, token_budget_exceeded
from src.generation.payload_parser import parse_model_json

_DECOMPOSE_SYSTEM_PROMPT = (
    "You are a financial research query decomposer. "
    "Break the multi-hop question into independent sub-questions, each answerable "
    "by a single fact lookup or a single document search. "
    "Preserve company names, metrics and periods exactly. "
    'Output one JSON object only: {"sub_questions": [{"id": "q1", "query": "...", "required_fields": []}]}. '
    "required_fields lists the evidence kinds each sub-question needs. At most 6 sub-questions."
)


def parse_sub_questions(
    payload: dict[str, Any] | None,
    *,
    query: str,
    max_sub_questions: int,
) -> list[dict[str, Any]] | None:
    """Validate and normalize a decomposition payload.

    Returns ``None`` when the payload has no usable sub-questions; otherwise a
    list of at most ``max_sub_questions`` entries. Empty/duplicate queries and
    non-dict entries are dropped.
    """
    if not isinstance(payload, dict):
        return None
    raw = payload.get("sub_questions")
    if not isinstance(raw, list) or not raw:
        return None
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            continue
        sub_query = str(item.get("query") or "").strip()
        if not sub_query:
            continue
        if sub_query in seen:
            continue
        seen.add(sub_query)
        required = item.get("required_fields")
        if not isinstance(required, list):
            required = []
        out.append(
            {
                "id": str(item.get("id") or f"q{index}"),
                "query": sub_query,
                "required_fields": [str(field) for field in required if isinstance(field, str) and field.strip()],
            }
        )
        if len(out) >= max_sub_questions:
            break
    return out or None


def make_decompose_query(llm, config: AgentConfig):
    def decompose_query(state: dict[str, Any]) -> dict[str, Any]:
        if not begin_node(state, config):
            return state
        if int(state.get("llm_call_count", 0)) >= config.max_llm_calls:
            state["termination_reason"] = "max_llm_calls"
            return state
        if token_budget_exceeded(state, config):
            return state

        user_prompt = (
            f"Original query: {state['query']}\n"
            f"Question type: {state.get('question_type', '')}\n"
            f"Answer mode: {state.get('answer_mode', '')}\n"
            f"Domain hint: {state.get('domain_hint', '') or state.get('query_domain_bucket', '')}\n"
        )
        response = llm.generate(
            messages=[
                {"role": "system", "content": _DECOMPOSE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=512,
        )
        record_llm_response(state, response, node="decompose_query")

        payload = parse_model_json(response.content) or {}
        subs = parse_sub_questions(payload, query=state["query"], max_sub_questions=config.max_sub_questions)
        if not subs:
            subs = [{"id": "q1", "query": state["query"], "required_fields": []}]
        state["sub_questions"] = subs
        return state

    return decompose_query
