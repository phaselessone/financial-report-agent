"""synthesize node: generate the draft answer via the existing AnswerService."""

from __future__ import annotations

from typing import Any

from src.agent.config import AgentConfig
from src.agent.policies import begin_node, record_llm_response, token_budget_exceeded


def make_synthesize(answerer, config: AgentConfig):
    def synthesize(state: dict[str, Any]) -> dict[str, Any]:
        if not begin_node(state, config):
            return state
        if int(state.get("generation_count", 0)) >= config.max_generation_attempts:
            state["termination_reason"] = "max_generation_attempts"
            return state
        if int(state.get("llm_call_count", 0)) >= config.max_llm_calls:
            state["termination_reason"] = "max_llm_calls"
            return state
        if token_budget_exceeded(state, config):
            return state

        retrieval_result = state.get("last_retrieval_result") or {"rerank_rows": [], "hybrid_rows": []}
        calculation_id = None
        calculation_record = None
        if state.get("tool_outcome") == "calculation_success":
            calculations = state.get("calculations") or {}
            if calculations:
                calculation_id, calculation_record = next(reversed(calculations.items()))
                retrieval_result = {
                    **retrieval_result,
                    "calculation": calculation_record,
                    "calculation_id": calculation_id,
                }
        if state.get("reasoning_plan"):
            retrieval_result = {
                **retrieval_result,
                "reasoning_plan": dict(state.get("reasoning_plan") or {}),
                "reasoning_step_results": list(state.get("reasoning_step_results") or []),
                "reasoning_conclusion": dict(state.get("reasoning_conclusion") or {}),
                "calculations": dict(state.get("calculations") or {}),
            }
        if int(state.get("generation_count", 0)) > 0:
            # Resynthesis after a verification retry: feed the answerer the
            # evidence pooled across ALL retrieval rounds (top-5 by score),
            # not just the latest round (checklist §P3 gate remediation).
            pool = state.get("evidence_pool") or {}
            if pool:
                merged_rows = sorted(
                    pool.values(),
                    key=lambda row: float(row.get("score") or row.get("rerank_score") or -999.0),
                    reverse=True,
                )[:5]
                retrieval_result = {**retrieval_result, "rerank_rows": merged_rows, "hybrid_rows": merged_rows}
        calls_before = len(getattr(answerer, "llm_calls", []))

        draft = answerer.answer(
            query=state["active_query"],
            question_type=state.get("question_type", ""),
            retrieval_result=retrieval_result,
            query_domain_hint=state.get("domain_hint", ""),
        )
        reasoning_conclusion = state.get("reasoning_conclusion")
        if (
            isinstance(reasoning_conclusion, dict)
            and reasoning_conclusion.get("answer")
            and bool((state.get("reasoning_coverage") or {}).get("deterministic_conclusion_allowed"))
        ):
            calculation_id = str(reasoning_conclusion.get("calculation_id") or "") or None
            provenance_ids: list[str] = []
            for result in state.get("reasoning_step_results") or []:
                if not isinstance(result, dict) or result.get("calculation_id") != calculation_id:
                    continue
                for evidence_id in result.get("evidence_ids") or []:
                    evidence_id = str(evidence_id)
                    if evidence_id and evidence_id not in provenance_ids:
                        provenance_ids.append(evidence_id)
            citations: list[dict[str, Any]] = []
            for evidence_id in provenance_ids:
                matching = next(
                    (
                        row
                        for row in (state.get("evidence_pool") or {}).values()
                        if isinstance(row, dict)
                        and evidence_id in {str(row.get("evidence_id") or ""), str(row.get("chunk_id") or "")}
                    ),
                    None,
                )
                citation = dict(matching or {"evidence_id": evidence_id})
                citation.setdefault("evidence_id", evidence_id)
                citations.append(citation)
            draft = {
                **draft,
                "query": state.get("query", ""),
                "question_type": state.get("question_type", "comparison"),
                "answer_mode": "comparison",
                "final_answer": str(reasoning_conclusion["answer"]),
                "answer": str(reasoning_conclusion["answer"]),
                "abstained": False,
                "abstain_reason": None,
                "used_evidence_ids": provenance_ids,
                "citations": citations,
                "support_validation": {
                    "supported": True,
                    "reasoning_gate": "deterministic_compare",
                },
                "calculation_id": calculation_id,
                "calculations": dict(state.get("calculations") or {}),
                "reasoning_conclusion": dict(reasoning_conclusion),
                "reasoning_plan": dict(state.get("reasoning_plan") or {}),
                "reasoning_step_results": list(state.get("reasoning_step_results") or []),
            }
            # The comparison record is an internal proof object.  Its numeric
            # difference (for example 0.15) is not the user-facing answer and
            # must not trigger the generic single-calculation fallback below.
            calculation_record = None
            calculation_id = None
        # The answerer remains the presentation layer, but a successful
        # calculator result is an auditable fact.  If the model omitted the
        # result or returned an unsupported/abstaining draft, fall back to a
        # deterministic one-line answer so the derived claim can be verified
        # against the calculation record instead of being silently discarded.
        if calculation_record and isinstance(calculation_record, dict):
            result = calculation_record.get("result") or {}
            formatted = str(result.get("formatted") or "").strip()
            answer_text = str(draft.get("final_answer") or draft.get("answer") or "")
            if formatted and not _calculation_text_matches(answer_text, formatted):
                provenance_ids = [
                    str(item.get("evidence_id") or item.get("fact_id") or "")
                    for item in (calculation_record.get("inputs") or [])
                    if isinstance(item, dict) and (item.get("evidence_id") or item.get("fact_id"))
                ]
                citations = [
                    dict(state.get("evidence_pool", {}).get(evidence_id) or {"evidence_id": evidence_id})
                    for evidence_id in provenance_ids
                ]
                draft = {
                    **draft,
                    "query": state.get("query", ""),
                    "question_type": state.get("question_type", "fact"),
                    "answer_mode": "numeric_fact",
                    "final_answer": f"针对“{state.get('query', '')}”，计算结果为 {formatted}。",
                    "abstained": False,
                    "abstain_reason": None,
                    "used_evidence_ids": provenance_ids,
                    "citations": citations,
                    "support_validation": {"supported": True, "matched_numeric_tokens": [formatted]},
                }
            draft["calculation_id"] = calculation_id
            draft["calculations"] = {str(calculation_id): calculation_record}
        state["draft_answer"] = draft
        state["support_validation"] = draft.get("support_validation") or {}
        state["generation_count"] = int(state.get("generation_count", 0)) + 1

        # Account for every LLM call the AnswerService made under the hood.
        new_calls = list(getattr(answerer, "llm_calls", [])[calls_before:])
        for response in new_calls:
            record_llm_response(state, response, node="synthesize")
        return state

    return synthesize


def _calculation_text_matches(text: str, formatted: str) -> bool:
    """Compare a rendered result without requiring calculator precision text."""
    compact_text = str(text or "").replace(" ", "")
    compact_result = str(formatted or "").replace(" ", "")
    if compact_result and compact_result in compact_text:
        return True
    # ``20.0000%`` and ``20%`` are equivalent for presentation purposes.
    if compact_result.endswith("%"):
        try:
            number = compact_result[:-1]
            number = number.rstrip("0").rstrip(".") or "0"
            return f"{number}%" in compact_text
        except Exception:
            return False
    return False
