"""Agent policies: deterministic evidence grading and budget/termination rules.

All grading here is deterministic (checklist v3.0 §P2: grade_evidence 优先
deterministic — no evidence / domain mismatch / source diversity / numeric
missing; API grading only when undecidable, not implemented in v1).
"""

from __future__ import annotations

from typing import Any

from src.generation.evidence_selector import _prepare_evidence, _selected_domain_mismatch
from src.generation.routing import is_numeric_or_table_query
from src.llm.types import LLMResponse
from src.retrieval.runtime import NUMERIC_QUERY_HINTS
from src.utils.text_utils import extract_numeric_tokens, normalize_for_match, normalize_text

GRADE_REASONS = (
    "no_evidence",
    "domain_mismatch",
    "source_diversity_missing",
    "numeric_missing",
)


def _rows_from_result(retrieval_result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = retrieval_result.get("rerank_rows") or retrieval_result.get("hybrid_rows") or []
    return [row for row in rows if isinstance(row, dict)]


def _evidence_text(row: dict[str, Any]) -> str:
    return normalize_text(
        " ".join(
            str(row.get(key, "") or "")
            for key in ("support_span", "child_text", "text")
        )
    )


def grade_evidence_deterministic(
    *,
    query: str,
    question_type: str,
    answer_mode: str,
    fact_subtype: str,
    query_domain_buckets: list[str],
    query_domain_bucket: str,
    retrieval_result: dict[str, Any],
) -> tuple[bool, str]:
    """Grade the current retrieval result. Returns (sufficient, missing_reason)."""
    rows = _rows_from_result(retrieval_result)
    if not rows:
        return False, "no_evidence"

    selected_evidence, doc_candidates, selected_doc_ids, forced_reason, _doc_guard = _prepare_evidence(
        query=query,
        question_type=question_type,
        answer_mode=answer_mode,
        fact_subtype=fact_subtype,
        query_domain_bucket=query_domain_bucket,
        query_domain_buckets=query_domain_buckets,
        retrieval_result=retrieval_result,
    )
    if forced_reason is not None:
        return False, forced_reason
    if not selected_evidence:
        return False, "no_evidence"

    selected_domain_buckets = sorted(
        {candidate.get("domain_bucket", "") for candidate in doc_candidates if candidate["doc_id"] in set(selected_doc_ids)}
    )
    if _selected_domain_mismatch(query_domain_buckets, selected_domain_buckets, answer_mode=answer_mode):
        return False, "domain_mismatch"

    source_count = len({row["doc_id"] for row in selected_evidence})
    if question_type in {"comparison", "inductive"} and source_count < 2:
        return False, "source_diversity_missing"

    if question_type == "fact" and answer_mode == "numeric_fact" and is_numeric_or_table_query(query):
        query_numerics = [normalize_for_match(token) for token in extract_numeric_tokens(query)]
        query_numerics = [token for token in query_numerics if token and len(token) >= 2]
        has_query_hint = any(hint in normalize_text(query) for hint in NUMERIC_QUERY_HINTS)
        if query_numerics:
            combined = normalize_for_match(" ".join(_evidence_text(row) for row in selected_evidence))
            if not any(token in combined for token in query_numerics):
                return False, "numeric_missing"
        elif has_query_hint and not any(
            normalize_for_match(token)
            for row in selected_evidence
            for token in extract_numeric_tokens(_evidence_text(row))
        ):
            return False, "numeric_missing"

    return True, ""


def new_evidence_ids(*, seen_chunk_ids: set[str], rows: list[dict[str, Any]]) -> list[str]:
    return [row["chunk_id"] for row in rows if row.get("chunk_id") not in seen_chunk_ids]


def record_llm_response(state: dict[str, Any], response: LLMResponse) -> None:
    """Accumulate LLM call/token counters from a single provider response."""
    state["llm_call_count"] = int(state.get("llm_call_count", 0)) + 1
    state["prompt_tokens"] = int(state.get("prompt_tokens", 0)) + response.prompt_tokens
    state["completion_tokens"] = int(state.get("completion_tokens", 0)) + response.completion_tokens
    state["total_tokens"] = int(state.get("total_tokens", 0)) + response.usage_total_tokens()


def begin_node(state: dict[str, Any], config: Any) -> bool:
    """Shared pre-node guard: counts the step and enforces the max_steps budget.

    Returns False (with ``termination_reason`` set) when the run must stop.
    """
    if state.get("termination_reason"):
        return False
    state["step_count"] = int(state.get("step_count", 0)) + 1
    if int(state["step_count"]) > config.max_steps:
        state["termination_reason"] = "max_steps"
        return False
    return True
