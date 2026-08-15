"""Abstention decision logic; support validation moved to src.generation.support_validator (P1)."""
from __future__ import annotations

from typing import Any

from src.generation.support_validator import COMMON_EXEMPT_TERMS, TIME_PATTERNS, TITLE_PATTERN, VALUE_UNIT_PATTERN, _evidence_text, _is_time_like_numeric_token, _is_value_numeric_token, _numeric_claim_text, _query_terms, _title_overlap_score, _title_text, detect_conflicting_evidence, evidence_source_count, validate_answer_support




import re
from typing import Any

from src.utils.text_utils import extract_numeric_tokens, extract_terms, normalize_for_match, normalize_text



def select_used_evidence_ids(parsed_ids: list[str], evidence_rows: list[dict[str, Any]], question_type: str) -> list[str]:
    valid_ids = [row["evidence_id"] for row in evidence_rows]
    filtered = [evidence_id for evidence_id in parsed_ids if evidence_id in valid_ids]
    if filtered:
        selected = list(dict.fromkeys(filtered))
    elif question_type == "fact":
        selected = valid_ids[:1]
    elif question_type == "comparison":
        selected = valid_ids[:2]
    else:
        selected = valid_ids[: min(3, len(valid_ids))]

    if question_type in {"comparison", "inductive"}:
        evidence_by_id = {row["evidence_id"]: row for row in evidence_rows}
        covered_docs = {evidence_by_id[evidence_id]["doc_id"] for evidence_id in selected if evidence_id in evidence_by_id}
        for row in evidence_rows:
            if row["evidence_id"] in selected:
                continue
            if row["doc_id"] in covered_docs:
                continue
            selected.append(row["evidence_id"])
            covered_docs.add(row["doc_id"])
            if len(covered_docs) >= 2:
                break
    return selected


def decide_abstention(
    *,
    question_type: str,
    evidence_rows: list[dict[str, Any]],
    retrieval_scores: dict[str, float],
    support_validation: dict[str, Any],
    json_parse_failures: int,
    answer_mode: str | None = None,
    fallback_reason: str | None = None,
) -> tuple[bool, str | None]:
    del retrieval_scores
    if fallback_reason is not None:
        return True, fallback_reason
    if support_validation.get("domain_mismatch"):
        return True, "domain_mismatch"
    if support_validation.get("low_title_overlap"):
        return True, "low_title_overlap"
    if support_validation.get("missing_query_terms_high"):
        return True, "missing_query_terms_high"
    if not evidence_rows:
        return True, "no_evidence"
    if json_parse_failures >= 2 and answer_mode != "report_lookup":
        return True, "invalid_model_json"
    if answer_mode == "report_lookup":
        return False, None
    if question_type in {"comparison", "inductive"} and evidence_source_count(evidence_rows) < 2:
        return True, "insufficient_source_diversity"
    if support_validation.get("missing_numeric_tokens"):
        return True, "unsupported_numeric_claim"
    if support_validation.get("conflict_detected"):
        return True, "conflicting_evidence"
    return False, None


def compute_confidence_label(*, retrieval_scores: dict[str, float], support_validation: dict[str, Any], abstained: bool) -> str:
    if abstained:
        return "low"
    rerank_top1 = retrieval_scores.get("rerank_top1", -999.0)
    if rerank_top1 >= 2 and support_validation.get("supported") and not support_validation.get("conflict_detected"):
        return "high"
    if rerank_top1 >= 0 and not support_validation.get("missing_numeric_tokens"):
        return "medium"
    return "low"
