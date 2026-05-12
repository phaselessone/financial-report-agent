from __future__ import annotations

import re
from typing import Any

from src.utils.text_utils import extract_numeric_tokens, extract_terms, normalize_for_match, normalize_text


TITLE_PATTERN = re.compile("《[^》]{1,120}》")
TIME_PATTERNS = (
    r"^\d{4}年$",
    r"^\d{1,2}月$",
    r"^\d{1,2}日$",
    r"^第\d+周$",
    r"^[QH][1-4]$",
    r"^\d{4}[QH][1-4]$",
)
VALUE_UNIT_PATTERN = re.compile(
    r"(?:%|万元/吨|万元|亿元|亿|元/吨|元|吨|台|辆|倍|家|个|mAh|Wh|GW|GWh|kWh|Ah|pct|pp)",
    flags=re.IGNORECASE,
)
COMMON_EXEMPT_TERMS = {
    "报告",
    "研报",
    "行业",
    "周报",
    "月报",
    "数据",
    "关键",
    "变化",
    "趋势",
    "哪份",
    "哪篇",
    "哪家",
    "主要",
    "共同",
    "分别",
    "聚焦",
    "讨论",
    "提到",
    "写到",
    "强调",
    "分析",
    "关于",
}


def evidence_source_count(evidence_rows: list[dict[str, Any]]) -> int:
    return len({row["doc_id"] for row in evidence_rows})


def _evidence_text(row: dict[str, Any]) -> str:
    return normalize_text(row.get("support_span") or row.get("child_text") or row.get("text", ""))


def _title_text(row: dict[str, Any]) -> str:
    return normalize_text("\n".join([row.get("file_name", ""), row.get("section_title", "") or "", row.get("section_path", "") or ""]))


def _numeric_claim_text(final_answer: str, evidence_summary: str, *, answer_mode: str | None = None) -> str:
    if answer_mode in {"comparison", "inductive"}:
        text = normalize_text(final_answer)
    else:
        text = normalize_text(f"{final_answer}\n{evidence_summary}")
    return TITLE_PATTERN.sub("", text)


def _is_time_like_numeric_token(token: str) -> bool:
    normalized = normalize_text(token)
    return any(re.fullmatch(pattern, normalized, flags=re.IGNORECASE) for pattern in TIME_PATTERNS)


def _is_value_numeric_token(token: str) -> bool:
    normalized = normalize_text(token)
    if _is_time_like_numeric_token(normalized):
        return False
    if VALUE_UNIT_PATTERN.search(normalized):
        return True
    return bool(re.fullmatch(r"\d+(?:\.\d+)?", normalized))


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


def detect_conflicting_evidence(question_type: str, evidence_rows: list[dict[str, Any]], *, answer_mode: str | None = None) -> bool:
    if question_type != "fact" or answer_mode == "report_lookup" or len(evidence_rows) < 2:
        return False
    if evidence_source_count(evidence_rows) < 2:
        return False
    terms_by_row = [set(extract_terms(_evidence_text(row), top_k=6)) for row in evidence_rows]
    overlap = set.intersection(*terms_by_row) if terms_by_row else set()
    numeric_sets = [set(extract_numeric_tokens(_evidence_text(row))) for row in evidence_rows]
    numeric_overlap = set.intersection(*numeric_sets) if numeric_sets else set()
    return not overlap and not numeric_overlap


def _query_terms(query: str) -> list[str]:
    return [term for term in extract_terms(query, top_k=8) if term not in COMMON_EXEMPT_TERMS]


def _title_overlap_score(query_terms: list[str], evidence_rows: list[dict[str, Any]]) -> int:
    title_text = normalize_for_match("\n".join(_title_text(row) for row in evidence_rows))
    return sum(normalize_for_match(term) in title_text for term in query_terms)


def validate_answer_support(
    *,
    query: str,
    question_type: str,
    final_answer: str,
    evidence_summary: str,
    used_evidence_ids: list[str],
    evidence_rows: list[dict[str, Any]],
    answer_mode: str | None = None,
) -> dict[str, Any]:
    referenced = [row for row in evidence_rows if row["evidence_id"] in set(used_evidence_ids)]
    if not referenced:
        referenced = list(evidence_rows)

    source_count = evidence_source_count(referenced)
    combined_evidence_text = "\n".join(_evidence_text(row) for row in referenced)
    combined_title_text = "\n".join(_title_text(row) for row in referenced)
    normalized_evidence = normalize_for_match(f"{combined_title_text}\n{combined_evidence_text}")
    answer_text = normalize_text(f"{final_answer}\n{evidence_summary}")
    answer_terms = extract_terms(answer_text, top_k=10)
    query_terms = _query_terms(query)
    missing_keywords = [term for term in query_terms if normalize_for_match(term) not in normalized_evidence]
    matched_query_terms = [term for term in query_terms if normalize_for_match(term) in normalized_evidence]
    title_overlap_score = _title_overlap_score(query_terms, referenced)
    low_title_overlap = answer_mode == "report_lookup" and bool(query_terms) and title_overlap_score < min(2, len(query_terms))

    numeric_text = re.sub(r"E\d+", "", _numeric_claim_text(final_answer, evidence_summary, answer_mode=answer_mode), flags=re.IGNORECASE)
    raw_numeric_tokens = [token for token in extract_numeric_tokens(numeric_text) if not re.fullmatch(r"\d{6,8}", token)]
    support_filter_applied = "default"
    if answer_mode in {"comparison", "inductive"}:
        answer_numeric_tokens = [token for token in raw_numeric_tokens if _is_value_numeric_token(token)]
        support_filter_applied = "value_numeric_only"
    else:
        answer_numeric_tokens = raw_numeric_tokens

    missing_numeric_tokens = [
        token
        for token in answer_numeric_tokens
        if normalize_for_match(token) and normalize_for_match(token) not in normalized_evidence
    ]
    matched_numeric_tokens = [
        token
        for token in answer_numeric_tokens
        if normalize_for_match(token) and normalize_for_match(token) in normalized_evidence
    ]

    if answer_mode == "report_lookup":
        missing_query_terms_high = len(missing_keywords) >= 2 and title_overlap_score < 2
    elif answer_mode in {"comparison", "inductive"}:
        missing_query_terms_high = len(query_terms) >= 2 and len(matched_query_terms) < 2 and len(missing_keywords) >= 2
        if source_count >= 2 and answer_mode == "inductive":
            missing_query_terms_high = False
    else:
        missing_query_terms_high = len(query_terms) >= 3 and len(matched_query_terms) == 0 and len(missing_keywords) >= 2
        if matched_numeric_tokens and source_count >= 1:
            missing_query_terms_high = False

    comparison_source_ok = question_type != "comparison" or source_count >= 2
    inductive_source_ok = question_type != "inductive" or source_count >= 2
    conflict_detected = detect_conflicting_evidence(question_type, referenced, answer_mode=answer_mode)
    return {
        "supported": not missing_numeric_tokens and comparison_source_ok and inductive_source_ok and not conflict_detected,
        "missing_numeric_tokens": missing_numeric_tokens,
        "matched_numeric_tokens": matched_numeric_tokens,
        "missing_keywords": missing_keywords,
        "answer_numeric_tokens": answer_numeric_tokens,
        "answer_terms": answer_terms,
        "matched_query_terms": matched_query_terms,
        "query_term_count": len(query_terms),
        "citation_source_count": source_count,
        "comparison_source_ok": comparison_source_ok,
        "inductive_source_ok": inductive_source_ok,
        "conflict_detected": conflict_detected,
        "support_filter_applied": support_filter_applied,
        "title_overlap_score": title_overlap_score,
        "low_title_overlap": low_title_overlap,
        "missing_query_terms_high": missing_query_terms_high,
        "domain_mismatch": False,
    }


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
