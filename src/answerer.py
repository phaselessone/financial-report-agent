from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from src.evaluation.benchmark_assets import industry_from_file_name, short_title_from_file_name, topic_hint_from_title
from src.generation.abstain import compute_confidence_label, decide_abstention, select_used_evidence_ids, validate_answer_support
from src.generation.citation_builder import build_citations
from src.generation.prompt import SYSTEM_PROMPT, build_answer_prompt
from src.retrieval.model_store import ensure_model_downloaded
from src.utils.text_utils import extract_terms, first_sentence, normalize_for_match, normalize_text, strip_file_extension

QUESTION_TYPE_KEYWORDS = {
    "comparison": ("\u6bd4\u8f83", "\u5206\u522b", "\u5dee\u5f02", "\u5bf9\u6bd4"),
    "inductive": ("\u5171\u540c", "\u5171\u6027", "\u603b\u7ed3", "\u5f52\u7eb3", "\u5f3a\u8c03"),
}
REPORT_LOOKUP_PATTERN = re.compile("(?:\u54ea(?:\u4efd|\u7bc7)?\u62a5\u544a|\u54ea(?:\u4efd|\u7bc7)?\u7814\u62a5|\u54ea\u5bb6\u5238\u5546)")
VALUE_QUERY_HINTS = (
    "\u591a\u5c11",
    "\u51e0",
    "\u540c\u6bd4",
    "\u73af\u6bd4",
    "\u589e\u957f\u7387",
    "\u4ef7\u683c",
    "\u62a5\u4ef7",
    "\u73b0\u8d27",
    "\u5747\u4ef7",
    "\u51fa\u8d27\u91cf",
    "\u9500\u91cf",
    "\u89c4\u6a21",
    "\u5e02\u5360\u7387",
    "\u5355\u4ef7",
    "\u4ea7\u91cf",
    "\u91d1\u989d",
    "\u62d0\u70b9",
    "%",
)
NUMERIC_QUERY_HINTS = VALUE_QUERY_HINTS + (
    "\u6570\u636e",
    "\u53d8\u5316",
    "\u5173\u952e\u6570\u636e",
    "\u5173\u952e\u53d8\u5316",
)
QUERY_DOMAIN_HINTS = {
    "semiconductor": ("\u534a\u5bfc\u4f53", "\u82af\u7247", "\u5c01\u6d4b", "\u6676\u5706", "\u5b58\u50a8", "\u6a21\u62df", "semicon", "gtc"),
    "new_energy": ("\u65b0\u80fd\u6e90", "\u5149\u4f0f", "\u9502\u7535", "\u50a8\u80fd", "\u65b0\u80fd\u6e90\u6c7d\u8f66", "\u52a8\u529b\u7535\u6c60", "\u56fa\u6001\u7535\u6c60", "\u78b3\u9178\u9502", "\u6c22\u6c27\u5316\u9502", "\u7279\u65af\u62c9"),
    "liquor": ("\u767d\u9152", "\u9152\u4f01", "\u8305\u53f0", "\u4e94\u7cae\u6db2", "\u98df\u54c1\u996e\u6599", "\u6625\u8282", "\u6279\u4ef7"),
    "consumer": ("\u6d88\u8d39", "\u5546\u793e", "\u7f8e\u62a4", "\u5bb6\u7535", "\u96f6\u552e", "\u5185\u9700", "\u73b0\u4ee3\u670d\u52a1\u4e1a", "\u5927\u4f17\u54c1", "\u5fc5\u9009\u6d88\u8d39\u54c1", "\u521b\u4e1a\u677f", "\u5a74\u914d\u7c89", "\u996e\u6599", "\u5546\u8d38"),
}
FALLBACK_ANSWER = "信息不足，暂无无法给出可靠答案。"

def is_fallback_answer(text: str) -> bool:
    return normalize_text(text) == FALLBACK_ANSWER


def _fallback_abstain_reason(*, final_answer: str, fallback_source: str | None) -> str | None:
    if not is_fallback_answer(final_answer):
        return None
    return fallback_source or "unsupported_answer"

GENERIC_THEME_TERMS = {
    "\u677f\u5757",
    "\u884c\u4e1a",
    "\u5468\u62a5",
    "\u62a5\u544a",
    "\u6570\u636e",
    "\u89c2\u70b9",
    "\u63a8\u8350",
    "\u5e02\u573a",
    "\u8868\u73b0",
    "\u4e13\u9898",
    "\u8ddf\u8e2a",
}


def infer_question_type(query: str) -> str:
    normalized = normalize_text(query)
    for question_type, keywords in QUESTION_TYPE_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return question_type
    return "fact"



def infer_answer_mode(query: str, question_type: str | None = None) -> str:
    resolved_question_type = question_type or infer_question_type(query)
    if resolved_question_type == "comparison":
        return "comparison"
    if resolved_question_type == "inductive":
        return "inductive"
    if REPORT_LOOKUP_PATTERN.search(normalize_text(query)):
        return "report_lookup"
    return "numeric_fact"



def infer_fact_subtype(query: str, answer_mode: str) -> str:
    if answer_mode == "report_lookup":
        return "report_lookup"
    normalized = normalize_text(query)
    if any(keyword in normalized for keyword in VALUE_QUERY_HINTS):
        return "value_fact"
    if re.search(r"\d+(?:\.\d+)?%?", normalized):
        return "value_fact"
    return "semantic_fact"



def is_numeric_or_table_query(query: str) -> bool:
    normalized = normalize_text(query)
    return any(keyword in normalized for keyword in NUMERIC_QUERY_HINTS) or any(char.isdigit() for char in normalized)



def infer_query_domain_bucket(query: str) -> str:
    normalized = normalize_text(query).lower()
    for bucket in ("semiconductor", "new_energy", "liquor", "consumer"):
        if any(token in normalized for token in QUERY_DOMAIN_HINTS[bucket]):
            return bucket
    return ""


def _row_score(row: dict[str, Any]) -> float:
    return float(row.get("rerank_score", row.get("score", 0.0)))


def _row_text(row: dict[str, Any], *, prefer_exact: bool = False) -> str:
    if prefer_exact:
        return normalize_text(row.get("support_span") or row.get("child_text") or row.get("text", ""))
    return normalize_text(row.get("child_text") or row.get("text", ""))


def _row_match_text(row: dict[str, Any]) -> str:
    return normalize_for_match(
        "\n".join(
            [
                _row_text(row, prefer_exact=True),
                normalize_text(row.get("section_title", "") or ""),
                normalize_text(row.get("section_path", "") or ""),
                normalize_text(row.get("file_name", "") or ""),
            ]
        )
    )


def _is_structured_row(row: dict[str, Any]) -> bool:
    return row.get("chunk_type") == "table_like" or row.get("element_type") in {"table", "figure"}


def _is_paragraph_like(row: dict[str, Any]) -> bool:
    return row.get("element_type") in {"paragraph", "figure_context", "figure_caption"}


def _same_context(anchor: dict[str, Any], candidate: dict[str, Any]) -> bool:
    if anchor["doc_id"] != candidate["doc_id"]:
        return False
    same_page = int(anchor["page_start"]) == int(candidate["page_start"]) and int(anchor["page_end"]) == int(candidate["page_end"])
    same_section = bool(anchor.get("section_path") and anchor.get("section_path") == candidate.get("section_path"))
    return same_page or same_section


def _short_report_title(file_name: str) -> str:
    return short_title_from_file_name(file_name)


def _title_topic_hint(file_name: str) -> str:
    return normalize_text(topic_hint_from_title(_short_report_title(file_name)))


def _row_topic_hint(row: dict[str, Any]) -> str:
    title_hint = _title_topic_hint(row["file_name"])
    if len(normalize_for_match(title_hint)) >= 6:
        return title_hint
    return first_sentence(_row_text(row, prefer_exact=True), max_chars=64)


def _query_terms(query: str, *, top_k: int = 8) -> list[str]:
    return [normalize_for_match(term) for term in extract_terms(query, top_k=top_k) if normalize_for_match(term)]


def _title_overlap_score(query: str, row_or_title: dict[str, Any] | str) -> int:
    query_terms = _query_terms(query)
    if not query_terms:
        return 0
    if isinstance(row_or_title, dict):
        file_name = row_or_title.get("file_name", "")
        section_title = row_or_title.get("section_title", "") or ""
    else:
        file_name = row_or_title
        section_title = ""
    title_text = normalize_for_match(
        "\n".join((
            _short_report_title(file_name),
            normalize_text(file_name or ""),
            normalize_text(section_title),
        ))
    )
    return sum(term in title_text for term in query_terms)


def _unique_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_chunk: dict[str, dict[str, Any]] = {}
    for row in rows:
        chunk_id = row["chunk_id"]
        payload = dict(row)
        if chunk_id not in by_chunk or _row_score(payload) > _row_score(by_chunk[chunk_id]):
            by_chunk[chunk_id] = payload
    return sorted(by_chunk.values(), key=_row_score, reverse=True)


def _ordered_pool(retrieval_result: dict[str, Any], *, extra_hybrid: int = 12) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    seen_chunk_ids: set[str] = set()
    for row in list(retrieval_result.get("rerank_rows", [])) + list(retrieval_result.get("hybrid_rows", []))[:extra_hybrid]:
        if row["chunk_id"] in seen_chunk_ids:
            continue
        ordered.append(dict(row))
        seen_chunk_ids.add(row["chunk_id"])
    return ordered


def _build_doc_candidates(query: str, answer_mode: str, retrieval_result: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _ordered_pool(retrieval_result):
        grouped[row["doc_id"]].append(dict(row))

    candidates: list[dict[str, Any]] = []
    for doc_id, rows in grouped.items():
        ordered_rows = _unique_rows(rows)
        if not ordered_rows:
            continue
        rerank_scores = [float(row["rerank_score"]) for row in ordered_rows if row.get("rerank_score") is not None]
        max_rerank_score = max(rerank_scores) if rerank_scores else _row_score(ordered_rows[0])
        top2_chunk_score_sum = sum(_row_score(row) for row in ordered_rows[:2])
        title_overlap = _title_overlap_score(query, ordered_rows[0])
        structured_bonus = sum(1 for row in ordered_rows[:2] if _is_structured_row(row))
        doc_score = max_rerank_score + 0.35 * top2_chunk_score_sum + 0.9 * title_overlap + 0.15 * structured_bonus
        if answer_mode == "report_lookup":
            doc_score += 0.75 * title_overlap
        candidates.append(
            {
                "doc_id": doc_id,
                "file_name": ordered_rows[0]["file_name"],
                "short_title": _short_report_title(ordered_rows[0]["file_name"]),
                "domain_bucket": industry_from_file_name(ordered_rows[0]["file_name"]),
                "doc_score": float(doc_score),
                "max_rerank_score": float(max_rerank_score),
                "top2_chunk_score_sum": float(top2_chunk_score_sum),
                "title_overlap_score": int(title_overlap),
                "structured_bonus": int(structured_bonus),
                "rows": ordered_rows,
            }
        )

    if answer_mode == "report_lookup":
        candidates.sort(key=lambda item: (item["title_overlap_score"], item["doc_score"], item["max_rerank_score"]), reverse=True)
    else:
        candidates.sort(key=lambda item: (item["doc_score"], item["title_overlap_score"], item["max_rerank_score"]), reverse=True)
    return candidates


def _apply_domain_guard(doc_candidates: list[dict[str, Any]], *, query_domain_bucket: str, answer_mode: str) -> tuple[list[dict[str, Any]], bool]:
    if not query_domain_bucket:
        return doc_candidates, False
    same_bucket = [candidate for candidate in doc_candidates if candidate.get("domain_bucket") == query_domain_bucket]
    if not same_bucket:
        return doc_candidates, False

    if answer_mode in {"comparison", "inductive"}:
        if len(same_bucket) >= 2:
            return same_bucket, len(same_bucket) != len(doc_candidates)
        return doc_candidates, False

    top_candidate = doc_candidates[0]
    if top_candidate.get("domain_bucket") == query_domain_bucket:
        return doc_candidates, False

    guarded_candidate = same_bucket[0]
    top_score = float(top_candidate.get("doc_score", 0.0))
    guarded_score = float(guarded_candidate.get("doc_score", 0.0))
    score_gap_ratio = (top_score - guarded_score) / max(abs(top_score), 1e-6)
    if guarded_candidate.get("title_overlap_score", 0) > 0 or score_gap_ratio <= 0.2:
        reordered = same_bucket + [candidate for candidate in doc_candidates if candidate.get("domain_bucket") != query_domain_bucket]
        return reordered, True
    return doc_candidates, False


def _select_doc_anchor(doc_rows: list[dict[str, Any]], *, answer_mode: str, fact_subtype: str, numeric_query: bool) -> dict[str, Any] | None:
    paragraph_rows = [row for row in doc_rows if _is_paragraph_like(row)]
    prefer_paragraph = answer_mode in {"report_lookup", "comparison", "inductive"} or fact_subtype == "semantic_fact"
    candidate_rows = paragraph_rows if prefer_paragraph and paragraph_rows else doc_rows

    best_row: dict[str, Any] | None = None
    best_score = float("-inf")
    for row in candidate_rows:
        score = _row_score(row)
        page_start = int(row.get("page_start", 1))
        is_structured = _is_structured_row(row)
        is_paragraph_like = _is_paragraph_like(row)
        if page_start <= 2:
            score += 0.18
        elif page_start <= 4:
            score += 0.08
        if row.get("support_span"):
            score += 0.03
        if row.get("section_path"):
            score += 0.02
        if answer_mode == "report_lookup":
            if is_paragraph_like:
                score += 0.55
            if is_structured:
                score -= 0.5
            if page_start <= 2:
                score += 0.4
        elif answer_mode == "comparison":
            if is_paragraph_like:
                score += 0.45
            if is_structured:
                score -= 0.2
        elif answer_mode == "inductive":
            if is_paragraph_like:
                score += 0.6
            if is_structured:
                score -= 0.35
        elif fact_subtype == "semantic_fact":
            if is_paragraph_like:
                score += 0.45
            if is_structured:
                score -= 0.35
        else:
            if numeric_query and is_structured:
                score += 0.22
            if is_paragraph_like:
                score += 0.08
        if score > best_score:
            best_score = score
            best_row = dict(row)
    return best_row


def _best_companion(anchor: dict[str, Any], candidates: list[dict[str, Any]], *, numeric_query: bool, require_same_section: bool = False, prefer_paragraph: bool = False) -> dict[str, Any] | None:
    candidate_rows = [row for row in candidates if row["chunk_id"] != anchor["chunk_id"] and row["doc_id"] == anchor["doc_id"]]
    if prefer_paragraph:
        paragraph_rows = [row for row in candidate_rows if _is_paragraph_like(row)]
        if paragraph_rows:
            candidate_rows = paragraph_rows

    best_row: dict[str, Any] | None = None
    best_score = float("-inf")
    for candidate in candidate_rows:
        if require_same_section and anchor.get("section_path") and anchor.get("section_path") != candidate.get("section_path"):
            continue
        score = _row_score(candidate)
        if _same_context(anchor, candidate):
            score += 0.35
        if anchor.get("section_path") and anchor.get("section_path") == candidate.get("section_path"):
            score += 0.15
        if _is_structured_row(anchor) and _is_paragraph_like(candidate):
            score += 0.25
        if numeric_query and _is_structured_row(candidate):
            score += 0.08
        if score > best_score:
            best_score = score
            best_row = dict(candidate)
    return best_row


def _select_rows_for_doc(doc_candidate: dict[str, Any], *, answer_mode: str, fact_subtype: str, numeric_query: bool, max_rows: int = 2, require_same_section: bool = False) -> list[dict[str, Any]]:
    doc_rows = list(doc_candidate.get("rows", []))
    anchor = _select_doc_anchor(doc_rows, answer_mode=answer_mode, fact_subtype=fact_subtype, numeric_query=numeric_query)
    if anchor is None:
        return []
    selected = [anchor]
    companion = _best_companion(
        anchor,
        doc_rows,
        numeric_query=numeric_query,
        require_same_section=require_same_section,
        prefer_paragraph=answer_mode in {"comparison", "inductive"} or fact_subtype == "semantic_fact" or (_is_structured_row(anchor) and fact_subtype == "value_fact"),
    )
    if companion is not None and not any(existing["chunk_id"] == companion["chunk_id"] for existing in selected):
        selected.append(companion)
    for row in doc_rows:
        if len(selected) >= max_rows:
            break
        if any(existing["chunk_id"] == row["chunk_id"] for existing in selected):
            continue
        if require_same_section and anchor.get("section_path") and anchor.get("section_path") != row.get("section_path"):
            continue
        selected.append(dict(row))
    return selected[:max_rows]


def _assign_evidence_ids(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assigned: list[dict[str, Any]] = []
    for index, row in enumerate(_unique_rows(rows), start=1):
        payload = dict(row)
        payload["evidence_id"] = f"E{index}"
        payload.setdefault("support_span", _row_text(row, prefer_exact=True))
        payload.setdefault("bundle_id", payload.get("parent_chunk_id") or payload.get("chunk_id"))
        payload.setdefault("bundle_rank", payload.get("bundle_rank") or 1)
        assigned.append(payload)
    return assigned


def _prepare_evidence(*, query: str, question_type: str, answer_mode: str, fact_subtype: str, query_domain_bucket: str, retrieval_result: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], str | None, bool]:
    del question_type
    numeric_query = is_numeric_or_table_query(query)
    doc_candidates = _build_doc_candidates(query, answer_mode, retrieval_result)
    doc_candidates, doc_guard_triggered = _apply_domain_guard(doc_candidates, query_domain_bucket=query_domain_bucket, answer_mode=answer_mode)
    if not doc_candidates:
        return [], [], [], "no_evidence", doc_guard_triggered

    if answer_mode == "report_lookup":
        primary = doc_candidates[0]
        selected_rows = _select_rows_for_doc(primary, answer_mode=answer_mode, fact_subtype=fact_subtype, numeric_query=numeric_query, max_rows=1)
        return _assign_evidence_ids(selected_rows), doc_candidates, [primary["doc_id"]], None, doc_guard_triggered

    if answer_mode == "numeric_fact":
        primary = doc_candidates[0]
        max_rows = 3 if fact_subtype == "value_fact" else 2
        selected_rows = _select_rows_for_doc(primary, answer_mode=answer_mode, fact_subtype=fact_subtype, numeric_query=numeric_query, max_rows=max_rows)
        return _assign_evidence_ids(selected_rows), doc_candidates, [primary["doc_id"]], None, doc_guard_triggered

    if answer_mode == "comparison":
        top_docs = doc_candidates[:2]
        if len(top_docs) < 2:
            return [], doc_candidates, [], "insufficient_source_diversity", doc_guard_triggered
        selected_rows: list[dict[str, Any]] = []
        for candidate in top_docs:
            selected_rows.extend(_select_rows_for_doc(candidate, answer_mode=answer_mode, fact_subtype=fact_subtype, numeric_query=numeric_query, max_rows=2))
        return _assign_evidence_ids(selected_rows), doc_candidates, [candidate["doc_id"] for candidate in top_docs], None, doc_guard_triggered

    top_docs = doc_candidates[:3]
    if len(top_docs) < 2:
        return [], doc_candidates, [], "insufficient_source_diversity", doc_guard_triggered
    selected_rows: list[dict[str, Any]] = []
    for candidate in top_docs:
        selected_rows.extend(_select_rows_for_doc(candidate, answer_mode=answer_mode, fact_subtype=fact_subtype, numeric_query=numeric_query, max_rows=1))
    return _assign_evidence_ids(selected_rows), doc_candidates, [candidate["doc_id"] for candidate in top_docs], None, doc_guard_triggered


def _finalize_used_evidence_ids(
    *,
    parsed_ids: list[str],
    selected_evidence: list[dict[str, Any]],
    question_type: str,
    answer_mode: str,
) -> list[str]:
    used_ids = select_used_evidence_ids(parsed_ids, selected_evidence, question_type)
    evidence_by_id = {row["evidence_id"]: row for row in selected_evidence}

    if answer_mode == "report_lookup":
        return used_ids[:1] or ([selected_evidence[0]["evidence_id"]] if selected_evidence else [])

    if answer_mode == "numeric_fact":
        if not selected_evidence:
            return []
        primary_doc_id = selected_evidence[0]["doc_id"]
        filtered = [evidence_id for evidence_id in used_ids if evidence_by_id[evidence_id]["doc_id"] == primary_doc_id]
        return filtered[:2] or [selected_evidence[0]["evidence_id"]]

    covered_docs = {evidence_by_id[evidence_id]["doc_id"] for evidence_id in used_ids if evidence_id in evidence_by_id}
    for row in selected_evidence:
        if row["evidence_id"] in used_ids:
            continue
        if row["doc_id"] in covered_docs:
            continue
        used_ids.append(row["evidence_id"])
        covered_docs.add(row["doc_id"])
        if len(covered_docs) >= 2:
            break
    return list(dict.fromkeys(used_ids))


def _evidence_line(row: dict[str, Any]) -> str:
    title = _short_report_title(row["file_name"])
    snippet = first_sentence(_row_text(row, prefer_exact=True), max_chars=72)
    return normalize_text(f"\u300a{title}\u300bP{row['page_start']} \u63d0\u5230 {snippet}")



def _default_evidence_summary(answer_mode: str, evidence_rows: list[dict[str, Any]]) -> str:
    if not evidence_rows:
        return ""
    if answer_mode in {"report_lookup", "numeric_fact"}:
        return normalize_text("; ".join(_evidence_line(row) for row in evidence_rows[:2]))

    by_doc: list[dict[str, Any]] = []
    seen_docs: set[str] = set()
    for row in evidence_rows:
        if row["doc_id"] in seen_docs:
            continue
        by_doc.append(row)
        seen_docs.add(row["doc_id"])
    limit = 2 if answer_mode == "comparison" else 3
    return normalize_text("; ".join(_evidence_line(row) for row in by_doc[:limit]))



def _common_terms_for_rows(evidence_rows: list[dict[str, Any]], *, min_doc_frequency: int = 2, top_k: int = 5) -> list[str]:
    rows_by_doc: dict[str, list[str]] = defaultdict(list)
    for row in evidence_rows:
        rows_by_doc[row["doc_id"]].append(_row_topic_hint(row))
        rows_by_doc[row["doc_id"]].append(first_sentence(_row_text(row, prefer_exact=True), max_chars=72))
    term_counter: Counter[str] = Counter()
    for texts in rows_by_doc.values():
        terms = set(extract_terms("\n".join(texts), top_k=8))
        for term in terms:
            if term in GENERIC_THEME_TERMS:
                continue
            term_counter[term] += 1
    common_terms = [term for term, count in term_counter.most_common() if count >= min_doc_frequency]
    return common_terms[:top_k]



def _fallback_payload(answer_mode: str, selected_evidence: list[dict[str, Any]], *, fact_subtype: str) -> dict[str, Any]:
    if not selected_evidence:
        return {
            "final_answer": FALLBACK_ANSWER,
            "evidence_summary": "",
            "uncertainty_note": "基于当前证据进行保守回答。",
            "used_evidence_ids": [],
        }

    if answer_mode == "report_lookup":
        title = _short_report_title(selected_evidence[0]["file_name"])
        return {
            "final_answer": title,
            "evidence_summary": _default_evidence_summary(answer_mode, selected_evidence),
            "uncertainty_note": "",
            "used_evidence_ids": [selected_evidence[0]["evidence_id"]],
        }

    if answer_mode == "numeric_fact":
        anchor = selected_evidence[0]
        answer_text = first_sentence(_row_text(anchor, prefer_exact=True), max_chars=96) or FALLBACK_ANSWER
        if fact_subtype == "semantic_fact":
            paragraph_rows = [row for row in selected_evidence if _is_paragraph_like(row)]
            if paragraph_rows:
                answer_text = first_sentence(_row_text(paragraph_rows[0], prefer_exact=True), max_chars=96) or answer_text
        return {
            "final_answer": answer_text,
            "evidence_summary": _default_evidence_summary(answer_mode, selected_evidence),
            "uncertainty_note": "",
            "used_evidence_ids": [row["evidence_id"] for row in selected_evidence[:2]],
        }

    by_doc: list[dict[str, Any]] = []
    seen_docs: set[str] = set()
    for row in selected_evidence:
        if row["doc_id"] in seen_docs:
            continue
        by_doc.append(row)
        seen_docs.add(row["doc_id"])

    if answer_mode == "comparison":
        lines = []
        for row in by_doc[:2]:
            topic = _row_topic_hint(row) or first_sentence(_row_text(row, prefer_exact=True), max_chars=64)
            lines.append(f"\u300a{_short_report_title(row['file_name'])}\u300b\u805a\u7126{topic}\u3002")
        return {
            "final_answer": normalize_text("".join(lines)) or FALLBACK_ANSWER,
            "evidence_summary": _default_evidence_summary(answer_mode, selected_evidence),
            "uncertainty_note": "",
            "used_evidence_ids": [row["evidence_id"] for row in by_doc[:2]],
        }

    common_terms = _common_terms_for_rows(selected_evidence)
    if len(common_terms) >= 3:
        common_text = f"{common_terms[0]}\u3001{common_terms[1]}\uff0c\u4ee5\u53ca{common_terms[2]}"
    elif len(common_terms) == 2:
        common_text = f"{common_terms[0]}\u3001{common_terms[1]}"
    elif common_terms:
        common_text = common_terms[0]
    else:
        common_text = "\u7a33\u5b9a\u4e3b\u9898"
    return {
        "final_answer": normalize_text(f"\u5171\u540c\u4e3b\u9898\u5305\u62ec{common_text}\u3002"),
        "evidence_summary": _default_evidence_summary(answer_mode, selected_evidence),
        "uncertainty_note": "",
        "used_evidence_ids": [row["evidence_id"] for row in by_doc[:3]],
    }


def _repair_answer_payload(
    *,
    answer_mode: str,
    fact_subtype: str,
    payload: dict[str, Any],
    selected_evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    fallback = _fallback_payload(answer_mode, selected_evidence, fact_subtype=fact_subtype)
    final_answer = normalize_text(payload.get("final_answer", "")) or fallback["final_answer"]
    evidence_summary = normalize_text(payload.get("evidence_summary", "")) or fallback["evidence_summary"]
    uncertainty_note = normalize_text(payload.get("uncertainty_note", ""))
    used_evidence_ids = payload.get("used_evidence_ids", [])

    if answer_mode in {"report_lookup", "inductive"}:
        return fallback

    if answer_mode == "comparison":
        topic_hints: list[str] = []
        for row in selected_evidence:
            hint = _row_topic_hint(row)
            if hint and hint not in topic_hints:
                topic_hints.append(hint)
        if topic_hints and not all(hint in final_answer for hint in topic_hints[:2]):
            final_answer = fallback["final_answer"]

    return {
        "final_answer": final_answer,
        "evidence_summary": evidence_summary,
        "uncertainty_note": uncertainty_note,
        "used_evidence_ids": used_evidence_ids,
    }


class LocalEvidenceAnswerer:
    def __init__(
        self,
        model_name: str,
        *,
        cache_dir: Path,
        device: str = "cuda",
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.8,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = ensure_model_downloaded(model_name, cache_dir)
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        self.model_name = model_name
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            dtype=torch_dtype if self.device.type == "cuda" else None,
        )
        self.model.to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def answer(
        self,
        *,
        query: str,
        question_type: str,
        retrieval_result: dict[str, Any],
    ) -> dict[str, Any]:
        question_type = question_type or infer_question_type(query)
        answer_mode = infer_answer_mode(query, question_type)
        fact_subtype = infer_fact_subtype(query, answer_mode)
        query_domain_bucket = infer_query_domain_bucket(query)
        numeric_query = is_numeric_or_table_query(query)
        selected_evidence, doc_candidates, selected_doc_ids, forced_abstain_reason, doc_guard_triggered = _prepare_evidence(
            query=query,
            question_type=question_type,
            answer_mode=answer_mode,
            fact_subtype=fact_subtype,
            query_domain_bucket=query_domain_bucket,
            retrieval_result=retrieval_result,
        )
        retrieval_scores = summarize_retrieval_scores(retrieval_result, selected_evidence)
        doc_candidate_summary = summarize_doc_candidates(doc_candidates)
        selected_domain_buckets = sorted({candidate.get("domain_bucket", "") for candidate in doc_candidates if candidate["doc_id"] in set(selected_doc_ids)})
        citation_rule_applied = "doc_level_report_lookup" if answer_mode == "report_lookup" else "default"

        if forced_abstain_reason is not None:
            return self._abstain_payload(
                query=query,
                question_type=question_type,
                answer_mode=answer_mode,
                fact_subtype=fact_subtype,
                selected_evidence=selected_evidence,
                retrieval_scores=retrieval_scores,
                doc_candidates=doc_candidate_summary,
                selected_doc_ids=selected_doc_ids,
                query_domain_bucket=query_domain_bucket,
                selected_domain_buckets=selected_domain_buckets,
                doc_guard_triggered=doc_guard_triggered,
                citation_rule_applied=citation_rule_applied,
                abstain_reason=forced_abstain_reason,
                generation_latency_ms=0.0,
                json_parse_failures=0,
            )

        generation_start = perf_counter()
        raw_model_output = ""
        json_parse_failures = 0
        parsed_payload: dict[str, Any] | None = None
        fallback_used = False
        fallback_source: str | None = None
        title_resolved_from = "short_title_from_file_name" if answer_mode == "report_lookup" else ""

        if answer_mode in {"report_lookup", "inductive"}:
            parsed_payload = _fallback_payload(answer_mode, selected_evidence, fact_subtype=fact_subtype)
            fallback_used = True
        else:
            for strict_json in (False, True):
                prompt = build_answer_prompt(
                    query=query,
                    question_type=question_type,
                    answer_mode=answer_mode,
                    evidence_rows=selected_evidence,
                    numeric_query=numeric_query,
                    strict_json=strict_json,
                )
                raw_model_output = self._generate(prompt)
                parsed_payload = parse_model_json(raw_model_output)
                if parsed_payload is not None:
                    break
                json_parse_failures += 1
            if parsed_payload is None:
                parsed_payload = _fallback_payload(answer_mode, selected_evidence, fact_subtype=fact_subtype)
                fallback_used = True
                fallback_source = "invalid_model_json"

        generation_latency_ms = round((perf_counter() - generation_start) * 1000, 2)
        payload = _repair_answer_payload(
            answer_mode=answer_mode,
            fact_subtype=fact_subtype,
            payload=parsed_payload or {},
            selected_evidence=selected_evidence,
        )
        used_evidence_ids = _finalize_used_evidence_ids(
            parsed_ids=payload.get("used_evidence_ids", []),
            selected_evidence=selected_evidence,
            question_type=question_type,
            answer_mode=answer_mode,
        )
        final_answer = payload.get("final_answer", FALLBACK_ANSWER)
        evidence_summary = payload.get("evidence_summary", "")
        uncertainty_note = payload.get("uncertainty_note", "")

        support_validation = validate_answer_support(
            query=query,
            question_type=question_type,
            final_answer=final_answer,
            evidence_summary=evidence_summary,
            used_evidence_ids=used_evidence_ids,
            evidence_rows=selected_evidence,
            answer_mode=answer_mode,
        )

        if answer_mode != "report_lookup" and not support_validation.get("supported", False):
            fallback_payload = _fallback_payload(answer_mode, selected_evidence, fact_subtype=fact_subtype)
            fallback_used = True
            fallback_source = "unsupported_answer"
            used_evidence_ids = _finalize_used_evidence_ids(
                parsed_ids=fallback_payload.get("used_evidence_ids", []),
                selected_evidence=selected_evidence,
                question_type=question_type,
                answer_mode=answer_mode,
            )
            final_answer = fallback_payload["final_answer"]
            evidence_summary = fallback_payload["evidence_summary"]
            uncertainty_note = fallback_payload["uncertainty_note"]
            support_validation = validate_answer_support(
                query=query,
                question_type=question_type,
                final_answer=final_answer,
                evidence_summary=evidence_summary,
                used_evidence_ids=used_evidence_ids,
                evidence_rows=selected_evidence,
                answer_mode=answer_mode,
            )

        effective_json_failures = 0 if fallback_used else json_parse_failures
        fallback_reason = _fallback_abstain_reason(final_answer=final_answer, fallback_source=fallback_source)
        referenced_rows = [row for row in selected_evidence if row["evidence_id"] in set(used_evidence_ids)] or selected_evidence
        abstained, abstain_reason = decide_abstention(
            question_type=question_type,
            evidence_rows=referenced_rows,
            retrieval_scores=retrieval_scores,
            support_validation=support_validation,
            json_parse_failures=effective_json_failures,
            answer_mode=answer_mode,
            fallback_reason=fallback_reason,
        )
        confidence_label = compute_confidence_label(
            retrieval_scores=retrieval_scores,
            support_validation=support_validation,
            abstained=abstained,
        )
        citations = build_citations(used_evidence_ids=used_evidence_ids, evidence_rows=selected_evidence) if not abstained else []
        return {
            "query": query,
            "question_type": question_type,
            "query_intent": answer_mode,
            "answer_mode": answer_mode,
            "fact_subtype": fact_subtype,
            "answer_source": "deterministic" if fallback_used or answer_mode in {"report_lookup", "inductive"} else "llm",
            "final_answer": FALLBACK_ANSWER if abstained else final_answer,
            "evidence_summary": "" if abstained else evidence_summary,
            "uncertainty_note": uncertainty_note,
            "used_evidence_ids": [] if abstained else used_evidence_ids,
            "abstained": abstained,
            "abstain_reason": abstain_reason,
            "abstain_gate": abstain_reason or "none",
            "confidence_label": confidence_label,
            "citations": citations,
            "retrieval_scores": retrieval_scores,
            "support_validation": support_validation,
            "matched_numeric_tokens": support_validation.get("matched_numeric_tokens", []),
            "title_resolved_from": title_resolved_from,
            "query_domain_bucket": query_domain_bucket,
            "selected_domain_buckets": selected_domain_buckets,
            "doc_guard_triggered": doc_guard_triggered,
            "citation_rule_applied": citation_rule_applied,
            "support_filter_applied": support_validation.get("support_filter_applied", "default"),
            "selected_evidence": summarize_selected_evidence(selected_evidence),
            "doc_candidates": doc_candidate_summary,
            "selected_doc_ids": selected_doc_ids,
            "raw_model_output": raw_model_output,
            "json_parse_failures": json_parse_failures,
            "generation_latency_ms": generation_latency_ms,
        }

    def _generate(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            rendered = f"{SYSTEM_PROMPT}\n\n{prompt}"

        inputs = self.tokenizer(rendered, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        prompt_length = inputs["input_ids"].shape[-1]
        output_ids = generated[0][prompt_length:]
        return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    def _abstain_payload(
        self,
        *,
        query: str,
        question_type: str,
        answer_mode: str,
        fact_subtype: str,
        selected_evidence: list[dict[str, Any]],
        retrieval_scores: dict[str, float],
        doc_candidates: list[dict[str, Any]],
        selected_doc_ids: list[str],
        query_domain_bucket: str,
        selected_domain_buckets: list[str],
        doc_guard_triggered: bool,
        citation_rule_applied: str,
        abstain_reason: str,
        generation_latency_ms: float,
        raw_model_output: str = "",
        json_parse_failures: int = 0,
    ) -> dict[str, Any]:
        support_validation = validate_answer_support(
            query=query,
            question_type=question_type,
            final_answer=FALLBACK_ANSWER,
            evidence_summary="",
            used_evidence_ids=[],
            evidence_rows=selected_evidence,
            answer_mode=answer_mode,
        )
        return {
            "query": query,
            "question_type": question_type,
            "query_intent": answer_mode,
            "answer_mode": answer_mode,
            "fact_subtype": fact_subtype,
            "answer_source": "deterministic",
            "final_answer": FALLBACK_ANSWER,
            "evidence_summary": "",
            "uncertainty_note": "基于当前证据进行保守回答。",
            "used_evidence_ids": [],
            "abstained": True,
            "abstain_reason": abstain_reason,
            "abstain_gate": abstain_reason,
            "confidence_label": "low",
            "citations": [],
            "retrieval_scores": retrieval_scores,
            "support_validation": support_validation,
            "matched_numeric_tokens": support_validation.get("matched_numeric_tokens", []),
            "title_resolved_from": "short_title_from_file_name" if answer_mode == "report_lookup" else "",
            "query_domain_bucket": query_domain_bucket,
            "selected_domain_buckets": selected_domain_buckets,
            "doc_guard_triggered": doc_guard_triggered,
            "citation_rule_applied": citation_rule_applied,
            "support_filter_applied": support_validation.get("support_filter_applied", "default"),
            "selected_evidence": summarize_selected_evidence(selected_evidence),
            "doc_candidates": doc_candidates,
            "selected_doc_ids": selected_doc_ids,
            "raw_model_output": raw_model_output,
            "json_parse_failures": json_parse_failures,
            "generation_latency_ms": generation_latency_ms,
        }


def summarize_retrieval_scores(retrieval_result: dict[str, Any], evidence_rows: list[dict[str, Any]]) -> dict[str, float]:
    dense_rows = retrieval_result.get("dense_rows", [])
    bm25_rows = retrieval_result.get("bm25_rows", [])
    hybrid_rows = retrieval_result.get("hybrid_rows", [])
    rerank_rows = retrieval_result.get("rerank_rows", [])
    return {
        "dense_top1": float(dense_rows[0]["score"]) if dense_rows else -999.0,
        "bm25_top1": float(bm25_rows[0]["score"]) if bm25_rows else -999.0,
        "hybrid_top1": float(hybrid_rows[0]["score"]) if hybrid_rows else -999.0,
        "rerank_top1": float(rerank_rows[0].get("rerank_score", rerank_rows[0].get("score", -999.0))) if rerank_rows else -999.0,
        "selected_evidence_count": float(len(evidence_rows)),
        "selected_source_count": float(len({row["doc_id"] for row in evidence_rows})),
    }


def summarize_selected_evidence(evidence_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in evidence_rows:
        rows.append(
            {
                "evidence_id": row["evidence_id"],
                "chunk_id": row["chunk_id"],
                "doc_id": row["doc_id"],
                "file_name": row["file_name"],
                "page_start": row["page_start"],
                "page_end": row["page_end"],
                "chunk_type": row.get("chunk_type"),
                "element_type": row.get("element_type"),
                "bundle_id": row.get("bundle_id"),
                "bundle_rank": row.get("bundle_rank"),
                "support_type": row.get("support_type"),
                "support_span": row.get("support_span"),
                "score": float(_row_score(row)),
                "text_snippet": _row_text(row, prefer_exact=True)[:220],
                "keywords": extract_terms(_row_text(row), top_k=5),
            }
        )
    return rows


def summarize_doc_candidates(doc_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for index, candidate in enumerate(doc_candidates[:5], start=1):
        summary.append(
            {
                "rank": index,
                "doc_id": candidate["doc_id"],
                "file_name": candidate["file_name"],
                "short_title": candidate["short_title"],
                "domain_bucket": candidate.get("domain_bucket", ""),
                "doc_score": round(float(candidate["doc_score"]), 4),
                "max_rerank_score": round(float(candidate["max_rerank_score"]), 4),
                "top2_chunk_score_sum": round(float(candidate["top2_chunk_score_sum"]), 4),
                "title_overlap_score": int(candidate["title_overlap_score"]),
                "structured_bonus": int(candidate["structured_bonus"]),
                "top_chunk_ids": [row["chunk_id"] for row in candidate.get("rows", [])[:3]],
            }
        )
    return summary


def parse_model_json(text: str) -> dict[str, Any] | None:
    payload = text.strip()
    if payload.startswith("```"):
        payload = payload.strip("`")
        if payload.lower().startswith("json"):
            payload = payload[4:].strip()
    match = re.search(r"\{.*\}", payload, re.DOTALL)
    if not match:
        return None
    try:
        loaded = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(loaded, dict):
        return None
    return loaded






