from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from src.generation.abstain import compute_confidence_label, decide_abstention, select_used_evidence_ids, validate_answer_support
from src.generation.citation_builder import build_citations
from src.generation.prompt import (
    SYSTEM_PROMPT,
    build_answer_prompt,
    build_comparison_focus_prompt,
    build_comparison_synthesis_prompt,
    build_inductive_observation_prompt,
    build_inductive_synthesis_prompt,
)
from src.retrieval.model_store import ensure_model_downloaded
from src.utils.text_utils import (
    extract_terms,
    first_sentence,
    industry_from_file_name,
    normalize_for_match,
    normalize_text,
    short_title_from_file_name,
    strip_file_extension,
    topic_hint_from_title,
)

QUESTION_TYPE_KEYWORDS = {
    "comparison": ("\u6bd4\u8f83", "\u5206\u522b", "\u5dee\u5f02", "\u5bf9\u6bd4"),
    "inductive": ("\u5171\u540c", "\u5171\u6027", "\u603b\u7ed3", "\u5f52\u7eb3", "\u5f3a\u8c03"),
}
SHARED_LOGIC_COMPARISON_MARKERS = (
    "\u5171\u540c\u5173\u6ce8",
    "\u5171\u540c\u903b\u8f91",
    "\u5171\u540c\u5f3a\u8c03",
    "\u90fd\u5728\u8ba8\u8bba",
    "\u90fd\u5728\u8c08",
    "\u90fd\u56f4\u7ed5",
    "\u90fd\u805a\u7126",
    "\u4e00\u81f4\u5173\u6ce8",
)
REPORT_LOOKUP_PATTERN = re.compile(
    r"(?:\u54ea(?:\u4efd|\u7bc7)?(?:[^\uff0c\u3002\uff01\uff1f\uff1b,]{0,24})?(?:\u62a5\u544a|\u7814\u62a5|\u5468\u62a5|\u6708\u62a5|\u767d\u76ae\u4e66|\u7b56\u7565(?:\u62a5\u544a)?|\u4e13\u9898(?:\u62a5\u544a)?|\u884c\u4e1a\u7814\u7a76|\u7814\u7a76(?:\u62a5\u544a)?|\u6d1e\u5bdf\u7b80\u62a5))"
)
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
    "semiconductor": ("\u534a\u5bfc\u4f53", "\u82af\u7247", "\u5c01\u6d4b", "\u6676\u5706", "\u5b58\u50a8", "\u6a21\u62df", "\u7b97\u529b", "\u6676\u5706\u4ee3\u5de5", "semicon", "gtc", "gpu", "token"),
    "new_energy": ("\u65b0\u80fd\u6e90", "\u5149\u4f0f", "\u9502\u7535", "\u50a8\u80fd", "\u65b0\u80fd\u6e90\u6c7d\u8f66", "\u52a8\u529b\u7535\u6c60", "\u56fa\u6001\u7535\u6c60", "\u78b3\u9178\u9502", "\u6c22\u6c27\u5316\u9502", "\u7279\u65af\u62c9"),
    "liquor": ("\u767d\u9152", "\u9152\u4f01", "\u8305\u53f0", "\u4e94\u7cae\u6db2", "\u98df\u54c1\u996e\u6599", "\u6625\u8282", "\u6279\u4ef7"),
    "consumer": ("\u6d88\u8d39", "\u5546\u793e", "\u7f8e\u62a4", "\u5bb6\u7535", "\u96f6\u552e", "\u5185\u9700", "\u73b0\u4ee3\u670d\u52a1\u4e1a", "\u5927\u4f17\u54c1", "\u5fc5\u9009\u6d88\u8d39\u54c1", "\u521b\u4e1a\u677f", "\u5a74\u914d\u7c89", "\u996e\u6599", "\u5546\u8d38"),
    "agriculture": ("\u519c\u6797\u7267\u6e14", "\u519c\u4e1a", "\u751f\u732a", "\u732a\u4ef7", "\u517b\u6b96", "\u79cd\u4e1a", "\u9972\u6599", "\u7267\u6e14", "\u725b\u4ef7", "\u4ed4\u732a", "\u767d\u7fbd\u9e21", "usda"),
    "healthcare": ("\u533b\u836f", "\u533b\u7597", "\u5065\u5eb7", "\u4e34\u5e8a", "\u521b\u65b0\u836f", "\u65b0\u836f", "\u75ab\u82d7", "\u547c\u5438\u9053", "\u98ce\u9669\u8bc4\u4f30", "\u65b0\u51a0", "\u80ba\u708e", "who", "iqvia", "car-t", "cagt"),
}
DOMAIN_COMPATIBILITY_GROUPS = {
    "consumer": {"consumer", "liquor"},
    "liquor": {"consumer", "liquor"},
}
TOPIC_NOISE_TERMS = {
    "\u677f\u5757",
    "\u884c\u4e1a",
    "\u5468\u62a5",
    "\u6708\u62a5",
    "\u53cc\u5468\u62a5",
    "\u62a5\u544a",
    "\u7814\u62a5",
    "\u767d\u76ae\u4e66",
    "\u4e13\u9898",
    "\u7b56\u7565",
    "\u7814\u7a76",
    "\u4e3b\u9898",
    "\u5efa\u8bae",
    "\u5173\u6ce8",
    "\u6570\u636e",
    "\u89c2\u70b9",
    "\u63a8\u8350",
    "\u5e02\u573a",
    "\u8868\u73b0",
    "\u8ddf\u8e2a",
    "\u672c\u5468",
    "\u6307\u6570",
    "\u7ae0\u8282\u8def\u5f84",
    "\u8bc1\u5238\u7814\u7a76\u62a5\u544a",
    "\u7814\u7a76\u62a5\u544a\u6b63\u6587",
    "\u6570\u636e\u4e2d\u5fc3",
    "\u4e1c\u65b9\u8d22\u5bcc\u7f51",
    "\u7535\u5b50",
    "\u534a\u5bfc\u4f53",
    "\u519c\u4e1a",
    "\u519c\u6797\u7267\u6e14",
    "\u533b\u836f",
    "\u533b\u7597",
    "\u5065\u5eb7",
    "\u65b0\u80fd\u6e90",
    "\u6d88\u8d39",
    "\u767d\u9152",
}
TOPIC_NOISE_PATTERNS = (
    "\u884c\u4e1a\u5468\u62a5",
    "\u884c\u4e1a\u53cc\u5468\u62a5",
    "\u884c\u4e1a\u6708\u62a5",
    "\u8bc1\u5238\u7814\u7a76\u62a5\u544a",
    "\u7814\u7a76\u62a5\u544a\u6b63\u6587",
    "\u6570\u636e\u4e2d\u5fc3",
    "\u4e1c\u65b9\u8d22\u5bcc\u7f51",
    "\u5e02\u573a\u8868\u73b0",
    "\u672c\u5468",
    "\u6307\u6570",
    "\u8bf7\u52a1\u5fc5\u9605\u8bfb",
    "\u98ce\u9669\u63d0\u793a",
    "\u6295\u8d44\u8bc4\u7ea7",
    "\u540c\u6b65\u5927\u5e02",
)
NOISY_SECTION_PATTERNS = (
    "\u76f8\u5173\u62a5\u544a",
    "\u62a5\u544a\u6c47\u603b",
    "\u56fe\u8868\u76ee\u5f55",
    "\u76ee\u5f55",
    "\u9644\u5f55",
    "\u98ce\u9669\u63d0\u793a",
    "\u514d\u8d23\u58f0\u660e",
    "\u6295\u8d44\u8bc4\u7ea7\u8bf4\u660e",
    "\u8bc4\u7ea7\u8bf4\u660e",
)
TOPIC_NUMERIC_RE = re.compile(r"^(?:20\d{2}(?:[-/]?\d{2})?(?:[-/]?\d{2})?|\d{6,8}|\d+(?:\.\d+)?%?)$")
TOPIC_DATE_RE = re.compile(r"^(?:20\d{2}[-/]?\d{2}[-/]?\d{2}|20\d{2}\u5e74\d{1,2}\u6708\d{1,2}\u65e5|\d{4}\u5e74|\d{1,2}\u6708|\d{1,2}\u65e5|\u7b2c\d+\u9875|\u7b2c\d+\u7248(?:\(\u82f1\u8bd1\u4e2d\))?)$")
QUERY_REPORT_TITLE_RE = re.compile(r"\u300a([^\u300b]{2,120})\u300b")
FALLBACK_ANSWER = "信息不足，暂时无法给出可靠答案。"

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
BROAD_INDUCTIVE_GENERIC_TERMS = {
    "近期",
    "最近",
    "最新",
    "共同",
    "共性",
    "主题",
    "报告",
    "研报",
    "总结",
    "趋势",
    "整体",
    "关注",
    "变化",
    "recent",
    "latest",
    "common",
    "shared",
    "themes",
    "theme",
    "reports",
    "report",
    "summary",
}
COARSE_DOMAIN_QUERY_TERMS = {
    "半导体",
    "电子",
    "农业",
    "医药",
    "医疗",
    "新能源",
    "光伏",
    "消费",
    "白酒",
    "储能",
    "agriculture",
    "semiconductor",
    "electronics",
    "healthcare",
    "medical",
    "consumer",
    "newenergy",
    "liquor",
}
ENGLISH_CLUSTER_NOISE_TERMS = {
    "weekly",
    "monthly",
    "report",
    "reports",
    "update",
    "updates",
    "industry",
    "sector",
    "tracker",
    "tracking",
    "overview",
    "coverage",
    "watch",
    "edition",
}
BROAD_INDUCTIVE_MARKER_PHRASES = (
    "近期研报",
    "最近报告",
    "共同主题",
    "共性主题",
    "共同关注",
    "整体趋势",
    "主要观点",
    "recent",
    "common themes",
    "shared themes",
)
CLUSTER_DATE_FRAGMENT_RE = re.compile(r"(?:20\d{2}(?:[-/]?\d{2})?(?:[-/]?\d{2})?|\d{6,8})")
CLUSTER_PERIOD_FRAGMENT_RE = re.compile(r"(?:\u7b2c)?\d+(?:\u5468|\u671f|\u7248)")
CLUSTER_REPORT_NOISE_RE = re.compile(
    r"(?:\u884c\u4e1a)?(?:\u53cc\u5468\u62a5|\u5468\u62a5|\u6708\u62a5|\u65e5\u62a5|\u5b63\u62a5|\u5e74\u62a5|\u4e13\u9898\u62a5\u544a|\u4e13\u9898|\u6df1\u5ea6\u62a5\u544a|\u7814\u7a76\u62a5\u544a|\u7814\u7a76|\u7814\u62a5|\u8ddf\u8e2a\u5468\u62a5|\u8ddf\u8e2a|\u89c2\u5bdf|\u70b9\u8bc4|\u7b56\u7565\u62a5\u544a|\u7b56\u7565|\u767d\u76ae\u4e66|\u7b80\u62a5)",
    re.IGNORECASE,
)
CLUSTER_ENGLISH_NOISE_RE = re.compile(r"\b(?:weekly|monthly|report|reports|update|updates|industry|sector|tracker|tracking|overview|coverage|watch|edition)\b", re.IGNORECASE)
SINGLE_LETTER_TOKEN_RE = re.compile(r"\b[a-zA-Z]\b")
REPORT_PERIOD_RE = re.compile(r"(?:_|-)(?:(?:20)?(\d{2})(\d{2})(\d{2})|(\d{4})-(\d{2})-(\d{2}))(?:_|-)?")
NORMALIZED_BROAD_INDUCTIVE_GENERIC_TERMS = {normalize_for_match(term) for term in BROAD_INDUCTIVE_GENERIC_TERMS}
NORMALIZED_COARSE_DOMAIN_QUERY_TERMS = {normalize_for_match(term) for term in COARSE_DOMAIN_QUERY_TERMS}
NORMALIZED_CLUSTER_NOISE_TERMS = {
    normalize_for_match(term)
    for term in {*TOPIC_NOISE_TERMS, *ENGLISH_CLUSTER_NOISE_TERMS, *BROAD_INDUCTIVE_GENERIC_TERMS}
}
NORMALIZED_FINE_GRAIN_QUERY_HINTS = {
    normalize_for_match(token)
    for tokens in QUERY_DOMAIN_HINTS.values()
    for token in tokens
    if normalize_for_match(token) and normalize_for_match(token) not in NORMALIZED_COARSE_DOMAIN_QUERY_TERMS
}
NORMALIZED_BROAD_INDUCTIVE_MARKER_PHRASES = {normalize_for_match(phrase) for phrase in BROAD_INDUCTIVE_MARKER_PHRASES}
GENERIC_NUMERIC_QUERY_TERMS = {
    "关键数据",
    "数据",
    "变化",
    "关键变化",
    "现货",
    "平均",
    "均价",
    "价格",
    "业绩",
    "食品",
    "饮料",
    "章节",
    "路径",
    "提升",
    "情况",
    "趋势",
    "report",
    "reports",
}
NORMALIZED_GENERIC_NUMERIC_QUERY_TERMS = {normalize_for_match(term) for term in GENERIC_NUMERIC_QUERY_TERMS}


def infer_question_type(query: str) -> str:
    normalized = normalize_text(query)
    for question_type, keywords in QUESTION_TYPE_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return question_type
    return "fact"



def infer_answer_mode(query: str, question_type: str | None = None) -> str:
    resolved_question_type = question_type or infer_question_type(query)
    normalized = normalize_text(query)
    if resolved_question_type == "comparison":
        return "comparison"
    if resolved_question_type == "inductive":
        return "inductive"
    if REPORT_LOOKUP_PATTERN.search(normalized):
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



def infer_query_domain_buckets(query: str) -> list[str]:
    normalized = normalize_text(query).lower()
    buckets: list[tuple[int, str]] = []
    for bucket, tokens in QUERY_DOMAIN_HINTS.items():
        hits = sum(token in normalized for token in tokens)
        if hits > 0:
            buckets.append((hits, bucket))
    buckets.sort(key=lambda item: (-item[0], item[1]))
    return [bucket for _, bucket in buckets]



def infer_query_domain_bucket(query: str) -> str:
    buckets = infer_query_domain_buckets(query)
    return buckets[0] if buckets else ""



def infer_comparison_subtype(query: str) -> str:
    normalized = normalize_text(query)
    if any(marker in normalized for marker in SHARED_LOGIC_COMPARISON_MARKERS):
        return "shared_logic_comparison"
    if "\u5171\u540c" in normalized and any(token in normalized for token in ("\u903b\u8f91", "\u4e3b\u7ebf", "\u5173\u6ce8", "\u8ba8\u8bba", "\u5f3a\u8c03")):
        return "shared_logic_comparison"
    return "contrast_comparison"



def _normalize_domain_bucket(bucket: str) -> str:
    return normalize_text(bucket).strip().lower()


def _is_weak_domain_bucket(bucket: str) -> bool:
    normalized = _normalize_domain_bucket(bucket)
    return not normalized or normalized == "other"


def _compatible_domain_buckets(bucket: str) -> set[str]:
    normalized = _normalize_domain_bucket(bucket)
    if not normalized:
        return set()
    return set(DOMAIN_COMPATIBILITY_GROUPS.get(normalized, {normalized}))


def _domains_compatible(left: str, right: str) -> bool:
    left_normalized = _normalize_domain_bucket(left)
    right_normalized = _normalize_domain_bucket(right)
    if not left_normalized or not right_normalized:
        return False
    if left_normalized == right_normalized:
        return True
    return (
        right_normalized in _compatible_domain_buckets(left_normalized)
        or left_normalized in _compatible_domain_buckets(right_normalized)
    )


def resolve_query_domain_buckets(query: str, *, domain_hint: str = "") -> list[str]:
    buckets = infer_query_domain_buckets(query)
    hint = _normalize_domain_bucket(domain_hint)
    if hint and not _is_weak_domain_bucket(hint):
        buckets = [bucket for bucket in buckets if bucket != hint]
        buckets.insert(0, hint)
    return buckets



def _meaningful_numeric_query_terms(query: str) -> list[str]:
    filtered_terms: list[str] = []
    for term in _query_terms(query, top_k=12):
        if (
            term in NORMALIZED_GENERIC_NUMERIC_QUERY_TERMS
            or term in NORMALIZED_BROAD_INDUCTIVE_GENERIC_TERMS
            or term in NORMALIZED_COARSE_DOMAIN_QUERY_TERMS
        ):
            continue
        filtered_terms.append(term)
    return filtered_terms


def _is_generic_numeric_semantic_query(query: str, *, fact_subtype: str, answer_mode: str) -> bool:
    if answer_mode != "numeric_fact" or fact_subtype != "semantic_fact":
        return False
    if _extract_query_report_titles(query):
        return False
    return len(_meaningful_numeric_query_terms(query)) <= 2



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


def _extract_query_report_titles(query: str) -> list[str]:
    titles: list[str] = []
    seen: set[str] = set()
    for title in QUERY_REPORT_TITLE_RE.findall(normalize_text(query)):
        cleaned = normalize_text(title).strip("\u300a\u300b\"' ")
        if not cleaned:
            continue
        normalized = normalize_for_match(cleaned)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        titles.append(cleaned)
    return titles


def _report_source_key(file_name: str) -> str:
    stem = strip_file_extension(Path(file_name).name)
    if "_" in stem:
        source = stem.split("_", 1)[0]
    elif "-" in stem:
        source = stem.split("-", 1)[0]
    else:
        source = stem
    return normalize_for_match(source)


def _title_topic_hint(file_name: str) -> str:
    return normalize_text(topic_hint_from_title(_short_report_title(file_name)))


def _is_noise_topic_text(text: str) -> bool:
    normalized = normalize_text(text)
    compact = normalize_for_match(normalized)
    if not compact:
        return True
    if TOPIC_NUMERIC_RE.fullmatch(normalized) or TOPIC_DATE_RE.fullmatch(normalized):
        return True
    lowered = normalized.lower()
    if any(pattern.lower() in lowered for pattern in TOPIC_NOISE_PATTERNS):
        return True
    if compact in {normalize_for_match(term) for term in TOPIC_NOISE_TERMS}:
        return True
    if normalized.endswith('?') and any(char.isdigit() for char in normalized):
        return True
    return False



def _is_noise_theme_term(term: str) -> bool:
    normalized = normalize_text(term)
    compact = normalize_for_match(normalized)
    if not compact or compact.isdigit():
        return True
    if TOPIC_NUMERIC_RE.fullmatch(normalized) or TOPIC_DATE_RE.fullmatch(normalized):
        return True
    if compact in {normalize_for_match(token) for token in TOPIC_NOISE_TERMS}:
        return True
    if normalized.lower().endswith('报告') or normalized.lower().endswith('研报'):
        return True
    return False


def _is_noise_evidence_row(row: dict[str, Any]) -> bool:
    candidates = [
        normalize_text(row.get("section_title", "") or ""),
        normalize_text(row.get("section_path", "") or ""),
        first_sentence(_row_text(row, prefer_exact=True), max_chars=64),
    ]
    normalized_candidates = [candidate for candidate in candidates if candidate]
    if not normalized_candidates:
        return False
    return any(any(pattern in candidate for pattern in NOISY_SECTION_PATTERNS) for candidate in normalized_candidates)


def _theme_terms(text: str, *, top_k: int = 8) -> list[str]:
    return [term for term in extract_terms(text, top_k=top_k) if not _is_noise_theme_term(term)]



def _row_topic_hint(row: dict[str, Any]) -> str:
    for candidate in (
        _title_topic_hint(row['file_name']),
        normalize_text(row.get('section_title', '') or ''),
        first_sentence(_row_text(row, prefer_exact=True), max_chars=64),
    ):
        if candidate and not _is_noise_topic_text(candidate):
            return candidate
    return ''



def _query_terms(query: str, *, top_k: int = 10) -> list[str]:
    return [normalize_for_match(term) for term in extract_terms(query, top_k=top_k) if normalize_for_match(term)]


def _cluster_stem_terms(text: str, *, top_k: int = 8) -> list[str]:
    normalized = normalize_text(text)
    normalized = CLUSTER_DATE_FRAGMENT_RE.sub(" ", normalized)
    normalized = CLUSTER_PERIOD_FRAGMENT_RE.sub(" ", normalized)
    normalized = CLUSTER_REPORT_NOISE_RE.sub(" ", normalized)
    normalized = CLUSTER_ENGLISH_NOISE_RE.sub(" ", normalized)
    normalized = SINGLE_LETTER_TOKEN_RE.sub(" ", normalized)
    normalized = re.sub(r"[_\-\s,:;./()]+", " ", normalized)
    terms: list[str] = []
    seen: set[str] = set()
    for term in extract_terms(normalized, top_k=top_k):
        key = normalize_for_match(term)
        if not key or len(key) < 2:
            continue
        if key in seen or key in NORMALIZED_CLUSTER_NOISE_TERMS:
            continue
        if _is_noise_theme_term(term):
            continue
        seen.add(key)
        terms.append(key)
    return terms


def clean_report_stem(text: str) -> str:
    terms = _cluster_stem_terms(text)
    if terms:
        return "-".join(terms[:3])
    cleaned = CLUSTER_REPORT_NOISE_RE.sub(" ", CLUSTER_DATE_FRAGMENT_RE.sub(" ", normalize_text(text)))
    return normalize_for_match(cleaned)


def clean_topic_stem(text: str) -> str:
    terms = _cluster_stem_terms(text)
    if terms:
        return "-".join(terms[:2])
    cleaned = CLUSTER_ENGLISH_NOISE_RE.sub(" ", CLUSTER_DATE_FRAGMENT_RE.sub(" ", normalize_text(text)))
    return normalize_for_match(cleaned)


def has_strong_entity_terms(query: str) -> bool:
    if _extract_query_report_titles(query):
        return True
    normalized_query = normalize_for_match(query)
    for token in NORMALIZED_FINE_GRAIN_QUERY_HINTS:
        if len(token) >= 2 and token in normalized_query:
            return True
    for term in _query_terms(query, top_k=12):
        if term in NORMALIZED_BROAD_INDUCTIVE_GENERIC_TERMS or term in NORMALIZED_COARSE_DOMAIN_QUERY_TERMS:
            continue
        if len(term) >= 2:
            return True
    return False


def is_broad_inductive_query(query: str) -> bool:
    if _extract_query_report_titles(query):
        return False
    query_terms = _query_terms(query, top_k=12)
    if not query_terms:
        return False
    normalized_query = normalize_for_match(query)
    has_marker_phrase = any(marker in normalized_query for marker in NORMALIZED_BROAD_INDUCTIVE_MARKER_PHRASES)
    generic_or_coarse_count = sum(
        term in NORMALIZED_BROAD_INDUCTIVE_GENERIC_TERMS or term in NORMALIZED_COARSE_DOMAIN_QUERY_TERMS
        for term in query_terms
    )
    if not has_marker_phrase and generic_or_coarse_count < max(2, len(query_terms) - 1):
        return False
    return not has_strong_entity_terms(query)


def _title_overlap_score(query: str, row_or_title: dict[str, Any] | str) -> int:
    query_terms = _query_terms(query)
    if not query_terms:
        return 0
    if isinstance(row_or_title, dict):
        file_name = row_or_title.get('file_name', '')
        section_title = row_or_title.get('section_title', '') or ''
    else:
        file_name = row_or_title
        section_title = ''
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


def _ordered_pool(
    retrieval_result: dict[str, Any],
    *,
    extra_hybrid: int = 12,
    include_dense_bm25: bool = False,
) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    seen_chunk_ids: set[str] = set()
    candidate_rows = list(retrieval_result.get("rerank_rows", [])) + list(retrieval_result.get("hybrid_rows", []))[:extra_hybrid]
    if include_dense_bm25:
        candidate_rows.extend(list(retrieval_result.get("bm25_rows", []))[:extra_hybrid])
        candidate_rows.extend(list(retrieval_result.get("dense_rows", []))[:extra_hybrid])
    for row in candidate_rows:
        if row["chunk_id"] in seen_chunk_ids:
            continue
        ordered.append(dict(row))
        seen_chunk_ids.add(row["chunk_id"])
    return ordered


def _build_doc_candidates(query: str, answer_mode: str, retrieval_result: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    query_titles = _extract_query_report_titles(query)
    extra_hybrid = 20 if answer_mode == "comparison" and query_titles else 12
    include_dense_bm25 = bool(query_titles)
    for row in _ordered_pool(
        retrieval_result,
        extra_hybrid=extra_hybrid,
        include_dense_bm25=include_dense_bm25,
    ):
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
        candidate = {
            "doc_id": doc_id,
            "file_name": ordered_rows[0]["file_name"],
            "short_title": _short_report_title(ordered_rows[0]["file_name"]),
            "source_key": _report_source_key(ordered_rows[0]["file_name"]),
            "topic_key": normalize_for_match(_title_topic_hint(ordered_rows[0]["file_name"])),
            "domain_bucket": industry_from_file_name(ordered_rows[0]["file_name"]),
            "doc_score": float(doc_score),
            "max_rerank_score": float(max_rerank_score),
            "top2_chunk_score_sum": float(top2_chunk_score_sum),
            "title_overlap_score": int(title_overlap),
            "structured_bonus": int(structured_bonus),
            "rows": ordered_rows,
        }
        candidate["cleaned_title_stem"] = clean_report_stem(candidate["short_title"])
        candidate["cleaned_topic_stem"] = clean_topic_stem(_title_topic_hint(candidate["file_name"]))
        candidate["cluster_key"] = cluster_key_for_candidate(candidate)
        candidates.append(candidate)

    if answer_mode == "report_lookup":
        candidates.sort(key=lambda item: (item["title_overlap_score"], item["doc_score"], item["max_rerank_score"]), reverse=True)
    else:
        candidates.sort(key=lambda item: (item["doc_score"], item["title_overlap_score"], item["max_rerank_score"]), reverse=True)
    return candidates


def _apply_domain_guard(doc_candidates: list[dict[str, Any]], *, query_domain_bucket: str, answer_mode: str) -> tuple[list[dict[str, Any]], bool]:
    if _is_weak_domain_bucket(query_domain_bucket):
        return doc_candidates, False
    exact_bucket_candidates = [
        candidate
        for candidate in doc_candidates
        if _normalize_domain_bucket(candidate.get("domain_bucket", "")) == _normalize_domain_bucket(query_domain_bucket)
    ]
    compatible_bucket_candidates = [
        candidate
        for candidate in doc_candidates
        if _domains_compatible(candidate.get("domain_bucket", ""), query_domain_bucket)
    ]
    if not compatible_bucket_candidates:
        return doc_candidates, False

    if answer_mode in {"comparison", "inductive"}:
        if len(exact_bucket_candidates) >= 2:
            return exact_bucket_candidates, len(exact_bucket_candidates) != len(doc_candidates)
        if len(compatible_bucket_candidates) >= 2:
            return compatible_bucket_candidates, len(compatible_bucket_candidates) != len(doc_candidates)
        return doc_candidates, False

    top_candidate = doc_candidates[0]
    if _normalize_domain_bucket(top_candidate.get("domain_bucket", "")) == _normalize_domain_bucket(query_domain_bucket):
        return doc_candidates, False

    if exact_bucket_candidates:
        guarded_candidate = exact_bucket_candidates[0]
    elif compatible_bucket_candidates:
        guarded_candidate = compatible_bucket_candidates[0]
    else:
        return doc_candidates, False
    top_score = float(top_candidate.get("doc_score", 0.0))
    guarded_score = float(guarded_candidate.get("doc_score", 0.0))
    score_gap_ratio = (top_score - guarded_score) / max(abs(top_score), 1e-6)
    if guarded_candidate.get("title_overlap_score", 0) > 0 or score_gap_ratio <= 0.2:
        preferred_candidates = exact_bucket_candidates or compatible_bucket_candidates
        reordered = preferred_candidates + [
            candidate
            for candidate in doc_candidates
            if candidate not in preferred_candidates
        ]
        return reordered, True
    return doc_candidates, False


def _select_doc_anchor(doc_rows: list[dict[str, Any]], *, query: str = "", answer_mode: str, fact_subtype: str, numeric_query: bool) -> dict[str, Any] | None:
    paragraph_rows = [row for row in doc_rows if _is_paragraph_like(row)]
    prefer_paragraph = answer_mode in {"report_lookup", "comparison", "inductive"} or fact_subtype == "semantic_fact"
    candidate_rows = paragraph_rows if prefer_paragraph and paragraph_rows else doc_rows
    generic_numeric_semantic = _is_generic_numeric_semantic_query(
        query,
        fact_subtype=fact_subtype,
        answer_mode=answer_mode,
    )

    best_row: dict[str, Any] | None = None
    best_score = float('-inf')
    for row in candidate_rows:
        score = _row_score(row)
        page_start = int(row.get('page_start', 1))
        is_structured = _is_structured_row(row)
        is_paragraph_like = _is_paragraph_like(row)
        topic_hint = _row_topic_hint(row)
        noisy_topic = not topic_hint or _is_noise_topic_text(topic_hint)
        if _is_noise_evidence_row(row):
            score -= 1.2
        if not noisy_topic:
            if page_start <= 2:
                score += 0.18
            elif page_start <= 4:
                score += 0.08
        else:
            score -= 0.9 if answer_mode in {"report_lookup", "comparison", "inductive"} else 0.2
        if row.get('support_span'):
            score += 0.03
        if row.get('section_path'):
            score += 0.02
        if generic_numeric_semantic:
            if is_paragraph_like:
                score += 0.65
            if is_structured:
                score -= 1.25
            if page_start <= 3 and not noisy_topic:
                score += 0.45
            if page_start > 5:
                score -= 0.2
        if answer_mode == 'report_lookup':
            if is_paragraph_like:
                score += 0.55
            if is_structured:
                score -= 0.5
            if page_start <= 2 and not noisy_topic:
                score += 0.4
        elif answer_mode == 'comparison':
            if is_paragraph_like:
                score += 0.45
            if is_structured:
                score -= 0.2
        elif answer_mode == 'inductive':
            if is_paragraph_like:
                score += 0.6
            if is_structured:
                score -= 0.35
        elif fact_subtype == 'semantic_fact':
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
    candidate_rows = [row for row in candidates if row['chunk_id'] != anchor['chunk_id'] and row['doc_id'] == anchor['doc_id']]
    if prefer_paragraph:
        paragraph_rows = [row for row in candidate_rows if _is_paragraph_like(row)]
        if paragraph_rows:
            candidate_rows = paragraph_rows

    best_row: dict[str, Any] | None = None
    best_score = float('-inf')
    for candidate in candidate_rows:
        if require_same_section and anchor.get('section_path') and anchor.get('section_path') != candidate.get('section_path'):
            continue
        score = _row_score(candidate)
        if _is_noise_evidence_row(candidate):
            score -= 1.0
        if _same_context(anchor, candidate):
            score += 0.35
        if anchor.get('section_path') and anchor.get('section_path') == candidate.get('section_path'):
            score += 0.15
        if _is_structured_row(anchor) and _is_paragraph_like(candidate):
            score += 0.25
        if numeric_query and _is_structured_row(candidate):
            score += 0.08
        if not _row_topic_hint(candidate):
            score -= 0.5
        if score > best_score:
            best_score = score
            best_row = dict(candidate)
    return best_row



def _select_rows_for_doc(doc_candidate: dict[str, Any], *, query: str = "", answer_mode: str, fact_subtype: str, numeric_query: bool, max_rows: int = 2, require_same_section: bool = False) -> list[dict[str, Any]]:
    doc_rows = list(doc_candidate.get("rows", []))
    anchor = _select_doc_anchor(
        doc_rows,
        query=query,
        answer_mode=answer_mode,
        fact_subtype=fact_subtype,
        numeric_query=numeric_query,
    )
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


def _title_page_candidates(doc_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    preferred = [
        dict(row)
        for row in doc_rows
        if int(row.get("page_start", 1)) <= 2 and not _is_noise_evidence_row(row)
    ]
    if preferred:
        return preferred
    return [dict(row) for row in doc_rows if not _is_noise_evidence_row(row)] or [dict(row) for row in doc_rows]


def _candidate_title_key(candidate: dict[str, Any]) -> str:
    return normalize_for_match(candidate.get("short_title") or _short_report_title(candidate.get("file_name", "")))


def _candidate_source_key(candidate: dict[str, Any]) -> str:
    source_key = normalize_for_match(str(candidate.get("source_key", "")))
    if source_key:
        return source_key
    return _report_source_key(candidate.get("file_name", ""))


def _candidate_topic_key(candidate: dict[str, Any]) -> str:
    topic_key = normalize_for_match(str(candidate.get("topic_key", "")))
    if topic_key:
        return topic_key
    return normalize_for_match(_title_topic_hint(candidate.get("file_name", "")))


def _candidate_diversity_key(candidate: dict[str, Any]) -> str:
    topic_key = normalize_for_match(str(candidate.get("cleaned_topic_stem", ""))) or _candidate_topic_key(candidate)
    if topic_key:
        return topic_key
    cluster_key = _candidate_cluster_key(candidate)
    if cluster_key:
        return cluster_key
    return _candidate_title_key(candidate)


def _candidate_topic_family_key(candidate: dict[str, Any]) -> str:
    normalized_text = normalize_text(
        "\n".join(
            [
                str(candidate.get("short_title", "") or ""),
                str(candidate.get("file_name", "") or ""),
                _title_topic_hint(str(candidate.get("file_name", "") or "")),
            ]
        )
    ).lower()
    if any(token in normalized_text for token in ("\u732a\u4ef7", "\u751f\u732a", "\u4ed4\u732a", "\u517b\u6b96", "\u53bb\u4ea7\u80fd", "hog", "pig", "destocking")):
        return "agriculture:hog"
    if any(token in normalized_text for token in ("\u79cd\u4e1a", "\u79cd\u690d", "\u80b2\u79cd", "\u632f\u5174", "seed", "breeding")):
        return "agriculture:seed"
    if any(token in normalized_text for token in ("\u539f\u6cb9", "\u519c\u4ea7\u54c1", "\u725b\u4ef7", "\u8089\u725b", "oil", "commodity", "cattle")):
        return "agriculture:commodity"
    if any(token in normalized_text for token in ("\u6676\u5706", "\u4ee3\u5de5", "\u6676\u5706\u4ee3\u5de5")):
        return "semiconductor:foundry"
    if any(token in normalized_text for token in ("\u5b58\u50a8", "dram", "nand")):
        return "semiconductor:memory"
    if any(token in normalized_text for token in ("ai", "gpu", "\u7b97\u529b", "\u6db2\u51b7")):
        return "semiconductor:compute"
    return _candidate_diversity_key(candidate)


def cluster_key_for_candidate(candidate: dict[str, Any]) -> str:
    source_key = _candidate_source_key(candidate) or "unknown"
    title_stem = normalize_for_match(str(candidate.get("cleaned_title_stem", ""))) or clean_report_stem(
        candidate.get("short_title") or _short_report_title(candidate.get("file_name", ""))
    )
    topic_stem = normalize_for_match(str(candidate.get("cleaned_topic_stem", ""))) or clean_topic_stem(
        candidate.get("topic_key") or _title_topic_hint(candidate.get("file_name", "")) or candidate.get("short_title", "")
    )
    if not topic_stem:
        topic_stem = title_stem
    if not title_stem:
        title_stem = topic_stem
    return ":".join(part for part in (source_key, title_stem, topic_stem) if part)


def _candidate_cluster_key(candidate: dict[str, Any]) -> str:
    cluster_key = normalize_for_match(str(candidate.get("cluster_key", "")))
    if cluster_key:
        return cluster_key
    return normalize_for_match(cluster_key_for_candidate(candidate))


def _candidate_title_match_score(candidate: dict[str, Any], target_title: str) -> int:
    normalized_target = normalize_for_match(target_title)
    if not normalized_target:
        return 0
    candidate_text = normalize_for_match(
        "\n".join(
            [
                candidate.get("short_title", "") or _short_report_title(candidate.get("file_name", "")),
                candidate.get("file_name", ""),
            ]
        )
    )
    if normalized_target in candidate_text:
        return 100
    target_terms = [normalize_for_match(term) for term in extract_terms(target_title, top_k=6) if normalize_for_match(term)]
    return sum(term in candidate_text for term in target_terms)


def _candidate_title_match_sort_key(candidate: dict[str, Any], target_title: str) -> tuple[int, int, int, float, float]:
    normalized_target = normalize_for_match(target_title)
    if not normalized_target:
        return (0, 0, 0, float(candidate.get("doc_score", 0.0)), float(candidate.get("max_rerank_score", 0.0)))
    short_title_key = _candidate_title_key(candidate)
    exact_short_title_containment = 0
    if short_title_key:
        if short_title_key == normalized_target:
            exact_short_title_containment = 2
        elif normalized_target in short_title_key or short_title_key in normalized_target:
            exact_short_title_containment = 1
    candidate_text = normalize_for_match(
        "\n".join(
            [
                candidate.get("short_title", "") or _short_report_title(candidate.get("file_name", "")),
                candidate.get("file_name", ""),
            ]
        )
    )
    normalized_title_containment = int(bool(normalized_target and normalized_target in candidate_text))
    title_term_overlap = _candidate_title_match_score(candidate, target_title)
    return (
        exact_short_title_containment,
        normalized_title_containment,
        title_term_overlap,
        float(candidate.get("doc_score", 0.0)),
        float(candidate.get("max_rerank_score", 0.0)),
    )


def _iter_candidate_pool(
    doc_candidates: list[dict[str, Any]],
    *,
    pre_guard_candidates: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    seen_doc_ids: set[str] = set()
    for pool in (doc_candidates, pre_guard_candidates or []):
        for candidate in pool:
            doc_id = candidate["doc_id"]
            if doc_id in seen_doc_ids:
                continue
            ordered.append(candidate)
            seen_doc_ids.add(doc_id)
    return ordered


def _select_diverse_doc_candidates(
    doc_candidates: list[dict[str, Any]],
    *,
    max_docs: int,
    pre_guard_candidates: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    pooled_candidates = _iter_candidate_pool(doc_candidates, pre_guard_candidates=pre_guard_candidates)
    if not pooled_candidates:
        return []

    selected: list[dict[str, Any]] = []
    selected_doc_ids: set[str] = set()
    selected_title_keys: set[str] = set()

    def try_add(candidate: dict[str, Any], *, require_new_title: bool) -> bool:
        doc_id = candidate["doc_id"]
        title_key = _candidate_title_key(candidate)
        if doc_id in selected_doc_ids:
            return False
        if require_new_title and title_key and title_key in selected_title_keys:
            return False
        selected.append(candidate)
        selected_doc_ids.add(doc_id)
        if title_key:
            selected_title_keys.add(title_key)
        return True

    try_add(pooled_candidates[0], require_new_title=False)
    for require_new_title in (True, False):
        for candidate in pooled_candidates[1:]:
            if len(selected) >= max_docs:
                return selected
            try_add(candidate, require_new_title=require_new_title)
    return selected


def _select_comparison_doc_candidates(
    doc_candidates: list[dict[str, Any]],
    *,
    query: str,
    max_docs: int,
    pre_guard_candidates: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    pooled_candidates = _iter_candidate_pool(doc_candidates, pre_guard_candidates=pre_guard_candidates)
    query_titles = _extract_query_report_titles(query)
    if len(query_titles) >= 2:
        selected: list[dict[str, Any]] = []
        selected_doc_ids: set[str] = set()
        for title in query_titles[:max_docs]:
            best_candidate: dict[str, Any] | None = None
            best_sort_key: tuple[int, int, int, float, float] | None = None
            for candidate in pooled_candidates:
                if candidate["doc_id"] in selected_doc_ids:
                    continue
                sort_key = _candidate_title_match_sort_key(candidate, title)
                if sort_key[:3] == (0, 0, 0):
                    continue
                if best_sort_key is None or sort_key > best_sort_key:
                    best_candidate = candidate
                    best_sort_key = sort_key
            if best_candidate is None:
                break
            selected.append(best_candidate)
            selected_doc_ids.add(best_candidate["doc_id"])
        if len(selected) >= min(max_docs, len(query_titles[:max_docs])):
            return selected[:max_docs]

    return _select_diverse_doc_candidates(
        doc_candidates,
        max_docs=max_docs,
        pre_guard_candidates=pre_guard_candidates,
    )


def _candidate_domain_bucket(candidate: dict[str, Any]) -> str:
    return normalize_text(str(candidate.get("domain_bucket", "")))


def _pick_ranked_candidate(
    pool: list[dict[str, Any]],
    *,
    selected_doc_ids: set[str],
    require_new_cluster: bool = False,
    require_new_source: bool = False,
    require_new_topic: bool = False,
    selected_cluster_keys: set[str] | None = None,
    selected_source_keys: set[str] | None = None,
    selected_topic_keys: set[str] | None = None,
) -> dict[str, Any] | None:
    selected_cluster_keys = selected_cluster_keys or set()
    selected_source_keys = selected_source_keys or set()
    selected_topic_keys = selected_topic_keys or set()
    for candidate in pool:
        if candidate["doc_id"] in selected_doc_ids:
            continue
        cluster_new = bool(_candidate_cluster_key(candidate) and _candidate_cluster_key(candidate) not in selected_cluster_keys)
        source_new = bool(_candidate_source_key(candidate) and _candidate_source_key(candidate) not in selected_source_keys)
        topic_new = bool(_candidate_topic_key(candidate) and _candidate_topic_key(candidate) not in selected_topic_keys)
        if require_new_cluster and not cluster_new:
            continue
        if require_new_source and not source_new:
            continue
        if require_new_topic and not topic_new:
            continue
        return candidate
    return None


def _append_candidate_selection(
    selected: list[dict[str, Any]],
    candidate: dict[str, Any],
    *,
    selected_doc_ids: set[str],
    selected_cluster_keys: set[str],
    selected_source_keys: set[str],
    selected_topic_keys: set[str],
) -> None:
    selected.append(candidate)
    selected_doc_ids.add(candidate["doc_id"])
    cluster_key = _candidate_cluster_key(candidate)
    source_key = _candidate_source_key(candidate)
    topic_key = _candidate_topic_key(candidate)
    if cluster_key:
        selected_cluster_keys.add(cluster_key)
    if source_key:
        selected_source_keys.add(source_key)
    if topic_key:
        selected_topic_keys.add(topic_key)



def _select_broad_inductive_doc_candidates(
    doc_candidates: list[dict[str, Any]],
    *,
    query_domain_bucket: str,
    max_docs: int,
    pre_guard_candidates: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not doc_candidates:
        return []

    primary_domain_bucket = query_domain_bucket or _candidate_domain_bucket(doc_candidates[0])
    same_domain_pool = [candidate for candidate in doc_candidates if not primary_domain_bucket or _candidate_domain_bucket(candidate) == primary_domain_bucket]
    primary_pool = same_domain_pool or list(doc_candidates)

    selected: list[dict[str, Any]] = []
    selected_doc_ids: set[str] = set()
    selected_cluster_keys: set[str] = set()
    selected_source_keys: set[str] = set()
    selected_topic_keys: set[str] = set()
    selected_diversity_keys: set[str] = set()

    def add_candidate(candidate: dict[str, Any]) -> None:
        _append_candidate_selection(
            selected,
            candidate,
            selected_doc_ids=selected_doc_ids,
            selected_cluster_keys=selected_cluster_keys,
            selected_source_keys=selected_source_keys,
            selected_topic_keys=selected_topic_keys,
        )
        diversity_key = _candidate_topic_family_key(candidate)
        if diversity_key:
            selected_diversity_keys.add(diversity_key)

    add_candidate(primary_pool[0])

    ranked_distinct_candidates: list[dict[str, Any]] = []
    seen_diversity_keys: set[str] = set()
    for candidate in primary_pool:
        diversity_key = _candidate_topic_family_key(candidate)
        if diversity_key and diversity_key in seen_diversity_keys:
            continue
        if diversity_key:
            seen_diversity_keys.add(diversity_key)
        ranked_distinct_candidates.append(candidate)

    for candidate in ranked_distinct_candidates[1:]:
        if len(selected) >= max_docs:
            break
        diversity_key = _candidate_topic_family_key(candidate)
        if diversity_key and diversity_key in selected_diversity_keys:
            continue
        add_candidate(candidate)

    for stage in (
        {"require_new_cluster": True},
        {"require_new_source": True},
        {"require_new_topic": True},
        {},
    ):
        if len(selected) >= max_docs:
            break
        candidate = _pick_ranked_candidate(
            primary_pool,
            selected_doc_ids=selected_doc_ids,
            selected_cluster_keys=selected_cluster_keys,
            selected_source_keys=selected_source_keys,
            selected_topic_keys=selected_topic_keys,
            **stage,
        )
        if candidate is None:
            continue
        diversity_key = _candidate_topic_family_key(candidate)
        if diversity_key and diversity_key in selected_diversity_keys and len(ranked_distinct_candidates) > len(selected_diversity_keys):
            continue
        add_candidate(candidate)

    if len(selected) < max_docs and not query_domain_bucket and pre_guard_candidates:
        pre_guard_pool = _iter_candidate_pool(pre_guard_candidates)
        for stage in (
            {"require_new_cluster": True},
            {"require_new_source": True},
            {"require_new_topic": True},
            {},
        ):
            if len(selected) >= max_docs:
                break
            candidate = _pick_ranked_candidate(
                pre_guard_pool,
                selected_doc_ids=selected_doc_ids,
                selected_cluster_keys=selected_cluster_keys,
                selected_source_keys=selected_source_keys,
                selected_topic_keys=selected_topic_keys,
                **stage,
            )
            if candidate is None:
                continue
            diversity_key = _candidate_topic_family_key(candidate)
            if diversity_key and diversity_key in selected_diversity_keys and len(ranked_distinct_candidates) > len(selected_diversity_keys):
                continue
            add_candidate(candidate)

    return selected[:max_docs]



def _select_conservative_inductive_doc_candidates(
    doc_candidates: list[dict[str, Any]],
    *,
    query_domain_bucket: str,
    max_docs: int,
) -> list[dict[str, Any]]:
    pool = [candidate for candidate in doc_candidates if not query_domain_bucket or _candidate_domain_bucket(candidate) == query_domain_bucket]
    if not pool:
        pool = list(doc_candidates)
    return pool[:max_docs]



def _select_inductive_doc_candidates(
    doc_candidates: list[dict[str, Any]],
    *,
    query: str,
    query_domain_bucket: str,
    max_docs: int,
    pre_guard_candidates: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not doc_candidates:
        return []
    if is_broad_inductive_query(query):
        return _select_broad_inductive_doc_candidates(
            doc_candidates,
            query_domain_bucket=query_domain_bucket,
            max_docs=max_docs,
            pre_guard_candidates=pre_guard_candidates,
        )
    return _select_conservative_inductive_doc_candidates(
        doc_candidates,
        query_domain_bucket=query_domain_bucket,
        max_docs=max_docs,
    )


def _row_query_overlap(query: str, row: dict[str, Any]) -> int:
    query_terms = _query_terms(query)
    if not query_terms:
        return 0
    match_text = _row_match_text(row)
    return sum(term in match_text for term in query_terms)


def _context_overlap_score(context_text: str, row: dict[str, Any]) -> int:
    context_terms = [normalize_for_match(term) for term in extract_terms(context_text, top_k=10) if normalize_for_match(term)]
    if not context_terms:
        return 0
    match_text = _row_match_text(row)
    return sum(term in match_text for term in context_terms)


def _best_support_row(
    query: str,
    anchor: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    numeric_query: bool,
    answer_mode: str,
    context_text: str = "",
) -> dict[str, Any] | None:
    best_row: dict[str, Any] | None = None
    best_sort_key: tuple[int, ...] | None = None
    anchor_sentence = normalize_for_match(first_sentence(_row_text(anchor, prefer_exact=True), max_chars=160))
    anchor_page = int(anchor.get("page_start", 1))
    for candidate in candidates:
        if candidate["chunk_id"] == anchor["chunk_id"]:
            continue
        same_section = int(bool(anchor.get("section_path") and anchor.get("section_path") == candidate.get("section_path")))
        same_page = int(anchor_page == int(candidate.get("page_start", 1)))
        paragraph_like = int(_is_paragraph_like(candidate))
        non_noise = int(not _is_noise_evidence_row(candidate))
        context_overlap = _context_overlap_score(context_text, candidate)
        deeper_page = int(int(candidate.get("page_start", 1)) > 2)
        structured_ok = int(not _is_structured_row(candidate) or numeric_query)
        generic_title_adjacent = int(
            int(candidate.get("page_start", 1)) <= 2
            and context_overlap == 0
            and (_is_noise_topic_text(_row_topic_hint(candidate)) or not paragraph_like)
        )
        base_score = int(_row_score(candidate) * 1000)
        candidate_sentence = normalize_for_match(first_sentence(_row_text(candidate, prefer_exact=True), max_chars=160))
        duplicate_penalty = int(bool(anchor_sentence and candidate_sentence and anchor_sentence == candidate_sentence))
        if answer_mode == "comparison":
            sort_key = (
                int(same_section and paragraph_like),
                int(same_page and paragraph_like),
                context_overlap,
                deeper_page,
                non_noise,
                paragraph_like,
                structured_ok,
                -duplicate_penalty,
                base_score,
            )
        elif answer_mode == "inductive":
            sort_key = (
                paragraph_like,
                context_overlap,
                deeper_page,
                int(not generic_title_adjacent),
                same_section,
                non_noise,
                structured_ok,
                -duplicate_penalty,
                base_score,
            )
        else:
            sort_key = (
                context_overlap,
                paragraph_like,
                deeper_page,
                non_noise,
                structured_ok,
                -duplicate_penalty,
                base_score,
            )
        if best_sort_key is None or sort_key > best_sort_key:
            best_sort_key = sort_key
            best_row = dict(candidate)
    return best_row


def _select_report_lookup_rows(doc_candidate: dict[str, Any]) -> list[dict[str, Any]]:
    doc_rows = list(doc_candidate.get("rows", []))
    if not doc_rows:
        return []
    anchor = _select_doc_anchor(
        _title_page_candidates(doc_rows),
        query=doc_candidate.get("short_title", ""),
        answer_mode="report_lookup",
        fact_subtype="report_lookup",
        numeric_query=False,
    )
    if anchor is None:
        return []
    selected = [anchor]
    companion = _best_companion(
        anchor,
        [row for row in doc_rows if not _is_noise_evidence_row(row)],
        numeric_query=False,
        require_same_section=False,
        prefer_paragraph=True,
    )
    if companion is not None and not any(existing["chunk_id"] == companion["chunk_id"] for existing in selected):
        selected.append(companion)
    return selected[:2]


def _select_multidoc_rows_for_doc(
    doc_candidate: dict[str, Any],
    *,
    query: str,
    answer_mode: str,
    fact_subtype: str,
    numeric_query: bool,
) -> list[dict[str, Any]]:
    doc_rows = list(doc_candidate.get("rows", []))
    if not doc_rows:
        return []
    context_text = normalize_text(
        "\n".join(
            [
                query,
                doc_candidate.get("short_title", ""),
                _title_topic_hint(doc_candidate.get("file_name", "")),
            ]
        )
    )
    if answer_mode in {"comparison", "inductive"}:
        overview_pool = [row for row in doc_rows if not _is_noise_evidence_row(row)] or list(doc_rows)
        overview_anchor = max(
            overview_pool,
            key=lambda row: (
                int(_is_paragraph_like(row)),
                _context_overlap_score(context_text, row) + _row_query_overlap(query, row),
                int(int(row.get("page_start", 1)) > 2),
                int(not _is_noise_topic_text(_row_topic_hint(row))),
                int(not _is_structured_row(row) or numeric_query),
                _row_score(row),
            ),
        )
        overview_anchor = dict(overview_anchor)
    else:
        overview_anchor = _select_doc_anchor(
            _title_page_candidates(doc_rows),
            query=query,
            answer_mode=answer_mode,
            fact_subtype=fact_subtype,
            numeric_query=numeric_query,
        )
    if overview_anchor is None:
        return []
    selected = [overview_anchor]
    support_pool = [row for row in doc_rows if row["chunk_id"] != overview_anchor["chunk_id"] and not _is_noise_evidence_row(row)]
    support_row = _best_support_row(
        query,
        overview_anchor,
        support_pool or doc_rows,
        numeric_query=numeric_query,
        answer_mode=answer_mode,
        context_text=context_text,
    )
    if support_row is not None and not any(existing["chunk_id"] == support_row["chunk_id"] for existing in selected):
        selected.append(support_row)
    return selected[:2]


def _assign_evidence_ids(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assigned: list[dict[str, Any]] = []
    for index, row in enumerate(_unique_rows(rows), start=1):
        payload = dict(row)
        payload['evidence_id'] = f'E{index}'
        payload.setdefault('support_span', _row_text(row, prefer_exact=True))
        payload.setdefault('bundle_id', payload.get('parent_chunk_id') or payload.get('chunk_id'))
        payload.setdefault('bundle_rank', payload.get('bundle_rank') or 1)
        assigned.append(payload)
    return assigned



def _query_bucket_tokens(query: str, bucket: str) -> list[str]:
    normalized_query = normalize_text(query).lower()
    return [token for token in QUERY_DOMAIN_HINTS.get(bucket, ()) if token.lower() in normalized_query]


def _uncovered_query_domains(
    *,
    query: str,
    query_domain_buckets: list[str],
    selected_domain_buckets: list[str],
    selected_rows: list[dict[str, Any]],
) -> list[str]:
    strong_query_domains = [_normalize_domain_bucket(bucket) for bucket in query_domain_buckets if not _is_weak_domain_bucket(bucket)]
    if len(strong_query_domains) < 2 or not selected_rows:
        return []
    selected_bucket_set = {
        _normalize_domain_bucket(bucket)
        for bucket in selected_domain_buckets
        if not _is_weak_domain_bucket(bucket)
    }
    if not selected_bucket_set:
        return []
    selected_text = normalize_text(
        "\n".join(
            [
                normalize_text(row.get("file_name", "") or "")
                + "\n"
                + _row_text(row, prefer_exact=True)
                + "\n"
                + normalize_text(row.get("section_title", "") or "")
                for row in selected_rows
            ]
        )
    ).lower()
    uncovered: list[str] = []
    for bucket in strong_query_domains:
        if any(_domains_compatible(bucket, selected_bucket) for selected_bucket in selected_bucket_set):
            continue
        bucket_tokens = _query_bucket_tokens(query, bucket)
        if not bucket_tokens:
            continue
        if not any(token.lower() in selected_text for token in bucket_tokens):
            uncovered.append(bucket)
    return uncovered


def _selected_domain_mismatch(query_domain_buckets: list[str], selected_domain_buckets: list[str], *, answer_mode: str) -> bool:
    del answer_mode
    strong_query_domains = [_normalize_domain_bucket(bucket) for bucket in query_domain_buckets if not _is_weak_domain_bucket(bucket)]
    selected = {
        _normalize_domain_bucket(bucket)
        for bucket in selected_domain_buckets
        if not _is_weak_domain_bucket(bucket)
    }
    if not strong_query_domains or not selected:
        return False
    if not any(
        _domains_compatible(query_bucket, selected_bucket)
        for query_bucket in strong_query_domains
        for selected_bucket in selected
    ):
        return True
    return False



def _prepare_evidence(*, query: str, question_type: str, answer_mode: str, fact_subtype: str, query_domain_bucket: str, query_domain_buckets: list[str], retrieval_result: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], str | None, bool]:
    numeric_query = is_numeric_or_table_query(query)
    pre_guard_doc_candidates = _build_doc_candidates(query, answer_mode, retrieval_result)
    doc_candidates, doc_guard_triggered = _apply_domain_guard(pre_guard_doc_candidates, query_domain_bucket=query_domain_bucket, answer_mode=answer_mode)
    if not doc_candidates:
        return [], [], [], 'no_evidence', doc_guard_triggered

    if answer_mode == 'report_lookup':
        primary = doc_candidates[0]
        selected_rows = _select_report_lookup_rows(primary)
        selected_domain_buckets = [primary.get('domain_bucket', '')]
        if _selected_domain_mismatch(query_domain_buckets, selected_domain_buckets, answer_mode=answer_mode):
            return [], doc_candidates, [primary['doc_id']], 'domain_mismatch', doc_guard_triggered
        if _uncovered_query_domains(
            query=query,
            query_domain_buckets=query_domain_buckets,
            selected_domain_buckets=selected_domain_buckets,
            selected_rows=selected_rows,
        ):
            return [], doc_candidates, [primary['doc_id']], 'domain_mismatch', doc_guard_triggered
        return _assign_evidence_ids(selected_rows), doc_candidates, [primary['doc_id']], None, doc_guard_triggered

    if answer_mode == 'numeric_fact':
        primary = doc_candidates[0]
        max_rows = 3 if fact_subtype == 'value_fact' else 2
        selected_rows = _select_rows_for_doc(
            primary,
            query=query,
            answer_mode=answer_mode,
            fact_subtype=fact_subtype,
            numeric_query=numeric_query,
            max_rows=max_rows,
        )
        selected_domain_buckets = [primary.get('domain_bucket', '')]
        if _selected_domain_mismatch(query_domain_buckets, selected_domain_buckets, answer_mode=answer_mode):
            return [], doc_candidates, [primary['doc_id']], 'domain_mismatch', doc_guard_triggered
        if _uncovered_query_domains(
            query=query,
            query_domain_buckets=query_domain_buckets,
            selected_domain_buckets=selected_domain_buckets,
            selected_rows=selected_rows,
        ):
            return [], doc_candidates, [primary['doc_id']], 'domain_mismatch', doc_guard_triggered
        return _assign_evidence_ids(selected_rows), doc_candidates, [primary['doc_id']], None, doc_guard_triggered

    if answer_mode == 'comparison':
        top_docs = _select_comparison_doc_candidates(
            doc_candidates,
            query=query,
            max_docs=2,
            pre_guard_candidates=pre_guard_doc_candidates if doc_guard_triggered else None,
        )
        if len(top_docs) < 2:
            return [], doc_candidates, [], 'insufficient_source_diversity', doc_guard_triggered
        selected_domain_buckets = [candidate.get('domain_bucket', '') for candidate in top_docs]
        if _selected_domain_mismatch(query_domain_buckets, selected_domain_buckets, answer_mode=answer_mode):
            return [], doc_candidates, [candidate['doc_id'] for candidate in top_docs], 'domain_mismatch', doc_guard_triggered
        selected_rows: list[dict[str, Any]] = []
        for candidate in top_docs:
            selected_rows.extend(
                _select_multidoc_rows_for_doc(
                    candidate,
                    query=query,
                    answer_mode=answer_mode,
                    fact_subtype=fact_subtype,
                    numeric_query=numeric_query,
                )
            )
        if _uncovered_query_domains(
            query=query,
            query_domain_buckets=query_domain_buckets,
            selected_domain_buckets=selected_domain_buckets,
            selected_rows=selected_rows,
        ):
            return [], doc_candidates, [candidate['doc_id'] for candidate in top_docs], 'domain_mismatch', doc_guard_triggered
        return _assign_evidence_ids(selected_rows), doc_candidates, [candidate['doc_id'] for candidate in top_docs], None, doc_guard_triggered

    top_docs = _select_inductive_doc_candidates(
        doc_candidates,
        query=query,
        query_domain_bucket=query_domain_bucket,
        max_docs=3,
        pre_guard_candidates=pre_guard_doc_candidates if doc_guard_triggered else None,
    )
    if len(top_docs) < 2:
        return [], doc_candidates, [], 'insufficient_source_diversity', doc_guard_triggered
    selected_domain_buckets = [candidate.get('domain_bucket', '') for candidate in top_docs]
    if _selected_domain_mismatch(query_domain_buckets, selected_domain_buckets, answer_mode=answer_mode):
        return [], doc_candidates, [candidate['doc_id'] for candidate in top_docs], 'domain_mismatch', doc_guard_triggered
    selected_rows: list[dict[str, Any]] = []
    for candidate in top_docs:
        selected_rows.extend(
            _select_multidoc_rows_for_doc(
                candidate,
                query=query,
                answer_mode=answer_mode,
                fact_subtype=fact_subtype,
                numeric_query=numeric_query,
            )
        )
    if _uncovered_query_domains(
        query=query,
        query_domain_buckets=query_domain_buckets,
        selected_domain_buckets=selected_domain_buckets,
        selected_rows=selected_rows,
    ):
        return [], doc_candidates, [candidate['doc_id'] for candidate in top_docs], 'domain_mismatch', doc_guard_triggered
    if len({row['doc_id'] for row in selected_rows}) < min(3, len(top_docs)):
        return [], doc_candidates, [candidate['doc_id'] for candidate in top_docs], 'insufficient_source_diversity', doc_guard_triggered
    return _assign_evidence_ids(selected_rows), doc_candidates, [candidate['doc_id'] for candidate in top_docs], None, doc_guard_triggered



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
        return used_ids[:2] or [row["evidence_id"] for row in selected_evidence[:2]]

    if answer_mode == "numeric_fact":
        if not selected_evidence:
            return []
        primary_doc_id = selected_evidence[0]["doc_id"]
        filtered = [evidence_id for evidence_id in used_ids if evidence_by_id[evidence_id]["doc_id"] == primary_doc_id]
        return filtered[:2] or [selected_evidence[0]["evidence_id"]]

    required_doc_count = 2 if answer_mode == "comparison" else min(3, len({row["doc_id"] for row in selected_evidence}))
    covered_docs = {evidence_by_id[evidence_id]["doc_id"] for evidence_id in used_ids if evidence_id in evidence_by_id}
    for row in selected_evidence:
        if row["evidence_id"] in used_ids:
            continue
        if row["doc_id"] in covered_docs:
            continue
        used_ids.append(row["evidence_id"])
        covered_docs.add(row["doc_id"])
        if len(covered_docs) >= required_doc_count:
            break

    if answer_mode in {"comparison", "inductive"}:
        rows_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in selected_evidence:
            rows_by_doc[row["doc_id"]].append(row)
        for doc_id, rows in rows_by_doc.items():
            doc_used = [
                evidence_id
                for evidence_id in used_ids
                if evidence_id in evidence_by_id and evidence_by_id[evidence_id]["doc_id"] == doc_id
            ]
            if len(doc_used) >= 2:
                continue
            support_candidates = [row for row in rows if row["evidence_id"] not in used_ids]
            support_candidates.sort(
                key=lambda row: (
                    _is_paragraph_like(row),
                    int(row.get("page_start", 1)) > 2,
                    _row_score(row),
                ),
                reverse=True,
            )
            if support_candidates:
                used_ids.append(support_candidates[0]["evidence_id"])

    return list(dict.fromkeys(used_ids))


def _citation_context_map(
    *,
    answer_mode: str,
    payload: dict[str, Any],
    selected_evidence: list[dict[str, Any]],
) -> dict[str, str]:
    context_map: dict[str, str] = {}
    if answer_mode == "comparison":
        doc_focus_map = {
            str(doc_id): normalize_text(str(summary))
            for doc_id, summary in dict(payload.get("doc_focus_map", {}) or {}).items()
            if normalize_text(str(summary))
        }
        differences = [normalize_text(item) for item in payload.get("differences", []) if normalize_text(item)]
        common_points = [normalize_text(item) for item in payload.get("common_points", []) if normalize_text(item)]
        conclusion = normalize_text(payload.get("conclusion", ""))
        final_answer = normalize_text(payload.get("final_answer", ""))
        for doc_id, focus in doc_focus_map.items():
            context_map[doc_id] = normalize_text("\n".join([focus, *differences[:2], *common_points[:2], conclusion, final_answer]))
        return context_map

    if answer_mode == "inductive":
        per_doc_observation = {
            str(doc_id): normalize_text(str(summary))
            for doc_id, summary in dict(payload.get("per_doc_observation", {}) or {}).items()
            if normalize_text(str(summary))
        }
        shared_themes = [normalize_text(item) for item in payload.get("shared_themes", []) if normalize_text(item)]
        synthesis = normalize_text(payload.get("evidence_backed_synthesis", "") or payload.get("synthesis", ""))
        final_answer = normalize_text(payload.get("final_answer", ""))
        for doc_id, observation in per_doc_observation.items():
            context_map[doc_id] = normalize_text("\n".join([observation, *shared_themes[:3], synthesis, final_answer]))
        return context_map

    for row in selected_evidence:
        context_map.setdefault(row["doc_id"], normalize_text(payload.get("final_answer", "")))
    return context_map


def _citation_row_sort_key(
    row: dict[str, Any],
    *,
    anchor_row: dict[str, Any],
    context_text: str,
    answer_mode: str,
) -> tuple[int, ...]:
    same_section = int(bool(anchor_row.get("section_path") and anchor_row.get("section_path") == row.get("section_path")))
    same_page = int(int(anchor_row.get("page_start", 1)) == int(row.get("page_start", 1)))
    paragraph_like = int(_is_paragraph_like(row))
    non_noise = int(not _is_noise_evidence_row(row))
    context_overlap = _context_overlap_score(context_text, row)
    deeper_page = int(int(row.get("page_start", 1)) > 2)
    generic_title_adjacent = int(
        int(row.get("page_start", 1)) <= 2
        and context_overlap == 0
        and (_is_noise_topic_text(_row_topic_hint(row)) or not paragraph_like)
    )
    base_score = int(_row_score(row) * 1000)
    if answer_mode == "comparison":
        return (
            int(same_section and paragraph_like),
            int(same_page and paragraph_like),
            context_overlap,
            deeper_page,
            non_noise,
            paragraph_like,
            base_score,
        )
    if answer_mode == "inductive":
        return (
            context_overlap,
            paragraph_like,
            deeper_page,
            int(not generic_title_adjacent),
            same_section,
            non_noise,
            base_score,
        )
    return (context_overlap, paragraph_like, non_noise, base_score)


def _refine_citation_evidence_ids(
    *,
    answer_mode: str,
    payload: dict[str, Any],
    selected_evidence: list[dict[str, Any]],
    used_evidence_ids: list[str],
) -> list[str]:
    if answer_mode not in {"comparison", "inductive"} or not selected_evidence:
        return list(dict.fromkeys(used_evidence_ids))

    evidence_by_id = {row["evidence_id"]: row for row in selected_evidence}
    rows_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_evidence:
        rows_by_doc[row["doc_id"]].append(row)

    doc_order: list[str] = []
    seen_docs: set[str] = set()
    for evidence_id in used_evidence_ids:
        row = evidence_by_id.get(evidence_id)
        if row is None or row["doc_id"] in seen_docs:
            continue
        doc_order.append(row["doc_id"])
        seen_docs.add(row["doc_id"])
    if not doc_order:
        doc_order = _required_doc_ids(selected_evidence)

    context_map = _citation_context_map(answer_mode=answer_mode, payload=payload, selected_evidence=selected_evidence)
    refined_ids: list[str] = []
    for doc_id in doc_order:
        doc_rows = rows_by_doc.get(doc_id, [])
        if not doc_rows:
            continue
        anchor_row = next(
            (
                evidence_by_id[evidence_id]
                for evidence_id in used_evidence_ids
                if evidence_id in evidence_by_id and evidence_by_id[evidence_id]["doc_id"] == doc_id
            ),
            doc_rows[0],
        )
        context_text = context_map.get(doc_id, "")
        support_rows = [row for row in doc_rows if row["chunk_id"] != anchor_row["chunk_id"]]
        ranked_support_rows = sorted(
            support_rows,
            key=lambda row: _citation_row_sort_key(
                row,
                anchor_row=anchor_row,
                context_text=context_text,
                answer_mode=answer_mode,
            ),
            reverse=True,
        )
        chosen_rows: list[dict[str, Any]] = []
        if ranked_support_rows:
            top_support = ranked_support_rows[0]
            support_overlap = _context_overlap_score(context_text, top_support)
            support_is_body = _is_paragraph_like(top_support) and int(top_support.get("page_start", 1)) > 2
            support_same_section = bool(anchor_row.get("section_path") and anchor_row.get("section_path") == top_support.get("section_path"))
            if support_overlap > 0 or support_is_body or (answer_mode == "comparison" and support_same_section):
                chosen_rows.append(top_support)
        if anchor_row["evidence_id"] not in {row["evidence_id"] for row in chosen_rows}:
            chosen_rows.append(anchor_row)
        ranked_rows = sorted(
            doc_rows,
            key=lambda row: _citation_row_sort_key(
                row,
                anchor_row=anchor_row,
                context_text=context_text,
                answer_mode=answer_mode,
            ),
            reverse=True,
        )
        for row in ranked_rows:
            if len(chosen_rows) >= min(2, len(doc_rows)):
                break
            if row["evidence_id"] in {item["evidence_id"] for item in chosen_rows}:
                continue
            chosen_rows.append(row)
        for row in chosen_rows:
            refined_ids.append(row["evidence_id"])

    for evidence_id in used_evidence_ids:
        if evidence_id not in refined_ids:
            refined_ids.append(evidence_id)
    return list(dict.fromkeys(refined_ids))


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



def _doc_theme_summary(rows: list[dict[str, Any]], *, top_k: int = 2) -> str:
    texts: list[str] = []
    for row in rows:
        hint = _row_topic_hint(row)
        if hint:
            texts.append(hint)
        sentence = first_sentence(_row_text(row, prefer_exact=True), max_chars=72)
        if sentence and not _is_noise_topic_text(sentence):
            texts.append(sentence)
    theme_terms = _theme_terms("\n".join(texts), top_k=top_k + 4)
    if theme_terms:
        return "、".join(theme_terms[:top_k])
    for candidate in texts:
        if candidate and not _is_noise_topic_text(candidate):
            return candidate
    return ""



def _common_terms_for_rows(evidence_rows: list[dict[str, Any]], *, min_doc_frequency: int = 2, top_k: int = 5) -> list[str]:
    rows_by_doc: dict[str, list[str]] = defaultdict(list)
    for row in evidence_rows:
        hint = _row_topic_hint(row)
        if hint:
            rows_by_doc[row["doc_id"]].append(hint)
        sentence = first_sentence(_row_text(row, prefer_exact=True), max_chars=72)
        if sentence and not _is_noise_topic_text(sentence):
            rows_by_doc[row["doc_id"]].append(sentence)
    doc_frequency: Counter[str] = Counter()
    term_score: Counter[str] = Counter()
    display_terms: dict[str, str] = {}
    for texts in rows_by_doc.values():
        seen_keys: set[str] = set()
        for rank, term in enumerate(_theme_terms("\n".join(texts), top_k=12), start=1):
            key = normalize_for_match(term)
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            display_terms.setdefault(key, term)
            doc_frequency[key] += 1
            term_score[key] += max(1, 12 - rank)
    common_terms = [
        display_terms[key]
        for key, count in sorted(
            doc_frequency.items(),
            key=lambda item: (-item[1], -term_score[item[0]], item[0]),
        )
        if count >= min_doc_frequency
    ]
    return common_terms[:top_k]


def _doc_title_variants(selected_evidence: list[dict[str, Any]]) -> dict[str, list[str]]:
    variants: dict[str, list[str]] = {}
    for row in selected_evidence:
        doc_id = row["doc_id"]
        file_name = str(row.get("file_name", ""))
        candidates = [
            _short_report_title(file_name),
            normalize_text(strip_file_extension(Path(file_name).name)),
            normalize_text(file_name),
        ]
        deduped: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized_candidate = normalize_text(candidate)
            if not normalized_candidate or normalized_candidate in seen:
                continue
            deduped.append(normalized_candidate)
            seen.add(normalized_candidate)
        if deduped:
            variants[doc_id] = deduped
    return variants


def _required_doc_ids(selected_evidence: list[dict[str, Any]]) -> list[str]:
    ordered_doc_ids: list[str] = []
    seen_docs: set[str] = set()
    for row in selected_evidence:
        if row["doc_id"] in seen_docs:
            continue
        ordered_doc_ids.append(row["doc_id"])
        seen_docs.add(row["doc_id"])
    return ordered_doc_ids


def _has_focus_overlap(final_answer: str, focus_text: str) -> bool:
    answer_terms = set(extract_terms(final_answer, top_k=10))
    focus_terms = set(extract_terms(focus_text, top_k=8))
    if not focus_terms:
        return False
    return bool(answer_terms & focus_terms)


def _rows_by_doc(selected_evidence: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    rows_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_evidence:
        rows_by_doc[row["doc_id"]].append(row)
    return rows_by_doc


def _default_doc_used_evidence_ids(required_doc_ids: list[str], selected_evidence: list[dict[str, Any]]) -> list[str]:
    by_doc = _rows_by_doc(selected_evidence)
    return [by_doc[doc_id][0]["evidence_id"] for doc_id in required_doc_ids if by_doc.get(doc_id)]


def _compact_summary_text(text: str, *, top_k: int = 3, fallback_chars: int = 36) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return ""
    theme_terms = _theme_terms(normalized, top_k=top_k + 2)
    if theme_terms:
        return "、".join(theme_terms[:top_k])
    return first_sentence(normalized, max_chars=fallback_chars)


def _coerce_doc_summary_map(raw_map: dict[str, Any], *, field_name: str = "focus") -> dict[str, str]:
    normalized: dict[str, str] = {}
    for doc_id, summary in dict(raw_map or {}).items():
        if isinstance(summary, dict):
            text = normalize_text(str(summary.get(field_name, "") or summary.get("summary", "") or ""))
        else:
            text = normalize_text(str(summary))
        if text:
            normalized[str(doc_id)] = text
    return normalized


def _ensure_doc_focus_map(required_doc_ids: list[str], selected_evidence: list[dict[str, Any]], doc_focus_map: dict[str, str]) -> dict[str, str]:
    by_doc = _rows_by_doc(selected_evidence)
    repaired = dict(doc_focus_map)
    for doc_id in required_doc_ids:
        if normalize_text(str(repaired.get(doc_id, ""))):
            continue
        summary = _doc_theme_summary(by_doc.get(doc_id, []))
        if summary:
            repaired[doc_id] = summary
    return repaired


def _ensure_per_doc_observation(required_doc_ids: list[str], selected_evidence: list[dict[str, Any]], per_doc_observation: dict[str, str]) -> dict[str, str]:
    by_doc = _rows_by_doc(selected_evidence)
    repaired = dict(per_doc_observation)
    for doc_id in required_doc_ids:
        if normalize_text(str(repaired.get(doc_id, ""))):
            continue
        summary = _doc_theme_summary(by_doc.get(doc_id, []))
        if summary:
            repaired[doc_id] = summary
    return repaired


def _normalize_theme_list(items: list[Any] | tuple[Any, ...] | str) -> list[str]:
    if isinstance(items, str):
        items = [items]
    return [normalize_text(str(item)) for item in items if normalize_text(str(item))]


def _normalize_theme_candidates(raw_candidates: dict[str, Any]) -> dict[str, list[str]]:
    normalized: dict[str, list[str]] = {}
    for doc_id, items in dict(raw_candidates or {}).items():
        themes = _normalize_theme_list(items)
        if themes:
            normalized[str(doc_id)] = themes
    return normalized


def _derive_shared_themes(
    *,
    observed_doc_ids: list[str],
    theme_candidates: dict[str, list[str]],
    selected_evidence: list[dict[str, Any]],
) -> list[str]:
    display_map: dict[str, str] = {}
    doc_frequency: Counter[str] = Counter()
    by_doc = _rows_by_doc(selected_evidence)
    for doc_id in observed_doc_ids:
        candidates = list(theme_candidates.get(doc_id, []))
        if not candidates:
            candidates = _theme_terms(
                "\n".join(
                    [
                        _doc_theme_summary(by_doc.get(doc_id, [])),
                        *[_row_text(row, prefer_exact=True) for row in by_doc.get(doc_id, [])[:2]],
                    ]
                ),
                top_k=6,
            )
        seen_keys: set[str] = set()
        for theme in candidates:
            key = normalize_for_match(theme)
            if not key or key in seen_keys or _is_noise_theme_term(theme):
                continue
            seen_keys.add(key)
            display_map.setdefault(key, theme)
            doc_frequency[key] += 1
    shared = [
        display_map[key]
        for key, count in sorted(doc_frequency.items(), key=lambda item: (-item[1], item[0]))
        if count >= 2
    ]
    if shared:
        return shared[:3]
    return _common_terms_for_rows(selected_evidence, min_doc_frequency=2, top_k=3)


def _report_period_label(file_name: str) -> str:
    stem = strip_file_extension(Path(file_name).name)
    match = REPORT_PERIOD_RE.search(stem)
    if not match:
        return ""
    if match.group(4) and match.group(5) and match.group(6):
        return f"{match.group(5)}{match.group(6)}"
    if match.group(2) and match.group(3):
        return f"{match.group(2)}{match.group(3)}"
    return ""


def _report_display_label(doc_id: str, selected_evidence: list[dict[str, Any]], duplicate_titles: set[str]) -> str:
    row = next(item for item in selected_evidence if item["doc_id"] == doc_id)
    title = _short_report_title(row["file_name"])
    if title in duplicate_titles:
        period_label = _report_period_label(row["file_name"])
        if period_label:
            return f"《{title}》({period_label})"
    return f"《{title}》"


def _duplicate_short_titles(required_doc_ids: list[str], selected_evidence: list[dict[str, Any]]) -> set[str]:
    titles: list[str] = []
    seen_docs: set[str] = set()
    required_set = set(required_doc_ids)
    for row in selected_evidence:
        if row["doc_id"] not in required_set or row["doc_id"] in seen_docs:
            continue
        titles.append(_short_report_title(row["file_name"]))
        seen_docs.add(row["doc_id"])
    return {title for title in titles if titles.count(title) > 1}


def _shared_logic_focus_text(required_doc_ids: list[str], doc_focus_map: dict[str, str], selected_evidence: list[dict[str, Any]]) -> str:
    by_doc = _rows_by_doc(selected_evidence)
    focus_texts: list[str] = []
    for doc_id in required_doc_ids:
        focus = normalize_text(doc_focus_map.get(doc_id, ""))
        if focus:
            focus_texts.append(focus)
            continue
        summary = _doc_theme_summary(by_doc.get(doc_id, []))
        if summary:
            focus_texts.append(summary)
    shared_terms = _common_terms_for_rows(selected_evidence, min_doc_frequency=2, top_k=3)
    if shared_terms:
        return "、".join(shared_terms)
    if focus_texts:
        compact_terms = _theme_terms("\n".join(focus_texts), top_k=4)
        if compact_terms:
            return "、".join(compact_terms[:3])
        return _compact_summary_text(focus_texts[0])
    return ""


def _preferred_numeric_semantic_answer(selected_evidence: list[dict[str, Any]]) -> str:
    paragraph_rows = [
        row
        for row in selected_evidence
        if _is_paragraph_like(row) and _row_topic_hint(row) and not _is_noise_evidence_row(row)
    ]
    if paragraph_rows:
        return first_sentence(_row_text(paragraph_rows[0], prefer_exact=True), max_chars=96)
    non_structured_rows = [row for row in selected_evidence if not _is_structured_row(row)]
    if non_structured_rows:
        return first_sentence(_row_text(non_structured_rows[0], prefer_exact=True), max_chars=96)
    return first_sentence(_row_text(selected_evidence[0], prefer_exact=True), max_chars=96) if selected_evidence else ""


def _shared_logic_comparison_answer(
    *,
    required_doc_ids: list[str],
    selected_evidence: list[dict[str, Any]],
    doc_focus_map: dict[str, str],
    shared_points: list[str],
    conclusion: str,
) -> str:
    if len(required_doc_ids) < 2:
        return ""
    duplicate_titles = _duplicate_short_titles(required_doc_ids, selected_evidence)
    left_doc_id, right_doc_id = required_doc_ids[:2]
    left_label = _report_display_label(left_doc_id, selected_evidence, duplicate_titles)
    right_label = _report_display_label(right_doc_id, selected_evidence, duplicate_titles)
    shared_focus = _compact_summary_text(
        "、".join(shared_points) or _shared_logic_focus_text(required_doc_ids, doc_focus_map, selected_evidence),
        top_k=3,
        fallback_chars=48,
    )
    if not shared_focus:
        return ""
    left_focus = _compact_summary_text(str(doc_focus_map.get(left_doc_id, "")), top_k=2, fallback_chars=24)
    right_focus = _compact_summary_text(str(doc_focus_map.get(right_doc_id, "")), top_k=2, fallback_chars=24)
    parts = [f"{left_label}与{right_label}都围绕{shared_focus}展开。"]
    if left_focus and right_focus and left_focus != right_focus:
        parts.append(f"其中{left_label}更偏向{left_focus}，{right_label}更偏向{right_focus}。")
    if conclusion:
        parts.append(_compact_summary_text(conclusion, top_k=4, fallback_chars=64).rstrip("。") + "。")
    return normalize_text("".join(parts))


def _compose_comparison_answer(
    *,
    query: str,
    required_doc_ids: list[str],
    selected_evidence: list[dict[str, Any]],
    doc_focus_map: dict[str, str],
    difference_dimension: str,
    difference_detail: str,
    shared_points: list[str],
    conclusion: str,
) -> str:
    if len(required_doc_ids) < 2:
        return ""
    left_doc_id, right_doc_id = required_doc_ids[:2]
    left_focus = _compact_summary_text(str(doc_focus_map.get(left_doc_id, "")))
    right_focus = _compact_summary_text(str(doc_focus_map.get(right_doc_id, "")))
    if not left_focus or not right_focus:
        return ""
    short_titles = [_short_report_title(row["file_name"]) for row in selected_evidence if row["doc_id"] in set(required_doc_ids)]
    duplicate_titles = {title for title in short_titles if short_titles.count(title) > 1}
    left_title = _report_display_label(left_doc_id, selected_evidence, duplicate_titles)
    right_title = _report_display_label(right_doc_id, selected_evidence, duplicate_titles)
    dimension_text = normalize_text(difference_dimension) or "关注重点"
    detail_text = _compact_summary_text(difference_detail, top_k=4, fallback_chars=64)
    if not detail_text:
        detail_text = f"{left_title}更强调{left_focus}，{right_title}更强调{right_focus}"
    parts = [
        f"{left_title}主要关注{left_focus}；{right_title}主要关注{right_focus}。",
        f"二者在{dimension_text}上的差异在于{detail_text.rstrip('。')}。",
    ]
    if shared_points:
        parts.append("共同点包括" + "、".join(_compact_summary_text(item) for item in shared_points[:2]) + "。")
    if conclusion:
        parts.append(conclusion.rstrip("。") + "。")
    return normalize_text("".join(parts))


def _compose_inductive_synthesis(
    *,
    observed_doc_ids: list[str],
    per_doc_observation: dict[str, str],
) -> str:
    observations = [normalize_text(str(per_doc_observation.get(doc_id, ""))) for doc_id in observed_doc_ids if normalize_text(str(per_doc_observation.get(doc_id, "")))]
    deduped: list[str] = []
    seen: set[str] = set()
    for observation in observations:
        normalized = normalize_for_match(observation)
        if not normalized or normalized in seen:
            continue
        deduped.append(observation.rstrip("。"))
        seen.add(normalized)
    if len(deduped) < 2:
        return ""
    return normalize_text("；".join(deduped[:2]) + "。")


def _compose_inductive_answer(
    *,
    query: str,
    observed_doc_ids: list[str],
    selected_evidence: list[dict[str, Any]],
    per_doc_observation: dict[str, str],
    shared_themes: list[str],
    synthesis_basis: str,
    synthesis: str,
) -> str:
    if len(observed_doc_ids) < 2:
        return ""
    by_doc = _rows_by_doc(selected_evidence)
    observation_parts: list[str] = []
    max_observation_docs = 3 if is_broad_inductive_query(query) else 2
    for doc_id in observed_doc_ids[:max_observation_docs]:
        doc_rows = by_doc.get(doc_id, [])
        if not doc_rows:
            continue
        title = _short_report_title(doc_rows[0]["file_name"])
        observation = first_sentence(normalize_text(str(per_doc_observation.get(doc_id, ""))), max_chars=48)
        if not observation:
            continue
        observation_parts.append(f"《{title}》强调{observation}")
    if len(observation_parts) < 2:
        return ""
    shared_theme_text = "、".join(normalize_text(theme) for theme in shared_themes[:3] if normalize_text(theme))
    parts = []
    if shared_theme_text:
        parts.append(f"共同主题包括：{shared_theme_text}。")
    parts.append("；".join(observation_parts) + "。")
    basis_text = normalize_text(synthesis_basis) or normalize_text(synthesis)
    if basis_text:
        parts.append(first_sentence(basis_text, max_chars=96).rstrip("。") + "。")
    return normalize_text("".join(parts))


def _is_single_evidence_rephrase(final_answer: str, selected_evidence: list[dict[str, Any]]) -> bool:
    normalized_final = normalize_for_match(first_sentence(final_answer, max_chars=160))
    if not normalized_final:
        return False
    for row in selected_evidence:
        normalized_evidence = normalize_for_match(first_sentence(_row_text(row, prefer_exact=True), max_chars=160))
        if normalized_evidence and normalized_evidence == normalized_final:
            return True
    return False


def _fallback_payload(answer_mode: str, selected_evidence: list[dict[str, Any]], *, fact_subtype: str, query: str = "") -> dict[str, Any]:
    conservative_note = "Conservative fallback based on current evidence."
    if not selected_evidence:
        return {
            "final_answer": FALLBACK_ANSWER,
            "evidence_summary": "",
            "uncertainty_note": conservative_note,
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
            preferred_answer = _preferred_numeric_semantic_answer(selected_evidence)
            if preferred_answer:
                answer_text = preferred_answer
            if _is_generic_numeric_semantic_query(query, fact_subtype=fact_subtype, answer_mode=answer_mode):
                answer_text = preferred_answer or answer_text
        return {
            "final_answer": answer_text,
            "evidence_summary": _default_evidence_summary(answer_mode, selected_evidence),
            "uncertainty_note": "",
            "used_evidence_ids": [row["evidence_id"] for row in selected_evidence[:2]],
        }

    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_evidence:
        by_doc[row["doc_id"]].append(row)
    ordered_docs: list[str] = []
    seen_docs: set[str] = set()
    for row in selected_evidence:
        if row["doc_id"] in seen_docs:
            continue
        ordered_docs.append(row["doc_id"])
        seen_docs.add(row["doc_id"])

    if answer_mode == "comparison":
        if len(ordered_docs) < 2:
            return {
                "final_answer": FALLBACK_ANSWER,
                "evidence_summary": _default_evidence_summary(answer_mode, selected_evidence),
                "uncertainty_note": conservative_note,
                "used_evidence_ids": [],
            }
        doc_summaries: list[tuple[dict[str, Any], str]] = []
        for doc_id in ordered_docs[:2]:
            rows = by_doc[doc_id]
            summary = _doc_theme_summary(rows)
            if not summary or _is_noise_topic_text(summary):
                return {
                    "final_answer": FALLBACK_ANSWER,
                    "evidence_summary": _default_evidence_summary(answer_mode, selected_evidence),
                    "uncertainty_note": conservative_note,
                    "used_evidence_ids": [],
                }
            doc_summaries.append((rows[0], summary))
        common_terms = _common_terms_for_rows(selected_evidence, min_doc_frequency=2, top_k=2)
        tail = (
            "二者共同点在于" + "、".join(common_terms) + "。"
            if common_terms
            else "二者关注点存在明显差异。"
        )
        final_answer = normalize_text(
            f"《{_short_report_title(doc_summaries[0][0]['file_name'])}》主要强调{doc_summaries[0][1]}；"
            f"《{_short_report_title(doc_summaries[1][0]['file_name'])}》主要强调{doc_summaries[1][1]}。{tail}"
        )
        return {
            "final_answer": final_answer,
            "evidence_summary": _default_evidence_summary(answer_mode, selected_evidence),
            "uncertainty_note": "",
            "used_evidence_ids": [by_doc[doc_id][0]["evidence_id"] for doc_id in ordered_docs[:2]],
        }

    common_terms = _common_terms_for_rows(selected_evidence)
    if len(common_terms) < 2:
        return {
            "final_answer": FALLBACK_ANSWER,
            "evidence_summary": _default_evidence_summary(answer_mode, selected_evidence),
            "uncertainty_note": conservative_note,
            "used_evidence_ids": [],
        }
    common_text = "、".join(common_terms[:3])
    return {
        "final_answer": normalize_text(f"共同主题包括：{common_text}。"),
        "evidence_summary": _default_evidence_summary(answer_mode, selected_evidence),
        "uncertainty_note": "",
        "used_evidence_ids": [by_doc[doc_id][0]["evidence_id"] for doc_id in ordered_docs[:3]],
    }


def _repair_answer_payload(
    *,
    query: str = "",
    answer_mode: str,
    fact_subtype: str,
    payload: dict[str, Any],
    selected_evidence: list[dict[str, Any]],
) -> tuple[dict[str, Any], str | None]:
    fallback = _fallback_payload(answer_mode, selected_evidence, fact_subtype=fact_subtype, query=query)
    final_answer = normalize_text(payload.get("final_answer", "")) or fallback["final_answer"]
    evidence_summary = normalize_text(payload.get("evidence_summary", "")) or fallback["evidence_summary"]
    uncertainty_note = normalize_text(payload.get("uncertainty_note", ""))
    used_evidence_ids = payload.get("used_evidence_ids", [])

    if answer_mode == "report_lookup":
        title_variants = _doc_title_variants(selected_evidence)
        canonical_titles: list[str] = []
        matched_title = ""
        normalized_answer = normalize_for_match(final_answer)
        for variants in title_variants.values():
            canonical_title = variants[0]
            canonical_titles.append(canonical_title)
            if any(normalize_for_match(variant) and normalize_for_match(variant) in normalized_answer for variant in variants):
                matched_title = canonical_title
                break
        if matched_title:
            final_answer = matched_title
        else:
            cleaned_answer = normalize_text(final_answer).strip("《》“”\"' ")
            if cleaned_answer in canonical_titles:
                final_answer = cleaned_answer
            else:
                return fallback, "report_lookup_validation"
        return {
            "final_answer": final_answer,
            "evidence_summary": evidence_summary,
            "uncertainty_note": uncertainty_note,
            "used_evidence_ids": used_evidence_ids or [row["evidence_id"] for row in selected_evidence[:2]],
        }, None

    if answer_mode == "numeric_fact" and fact_subtype == "semantic_fact":
        preferred_answer = _preferred_numeric_semantic_answer(selected_evidence)
        if preferred_answer and _is_generic_numeric_semantic_query(query, fact_subtype=fact_subtype, answer_mode=answer_mode):
            final_answer = preferred_answer

    if answer_mode == "comparison":
        doc_focus_map = _coerce_doc_summary_map(payload.get("doc_focus_map", {}) or {}, field_name="focus")
        required_doc_ids = _required_doc_ids(selected_evidence)[:2]
        doc_focus_map = _ensure_doc_focus_map(required_doc_ids, selected_evidence, doc_focus_map)
        if any(not normalize_text(str(doc_focus_map.get(doc_id, ""))) for doc_id in required_doc_ids):
            return fallback, "comparison_validation"
        used_evidence_ids = list(dict.fromkeys(list(used_evidence_ids) + _default_doc_used_evidence_ids(required_doc_ids, selected_evidence)))
        comparison_subtype = infer_comparison_subtype(query)
        if _duplicate_short_titles(required_doc_ids, selected_evidence):
            comparison_subtype = "shared_logic_comparison"
        difference_dimension = normalize_text(payload.get("difference_dimension", ""))
        difference_detail = normalize_text(payload.get("difference_detail", ""))
        differences = [normalize_text(item) for item in payload.get("differences", []) if normalize_text(item)]
        if not difference_detail and differences:
            difference_detail = "；".join(differences[:2])
        if not difference_dimension and differences:
            difference_dimension = "关注重点"
        shared_points = [normalize_text(item) for item in payload.get("shared_points", []) if normalize_text(item)]
        if not shared_points:
            shared_points = [normalize_text(item) for item in payload.get("common_points", []) if normalize_text(item)]
        if not shared_points:
            shared_points = _common_terms_for_rows(selected_evidence, min_doc_frequency=2, top_k=3)
        conclusion = normalize_text(payload.get("conclusion", ""))
        if comparison_subtype == "shared_logic_comparison":
            candidate_answer = _shared_logic_comparison_answer(
                required_doc_ids=required_doc_ids,
                selected_evidence=selected_evidence,
                doc_focus_map=doc_focus_map,
                shared_points=shared_points,
                conclusion=conclusion,
            )
            combined_text = normalize_text("\n".join([candidate_answer, *shared_points, conclusion]))
        else:
            if not difference_detail:
                return fallback, "comparison_validation"
            candidate_answer = _compose_comparison_answer(
                query=query,
                required_doc_ids=required_doc_ids,
                selected_evidence=selected_evidence,
                doc_focus_map=doc_focus_map,
                difference_dimension=difference_dimension,
                difference_detail=difference_detail,
                shared_points=shared_points,
                conclusion=conclusion,
            )
            combined_text = normalize_text("\n".join([candidate_answer, difference_dimension, difference_detail, conclusion, *shared_points]))
        if not candidate_answer or not combined_text:
            return fallback, "comparison_validation"
        if not all(_has_focus_overlap(combined_text, doc_focus_map[doc_id]) for doc_id in required_doc_ids):
            return fallback, "comparison_validation"
        final_answer = candidate_answer

    if answer_mode == "inductive":
        required_doc_ids = _required_doc_ids(selected_evidence)
        per_doc_observation = _coerce_doc_summary_map(payload.get("per_doc_observation", {}) or {}, field_name="observation")
        per_doc_observation = _ensure_per_doc_observation(required_doc_ids, selected_evidence, per_doc_observation)
        observed_doc_ids = [doc_id for doc_id in required_doc_ids if normalize_text(str(per_doc_observation.get(doc_id, "")))]
        if len(observed_doc_ids) < 2:
            return fallback, "inductive_validation"
        theme_candidates = _normalize_theme_candidates(payload.get("theme_candidates", {}) or {})
        shared_themes = [normalize_text(item) for item in payload.get("shared_themes", []) if normalize_text(item)]
        if not shared_themes:
            shared_themes = _derive_shared_themes(
                observed_doc_ids=observed_doc_ids,
                theme_candidates=theme_candidates,
                selected_evidence=selected_evidence,
            )
        synthesis_basis = normalize_text(payload.get("synthesis_basis", ""))
        synthesis = normalize_text(payload.get("evidence_backed_synthesis", "") or payload.get("synthesis", ""))
        evidence_by_id = {row["evidence_id"]: row for row in selected_evidence}
        used_evidence_ids = list(dict.fromkeys(list(used_evidence_ids) + _default_doc_used_evidence_ids(observed_doc_ids, selected_evidence)))
        covered_docs = {
            evidence_by_id[evidence_id]["doc_id"]
            for evidence_id in used_evidence_ids
            if evidence_id in evidence_by_id
        }
        if len(covered_docs & set(observed_doc_ids)) < 2:
            return fallback, "inductive_validation"
        if len(shared_themes) < 1:
            return fallback, "inductive_validation"
        if not synthesis:
            synthesis = _compose_inductive_synthesis(
                observed_doc_ids=observed_doc_ids,
                per_doc_observation=per_doc_observation,
            )
        if not synthesis and not synthesis_basis:
            return fallback, "inductive_validation"
        if is_broad_inductive_query(query):
            doc_rows = _rows_by_doc(selected_evidence)
            diversity_keys = set()
            for doc_id in observed_doc_ids:
                rows = doc_rows.get(doc_id, [])
                if not rows:
                    continue
                diversity_keys.add(
                    _candidate_diversity_key(
                        {
                            "doc_id": doc_id,
                            "file_name": rows[0]["file_name"],
                            "short_title": _short_report_title(rows[0]["file_name"]),
                            "cleaned_topic_stem": clean_topic_stem(_title_topic_hint(rows[0]["file_name"])),
                            "cleaned_title_stem": clean_report_stem(_short_report_title(rows[0]["file_name"])),
                            "cluster_key": "",
                            "source_key": _report_source_key(rows[0]["file_name"]),
                            "topic_key": _title_topic_hint(rows[0]["file_name"]),
                        }
                    )
                )
            diversity_keys = {key for key in diversity_keys if key}
            if len(diversity_keys) < 2:
                return fallback, "inductive_validation"
        final_answer = _compose_inductive_answer(
            query=query,
            observed_doc_ids=observed_doc_ids,
            selected_evidence=selected_evidence,
            per_doc_observation=per_doc_observation,
            shared_themes=shared_themes,
            synthesis_basis=synthesis_basis,
            synthesis=synthesis,
        )
        normalized_final = normalize_for_match(final_answer)
        if _is_noise_topic_text(final_answer) or len(normalized_final) < 8:
            return fallback, "inductive_validation"
        combined_inductive_text = normalize_text("\n".join([final_answer, synthesis_basis, synthesis, *shared_themes]))
        observation_hits = sum(
            1
            for doc_id in observed_doc_ids
            if _has_focus_overlap(combined_inductive_text, normalize_text(str(per_doc_observation[doc_id])))
        )
        if observation_hits < 1:
            return fallback, "inductive_validation"
        if _is_single_evidence_rephrase(final_answer, selected_evidence):
            return fallback, "inductive_validation"

    return {
        "final_answer": final_answer,
        "evidence_summary": evidence_summary,
        "uncertainty_note": uncertainty_note,
        "used_evidence_ids": used_evidence_ids,
    }, None



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
        self.llm_provider = "local"
        self.llm_model = model_name
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

    def _generate_json_payload(self, prompt_builder: Any, **prompt_kwargs: Any) -> tuple[dict[str, Any] | None, str, int]:
        raw_model_output = ""
        json_parse_failures = 0
        parsed_payload: dict[str, Any] | None = None
        for strict_json in (False, True):
            prompt = prompt_builder(strict_json=strict_json, **prompt_kwargs)
            raw_model_output = self._generate(prompt)
            parsed_payload = parse_model_json(raw_model_output)
            if parsed_payload is not None:
                break
            json_parse_failures += 1
        return parsed_payload, raw_model_output, json_parse_failures

    def _generate_comparison_payload(
        self,
        *,
        query: str,
        selected_evidence: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, str, int]:
        focus_payload, focus_output, focus_failures = self._generate_json_payload(
            build_comparison_focus_prompt,
            query=query,
            evidence_rows=selected_evidence,
        )
        if focus_payload is None:
            return None, focus_output, focus_failures
        doc_focus_map = {
            str(doc_id): normalize_text(str(summary))
            for doc_id, summary in dict(focus_payload.get("doc_focus_map", {}) or {}).items()
            if normalize_text(str(summary))
        }
        synthesis_payload, synthesis_output, synthesis_failures = self._generate_json_payload(
            build_comparison_synthesis_prompt,
            query=query,
            evidence_rows=selected_evidence,
            doc_focus_map=doc_focus_map,
        )
        if synthesis_payload is None:
            return None, f"{focus_output}\n\n---\n\n{synthesis_output}", focus_failures + synthesis_failures
        synthesis_payload["doc_focus_map"] = doc_focus_map
        merged_used_evidence_ids = list(
            dict.fromkeys(
                list(focus_payload.get("used_evidence_ids", []))
                + list(synthesis_payload.get("used_evidence_ids", []))
            )
        )
        synthesis_payload["used_evidence_ids"] = merged_used_evidence_ids
        return synthesis_payload, f"{focus_output}\n\n---\n\n{synthesis_output}", focus_failures + synthesis_failures

    def _generate_inductive_payload(
        self,
        *,
        query: str,
        selected_evidence: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, str, int]:
        observation_payload, observation_output, observation_failures = self._generate_json_payload(
            build_inductive_observation_prompt,
            query=query,
            evidence_rows=selected_evidence,
        )
        if observation_payload is None:
            return None, observation_output, observation_failures
        per_doc_observation = {
            str(doc_id): normalize_text(str(summary))
            for doc_id, summary in dict(observation_payload.get("per_doc_observation", {}) or {}).items()
            if normalize_text(str(summary))
        }
        theme_candidates = _normalize_theme_candidates(observation_payload.get("theme_candidates", {}) or {})
        synthesis_payload, synthesis_output, synthesis_failures = self._generate_json_payload(
            build_inductive_synthesis_prompt,
            query=query,
            evidence_rows=selected_evidence,
            per_doc_observation=per_doc_observation,
            theme_candidates=theme_candidates,
        )
        if synthesis_payload is None:
            return None, f"{observation_output}\n\n---\n\n{synthesis_output}", observation_failures + synthesis_failures
        synthesis_payload["per_doc_observation"] = per_doc_observation
        synthesis_payload["theme_candidates"] = theme_candidates
        merged_used_evidence_ids = list(
            dict.fromkeys(
                list(observation_payload.get("used_evidence_ids", []))
                + list(synthesis_payload.get("used_evidence_ids", []))
            )
        )
        synthesis_payload["used_evidence_ids"] = merged_used_evidence_ids
        return synthesis_payload, f"{observation_output}\n\n---\n\n{synthesis_output}", observation_failures + synthesis_failures

    def answer(
        self,
        *,
        query: str,
        question_type: str,
        retrieval_result: dict[str, Any],
        query_domain_hint: str = "",
    ) -> dict[str, Any]:
        question_type = question_type or infer_question_type(query)
        answer_mode = infer_answer_mode(query, question_type)
        fact_subtype = infer_fact_subtype(query, answer_mode)
        query_domain_buckets = resolve_query_domain_buckets(query, domain_hint=query_domain_hint)
        query_domain_bucket = query_domain_buckets[0] if query_domain_buckets else ""
        numeric_query = is_numeric_or_table_query(query)
        selected_evidence, doc_candidates, selected_doc_ids, forced_abstain_reason, doc_guard_triggered = _prepare_evidence(
            query=query,
            question_type=question_type,
            answer_mode=answer_mode,
            fact_subtype=fact_subtype,
            query_domain_bucket=query_domain_bucket,
            query_domain_buckets=query_domain_buckets,
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

        if answer_mode == "comparison":
            parsed_payload, raw_model_output, json_parse_failures = self._generate_comparison_payload(
                query=query,
                selected_evidence=selected_evidence,
            )
        elif answer_mode == "inductive":
            parsed_payload, raw_model_output, json_parse_failures = self._generate_inductive_payload(
                query=query,
                selected_evidence=selected_evidence,
            )
        else:
            parsed_payload, raw_model_output, json_parse_failures = self._generate_json_payload(
                build_answer_prompt,
                query=query,
                question_type=question_type,
                answer_mode=answer_mode,
                evidence_rows=selected_evidence,
                numeric_query=numeric_query,
            )
        if parsed_payload is None:
            parsed_payload = _fallback_payload(answer_mode, selected_evidence, fact_subtype=fact_subtype, query=query)
            fallback_used = True
            fallback_source = "invalid_model_json"

        generation_latency_ms = round((perf_counter() - generation_start) * 1000, 2)
        if fallback_source == "invalid_model_json":
            payload = parsed_payload or {}
        else:
            payload, repair_fallback_reason = _repair_answer_payload(
                query=query,
                answer_mode=answer_mode,
                fact_subtype=fact_subtype,
                payload=parsed_payload or {},
                selected_evidence=selected_evidence,
            )
            if repair_fallback_reason is not None:
                fallback_used = True
                fallback_source = repair_fallback_reason
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
        support_validation["domain_mismatch"] = _selected_domain_mismatch(
            query_domain_buckets,
            selected_domain_buckets,
            answer_mode=answer_mode,
        )

        if answer_mode != "report_lookup" and not support_validation.get("supported", False):
            fallback_payload = _fallback_payload(answer_mode, selected_evidence, fact_subtype=fact_subtype, query=query)
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
            support_validation["domain_mismatch"] = _selected_domain_mismatch(
                query_domain_buckets,
                selected_domain_buckets,
                answer_mode=answer_mode,
            )

        effective_json_failures = json_parse_failures
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
        citation_evidence_ids = []
        if not abstained:
            citation_payload = dict(payload)
            citation_payload.update(
                {
                    "final_answer": final_answer,
                    "evidence_summary": evidence_summary,
                    "uncertainty_note": uncertainty_note,
                }
            )
            citation_evidence_ids = _refine_citation_evidence_ids(
                answer_mode=answer_mode,
                payload=citation_payload,
                selected_evidence=selected_evidence,
                used_evidence_ids=used_evidence_ids,
            )
        citations = build_citations(used_evidence_ids=citation_evidence_ids, evidence_rows=selected_evidence) if not abstained else []
        used_evidence_ids = citation_evidence_ids if not abstained else []
        return {
            "query": query,
            "question_type": question_type,
            "query_intent": answer_mode,
            "answer_mode": answer_mode,
            "fact_subtype": fact_subtype,
            "llm_provider": getattr(self, "llm_provider", "local"),
            "llm_model": getattr(self, "llm_model", getattr(self, "model_name", "")),
            "answer_source": "deterministic" if fallback_used else "llm",
            "fallback_used": fallback_used,
            "fallback_reason": fallback_source,
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
            "query_domain_buckets": query_domain_buckets,
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
        support_validation["domain_mismatch"] = False
        return {
            "query": query,
            "question_type": question_type,
            "query_intent": answer_mode,
            "answer_mode": answer_mode,
            "fact_subtype": fact_subtype,
            "llm_provider": getattr(self, "llm_provider", "local"),
            "llm_model": getattr(self, "llm_model", getattr(self, "model_name", "")),
            "answer_source": "deterministic",
            "fallback_used": True,
            "fallback_reason": abstain_reason,
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


def _strip_code_fences(text: str) -> str:
    payload = text.strip()
    if payload.startswith("```"):
        payload = payload.strip("`").strip()
        if payload.lower().startswith("json"):
            payload = payload[4:].strip()
    if payload.endswith("```"):
        payload = payload.rstrip("`").strip()
    return payload


def parse_model_json(text: str) -> dict[str, Any] | None:
    payload = _strip_code_fences(text)
    if not payload:
        return None
    decoder = json.JSONDecoder()
    best_end = -1
    best_value: Any = None
    for index, char in enumerate(payload):
        if char not in "{[":
            continue
        try:
            value, end = decoder.raw_decode(payload, index)
        except json.JSONDecodeError:
            continue
        if end > best_end:
            best_end = end
            best_value = value
    return best_value if isinstance(best_value, dict) else None
