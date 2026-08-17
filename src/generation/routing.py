"""P1 module extracted from src/generation/answerer.py (behavior preserved)."""
from __future__ import annotations

from src.utils.text_utils import extract_terms, normalize_for_match, normalize_text
import re


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


QUERY_REPORT_TITLE_RE = re.compile(r"\u300a([^\u300b]{2,120})\u300b")


FALLBACK_ANSWER = "信息不足，暂时无法给出可靠答案。"


def is_fallback_answer(text: str) -> bool:
    return normalize_text(text) == FALLBACK_ANSWER


def _fallback_abstain_reason(*, final_answer: str, fallback_source: str | None) -> str | None:
    if not is_fallback_answer(final_answer):
        return None
    return fallback_source or "unsupported_answer"


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


NORMALIZED_BROAD_INDUCTIVE_GENERIC_TERMS = {normalize_for_match(term) for term in BROAD_INDUCTIVE_GENERIC_TERMS}
NORMALIZED_COARSE_DOMAIN_QUERY_TERMS = {normalize_for_match(term) for term in COARSE_DOMAIN_QUERY_TERMS}


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


_MULTI_HOP_METRIC_HINTS = (
    "营业收入",
    "营收",
    "净利润",
    "毛利率",
    "研发费用",
    "经营性现金流",
    "归母",
)

_MULTI_HOP_OPINION_HINTS = (
    "怎么看",
    "如何看",
    "观点",
    "看好",
    "认为",
    "展望",
    "逻辑",
    "成长性",
    "前景",
    "目标价",
    "评级",
    "机构看法",
)


def is_multi_hop_query(
    query: str,
    *,
    question_type: str = "",
    answer_mode: str = "",
    fact_subtype: str = "",
) -> bool:
    """Deterministic multi-hop trigger (checklist v3.0 §P6, no LLM).

    Multi-hop only for fact-type queries that mix a numeric/metric anchor with
    an opinion/outlook question — a shape the single-hop fact path cannot answer
    in one retrieval. Comparison and inductive queries stay on the existing
    single-hop path (they already have grade/rewrite recovery; §P3 gate).
    """
    del answer_mode, fact_subtype  # reserved for future refinement
    resolved_type = question_type or infer_question_type(query)
    if resolved_type != "fact":
        return False
    normalized = normalize_text(query)
    has_metric = any(hint in normalized for hint in _MULTI_HOP_METRIC_HINTS)
    has_opinion = any(hint in normalized for hint in _MULTI_HOP_OPINION_HINTS)
    return has_metric and has_opinion


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


def _query_terms(query: str, *, top_k: int = 10) -> list[str]:
    return [normalize_for_match(term) for term in extract_terms(query, top_k=top_k) if normalize_for_match(term)]


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
