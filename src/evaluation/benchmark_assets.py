from __future__ import annotations

from collections import defaultdict
from hashlib import sha1
from pathlib import Path
import re
from typing import Any

from src.utils.io import write_jsonl
from src.utils.text_utils import (
    extract_numeric_tokens,
    extract_terms,
    first_sentence,
    guess_industry,
    industry_from_file_name,
    normalize_for_match,
    normalize_text,
    short_title_from_file_name,
    strip_file_extension,
    topic_hint_from_title,
)

INDUSTRY_PREFIX_TO_KEY = {
    "\u534a\u5bfc\u4f53": "semiconductor",
    "\u65b0\u80fd\u6e90": "new_energy",
    "\u6d88\u8d39": "consumer",
    "\u767d\u9152": "liquor",
}

INDUSTRY_LABELS = {
    "semiconductor": "\u7535\u5b50/\u534a\u5bfc\u4f53",
    "new_energy": "\u65b0\u80fd\u6e90",
    "consumer": "\u6d88\u8d39",
    "liquor": "\u767d\u9152",
}

ANSWER_NOISE_PATTERNS = (
    "\u7ae0\u8282\u8def\u5f84",
    "\u8868\u683c\u884c\u7ec4",
    "\u8868\u683c\u6458\u8981",
    "\u56fe\u8868\u76ee\u5f55",
    "\u56fe\u8868\u6458\u8981",
    "\u56fe\u8868 ",
    "\u8868\u683c ",
)


def make_doc_key(file_name: str, duplicate_count: int = 1) -> str:
    title = short_title_from_file_name(file_name)
    base = normalize_for_match(title) or normalize_for_match(strip_file_extension(Path(file_name).name)) or "doc"
    if duplicate_count <= 1:
        return base
    digest = sha1(file_name.encode("utf-8")).hexdigest()[:8]
    return f"{base}-{digest}"


def industry_label(industry: str) -> str:
    return INDUSTRY_LABELS.get(industry, "\u76f8\u5173\u884c\u4e1a")


def _row_text(row: dict[str, Any]) -> str:
    return normalize_text(row.get("child_text") or row.get("text") or "")


def _row_terms(row: dict[str, Any], *, top_k: int = 6) -> list[str]:
    text = _row_text(row)
    if not text:
        return []
    return extract_terms(text, top_k=top_k)


def _summary_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(rows, key=_summary_score, reverse=True)
    return ranked[0]


def _summary_score(row: dict[str, Any]) -> float:
    text = _row_text(row)
    if not text:
        return -1e9
    score = 0.0
    if row.get("element_type") == "paragraph":
        score += 3.0
    if row.get("element_type") in {"figure_context", "figure_caption"}:
        score += 1.0
    if row.get("chunk_type") != "table_like":
        score += 1.0
    length = len(text)
    if 80 <= length <= 320:
        score += 2.5
    elif 40 <= length <= 420:
        score += 1.5
    score += max(0.0, 2.0 - (max(int(row.get("page_start", 1)) - 1, 0) * 0.4))
    score += min(len(_row_terms(row, top_k=6)), 6) * 0.2
    return score


def _fact_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(rows, key=_fact_score, reverse=True)
    return ranked[0]


def _fact_score(row: dict[str, Any]) -> float:
    text = _row_text(row)
    if not text:
        return -1e9
    numeric_tokens = extract_numeric_tokens(text)
    score = float(len(numeric_tokens))
    if row.get("chunk_type") == "table_like" or row.get("element_type") == "table":
        score += 3.0
    if row.get("element_type") == "figure":
        score += 1.0
    length = len(text)
    if 40 <= length <= 260:
        score += 1.5
    elif 20 <= length <= 360:
        score += 0.5
    score += max(0.0, 1.5 - (max(int(row.get("page_start", 1)) - 1, 0) * 0.3))
    if not numeric_tokens:
        score -= 1.0
    return score


def _sentence_from_row(row: dict[str, Any], *, max_chars: int = 80) -> str:
    text = _row_text(row)
    if not text:
        return ""
    return first_sentence(text, max_chars=max_chars)


def _clean_answer_text(text: str) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return ""
    lines = [line.strip() for line in normalized.split("\n") if line.strip()]
    filtered: list[str] = []
    for line in lines:
        if any(line.startswith(pattern) for pattern in ANSWER_NOISE_PATTERNS):
            continue
        if re.fullmatch(r"[\W_]+", line):
            continue
        filtered.append(line)
    return normalize_text("\n".join(filtered))


def _clean_answer_fragment(text: str, *, max_chars: int = 88) -> str:
    cleaned = _clean_answer_text(text)
    if not cleaned:
        return ""
    return first_sentence(cleaned, max_chars=max_chars)


def _good_answer_fragment(text: str) -> bool:
    cleaned = _clean_answer_fragment(text)
    if not cleaned:
        return False
    if any(cleaned.startswith(pattern) for pattern in ANSWER_NOISE_PATTERNS):
        return False
    if re.fullmatch(r"\d+(?:\.\d+)?", cleaned):
        return False
    if len(normalize_for_match(cleaned)) < 8:
        return False
    return True


def _keyword_hints(*texts: str, limit: int = 6) -> list[str]:
    merged = normalize_text("\n".join(text for text in texts if text))
    if not merged:
        return []
    return extract_terms(merged, top_k=limit)


def _page_hints(*rows: dict[str, Any]) -> list[int]:
    pages: set[int] = set()
    for row in rows:
        for page in range(int(row.get("page_start", 1)), int(row.get("page_end", row.get("page_start", 1))) + 1):
            pages.add(page)
    return sorted(pages)


def build_doc_manifest(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in chunks:
        grouped[row["doc_id"]].append(row)

    duplicate_counter: defaultdict[str, int] = defaultdict(int)
    for rows in grouped.values():
        file_name = rows[0]["file_name"]
        duplicate_counter[normalize_for_match(short_title_from_file_name(file_name)) or "doc"] += 1

    manifest: list[dict[str, Any]] = []
    for doc_id, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: (int(row.get("page_start", 1)), int(row.get("page_end", row.get("page_start", 1))), row["chunk_id"]))
        file_name = ordered[0]["file_name"]
        short_title = short_title_from_file_name(file_name)
        base_key = normalize_for_match(short_title) or "doc"
        summary_row = _summary_row(ordered)
        fact_row = _fact_row(ordered)
        page_numbers = {
            page
            for row in ordered
            for page in range(int(row.get("page_start", 1)), int(row.get("page_end", row.get("page_start", 1))) + 1)
        }
        manifest.append(
            {
                "doc_id": doc_id,
                "doc_key": make_doc_key(file_name, duplicate_counter[base_key]),
                "base_doc_key": base_key,
                "file_name": file_name,
                "short_title": short_title,
                "title_topic": topic_hint_from_title(short_title),
                "title_terms": _keyword_hints(short_title, limit=6),
                "industry": industry_from_file_name(file_name),
                "page_count": len(page_numbers),
                "chunk_count": len(ordered),
                "summary_chunk_id": summary_row["chunk_id"],
                "summary_page_start": int(summary_row.get("page_start", 1)),
                "summary_page_end": int(summary_row.get("page_end", summary_row.get("page_start", 1))),
                "summary_fragment": _sentence_from_row(summary_row, max_chars=72),
                "summary_terms": _row_terms(summary_row, top_k=6),
                "fact_chunk_id": fact_row["chunk_id"],
                "fact_page_start": int(fact_row.get("page_start", 1)),
                "fact_page_end": int(fact_row.get("page_end", fact_row.get("page_start", 1))),
                "fact_fragment": _sentence_from_row(fact_row, max_chars=88),
                "fact_terms": _row_terms(fact_row, top_k=6),
            }
        )

    manifest.sort(key=lambda row: (row["industry"], row["file_name"]))
    return manifest


def write_doc_manifest(manifest: list[dict[str, Any]], output_path: Path) -> None:
    write_jsonl(output_path, manifest)


def _evenly_spaced_docs(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(rows) < count:
        raise ValueError(f"expected at least {count} docs, got {len(rows)}")
    if count == 1:
        return [rows[len(rows) // 2]]
    indices = []
    for idx in range(count):
        position = round(idx * (len(rows) - 1) / (count - 1))
        indices.append(position)
    deduped = list(dict.fromkeys(indices))
    if len(deduped) < count:
        cursor = 0
        while len(deduped) < count and cursor < len(rows):
            if cursor not in deduped:
                deduped.append(cursor)
            cursor += 1
        deduped.sort()
    return [rows[index] for index in deduped[:count]]


def _report_lookup_row(industry: str, ordinal: int, doc: dict[str, Any]) -> dict[str, Any]:
    topic = doc["title_topic"] or doc["short_title"]
    question_id = f"{industry}_report_lookup_{ordinal:02d}"
    return {
        "qid": question_id,
        "question_id": question_id,
        "question_type": "fact",
        "intent": "report_lookup",
        "industry": industry,
        "query": f"哪份报告讨论了“{topic}”？",
        "target_doc_keys": [doc["doc_key"]],
        "page_hints": [doc["summary_page_start"]],
        "keyword_hints": _keyword_hints(doc["short_title"], doc["summary_fragment"]),
        "gold_answer_hint": doc["short_title"],
    }


def _numeric_fact_row(industry: str, ordinal: int, doc: dict[str, Any]) -> dict[str, Any]:
    focus_terms = doc["fact_terms"][:3] or doc["title_terms"][:3]
    focus = "、".join(focus_terms) if focus_terms else (doc["title_topic"] or doc["short_title"])
    question_id = f"{industry}_numeric_fact_{ordinal:02d}"
    return {
        "qid": question_id,
        "question_id": question_id,
        "question_type": "fact",
        "intent": "numeric_fact",
        "industry": industry,
        "query": f"报告中关于“{focus}”的关键数据或变化是怎样的？",
        "target_doc_keys": [doc["doc_key"]],
        "page_hints": _page_hints(
            {
                "page_start": doc["fact_page_start"],
                "page_end": doc["fact_page_end"],
            }
        ),
        "keyword_hints": _keyword_hints(doc["short_title"], doc["fact_fragment"]),
        "gold_answer_hint": doc["fact_fragment"],
    }


def _comparison_row(industry: str, ordinal: int, first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    question_id = f"{industry}_comparison_{ordinal:02d}"
    return {
        "qid": question_id,
        "question_id": question_id,
        "question_type": "comparison",
        "intent": "comparison",
        "industry": industry,
        "query": f"《{first['short_title']}》与《{second['short_title']}》分别关注哪些重点？",
        "target_doc_keys": [first["doc_key"], second["doc_key"]],
        "page_hints": sorted({first["summary_page_start"], second["summary_page_start"]}),
        "keyword_hints": _keyword_hints(first["short_title"], second["short_title"], first["summary_fragment"], second["summary_fragment"]),
        "gold_answer_hint": f"《{first['short_title']}》主要关注{first['summary_fragment']}；《{second['short_title']}》主要关注{second['summary_fragment']}。",
    }


def _inductive_row(industry: str, ordinal: int, docs: list[dict[str, Any]]) -> dict[str, Any]:
    common_terms = _keyword_hints(*(doc["summary_fragment"] for doc in docs), *(doc["short_title"] for doc in docs), limit=6)
    question_id = f"{industry}_inductive_{ordinal:02d}"
    return {
        "qid": question_id,
        "question_id": question_id,
        "question_type": "inductive",
        "intent": "inductive",
        "industry": industry,
        "query": f"近期{industry_label(industry)}报告共同强调了哪些主题？",
        "target_doc_keys": [doc["doc_key"] for doc in docs],
        "page_hints": sorted({doc["summary_page_start"] for doc in docs}),
        "keyword_hints": common_terms,
        "gold_answer_hint": f"共同主题围绕：{'、'.join(common_terms)}" if common_terms else "",
    }


def build_retrieval_seed(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_industry: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest:
        by_industry[row["industry"]].append(row)

    seed_rows: list[dict[str, Any]] = []
    for industry in ("semiconductor", "new_energy", "consumer", "liquor"):
        docs = list(by_industry.get(industry, []))
        selected = _evenly_spaced_docs(docs, 12)
        seed_rows.extend(
            [
                _report_lookup_row(industry, 1, selected[0]),
                _numeric_fact_row(industry, 1, selected[1]),
                _report_lookup_row(industry, 2, selected[2]),
                _numeric_fact_row(industry, 2, selected[3]),
                _report_lookup_row(industry, 3, selected[4]),
                _numeric_fact_row(industry, 3, selected[5]),
                _comparison_row(industry, 1, selected[6], selected[7]),
                _comparison_row(industry, 2, selected[8], selected[9]),
                _comparison_row(industry, 3, selected[10], selected[11]),
                _inductive_row(industry, 1, [selected[0], selected[6], selected[8]]),
                _inductive_row(industry, 2, [selected[2], selected[7], selected[10]]),
                _inductive_row(industry, 3, [selected[4], selected[9], selected[11]]),
            ]
        )
    return seed_rows


def prepare_benchmark_assets(*, chunks: list[dict[str, Any]], manifest_output_path: Path, seed_output_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = build_doc_manifest(chunks)
    seed_rows = build_retrieval_seed(manifest)
    write_doc_manifest(manifest, manifest_output_path)
    write_jsonl(seed_output_path, seed_rows)
    return manifest, seed_rows


def _select_answer_seed_rows(retrieval_seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for industry in ("semiconductor", "new_energy", "consumer", "liquor"):
        group = [row for row in retrieval_seed_rows if row.get("industry") == industry]
        selected.extend(
            [
                next(row for row in group if row.get("intent") == "report_lookup"),
                next(row for row in group if row.get("intent") == "numeric_fact"),
                next(row for row in group if row.get("intent") == "comparison"),
                next(row for row in group if row.get("intent") == "inductive"),
            ]
        )
    return selected


def _select_current_full_answer_seed_rows(retrieval_seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return list(retrieval_seed_rows)


def _answer_chunk_score(row: dict[str, Any], *, mode: str) -> float:
    raw_text = _row_text(row)
    cleaned = _clean_answer_text(raw_text)
    if not cleaned:
        return -1e9

    score = 0.0
    if mode == "numeric_fact":
        score += len(extract_numeric_tokens(cleaned)) * 2.0
        if row.get("chunk_type") == "table_like" or row.get("element_type") == "table":
            score += 3.0
        if 20 <= len(cleaned) <= 220:
            score += 2.0
    else:
        if row.get("element_type") == "paragraph":
            score += 3.0
        if row.get("chunk_type") != "table_like":
            score += 1.0
        if 40 <= len(cleaned) <= 180:
            score += 2.0
        elif 20 <= len(cleaned) <= 240:
            score += 1.0
        score += len(extract_terms(cleaned, top_k=6)) * 0.2

    if _good_answer_fragment(raw_text):
        score += 3.0
    score += max(0.0, 1.5 - max(int(row.get("page_start", 1)) - 1, 0) * 0.2)
    return score


def _answer_chunk_junk_penalty(raw_text: str) -> float:
    """Penalize non-content chunks (analyst headers, cells, dates, disclaimers)."""
    cleaned = _clean_answer_text(raw_text)
    if not cleaned:
        return 1e9
    penalty = 0.0
    lowered = cleaned.lower()
    if any(marker in lowered for marker in ("@", "证书编号", "邮箱")) or "证券分析师" in cleaned or "分析师" in cleaned:
        penalty += 20.0
    for pattern in ("投资评级", "行业走势图", "公司代码", "优于大市", "风险提示", "表头", "行业研究·行业"):
        if pattern in cleaned:
            penalty += 10.0
    normalized_len = len(normalize_for_match(cleaned))
    if normalized_len < 8:
        penalty += 30.0
    if cleaned.startswith("\uf06e") and cleaned.rstrip().endswith(":"):
        penalty += 10.0
    if re.fullmatch(r"[\d\s年月日.\-:/]+", cleaned):
        penalty += 30.0
    return penalty


def _select_answer_chunk(rows: list[dict[str, Any]], *, mode: str) -> dict[str, Any]:
    ranked = sorted(
        rows,
        key=lambda row: _answer_chunk_score(row, mode=mode) - _answer_chunk_junk_penalty(row.get("text", "")),
        reverse=True,
    )
    return ranked[0]


def _comparison_gold_answer(first_doc: dict[str, Any], second_doc: dict[str, Any], first_chunk: dict[str, Any], second_chunk: dict[str, Any]) -> str:
    first_fragment = _clean_answer_fragment(first_chunk.get("text", "")) or first_doc.get("title_topic") or first_doc["short_title"]
    second_fragment = _clean_answer_fragment(second_chunk.get("text", "")) or second_doc.get("title_topic") or second_doc["short_title"]
    return f"《{first_doc['short_title']}》主要关注{first_fragment}；《{second_doc['short_title']}》主要关注{second_fragment}。"


def _inductive_gold_answer(docs: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> str:
    candidate_terms = extract_terms(
        normalize_text(
            "\n".join(
                [
                    *(doc["short_title"] for doc in docs),
                    *(_clean_answer_text(chunk.get("text", "")) for chunk in chunks),
                ]
            )
        ),
        top_k=10,
    )
    filtered_terms = [term for term in candidate_terms if len(normalize_for_match(term)) >= 4][:5]
    if filtered_terms:
        return "共同主题包括：" + "、".join(filtered_terms)
    fallback = [doc.get("title_topic") or doc["short_title"] for doc in docs]
    return "共同主题围绕：" + "、".join(fallback[:3])


def build_answer_seed_draft(
    *,
    retrieval_seed_rows: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
    scope: str = "dev",
) -> list[dict[str, Any]]:
    manifest_by_key = {row["doc_key"]: row for row in manifest}
    chunks_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_doc[chunk["doc_id"]].append(chunk)

    draft_rows: list[dict[str, Any]] = []
    if scope == "dev":
        selected_seed_rows = _select_answer_seed_rows(retrieval_seed_rows)
    elif scope == "current-full":
        selected_seed_rows = _select_current_full_answer_seed_rows(retrieval_seed_rows)
    else:
        raise ValueError(f"Unsupported answer seed draft scope: {scope}")

    for row in selected_seed_rows:
        intent = row["intent"]
        target_docs = [manifest_by_key[doc_key] for doc_key in row["target_doc_keys"]]
        selected_chunks = [
            _select_answer_chunk(chunks_by_doc[doc["doc_id"]], mode="numeric_fact" if intent == "numeric_fact" else "summary")
            for doc in target_docs
        ]

        if intent == "report_lookup":
            gold_answer = target_docs[0]["short_title"]
        elif intent == "numeric_fact":
            gold_answer = _clean_answer_fragment(selected_chunks[0].get("text", ""), max_chars=108) or target_docs[0]["short_title"]
        elif intent == "comparison":
            gold_answer = _comparison_gold_answer(target_docs[0], target_docs[1], selected_chunks[0], selected_chunks[1])
        else:
            gold_answer = _inductive_gold_answer(target_docs, selected_chunks)

        draft_rows.append(
            {
                "question_id": row["question_id"],
                "query": row["query"],
                "question_type": row["question_type"],
                "intent": row["intent"],
                "industry": row["industry"],
                "target_doc_keys": row["target_doc_keys"],
                "target_titles": [doc["short_title"] for doc in target_docs],
                "gold_answer": gold_answer,
                "gold_chunk_ids": [chunk["chunk_id"] for chunk in selected_chunks],
                "candidate_snippets": [_clean_answer_fragment(chunk.get("text", ""), max_chars=108) for chunk in selected_chunks],
                "review_notes": f"draft_auto_generated_from_retrieval_seed:{scope}",
            }
        )
    return draft_rows


def write_answer_seed_draft(draft_rows: list[dict[str, Any]], output_path: Path) -> None:
    write_jsonl(output_path, draft_rows)


# ---------------------------------------------------------------------------
# Agent benchmark seeds (checklist v3.0 §P3 benchmark categories)
# ---------------------------------------------------------------------------

AGENT_BENCHMARK_INDUSTRIES = ("semiconductor", "new_energy", "consumer", "liquor")

_AGENT_CATEGORY_PREFIX = {
    "agent_recovery": "rec",
    "agent_multi_source": "ms",
    "agent_numeric_missing": "num",
    "agent_abstain": "abs",
}

_AGENT_ABSTAIN_QUERY_TEMPLATES = {
    "semiconductor": "半导体行业2027年度资本开支总额是多少？",
    "new_energy": "新能源行业2027年度锂电装机总量是多少？",
    "consumer": "消费行业2027年度社会零售总额预测是多少？",
    "liquor": "白酒行业2027年度销售回款总额是多少？",
}

# Decoy numbers embedded in numeric_missing queries: absent from the corpus,
# so the deterministic grade flags numeric_missing on round 1 (recovery design).
_AGENT_DECOY_NUMBERS = {
    "semiconductor": "47.2%",
    "new_energy": "53.6%",
    "consumer": "61.4%",
    "liquor": "39.8%",
}


def _anchored_comparison_query(first_doc: dict[str, Any]) -> str:
    """Single-doc anchored comparison: round-1 retrieval pins on the first
    report's title, so the deterministic grade fails with source_diversity_missing."""
    return f"《{first_doc['short_title']}》与同行业其他研报的关注重点有何不同？"


def _numeric_missing_query(industry: str, doc: dict[str, Any]) -> str:
    focus_terms = (doc.get("fact_terms") or doc.get("title_terms") or [])[:3]
    focus = "、".join(focus_terms) if focus_terms else (doc.get("title_topic") or doc["short_title"])
    return f"报告中关于“{focus}”的关键数据从{_AGENT_DECOY_NUMBERS[industry]}变化到多少？"


def _agent_row(
    *,
    question_id: str,
    category: str,
    query: str,
    question_type: str,
    intent: str,
    industry: str,
    target_docs: list[dict[str, Any]],
    gold_chunks: list[dict[str, Any]],
    gold_answer: str,
    must_abstain: bool = False,
    must_recover: bool = False,
    expected_first_failure: str = "",
    review_notes: str,
    domain_hint: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "question_id": question_id,
        "category": category,
        "query": query,
        "question_type": question_type,
        "intent": intent,
        "industry": industry,
        "target_doc_keys": [doc["doc_key"] for doc in target_docs],
        "target_titles": [doc["short_title"] for doc in target_docs],
        "gold_answer": gold_answer,
        "gold_chunk_ids": [chunk["chunk_id"] for chunk in gold_chunks],
        "candidate_snippets": [_clean_answer_fragment(chunk.get("text", ""), max_chars=108) for chunk in gold_chunks],
        "must_abstain": must_abstain,
        "must_recover": must_recover,
        "expected_first_failure": expected_first_failure,
        "review_notes": review_notes,
        "domain_hint": industry if domain_hint is None else domain_hint,
    }
    return row


def _agent_ordinal(industry: str, category: str) -> str:
    return f"{industry}_{_AGENT_CATEGORY_PREFIX[category]}_01"


def _agent_target_docs(manifest_by_key: dict[str, dict[str, Any]], seed_row: dict[str, Any]) -> list[dict[str, Any]]:
    return [manifest_by_key[doc_key] for doc_key in seed_row.get("target_doc_keys", [])]


def _agent_select_chunks(
    chunks_by_doc: dict[str, list[dict[str, Any]]],
    target_docs: list[dict[str, Any]],
    *,
    mode: str,
) -> list[dict[str, Any]]:
    return [_select_answer_chunk(chunks_by_doc[doc["doc_id"]], mode=mode) for doc in target_docs]


def build_agent_seed_draft(
    *,
    retrieval_seed_rows: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the 4-category dev agent benchmark seed (20 rows) from retrieval seeds.

    Categories (checklist v3.0 §P3): agent_recovery x4, agent_multi_source x8,
    agent_numeric_missing x4, agent_abstain x4 (1 per industry each).
    """
    manifest_by_key = {row["doc_key"]: row for row in manifest}
    chunks_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_doc[chunk["doc_id"]].append(chunk)

    rows: list[dict[str, Any]] = []
    for industry in AGENT_BENCHMARK_INDUSTRIES:
        group = [row for row in retrieval_seed_rows if row.get("industry") == industry]
        comparisons = [row for row in group if row.get("intent") == "comparison"]
        numerics = [row for row in group if row.get("intent") == "numeric_fact"]
        inductives = [row for row in group if row.get("intent") == "inductive"]

        # agent_recovery: single-doc anchored comparison query (round-1 grade
        # fails with source_diversity_missing), same dual-doc targets.
        recovery_seed = comparisons[1]
        recovery_docs = _agent_target_docs(manifest_by_key, recovery_seed)
        recovery_chunks = _agent_select_chunks(chunks_by_doc, recovery_docs, mode="summary")
        rows.append(
            _agent_row(
                question_id=_agent_ordinal(industry, "agent_recovery"),
                category="agent_recovery",
                query=_anchored_comparison_query(recovery_docs[0]),
                question_type="comparison",
                intent="comparison",
                industry=industry,
                target_docs=recovery_docs,
                gold_chunks=recovery_chunks,
                gold_answer=_comparison_gold_answer(recovery_docs[0], recovery_docs[1], recovery_chunks[0], recovery_chunks[1]),
                must_recover=True,
                expected_first_failure="source_diversity_missing",
                review_notes="agent_recovery_anchored_comparison:dev",
            )
        )

        # agent_multi_source: one original comparison + one inductive per industry.
        comparison_seed = comparisons[0]
        comparison_docs = _agent_target_docs(manifest_by_key, comparison_seed)
        comparison_chunks = _agent_select_chunks(chunks_by_doc, comparison_docs, mode="summary")
        rows.append(
            _agent_row(
                question_id=f"{industry}_ms_comp_01",
                category="agent_multi_source",
                query=comparison_seed["query"],
                question_type="comparison",
                intent="comparison",
                industry=industry,
                target_docs=comparison_docs,
                gold_chunks=comparison_chunks,
                gold_answer=_comparison_gold_answer(
                    comparison_docs[0], comparison_docs[1], comparison_chunks[0], comparison_chunks[1]
                ),
                review_notes="agent_multi_source_comparison:dev",
            )
        )
        inductive_seed = inductives[0]
        inductive_docs = _agent_target_docs(manifest_by_key, inductive_seed)
        inductive_chunks = _agent_select_chunks(chunks_by_doc, inductive_docs, mode="summary")
        rows.append(
            _agent_row(
                question_id=f"{industry}_ms_ind_01",
                category="agent_multi_source",
                query=inductive_seed["query"],
                question_type="inductive",
                intent="inductive",
                industry=industry,
                target_docs=inductive_docs,
                gold_chunks=inductive_chunks,
                gold_answer=_inductive_gold_answer(inductive_docs, inductive_chunks),
                review_notes="agent_multi_source_inductive:dev",
            )
        )

        # agent_numeric_missing: decoy-number query forces round-1 numeric_missing.
        numeric_seed = numerics[0]
        numeric_docs = _agent_target_docs(manifest_by_key, numeric_seed)
        numeric_chunks = _agent_select_chunks(chunks_by_doc, numeric_docs, mode="numeric_fact")
        rows.append(
            _agent_row(
                question_id=_agent_ordinal(industry, "agent_numeric_missing"),
                category="agent_numeric_missing",
                query=_numeric_missing_query(industry, numeric_docs[0]),
                question_type="fact",
                intent="numeric_fact",
                industry=industry,
                target_docs=numeric_docs,
                gold_chunks=numeric_chunks,
                gold_answer=_clean_answer_fragment(numeric_chunks[0].get("text", ""), max_chars=108)
                or numeric_docs[0]["short_title"],
                must_recover=True,
                expected_first_failure="numeric_missing",
                review_notes="agent_numeric_missing_decoy:dev",
            )
        )

        # agent_abstain: synthetic out-of-doc query, no gold targets.
        rows.append(
            _agent_row(
                question_id=_agent_ordinal(industry, "agent_abstain"),
                category="agent_abstain",
                query=_AGENT_ABSTAIN_QUERY_TEMPLATES[industry],
                question_type="fact",
                intent="numeric_fact",
                industry=industry,
                target_docs=[],
                gold_chunks=[],
                gold_answer="",
                must_abstain=True,
                expected_first_failure="no_evidence",
                review_notes="agent_abstain_synthetic:dev",
            )
        )

    return rows


def write_agent_seed_draft(draft_rows: list[dict[str, Any]], output_path: Path) -> None:
    write_jsonl(output_path, draft_rows)

