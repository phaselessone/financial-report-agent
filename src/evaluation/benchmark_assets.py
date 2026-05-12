from __future__ import annotations

from collections import defaultdict
from hashlib import sha1
from pathlib import Path
import re
from typing import Any

from src.utils.io import write_jsonl
from src.utils.text_utils import extract_numeric_tokens, extract_terms, first_sentence, guess_industry, normalize_for_match, normalize_text, strip_file_extension

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

NEW_FILE_NAME_RE = re.compile(r"^(?P<prefix>[^_]+)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<ref>[A-Za-z0-9]+)_(?P<title>.+)$")
TITLE_DATE_SUFFIX_RE = re.compile(r"(?:[-_ ]?(?:20\d{2}[-/]?\d{2}[-/]?\d{2}|\d{6,8}|\u7b2c?\d+\u7248(?:\(\u82f1\u8bd1\u4e2d\))?))+$")
TITLE_NOISE_PATTERNS = (
    "\u7814\u7a76\u62a5\u544a\u6b63\u6587 _ \u6570\u636e\u4e2d\u5fc3 _ \u4e1c\u65b9\u8d22\u5bcc\u7f51",
    "\u7814\u7a76\u62a5\u544a\u6b63\u6587 _ \u6570\u636e\u4e2d\u5fc3",
    "\u6570\u636e\u4e2d\u5fc3 _ \u4e1c\u65b9\u8d22\u5bcc\u7f51",
    "\u6570\u636e\u4e2d\u5fc3",
)
ANSWER_NOISE_PATTERNS = (
    "\u7ae0\u8282\u8def\u5f84",
    "\u8868\u683c\u884c\u7ec4",
    "\u8868\u683c\u6458\u8981",
    "\u56fe\u8868\u76ee\u5f55",
    "\u56fe\u8868\u6458\u8981",
    "\u56fe\u8868 ",
    "\u8868\u683c ",
)


def industry_from_file_name(file_name: str) -> str:
    return guess_industry(file_name)



def short_title_from_file_name(file_name: str) -> str:
    stem = strip_file_extension(Path(file_name).name)
    match = NEW_FILE_NAME_RE.match(stem)
    if match:
        title = match.group("title").strip()
    elif "\uff1a" in stem:
        title = stem.split("\uff1a", 1)[1].strip()
    elif ":" in stem:
        title = stem.split(":", 1)[1].strip()
    else:
        parts = [part for part in stem.split("-") if part]
        if len(parts) >= 3:
            title = "-".join(parts[1:-1]).strip()
        elif len(parts) >= 2:
            title = "-".join(parts[1:]).strip()
        else:
            title = stem.strip()

    normalized = normalize_text(title)
    for pattern in TITLE_NOISE_PATTERNS:
        normalized = normalized.replace(pattern, "").strip(" -_:\uff1a,\uff0c")
    normalized = TITLE_DATE_SUFFIX_RE.sub("", normalized).strip(" -_:\uff1a,\uff0c")
    return normalized or stem.strip()



def make_doc_key(file_name: str, duplicate_count: int = 1) -> str:
    title = short_title_from_file_name(file_name)
    base = normalize_for_match(title) or normalize_for_match(strip_file_extension(Path(file_name).name)) or "doc"
    if duplicate_count <= 1:
        return base
    digest = sha1(file_name.encode("utf-8")).hexdigest()[:8]
    return f"{base}-{digest}"



def topic_hint_from_title(title: str) -> str:
    normalized = normalize_text(title)
    if not normalized:
        return ""
    for separator in ("\uff1a", ":", "\u2014\u2014", "-", "\uff0c", ","):
        if separator in normalized:
            suffix = normalized.split(separator, 1)[1].strip()
            suffix = TITLE_DATE_SUFFIX_RE.sub("", suffix).strip(" -_:\uff1a,\uff0c")
            if (
                6 <= len(suffix) <= 36
                and not re.fullmatch(r"(?:20\d{2}[-/]?\d{2}[-/]?\d{2}|\d{6,8}|\u7b2c?\d+\u7248(?:\(\u82f1\u8bd1\u4e2d\))?)", suffix)
                and not suffix.isdigit()
            ):
                return suffix
    cleaned = TITLE_DATE_SUFFIX_RE.sub("", normalized).strip(" -_:\uff1a,\uff0c")
    if len(cleaned) > 36:
        return cleaned[:36].rstrip("\uff0c,\uff1b; ") + "\u2026"
    return cleaned



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


def _select_answer_chunk(rows: list[dict[str, Any]], *, mode: str) -> dict[str, Any]:
    ranked = sorted(rows, key=lambda row: _answer_chunk_score(row, mode=mode), reverse=True)
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

