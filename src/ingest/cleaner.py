from __future__ import annotations

from collections import Counter
from typing import Any

from src.utils.text_utils import is_probable_noise_line, normalize_line, normalize_text

DISCLAIMER_KEYWORDS = (
    "免责声明",
    "风险提示",
    "投资评级说明",
    "评级说明",
    "分析师声明",
    "法律声明",
    "披露声明",
)
APPENDIX_KEYWORDS = (
    "相关报告",
    "报告汇总",
    "图表目录",
    "附录",
    "投资评级说明",
    "评级说明",
)
WEB_EXPORT_NOISE_KEYWORDS = (
    "www.eastmoney.com",
    "choice数据",
    "查看pdf原文",
    "研报大全",
    "数据中心",
    "热门个股",
    "相关基金",
    "回到顶部",
    "意见反馈",
)


def _detect_boilerplate(page_records: list[dict[str, Any]]) -> set[str]:
    if len(page_records) < 3:
        return set()
    seen = Counter()
    original_line_by_key: dict[str, str] = {}
    for record in page_records:
        lines = [line.strip() for line in record["text"].split("\n") if line.strip()]
        candidates = lines[:2] + lines[-2:]
        for line in candidates:
            key = normalize_line(line)
            if not key or len(key) < 3 or len(line) > 80 or is_probable_noise_line(line):
                continue
            seen[key] += 1
            original_line_by_key.setdefault(key, line)
    threshold = max(3, int(len(page_records) * 0.4))
    return {original_line_by_key[key] for key, count in seen.items() if count >= threshold}


def _clean_block_lines(lines: list[str], removable_lines: set[str]) -> list[str]:
    cleaned: list[str] = []
    for line in lines:
        normalized = normalize_text(line)
        if not normalized:
            continue
        if normalized in removable_lines:
            continue
        if is_probable_noise_line(normalized):
            continue
        cleaned.append(normalized)
    return cleaned


def _is_disclaimer_text(text: str, *, page_num: int, total_pages: int) -> bool:
    normalized = normalize_text(text)
    if not normalized or total_pages <= 0:
        return False
    if page_num < max(2, int(total_pages * 0.6)):
        return False
    return any(keyword in normalized for keyword in DISCLAIMER_KEYWORDS)


def _is_appendix_text(text: str, *, page_num: int, total_pages: int) -> bool:
    normalized = normalize_text(text)
    if not normalized or total_pages <= 0:
        return False
    if page_num < max(3, int(total_pages * 0.5)):
        return False
    return any(keyword in normalized for keyword in APPENDIX_KEYWORDS)


def _is_web_export_noise(text: str, *, source_type: str) -> bool:
    if source_type != "web_export_pdf":
        return False
    normalized = normalize_text(text).lower()
    return any(keyword in normalized for keyword in WEB_EXPORT_NOISE_KEYWORDS)


def _clean_elements(
    elements: list[dict[str, Any]],
    removable_lines: set[str],
    *,
    source_type: str,
    page_num: int,
    total_pages: int,
) -> list[dict[str, Any]]:
    cleaned_elements: list[dict[str, Any]] = []
    for element in elements:
        raw_lines = element.get("raw_lines") or element.get("text", "").split("\n")
        kept_lines = _clean_block_lines(list(raw_lines), removable_lines)
        cleaned_text = normalize_text("\n".join(kept_lines))
        if not cleaned_text:
            continue
        if _is_disclaimer_text(cleaned_text, page_num=page_num, total_pages=total_pages):
            continue
        if _is_appendix_text(cleaned_text, page_num=page_num, total_pages=total_pages):
            continue
        if _is_web_export_noise(cleaned_text, source_type=source_type):
            continue
        cleaned = dict(element)
        cleaned["raw_lines"] = kept_lines
        cleaned["text"] = cleaned_text
        cleaned_elements.append(cleaned)
    return cleaned_elements


def clean_pages(page_records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    removable_lines = _detect_boilerplate(page_records)
    cleaned_records: list[dict[str, Any]] = []
    notes: list[dict[str, Any]] = []
    total_pages = len(page_records)

    for record in page_records:
        cleaned_blocks: list[dict[str, Any]] = []
        for block in record.get("blocks", []):
            raw_lines = block.get("raw_lines") or block["text"].split("\n")
            kept_lines = _clean_block_lines(raw_lines, removable_lines)
            cleaned_text = normalize_text("\n".join(kept_lines))
            if not cleaned_text:
                continue
            cleaned_block = dict(block)
            cleaned_block["text"] = cleaned_text
            cleaned_block["raw_lines"] = kept_lines
            cleaned_blocks.append(cleaned_block)

        cleaned_elements = _clean_elements(
            record.get("elements", []),
            removable_lines,
            source_type=record.get("source_type", "report_pdf"),
            page_num=int(record["page_num"]),
            total_pages=total_pages,
        )
        cleaned_text = normalize_text(
            "\n\n".join(
                element["text"]
                for element in cleaned_elements
                if element.get("element_type") not in {"noise"} and element.get("text")
            )
        )
        if not cleaned_text:
            cleaned_text = normalize_text("\n\n".join(block["text"] for block in cleaned_blocks))
        cleaned = dict(record)
        cleaned["blocks"] = cleaned_blocks
        cleaned["elements"] = cleaned_elements
        cleaned["text"] = cleaned_text
        cleaned["char_count"] = len(cleaned_text)
        cleaned_records.append(cleaned)

        if not cleaned_text:
            notes.append(
                {
                    "doc_id": record["doc_id"],
                    "file_name": record["file_name"],
                    "page_num": record["page_num"],
                    "issue_type": "empty_after_cleaning",
                    "detail": "Page text became empty after normalization.",
                }
            )
    return cleaned_records, notes
