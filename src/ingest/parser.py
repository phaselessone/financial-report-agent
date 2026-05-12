from __future__ import annotations

import re
from pathlib import Path
from statistics import median
from typing import Any

import fitz

from src.utils.text_utils import (
    guess_industry,
    guess_source_type,
    is_probable_noise_line,
    is_table_like,
    is_title_like,
    make_doc_id,
    normalize_text,
)

FIGURE_CAPTION_RE = re.compile(r"^(图表?|Figure)\s*[0-9一二三四五六七八九十]+", re.IGNORECASE)
TABLE_CAPTION_RE = re.compile(r"^(表|Table)\s*[0-9一二三四五六七八九十]+", re.IGNORECASE)
NUMBERED_H1_RE = re.compile(r"^(?:\d+|[一二三四五六七八九十]+)\s*[、.]")
NUMBERED_H2_RE = re.compile(r"^(?:\d+\.\d+|[(（][一二三四五六七八九十\d]+[)）])")
NUMBERED_H3_RE = re.compile(r"^\d+\.\d+\.\d+")


def _extract_block_record(block: dict[str, Any]) -> dict[str, Any] | None:
    if block.get("type") != 0:
        return None

    line_texts: list[str] = []
    span_sizes: list[float] = []
    bold_hits = 0

    for line in block.get("lines", []):
        spans = line.get("spans", [])
        pieces: list[str] = []
        for span in spans:
            text = span.get("text", "").strip()
            if not text:
                continue
            pieces.append(text)
            span_sizes.append(float(span.get("size", 0.0)))
            font_name = str(span.get("font", "")).lower()
            flags = int(span.get("flags", 0))
            if "bold" in font_name or flags & 16:
                bold_hits += 1
        if pieces:
            line_texts.append(" ".join(pieces))

    if not line_texts:
        return None

    x0, y0, x1, y1 = block.get("bbox", (0.0, 0.0, 0.0, 0.0))
    raw_lines = [normalize_text(line) for line in line_texts if normalize_text(line)]
    text = normalize_text("\n".join(raw_lines))
    if not text:
        return None

    return {
        "bbox": [float(x0), float(y0), float(x1), float(y1)],
        "text": text,
        "raw_lines": raw_lines,
        "max_font_size": max(span_sizes) if span_sizes else 0.0,
        "avg_font_size": sum(span_sizes) / len(span_sizes) if span_sizes else 0.0,
        "is_bold": bold_hits > 0,
        "line_count": len(raw_lines),
        "width": float(x1) - float(x0),
        "height": float(y1) - float(y0),
    }


def _detect_columns(blocks: list[dict[str, Any]], page_width: float) -> dict[str, float]:
    narrow_blocks = [block for block in blocks if block["width"] <= page_width * 0.72 and block["line_count"] <= 12]
    if len(narrow_blocks) < 6:
        return {"column_count": 1, "split_x": 0.0, "column_top_y": 0.0, "column_bottom_y": 0.0}

    x_positions = sorted({round(block["bbox"][0], 1) for block in narrow_blocks})
    best_gap = 0.0
    split_x = 0.0
    for left, right in zip(x_positions, x_positions[1:], strict=False):
        gap = right - left
        midpoint = (left + right) / 2.0
        left_count = sum(1 for block in narrow_blocks if ((block["bbox"][0] + block["bbox"][2]) / 2.0) < midpoint)
        right_count = sum(1 for block in narrow_blocks if ((block["bbox"][0] + block["bbox"][2]) / 2.0) >= midpoint)
        if gap > best_gap and min(left_count, right_count) >= 2:
            best_gap = gap
            split_x = midpoint

    if best_gap < page_width * 0.12:
        return {"column_count": 1, "split_x": 0.0, "column_top_y": 0.0, "column_bottom_y": 0.0}

    left_blocks = [block for block in narrow_blocks if ((block["bbox"][0] + block["bbox"][2]) / 2.0) < split_x]
    right_blocks = [block for block in narrow_blocks if ((block["bbox"][0] + block["bbox"][2]) / 2.0) >= split_x]
    if min(len(left_blocks), len(right_blocks)) < 2:
        return {"column_count": 1, "split_x": 0.0, "column_top_y": 0.0, "column_bottom_y": 0.0}

    return {
        "column_count": 2,
        "split_x": split_x,
        "column_top_y": min(block["bbox"][1] for block in narrow_blocks),
        "column_bottom_y": max(block["bbox"][3] for block in narrow_blocks),
    }


def _column_id(block: dict[str, Any], layout: dict[str, float], page_width: float) -> int:
    if layout["column_count"] == 1:
        return 0
    if block["width"] >= page_width * 0.82:
        return -1
    center_x = (block["bbox"][0] + block["bbox"][2]) / 2.0
    return 0 if center_x < layout["split_x"] else 1


def _reading_order_key(block: dict[str, Any], layout: dict[str, float], page_width: float) -> tuple[float, float, float]:
    x0, y0, _, y1 = block["bbox"]
    if layout["column_count"] == 1:
        return (0.0, round(y0, 1), round(x0, 1))

    column_id = _column_id(block, layout, page_width)
    if column_id == -1 and y1 <= layout["column_top_y"] + 8:
        return (0.0, round(y0, 1), round(x0, 1))
    if column_id == 0:
        return (1.0, round(y0, 1), round(x0, 1))
    if column_id == 1:
        return (2.0, round(y0, 1), round(x0, 1))
    if y0 >= layout["column_bottom_y"] - 8:
        return (3.0, round(y0, 1), round(x0, 1))
    return (2.5, round(y0, 1), round(x0, 1))


def _extract_sorted_blocks(page: fitz.Page) -> list[dict[str, Any]]:
    raw = page.get_text("dict")
    page_width = float(page.rect.width)
    blocks: list[dict[str, Any]] = []
    for block in raw.get("blocks", []):
        block_record = _extract_block_record(block)
        if not block_record:
            continue
        blocks.append(block_record)

    layout = _detect_columns(blocks, page_width)
    blocks.sort(key=lambda item: _reading_order_key(item, layout, page_width))

    font_sizes = [record["avg_font_size"] for record in blocks if record["avg_font_size"]]
    baseline_font_size = median(font_sizes) if font_sizes else 0.0
    for index, record in enumerate(blocks, start=1):
        record["block_id"] = index
        record["page_block_index"] = index
        record["page_font_baseline"] = baseline_font_size
        record["column_id"] = _column_id(record, layout, page_width)
        record["column_count"] = int(layout["column_count"])
        record["page_width"] = page_width
    return blocks


def _classify_block(block: dict[str, Any], baseline_font_size: float) -> str:
    text = block["text"]
    if not text:
        return "noise"
    if TABLE_CAPTION_RE.match(text) or is_table_like(text):
        return "table"
    if FIGURE_CAPTION_RE.match(text):
        return "figure_caption"
    if is_title_like(
        text,
        font_size=float(block.get("max_font_size", 0.0)),
        baseline_font_size=baseline_font_size,
        is_bold=bool(block.get("is_bold")),
    ):
        return "title"
    if is_probable_noise_line(text):
        return "noise"
    return "paragraph"


def _infer_title_level(text: str, font_size: float, baseline_font_size: float, is_bold: bool) -> int:
    stripped = normalize_text(text)
    if NUMBERED_H3_RE.match(stripped):
        return 3
    if NUMBERED_H2_RE.match(stripped):
        return 2
    if NUMBERED_H1_RE.match(stripped):
        return 1
    if baseline_font_size and font_size >= baseline_font_size * 1.32:
        return 1
    if baseline_font_size and font_size >= baseline_font_size * 1.16:
        return 2
    if is_bold:
        return 2
    return 3


def _mergeable(prev: dict[str, Any], current: dict[str, Any]) -> bool:
    if prev["element_type"] != current["element_type"]:
        return False
    if prev["page_start"] != current["page_start"]:
        return False
    if prev.get("section_path") != current.get("section_path"):
        return False
    if prev.get("column_id") != current.get("column_id"):
        return False

    vertical_gap = float(current["bbox"][1]) - float(prev["bbox"][3])
    max_chars = 1600 if current["element_type"] == "paragraph" else 5000
    gap_limit = 26.0 if current["element_type"] == "paragraph" else 18.0
    return vertical_gap <= gap_limit and (len(prev["text"]) + len(current["text"])) <= max_chars


def _merge_element(prev: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    merged = dict(prev)
    merged["text"] = normalize_text(f"{prev['text']}\n{current['text']}")
    merged["raw_lines"] = list(prev.get("raw_lines", [])) + list(current.get("raw_lines", []))
    merged["page_end"] = current["page_end"]
    merged["bbox"] = [
        min(prev["bbox"][0], current["bbox"][0]),
        min(prev["bbox"][1], current["bbox"][1]),
        max(prev["bbox"][2], current["bbox"][2]),
        max(prev["bbox"][3], current["bbox"][3]),
    ]
    return merged


def _annotate_figure_context(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    figure_index = 0
    for index, element in enumerate(elements):
        if element["element_type"] != "figure_caption":
            continue
        figure_index += 1
        figure_id = f"fig-{figure_index:04d}"
        element["figure_id"] = figure_id
        attached = 0
        for follow in elements[index + 1 :]:
            if follow["element_type"] in {"title", "table", "figure_caption"}:
                break
            if follow["element_type"] != "paragraph":
                continue
            if follow["page_start"] > element["page_end"] + 1:
                break
            follow["element_type"] = "figure_context"
            follow["figure_id"] = figure_id
            attached += 1
            if attached >= 2:
                break
    table_index = 0
    for element in elements:
        if element["element_type"] == "table":
            table_index += 1
            element["table_id"] = f"tbl-{table_index:04d}"
    return elements


def _build_page_elements(blocks: list[dict[str, Any]], page_num: int) -> list[dict[str, Any]]:
    baseline_font_size = median([block["avg_font_size"] for block in blocks if block["avg_font_size"]]) if blocks else 0.0
    elements: list[dict[str, Any]] = []
    title_stack: list[str] = []

    for block in blocks:
        element_type = _classify_block(block, baseline_font_size)
        if element_type == "title":
            title_level = _infer_title_level(
                block["text"],
                float(block.get("max_font_size", 0.0)),
                baseline_font_size,
                bool(block.get("is_bold")),
            )
            while len(title_stack) >= title_level:
                title_stack.pop()
            title_stack.append(block["text"])
            section_path = " > ".join(title_stack)
            elements.append(
                {
                    "element_id": f"p{page_num:04d}-e{len(elements) + 1:03d}",
                    "page_start": page_num,
                    "page_end": page_num,
                    "element_type": "title",
                    "text": block["text"],
                    "raw_lines": list(block.get("raw_lines", [])),
                    "bbox": list(block["bbox"]),
                    "column_id": block.get("column_id", 0),
                    "section_title": block["text"],
                    "section_path": section_path,
                    "max_font_size": float(block.get("max_font_size", 0.0)),
                    "avg_font_size": float(block.get("avg_font_size", 0.0)),
                    "is_bold": bool(block.get("is_bold")),
                }
            )
            continue

        section_path = " > ".join(title_stack)
        section_title = title_stack[-1] if title_stack else None
        candidate = {
            "element_id": f"p{page_num:04d}-e{len(elements) + 1:03d}",
            "page_start": page_num,
            "page_end": page_num,
            "element_type": element_type,
            "text": block["text"],
            "raw_lines": list(block.get("raw_lines", [])),
            "bbox": list(block["bbox"]),
            "column_id": block.get("column_id", 0),
            "section_title": section_title,
            "section_path": section_path,
            "max_font_size": float(block.get("max_font_size", 0.0)),
            "avg_font_size": float(block.get("avg_font_size", 0.0)),
            "is_bold": bool(block.get("is_bold")),
        }
        if elements and _mergeable(elements[-1], candidate):
            elements[-1] = _merge_element(elements[-1], candidate)
        else:
            elements.append(candidate)

    return _annotate_figure_context(elements)


def parse_pdf(pdf_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    doc_id = make_doc_id(pdf_path)
    industry = guess_industry(pdf_path.name)
    pages: list[dict[str, Any]] = []
    badcases: list[dict[str, Any]] = []

    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document, start=1):
            blocks = _extract_sorted_blocks(page)
            page_text = normalize_text("\n\n".join(block["text"] for block in blocks if block["text"]))
            if not page_text:
                page_text = normalize_text(page.get_text("text", sort=True))
            source_type = guess_source_type(pdf_path.name, page_text[:500])
            elements = _build_page_elements(blocks, page_index)
            element_text = normalize_text(
                "\n\n".join(
                    element["text"]
                    for element in elements
                    if element["element_type"] not in {"noise"} and element.get("text")
                )
            )
            page_record = {
                "doc_id": doc_id,
                "file_name": pdf_path.name,
                "source_path": str(pdf_path.as_posix()),
                "industry": industry,
                "source_type": source_type,
                "page_num": page_index,
                "text": element_text or page_text,
                "char_count": len(element_text or page_text),
                "blocks": blocks,
                "elements": elements,
            }
            pages.append(page_record)
            if not (element_text or page_text):
                badcases.append(
                    {
                        "doc_id": doc_id,
                        "file_name": pdf_path.name,
                        "page_num": page_index,
                        "issue_type": "empty_page_text",
                        "detail": "No extractable text from page.",
                    }
                )
    return pages, badcases
