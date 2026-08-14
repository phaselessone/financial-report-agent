from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any

from src.utils.text_utils import (
    first_sentence,
    is_table_like,
    is_title_like,
    normalize_text,
    split_paragraphs,
    split_with_overlap,
)

STRATEGIES = ("fixed_window", "title_aware", "table_protected")
PRODUCTION_STRATEGY = "table_protected"

PARENT_TARGET_SIZE = 1050
PARENT_OVERLAP = 120
CHILD_TARGET_SIZE = 360
CHILD_OVERLAP = 40
MIN_CHILD_SIZE = 120
TABLE_ROW_GROUP_SIZE = 10


def _make_chunk_id(doc_id: str, strategy_name: str, chunk_index: int, *, granularity: str) -> str:
    if granularity == "parent":
        return f"{doc_id}-{strategy_name}-p{chunk_index:04d}"
    return f"{doc_id}-{strategy_name}-c{chunk_index:04d}"


def _base_chunk(
    *,
    doc_id: str,
    file_name: str,
    industry: str,
    source_type: str,
    page_start: int,
    page_end: int,
    section_title: str | None,
    section_path: str | None,
    chunk_type: str,
    strategy_name: str,
    source_path: str,
    text: str,
    chunk_index: int,
    granularity: str = "child",
    parent_chunk_id: str | None = None,
    element_type: str = "paragraph",
    table_id: str | None = None,
    figure_id: str | None = None,
    parent_text: str | None = None,
    bundle_id: str | None = None,
    bundle_rank: int | None = None,
    support_type: str | None = None,
    support_span: str | None = None,
) -> dict[str, Any]:
    normalized_text = normalize_text(text)
    chunk_id = _make_chunk_id(doc_id, strategy_name, chunk_index, granularity=granularity)
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "file_name": file_name,
        "industry": industry,
        "source_type": source_type,
        "source_path": source_path,
        "page_start": page_start,
        "page_end": page_end,
        "source_page_range": [page_start, page_end],
        "section_title": section_title,
        "section_path": section_path,
        "chunk_type": chunk_type,
        "strategy_name": strategy_name,
        "granularity": granularity,
        "parent_chunk_id": parent_chunk_id,
        "element_type": element_type,
        "table_id": table_id,
        "figure_id": figure_id,
        "bundle_id": bundle_id,
        "bundle_rank": bundle_rank,
        "support_type": support_type,
        "support_span": normalize_text(support_span) if support_span else None,
        "text": normalized_text,
        "parent_text": normalize_text(parent_text) if parent_text else None,
        "char_count": len(normalized_text),
    }


def _page_metadata(page_records: list[dict[str, Any]]) -> dict[str, Any]:
    if not page_records:
        return {}
    sample = page_records[0]
    return {
        "doc_id": sample["doc_id"],
        "file_name": sample["file_name"],
        "industry": sample["industry"],
        "source_type": sample["source_type"],
        "source_path": sample["source_path"],
    }


def _flatten_blocks(page_records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float]:
    blocks: list[dict[str, Any]] = []
    font_sizes: list[float] = []
    for record in page_records:
        for block in record.get("blocks", []):
            block_record = dict(block)
            block_record["page_num"] = record["page_num"]
            blocks.append(block_record)
            if block_record.get("avg_font_size"):
                font_sizes.append(float(block_record["avg_font_size"]))
    baseline_font_size = mean(font_sizes) if font_sizes else 0.0
    return blocks, baseline_font_size


def _flatten_elements(page_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    elements: list[dict[str, Any]] = []
    for record in page_records:
        for element in record.get("elements", []):
            payload = dict(element)
            payload.setdefault("page_start", record["page_num"])
            payload.setdefault("page_end", record["page_num"])
            elements.append(payload)
    return elements


def _split_paragraph_group(paragraphs: list[dict[str, Any]], target_size: int, overlap: int) -> list[dict[str, Any]]:
    if not paragraphs:
        return []

    joined = [paragraph for paragraph in paragraphs if paragraph.get("text", "").strip()]
    if not joined:
        return []

    # 先把超长段落按 target_size(留 overlap)切分为受控小片,避免整段原样保留。
    units: list[dict[str, Any]] = []
    for paragraph in joined:
        text = normalize_text(paragraph["text"])
        if not text:
            continue
        pieces = split_with_overlap(text, target_size=target_size, overlap=overlap) if len(text) > target_size else [text]
        for piece in pieces:
            units.append(
                {
                    "text": piece,
                    "page_num": paragraph["page_num"],
                    "section_title": paragraph.get("section_title"),
                    "section_path": paragraph.get("section_path"),
                    "chunk_type": paragraph.get("chunk_type", "text"),
                    "element_type": paragraph.get("element_type", "paragraph"),
                }
            )
    if not units:
        return []

    segments: list[dict[str, Any]] = []
    current_text = ""
    current_pages: list[int] = []
    current_section = units[0]["section_title"]
    current_section_path = units[0]["section_path"]
    current_chunk_type = units[0]["chunk_type"]
    current_element_type = units[0]["element_type"]

    for unit in units:
        unit_section = unit["section_title"]
        unit_section_path = unit["section_path"]
        unit_chunk_type = unit["chunk_type"]
        unit_element_type = unit["element_type"]

        if current_text and (
            unit_section != current_section
            or unit_section_path != current_section_path
            or unit_chunk_type != current_chunk_type
            or unit_element_type != current_element_type
        ):
            segments.append(
                {
                    "text": current_text,
                    "page_start": min(current_pages),
                    "page_end": max(current_pages),
                    "section_title": current_section,
                    "section_path": current_section_path,
                    "chunk_type": current_chunk_type,
                    "element_type": current_element_type,
                }
            )
            current_text = ""
            current_pages = []
            current_section = unit_section
            current_section_path = unit_section_path
            current_chunk_type = unit_chunk_type
            current_element_type = unit_element_type

        unit_text = unit["text"]
        candidate = f"{current_text}\n\n{unit_text}".strip() if current_text else unit_text
        if current_text and len(candidate) > target_size:
            segments.append(
                {
                    "text": current_text,
                    "page_start": min(current_pages),
                    "page_end": max(current_pages),
                    "section_title": current_section,
                    "section_path": current_section_path,
                    "chunk_type": current_chunk_type,
                    "element_type": current_element_type,
                }
            )
            overlap_text = current_text[-overlap:].strip() if overlap > 0 else ""
            current_text = f"{overlap_text}\n\n{unit_text}".strip() if overlap_text else unit_text
            current_pages = [current_pages[-1], unit["page_num"]]
        else:
            current_text = candidate
            current_pages.append(unit["page_num"])
            current_section = unit_section
            current_section_path = unit_section_path
            current_chunk_type = unit_chunk_type
            current_element_type = unit_element_type

    if current_text:
        segments.append(
            {
                "text": current_text,
                "page_start": min(current_pages),
                "page_end": max(current_pages),
                "section_title": current_section,
                "section_path": current_section_path,
                "chunk_type": current_chunk_type,
                "element_type": current_element_type,
            }
        )
    return segments


def _prefixed_text(section_path: str | None, text: str, *, label: str | None = None) -> str:
    lines: list[str] = []
    if section_path:
        lines.append(f"章节路径: {section_path}")
    if label:
        lines.append(label)
    lines.append(normalize_text(text))
    return normalize_text("\n".join(line for line in lines if line))


def _support_span(text: str, *, max_chars: int = 180) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return ""
    lines = [line.strip() for line in normalized.split("\n") if line.strip()]
    if is_table_like(normalized):
        return " | ".join(lines[:3])[:max_chars].rstrip()
    return first_sentence(normalized, max_chars=max_chars)


def _merge_small_child_chunks(chunks: list[str], *, target_size: int, min_size: int) -> list[str]:
    merged: list[str] = []
    for chunk in chunks:
        normalized = normalize_text(chunk)
        if not normalized:
            continue
        if merged and (
            len(normalized) < min_size
            or len(merged[-1]) < max(min_size, target_size // 2)
            or len(merged[-1]) + len(normalized) <= int(target_size * 1.15)
        ):
            merged[-1] = normalize_text(f"{merged[-1]}\n\n{normalized}")
        else:
            merged.append(normalized)
    if len(merged) >= 2 and len(merged[-1]) < min_size:
        merged[-2] = normalize_text(f"{merged[-2]}\n\n{merged[-1]}")
        merged.pop()
    return merged


def _split_child_text(text: str, *, target_size: int = CHILD_TARGET_SIZE, overlap: int = CHILD_OVERLAP) -> list[str]:
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        normalized = normalize_text(text)
        return [normalized] if normalized else []

    # 先把超长段落按 target_size(留 overlap)切分为受控小片,避免整段原样保留。
    pieces: list[str] = []
    for paragraph in paragraphs:
        paragraph = normalize_text(paragraph)
        if not paragraph:
            continue
        if len(paragraph) > target_size:
            pieces.extend(split_with_overlap(paragraph, target_size=target_size, overlap=overlap))
        else:
            pieces.append(paragraph)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n\n{piece}".strip() if current else piece
        if current and len(candidate) > target_size:
            chunks.append(normalize_text(current))
            overlap_text = current[-overlap:].strip() if overlap > 0 else ""
            current = f"{overlap_text}\n\n{piece}".strip() if overlap_text else piece
        else:
            current = candidate
    if current:
        chunks.append(normalize_text(current))
    return _merge_small_child_chunks(chunks, target_size=target_size, min_size=MIN_CHILD_SIZE)


def _normalize_table_lines(text: str) -> list[str]:
    return [normalize_text(line) for line in text.split("\n") if normalize_text(line)]


def _table_summary_text(text: str) -> str:
    lines = _normalize_table_lines(text)
    if not lines:
        return ""
    header_lines = lines[: min(2, len(lines))]
    data_lines = lines[len(header_lines) : len(header_lines) + 6]
    summary_lines: list[str] = []
    if header_lines:
        summary_lines.append(f"表头: {' | '.join(header_lines)}")
    for line in data_lines:
        summary_lines.append(f"数据行: {line}")
    if not data_lines and len(lines) > len(header_lines):
        summary_lines.append(f"正文: {' '.join(lines[len(header_lines):])}")
    return normalize_text("\n".join(summary_lines))


def _table_row_groups(text: str, *, group_size: int = TABLE_ROW_GROUP_SIZE) -> list[str]:
    lines = _normalize_table_lines(text)
    if len(lines) <= 2:
        return []
    header_lines = lines[:2]
    data_lines = lines[2:]
    groups: list[str] = []
    for start in range(0, len(data_lines), group_size):
        group = data_lines[start : start + group_size]
        if groups and len(group) < max(4, group_size // 2):
            groups[-1] = normalize_text(f"{groups[-1]}\n" + "\n".join(group))
            continue
        groups.append(normalize_text("\n".join(header_lines + group)))
    return groups


def _continued_table(prev_element: dict[str, Any], element: dict[str, Any]) -> bool:
    if prev_element.get("element_type") != "table" or element.get("element_type") != "table":
        return False
    if prev_element.get("section_path") != element.get("section_path"):
        return False
    if int(element["page_start"]) != int(prev_element["page_end"]) + 1:
        return False

    prev_lines = _normalize_table_lines(prev_element.get("text", ""))
    curr_lines = _normalize_table_lines(element.get("text", ""))
    if not prev_lines or not curr_lines:
        return False
    prev_header = normalize_text(" ".join(prev_lines[:2]))
    curr_header = normalize_text(" ".join(curr_lines[:2]))
    if curr_lines[0].startswith("续表"):
        return True
    return bool(prev_header and curr_header and prev_header[:40] == curr_header[:40])


def _build_structured_artifacts(page_records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    meta = _page_metadata(page_records)
    elements = _flatten_elements(page_records)
    parent_rows: list[dict[str, Any]] = []
    child_rows: list[dict[str, Any]] = []
    parent_index = 0
    child_index = 0
    text_buffer: list[dict[str, Any]] = []

    def flush_text_buffer() -> None:
        nonlocal parent_index, child_index
        if not text_buffer:
            return

        paragraph_rows = [
            {
                "text": element["text"],
                "page_num": int(element["page_start"]),
                "section_title": element.get("section_title"),
                "section_path": element.get("section_path"),
                "chunk_type": "text",
                "element_type": element.get("element_type", "paragraph"),
            }
            for element in text_buffer
        ]
        parent_segments = _split_paragraph_group(paragraph_rows, target_size=PARENT_TARGET_SIZE, overlap=PARENT_OVERLAP)
        text_buffer.clear()

        for segment in parent_segments:
            parent_index += 1
            parent_text = _prefixed_text(segment.get("section_path"), segment["text"])
            parent_row = _base_chunk(
                **meta,
                page_start=segment["page_start"],
                page_end=segment["page_end"],
                section_title=segment.get("section_title"),
                section_path=segment.get("section_path"),
                chunk_type="parent_context",
                strategy_name="table_protected",
                text=parent_text,
                chunk_index=parent_index,
                granularity="parent",
                element_type="paragraph",
            )
            parent_rows.append(parent_row)

            for bundle_rank, child_text in enumerate(_split_child_text(segment["text"]), start=1):
                child_index += 1
                child_rows.append(
                    _base_chunk(
                        **meta,
                        page_start=segment["page_start"],
                        page_end=segment["page_end"],
                        section_title=segment.get("section_title"),
                        section_path=segment.get("section_path"),
                        chunk_type="text",
                        strategy_name="table_protected",
                        text=_prefixed_text(segment.get("section_path"), child_text),
                        chunk_index=child_index,
                        granularity="child",
                        parent_chunk_id=parent_row["chunk_id"],
                        element_type="paragraph",
                        parent_text=parent_row["text"],
                        bundle_id=parent_row["chunk_id"],
                        bundle_rank=bundle_rank,
                        support_type="text_block",
                        support_span=_support_span(child_text),
                    )
                )

    index = 0
    while index < len(elements):
        element = elements[index]
        element_type = element.get("element_type")

        if element_type == "title":
            flush_text_buffer()
            index += 1
            continue

        if element_type in {"paragraph", "figure_context"}:
            text_buffer.append(element)
            index += 1
            continue

        flush_text_buffer()

        if element_type == "figure_caption":
            bundle = [element]
            page_end = int(element["page_end"])
            section_title = element.get("section_title")
            section_path = element.get("section_path")
            figure_id = element.get("figure_id")
            scan = index + 1
            while scan < len(elements):
                follow = elements[scan]
                if follow.get("figure_id") != figure_id or follow.get("element_type") != "figure_context":
                    break
                bundle.append(follow)
                page_end = max(page_end, int(follow["page_end"]))
                scan += 1

            parent_index += 1
            bundle_text = normalize_text("\n\n".join(item["text"] for item in bundle))
            parent_text = _prefixed_text(section_path, bundle_text, label="图表语义")
            parent_row = _base_chunk(
                **meta,
                page_start=int(element["page_start"]),
                page_end=page_end,
                section_title=section_title,
                section_path=section_path,
                chunk_type="figure_parent",
                strategy_name="table_protected",
                text=parent_text,
                chunk_index=parent_index,
                granularity="parent",
                element_type="figure",
                figure_id=figure_id,
            )
            parent_rows.append(parent_row)

            child_index += 1
            child_rows.append(
                _base_chunk(
                    **meta,
                    page_start=int(element["page_start"]),
                    page_end=page_end,
                    section_title=section_title,
                    section_path=section_path,
                    chunk_type="figure_bundle",
                    strategy_name="table_protected",
                    text=_prefixed_text(section_path, bundle_text, label="图表语义"),
                    chunk_index=child_index,
                    granularity="child",
                    parent_chunk_id=parent_row["chunk_id"],
                    element_type="figure",
                    figure_id=figure_id,
                    parent_text=parent_row["text"],
                    bundle_id=parent_row["chunk_id"],
                    bundle_rank=1,
                    support_type="figure_bundle",
                    support_span=_support_span(bundle_text),
                )
            )
            index = scan
            continue

        if element_type == "table":
            table_elements = [element]
            page_end = int(element["page_end"])
            scan = index + 1
            while scan < len(elements):
                follow = elements[scan]
                if _continued_table(table_elements[-1], follow):
                    table_elements.append(follow)
                    page_end = max(page_end, int(follow["page_end"]))
                    scan += 1
                    continue
                break

            section_title = element.get("section_title")
            section_path = element.get("section_path")
            table_id = table_elements[0].get("table_id")
            bundle_text = normalize_text("\n\n".join(item["text"] for item in table_elements))
            summary_text = _table_summary_text(bundle_text)
            parent_index += 1
            parent_body = normalize_text("\n\n".join(part for part in [summary_text, bundle_text] if part))
            parent_text = _prefixed_text(section_path, parent_body, label="表格语义")
            parent_row = _base_chunk(
                **meta,
                page_start=int(element["page_start"]),
                page_end=page_end,
                section_title=section_title,
                section_path=section_path,
                chunk_type="table_parent",
                strategy_name="table_protected",
                text=parent_text,
                chunk_index=parent_index,
                granularity="parent",
                element_type="table",
                table_id=table_id,
            )
            parent_rows.append(parent_row)

            child_index += 1
            child_rows.append(
                _base_chunk(
                    **meta,
                    page_start=int(element["page_start"]),
                    page_end=page_end,
                    section_title=section_title,
                    section_path=section_path,
                    chunk_type="table_like",
                    strategy_name="table_protected",
                    text=_prefixed_text(section_path, summary_text or bundle_text, label="表格摘要"),
                    chunk_index=child_index,
                    granularity="child",
                    parent_chunk_id=parent_row["chunk_id"],
                    element_type="table",
                    table_id=table_id,
                    parent_text=parent_row["text"],
                    bundle_id=parent_row["chunk_id"],
                    bundle_rank=1,
                    support_type="table_summary",
                    support_span=_support_span(summary_text or bundle_text),
                )
            )

            for bundle_rank, group_text in enumerate(_table_row_groups(bundle_text), start=2):
                child_index += 1
                child_rows.append(
                    _base_chunk(
                        **meta,
                        page_start=int(element["page_start"]),
                        page_end=page_end,
                        section_title=section_title,
                        section_path=section_path,
                        chunk_type="table_like",
                        strategy_name="table_protected",
                        text=_prefixed_text(section_path, group_text, label="表格行组"),
                        chunk_index=child_index,
                        granularity="child",
                        parent_chunk_id=parent_row["chunk_id"],
                        element_type="table",
                        table_id=table_id,
                        parent_text=parent_row["text"],
                        bundle_id=parent_row["chunk_id"],
                        bundle_rank=bundle_rank,
                        support_type="table_rows",
                        support_span=_support_span(group_text),
                    )
                )
            index = scan
            continue

        index += 1

    flush_text_buffer()
    return parent_rows, child_rows


def build_parent_context_chunks(page_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parent_rows, _ = _build_structured_artifacts(page_records)
    return parent_rows


def build_fixed_window_chunks(page_records: list[dict[str, Any]], *, target_size: int = 700, overlap: int = 100) -> list[dict[str, Any]]:
    meta = _page_metadata(page_records)
    paragraphs: list[dict[str, Any]] = []
    for record in page_records:
        paragraphs.extend(
            {
                "text": text_part,
                "page_num": record["page_num"],
                "section_title": None,
                "section_path": None,
                "chunk_type": "text",
                "element_type": "paragraph",
            }
            for paragraph in split_paragraphs(record["text"])
            for text_part in split_with_overlap(paragraph, target_size=target_size, overlap=overlap)
        )
    segments = _split_paragraph_group(paragraphs, target_size=target_size, overlap=overlap)
    return [
        _base_chunk(
            **meta,
            page_start=segment["page_start"],
            page_end=segment["page_end"],
            section_title=None,
            section_path=None,
            chunk_type="text",
            strategy_name="fixed_window",
            text=segment["text"],
            chunk_index=index,
            granularity="child",
            element_type="paragraph",
            support_type="text_block",
            support_span=_support_span(segment["text"]),
        )
        for index, segment in enumerate(segments, start=1)
    ]


def build_title_aware_chunks(page_records: list[dict[str, Any]], *, target_size: int = 900, overlap: int = 120) -> list[dict[str, Any]]:
    meta = _page_metadata(page_records)
    blocks, baseline_font_size = _flatten_blocks(page_records)
    paragraphs: list[dict[str, Any]] = []
    current_title: str | None = None

    for block in blocks:
        text = block["text"]
        title_like = is_title_like(
            text,
            font_size=float(block.get("max_font_size", 0.0)),
            baseline_font_size=baseline_font_size,
            is_bold=bool(block.get("is_bold")),
        )
        if title_like:
            current_title = text
            continue
        paragraphs.append(
            {
                "text": text,
                "page_num": block["page_num"],
                "section_title": current_title,
                "section_path": current_title,
                "chunk_type": "text",
                "element_type": "paragraph",
            }
        )

    segments = _split_paragraph_group(paragraphs, target_size=target_size, overlap=overlap)
    return [
        _base_chunk(
            **meta,
            page_start=segment["page_start"],
            page_end=segment["page_end"],
            section_title=segment["section_title"],
            section_path=segment["section_path"],
            chunk_type=segment["chunk_type"],
            strategy_name="title_aware",
            text=segment["text"],
            chunk_index=index,
            granularity="child",
            element_type=segment.get("element_type", "paragraph"),
            support_type="text_block",
            support_span=_support_span(segment["text"]),
        )
        for index, segment in enumerate(segments, start=1)
    ]


def build_table_protected_chunks(page_records: list[dict[str, Any]], *, target_size: int = 900, overlap: int = 120) -> list[dict[str, Any]]:
    del target_size, overlap
    _, child_rows = _build_structured_artifacts(page_records)
    return child_rows


def build_chunk_strategies(page_records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        "fixed_window": build_fixed_window_chunks(page_records),
        "title_aware": build_title_aware_chunks(page_records),
        "table_protected": build_table_protected_chunks(page_records),
    }


def summarize_chunk_strategies(strategy_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for strategy_name, rows in strategy_rows.items():
        if not rows:
            summary[strategy_name] = {"chunk_count": 0, "avg_char_count": 0, "table_like_ratio": 0.0}
            continue
        table_like_count = sum(row["chunk_type"] == "table_like" for row in rows)
        summary[strategy_name] = {
            "chunk_count": len(rows),
            "avg_char_count": round(sum(row["char_count"] for row in rows) / len(rows), 2),
            "table_like_ratio": round(table_like_count / len(rows), 4),
        }
    return summary


def aggregate_strategy_stats(per_doc_stats: list[dict[str, Any]]) -> dict[str, Any]:
    combined: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    doc_count = max(len(per_doc_stats), 1)
    for doc_stats in per_doc_stats:
        for strategy_name, stats in doc_stats.items():
            for key, value in stats.items():
                combined[strategy_name][key] += value
    return {
        strategy_name: {
            "avg_chunk_count": round(stats["chunk_count"] / doc_count, 2),
            "avg_char_count": round(stats["avg_char_count"] / doc_count, 2),
            "avg_table_like_ratio": round(stats["table_like_ratio"] / doc_count, 4),
        }
        for strategy_name, stats in combined.items()
    }
