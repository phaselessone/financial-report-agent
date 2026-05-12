from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.ingest.chunker import (
    PRODUCTION_STRATEGY,
    aggregate_strategy_stats,
    build_chunk_strategies,
    build_parent_context_chunks,
    summarize_chunk_strategies,
)
from src.ingest.cleaner import clean_pages
from src.ingest.parser import parse_pdf
from src.utils.io import JsonlWriter, ensure_dir, write_json


def run_ingest_pipeline(
    *,
    input_dir: Path,
    output_dir: Path,
    badcase_dir: Path,
    limit: int = 0,
) -> dict[str, Any]:
    pdf_paths = sorted(input_dir.rglob("*.pdf"))
    if limit > 0:
        pdf_paths = pdf_paths[:limit]
    if not pdf_paths:
        raise SystemExit(f"No PDF files found under {input_dir}")

    parsed_path = output_dir / "parsed_pages" / "parsed_pages.jsonl"
    cleaned_path = output_dir / "cleaned_pages" / "cleaned_pages.jsonl"
    elements_path = output_dir / "elements" / "elements.jsonl"
    parent_chunks_path = output_dir / "chunks" / "parent_chunks.jsonl"
    child_chunks_path = output_dir / "chunks" / "child_chunks.jsonl"
    chunks_path = output_dir / "chunks" / "chunks.jsonl"
    experiment_dir = output_dir / "chunks" / "experiments"
    badcase_path = badcase_dir / "ingest_badcases.jsonl"
    summary_path = output_dir / "reports" / "ingest_summary.json"
    chunk_report_path = output_dir / "reports" / "chunk_strategy_report.json"

    ensure_dir(summary_path.parent)
    ensure_dir(experiment_dir)

    totals = {
        "pdf_count": len(pdf_paths),
        "page_count": 0,
        "element_count": 0,
        "parent_chunk_count": 0,
        "chunk_count": 0,
        "badcase_count": 0,
        "production_strategy": PRODUCTION_STRATEGY,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
    per_doc_stats: list[dict[str, dict[str, Any]]] = []

    with (
        JsonlWriter(parsed_path) as parsed_writer,
        JsonlWriter(cleaned_path) as cleaned_writer,
        JsonlWriter(elements_path) as element_writer,
        JsonlWriter(parent_chunks_path) as parent_chunk_writer,
        JsonlWriter(child_chunks_path) as child_chunk_writer,
        JsonlWriter(chunks_path) as chunk_writer,
        JsonlWriter(experiment_dir / "fixed_window.jsonl") as fixed_writer,
        JsonlWriter(experiment_dir / "title_aware.jsonl") as title_writer,
        JsonlWriter(experiment_dir / "table_protected.jsonl") as table_writer,
        JsonlWriter(badcase_path) as badcase_writer,
    ):
        for index, pdf_path in enumerate(pdf_paths, start=1):
            print(f"[{index}/{len(pdf_paths)}] Processing {pdf_path.name}")
            parsed_pages, parse_badcases = parse_pdf(pdf_path)
            cleaned_pages, clean_notes = clean_pages(parsed_pages)
            strategy_rows = build_chunk_strategies(cleaned_pages)
            parent_rows = build_parent_context_chunks(cleaned_pages)
            per_doc_stats.append(summarize_chunk_strategies(strategy_rows))

            for row in parsed_pages:
                parsed_writer.write_row(row)
            for row in cleaned_pages:
                cleaned_writer.write_row(row)
                for element in row.get("elements", []):
                    payload = dict(element)
                    payload.update(
                        {
                            "doc_id": row["doc_id"],
                            "file_name": row["file_name"],
                            "source_path": row["source_path"],
                            "industry": row["industry"],
                            "source_type": row["source_type"],
                            "page_num": row["page_num"],
                        }
                    )
                    element_writer.write_row(payload)

            for row in parent_rows:
                parent_chunk_writer.write_row(row)

            for row in strategy_rows["fixed_window"]:
                fixed_writer.write_row(row)
            for row in strategy_rows["title_aware"]:
                title_writer.write_row(row)
            for row in strategy_rows["table_protected"]:
                table_writer.write_row(row)
                child_chunk_writer.write_row(row)
                chunk_writer.write_row(row)

            for row in parse_badcases + clean_notes:
                badcase_writer.write_row(row)

            totals["page_count"] += len(parsed_pages)
            totals["element_count"] += sum(len(row.get("elements", [])) for row in cleaned_pages)
            totals["parent_chunk_count"] += len(parent_rows)
            totals["chunk_count"] += len(strategy_rows[PRODUCTION_STRATEGY])
            totals["badcase_count"] += len(parse_badcases) + len(clean_notes)

    strategy_report = {
        "production_strategy": PRODUCTION_STRATEGY,
        "aggregated": aggregate_strategy_stats(per_doc_stats),
        "doc_count": len(per_doc_stats),
    }
    write_json(summary_path, totals)
    write_json(chunk_report_path, strategy_report)
    return {
        "totals": totals,
        "strategy_report": strategy_report,
        "paths": {
            "parsed_path": str(parsed_path),
            "cleaned_path": str(cleaned_path),
            "elements_path": str(elements_path),
            "parent_chunks_path": str(parent_chunks_path),
            "child_chunks_path": str(child_chunks_path),
            "chunks_path": str(chunks_path),
            "badcase_path": str(badcase_path),
            "summary_path": str(summary_path),
            "chunk_report_path": str(chunk_report_path),
        },
    }
