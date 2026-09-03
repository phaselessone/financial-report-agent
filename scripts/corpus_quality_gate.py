"""Offline corpus-quality gate for deciding whether M10 should be promoted.

The gate consumes the JSONL products already emitted by the ingest pipeline.
It never parses a PDF, loads a model, calls a provider, or downloads anything.
The result is a diagnostic decision, not an OCR implementation: ``promote_m10``
is true only when the measured extraction/table signals cross explicit gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAGES_PATH = Path("data/cleaned_pages/cleaned_pages.jsonl")
DEFAULT_CHUNKS_PATH = Path("data/chunks/chunks.jsonl")
DEFAULT_OUTPUT_PATH = Path("outputs/quality/corpus_quality_gate.json")

# These are promotion triggers rather than claims about a corpus-wide OCR
# quality score. They deliberately favour a false negative over a premature
# OCR/table rewrite; a reviewer can inspect the emitted samples before acting.
DEFAULT_THRESHOLDS = {
    "min_pages": 10,
    "empty_page_ratio": 0.10,
    "low_text_page_ratio": 0.20,
    "low_text_chars": 60,
    "replacement_char_ratio": 0.01,
    "table_representation_gap_ratio": 0.15,
}


class QualityGateError(RuntimeError):
    """Raised for malformed or unavailable local ingest artifacts."""


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise QualityGateError(f"cannot read JSONL artifact {path}: {exc}") from exc
    if not all(isinstance(row, dict) for row in rows):
        raise QualityGateError(f"{path} must contain JSON objects")
    return rows


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _sample(values: Iterable[Mapping[str, Any]], *, limit: int = 20) -> list[dict[str, Any]]:
    return [dict(value) for value in list(values)[:limit]]


def assess_corpus_quality(
    pages: Iterable[Mapping[str, Any]],
    chunks: Iterable[Mapping[str, Any]] | None = None,
    *,
    thresholds: Mapping[str, float | int] | None = None,
) -> dict[str, Any]:
    """Return a deterministic M10 promotion decision and inspectable signals.

    ``pages`` must be the cleaned page JSONL emitted by ``run_pipeline.py``.
    ``chunks`` is optional, but without it the table-representation signal is
    reported as unavailable rather than guessed.
    """

    policy = {**DEFAULT_THRESHOLDS, **dict(thresholds or {})}
    page_rows = [dict(row) for row in pages]
    chunk_rows = [dict(row) for row in chunks] if chunks is not None else None
    if not page_rows:
        return {
            "schema_version": 1,
            "decision": "INSUFFICIENT_DATA",
            "promote_m10": False,
            "reason_codes": ["no_pages"],
            "thresholds": policy,
            "summary": {"page_count": 0, "chunk_count": 0 if chunk_rows is not None else None},
            "samples": {"empty_or_low_text_pages": [], "table_gap_documents": []},
        }

    page_count = len(page_rows)
    empty_pages: list[dict[str, Any]] = []
    low_text_pages: list[dict[str, Any]] = []
    replacement_pages: list[dict[str, Any]] = []
    table_pages_by_doc: Counter[str] = Counter()

    for row in page_rows:
        text = str(row.get("text") or "")
        char_count = int(row.get("char_count", len(text)) or 0)
        identity = {
            "doc_id": str(row.get("doc_id") or ""),
            "file_name": str(row.get("file_name") or ""),
            "page_num": row.get("page_num"),
            "char_count": char_count,
        }
        if char_count == 0:
            empty_pages.append(identity)
        elif char_count < int(policy["low_text_chars"]):
            low_text_pages.append(identity)
        if text and (text.count("\ufffd") / max(len(text), 1)) >= float(policy["replacement_char_ratio"]):
            replacement_pages.append(identity)
        if any(str(element.get("element_type") or "") == "table" for element in row.get("elements") or []):
            table_pages_by_doc[identity["doc_id"]] += 1

    table_chunks_by_doc: Counter[str] = Counter()
    cross_page_table_chunks = 0
    if chunk_rows is not None:
        for row in chunk_rows:
            if str(row.get("element_type") or "") == "table" or str(row.get("chunk_type") or "") in {"table_like", "table_parent"}:
                doc_id = str(row.get("doc_id") or "")
                table_chunks_by_doc[doc_id] += 1
                if int(row.get("page_end", 0) or 0) > int(row.get("page_start", 0) or 0):
                    cross_page_table_chunks += 1

    table_gap_documents = [
        {"doc_id": doc_id, "table_page_count": count, "table_chunk_count": table_chunks_by_doc.get(doc_id, 0)}
        for doc_id, count in sorted(table_pages_by_doc.items())
        if count and not table_chunks_by_doc.get(doc_id)
    ]
    table_doc_count = len(table_pages_by_doc)
    table_gap_ratio = _ratio(len(table_gap_documents), table_doc_count)
    signals = {
        "empty_page_ratio": _ratio(len(empty_pages), page_count),
        "low_text_page_ratio": _ratio(len(low_text_pages), page_count),
        "replacement_char_page_ratio": _ratio(len(replacement_pages), page_count),
        "table_representation_gap_ratio": table_gap_ratio if chunk_rows is not None else None,
    }
    reasons: list[str] = []
    if page_count < int(policy["min_pages"]):
        reasons.append("sample_too_small")
    if signals["empty_page_ratio"] >= float(policy["empty_page_ratio"]):
        reasons.append("empty_page_ratio")
    if signals["low_text_page_ratio"] >= float(policy["low_text_page_ratio"]):
        reasons.append("low_text_page_ratio")
    if signals["replacement_char_page_ratio"] >= float(policy["replacement_char_ratio"]):
        reasons.append("replacement_character_ratio")
    if chunk_rows is not None and table_doc_count and table_gap_ratio >= float(policy["table_representation_gap_ratio"]):
        reasons.append("table_representation_gap")

    actionable = [reason for reason in reasons if reason != "sample_too_small"]
    if "sample_too_small" in reasons:
        decision = "INSUFFICIENT_DATA"
    else:
        decision = "PROMOTE_M10" if actionable else "KEEP_M10_P2"
    return {
        "schema_version": 1,
        "decision": decision,
        "promote_m10": decision == "PROMOTE_M10",
        "reason_codes": reasons,
        "thresholds": policy,
        "summary": {
            "page_count": page_count,
            "chunk_count": len(chunk_rows) if chunk_rows is not None else None,
            "document_count": len({str(row.get("doc_id") or "") for row in page_rows}),
            "table_document_count": table_doc_count,
            "cross_page_table_chunk_count": cross_page_table_chunks if chunk_rows is not None else None,
            **signals,
        },
        "samples": {
            "empty_or_low_text_pages": _sample([*empty_pages, *low_text_pages, *replacement_pages]),
            "table_gap_documents": _sample(table_gap_documents),
        },
    }


def run_gate(
    *,
    pages_path: Path,
    chunks_path: Path | None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    resolved_pages_path = _repo_path(pages_path).resolve()
    resolved_chunks_path = _repo_path(chunks_path).resolve() if chunks_path is not None else None
    pages = _read_jsonl(resolved_pages_path)
    chunks = _read_jsonl(resolved_chunks_path) if resolved_chunks_path is not None else None
    report = assess_corpus_quality(pages, chunks)
    report["input_artifacts"] = {
        "pages": {
            "path": str(resolved_pages_path),
            "sha256": hashlib.sha256(resolved_pages_path.read_bytes()).hexdigest(),
            "size_bytes": resolved_pages_path.stat().st_size,
            "row_count": len(pages),
        },
        "chunks": (
            {
                "path": str(resolved_chunks_path),
                "sha256": hashlib.sha256(resolved_chunks_path.read_bytes()).hexdigest(),
                "size_bytes": resolved_chunks_path.stat().st_size,
                "row_count": len(chunks or []),
            }
            if resolved_chunks_path is not None
            else None
        ),
    }
    if output_path is not None:
        destination = _repo_path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline M10 OCR/table promotion gate.")
    parser.add_argument("--pages-path", type=Path, default=DEFAULT_PAGES_PATH)
    parser.add_argument("--chunks-path", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--without-chunks", action="store_true", help="Assess extraction signals only.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = run_gate(
            pages_path=args.pages_path,
            chunks_path=None if args.without_chunks else args.chunks_path,
            output_path=args.output,
        )
    except QualityGateError as exc:
        print(f"CORPUS_QUALITY_GATE_FAILED: {exc}")
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
