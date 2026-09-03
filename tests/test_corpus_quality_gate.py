from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path

from scripts.corpus_quality_gate import assess_corpus_quality, run_gate
from scripts.hard_case_subset_gate import (
    EXPECTED_CATEGORIES,
    load_balanced_subset,
    validate_contract_manifest,
)


def _page(doc_id: str, page_num: int, text: str, *, table: bool = False) -> dict[str, object]:
    return {
        "doc_id": doc_id,
        "file_name": f"{doc_id}.pdf",
        "page_num": page_num,
        "text": text,
        "char_count": len(text),
        "elements": [{"element_type": "table", "text": "table"}] if table else [],
    }


def test_small_or_clean_corpus_does_not_promote_m10() -> None:
    pages = [_page("doc", index, "正常文本" * 40) for index in range(1, 11)]
    chunks = [{"doc_id": "doc", "element_type": "paragraph", "chunk_type": "text", "page_start": 1, "page_end": 1}]
    report = assess_corpus_quality(pages, chunks)
    assert report["decision"] == "KEEP_M10_P2"
    assert report["promote_m10"] is False


def test_too_small_corpus_is_insufficient_even_when_chunks_are_present() -> None:
    pages = [_page("doc", index, "正常文本" * 40) for index in range(1, 10)]
    chunks = [
        {
            "doc_id": "doc",
            "element_type": "paragraph",
            "chunk_type": "text",
            "page_start": 1,
            "page_end": 1,
        }
    ]

    report = assess_corpus_quality(pages, chunks)

    assert report["decision"] == "INSUFFICIENT_DATA"
    assert report["promote_m10"] is False
    assert report["reason_codes"] == ["sample_too_small"]


def test_empty_pages_or_table_gap_promote_m10() -> None:
    pages = [_page("doc", index, "") for index in range(1, 3)] + [
        _page("doc", index, "正常文本" * 40, table=index in {3, 4}) for index in range(3, 11)
    ]
    report = assess_corpus_quality(pages, [])
    assert report["decision"] == "PROMOTE_M10"
    assert "empty_page_ratio" in report["reason_codes"]
    assert "table_representation_gap" in report["reason_codes"]


def test_gate_writes_json_without_private_assets(tmp_path: Path) -> None:
    pages_path = tmp_path / "pages.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    output_path = tmp_path / "report.json"
    pages_path.write_text("\n".join(json.dumps(_page("doc", index, "文本" * 50)) for index in range(1, 11)), encoding="utf-8")
    chunks_path.write_text(json.dumps({"doc_id": "doc", "element_type": "paragraph", "chunk_type": "text", "page_start": 1, "page_end": 1}) + "\n", encoding="utf-8")
    result = run_gate(pages_path=pages_path, chunks_path=chunks_path, output_path=output_path)
    assert result["decision"] == "KEEP_M10_P2"
    assert json.loads(output_path.read_text(encoding="utf-8"))["schema_version"] == 1
    assert result["input_artifacts"] == {
        "pages": {
            "path": str(pages_path.resolve()),
            "sha256": hashlib.sha256(pages_path.read_bytes()).hexdigest(),
            "size_bytes": pages_path.stat().st_size,
            "row_count": 10,
        },
        "chunks": {
            "path": str(chunks_path.resolve()),
            "sha256": hashlib.sha256(chunks_path.read_bytes()).hexdigest(),
            "size_bytes": chunks_path.stat().st_size,
            "row_count": 1,
        },
    }


def test_balanced_subset_includes_each_category() -> None:
    subset = load_balanced_subset(Path("benchmarks/hard_cases/cases.jsonl"))
    assert [row["category"] for row in subset] == list(EXPECTED_CATEGORIES)


def test_subset_cli_is_offline_and_deterministic() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "hard_case_subset_gate.py"
    completed = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, check=False)
    assert completed.returncode == 0
    report = json.loads(completed.stdout)
    assert report["case_count"] == 10
    assert set(report["category_counts"]) == set(EXPECTED_CATEGORIES)
    assert report["contract"]["cases_file_sha256"] == (
        "32576e9f9acf39b0905841f221ff22ac2bf1a817fe63468aa8f1a801e72fe29f"
    )


def test_hash_pinned_hard_case_bytes_are_lf_and_match_manifest() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cases_path = repo_root / "benchmarks/hard_cases/cases.jsonl"
    manifest_path = repo_root / "benchmarks/hard_cases/contract-v1.manifest.json"
    payload = cases_path.read_bytes()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert b"\r" not in payload
    assert hashlib.sha256(payload).hexdigest() == manifest["cases_file_sha256"]
    assert validate_contract_manifest(cases_path, manifest_path)["cases_file_sha256"] == (
        manifest["cases_file_sha256"]
    )
    attributes = (repo_root / ".gitattributes").read_text(encoding="utf-8")
    assert "benchmarks/**/*.jsonl text eol=lf" in attributes


def test_subset_manifest_gate_rejects_content_drift(tmp_path: Path) -> None:
    source_cases = Path(__file__).resolve().parents[1] / "benchmarks/hard_cases/cases.jsonl"
    source_manifest = (
        Path(__file__).resolve().parents[1]
        / "benchmarks/hard_cases/contract-v1.manifest.json"
    )
    cases_path = tmp_path / "cases.jsonl"
    manifest_path = tmp_path / "manifest.json"
    cases_path.write_bytes(source_cases.read_bytes() + b"\n")
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    manifest["cases_file"] = cases_path.name
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    try:
        validate_contract_manifest(cases_path, manifest_path)
    except ValueError as exc:
        assert "cases_file_sha256" in str(exc)
    else:  # pragma: no cover - proves the pin fails closed
        raise AssertionError("tampered hard-case contract unexpectedly passed")
