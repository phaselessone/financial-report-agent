from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from src.utils.io import write_json

SOURCE_MANIFEST_INCLUDE_DIRS = ("src", "tests", "scripts", "bootstrap")
SOURCE_MANIFEST_INCLUDE_GLOBS = ("run_*.py", "Makefile", "requirements*.txt", ".env.example")


def _sha1_hexdigest(payload: bytes) -> str:
    return hashlib.sha1(payload).hexdigest()


def file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows_sha1(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha1_hexdigest(payload)


def detect_git_sha(cwd: Path | None = None) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd or Path.cwd()),
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except Exception:
        return ""
    return completed.stdout.strip()


def _iter_source_files(cwd: Path) -> list[Path]:
    files: dict[str, Path] = {}
    for dirname in SOURCE_MANIFEST_INCLUDE_DIRS:
        root = cwd / dirname
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file():
                rel_path = path.relative_to(cwd).as_posix()
                files.setdefault(rel_path, path)
    for pattern in SOURCE_MANIFEST_INCLUDE_GLOBS:
        for path in cwd.glob(pattern):
            if path.is_file():
                rel_path = path.relative_to(cwd).as_posix()
                files.setdefault(rel_path, path)
    return [files[key] for key in sorted(files)]


def build_source_manifest(*, cwd: Path | None = None) -> dict[str, Any]:
    root = (cwd or Path.cwd()).resolve()
    entries: list[dict[str, Any]] = []
    for path in _iter_source_files(root):
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha1": file_sha1(path),
                "size_bytes": path.stat().st_size,
            }
        )
    manifest_digest = rows_sha1(entries) if entries else _sha1_hexdigest(b"")
    return {
        "source_manifest_id": f"source:{manifest_digest[:12]}",
        "source_root": str(root),
        "file_count": len(entries),
        "files": entries,
    }


def write_source_manifest(path: Path, *, cwd: Path | None = None) -> dict[str, Any]:
    manifest = build_source_manifest(cwd=cwd)
    write_json(path, manifest)
    return manifest


def build_run_metadata(
    *,
    chunks_path: Path,
    eval_rows: list[dict[str, Any]],
    benchmark_source_path: Path,
    benchmark_label: str,
    benchmark_profile: str,
    corpus_label: str,
    split: str,
    cwd: Path | None = None,
    source_manifest_path: Path | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_cwd = (cwd or Path.cwd()).resolve()
    corpus_sha1 = file_sha1(chunks_path)
    benchmark_source_sha1 = file_sha1(benchmark_source_path) if benchmark_source_path.exists() else ""
    benchmark_rows_sha1 = rows_sha1(eval_rows)
    git_sha = detect_git_sha(cwd=resolved_cwd)
    source_manifest: dict[str, Any] | None = None
    if source_manifest_path is not None:
        source_manifest = write_source_manifest(source_manifest_path, cwd=resolved_cwd)
    elif not git_sha:
        source_manifest = build_source_manifest(cwd=resolved_cwd)

    metadata = {
        "split": split,
        "benchmark_label": benchmark_label,
        "benchmark_profile": benchmark_profile,
        "benchmark_id": f"benchmark:{benchmark_rows_sha1[:12]}",
        "benchmark_source_path": str(benchmark_source_path),
        "benchmark_source_sha1": benchmark_source_sha1,
        "benchmark_row_count": len(eval_rows),
        "corpus_label": corpus_label,
        "corpus_id": f"corpus:{corpus_sha1[:12]}",
        "corpus_path": str(chunks_path),
        "corpus_sha1": corpus_sha1,
        "git_sha": git_sha,
    }
    if source_manifest is not None:
        metadata["source_manifest_id"] = source_manifest["source_manifest_id"]
        metadata["source_file_count"] = source_manifest["file_count"]
        if source_manifest_path is not None:
            metadata["source_manifest_path"] = str(source_manifest_path)
    if extra_fields:
        metadata.update(extra_fields)
    return metadata
