from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from src.utils.io import write_json

SOURCE_MANIFEST_INCLUDE_DIRS = ("src", "tests", "scripts", "bootstrap")
SOURCE_MANIFEST_INCLUDE_GLOBS = (
    "run_*.py",
    "Makefile",
    "requirements*.txt",
    "requirements.lock",
    ".env.example",
)
SOURCE_MANIFEST_EXCLUDED_DIRS = {"__pycache__", ".pytest_cache"}
SOURCE_MANIFEST_EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
SOURCE_MANIFEST_HASH_POLICY = "text-lf-normalized-v1"
CANONICAL_BENCHMARK_PROFILES = (
    "current-dev",
    "historical-full-core",
    "historical-full-raw",
)
LEGACY_BENCHMARK_PROFILE_ALIASES = {"historical-full": "historical-full-raw"}


def normalize_benchmark_profile(profile: str) -> str:
    """Normalize the one legacy full profile alias used by old artifacts."""

    value = str(profile or "").strip()
    return LEGACY_BENCHMARK_PROFILE_ALIASES.get(value, value)


def _sha1_hexdigest(payload: bytes) -> str:
    return hashlib.sha1(payload).hexdigest()


def _sha256_hexdigest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_manifest_bytes(path: Path) -> bytes:
    """Return checkout-stable bytes for files in the text-only source scope."""

    return path.read_bytes().replace(b"\r\n", b"\n")


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
            relative_parts = path.relative_to(root).parts
            if not path.is_file():
                continue
            if SOURCE_MANIFEST_EXCLUDED_DIRS.intersection(relative_parts):
                continue
            if path.suffix.lower() in SOURCE_MANIFEST_EXCLUDED_SUFFIXES:
                continue
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
        canonical_bytes = _source_manifest_bytes(path)
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha1": _sha1_hexdigest(canonical_bytes),
                "size_bytes": len(canonical_bytes),
            }
        )
    canonical_entries = json.dumps(entries, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest_digest = rows_sha1(entries) if entries else _sha1_hexdigest(b"")
    manifest_hash = _sha256_hexdigest(canonical_entries)
    return {
        "source_manifest_id": f"source:{manifest_digest[:12]}",
        "source_manifest_hash": manifest_hash,
        "hash_policy": SOURCE_MANIFEST_HASH_POLICY,
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

    normalized_profile = normalize_benchmark_profile(benchmark_profile)
    metadata = {
        "split": split,
        "benchmark_label": benchmark_label,
        "benchmark_profile": normalized_profile,
        "benchmark_profile_contract": {
            "canonical_profiles": list(CANONICAL_BENCHMARK_PROFILES),
            "legacy_aliases": dict(LEGACY_BENCHMARK_PROFILE_ALIASES),
            "metric_merge_policy": "forbid_cross_profile_merge",
        },
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
        metadata["source_manifest_hash"] = source_manifest["source_manifest_hash"]
        metadata["source_file_count"] = source_manifest["file_count"]
        if source_manifest_path is not None:
            metadata["source_manifest_path"] = str(source_manifest_path)
    if extra_fields:
        metadata.update(extra_fields)
    return metadata
