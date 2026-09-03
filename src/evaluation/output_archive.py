from __future__ import annotations

import re
import shutil
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from src.evaluation.run_metadata import normalize_benchmark_profile
from src.utils.io import ensure_dir, write_json

BENCHMARK_PROFILE_CONTRACT = {
    "canonical_profiles": ["current-dev", "historical-full-core", "historical-full-raw"],
    "legacy_aliases": {"historical-full": "historical-full-raw"},
    "metric_merge_policy": "forbid_cross_profile_merge",
}

ARTIFACT_LABEL_RE = re.compile(r"[^0-9A-Za-z._-]+")


def sanitize_artifact_label(label: str) -> str:
    normalized = ARTIFACT_LABEL_RE.sub("-", label.strip()).strip("-._")
    return normalized or datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")


def default_artifact_label(*, split: str, benchmark_profile: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    profile = sanitize_artifact_label(benchmark_profile or "default")
    return sanitize_artifact_label(f"remote_{timestamp}_{split}_{profile}")


def resolve_artifact_dir(*, artifact_root: Path, artifact_label: str) -> Path:
    root = ensure_dir(artifact_root)
    base_name = sanitize_artifact_label(artifact_label)
    candidate = root / base_name
    suffix = 2
    while candidate.exists():
        candidate = root / f"{base_name}-{suffix}"
        suffix += 1
    return candidate


def archive_output_bundle(
    *,
    output_dir: Path,
    artifact_dir: Path,
    include_names: tuple[str, ...] = ("reports", "badcases", "eval"),
) -> Path:
    ensure_dir(artifact_dir)
    for name in include_names:
        source = output_dir / name
        if not source.exists():
            continue
        shutil.copytree(source, artifact_dir / name, dirs_exist_ok=True)
    return artifact_dir


def write_scratch_run_policy(
    *,
    output_dir: Path,
    artifact_dir: Path | None,
    split: str,
    benchmark_profile: str,
    summary_path: Path,
    source_manifest_path: Path | None = None,
    treatments: Mapping[str, object] | None = None,
) -> Path:
    policy_path = output_dir / "RUN_POLICY.json"
    benchmark_profile = normalize_benchmark_profile(benchmark_profile)
    policy = {
        "mode": "scratch",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "benchmark_profile": benchmark_profile,
        "benchmark_profile_contract": BENCHMARK_PROFILE_CONTRACT,
        "treatments": dict(treatments or {}),
        "scratch_output_dir": str(output_dir),
        "latest_summary_path": str(summary_path),
        "latest_source_manifest_path": str(source_manifest_path) if source_manifest_path is not None else "",
        "last_archived_run": str(artifact_dir) if artifact_dir is not None else "",
    }
    write_json(policy_path, policy)
    return policy_path
