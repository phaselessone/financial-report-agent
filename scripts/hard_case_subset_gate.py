"""Offline regression gate for a balanced hard-case benchmark subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.hard_case_benchmark import validate_cases
from src.evaluation.run_identity import hash_benchmark_rows, hash_case_ids

DEFAULT_CASES = Path("benchmarks/hard_cases/cases.jsonl")
DEFAULT_MANIFEST = Path("benchmarks/hard_cases/contract-v1.manifest.json")
EXPECTED_CATEGORIES = (
    "simple_factual",
    "numeric_disambiguation",
    "cross_report_comparison",
    "derived_calculation",
    "multi_hop",
    "rewrite_required",
    "misleading_top_1",
    "conflicting_evidence",
    "must_abstain",
    "partial_answer",
)


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def load_balanced_subset(path: Path, *, per_category: int = 1) -> list[dict[str, Any]]:
    if per_category < 1:
        raise ValueError("per_category must be positive")
    source = _repo_path(path)
    rows = [
        json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    # Reuse the benchmark's strict contract validator before selecting the
    # subset; the CI gate must not accept malformed synthetic rows.
    rows = validate_cases(rows, require_exact_100=True)
    grouped: dict[str, list[dict[str, Any]]] = {category: [] for category in EXPECTED_CATEGORIES}
    for row in rows:
        category = row.get("category")
        if category in grouped:
            grouped[category].append(row)
    missing = [category for category, cases in grouped.items() if len(cases) < per_category]
    if missing:
        raise ValueError(f"benchmark lacks {per_category} case(s) for: {', '.join(missing)}")
    return [case for category in EXPECTED_CATEGORIES for case in grouped[category][:per_category]]


def validate_contract_manifest(cases_path: Path, manifest_path: Path) -> dict[str, Any]:
    """Bind the CI subset gate to the immutable checked-in contract identity."""

    source = _repo_path(cases_path).resolve()
    manifest_source = _repo_path(manifest_path).resolve()
    manifest = json.loads(manifest_source.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("hard-case contract manifest must be a JSON object")
    expected_fields = {
        "schema_version": "synthetic-hard-cases-manifest-v1",
        "benchmark_kind": "synthetic_contract",
        "benchmark_version": "contract-v1",
        "review_status": "synthetic_not_human_reviewed",
        "case_count": 100,
    }
    mismatches = [
        field for field, expected in expected_fields.items() if manifest.get(field) != expected
    ]
    if mismatches:
        raise ValueError("hard-case contract manifest metadata mismatch: " + ", ".join(mismatches))
    declared_cases = (manifest_source.parent / str(manifest.get("cases_file") or "")).resolve()
    if declared_cases != source:
        raise ValueError("hard-case contract manifest cases_file does not match --cases")

    rows = [
        json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    rows = validate_cases(rows, require_exact_100=True)
    actual = {
        "cases_file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "canonical_benchmark_hash": hash_benchmark_rows(rows),
        "case_ids_hash": hash_case_ids(str(row["case_id"]) for row in rows),
        "category_counts": dict(Counter(str(row["category"]) for row in rows)),
    }
    for field, value in actual.items():
        if manifest.get(field) != value:
            raise ValueError(f"hard-case contract manifest {field} mismatch")
    return {
        "manifest_path": str(manifest_source),
        "manifest_sha256": hashlib.sha256(manifest_source.read_bytes()).hexdigest(),
        **actual,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a balanced offline hard-case subset.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--per-category", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        contract = validate_contract_manifest(args.cases, args.manifest)
        subset = load_balanced_subset(args.cases, per_category=args.per_category)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"HARD_CASE_SUBSET_GATE_FAILED: {exc}")
        return 2
    encoded = "\n".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for row in subset
    ).encode("utf-8")
    print(
        json.dumps(
            {
                "case_count": len(subset),
                "category_counts": dict(Counter(row["category"] for row in subset)),
                "subset_sha256": hashlib.sha256(encoded).hexdigest(),
                "contract": contract,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
