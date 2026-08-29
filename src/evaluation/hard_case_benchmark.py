"""Hard-case benchmark contracts and deterministic evaluation (M4/M5).

No model or network calls are made. Callers provide predictions and optional
trajectories.  The checked-in 100-row synthetic asset is a fixed behavioural
contract; separately reviewed assets are the only source of quality accuracy.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.evaluation.claim_eval import evaluate_claim_verification
from src.evaluation.failure_attribution import (
    FailureType,
    attribute_failure,
    build_failure_summary,
)
from src.evaluation.profile_runtime import PROFILE_IDS, PROFILE_SPECS
from src.evaluation.run_identity import hash_benchmark_rows, hash_case_ids
from src.utils.io import ensure_dir, write_json, write_jsonl

CATEGORIES = (
    "simple_factual", "numeric_disambiguation", "cross_report_comparison",
    "derived_calculation", "multi_hop", "rewrite_required",
    "misleading_top_1", "conflicting_evidence", "must_abstain", "partial_answer",
)
SYNTHETIC_CONTRACT_VERSION = "contract-v1"
REVIEWED_MANIFEST_SCHEMA_VERSION = "reviewed-hard-cases-manifest-v1"
_HARD_CASE_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "hard_cases"
DEFAULT_REVIEWED_CASES_PATH = _HARD_CASE_DIR / "reviewed_cases.jsonl"
DEFAULT_REVIEWED_MANIFEST_PATH = _HARD_CASE_DIR / "reviewed_manifest.json"
PROFILES = PROFILE_IDS
# Compatibility/reporting view only. Runtime execution is selected exclusively
# through PROFILE_SPECS, so the evaluator derives rather than duplicates it.
PROFILE_TREATMENTS: dict[str, dict[str, bool]] = {
    profile_id: dict(PROFILE_SPECS[profile_id].capabilities.to_dict())
    for profile_id in PROFILE_IDS
}
_LEGACY_PROFILE_ALIASES = {
    "Baseline RAG": "baseline-rag",
    "Agentic RAG": "agentic-rag",
    "Structured Agent": "structured-agent",
    "Structured+Tool+Claim Verification": "full-agent",
}
_REVIEWED_CLAIM_STATUSES = {"ENTAILED", "CONTRADICTED", "INSUFFICIENT"}
_CALCULATION_STATUSES = {"SUCCESS", "FAILED", "INSUFFICIENT", "AMBIGUOUS", "SKIPPED"}


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(char in "0123456789abcdefABCDEF" for char in text)


def normalize_profile(profile: Any) -> str:
    """Return a canonical profile ID while accepting one release of labels."""
    text = str(profile or "").strip()
    canonical = _LEGACY_PROFILE_ALIASES.get(text, text.lower().replace("_", "-"))
    if canonical not in PROFILE_TREATMENTS:
        raise ValueError(f"unknown profile: {profile}")
    return canonical


def _benchmark_kind(case: Mapping[str, Any]) -> str:
    if case.get("synthetic") is True or case.get("review_status") == "synthetic_not_human_reviewed":
        return "synthetic"
    if case.get("synthetic") is False and case.get("review_status") == "reviewed":
        return "reviewed"
    raise ValueError("benchmark row must be either synthetic contract-v1 or reviewed")


def _validate_evidence(
    case: Mapping[str, Any],
    *,
    allow_empty: bool = False,
    require_source_id: bool = False,
) -> set[str]:
    evidence = case.get("evidence")
    if not isinstance(evidence, list) or (not evidence and not allow_empty):
        raise ValueError("evidence must be a list" + ("" if allow_empty else " with at least one item"))
    if not all(isinstance(item, Mapping) and item.get("evidence_id") and item.get("text") and item.get("source") for item in evidence):
        raise ValueError(f"invalid evidence in {case.get('case_id', '<unknown>')}")
    if require_source_id and not all(str(item.get("source_id") or "").strip() for item in evidence):
        raise ValueError(f"reviewed evidence requires source_id in {case.get('case_id', '<unknown>')}")
    ids = [str(item["evidence_id"]) for item in evidence]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate evidence_id in {case.get('case_id', '<unknown>')}")
    return set(ids)


def _validate_synthetic_case(case: Mapping[str, Any]) -> None:
    required = {
        "case_id", "category", "question", "evidence", "gold_answer",
        "required_claims", "required_evidence_ids", "must_abstain",
        "synthetic", "review_status",
    }
    missing = required - set(case)
    if missing:
        raise ValueError(f"missing fields: {sorted(missing)}")
    if not isinstance(case["required_claims"], list) or not isinstance(case["required_evidence_ids"], list):
        raise ValueError("required claims and evidence ids must be lists")
    if case["synthetic"] is not True or case["review_status"] != "synthetic_not_human_reviewed":
        raise ValueError("synthetic hard cases must be labelled synthetic_not_human_reviewed")
    version = case.get("benchmark_version", case.get("contract_version", SYNTHETIC_CONTRACT_VERSION))
    if version != SYNTHETIC_CONTRACT_VERSION:
        raise ValueError(f"synthetic benchmark_version must equal {SYNTHETIC_CONTRACT_VERSION}")
    evidence_ids = _validate_evidence(case)
    if not set(case["required_evidence_ids"]).issubset(evidence_ids):
        raise ValueError(f"required evidence missing in {case['case_id']}")


def _validate_reviewed_case(case: Mapping[str, Any]) -> None:
    required = {
        "case_id", "category", "question", "evidence", "gold_answer",
        "gold_claims", "gold_calculations", "required_tools", "forbidden_tools",
        "must_abstain", "partial_answer_gold", "synthetic", "review_status",
        "reviewer", "review_time", "source_provenance", "benchmark_version",
    }
    missing = required - set(case)
    if missing:
        raise ValueError(f"missing reviewed fields: {sorted(missing)}")
    reviewer = case.get("reviewer")
    reviewer_valid = bool(str(reviewer).strip()) if isinstance(reviewer, str) else isinstance(reviewer, Mapping) and bool(reviewer.get("reviewer_id") or reviewer.get("name"))
    if not reviewer_valid:
        raise ValueError("reviewer must identify a human reviewer")
    try:
        review_time = datetime.fromisoformat(str(case.get("review_time") or "").replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("review_time must be an ISO-8601 timestamp") from exc
    if review_time.tzinfo is None:
        raise ValueError("review_time must include a timezone")
    if not str(case.get("benchmark_version") or "").strip():
        raise ValueError("benchmark_version must be non-empty")
    if "benchmark_hash" in case and not _is_sha256(case.get("benchmark_hash")):
        raise ValueError("benchmark_hash must be a SHA-256 hex digest")

    provenance = case.get("source_provenance")
    if not isinstance(provenance, list) or not provenance:
        raise ValueError("source_provenance must be a non-empty list")
    provenance_ids: set[str] = set()
    for item in provenance:
        if not isinstance(item, Mapping) or not item.get("source_id") or not _is_sha256(item.get("sha256")):
            raise ValueError("source_provenance entries require source_id and SHA-256")
        source_id = str(item["source_id"])
        if source_id in provenance_ids:
            raise ValueError("source_provenance requires unique source_id values")
        provenance_ids.add(source_id)
        if not any(item.get(locator) for locator in ("uri", "path", "document_id")):
            raise ValueError("source_provenance entries require a source locator")

    evidence_ids = _validate_evidence(
        case,
        allow_empty=bool(case.get("must_abstain")),
        require_source_id=True,
    )
    evidence_source_ids = {
        str(item.get("source_id") or "")
        for item in case.get("evidence") or []
        if isinstance(item, Mapping)
    }
    if not evidence_source_ids.issubset(provenance_ids):
        raise ValueError("reviewed evidence source_id must reference source_provenance")
    gold_claims = case.get("gold_claims")
    if not isinstance(gold_claims, list) or (not gold_claims and not case.get("must_abstain")):
        raise ValueError("gold_claims must be a non-empty list unless must_abstain is true")
    gold_claim_ids: set[str] = set()
    for claim in gold_claims:
        if not isinstance(claim, Mapping) or not claim.get("claim_id") or not isinstance(claim.get("text"), str):
            raise ValueError("gold_claims entries require claim_id and text")
        claim_id = str(claim["claim_id"])
        if claim_id in gold_claim_ids:
            raise ValueError(f"duplicate gold claim_id: {claim_id}")
        gold_claim_ids.add(claim_id)
        status = str(claim.get("status") or "").upper()
        if status not in _REVIEWED_CLAIM_STATUSES:
            raise ValueError(f"invalid gold claim status: {status}")
        claim_evidence = claim.get("evidence_ids")
        if not isinstance(claim_evidence, list) or not set(map(str, claim_evidence)).issubset(evidence_ids):
            raise ValueError(f"invalid gold claim evidence_ids for {claim_id}")
        if status in {"ENTAILED", "CONTRADICTED"} and not claim_evidence:
            raise ValueError(f"gold claim {claim_id} requires evidence_ids for status {status}")

    calculations = case.get("gold_calculations")
    if not isinstance(calculations, list):
        raise ValueError("gold_calculations must be a list")
    calculation_ids: set[str] = set()
    for calculation in calculations:
        if not isinstance(calculation, Mapping):
            raise ValueError("gold_calculations entries must be objects")
        calculation_id = str(calculation.get("calculation_id") or "")
        if not calculation_id or calculation_id in calculation_ids:
            raise ValueError("gold_calculations require unique calculation_id values")
        calculation_ids.add(calculation_id)
        status = str(calculation.get("status") or "").upper()
        if status not in _CALCULATION_STATUSES:
            raise ValueError(f"invalid gold calculation status: {status}")
        if not str(calculation.get("operation") or "").strip():
            raise ValueError(f"gold calculation {calculation_id} requires operation")
        if status == "SUCCESS" and "inputs" not in calculation:
            raise ValueError(f"gold calculation {calculation_id} inputs must be a list")
        inputs = calculation.get("inputs", [])
        if not isinstance(inputs, list):
            raise ValueError(f"gold calculation {calculation_id} inputs must be a list")
        for operand in inputs:
            if not isinstance(operand, Mapping) or not operand.get("name") or "value" not in operand:
                raise ValueError(f"gold calculation {calculation_id} has an invalid input")
            operand_evidence = operand.get("evidence_ids", [])
            if not isinstance(operand_evidence, list) or not set(map(str, operand_evidence)).issubset(evidence_ids):
                raise ValueError(f"gold calculation {calculation_id} has invalid input evidence_ids")
        calc_evidence = calculation.get("evidence_ids", [])
        if not isinstance(calc_evidence, list) or not set(map(str, calc_evidence)).issubset(evidence_ids):
            raise ValueError(f"gold calculation {calculation_id} has invalid evidence_ids")
        result = calculation.get("result")
        if status == "SUCCESS" and (not isinstance(result, Mapping) or "value" not in result):
            raise ValueError(f"gold calculation {calculation_id} requires a result with value")
        if result is not None and (not isinstance(result, Mapping) or "value" not in result):
            raise ValueError(f"gold calculation {calculation_id} result must contain value when provided")

    for field in ("required_tools", "forbidden_tools"):
        if not isinstance(case.get(field), list) or any(not str(item).strip() for item in case[field]):
            raise ValueError(f"{field} must be a list of non-empty tool names")
    required_tools = set(map(str, case["required_tools"]))
    forbidden_tools = set(map(str, case["forbidden_tools"]))
    if required_tools & forbidden_tools:
        raise ValueError("required_tools and forbidden_tools must be disjoint")

    partial_gold = case.get("partial_answer_gold")
    if not isinstance(partial_gold, Mapping) or not isinstance(partial_gold.get("allowed"), bool) or not isinstance(partial_gold.get("required_claim_ids"), list):
        raise ValueError("partial_answer_gold requires allowed and required_claim_ids")
    if not set(map(str, partial_gold["required_claim_ids"])).issubset(gold_claim_ids):
        raise ValueError("partial_answer_gold references unknown gold claims")
    if case.get("must_abstain") and partial_gold["allowed"]:
        raise ValueError("partial_answer_gold cannot be allowed when must_abstain is true")


def _norm(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _reviewed_hash_row(case: Mapping[str, Any]) -> dict[str, Any]:
    """Remove self-referential identity fields before canonical hashing."""
    row = dict(case)
    row.pop("benchmark_hash", None)
    row.pop("cases_hash", None)
    row.pop("cases_sha256", None)
    row.pop("case_ids_hash", None)
    row.pop("benchmark_kind", None)
    return row


def canonical_reviewed_cases_hash(cases: Iterable[Mapping[str, Any]]) -> str:
    """Hash reviewed rows canonically and independently of JSONL order."""
    return hash_benchmark_rows(_reviewed_hash_row(case) for case in cases)


def canonical_reviewed_case_ids_hash(cases: Iterable[Mapping[str, Any]]) -> str:
    """Hash the reviewed case-id set through the shared RunIdentity seam."""
    return hash_case_ids(str(case.get("case_id") or "") for case in cases)


def _quota_map(value: Any, *, label: str, allowed: set[str] | None = None) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    result: dict[str, int] = {}
    for raw_key, raw_quota in value.items():
        key = str(raw_key or "").strip()
        if not key or isinstance(raw_quota, bool) or not isinstance(raw_quota, int):
            raise ValueError(f"{label} requires non-empty keys and integer quotas")
        quota = raw_quota
        if quota < 0:
            raise ValueError(f"{label} quotas must be non-negative integers")
        if allowed is not None and key not in allowed:
            raise ValueError(f"{label} references unknown key: {key}")
        result[key] = quota
    return result


def validate_reviewed_manifest(
    manifest: Mapping[str, Any],
    cases: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate a reviewed benchmark manifest against recomputed case facts.

    The manifest is authoritative for the benchmark asset as a whole. Per-row
    ``benchmark_hash`` values are accepted as legacy annotations but are never
    trusted as the source of benchmark identity.
    """
    if not isinstance(manifest, Mapping):
        raise ValueError("reviewed manifest must be an object")
    rows = validate_cases(cases, require_exact_100=False)
    if not rows or any(_benchmark_kind(row) != "reviewed" for row in rows):
        raise ValueError("reviewed manifest requires reviewed benchmark rows")
    if manifest.get("schema_version") != REVIEWED_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"schema_version must equal {REVIEWED_MANIFEST_SCHEMA_VERSION}")
    if manifest.get("review_status") != "reviewed":
        raise ValueError("reviewed manifest review_status must equal reviewed")
    versions = {str(row.get("benchmark_version") or "") for row in rows}
    manifest_version = str(manifest.get("benchmark_version") or "")
    if versions != {manifest_version}:
        raise ValueError("manifest benchmark_version does not match reviewed cases")

    actual_cases_hash = canonical_reviewed_cases_hash(rows)
    declared_cases_hash = str(
        manifest.get("cases_hash")
        or manifest.get("cases_sha256")
        or manifest.get("benchmark_hash")
        or ""
    )
    if declared_cases_hash != actual_cases_hash:
        raise ValueError(
            f"cases_hash mismatch: expected recomputed {actual_cases_hash}, got {declared_cases_hash or '<missing>'}"
        )
    actual_case_ids_hash = canonical_reviewed_case_ids_hash(rows)
    declared_case_ids_hash = str(
        manifest.get("case_ids_hash") or manifest.get("case_ids_sha256") or ""
    )
    if declared_case_ids_hash != actual_case_ids_hash:
        raise ValueError(
            "case_ids_hash mismatch: "
            f"expected recomputed {actual_case_ids_hash}, got {declared_case_ids_hash or '<missing>'}"
        )
    if int(manifest.get("case_count", -1)) != len(rows):
        raise ValueError(f"case_count mismatch: expected {len(rows)}")

    provenance = manifest.get("source_provenance")
    if not isinstance(provenance, list) or not provenance:
        raise ValueError("source_provenance must be a non-empty list")
    provenance_by_id: dict[str, dict[str, Any]] = {}
    for item in provenance:
        if not isinstance(item, Mapping):
            raise ValueError("source_provenance entries must be objects")
        source_id = str(item.get("source_id") or "").strip()
        if not source_id or source_id in provenance_by_id or not _is_sha256(item.get("sha256")):
            raise ValueError("source_provenance requires unique source_id and SHA-256")
        if not any(item.get(locator) for locator in ("uri", "path", "document_id")):
            raise ValueError("source_provenance entries require a source locator")
        provenance_by_id[source_id] = dict(item)

    category_counts = {category: 0 for category in CATEGORIES}
    source_case_counts: dict[str, int] = {source_id: 0 for source_id in provenance_by_id}
    for row in rows:
        category_counts[str(row["category"])] += 1
        row_provenance = {
            str(item.get("source_id") or "").strip(): item
            for item in row.get("source_provenance") or []
            if isinstance(item, Mapping)
        }
        for source_id, item in row_provenance.items():
            declared = provenance_by_id.get(source_id)
            if declared is None:
                raise ValueError(
                    f"case {row['case_id']} source_id is missing manifest provenance: {source_id}"
                )
            locators_match = all(
                not item.get(locator)
                or item.get(locator) == declared.get(locator)
                for locator in ("uri", "path", "document_id")
            )
            if item.get("sha256") != declared.get("sha256") or not locators_match:
                raise ValueError(
                    f"case {row['case_id']} provenance does not match manifest source_id {source_id}"
                )
        row_source_ids = {
            str(item.get("source_id") or "").strip()
            for item in row.get("evidence") or []
            if isinstance(item, Mapping) and str(item.get("source_id") or "").strip()
        }
        unknown = sorted(row_source_ids - set(provenance_by_id))
        if unknown:
            raise ValueError(
                f"case {row['case_id']} evidence source_id is missing provenance: {', '.join(unknown)}"
            )
        for source_id in row_source_ids:
            source_case_counts[source_id] += 1

    category_quotas = _quota_map(
        manifest.get("category_quotas"),
        label="category_quotas",
        allowed=set(CATEGORIES),
    )
    source_quotas = _quota_map(
        manifest.get("source_quotas"),
        label="source_quotas",
        allowed=set(provenance_by_id),
    )
    category_shortfalls = {
        category: quota - category_counts.get(category, 0)
        for category, quota in category_quotas.items()
        if category_counts.get(category, 0) < quota
    }
    source_shortfalls = {
        source_id: quota - source_case_counts.get(source_id, 0)
        for source_id, quota in source_quotas.items()
        if source_case_counts.get(source_id, 0) < quota
    }
    coverage = {
        "category_counts": category_counts,
        "source_case_counts": source_case_counts,
        "category_quotas": category_quotas,
        "source_quotas": source_quotas,
        "category_shortfalls": category_shortfalls,
        "source_shortfalls": source_shortfalls,
        "category_quotas_met": not category_shortfalls,
        "source_quotas_met": not source_shortfalls,
    }
    declared_coverage = manifest.get("coverage")
    if declared_coverage is not None and declared_coverage != coverage:
        raise ValueError("manifest coverage does not match recomputed category/source coverage")
    return {
        **dict(manifest),
        "cases_hash": actual_cases_hash,
        "cases_sha256": actual_cases_hash,
        "benchmark_hash": actual_cases_hash,
        "case_ids_hash": actual_case_ids_hash,
        "case_ids_sha256": actual_case_ids_hash,
        "case_count": len(rows),
        "source_provenance": list(provenance_by_id.values()),
        "category_quotas": category_quotas,
        "source_quotas": source_quotas,
        "coverage": coverage,
    }


def validate_case(case: Mapping[str, Any]) -> None:
    """Validate one synthetic contract row or one reviewed gold row."""
    if case["category"] not in CATEGORIES:
        raise ValueError(f"unknown category: {case['category']}")
    if not isinstance(case["case_id"], str) or not case["case_id"]:
        raise ValueError("case_id must be a non-empty string")
    if not isinstance(case["question"], str) or not isinstance(case["gold_answer"], str):
        raise ValueError("question and gold_answer must be strings")
    for field in (
        "required_facts",
        "required_tools",
        "allowed_variants",
        "acceptable_answers",
    ):
        if field in case and not isinstance(case[field], list):
            raise ValueError(f"{field} must be a list")
    if "notes" in case and not isinstance(case["notes"], str):
        raise ValueError("notes must be a string")
    if not isinstance(case["must_abstain"], bool):
        raise ValueError("must_abstain must be boolean")
    kind = _benchmark_kind(case)
    if case.get("benchmark_kind") not in (None, kind):
        raise ValueError(f"benchmark_kind must equal {kind}")
    if kind == "synthetic":
        _validate_synthetic_case(case)
    else:
        _validate_reviewed_case(case)


def validate_cases(cases: Iterable[Mapping[str, Any]], *, require_exact_100: bool | None = None) -> list[dict[str, Any]]:
    rows = [dict(case) for case in cases]
    kinds = {_benchmark_kind(row) for row in rows}
    if len(kinds) > 1:
        raise ValueError("synthetic and reviewed rows must be separate benchmark assets")
    kind = next(iter(kinds), "synthetic")
    if require_exact_100 is None:
        require_exact_100 = kind == "synthetic"
    if require_exact_100 and len(rows) != 100:
        raise ValueError(f"expected exactly 100 cases, got {len(rows)}")
    seen: set[str] = set()
    counts = {category: 0 for category in CATEGORIES}
    for row in rows:
        row.setdefault("benchmark_kind", kind)
        if kind == "synthetic":
            row.setdefault("benchmark_version", SYNTHETIC_CONTRACT_VERSION)
            row.setdefault("contract_version", SYNTHETIC_CONTRACT_VERSION)
        validate_case(row)
        if row["case_id"] in seen:
            raise ValueError(f"duplicate case_id: {row['case_id']}")
        seen.add(row["case_id"])
        counts[row["category"]] += 1
    if require_exact_100 and counts != {category: 10 for category in CATEGORIES}:
        raise ValueError(f"category counts must be ten each, got {counts}")
    if kind == "reviewed" and rows:
        versions = {row["benchmark_version"] for row in rows}
        if len(versions) != 1:
            raise ValueError("reviewed rows must share one benchmark_version")
    return rows


def load_cases(path: str | Path | None = None) -> list[dict[str, Any]]:
    source = Path(path) if path else Path(__file__).resolve().parents[2] / "benchmarks" / "hard_cases" / "cases.jsonl"
    rows = []
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return validate_cases(rows, require_exact_100=True if path is None else None)


def load_reviewed_cases(
    cases_path: str | Path = DEFAULT_REVIEWED_CASES_PATH,
    manifest_path: str | Path = DEFAULT_REVIEWED_MANIFEST_PATH,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load reviewed JSONL rows and return them with their validated manifest."""
    rows = load_cases(Path(cases_path))
    if not rows or any(_benchmark_kind(row) != "reviewed" for row in rows):
        raise ValueError("reviewed cases path must contain reviewed rows only")
    with Path(manifest_path).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, Mapping):
        raise ValueError("reviewed manifest must be a JSON object")
    return rows, validate_reviewed_manifest(manifest, rows)


def _contains_claim(answer: str, claim: str) -> bool:
    return _norm(claim) in _norm(answer)


def _trace_items(prediction: Mapping[str, Any]) -> list[Any]:
    """Return trajectory/tool items from the common prediction shapes."""
    for key in ("trace_events", "trajectory", "agent_steps"):
        value = prediction.get(key)
        if isinstance(value, list):
            return value
    return []


def _tool_names(prediction: Mapping[str, Any]) -> list[str]:
    raw = prediction.get("tool_calls")
    if not isinstance(raw, list):
        raw = _trace_items(prediction)
    names: list[str] = []
    for item in raw:
        if isinstance(item, str):
            candidate = item
        elif isinstance(item, Mapping):
            candidate = item.get("tool_name") or item.get("tool") or item.get("name") or item.get("action") or ""
        else:
            candidate = ""
        candidate = str(candidate or "").strip()
        if candidate and candidate not in {"rewrite", "retrieve", "synthesize", "verify", "finish"}:
            names.append(candidate)
    return names


def _required_tools(case: Mapping[str, Any]) -> set[str]:
    explicit = case.get("required_tools") or case.get("expected_tools")
    if isinstance(explicit, (list, tuple, set)):
        return {str(item).strip() for item in explicit if str(item).strip()}
    # The checked-in synthetic fixture predates the optional required_tools
    # field.  Keep its contract useful by deriving the minimum expected action
    # from the category; reviewed benchmarks should provide the field directly.
    if case.get("category") == "derived_calculation":
        return {"calculator"}
    if case.get("category") in {"simple_factual", "numeric_disambiguation", "cross_report_comparison", "multi_hop", "rewrite_required", "misleading_top_1", "conflicting_evidence", "must_abstain", "partial_answer"}:
        return {"report_search"}
    return set()


def _required_evidence(case: Mapping[str, Any]) -> set[str]:
    explicit = case.get("required_evidence_ids")
    if isinstance(explicit, (list, tuple, set)):
        return {str(item) for item in explicit if str(item)}
    if _benchmark_kind(case) != "reviewed":
        return set()
    result: set[str] = set()
    for claim in case.get("gold_claims") or []:
        if isinstance(claim, Mapping):
            result.update(str(item) for item in claim.get("evidence_ids") or [] if str(item))
    for calculation in case.get("gold_calculations") or []:
        if isinstance(calculation, Mapping):
            result.update(_calculation_evidence_ids(calculation))
    return result


def _prediction_counter(prediction: Mapping[str, Any], key: str, *aliases: str) -> float:
    value: Any = prediction.get(key)
    if value is None:
        counters = prediction.get("counters")
        if isinstance(counters, Mapping):
            value = counters.get(key)
            if value is None:
                for alias in aliases:
                    value = counters.get(alias)
                    if value is not None:
                        break
    if value is None:
        for alias in aliases:
            value = prediction.get(alias)
            if value is not None:
                break
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _calculation_execution_counts(prediction: Mapping[str, Any]) -> tuple[int, int]:
    """Count successful and emitted calculation records from the run itself."""
    records = prediction.get("calculations")
    if isinstance(records, Mapping):
        records = list(records.values())
    if not isinstance(records, list):
        records = []
    successful_count = sum(
        1
        for item in records
        if isinstance(item, Mapping)
        and str(item.get("status") or "").upper() in {"SUCCESS", "SUCCEEDED", "OK"}
    )
    if not records:
        calculation = prediction.get("calculation")
        if isinstance(calculation, Mapping):
            records = [calculation]
            successful_count = int(str(calculation.get("status") or "").upper() in {"SUCCESS", "SUCCEEDED", "OK"})
    return successful_count, len(records)


def _calculation_execution_success_rate(prediction: Mapping[str, Any]) -> float | None:
    """Return a process rate with an emitted-record denominator, never gold."""
    successful_count, record_count = _calculation_execution_counts(prediction)
    return successful_count / record_count if record_count else None


def _claim_entailment_counts(prediction: Mapping[str, Any]) -> tuple[int, int]:
    """Count ENTAILED and total claims emitted by the run."""
    claims = prediction.get("claims")
    if not isinstance(claims, list) or not claims:
        return (0, 0)
    entailed = 0
    for claim in claims:
        if not isinstance(claim, Mapping):
            continue
        verification = claim.get("verification")
        if isinstance(verification, Mapping) and "status" in verification:
            is_entailed = str(verification.get("status") or "").upper() == "ENTAILED"
        elif "status" in claim:
            is_entailed = str(claim.get("status") or "").upper() == "ENTAILED"
        else:
            # Compatibility for legacy traces that predate explicit verification.
            is_entailed = claim.get("supported") is True
        entailed += int(is_entailed)
    return entailed, len(claims)


def _claim_entailment_yield(prediction: Mapping[str, Any]) -> float | None:
    entailed, claim_count = _claim_entailment_counts(prediction)
    return entailed / claim_count if claim_count else None


def _calculation_records(prediction: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    records = prediction.get("calculations")
    if isinstance(records, Mapping):
        return [item for item in records.values() if isinstance(item, Mapping)]
    if isinstance(records, list):
        return [item for item in records if isinstance(item, Mapping)]
    calculation = prediction.get("calculation")
    return [calculation] if isinstance(calculation, Mapping) else []


def _canonical_scalar(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    try:
        number = Decimal(text.replace(",", ""))
    except (InvalidOperation, ValueError):
        return _norm(text)
    if not number.is_finite():
        return _norm(text)
    return format(number.normalize(), "f")


def _canonical_formula(value: Any) -> str:
    return "".join(str(value or "").lower().split())


def _calculation_inputs(record: Mapping[str, Any]) -> tuple[tuple[str, str, str], ...]:
    raw = record.get("inputs")
    if isinstance(raw, Mapping):
        raw = list(raw.values())
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(
        (
            _norm(item.get("name")),
            _canonical_scalar(item.get("value")),
            _norm(item.get("unit")),
        )
        for item in raw
        if isinstance(item, Mapping)
    )


def _calculation_evidence_ids(record: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()
    raw = record.get("evidence_ids")
    if isinstance(raw, (list, tuple, set)):
        result.update(str(item) for item in raw if str(item))
    inputs = record.get("inputs")
    if isinstance(inputs, Mapping):
        inputs = list(inputs.values())
    if isinstance(inputs, (list, tuple)):
        for operand in inputs:
            if not isinstance(operand, Mapping):
                continue
            many = operand.get("evidence_ids")
            if isinstance(many, (list, tuple, set)):
                result.update(str(item) for item in many if str(item))
            one = operand.get("evidence_id")
            if one:
                result.add(str(one))
    return result


def _calculation_result(record: Mapping[str, Any]) -> tuple[str, str]:
    result = record.get("result")
    if not isinstance(result, Mapping):
        return ("", "")
    return (_canonical_scalar(result.get("value")), _norm(result.get("unit")))


def _evaluate_calculation_gold(
    prediction: Mapping[str, Any],
    gold_calculations: list[Mapping[str, Any]],
) -> dict[str, Any]:
    predicted = _calculation_records(prediction)
    by_id = {str(record.get("calculation_id")): index for index, record in enumerate(predicted) if record.get("calculation_id")}
    used: set[int] = set()
    comparisons: list[dict[str, Any]] = []
    for gold in gold_calculations:
        match_index = by_id.get(str(gold.get("calculation_id") or ""))
        if match_index in used:
            match_index = None
        if match_index is None:
            operation = _norm(gold.get("operation"))
            match_index = next((index for index, record in enumerate(predicted) if index not in used and _norm(record.get("operation")) == operation), None)
        actual = predicted[match_index] if match_index is not None else {}
        if match_index is not None:
            used.add(match_index)
        gold_status = str(gold.get("status") or "").upper()
        status_match = _norm(actual.get("status")) == _norm(gold_status)
        operation_match = _norm(actual.get("operation")) == _norm(gold.get("operation"))
        evidence_applicable = gold_status == "SUCCESS" or bool(_calculation_evidence_ids(gold))
        formula_applicable = "formula" in gold
        inputs_applicable = gold_status == "SUCCESS" or "inputs" in gold
        result_applicable = gold_status == "SUCCESS" or "result" in gold
        evidence_match = (
            _calculation_evidence_ids(actual) == _calculation_evidence_ids(gold)
            if evidence_applicable
            else None
        )
        formula_match = (
            _canonical_formula(actual.get("formula")) == _canonical_formula(gold.get("formula"))
            if formula_applicable
            else None
        )
        inputs_match = (
            _calculation_inputs(actual) == _calculation_inputs(gold)
            if inputs_applicable
            else None
        )
        result_match = (
            _calculation_result(actual) == _calculation_result(gold)
            if result_applicable
            else None
        )
        applicable_matches = [
            value
            for value in (
                status_match,
                operation_match,
                evidence_match,
                formula_match,
                inputs_match,
                result_match,
            )
            if value is not None
        ]
        comparisons.append(
            {
                "gold_calculation_id": str(gold.get("calculation_id") or ""),
                "predicted_calculation_id": str(actual.get("calculation_id") or "") or None,
                "unmatched_prediction": False,
                "status_match": status_match,
                "operation_match": operation_match,
                "evidence_match": evidence_match,
                "formula_match": formula_match,
                "inputs_match": inputs_match,
                "result_match": result_match,
                "record_match": all(applicable_matches),
            }
        )
    for index, actual in enumerate(predicted):
        if index in used:
            continue
        comparisons.append(
            {
                "gold_calculation_id": None,
                "predicted_calculation_id": str(actual.get("calculation_id") or "") or None,
                "unmatched_prediction": True,
                # An unexpected calculation is a false positive in every
                # reviewed calculation dimension, not merely its final result.
                # Otherwise a system can over-predict records while retaining
                # perfect component-level accuracy.
                "status_match": False,
                "operation_match": False,
                "evidence_match": False,
                "formula_match": False,
                "inputs_match": False,
                "result_match": False,
                "record_match": False,
            }
        )
    def metric(field: str) -> tuple[int, int, float | None]:
        values = [row[field] for row in comparisons if row.get(field) is not None]
        numerator = sum(bool(value) for value in values)
        denominator = len(values)
        return numerator, denominator, (numerator / denominator) if denominator else None

    status_n, status_d, status_v = metric("status_match")
    operation_n, operation_d, operation_v = metric("operation_match")
    evidence_n, evidence_d, evidence_v = metric("evidence_match")
    formula_n, formula_d, formula_v = metric("formula_match")
    inputs_n, inputs_d, inputs_v = metric("inputs_match")
    result_n, result_d, result_v = metric("result_match")
    record_n, record_d, record_v = metric("record_match")

    return {
        "evaluated_calculation_count": len(comparisons),
        "calculation_status_accuracy": status_v,
        "calculation_status_numerator": status_n,
        "calculation_status_denominator": status_d,
        "calculation_operation_accuracy": operation_v,
        "calculation_operation_numerator": operation_n,
        "calculation_operation_denominator": operation_d,
        "calculation_evidence_accuracy": evidence_v,
        "calculation_evidence_numerator": evidence_n,
        "calculation_evidence_denominator": evidence_d,
        "calculation_formula_accuracy": formula_v,
        "calculation_formula_numerator": formula_n,
        "calculation_formula_denominator": formula_d,
        "calculation_inputs_accuracy": inputs_v,
        "calculation_inputs_numerator": inputs_n,
        "calculation_inputs_denominator": inputs_d,
        "calculation_result_accuracy": result_v,
        "calculation_result_numerator": result_n,
        "calculation_result_denominator": result_d,
        "calculation_record_accuracy": record_v,
        "calculation_record_numerator": record_n,
        "calculation_record_denominator": record_d,
        "calculation_comparison": comparisons,
    }


def _reviewed_gold_outcomes(
    case: Mapping[str, Any],
    row: Mapping[str, Any],
    tools: Sequence[str],
) -> list[dict[str, Any]]:
    """Materialize independent reviewed outcomes for failure attribution.

    These records intentionally carry no dependency information.  Metric
    disagreement proves an outcome mismatch, not a causal edge between the
    agent's runtime, calculations, tools, answer, or partial answer.
    """
    outcomes: list[dict[str, Any]] = []
    match_dimensions = (
        ("status_match", "calculation.status"),
        ("operation_match", "calculation.operation"),
        ("evidence_match", "calculation.evidence"),
        ("formula_match", "calculation.formula"),
        ("inputs_match", "calculation.inputs"),
        ("result_match", "calculation.result"),
    )
    for comparison in row.get("calculation_comparison") or []:
        if not isinstance(comparison, Mapping):
            continue
        dimensions = [
            dimension
            for field, dimension in match_dimensions
            if comparison.get(field) is False
        ]
        if comparison.get("unmatched_prediction"):
            dimensions.append("calculation.unexpected")
        if dimensions:
            outcomes.append(
                {
                    "failure_type": FailureType.CALCULATION_FAILED.value,
                    "outcome_dimensions": dimensions,
                    "gold_id": comparison.get("gold_calculation_id"),
                    "predicted_id": comparison.get("predicted_calculation_id"),
                    "node": "reviewed_gold_calculation",
                    "message": "reviewed calculation contract mismatch",
                }
            )

    actual_tools = set(map(str, tools))
    missing_required = sorted(_required_tools(case) - actual_tools)
    if missing_required:
        outcomes.append(
            {
                "failure_type": FailureType.TOOL_SELECTION_ERROR.value,
                "outcome_dimensions": ["tools.required"],
                "node": "reviewed_gold_tool_selection",
                "message": f"missing required tools: {', '.join(missing_required)}",
            }
        )
    forbidden_used = sorted(
        actual_tools & {str(item) for item in case.get("forbidden_tools") or []}
    )
    if forbidden_used:
        outcomes.append(
            {
                "failure_type": FailureType.TOOL_SELECTION_ERROR.value,
                "outcome_dimensions": ["tools.forbidden"],
                "node": "reviewed_gold_tool_selection",
                "message": f"forbidden tools used: {', '.join(forbidden_used)}",
            }
        )

    if not case.get("must_abstain") and not bool(row.get("exact_match")):
        outcomes.append(
            {
                "failure_type": FailureType.SYNTHESIS_ERROR.value,
                "outcome_dimensions": ["answer"],
                "node": "reviewed_gold_answer",
                "message": "answer does not match the reviewed answer contract",
            }
        )
    partial_gold = case.get("partial_answer_gold") or {}
    if (
        not case.get("must_abstain")
        and partial_gold.get("allowed")
        and float(row.get("partial_answer_gold_score", 0.0) or 0.0) < 1.0
    ):
        outcomes.append(
            {
                "failure_type": FailureType.SYNTHESIS_ERROR.value,
                "outcome_dimensions": ["partial_answer"],
                "node": "reviewed_gold_partial_answer",
                "message": "partial answer does not satisfy the reviewed gold contract",
            }
        )
    return outcomes


def _classify_expected_abstention(
    attribution: Mapping[str, Any], *, expected: bool, actual: bool
) -> dict[str, Any]:
    """Treat a gold-correct abstention as an outcome success without erasing its audit."""

    if not (expected and actual):
        return dict(attribution)
    observed = dict(attribution)
    classified = attribute_failure({})
    classified.update(
        {
            "expected_abstention": True,
            "outcome_status": "EXPECTED_ABSTENTION",
            "observed_process_attribution": observed,
        }
    )
    return classified


def evaluate_case(case: Mapping[str, Any], prediction: Mapping[str, Any] | None = None) -> dict[str, Any]:
    prediction = prediction or {}
    benchmark_kind = _benchmark_kind(case)
    answer = str(prediction.get("answer", prediction.get("final_answer", "")))
    abstained = bool(prediction.get("abstained", False))
    process_failure_attribution = _classify_expected_abstention(
        attribute_failure(prediction),
        expected=bool(case["must_abstain"]),
        actual=abstained,
    )
    if case["must_abstain"]:
        exact = abstained
    else:
        acceptable = [case["gold_answer"]]
        for field in ("allowed_variants", "acceptable_answers"):
            variants = case.get(field)
            if isinstance(variants, list):
                acceptable.extend(variants)
        exact = any(_norm(item) == _norm(answer) for item in acceptable)
    claims = list(case.get("required_claims") or [])
    claim_hits = sum(_contains_claim(answer, claim) for claim in claims)
    claim_coverage = claim_hits / len(claims) if claims else (1.0 if exact else 0.0)
    raw_citations = prediction.get("cited_evidence_ids") or prediction.get("citations") or []
    cited = {
        str(item.get("evidence_id", item.get("id", ""))) if isinstance(item, Mapping) else str(item)
        for item in raw_citations
    }
    cited.discard("")
    required_evidence = _required_evidence(case)
    evidence_recall = len(cited & required_evidence) / len(required_evidence) if required_evidence else 1.0
    citation_precision = len(cited & required_evidence) / len(cited) if cited else (1.0 if not required_evidence else 0.0)
    abstention_contract_match = float(abstained == bool(case["must_abstain"]))
    partial = min(1.0, max(claim_coverage, float(exact))) if case.get("partial_credit") else float(exact)
    trace = _trace_items(prediction)
    required_event = case.get("required_trace_event")
    trace_tokens = {
        str(item.get("action") or item.get("event") or item.get("node") or "")
        for item in trace
        if isinstance(item, Mapping)
    }
    trajectory_compliant = (required_event in trace or required_event in trace_tokens) if required_event else True
    tools = _tool_names(prediction)
    required_tools = _required_tools(case)
    forbidden_tools = {str(item) for item in case.get("forbidden_tools") or []}
    tool_selection = (
        len(required_tools & set(tools)) / len(required_tools)
        if required_tools
        else 1.0
    )
    unnecessary_tools = len([tool for tool in tools if required_tools and tool not in required_tools]) / len(tools) if tools else 0.0
    forbidden_tool_calls = sum(tool in forbidden_tools for tool in tools)
    forbidden_tool_call_rate = forbidden_tool_calls / len(tools) if tools else 0.0
    recovery_required = bool(case.get("must_recover") or case.get("category") == "rewrite_required")
    recovery_value = prediction.get("recovery_success")
    if recovery_value is None:
        attribution = prediction.get("failure_attribution")
        if isinstance(attribution, Mapping):
            recovery_value = attribution.get("recovered", attribution.get("recovery"))
        if recovery_value is None:
            recovery_value = prediction.get(
                "recovered", process_failure_attribution.get("recovered")
            )
    recovery_success = (float(bool(recovery_value)) if recovery_required else None)
    latency = _prediction_counter(prediction, "latency_ms", "end_to_end_latency_ms", "latency")
    llm_calls = _prediction_counter(prediction, "llm_call_count", "llm_calls")
    tool_calls = _prediction_counter(prediction, "tool_call_count", "tool_calls")
    if tool_calls == 0 and tools:
        tool_calls = float(len(tools))
    tokens = _prediction_counter(prediction, "total_tokens", "tokens")
    calculation_success_count, calculation_record_count = _calculation_execution_counts(prediction)
    calculation_execution_success_rate = _calculation_execution_success_rate(prediction)
    entailed_claim_count, emitted_claim_count = _claim_entailment_counts(prediction)
    claim_entailment_yield = _claim_entailment_yield(prediction)
    citation_contract_satisfied = float(required_evidence.issubset(cited)) if required_evidence else abstention_contract_match
    row = {
        "case_id": case["case_id"],
        "category": case["category"],
        "benchmark_kind": benchmark_kind,
        "exact_match": float(exact),
        "answer_contract_match": float(exact),
        "claim_coverage": round(claim_coverage, 4),
        "claim_entailment_yield": None if claim_entailment_yield is None else round(claim_entailment_yield, 4),
        "claim_entailment_numerator": entailed_claim_count,
        "claim_entailment_denominator": emitted_claim_count,
        "evidence_recall": round(evidence_recall, 4),
        "citation_precision": round(citation_precision, 4),
        "citation_contract_satisfied": round(citation_contract_satisfied, 4),
        "calculation_execution_success_rate": None if calculation_execution_success_rate is None else round(calculation_execution_success_rate, 4),
        "calculation_execution_numerator": calculation_success_count,
        "calculation_execution_denominator": calculation_record_count,
        "abstention_contract_match": abstention_contract_match,
        "partial_credit": round(partial, 4),
        "trajectory_compliant": float(trajectory_compliant),
        "recovery_success": recovery_success,
        "tool_selection_recall": round(tool_selection, 4),
        "forbidden_tool_contract_satisfied": float(forbidden_tool_calls == 0),
        "forbidden_tool_call_rate": round(forbidden_tool_call_rate, 4),
        "unnecessary_tool_call_rate": round(unnecessary_tools, 4),
        "llm_calls": llm_calls,
        "tool_calls": tool_calls,
        "total_tokens": tokens,
        "latency_ms": latency,
        "answer": answer,
        "abstained": abstained,
        "failure_attribution": process_failure_attribution,
        "process_failure_numerator": int(bool(process_failure_attribution.get("recovered"))),
        "process_failure_denominator": int(bool(process_failure_attribution.get("has_failure"))),
        "process_recovery_success": (
            float(bool(process_failure_attribution.get("recovered")))
            if process_failure_attribution.get("has_failure")
            else None
        ),
    }
    if benchmark_kind == "reviewed":
        predicted_claims = prediction.get("claims")
        claim_gold = evaluate_claim_verification(
            predicted_claims if isinstance(predicted_claims, list) else [],
            case.get("gold_claims") or [],
        )
        row.update({key: value for key, value in claim_gold.items() if key != "metric_scope"})
        for field, prefix in (
            ("status_match", "claim_status"),
            ("evidence_match", "claim_evidence"),
            ("verification_match", "claim_verification"),
        ):
            values = [
                comparison.get(field)
                for comparison in claim_gold.get("comparisons", [])
                if comparison.get(field) is not None
            ]
            row[f"{prefix}_numerator"] = sum(bool(value) for value in values)
            row[f"{prefix}_denominator"] = len(values)
        row.update(_evaluate_calculation_gold(prediction, list(case.get("gold_calculations") or [])))
        partial_gold = case.get("partial_answer_gold") or {}
        required_partial_claims = set(map(str, partial_gold.get("required_claim_ids") or []))
        if case.get("must_abstain"):
            partial_gold_score = float(abstained)
        elif partial_gold.get("allowed") and required_partial_claims:
            matched = {
                str(comparison.get("gold_claim_id"))
                for comparison in row.get("comparisons", [])
                if comparison.get("verification_match")
            }
            partial_gold_score = len(required_partial_claims & matched) / len(required_partial_claims)
        else:
            partial_gold_score = float(exact)
        row["partial_answer_gold_score"] = round(partial_gold_score, 4)
        row["partial_answer_gold_numerator"] = partial_gold_score
        row["partial_answer_gold_denominator"] = 1
        row["failure_attribution"] = _classify_expected_abstention(
            attribute_failure(
                prediction,
                gold=case,
                gold_outcomes=_reviewed_gold_outcomes(case, row, tools),
            ),
            expected=bool(case["must_abstain"]),
            actual=abstained,
        )
    return row


def evaluate_predictions(
    cases: Iterable[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]] | Iterable[Mapping[str, Any]] | None = None,
    *,
    reviewed_target_count: int | None = None,
    reviewed_manifest: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = list(cases)
    kinds = {_benchmark_kind(case) for case in rows}
    if len(kinds) > 1:
        raise ValueError("synthetic and reviewed predictions must be evaluated separately")
    benchmark_kind = next(iter(kinds), "synthetic")
    if reviewed_target_count is not None and reviewed_target_count < 0:
        raise ValueError("reviewed_target_count must be non-negative")
    if benchmark_kind == "synthetic" and reviewed_manifest is not None:
        raise ValueError("synthetic evaluation must not receive a reviewed manifest")
    validated_manifest = (
        validate_reviewed_manifest(reviewed_manifest, rows)
        if benchmark_kind == "reviewed" and reviewed_manifest is not None
        else None
    )
    if predictions is None:
        pred_by_id: dict[str, Mapping[str, Any]] = {}
    elif isinstance(predictions, Mapping):
        pred_by_id = dict(predictions)
    else:
        pred_by_id = {str(row.get("case_id")): row for row in predictions}
    per_case = [evaluate_case(case, pred_by_id.get(case["case_id"])) for case in rows]
    def _mean(key: str, *, skip_none: bool = False) -> float | None:
        values = [r[key] for r in per_case if not skip_none or r.get(key) is not None]
        if not values:
            return None if skip_none else 0.0
        return round(sum(float(value or 0.0) for value in values) / len(values), 4)

    def _rate_detail(
        numerator_key: str,
        denominator_key: str,
        ratio_key: str,
        source_rows: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, int | float | None]:
        selected_rows = per_case if source_rows is None else source_rows
        numerator_value = sum(float(row.get(numerator_key, 0) or 0) for row in selected_rows)
        numerator: int | float = (
            int(numerator_value) if numerator_value.is_integer() else round(numerator_value, 4)
        )
        denominator = sum(int(row.get(denominator_key, 0) or 0) for row in selected_rows)
        macro_values = [
            float(row[ratio_key])
            for row in selected_rows
            if row.get(ratio_key) is not None
        ]
        return {
            "numerator": numerator,
            "denominator": denominator,
            "micro": round(float(numerator) / denominator, 4) if denominator else None,
            "macro": round(sum(macro_values) / len(macro_values), 4) if macro_values else None,
        }

    def _average_detail(
        key: str,
        source_rows: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, int | float | None]:
        selected_rows = per_case if source_rows is None else source_rows
        values = [float(row.get(key, 0.0) or 0.0) for row in selected_rows]
        total = round(sum(values), 4)
        average = round(total / len(values), 4) if values else None
        return {
            "numerator": total,
            "denominator": len(values),
            "micro": average,
            "macro": average,
        }

    metrics = {
        "benchmark_kind": benchmark_kind,
        "total": len(per_case),
        "exact_match_rate": _mean("exact_match"),
        "answer_contract_match_rate": _mean("answer_contract_match"),
        "Answer Contract Match Rate": _mean("answer_contract_match"),
        "claim_coverage": _mean("claim_coverage"),
        "claim_entailment_yield": _mean("claim_entailment_yield", skip_none=True),
        "Claim Entailment Yield": _mean("claim_entailment_yield", skip_none=True),
        "evidence_recall": _mean("evidence_recall"),
        "citation_precision": _mean("citation_precision"),
        "citation_contract_satisfaction_rate": _mean("citation_contract_satisfied"),
        "Citation Contract Satisfaction Rate": _mean("citation_contract_satisfied"),
        "calculation_execution_success_rate": _mean("calculation_execution_success_rate", skip_none=True),
        "Calculation Execution Success Rate": _mean("calculation_execution_success_rate", skip_none=True),
        "abstention_contract_match_rate": _mean("abstention_contract_match"),
        "Abstention Contract Match Rate": _mean("abstention_contract_match"),
        "partial_credit": _mean("partial_credit"),
        "trajectory_compliance": _mean("trajectory_compliant"),
        "recovery_success_rate": _mean("recovery_success", skip_none=True),
        "Recovery Success Rate": _mean("recovery_success", skip_none=True),
        "tool_selection_recall": _mean("tool_selection_recall"),
        "Tool Selection Recall": _mean("tool_selection_recall"),
        "forbidden_tool_contract_satisfaction_rate": _mean("forbidden_tool_contract_satisfied"),
        "Forbidden Tool Contract Satisfaction Rate": _mean("forbidden_tool_contract_satisfied"),
        "forbidden_tool_call_rate": _mean("forbidden_tool_call_rate"),
        "Forbidden Tool Call Rate": _mean("forbidden_tool_call_rate"),
        "unnecessary_tool_call_rate": _mean("unnecessary_tool_call_rate"),
        "Unnecessary Tool Call Rate": _mean("unnecessary_tool_call_rate"),
        "avg_llm_calls": _mean("llm_calls"),
        "Avg LLM Calls": _mean("llm_calls"),
        "avg_tool_calls": _mean("tool_calls"),
        "Avg Tool Calls": _mean("tool_calls"),
        "avg_tokens": _mean("total_tokens"),
        "Avg Tokens": _mean("total_tokens"),
        "avg_latency_ms": _mean("latency_ms"),
        "Avg Latency": _mean("latency_ms"),
    }
    process_metrics = {
        "Claim Entailment Yield": _rate_detail(
            "claim_entailment_numerator",
            "claim_entailment_denominator",
            "claim_entailment_yield",
        ),
        "Calculation Execution Success Rate": _rate_detail(
            "calculation_execution_numerator",
            "calculation_execution_denominator",
            "calculation_execution_success_rate",
        ),
        "Recovery Success Rate": _rate_detail(
            "process_failure_numerator",
            "process_failure_denominator",
            "process_recovery_success",
        ),
        "Avg LLM Calls": _average_detail("llm_calls"),
        "Avg Tool Calls": _average_detail("tool_calls"),
        "Avg Tokens": _average_detail("total_tokens"),
        "Avg Latency": _average_detail("latency_ms"),
    }
    metrics["process_metrics"] = process_metrics
    metrics["Claim Entailment Yield"] = process_metrics["Claim Entailment Yield"]["micro"]
    metrics["claim_entailment_yield"] = metrics["Claim Entailment Yield"]
    metrics["Calculation Execution Success Rate"] = process_metrics["Calculation Execution Success Rate"]["micro"]
    metrics["calculation_execution_success_rate"] = metrics["Calculation Execution Success Rate"]
    metrics["Recovery Success Rate"] = process_metrics["Recovery Success Rate"]["micro"]
    metrics["recovery_success_rate"] = metrics["Recovery Success Rate"]
    gold_specs = (
        ("claim_status", "claim_status_accuracy", "Claim Status Accuracy"),
        ("claim_evidence", "claim_evidence_accuracy", "Claim Evidence Accuracy"),
        ("claim_verification", "claim_verification_accuracy", "Claim Verification Accuracy"),
        ("calculation_status", "calculation_status_accuracy", "Calculation Status Accuracy"),
        ("calculation_operation", "calculation_operation_accuracy", "Calculation Operation Accuracy"),
        ("calculation_evidence", "calculation_evidence_accuracy", "Calculation Evidence Accuracy"),
        ("calculation_formula", "calculation_formula_accuracy", "Calculation Formula Accuracy"),
        ("calculation_inputs", "calculation_inputs_accuracy", "Calculation Inputs Accuracy"),
        ("calculation_result", "calculation_result_accuracy", "Calculation Result Accuracy"),
        ("calculation_record", "calculation_record_accuracy", "Calculation Record Accuracy"),
        ("partial_answer_gold", "partial_answer_gold_score", "Partial Answer Gold Score"),
    )
    if benchmark_kind == "reviewed":
        gold_metrics: dict[str, dict[str, int | float | None]] = {}
        for count_prefix, ratio_key, title in gold_specs:
            detail = _rate_detail(
                f"{count_prefix}_numerator",
                f"{count_prefix}_denominator",
                ratio_key,
            )
            gold_metrics[title] = detail
            metrics[title] = detail["micro"]
            metrics[ratio_key] = detail["micro"]
        metrics["gold_metrics"] = gold_metrics

    category_metrics: dict[str, dict[str, Any]] = {}
    for category in CATEGORIES:
        category_rows = [row for row in per_case if row["category"] == category]
        category_process_metrics = {
            "Claim Entailment Yield": _rate_detail(
                "claim_entailment_numerator",
                "claim_entailment_denominator",
                "claim_entailment_yield",
                category_rows,
            ),
            "Calculation Execution Success Rate": _rate_detail(
                "calculation_execution_numerator",
                "calculation_execution_denominator",
                "calculation_execution_success_rate",
                category_rows,
            ),
            "Recovery Success Rate": _rate_detail(
                "process_failure_numerator",
                "process_failure_denominator",
                "process_recovery_success",
                category_rows,
            ),
            "Avg LLM Calls": _average_detail("llm_calls", category_rows),
            "Avg Tool Calls": _average_detail("tool_calls", category_rows),
            "Avg Tokens": _average_detail("total_tokens", category_rows),
            "Avg Latency": _average_detail("latency_ms", category_rows),
        }
        category_result: dict[str, Any] = {
            "total": len(category_rows),
            "exact_match_rate": round(sum(row["exact_match"] for row in category_rows) / max(1, len(category_rows)), 4),
            "process_metrics": category_process_metrics,
            "contract_metrics": {
                "Tool Selection Recall": _average_detail(
                    "tool_selection_recall", category_rows
                ),
                "Forbidden Tool Contract Satisfaction Rate": _average_detail(
                    "forbidden_tool_contract_satisfied", category_rows
                ),
                "Abstention Contract Match Rate": _average_detail(
                    "abstention_contract_match", category_rows
                ),
                "Partial Credit": _average_detail("partial_credit", category_rows),
            },
            "gold_metrics": {},
            "failure_summary": build_failure_summary(category_rows),
        }
        if benchmark_kind == "reviewed" and category_rows:
            category_gold_metrics: dict[str, dict[str, int | float | None]] = {}
            for count_prefix, ratio_key, title in gold_specs:
                detail = _rate_detail(
                    f"{count_prefix}_numerator",
                    f"{count_prefix}_denominator",
                    ratio_key,
                    category_rows,
                )
                category_gold_metrics[title] = detail
                category_result[ratio_key] = detail["micro"]
            category_result["gold_metrics"] = category_gold_metrics
        category_metrics[category] = category_result
    metrics["category_metrics"] = category_metrics
    metrics["failure_summary"] = build_failure_summary(per_case)
    if benchmark_kind == "reviewed":
        target = int(reviewed_target_count or 0)
        target_met = target > 0 and len(rows) >= target
        metrics["reviewed_coverage"] = {
            "reviewed_case_count": len(rows),
            "target_case_count": target or None,
            "coverage_rate": round(len(rows) / target, 4) if target > 0 else None,
            "target_met": target_met,
        }
        manifest_coverage = (
            dict(validated_manifest["coverage"])
            if validated_manifest is not None
            else None
        )
        metrics["reviewed_manifest_coverage"] = manifest_coverage
        category_quotas = (
            dict(validated_manifest.get("category_quotas") or {})
            if validated_manifest is not None
            else {}
        )
        source_quotas = (
            dict(validated_manifest.get("source_quotas") or {})
            if validated_manifest is not None
            else {}
        )
        manifest_source_ids = {
            str(item.get("source_id") or "")
            for item in (validated_manifest or {}).get("source_provenance", [])
            if isinstance(item, Mapping)
        }
        checks = {
            "reviewed_manifest_valid": validated_manifest is not None,
            "reviewed_target_declared": target > 0,
            "reviewed_target_met": target_met,
            "reviewed_claim_gold_denominator_nonzero": int(
                gold_metrics["Claim Verification Accuracy"]["denominator"] or 0
            ) > 0,
            "reviewed_calculation_gold_denominator_nonzero": int(
                gold_metrics["Calculation Record Accuracy"]["denominator"] or 0
            ) > 0,
            "all_categories_declared": (
                set(category_quotas) == set(CATEGORIES)
                and all(category_quotas[category] > 0 for category in CATEGORIES)
            ),
            "category_quotas_met": bool(
                manifest_coverage is not None
                and manifest_coverage.get("category_quotas_met")
            ),
            "source_quotas_declared": (
                bool(source_quotas)
                and set(source_quotas) == manifest_source_ids
                and all(quota > 0 for quota in source_quotas.values())
            ),
            "source_quotas_met": bool(
                manifest_coverage is not None
                and manifest_coverage.get("source_quotas_met")
            ),
            "prediction_coverage_met": bool(rows) and {
                str(row["case_id"]) for row in rows
            }.issubset(pred_by_id),
        }
        reason_for_check = {
            "reviewed_manifest_valid": "reviewed_manifest_required",
            "reviewed_target_declared": "reviewed_target_not_declared",
            "reviewed_target_met": "reviewed_target_not_met",
            "reviewed_claim_gold_denominator_nonzero": (
                "reviewed_claim_gold_denominator_zero"
            ),
            "reviewed_calculation_gold_denominator_nonzero": (
                "reviewed_calculation_gold_denominator_zero"
            ),
            "all_categories_declared": "category_quotas_incomplete",
            "category_quotas_met": "category_quota_shortfall",
            "source_quotas_declared": "source_quotas_incomplete",
            "source_quotas_met": "source_quota_shortfall",
            "prediction_coverage_met": "prediction_coverage_incomplete",
        }
        blocking_reasons = [
            reason_for_check[name]
            for name, passed in checks.items()
            if not passed
        ]
        claim_allowed = not blocking_reasons
        metrics["publish_gate"] = {
            "status": "READY" if claim_allowed else "BLOCKED",
            "allowed": claim_allowed,
            "checks": checks,
            "blocking_reasons": blocking_reasons,
        }
        metrics["performance_claim_allowed"] = claim_allowed
        metrics["performance_claim_status"] = (
            "REVIEWED_TARGET_MET" if claim_allowed else "COVERAGE_ONLY"
        )
    else:
        metrics["reviewed_coverage"] = {
            "reviewed_case_count": 0,
            "target_case_count": reviewed_target_count,
            "coverage_rate": 0.0 if reviewed_target_count else None,
            "target_met": False,
        }
        metrics["reviewed_manifest_coverage"] = None
        metrics["publish_gate"] = {
            "status": "SYNTHETIC_CONTRACT_ONLY",
            "allowed": False,
            "checks": {"reviewed_benchmark": False},
            "blocking_reasons": ["synthetic_contract_only"],
        }
        metrics["performance_claim_allowed"] = False
        metrics["performance_claim_status"] = "SYNTHETIC_CONTRACT_ONLY"
    return metrics, per_case


def write_run_bundle(
    output_dir: str | Path,
    *,
    cases: Iterable[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]] | Iterable[Mapping[str, Any]] | None = None,
    profile: str = PROFILES[0],
    seed: int = 0,
    model: str = "none",
    provider: str = "none",
    config: Mapping[str, Any] | None = None,
    trajectories: Iterable[Mapping[str, Any]] | None = None,
    corpus_hash: str = "",
    prompt_version: str = "",
    reviewed_manifest: Mapping[str, Any] | None = None,
    reviewed_target_count: int | None = None,
) -> dict[str, Any]:
    profile = normalize_profile(profile)
    rows = validate_cases(cases)
    metrics, per_case = evaluate_predictions(
        rows,
        predictions,
        reviewed_manifest=reviewed_manifest,
        reviewed_target_count=reviewed_target_count,
    )
    out = ensure_dir(Path(output_dir))
    cfg = {"seed": seed, "model": model, "provider": provider, **dict(config or {})}
    cfg["profile"] = profile
    cfg["feature_treatment"] = dict(PROFILE_TREATMENTS[profile])
    cfg["reviewed_target_count"] = reviewed_target_count
    if reviewed_manifest is not None:
        cfg["reviewed_manifest_hash"] = _sha256_json(
            validate_reviewed_manifest(reviewed_manifest, rows)
        )
    cfg["config_hash"] = _sha256_json(cfg)
    write_json(out / "config.json", cfg)
    write_json(out / "metrics.json", metrics)
    write_jsonl(out / "per_case.jsonl", per_case)
    trajectory_rows = list(trajectories or [])
    write_jsonl(out / "trajectories.jsonl", trajectory_rows)
    case_hash = _sha256_json(rows)
    benchmark_kind = _benchmark_kind(rows[0]) if rows else "synthetic"
    benchmark_version = (
        str(rows[0].get("benchmark_version") or "")
        if rows
        else SYNTHETIC_CONTRACT_VERSION
    )
    declared_benchmark_hash = (
        canonical_reviewed_cases_hash(rows)
        if benchmark_kind == "reviewed" and rows
        else case_hash
    )
    case_ids_hash = (
        canonical_reviewed_case_ids_hash(rows)
        if benchmark_kind == "reviewed" and rows
        else hash_case_ids(str(row["case_id"]) for row in rows)
        if rows
        else _sha256_json([])
    )
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(Path(__file__).resolve().parents[2]), capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = ""
    metadata = {
        "case_hash": case_hash,
        "benchmark_hash": declared_benchmark_hash,
        "case_ids_hash": case_ids_hash,
        "benchmark_kind": benchmark_kind,
        "benchmark_version": benchmark_version,
        "corpus_hash": str(corpus_hash or ""),
        "config_hash": cfg["config_hash"],
        "hashes": {
            "cases_sha256": case_hash,
            "benchmark_sha256": declared_benchmark_hash,
            "case_ids_sha256": case_ids_hash,
            "corpus_sha256": str(corpus_hash or ""),
            "config_sha256": cfg["config_hash"],
            "per_case_sha256": _sha256_json(per_case),
            "trajectories_sha256": _sha256_json(trajectory_rows),
        },
        "commit": commit,
        "git_commit": commit,
        "model": model,
        "provider": provider,
        "prompt_version": str(prompt_version or ""),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "profile": profile,
        "synthetic": benchmark_kind == "synthetic",
        "review_status": "synthetic_not_human_reviewed" if benchmark_kind == "synthetic" else "reviewed",
    }
    if reviewed_manifest is not None:
        metadata["reviewed_manifest_hash"] = cfg["reviewed_manifest_hash"]
    write_json(out / "metadata.json", metadata)
    return {"config": cfg, "metrics": metrics, "metadata": metadata}
