"""Content-level integrity checks for the historical full benchmark corpus.

The reviewed full seed and historical result JSONL files prove benchmark-row
identity, but a matching ``chunk_id`` alone does not prove that a rebuilt
corpus contains the same evidence text.  This module binds those assets to a
candidate chunks file and fails closed when text, page, document, or corpus
identity drifts.

Historical results do not contain a snippet for every reviewed gold chunk.
Those chunks remain visible as ``ATTESTED_UNANCHORED`` and may count as ready
only when the caller has independently validated the reviewed seed
attestation.  They are never reported as content-verified.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from src.utils.io import read_jsonl
from src.utils.text_utils import normalize_for_match


CHUNK_ID_RE = re.compile(
    r"^(?P<doc_id>.+)-(?P<strategy>fixed_window|title_aware|table_protected)-(?P<ordinal>[cp]\d{4})$"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class HistoricalEvidenceAuditError(ValueError):
    """Raised when an audit input violates the benchmark contract."""


def _normalized_path(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


def _required_text(value: Any, *, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise HistoricalEvidenceAuditError(f"stage6 corpus attestation {label} must be non-empty")
    return text


def _required_sha256(value: Any, *, label: str) -> str:
    digest = str(value or "").strip().lower()
    if not SHA256_RE.fullmatch(digest):
        raise HistoricalEvidenceAuditError(
            f"stage6 corpus attestation {label} must be a 64-character SHA-256"
        )
    return digest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_stage6_corpus_attestation(
    attestation_path: Path,
    *,
    repo_root: Path,
    candidate_chunks_path: Path,
    seed_path: Path,
    historical_results_path: Path,
) -> dict[str, Any]:
    """Validate provenance and hashes for one reviewed Stage 6 corpus release."""

    if not attestation_path.is_file():
        raise HistoricalEvidenceAuditError(
            f"missing reviewed stage6 corpus attestation: {attestation_path}"
        )
    try:
        payload = json.loads(attestation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HistoricalEvidenceAuditError(
            f"cannot read stage6 corpus attestation {attestation_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise HistoricalEvidenceAuditError("stage6 corpus attestation must contain a JSON object")
    if payload.get("schema_version") != 1:
        raise HistoricalEvidenceAuditError("stage6 corpus attestation schema_version must equal 1")
    if payload.get("benchmark_profile") != "historical-full-raw":
        raise HistoricalEvidenceAuditError(
            "stage6 corpus attestation benchmark_profile must equal 'historical-full-raw'"
        )
    corpus_id = _required_text(payload.get("corpus_id"), label="corpus_id")

    asset = payload.get("asset")
    if not isinstance(asset, dict):
        raise HistoricalEvidenceAuditError("stage6 corpus attestation asset must be an object")
    asset_path_text = _required_text(asset.get("path"), label="asset.path")
    asset_path = Path(asset_path_text)
    if not asset_path.is_absolute():
        asset_path = repo_root / asset_path
    if _normalized_path(asset_path) != _normalized_path(candidate_chunks_path):
        raise HistoricalEvidenceAuditError(
            "stage6 corpus attestation asset.path does not match the candidate chunks path"
        )
    expected_chunks_sha256 = _required_sha256(asset.get("sha256"), label="asset.sha256")
    if asset.get("format") != "chunks-jsonl":
        raise HistoricalEvidenceAuditError(
            "stage6 corpus attestation asset.format must equal 'chunks-jsonl'"
        )
    chunk_count = asset.get("chunk_count")
    if isinstance(chunk_count, bool) or not isinstance(chunk_count, int) or chunk_count < 1:
        raise HistoricalEvidenceAuditError(
            "stage6 corpus attestation asset.chunk_count must be a positive integer"
        )
    if not candidate_chunks_path.is_file():
        raise HistoricalEvidenceAuditError(
            f"stage6 corpus attestation candidate asset is missing: {candidate_chunks_path}"
        )
    actual_chunks_sha256 = sha256_file(candidate_chunks_path)
    if actual_chunks_sha256 != expected_chunks_sha256:
        raise HistoricalEvidenceAuditError(
            "stage6 corpus attestation candidate SHA-256 mismatch: "
            f"expected {expected_chunks_sha256}, got {actual_chunks_sha256}"
        )
    actual_chunk_count = len(read_jsonl(candidate_chunks_path))
    if actual_chunk_count != chunk_count:
        raise HistoricalEvidenceAuditError(
            "stage6 corpus attestation chunk_count mismatch: "
            f"expected {chunk_count}, got {actual_chunk_count}"
        )

    source = payload.get("source")
    if not isinstance(source, dict):
        raise HistoricalEvidenceAuditError("stage6 corpus attestation source must be an object")
    source_record = {
        field: _required_text(source.get(field), label=f"source.{field}")
        for field in ("origin", "owner", "acquired_at", "source_ref")
    }

    review = payload.get("review")
    if not isinstance(review, dict):
        raise HistoricalEvidenceAuditError("stage6 corpus attestation review must be an object")
    if review.get("status") != "reviewed":
        raise HistoricalEvidenceAuditError(
            "stage6 corpus attestation review.status must equal 'reviewed'"
        )
    review_record = {
        "status": "reviewed",
        **{
            field: _required_text(review.get(field), label=f"review.{field}")
            for field in ("reviewer_id", "review_batch", "reviewed_at")
        },
    }

    binding = payload.get("benchmark_binding")
    if not isinstance(binding, dict):
        raise HistoricalEvidenceAuditError(
            "stage6 corpus attestation benchmark_binding must be an object"
        )
    expected_seed_sha256 = _required_sha256(
        binding.get("seed_sha256"), label="benchmark_binding.seed_sha256"
    )
    expected_results_sha256 = _required_sha256(
        binding.get("historical_results_sha256"),
        label="benchmark_binding.historical_results_sha256",
    )
    actual_seed_sha256 = sha256_file(seed_path)
    actual_results_sha256 = sha256_file(historical_results_path)
    if actual_seed_sha256 != expected_seed_sha256:
        raise HistoricalEvidenceAuditError(
            "stage6 corpus attestation seed SHA-256 mismatch: "
            f"expected {expected_seed_sha256}, got {actual_seed_sha256}"
        )
    if actual_results_sha256 != expected_results_sha256:
        raise HistoricalEvidenceAuditError(
            "stage6 corpus attestation historical results SHA-256 mismatch: "
            f"expected {expected_results_sha256}, got {actual_results_sha256}"
        )

    return {
        "path": str(attestation_path),
        "sha256": sha256_file(attestation_path),
        "schema_version": 1,
        "corpus_id": corpus_id,
        "benchmark_profile": "historical-full-raw",
        "asset_path": str(candidate_chunks_path),
        "asset_sha256": actual_chunks_sha256,
        "chunk_count": actual_chunk_count,
        "source": source_record,
        "review": review_record,
        "benchmark_binding": {
            "seed_sha256": actual_seed_sha256,
            "historical_results_sha256": actual_results_sha256,
        },
    }


def _unique_nonempty(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _indexed_unique_rows(rows: list[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        question_id = str(row.get("question_id") or "").strip()
        if not question_id:
            raise HistoricalEvidenceAuditError(f"{label} row {index} has no question_id")
        if question_id in indexed:
            raise HistoricalEvidenceAuditError(f"{label} contains duplicate question_id {question_id!r}")
        indexed[question_id] = row
    return indexed


def _expected_doc_id(chunk_id: str) -> str:
    match = CHUNK_ID_RE.fullmatch(chunk_id)
    if match is None:
        raise HistoricalEvidenceAuditError(f"invalid historical gold chunk_id: {chunk_id!r}")
    return match.group("doc_id")


def _page_range(item: dict[str, Any]) -> tuple[int, int] | None:
    start = item.get("page_start")
    end = item.get("page_end")
    if isinstance(start, bool) or isinstance(end, bool):
        return None
    if not isinstance(start, int) or not isinstance(end, int) or start < 1 or end < start:
        return None
    return start, end


def _pages_overlap(left: tuple[int, int] | None, right: tuple[int, int] | None) -> bool:
    if left is None or right is None:
        return False
    return max(left[0], right[0]) <= min(left[1], right[1])


def _same_file_name(left: Any, right: Any) -> bool:
    left_name = Path(str(left or "")).name
    right_name = Path(str(right or "")).name
    left_key = normalize_for_match(left_name)
    right_key = normalize_for_match(right_name)
    return bool(left_key and right_key and left_key == right_key)


def _snippet(item: dict[str, Any]) -> str:
    for field in ("snippet", "text_snippet", "text"):
        value = str(item.get(field) or "").strip()
        if value:
            return value
    return ""


def _snippet_matches(candidate_text: str, reference_text: str) -> bool:
    """Use strict normalized containment, never similarity-only acceptance."""

    candidate = normalize_for_match(candidate_text)
    reference = normalize_for_match(reference_text)
    if not candidate or not reference:
        return False
    return reference in candidate or candidate in reference


def _references_by_chunk(
    result_rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    references: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in result_rows:
        question_id = str(row["question_id"])
        for source_field in ("citations", "selected_evidence"):
            raw_items = row.get(source_field, [])
            if raw_items is None:
                continue
            if not isinstance(raw_items, list):
                raise HistoricalEvidenceAuditError(
                    f"historical result {question_id!r} field {source_field} must be a list or null"
                )
            for item in raw_items:
                if not isinstance(item, dict):
                    raise HistoricalEvidenceAuditError(
                        f"historical result {question_id!r} contains a non-object {source_field} item"
                    )
                chunk_id = str(item.get("chunk_id") or "").strip()
                if not chunk_id:
                    continue
                references[chunk_id].append(
                    {
                        "source_field": source_field,
                        "question_id": question_id,
                        "doc_id": str(item.get("doc_id") or "").strip(),
                        "file_name": str(item.get("file_name") or "").strip(),
                        "page_start": item.get("page_start"),
                        "page_end": item.get("page_end"),
                        "snippet": _snippet(item),
                    }
                )
    return references


def _seed_gold_contract(
    seed_rows: list[dict[str, Any]],
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    ordered_gold_ids: list[str] = []
    seen: set[str] = set()
    metadata: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "question_ids": [],
            "source_labels": set(),
            "target_doc_keys": set(),
            "manual_review_required": False,
        }
    )

    for row in seed_rows:
        question_id = str(row["question_id"])
        raw_ids = row.get("gold_chunk_ids", [])
        raw_keys = row.get("target_doc_keys", [])
        if not isinstance(raw_ids, list) or not isinstance(raw_keys, list):
            raise HistoricalEvidenceAuditError(
                f"seed row {question_id!r} gold_chunk_ids and target_doc_keys must be lists"
            )
        gold_ids = _unique_nonempty(raw_ids)
        doc_ids = _unique_nonempty(_expected_doc_id(chunk_id) for chunk_id in gold_ids)
        target_doc_keys = _unique_nonempty(raw_keys)
        if gold_ids and len(doc_ids) != len(target_doc_keys):
            raise HistoricalEvidenceAuditError(
                f"seed row {question_id!r} has {len(doc_ids)} gold documents but "
                f"{len(target_doc_keys)} target_doc_keys"
            )
        doc_key_by_id = dict(zip(doc_ids, target_doc_keys))
        for chunk_id in gold_ids:
            if chunk_id not in seen:
                seen.add(chunk_id)
                ordered_gold_ids.append(chunk_id)
            item = metadata[chunk_id]
            item["question_ids"].append(question_id)
            item["source_labels"].add(str(row.get("source_label") or ""))
            item["target_doc_keys"].add(doc_key_by_id[_expected_doc_id(chunk_id)])
            item["manual_review_required"] = bool(
                item["manual_review_required"] or row.get("manual_review_required", False)
            )

    for chunk_id, item in metadata.items():
        if len(item["target_doc_keys"]) != 1:
            raise HistoricalEvidenceAuditError(
                f"gold chunk {chunk_id!r} maps to conflicting target_doc_keys: "
                f"{sorted(item['target_doc_keys'])}"
            )
    return ordered_gold_ids, metadata


def _candidate_lookup(
    candidate_chunks: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    lookup: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for index, chunk in enumerate(candidate_chunks, start=1):
        chunk_id = str(chunk.get("chunk_id") or "").strip()
        if not chunk_id:
            raise HistoricalEvidenceAuditError(f"candidate chunk row {index} has no chunk_id")
        if chunk_id in lookup:
            duplicates.append(chunk_id)
            continue
        lookup[chunk_id] = chunk
    return lookup, sorted(set(duplicates))


def audit_historical_evidence(
    *,
    seed_rows: list[dict[str, Any]],
    historical_result_rows: list[dict[str, Any]],
    candidate_chunks: list[dict[str, Any]],
    candidate_corpus_sha256: str | None,
    expected_corpus_sha256: str | None,
    reviewed_attestation_validated: bool = False,
    expected_question_count: int | None = None,
) -> dict[str, Any]:
    """Audit a candidate corpus against reviewed historical benchmark evidence.

    ``reviewed_attestation_validated`` must only be set after validating the
    benchmark sidecar and its asset hashes.  It permits reviewed badcase gold
    chunks without historical snippets to remain explicitly unanchored; it
    does not promote them to content-verified.
    """

    seed_by_id = _indexed_unique_rows(seed_rows, label="seed")
    results_by_id = _indexed_unique_rows(historical_result_rows, label="historical results")
    if expected_question_count is not None and len(seed_rows) != expected_question_count:
        raise HistoricalEvidenceAuditError(
            f"expected {expected_question_count} seed rows, got {len(seed_rows)}"
        )
    if set(seed_by_id) != set(results_by_id):
        missing_results = sorted(set(seed_by_id) - set(results_by_id))
        extra_results = sorted(set(results_by_id) - set(seed_by_id))
        raise HistoricalEvidenceAuditError(
            "seed/result question_id mismatch: "
            f"missing_results={missing_results}, extra_results={extra_results}"
        )

    actual_corpus_hash = str(candidate_corpus_sha256 or "").strip().lower()
    expected_hash = str(expected_corpus_sha256 or "").strip().lower()
    corpus_hash_pinned = bool(
        SHA256_RE.fullmatch(actual_corpus_hash)
        and SHA256_RE.fullmatch(expected_hash)
        and actual_corpus_hash == expected_hash
    )

    ordered_gold_ids, seed_metadata = _seed_gold_contract(seed_rows)
    references = _references_by_chunk(historical_result_rows)
    candidates, duplicate_chunk_ids = _candidate_lookup(candidate_chunks)
    chunk_reports: list[dict[str, Any]] = []

    for chunk_id in ordered_gold_ids:
        expected_doc_id = _expected_doc_id(chunk_id)
        seed_meta = seed_metadata[chunk_id]
        target_doc_key = next(iter(seed_meta["target_doc_keys"]))
        chunk = candidates.get(chunk_id)
        refs = [ref for ref in references.get(chunk_id, []) if ref["snippet"]]
        report: dict[str, Any] = {
            "chunk_id": chunk_id,
            "expected_doc_id": expected_doc_id,
            "question_ids": list(seed_meta["question_ids"]),
            "source_labels": sorted(seed_meta["source_labels"]),
            "target_doc_key": target_doc_key,
            "historical_reference_count": len(refs),
            "status": "BLOCKED",
            "verification_basis": None,
            "issues": [],
            "candidate": None,
            "reference_checks": [],
        }
        if chunk is None:
            report["issues"].append("missing_exact_gold_chunk_id")
            chunk_reports.append(report)
            continue

        candidate_doc_id = str(chunk.get("doc_id") or "").strip()
        candidate_file_name = str(chunk.get("file_name") or "").strip()
        candidate_text = str(chunk.get("text") or "").strip()
        candidate_pages = _page_range(chunk)
        report["candidate"] = {
            "doc_id": candidate_doc_id,
            "file_name": candidate_file_name,
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "normalized_text_chars": len(normalize_for_match(candidate_text)),
        }
        if candidate_doc_id != expected_doc_id:
            report["issues"].append("candidate_doc_id_mismatch")
        if not candidate_text:
            report["issues"].append("empty_candidate_text")
        if candidate_pages is None:
            report["issues"].append("invalid_candidate_page_range")

        if refs:
            verified_reference = False
            for ref in refs:
                ref_pages = _page_range(ref)
                checks = {
                    "source_field": ref["source_field"],
                    "question_id": ref["question_id"],
                    "text_match": _snippet_matches(candidate_text, ref["snippet"]),
                    "page_match": _pages_overlap(candidate_pages, ref_pages),
                    "doc_id_match": ref["doc_id"] == expected_doc_id,
                    "file_name_match": _same_file_name(candidate_file_name, ref["file_name"]),
                }
                checks["verified"] = all(
                    checks[field]
                    for field in ("text_match", "page_match", "doc_id_match", "file_name_match")
                )
                verified_reference = bool(verified_reference or checks["verified"])
                report["reference_checks"].append(checks)
            if not verified_reference:
                if not any(item["text_match"] for item in report["reference_checks"]):
                    report["issues"].append("historical_snippet_mismatch")
                if not any(item["page_match"] for item in report["reference_checks"]):
                    report["issues"].append("historical_page_mismatch")
                if not any(item["doc_id_match"] for item in report["reference_checks"]):
                    report["issues"].append("historical_doc_id_mismatch")
                if not any(item["file_name_match"] for item in report["reference_checks"]):
                    report["issues"].append("historical_file_name_mismatch")
            elif not report["issues"]:
                report["status"] = "VERIFIED"
                report["verification_basis"] = "historical_result_content"
        else:
            source_labels = set(seed_meta["source_labels"])
            candidate_title = normalize_for_match(f"{candidate_file_name}\n{candidate_doc_id}")
            target_key = normalize_for_match(target_doc_key)
            if not target_key or target_key not in candidate_title:
                report["issues"].append("target_document_identity_mismatch")
            if source_labels != {"artifact_badcase_gold"}:
                report["issues"].append("unanchored_gold_is_not_badcase_review_source")
            if seed_meta["manual_review_required"]:
                report["issues"].append("unanchored_gold_still_requires_manual_review")
            if not reviewed_attestation_validated:
                report["issues"].append("review_attestation_not_validated")
            if not report["issues"]:
                report["status"] = "ATTESTED_UNANCHORED"
                report["verification_basis"] = "reviewed_seed_attestation"

        chunk_reports.append(report)

    status_counts = Counter(item["status"] for item in chunk_reports)
    blocking_reasons: list[str] = []
    if not expected_hash:
        blocking_reasons.append("expected candidate corpus SHA-256 is required")
    elif not SHA256_RE.fullmatch(expected_hash):
        blocking_reasons.append("expected candidate corpus SHA-256 is invalid")
    elif not SHA256_RE.fullmatch(actual_corpus_hash):
        blocking_reasons.append("actual candidate corpus SHA-256 is invalid")
    elif actual_corpus_hash != expected_hash:
        blocking_reasons.append(
            f"candidate corpus SHA-256 mismatch: expected {expected_hash}, got {actual_corpus_hash}"
        )
    if duplicate_chunk_ids:
        blocking_reasons.append(f"candidate corpus contains {len(duplicate_chunk_ids)} duplicate chunk_ids")
    blocked_chunk_count = status_counts.get("BLOCKED", 0)
    if blocked_chunk_count:
        blocking_reasons.append(f"{blocked_chunk_count} gold chunks failed content/identity checks")

    ready = not blocking_reasons
    if ready and status_counts.get("ATTESTED_UNANCHORED", 0):
        proof_level = "MIXED_CONTENT_AND_REVIEW_ATTESTATION"
    elif ready:
        proof_level = "CONTENT_ANCHORED"
    else:
        proof_level = "INCOMPLETE"
    return {
        "schema_version": 1,
        "status": "READY" if ready else "BLOCKED",
        "benchmark_profile": "historical-full-raw",
        "proof_level": proof_level,
        "question_count": len(seed_rows),
        "non_abstain_question_count": sum(not bool(row.get("must_abstain")) for row in seed_rows),
        "candidate_corpus": {
            "sha256": actual_corpus_hash or None,
            "expected_sha256": expected_hash or None,
            "hash_pinned": corpus_hash_pinned,
            "chunk_count": len(candidate_chunks),
            "unique_chunk_count": len(candidates),
            "duplicate_chunk_ids": duplicate_chunk_ids,
        },
        "reviewed_attestation_validated": reviewed_attestation_validated,
        "summary": {
            "unique_gold_chunk_count": len(ordered_gold_ids),
            "content_anchored_gold_chunk_count": sum(
                bool(item["historical_reference_count"]) for item in chunk_reports
            ),
            "verified_content_gold_chunk_count": status_counts.get("VERIFIED", 0),
            "attested_unanchored_gold_chunk_count": status_counts.get("ATTESTED_UNANCHORED", 0),
            "blocked_gold_chunk_count": blocked_chunk_count,
            "exact_id_present_count": sum(item["candidate"] is not None for item in chunk_reports),
            "status_counts": dict(sorted(status_counts.items())),
        },
        "blocking_reasons": blocking_reasons,
        "chunks": chunk_reports,
    }


def audit_historical_evidence_files(
    *,
    seed_path: Path,
    historical_results_path: Path,
    candidate_chunks_path: Path,
    expected_corpus_sha256: str | None,
    reviewed_attestation_validated: bool,
    expected_question_count: int | None = 50,
) -> dict[str, Any]:
    """Read three JSONL assets and add file identity to the audit report."""

    for label, path in (
        ("seed", seed_path),
        ("historical results", historical_results_path),
        ("candidate chunks", candidate_chunks_path),
    ):
        if not path.is_file():
            return {
                "schema_version": 1,
                "status": "BLOCKED",
                "benchmark_profile": "historical-full-raw",
                "proof_level": "INCOMPLETE",
                "blocking_reasons": [f"missing {label} asset: {path}"],
                "inputs": {
                    "seed_path": str(seed_path),
                    "historical_results_path": str(historical_results_path),
                    "candidate_chunks_path": str(candidate_chunks_path),
                },
            }

    candidate_hash = sha256_file(candidate_chunks_path)
    report = audit_historical_evidence(
        seed_rows=read_jsonl(seed_path),
        historical_result_rows=read_jsonl(historical_results_path),
        candidate_chunks=read_jsonl(candidate_chunks_path),
        candidate_corpus_sha256=candidate_hash,
        expected_corpus_sha256=expected_corpus_sha256,
        reviewed_attestation_validated=reviewed_attestation_validated,
        expected_question_count=expected_question_count,
    )
    report["inputs"] = {
        "seed_path": str(seed_path),
        "seed_sha256": sha256_file(seed_path),
        "historical_results_path": str(historical_results_path),
        "historical_results_sha256": sha256_file(historical_results_path),
        "candidate_chunks_path": str(candidate_chunks_path),
        "candidate_chunks_sha256": candidate_hash,
    }
    return report
