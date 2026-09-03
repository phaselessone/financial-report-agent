"""Strict evidence and claim provenance checks for agent trace JSONL files."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from src.agent.claims import CLAIM_TYPES, stable_claim_id
from src.agent.claim_verifier import ENTAILED, verify_claim


VERIFICATION_STATUSES = frozenset({"ENTAILED", "CONTRADICTED", "INSUFFICIENT"})
CLAIM_RECORD_FIELDS = frozenset(
    {
        "claim_id",
        "text",
        "claim_type",
        "is_core",
        "source_step_ids",
        "evidence_ids",
        "calculation_id",
        "parent_claim_ids",
        "verification",
    }
)


def _issue(
    issues: list[dict[str, Any]],
    *,
    line: int,
    code: str,
    message: str,
    claim_id: str = "",
    evidence_id: str = "",
) -> None:
    item: dict[str, Any] = {
        "line": int(line),
        "code": code,
        "error_code": code,
        "message": message,
    }
    if claim_id:
        item["claim_id"] = claim_id
    if evidence_id:
        item["evidence_id"] = evidence_id
    issues.append(item)


def _read_jsonl(
    path: str | Path, *, kind: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    source = Path(path)
    if not source.is_file():
        _issue(issues, line=0, code="MISSING_FILE", message=f"{kind} file does not exist: {source}")
        return rows, issues
    try:
        with source.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    _issue(
                        issues,
                        line=line_number,
                        code="INVALID_JSON",
                        message=f"invalid {kind} JSON: {exc.msg}",
                    )
                    continue
                if not isinstance(value, Mapping):
                    _issue(
                        issues,
                        line=line_number,
                        code="INVALID_ROW",
                        message=f"{kind} row must be an object",
                    )
                    continue
                rows.append(dict(value))
    except OSError as exc:
        _issue(issues, line=0, code="READ_ERROR", message=f"cannot read {kind} file: {exc}")
    return rows, issues


def _id_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values: list[str] = []
        for item in value:
            values.extend(_id_values(item))
        return values
    text = str(value).strip()
    return [text] if text else []


def _citation_id(citation: Mapping[str, Any]) -> str:
    return str(citation.get("evidence_id") or citation.get("chunk_id") or "").strip()


def _evidence_aliases(identifier: str, resolved: Mapping[str, Any] | None) -> set[str]:
    aliases = {str(identifier).strip()} if str(identifier).strip() else set()
    if isinstance(resolved, Mapping):
        for key in ("evidence_id", "chunk_id", "id"):
            value = str(resolved.get(key) or "").strip()
            if value:
                aliases.add(value)
    return aliases


def _has_document_page(payload: Mapping[str, Any]) -> bool:
    document = payload.get("doc_id") or payload.get("document_id") or payload.get("document")
    page = payload.get("page") or payload.get("page_num") or payload.get("page_number")
    if page is None:
        page = payload.get("page_start") or payload.get("page_end")
    return bool(str(document or "").strip()) and page not in (None, "")


def _index_evidence(rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    index: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        for key in (row.get("evidence_id"), row.get("chunk_id"), row.get("id")):
            value = str(key or "").strip()
            if value:
                index.setdefault(value, row)
    return index


def _calculation_map(value: Any) -> dict[str, Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return {str(key): record for key, record in value.items() if isinstance(record, Mapping)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return {
            str(record.get("calculation_id")): record
            for record in value
            if isinstance(record, Mapping) and str(record.get("calculation_id") or "").strip()
        }
    return {}


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _reasoning_step_ids(row: Mapping[str, Any]) -> set[str]:
    identifiers: set[str] = set()
    plan = row.get("reasoning_plan")
    if isinstance(plan, Mapping):
        for step in plan.get("steps") or []:
            if isinstance(step, Mapping) and str(step.get("step_id") or "").strip():
                identifiers.add(str(step["step_id"]).strip())
    for result in row.get("reasoning_step_results") or []:
        if isinstance(result, Mapping) and str(result.get("step_id") or "").strip():
            identifiers.add(str(result["step_id"]).strip())
    return identifiers


def _calculation_evidence_rows(
    calculation: Mapping[str, Any],
    *,
    citation_index: Mapping[str, Mapping[str, Any]],
    evidence_index: Mapping[str, Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for operand in calculation.get("inputs") or []:
        if not isinstance(operand, Mapping):
            continue
        evidence_id = str(operand.get("evidence_id") or operand.get("chunk_id") or "").strip()
        resolved = citation_index.get(evidence_id) or evidence_index.get(evidence_id)
        if evidence_id and isinstance(resolved, Mapping) and evidence_id not in seen:
            seen.add(evidence_id)
            rows.append(resolved)
    return rows


def _record_evidence_ids(record: Mapping[str, Any]) -> list[str]:
    declared = _id_values(record.get("evidence_ids"))
    if declared:
        return list(dict.fromkeys(declared))
    values: list[str] = []
    for operand in record.get("inputs") or []:
        if not isinstance(operand, Mapping):
            continue
        for evidence_id in [
            *_id_values(operand.get("evidence_id") or operand.get("chunk_id")),
            *_id_values(operand.get("upstream_evidence_ids")),
        ]:
            if evidence_id not in values:
                values.append(evidence_id)
    return values


def _validate_calculation_lineage(
    calculations: Mapping[str, Mapping[str, Any]],
    *,
    line: int,
    known_step_ids: set[str],
    citation_index: Mapping[str, Mapping[str, Any]],
    evidence_index: Mapping[str, Mapping[str, Any]],
    issues: list[dict[str, Any]],
) -> None:
    adjacency: dict[str, list[str]] = {}
    for map_id, record in calculations.items():
        calculation_id = str(record.get("calculation_id") or "").strip()
        if calculation_id != map_id:
            _issue(
                issues,
                line=line,
                code="CALCULATION_ID_MISMATCH",
                message="calculation map key must match calculation_id",
            )
        if str(record.get("status") or "").upper() != "SUCCESS":
            continue
        if record.get("verified") is not True:
            _issue(
                issues,
                line=line,
                code="CALCULATION_SUCCESS_NOT_VERIFIED",
                message=f"SUCCESS calculation must be verified: {map_id}",
            )
        inputs = record.get("inputs")
        if not _is_sequence(inputs) or not inputs:
            _issue(
                issues,
                line=line,
                code="CALCULATION_INPUTS_INVALID",
                message=f"SUCCESS calculation requires non-empty inputs: {map_id}",
            )
            continue
        actual_upstream_ids: list[str] = []
        actual_source_steps: list[str] = []
        actual_evidence_ids: list[str] = []
        for operand in inputs:
            if not isinstance(operand, Mapping):
                _issue(
                    issues,
                    line=line,
                    code="CALCULATION_INPUT_INVALID",
                    message=f"calculation input must be an object: {map_id}",
                )
                continue
            evidence_id = str(operand.get("evidence_id") or operand.get("chunk_id") or "").strip()
            fact_id = str(operand.get("fact_id") or "").strip()
            upstream_id = str(operand.get("upstream_calculation_id") or "").strip()
            source_step_id = str(operand.get("source_step_id") or "").strip()
            upstream_evidence_ids = _id_values(operand.get("upstream_evidence_ids"))
            if source_step_id:
                if source_step_id not in actual_source_steps:
                    actual_source_steps.append(source_step_id)
                if source_step_id not in known_step_ids:
                    _issue(
                        issues,
                        line=line,
                        code="CALCULATION_SOURCE_STEP_MISSING",
                        message=f"calculation source step does not resolve: {source_step_id}",
                    )
            if fact_id and fact_id in calculations:
                _issue(
                    issues,
                    line=line,
                    code="CALCULATION_ID_AS_FACT_ID",
                    message=f"calculation lineage must not impersonate fact_id: {fact_id}",
                )
            if evidence_id:
                resolved = citation_index.get(evidence_id) or evidence_index.get(evidence_id)
                if resolved is None:
                    _issue(
                        issues,
                        line=line,
                        code="CALCULATION_ORPHAN_EVIDENCE",
                        message=f"calculation input evidence does not resolve: {evidence_id}",
                        evidence_id=evidence_id,
                    )
                if evidence_id not in actual_evidence_ids:
                    actual_evidence_ids.append(evidence_id)
            if upstream_id:
                if upstream_id not in actual_upstream_ids:
                    actual_upstream_ids.append(upstream_id)
                upstream = calculations.get(upstream_id)
                if upstream is None:
                    _issue(
                        issues,
                        line=line,
                        code="CALCULATION_UPSTREAM_MISSING",
                        message=f"upstream calculation does not resolve: {upstream_id}",
                    )
                elif str(upstream.get("status") or "").upper() != "SUCCESS":
                    _issue(
                        issues,
                        line=line,
                        code="CALCULATION_UPSTREAM_NOT_SUCCESS",
                        message=f"upstream calculation is not SUCCESS: {upstream_id}",
                    )
                if not source_step_id:
                    _issue(
                        issues,
                        line=line,
                        code="CALCULATION_SOURCE_STEP_MISSING",
                        message=f"calculation source step does not resolve: {source_step_id}",
                    )
                if not upstream_evidence_ids:
                    _issue(
                        issues,
                        line=line,
                        code="CALCULATION_UPSTREAM_EVIDENCE_MISSING",
                        message=f"upstream calculation evidence is missing: {upstream_id}",
                    )
                expected_upstream_evidence = (
                    set(_record_evidence_ids(upstream)) if upstream is not None else set()
                )
                for upstream_evidence_id in upstream_evidence_ids:
                    if upstream_evidence_id not in actual_evidence_ids:
                        actual_evidence_ids.append(upstream_evidence_id)
                    if (
                        upstream is not None
                        and upstream_evidence_id not in expected_upstream_evidence
                    ):
                        _issue(
                            issues,
                            line=line,
                            code="CALCULATION_UPSTREAM_EVIDENCE_MISMATCH",
                            message=(
                                "operand upstream_evidence_ids do not match upstream "
                                f"calculation: {upstream_id}"
                            ),
                            evidence_id=upstream_evidence_id,
                        )
                    resolved = citation_index.get(upstream_evidence_id) or evidence_index.get(
                        upstream_evidence_id
                    )
                    if resolved is None:
                        _issue(
                            issues,
                            line=line,
                            code="CALCULATION_ORPHAN_EVIDENCE",
                            message=(
                                "upstream calculation evidence does not resolve: "
                                f"{upstream_evidence_id}"
                            ),
                            evidence_id=upstream_evidence_id,
                        )
            elif not evidence_id:
                if fact_id:
                    _issue(
                        issues,
                        line=line,
                        code="CALCULATION_FACT_PROVENANCE_UNRESOLVED",
                        message=(
                            "strict bundle cannot resolve a calculation fact_id without "
                            f"its source evidence: {fact_id}"
                        ),
                    )
                else:
                    _issue(
                        issues,
                        line=line,
                        code="CALCULATION_INPUT_PROVENANCE_MISSING",
                        message=f"calculation input lacks leaf or upstream provenance: {map_id}",
                    )
        adjacency[map_id] = actual_upstream_ids
        if set(_id_values(record.get("input_calculation_ids"))) != set(actual_upstream_ids):
            _issue(
                issues,
                line=line,
                code="CALCULATION_INPUT_LINEAGE_MISMATCH",
                message=f"input_calculation_ids disagree with calculation inputs: {map_id}",
            )
        if set(_id_values(record.get("source_step_ids"))) != set(actual_source_steps):
            _issue(
                issues,
                line=line,
                code="CALCULATION_STEP_LINEAGE_MISMATCH",
                message=f"source_step_ids disagree with calculation inputs: {map_id}",
            )
        if set(_id_values(record.get("evidence_ids"))) != set(actual_evidence_ids):
            _issue(
                issues,
                line=line,
                code="CALCULATION_EVIDENCE_LINEAGE_MISMATCH",
                message=f"evidence_ids disagree with calculation inputs: {map_id}",
            )

    visiting: set[str] = set()
    completed: set[str] = set()

    def visit(calculation_id: str) -> None:
        if calculation_id in completed:
            return
        if calculation_id in visiting:
            _issue(
                issues,
                line=line,
                code="CALCULATION_LINEAGE_CYCLE",
                message=f"calculation lineage contains a cycle: {calculation_id}",
            )
            return
        visiting.add(calculation_id)
        for upstream_id in adjacency.get(calculation_id, []):
            if upstream_id in calculations:
                visit(upstream_id)
        visiting.remove(calculation_id)
        completed.add(calculation_id)

    for calculation_id in calculations:
        visit(calculation_id)


def validate_trace_rows(
    trace_rows: Sequence[Mapping[str, Any]],
    *,
    chunks: Sequence[Mapping[str, Any]] = (),
    initial_issues: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Validate already-loaded trace rows and return a machine-readable report."""

    issues: list[dict[str, Any]] = [
        dict(item) for item in initial_issues if isinstance(item, Mapping)
    ]
    evidence_index = _index_evidence(chunks)
    trace_count = 0
    claim_count = 0
    evidence_reference_count = 0
    entailed_evidence_count = 0

    for line_number, row in enumerate(trace_rows, start=1):
        trace_count += 1
        if not isinstance(row, Mapping):
            _issue(
                issues,
                line=line_number,
                code="INVALID_TRACE_ROW",
                message="trace row must be an object",
            )
            continue
        citations = [item for item in (row.get("citations") or []) if isinstance(item, Mapping)]
        calculations = _calculation_map(row.get("calculations"))
        known_step_ids = _reasoning_step_ids(row)
        citation_index: dict[str, Mapping[str, Any]] = {}
        for citation in citations:
            citation_id = _citation_id(citation)
            if not citation_id:
                _issue(
                    issues,
                    line=line_number,
                    code="CITATION_MISSING_ID",
                    message="citation has no evidence_id or chunk_id",
                )
                continue
            for alias in _evidence_aliases(citation_id, citation):
                citation_index.setdefault(alias, citation)
            if not _has_document_page(citation):
                _issue(
                    issues,
                    line=line_number,
                    code="CITATION_MISSING_METADATA",
                    message="citation must include document and page metadata",
                    evidence_id=citation_id,
                )

        _validate_calculation_lineage(
            calculations,
            line=line_number,
            known_step_ids=known_step_ids,
            citation_index=citation_index,
            evidence_index=evidence_index,
            issues=issues,
        )

        claims = row.get("claims") or []
        if not isinstance(claims, Sequence) or isinstance(claims, (str, bytes, bytearray)):
            _issue(issues, line=line_number, code="INVALID_CLAIMS", message="claims must be a list")
            claims = []
        row_claim_ids: set[str] = set()
        claim_by_id: dict[str, Mapping[str, Any]] = {}
        status_by_id: dict[str, str] = {}
        entailed_evidence: set[str] = set()
        entailed_core_count = 0
        for claim in claims:
            claim_count += 1
            if not isinstance(claim, Mapping):
                _issue(
                    issues,
                    line=line_number,
                    code="INVALID_CLAIM",
                    message="claim must be an object",
                )
                continue
            missing_fields = sorted(CLAIM_RECORD_FIELDS - set(claim))
            if missing_fields:
                _issue(
                    issues,
                    line=line_number,
                    code="CLAIM_RECORD_INCOMPLETE",
                    message="claim record is missing fields: " + ", ".join(missing_fields),
                    claim_id=str(claim.get("claim_id") or "").strip(),
                )
            claim_id = str(claim.get("claim_id") or "").strip()
            claim_text = str(claim.get("text") or "").strip()
            if not claim_id:
                _issue(
                    issues,
                    line=line_number,
                    code="CLAIM_MISSING_ID",
                    message="claim_id is required",
                )
            elif claim_id in row_claim_ids:
                _issue(
                    issues,
                    line=line_number,
                    code="DUPLICATE_CLAIM_ID",
                    message="claim_id must be unique within one case trace",
                    claim_id=claim_id,
                )
            else:
                row_claim_ids.add(claim_id)
                claim_by_id[claim_id] = claim
            if not claim_text:
                _issue(
                    issues,
                    line=line_number,
                    code="CLAIM_TEXT_MISSING",
                    message="claim text must be non-empty",
                    claim_id=claim_id,
                )
            elif claim_id and stable_claim_id(claim_text) != claim_id:
                _issue(
                    issues,
                    line=line_number,
                    code="UNSTABLE_CLAIM_ID",
                    message="claim_id must equal the stable id of claim text",
                    claim_id=claim_id,
                )
            claim_type = str(claim.get("claim_type") or "").strip().upper()
            if claim_type not in CLAIM_TYPES:
                _issue(
                    issues,
                    line=line_number,
                    code="INVALID_CLAIM_TYPE",
                    message=f"claim_type must be one of {sorted(CLAIM_TYPES)}",
                    claim_id=claim_id,
                )
            if not isinstance(claim.get("is_core"), bool):
                _issue(
                    issues,
                    line=line_number,
                    code="CLAIM_CORE_INVALID",
                    message="claim is_core must be a boolean",
                    claim_id=claim_id,
                )
            for field_name in ("source_step_ids", "evidence_ids", "parent_claim_ids"):
                if field_name in claim and not _is_sequence(claim.get(field_name)):
                    _issue(
                        issues,
                        line=line_number,
                        code="CLAIM_RECORD_FIELD_INVALID",
                        message=f"claim {field_name} must be a list",
                        claim_id=claim_id,
                    )
            source_step_ids = _id_values(claim.get("source_step_ids"))
            if claim_type in {"DERIVED", "SYNTHESIZED"} and not source_step_ids:
                _issue(
                    issues,
                    line=line_number,
                    code="CLAIM_SOURCE_STEP_REQUIRED",
                    message=f"{claim_type} claim requires source_step_ids",
                    claim_id=claim_id,
                )
            for step_id in source_step_ids:
                if step_id not in known_step_ids:
                    _issue(
                        issues,
                        line=line_number,
                        code="CLAIM_SOURCE_STEP_MISSING",
                        message=f"claim source_step_id does not resolve: {step_id}",
                        claim_id=claim_id,
                    )

            verification = claim.get("verification")
            if not isinstance(verification, Mapping):
                verification = {}
                _issue(
                    issues,
                    line=line_number,
                    code="CLAIM_VERIFICATION_INVALID",
                    message="claim verification must be an object",
                    claim_id=claim_id,
                )
            else:
                verification_missing = sorted(
                    {"status", "method", "score", "reasons"} - set(verification)
                )
                if verification_missing:
                    _issue(
                        issues,
                        line=line_number,
                        code="CLAIM_VERIFICATION_INCOMPLETE",
                        message="claim verification is missing fields: "
                        + ", ".join(verification_missing),
                        claim_id=claim_id,
                    )
                if not str(verification.get("method") or "").strip():
                    _issue(
                        issues,
                        line=line_number,
                        code="CLAIM_VERIFICATION_METHOD_INVALID",
                        message="claim verification method must be non-empty",
                        claim_id=claim_id,
                    )
                try:
                    score = float(verification.get("score"))
                except (TypeError, ValueError):
                    score = -1.0
                if not 0.0 <= score <= 1.0:
                    _issue(
                        issues,
                        line=line_number,
                        code="CLAIM_VERIFICATION_SCORE_INVALID",
                        message="claim verification score must be between 0 and 1",
                        claim_id=claim_id,
                    )
                if "reasons" in verification and not _is_sequence(verification.get("reasons")):
                    _issue(
                        issues,
                        line=line_number,
                        code="CLAIM_VERIFICATION_REASONS_INVALID",
                        message="claim verification reasons must be a list",
                        claim_id=claim_id,
                    )
            status = str(
                verification.get("status")
                or claim.get("status")
                or claim.get("verification_status")
                or ""
            ).strip()
            if status not in VERIFICATION_STATUSES:
                _issue(
                    issues,
                    line=line_number,
                    code="INVALID_VERIFICATION_STATUS",
                    message=f"verification status must be one of {sorted(VERIFICATION_STATUSES)}",
                    claim_id=claim_id,
                )
            elif claim_id:
                status_by_id[claim_id] = status
                if status == ENTAILED and claim.get("is_core") is True:
                    entailed_core_count += 1

            evidence_ids = _id_values(claim.get("evidence_ids") or claim.get("evidence_id"))
            evidence_reference_count += len(evidence_ids)
            if status == "ENTAILED" and not evidence_ids:
                _issue(
                    issues,
                    line=line_number,
                    code="ENTAILED_WITHOUT_EVIDENCE",
                    message="ENTAILED claim must contain at least one evidence_id",
                    claim_id=claim_id,
                )
            for evidence_id in evidence_ids:
                resolved = citation_index.get(evidence_id) or evidence_index.get(evidence_id)
                if resolved is None:
                    _issue(
                        issues,
                        line=line_number,
                        code="ORPHAN_EVIDENCE",
                        message="claim evidence_id does not resolve to a citation or chunk",
                        claim_id=claim_id,
                        evidence_id=evidence_id,
                    )
                elif not _has_document_page(resolved):
                    _issue(
                        issues,
                        line=line_number,
                        code="EVIDENCE_MISSING_METADATA",
                        message="resolved evidence must include document and page metadata",
                        claim_id=claim_id,
                        evidence_id=evidence_id,
                    )
                if status == "ENTAILED":
                    entailed_evidence.update(_evidence_aliases(evidence_id, resolved))
                    entailed_evidence_count += 1

            if claim_type == "DERIVED":
                calculation_id = str(claim.get("calculation_id") or "").strip()
                if not calculation_id:
                    _issue(
                        issues,
                        line=line_number,
                        code="DERIVED_MISSING_CALCULATION",
                        message="DERIVED claim must contain calculation_id",
                        claim_id=claim_id,
                    )
                else:
                    calculation = calculations.get(calculation_id)
                    if (
                        calculation is None
                        or str(calculation.get("status") or "").upper() != "SUCCESS"
                    ):
                        _issue(
                            issues,
                            line=line_number,
                            code="CALCULATION_NOT_SUCCESS",
                            message="DERIVED claim calculation_id must resolve to a SUCCESS record",
                            claim_id=claim_id,
                        )
                    elif calculation.get("verified") is False:
                        _issue(
                            issues,
                            line=line_number,
                            code="CALCULATION_NOT_VERIFIED",
                            message="successful calculation record is not verified",
                            claim_id=claim_id,
                        )
                    else:
                        inputs = calculation.get("inputs")
                        if (
                            not isinstance(inputs, Sequence)
                            or isinstance(inputs, (str, bytes, bytearray))
                            or not inputs
                        ):
                            _issue(
                                issues,
                                line=line_number,
                                code="CALCULATION_INPUT_PROVENANCE_MISSING",
                                message="calculation inputs must contain evidence/fact provenance",
                                claim_id=claim_id,
                            )
                        else:
                            for operand in inputs:
                                if not isinstance(operand, Mapping):
                                    _issue(
                                        issues,
                                        line=line_number,
                                        code="CALCULATION_INPUT_INVALID",
                                        message="calculation input must be an object",
                                        claim_id=claim_id,
                                    )
                                    continue
                                provenance = str(
                                    operand.get("evidence_id")
                                    or operand.get("fact_id")
                                    or operand.get("chunk_id")
                                    or ""
                                ).strip()
                                if not provenance:
                                    _issue(
                                        issues,
                                        line=line_number,
                                        code="CALCULATION_INPUT_PROVENANCE_MISSING",
                                        message="each calculation input needs evidence_id or fact_id provenance",
                                        claim_id=claim_id,
                                    )
                                elif (
                                    operand.get("evidence_id")
                                    and provenance not in citation_index
                                    and provenance not in evidence_index
                                ):
                                    _issue(
                                        issues,
                                        line=line_number,
                                        code="CALCULATION_ORPHAN_EVIDENCE",
                                        message="calculation input evidence_id does not resolve",
                                        claim_id=claim_id,
                                        evidence_id=provenance,
                                    )
                        if status == ENTAILED:
                            independent = verify_claim(
                                claim,
                                _calculation_evidence_rows(
                                    calculation,
                                    citation_index=citation_index,
                                    evidence_index=evidence_index,
                                ),
                                calculation_lookup=calculations,
                                llm_budget=0,
                            )
                            if independent["verification"]["status"] != ENTAILED:
                                _issue(
                                    issues,
                                    line=line_number,
                                    code="DERIVED_RESULT_MISMATCH",
                                    message=(
                                        "ENTAILED DERIVED claim does not independently match "
                                        "its calculation result and provenance"
                                    ),
                                    claim_id=claim_id,
                                )

        for claim_id, claim in claim_by_id.items():
            claim_type = str(claim.get("claim_type") or "").strip().upper()
            parent_ids = _id_values(claim.get("parent_claim_ids"))
            if claim_type == "SYNTHESIZED" and not parent_ids:
                _issue(
                    issues,
                    line=line_number,
                    code="SYNTHESIZED_PARENTS_REQUIRED",
                    message="SYNTHESIZED claim requires parent_claim_ids",
                    claim_id=claim_id,
                )
            for parent_id in parent_ids:
                if parent_id not in claim_by_id:
                    _issue(
                        issues,
                        line=line_number,
                        code="CLAIM_PARENT_MISSING",
                        message=f"parent_claim_id does not resolve: {parent_id}",
                        claim_id=claim_id,
                    )
                elif status_by_id.get(parent_id) != ENTAILED:
                    _issue(
                        issues,
                        line=line_number,
                        code="CLAIM_PARENT_NOT_ENTAILED",
                        message=f"parent claim is not ENTAILED: {parent_id}",
                        claim_id=claim_id,
                    )

        visit_state: dict[str, int] = {}

        def visit_parent_chain(claim_id: str) -> None:
            marker = visit_state.get(claim_id, 0)
            if marker == 2:
                return
            if marker == 1:
                _issue(
                    issues,
                    line=line_number,
                    code="CLAIM_PARENT_CYCLE",
                    message="claim parent dependency graph contains a cycle",
                    claim_id=claim_id,
                )
                return
            visit_state[claim_id] = 1
            claim = claim_by_id.get(claim_id) or {}
            for parent_id in _id_values(claim.get("parent_claim_ids")):
                if parent_id in claim_by_id:
                    visit_parent_chain(parent_id)
            visit_state[claim_id] = 2

        for claim_id in claim_by_id:
            visit_parent_chain(claim_id)

        failed = row.get("failed") is True
        if not failed and not bool(row.get("abstained")) and entailed_core_count == 0:
            _issue(
                issues,
                line=line_number,
                code="NON_ABSTAIN_WITHOUT_ENTAILED_CORE",
                message="non-abstained answer requires at least one ENTAILED core claim",
            )

        final_evidence_ids = _id_values(row.get("used_evidence_ids"))
        for evidence_id in final_evidence_ids:
            resolved = citation_index.get(evidence_id) or evidence_index.get(evidence_id)
            if resolved is None:
                _issue(
                    issues,
                    line=line_number,
                    code="FINAL_ORPHAN_EVIDENCE",
                    message="final answer references an unknown evidence_id",
                    evidence_id=evidence_id,
                )
            elif evidence_id not in entailed_evidence:
                _issue(
                    issues,
                    line=line_number,
                    code="FINAL_UNVERIFIED_EVIDENCE",
                    message="final answer references evidence not attached to an ENTAILED claim",
                    evidence_id=evidence_id,
                )
        if bool(row.get("abstained")) and final_evidence_ids:
            _issue(
                issues,
                line=line_number,
                code="ABSTAIN_WITH_EVIDENCE",
                message="abstain trace must not publish used_evidence_ids",
            )
        if bool(row.get("abstained")) and citations:
            _issue(
                issues,
                line=line_number,
                code="ABSTAIN_WITH_CITATION",
                message="abstain trace must not publish citations",
            )
        if failed and str(row.get("final_answer") or "").strip():
            _issue(
                issues,
                line=line_number,
                code="FAILED_WITH_FINAL_ANSWER",
                message="failed trace must not publish a final answer",
            )
        if failed and final_evidence_ids:
            _issue(
                issues,
                line=line_number,
                code="FAILED_WITH_EVIDENCE",
                message="failed trace must not publish used_evidence_ids",
            )
        if failed and citations:
            _issue(
                issues,
                line=line_number,
                code="FAILED_WITH_CITATION",
                message="failed trace must not publish citations",
            )
        # Citations that are published with the final answer must also be
        # backed by a verified claim, even when used_evidence_ids was omitted.
        for citation in citations:
            citation_id = _citation_id(citation)
            if (
                citation_id
                and citation_id not in entailed_evidence
                and not bool(row.get("abstained"))
                and not failed
            ):
                _issue(
                    issues,
                    line=line_number,
                    code="FINAL_UNVERIFIED_CITATION",
                    message="final citation is not attached to an ENTAILED claim",
                    evidence_id=citation_id,
                )

    summary = {
        "trace_rows": trace_count,
        "claim_count": claim_count,
        "evidence_reference_count": evidence_reference_count,
        "entailed_evidence_reference_count": entailed_evidence_count,
        "issue_count": len(issues),
        "error_count": len(issues),
    }
    return {
        "status": "READY" if not issues else "BLOCKED",
        "issues": issues,
        "errors": issues,
        "summary": summary,
    }


def validate_evidence_integrity(
    trace_path: str | Path,
    chunks_path: str | Path,
) -> dict[str, Any]:
    """Load trace/chunk JSONL assets and run :func:`validate_trace_rows`."""

    trace_rows, trace_issues = _read_jsonl(trace_path, kind="trace")
    chunk_rows, chunk_issues = _read_jsonl(chunks_path, kind="chunks")
    report = validate_trace_rows(
        trace_rows, chunks=chunk_rows, initial_issues=[*trace_issues, *chunk_issues]
    )
    report["trace_path"] = str(Path(trace_path))
    report["chunks_path"] = str(Path(chunks_path))
    report["asset_status"] = {
        "trace": "READY" if not trace_issues else "BLOCKED",
        "chunks": "READY" if not chunk_issues else "BLOCKED",
    }
    return report


def _bundle_identity_issues(bundle: Any) -> list[dict[str, Any]]:
    expected_identity = bundle.identity.to_dict()
    issues: list[dict[str, Any]] = []
    for line, row in enumerate(bundle.trajectories, start=1):
        if not isinstance(row, Mapping):
            continue
        question_id = str(row.get("question_id") or row.get("case_id") or "").strip()
        list_sections = (
            "claims",
            "citations",
            "used_evidence_ids",
            "tool_calls",
            "trajectory_events",
        )
        mapping_sections = (
            "calculations",
            "dependency_coverage",
            "failure_attribution",
        )
        for section in list_sections:
            value = row.get(section)
            if section not in row:
                _issue(
                    issues,
                    line=line,
                    code="TRACE_SECTION_MISSING",
                    message=f"strict trajectory requires section: {section}",
                )
                issues[-1].update({"question_id": question_id, "section": section})
            elif not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
                _issue(
                    issues,
                    line=line,
                    code="TRACE_SECTION_INVALID",
                    message=f"strict trajectory section must be a list: {section}",
                )
                issues[-1].update({"question_id": question_id, "section": section})
        for section in mapping_sections:
            if section not in row:
                _issue(
                    issues,
                    line=line,
                    code="TRACE_SECTION_MISSING",
                    message=f"strict trajectory requires section: {section}",
                )
                issues[-1].update({"question_id": question_id, "section": section})
            elif not isinstance(row.get(section), Mapping):
                _issue(
                    issues,
                    line=line,
                    code="TRACE_SECTION_INVALID",
                    message=f"strict trajectory section must be an object: {section}",
                )
                issues[-1].update({"question_id": question_id, "section": section})
        trajectory_events = row.get("trajectory_events")
        if (
            isinstance(trajectory_events, Sequence)
            and not isinstance(trajectory_events, (str, bytes, bytearray))
            and not trajectory_events
        ):
            _issue(
                issues,
                line=line,
                code="TRACE_EVENTS_EMPTY",
                message="strict trajectory requires at least one graph event",
            )
            issues[-1]["question_id"] = question_id
        if "abstained" not in row or not isinstance(row.get("abstained"), bool):
            _issue(
                issues,
                line=line,
                code="TRACE_ABSTENTION_INVALID",
                message="strict trajectory requires a boolean abstained field",
            )
            issues[-1]["question_id"] = question_id
        failed = row.get("failed", False)
        if "failed" in row and not isinstance(failed, bool):
            _issue(
                issues,
                line=line,
                code="TRACE_FAILED_INVALID",
                message="strict trajectory failed field must be boolean when present",
            )
            issues[-1]["question_id"] = question_id
        elif failed:
            if not str(row.get("error_type") or "").strip():
                _issue(
                    issues,
                    line=line,
                    code="FAILED_ERROR_TYPE_MISSING",
                    message="failed trajectory requires a non-empty error_type",
                )
                issues[-1]["question_id"] = question_id
            failure_attribution = row.get("failure_attribution")
            if (
                not isinstance(failure_attribution, Mapping)
                or failure_attribution.get("has_failure") is not True
            ):
                _issue(
                    issues,
                    line=line,
                    code="FAILED_ATTRIBUTION_MISSING",
                    message="failed trajectory requires failure_attribution.has_failure=true",
                )
                issues[-1]["question_id"] = question_id
            has_failed_event = isinstance(trajectory_events, Sequence) and any(
                isinstance(event, Mapping)
                and str(event.get("status") or "").upper() in {"FAILED", "ERROR", "EXCEPTION"}
                for event in trajectory_events
            )
            if not has_failed_event:
                _issue(
                    issues,
                    line=line,
                    code="FAILED_EVENT_MISSING",
                    message="failed trajectory requires at least one failed trajectory event",
                )
                issues[-1]["question_id"] = question_id
        claims = row.get("claims")
        if (
            failed is not True
            and row.get("abstained") is False
            and isinstance(claims, Sequence)
            and not isinstance(claims, (str, bytes, bytearray))
            and not claims
        ):
            _issue(
                issues,
                line=line,
                code="NON_ABSTAIN_WITHOUT_CLAIMS",
                message="a non-abstained strict trajectory requires verified claims",
            )
            issues[-1]["question_id"] = question_id
        if row.get("run_id") != bundle.identity.run_id:
            _issue(
                issues,
                line=line,
                code="TRACE_RUN_ID_MISMATCH",
                message="trajectory run_id does not match the strict eval bundle",
            )
            issues[-1]["question_id"] = question_id
        if row.get("run_identity") != expected_identity:
            _issue(
                issues,
                line=line,
                code="TRACE_RUN_IDENTITY_MISMATCH",
                message="trajectory run_identity does not match the strict eval bundle",
            )
            issues[-1]["question_id"] = question_id
    return issues


def _merge_issues(report: dict[str, Any], issues: Sequence[Mapping[str, Any]]) -> None:
    if not issues:
        return
    merged = [*report.get("issues", []), *(dict(issue) for issue in issues)]
    report["issues"] = merged
    report["errors"] = merged
    report.setdefault("summary", {})["issue_count"] = len(merged)
    report["summary"]["error_count"] = len(merged)
    report["status"] = "BLOCKED"


def validate_bundle_evidence_integrity(bundle: Any, chunks_path: str | Path) -> dict[str, Any]:
    """Validate agent provenance and exact RunIdentity for a strict eval bundle."""

    report = validate_evidence_integrity(bundle.path / "trajectories.jsonl", chunks_path)
    identity_issues = _bundle_identity_issues(bundle)
    if not bundle.trajectories:
        _issue(
            identity_issues,
            line=0,
            code="BUNDLE_TRAJECTORIES_EMPTY",
            message="agent evidence integrity requires at least one trajectory row",
        )
    if str(bundle.metadata.get("mode") or "").lower() != "agentic":
        _issue(
            identity_issues,
            line=0,
            code="BUNDLE_NOT_AGENTIC",
            message="evidence-integrity bundle must declare metadata.mode=agentic",
        )
    _merge_issues(report, identity_issues)
    report["bundle"] = {
        "path": str(bundle.path),
        "run_id": bundle.identity.run_id,
        "identity_hash": bundle.identity.identity_hash,
    }
    return report


def validate_trace(
    trace: str | Path | Sequence[Mapping[str, Any]],
    chunks: str | Path | Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate either JSONL paths or already-loaded trace/chunk rows."""

    if isinstance(trace, (str, Path)):
        if not isinstance(chunks, (str, Path)):
            raise TypeError("chunks must be a path when trace is a path")
        return validate_evidence_integrity(trace, chunks)
    if isinstance(chunks, (str, Path)):
        chunk_rows, chunk_issues = _read_jsonl(chunks, kind="chunks")
        return validate_trace_rows(trace, chunks=chunk_rows, initial_issues=chunk_issues)
    return validate_trace_rows(trace, chunks=chunks)


check_trace_integrity = validate_trace
validate_trace_file = validate_evidence_integrity


__all__ = [
    "VERIFICATION_STATUSES",
    "check_trace_integrity",
    "validate_bundle_evidence_integrity",
    "validate_evidence_integrity",
    "validate_trace",
    "validate_trace_file",
    "validate_trace_rows",
]
