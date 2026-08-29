from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.agent.claims import stable_claim_id
from src.evaluation.evidence_integrity import validate_trace_rows, validate_evidence_integrity


CHUNK = {"chunk_id": "c1", "doc_id": "d1", "page_start": 3, "page_end": 3, "text": "source"}


def _claim(*, status: str = "ENTAILED", evidence_ids=None, claim_type: str = "EXTRACTED", **extra):
    text = str(extra.pop("text", "事实"))
    item = {
        "claim_id": stable_claim_id(text),
        "claim_type": claim_type,
        "text": text,
        "is_core": True,
        "source_step_ids": [],
        "evidence_ids": ["c1"] if evidence_ids is None else evidence_ids,
        "calculation_id": None,
        "parent_claim_ids": [],
        "verification": {"status": status, "method": "deterministic", "score": 1.0, "reasons": []},
        "supported": status == "ENTAILED",
    }
    item.update(extra)
    return item


def _trace(*, claims=None, citations=None, used=None, abstained=False, calculations=None):
    claim_values = [_claim()] if claims is None else claims
    return {
        "question_id": "q1",
        "claims": list(claim_values or []),
        "citations": citations if citations is not None else [{"evidence_id": "c1", "doc_id": "d1", "page": 3}],
        "used_evidence_ids": ["c1"] if used is None and not abstained else (used or []),
        "abstained": abstained,
        "calculations": calculations or {},
    }


def test_complete_trace_is_ready():
    report = validate_trace_rows([_trace()], chunks=[CHUNK])
    assert report["status"] == "READY"
    assert report["summary"]["claim_count"] == 1
    assert report["issues"] == []


def test_citation_without_document_page_is_blocked():
    report = validate_trace_rows(
        [_trace(citations=[{"evidence_id": "c1", "doc_id": "d1"}])],
        chunks=[],
    )
    assert report["status"] == "BLOCKED"
    assert any(issue["code"] == "CITATION_MISSING_METADATA" for issue in report["issues"])


def test_orphan_evidence_is_blocked():
    report = validate_trace_rows(
        [_trace(claims=[_claim(evidence_ids=["missing"])], citations=[], used=["missing"])],
        chunks=[],
    )
    codes = {issue["code"] for issue in report["issues"]}
    assert "ORPHAN_EVIDENCE" in codes
    assert "FINAL_ORPHAN_EVIDENCE" in codes


def test_claim_id_uniqueness_is_scoped_to_one_case_trace():
    first = _trace()
    second = {**_trace(), "question_id": "q2"}
    report = validate_trace_rows([first, second], chunks=[CHUNK])
    assert report["status"] == "READY"

    duplicate_in_one_case = validate_trace_rows(
        [_trace(claims=[_claim(), _claim()])],
        chunks=[CHUNK],
    )
    assert any(
        issue["code"] == "DUPLICATE_CLAIM_ID"
        for issue in duplicate_in_one_case["issues"]
    )


def test_invalid_claim_status_is_blocked():
    invalid = validate_trace_rows([_trace(claims=[_claim(status="UNKNOWN")])], chunks=[CHUNK])
    assert any(issue["code"] == "INVALID_VERIFICATION_STATUS" for issue in invalid["issues"])


def test_non_abstain_requires_at_least_one_entailed_core_claim():
    report = validate_trace_rows(
        [
            _trace(
                claims=[_claim(status="INSUFFICIENT", evidence_ids=[])],
                citations=[],
                used=[],
            )
        ],
        chunks=[CHUNK],
    )

    assert any(
        issue["code"] == "NON_ABSTAIN_WITHOUT_ENTAILED_CORE"
        for issue in report["issues"]
    )


def test_failed_trace_may_have_no_claims_but_cannot_publish_an_answer_or_evidence():
    failed_trace = {
        **_trace(claims=[], citations=[], used=[]),
        "failed": True,
        "error_type": "RuntimeError",
        "final_answer": "",
    }
    report = validate_trace_rows([failed_trace], chunks=[CHUNK])
    assert report["status"] == "READY"

    published = {
        **failed_trace,
        "final_answer": "不应发布的答案",
        "citations": [{"evidence_id": "c1", "doc_id": "d1", "page": 3}],
        "used_evidence_ids": ["c1"],
    }
    blocked = validate_trace_rows([published], chunks=[CHUNK])
    codes = {issue["code"] for issue in blocked["issues"]}
    assert "FAILED_WITH_FINAL_ANSWER" in codes
    assert "FAILED_WITH_CITATION" in codes
    assert "FAILED_WITH_EVIDENCE" in codes


def test_claim_record_schema_and_dependency_ids_are_strictly_validated():
    incomplete = _claim()
    incomplete.pop("is_core")
    incomplete.pop("source_step_ids")
    incomplete.pop("parent_claim_ids")
    schema_report = validate_trace_rows([_trace(claims=[incomplete])], chunks=[CHUNK])
    assert any(issue["code"] == "CLAIM_RECORD_INCOMPLETE" for issue in schema_report["issues"])

    parent = _claim(text="父事实")
    child = _claim(
        text="综合结论",
        claim_type="SYNTHESIZED",
        parent_claim_ids=["missing-parent"],
        source_step_ids=["missing-step"],
    )
    dependency_report = validate_trace_rows(
        [
            {
                **_trace(claims=[parent, child]),
                "reasoning_plan": {
                    "plan_id": "P1",
                    "steps": [{"step_id": "lookup-1", "kind": "LOOKUP"}],
                    "answer_requirement_ids": ["lookup-1"],
                },
            }
        ],
        chunks=[CHUNK],
    )
    codes = {issue["code"] for issue in dependency_report["issues"]}
    assert "CLAIM_PARENT_MISSING" in codes
    assert "CLAIM_SOURCE_STEP_MISSING" in codes


def test_entailed_claim_requires_evidence_id():
    report = validate_trace_rows(
        [_trace(claims=[_claim(evidence_ids=[])], citations=[], used=[])],
        chunks=[CHUNK],
    )
    assert any(issue["code"] == "ENTAILED_WITHOUT_EVIDENCE" for issue in report["issues"])


def test_derived_claim_requires_successful_calculation_and_provenance():
    reasoning_plan = {
        "plan_id": "P-CALC",
        "steps": [{"step_id": "calculate_yoy", "kind": "CALCULATE"}],
        "answer_requirement_ids": ["calculate_yoy"],
    }
    citations = [
        {"evidence_id": "c1", "doc_id": "d1", "page": 3},
        {"evidence_id": "c2", "doc_id": "d2", "page": 4},
    ]
    chunks = [
        CHUNK,
        {"chunk_id": "c2", "doc_id": "d2", "page": 4, "text": "甲公司2025年营业收入120元"},
    ]
    failed = validate_trace_rows(
        [
            {
                **_trace(
                    claims=[_claim(text="甲公司2025年营业收入同比增长20%。", claim_type="DERIVED", calculation_id="calc-1", source_step_ids=["calculate_yoy"])],
                    calculations={"calc-1": {"calculation_id": "calc-1", "status": "FAILED", "inputs": []}},
                ),
                "reasoning_plan": reasoning_plan,
            }
        ],
        chunks=[CHUNK],
    )
    assert any(issue["code"] == "CALCULATION_NOT_SUCCESS" for issue in failed["issues"])

    valid = validate_trace_rows(
        [
            {
                **_trace(
                    claims=[_claim(text="甲公司2025年营业收入同比增长20%。", claim_type="DERIVED", calculation_id="calc-1", source_step_ids=["calculate_yoy"], evidence_ids=["c1", "c2"])],
                    citations=citations,
                    used=["c1", "c2"],
                    calculations={
                        "calc-1": {
                            "calculation_id": "calc-1",
                        "operation": "yoy",
                        "status": "SUCCESS",
                        "verified": True,
                        "input_calculation_ids": [],
                        "source_step_ids": [],
                        "evidence_ids": ["c1", "c2"],
                            "inputs": [
                                {"name": "prior", "value": "100", "unit": "元", "evidence_id": "c1"},
                                {"name": "current", "value": "120", "unit": "元", "evidence_id": "c2"},
                            ],
                            "result": {"value": "0.2", "unit": "", "formatted": "20.0000%"},
                        }
                    },
                ),
                "reasoning_plan": reasoning_plan,
            }
        ],
        chunks=chunks,
    )
    assert valid["status"] == "READY"

    mismatch = validate_trace_rows(
        [
            {
                **_trace(
                    claims=[_claim(text="甲公司2025年营业收入同比增长30%。", claim_type="DERIVED", calculation_id="calc-1", source_step_ids=["calculate_yoy"], evidence_ids=["c1", "c2"])],
                    citations=citations,
                    used=["c1", "c2"],
                    calculations={
                        "calc-1": {
                            "calculation_id": "calc-1",
                        "operation": "yoy",
                        "status": "SUCCESS",
                        "verified": True,
                        "input_calculation_ids": [],
                        "source_step_ids": [],
                        "evidence_ids": ["c1", "c2"],
                            "inputs": [
                                {"name": "prior", "value": "100", "unit": "元", "evidence_id": "c1"},
                                {"name": "current", "value": "120", "unit": "元", "evidence_id": "c2"},
                            ],
                            "result": {"value": "0.2", "unit": "", "formatted": "20.0000%"},
                        }
                    },
                ),
                "reasoning_plan": reasoning_plan,
            }
        ],
        chunks=chunks,
    )
    assert any(issue["code"] == "DERIVED_RESULT_MISMATCH" for issue in mismatch["issues"])


def test_calculation_lineage_rejects_calculation_ids_disguised_as_fact_ids():
    claim = _claim()
    row = {
        **_trace(claims=[claim]),
        "reasoning_plan": {
            "plan_id": "P-COMPARE",
            "steps": [
                {"step_id": "calculate_left", "kind": "CALCULATE"},
                {"step_id": "compare", "kind": "COMPARE"},
            ],
            "answer_requirement_ids": ["compare"],
        },
        "calculations": {
            "CALC-LEFT": {
                "calculation_id": "CALC-LEFT",
                "operation": "yoy",
                "status": "SUCCESS",
                "verified": True,
                "inputs": [
                    {"name": "prior", "value": "100", "evidence_id": "c1"},
                    {"name": "current", "value": "120", "evidence_id": "c1"},
                ],
                "input_calculation_ids": [],
                "source_step_ids": [],
                "evidence_ids": ["c1"],
                "result": {"value": "0.2", "formatted": "20.0000%", "unit": ""},
            },
            "CALC-COMPARE": {
                "calculation_id": "CALC-COMPARE",
                "operation": "difference",
                "status": "SUCCESS",
                "verified": True,
                "inputs": [
                    {"name": "left", "value": "0.2", "fact_id": "CALC-LEFT"},
                    {"name": "right", "value": "0.1", "fact_id": "missing-upstream"},
                ],
                "input_calculation_ids": [],
                "source_step_ids": [],
                "evidence_ids": [],
                "result": {"value": "0.1", "formatted": "0.1000", "unit": None},
            },
        },
    }

    report = validate_trace_rows([row], chunks=[CHUNK])
    codes = {issue["code"] for issue in report["issues"]}
    assert "CALCULATION_ID_AS_FACT_ID" in codes
    assert "CALCULATION_FACT_PROVENANCE_UNRESOLVED" in codes


def test_abstain_trace_is_valid_without_evidence():
    report = validate_trace_rows([_trace(claims=[], citations=[], used=[], abstained=True)], chunks=[])
    assert report["status"] == "READY"


def test_final_answer_cannot_reference_unverified_evidence():
    report = validate_trace_rows(
        [_trace(claims=[_claim(status="INSUFFICIENT")])],
        chunks=[CHUNK],
    )
    assert any(issue["code"] == "FINAL_UNVERIFIED_EVIDENCE" for issue in report["issues"])


def test_cli_returns_blocked_exit_code(tmp_path: Path):
    trace = tmp_path / "trace.jsonl"
    chunks = tmp_path / "chunks.jsonl"
    trace.write_text(json.dumps(_trace(claims=[_claim(evidence_ids=["missing"])], citations=[], used=["missing"])) + "\n", encoding="utf-8")
    chunks.write_text(json.dumps(CHUNK) + "\n", encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "evidence_integrity_gate.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--trace-path", str(trace), "--chunks-path", str(chunks)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "EVIDENCE_INTEGRITY_BLOCKED" in completed.stderr


def test_file_loader_reports_missing_asset(tmp_path: Path):
    report = validate_evidence_integrity(tmp_path / "trace.jsonl", tmp_path / "chunks.jsonl")
    assert report["status"] == "BLOCKED"
    assert report["summary"]["issue_count"] >= 2
