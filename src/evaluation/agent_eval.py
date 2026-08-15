"""Agent evaluation (checklist v3.0 §P3): quality metrics, baseline comparison, P3 gate.

Quality metrics reuse the existing answer-eval hit functions unchanged; trace
rows produced by :mod:`src.evaluation.trajectory_eval` carry the same
final-answer fields those functions expect.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from src.evaluation.answer_eval import (
    _materialize_answer_eval_seed,
    compute_answer_semantic_hit,
    compute_citation_doc_hit,
    compute_citation_span_hit,
    compute_support_hit,
)
from src.evaluation.benchmark_assets import build_doc_manifest
from src.utils.io import write_json, write_jsonl

AGENT_SEED_PRESERVE_FIELDS = ("category", "must_recover", "expected_first_failure")

# P3 hard gate thresholds (checklist §P3 Gate). Tunable module-level constants.
P3_GATE_QUALITY_GAIN = 0.05
P3_GATE_RECOVERY_GAIN = 0.10
P3_GATE_REGRESSION_TOLERANCE = 0.05

_MISSING_RESULT_ROW: dict[str, Any] = {
    "failed": True,
    "abstained": False,
    "final_answer": "",
    "evidence_summary": "",
    "fact_subtype": "",
    "citations": [],
    "support_validation": {},
    "matched_numeric_tokens": [],
}


def materialize_agent_eval_set(
    *,
    seed_path: Path,
    chunks: list[dict[str, Any]],
    output_path: Path,
    report_output_path: Path,
) -> list[dict[str, Any]]:
    """Resolve the agent benchmark seed against the corpus, preserving category metadata."""
    chunk_lookup = {chunk["chunk_id"]: chunk for chunk in chunks}
    manifest = build_doc_manifest(chunks)
    docs_by_key = {row["doc_key"]: row for row in manifest}
    resolved_rows, missing_rows = _materialize_answer_eval_seed(
        seed_path=seed_path,
        chunks=chunks,
        chunk_lookup=chunk_lookup,
        docs_by_key=docs_by_key,
        preserve_fields=AGENT_SEED_PRESERVE_FIELDS,
    )
    write_jsonl(output_path, resolved_rows)
    write_json(
        report_output_path,
        {
            "seed_path": str(seed_path),
            "row_count": len(resolved_rows) + len(missing_rows),
            "resolved_row_count": len(resolved_rows),
            "missing_row_count": len(missing_rows),
            "category_distribution": dict(sorted(Counter(row.get("category", "other") for row in resolved_rows).items())),
            "intent_distribution": dict(
                sorted(Counter(row.get("intent", row.get("question_type", "fact")) for row in resolved_rows).items())
            ),
            "missing_rows": missing_rows,
        },
    )
    if missing_rows:
        raise ValueError(f"failed to resolve {len(missing_rows)} agent eval rows; see {report_output_path}")
    return resolved_rows


def _row_hits(eval_row: dict[str, Any], result_row: dict[str, Any], chunk_lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    answer_semantic_hit = compute_answer_semantic_hit(eval_row, result_row, chunk_lookup)
    citation_doc_hit = compute_citation_doc_hit(eval_row, result_row)
    citation_span_hit = compute_citation_span_hit(eval_row, result_row, chunk_lookup)
    support_hit = compute_support_hit(
        eval_row,
        result_row,
        answer_semantic_hit=answer_semantic_hit,
        citation_doc_hit=citation_doc_hit,
    )
    return {
        "answer_semantic_hit": answer_semantic_hit,
        "citation_doc_hit": citation_doc_hit,
        "citation_span_hit": citation_span_hit,
        "support_hit": support_hit,
        "abstained": bool(result_row.get("abstained", False)),
        "failed": bool(result_row.get("failed", False)),
    }


def evaluate_agent_quality(
    eval_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    chunk_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Quality metrics (checklist §P3): answer/citation/abstain accuracy and recovery success."""
    results_by_id = {row["question_id"]: row for row in result_rows}
    answerable_rows = [row for row in eval_rows if not row.get("must_abstain")]
    abstain_rows = [row for row in eval_rows if row.get("must_abstain")]
    hits_by_id: dict[str, dict[str, Any]] = {}
    failed_query_count = 0
    for eval_row in eval_rows:
        result = results_by_id.get(eval_row["question_id"])
        if result is None:
            result = dict(_MISSING_RESULT_ROW)
            failed_query_count += 1
        elif result.get("failed"):
            failed_query_count += 1
        hits_by_id[eval_row["question_id"]] = _row_hits(eval_row, result, chunk_lookup)

    answer_denominator = max(len(answerable_rows), 1)
    summary: dict[str, Any] = {
        "query_count": len(eval_rows),
        "answerable_query_count": len(answerable_rows),
        "must_abstain_query_count": len(abstain_rows),
        "Failed Query Count": failed_query_count,
        "Answer Accuracy": round(
            sum(1 for row in answerable_rows if hits_by_id[row["question_id"]]["answer_semantic_hit"]) / answer_denominator, 4
        ),
        "Citation Accuracy": round(
            sum(1 for row in answerable_rows if hits_by_id[row["question_id"]]["citation_doc_hit"]) / answer_denominator, 4
        ),
        "Citation Span Accuracy": round(
            sum(1 for row in answerable_rows if hits_by_id[row["question_id"]]["citation_span_hit"]) / answer_denominator, 4
        ),
        "Support Hit Rate": round(
            sum(1 for row in answerable_rows if hits_by_id[row["question_id"]]["support_hit"]) / answer_denominator, 4
        ),
    }
    predicted_abstains = sum(1 for row in eval_rows if hits_by_id[row["question_id"]]["abstained"])
    correct_abstains = sum(1 for row in abstain_rows if hits_by_id[row["question_id"]]["abstained"])
    if abstain_rows:
        summary["Abstain Accuracy"] = round(correct_abstains / len(abstain_rows), 4)
    else:
        summary["Abstain Accuracy"] = None
    summary["Abstain Precision"] = round(correct_abstains / max(predicted_abstains, 1), 4)

    recovery_rows = [row for row in answerable_rows if row.get("must_recover")]
    if recovery_rows:
        recovered = sum(
            1
            for row in recovery_rows
            if not hits_by_id[row["question_id"]]["abstained"]
            and hits_by_id[row["question_id"]]["answer_semantic_hit"]
            and hits_by_id[row["question_id"]]["citation_doc_hit"]
        )
        triggered = sum(1 for row in recovery_rows if int(results_by_id.get(row["question_id"], {}).get("rewrite_count", 0) or 0) > 0)
        summary["Recovery Success Rate"] = round(recovered / len(recovery_rows), 4)
        summary["Recovery Trigger Rate"] = round(triggered / len(recovery_rows), 4)
    else:
        summary["Recovery Success Rate"] = None
        summary["Recovery Trigger Rate"] = None
    return summary


def build_baseline_agentic_comparison(
    eval_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    agentic_rows: list[dict[str, Any]],
    chunk_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Baseline RAG vs Agentic RAG comparison (checklist §P3 required comparison)."""
    baseline_by_id = {row["question_id"]: row for row in baseline_rows}
    agentic_by_id = {row["question_id"]: row for row in agentic_rows}
    per_question: list[dict[str, Any]] = []
    for eval_row in eval_rows:
        question_id = eval_row["question_id"]
        baseline_result = baseline_by_id.get(question_id)
        if baseline_result is None:
            baseline_result = dict(_MISSING_RESULT_ROW)
        agentic_result = agentic_by_id.get(question_id)
        if agentic_result is None:
            agentic_result = dict(_MISSING_RESULT_ROW)
        per_question.append(
            {
                "question_id": question_id,
                "category": eval_row.get("category", ""),
                "intent": eval_row.get("intent", eval_row.get("question_type", "fact")),
                "baseline": _row_hits(eval_row, baseline_result, chunk_lookup),
                "agentic": _row_hits(eval_row, agentic_result, chunk_lookup),
            }
        )
    baseline_summary = evaluate_agent_quality(eval_rows, baseline_rows, chunk_lookup)
    agentic_summary = evaluate_agent_quality(eval_rows, agentic_rows, chunk_lookup)
    deltas: dict[str, Any] = {}
    for key in ("Answer Accuracy", "Citation Accuracy", "Citation Span Accuracy", "Abstain Accuracy", "Recovery Success Rate"):
        baseline_value = baseline_summary.get(key)
        agentic_value = agentic_summary.get(key)
        deltas[key] = None if baseline_value is None or agentic_value is None else round(agentic_value - baseline_value, 4)
    return {
        "question_count": len(eval_rows),
        "per_question": per_question,
        "baseline": baseline_summary,
        "agentic": agentic_summary,
        "deltas": deltas,
    }


def evaluate_p3_gate(*, baseline: dict[str, Any], agentic: dict[str, Any]) -> dict[str, Any]:
    """P3 hard gate: provable quality/recovery gain without meaningful regression.

    PASS iff (quality gain OR recovery gain) AND no regression beyond tolerance.
    """
    baseline_answer = float(baseline.get("Answer Accuracy") or 0.0)
    agentic_answer = float(agentic.get("Answer Accuracy") or 0.0)
    baseline_citation = float(baseline.get("Citation Accuracy") or 0.0)
    agentic_citation = float(agentic.get("Citation Accuracy") or 0.0)
    baseline_recovery = float(baseline.get("Recovery Success Rate") or 0.0)
    agentic_recovery = float(agentic.get("Recovery Success Rate") or 0.0)

    gain_quality = (agentic_answer - baseline_answer) >= P3_GATE_QUALITY_GAIN or (
        agentic_citation - baseline_citation
    ) >= P3_GATE_QUALITY_GAIN
    gain_recovery = (agentic_recovery - baseline_recovery) >= P3_GATE_RECOVERY_GAIN
    no_regression = (
        agentic_answer >= baseline_answer - P3_GATE_REGRESSION_TOLERANCE
        and agentic_citation >= baseline_citation - P3_GATE_REGRESSION_TOLERANCE
    )
    baseline_abstain = baseline.get("Abstain Accuracy")
    agentic_abstain = agentic.get("Abstain Accuracy")
    if baseline_abstain is not None and agentic_abstain is not None:
        no_regression = no_regression and float(agentic_abstain) >= float(baseline_abstain) - P3_GATE_REGRESSION_TOLERANCE

    baseline_recovery_raw = baseline.get("Recovery Success Rate")
    agentic_recovery_raw = agentic.get("Recovery Success Rate")
    return {
        "gate_passed": (gain_quality or gain_recovery) and no_regression,
        "gain_quality": gain_quality,
        "gain_recovery": gain_recovery,
        "no_regression": no_regression,
        "deltas": {
            "Answer Accuracy": round(agentic_answer - baseline_answer, 4),
            "Citation Accuracy": round(agentic_citation - baseline_citation, 4),
            "Abstain Accuracy": None
            if baseline_abstain is None or agentic_abstain is None
            else round(float(agentic_abstain) - float(baseline_abstain), 4),
            "Recovery Success Rate": None
            if baseline_recovery_raw is None or agentic_recovery_raw is None
            else round(agentic_recovery - baseline_recovery, 4),
        },
        "thresholds": {
            "quality_gain": P3_GATE_QUALITY_GAIN,
            "recovery_gain": P3_GATE_RECOVERY_GAIN,
            "regression_tolerance": P3_GATE_REGRESSION_TOLERANCE,
        },
    }
