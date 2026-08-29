"""State adapter for the dependency coverage gate."""

from __future__ import annotations

from typing import Any

from src.agent.dependency_reasoning import assess_coverage
from src.agent.reasoning_plan import assess_reasoning_coverage


def _recoverable_single_step(state: dict[str, Any]) -> bool:
    steps = list((state.get("reasoning_plan") or {}).get("steps") or [])
    if not (
        len(steps) == 1
        and str((steps[0] or {}).get("kind") or "") in {"LOOKUP", "SEARCH"}
    ):
        return False

    # A retrieval miss can still use the existing bounded rewrite/fallback
    # path.  A structured ambiguity is different: the evidence contains
    # multiple incompatible values and synthesis must not be asked to guess
    # which financial coordinate the user meant.
    results = [
        result
        for result in (state.get("reasoning_step_results") or [])
        if isinstance(result, dict)
    ]
    error_type = str((results[-1] if results else {}).get("error_type") or "").upper()
    return error_type not in {
        "AMBIGUOUS_FACT",
        "AMBIGUOUS_OPERAND",
        "AMBIGUOUS_PERIOD_BASIS",
        "MISSING_OPERAND_COORDINATES",
    }

def dependency_gate(state: dict[str, Any]) -> dict[str, Any]:
    if state.get("reasoning_plan"):
        plan_steps = list((state.get("reasoning_plan") or {}).get("steps") or [])
        report = assess_reasoning_coverage(
            state["reasoning_plan"],
            state.get("reasoning_step_results") or [],
        )
        state["reasoning_coverage"] = report
        # Multi-step synthesis consumes one merged evidence view, while each
        # StepResult retains its own provenance in reasoning_step_artifacts.
        if len(plan_steps) > 1:
            rows = [
                row
                for row in (state.get("evidence_pool") or {}).values()
                if isinstance(row, dict)
            ]
            rows.sort(
                key=lambda row: float(row.get("rerank_score") or row.get("score") or 0.0),
                reverse=True,
            )
            state["last_retrieval_result"] = {
                "query_mode": "reasoning_plan",
                "numeric_query": bool(state.get("numeric_query")),
                "dense_rows": [],
                "bm25_rows": [],
                "hybrid_rows": rows,
                "rerank_rows": rows,
                "timings": {},
            }
    else:
        required = state.get("required_sub_questions") or state.get("sub_questions") or []
        report = assess_coverage(
            required,
            state.get("sub_question_results") or [],
            edges=state.get("dependency_edges"),
            evidence_pool=state.get("evidence_pool"),
        )
    state["dependency_coverage"] = report
    state["coverage_report"] = report
    state["deterministic_conclusion_allowed"] = report["deterministic_conclusion_allowed"]
    state["partial_answer"] = report["partial"]
    if report["abstain"] and not _recoverable_single_step(state):
        reason = (
            "abstain_core_dependency_missing"
            if state.get("reasoning_plan") and report.get("missing_answer_requirements")
            else "abstain_reasoning_incomplete"
        )
        state["termination_reason"] = state.get("termination_reason") or reason
    elif report["partial"]:
        # This is an enforcement signal consumed by finalize; preserving a
        # previous mode here would let synthesis render a definitive answer.
        state["answer_mode"] = "partial"
        state["missing_information"] = ",".join(report["missing"])
    return state


def make_dependency_gate(*_args: Any, **_kwargs: Any):
    return dependency_gate
