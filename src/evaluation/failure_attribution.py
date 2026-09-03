"""Failure attribution for agent trajectories.

This module deliberately works on plain dictionaries so it can consume both
runtime state and materialized trajectory rows.  It exposes one stable
taxonomy and records the first/root cause, downstream failures, final failure,
recovery and attributable cost without changing the agent runtime contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from src.agent.trajectory import canonicalize_trajectory_events
from src.evaluation.claim_eval import evaluate_claim_verification


class FailureType(str, Enum):
    # Fine-grained categories from the trajectory contract.  The original
    # runtime labels below remain valid aliases for backwards-compatible
    # summaries; these names let reviewed evals attribute failures before they
    # collapse into a generic retrieval/validation bucket.
    QUERY_ANALYSIS_ERROR = "query_analysis_error"
    ROUTING_ERROR = "routing_error"
    DECOMPOSITION_ERROR = "decomposition_error"
    STRUCTURED_LOOKUP_MISS = "structured_lookup_miss"
    RETRIEVAL_MISS = "retrieval_miss"
    RERANK_ERROR = "rerank_error"
    EVIDENCE_SELECTION_ERROR = "evidence_selection_error"
    TOOL_SELECTION_ERROR = "tool_selection_error"
    SYNTHESIS_ERROR = "synthesis_error"
    CITATION_ERROR = "citation_error"
    CLAIM_EXTRACTION_ERROR = "claim_extraction_error"
    VERIFICATION_FALSE_POSITIVE = "verification_false_positive"
    VERIFICATION_FALSE_NEGATIVE = "verification_false_negative"
    ABSTAIN_FALSE_POSITIVE = "abstain_false_positive"
    ABSTAIN_FALSE_NEGATIVE = "abstain_false_negative"
    NO_EVIDENCE = "no_evidence"
    NUMERIC_MISSING = "numeric_missing"
    SOURCE_DIVERSITY_MISSING = "source_diversity_missing"
    UNSUPPORTED = "unsupported"
    RETRIEVAL_ERROR = "retrieval_error"
    TOOL_EXECUTION_FAILED = "tool_execution_failed"
    CALCULATION_FAILED = "calculation_failed"
    VALIDATION_FAILED = "validation_failed"
    PARSE_ERROR = "parse_error"
    LLM_ERROR = "llm_error"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NO_IMPROVEMENT = "no_improvement"
    UNRESOLVED = "unresolved"
    UNKNOWN = "unknown"


FAILURE_TAXONOMY = {item.value for item in FailureType}

_OUTCOME_SYMPTOMS = {
    FailureType.SYNTHESIS_ERROR.value,
    FailureType.CITATION_ERROR.value,
    FailureType.VERIFICATION_FALSE_POSITIVE.value,
    FailureType.VERIFICATION_FALSE_NEGATIVE.value,
    FailureType.ABSTAIN_FALSE_POSITIVE.value,
    FailureType.ABSTAIN_FALSE_NEGATIVE.value,
    FailureType.UNSUPPORTED.value,
    FailureType.VALIDATION_FAILED.value,
}


@dataclass(frozen=True)
class FailureEvent:
    """Normalized event used internally and safe to serialize with ``asdict``."""

    index: int
    failure_type: str
    node: str = ""
    action: str = ""
    step: int | None = None
    status: str = ""
    message: str = ""
    latency_ms: float = 0.0
    tokens: int = 0
    retries: int = 0
    event_id: str = ""
    dependencies: tuple[str, ...] = ()
    recovery_of: tuple[str, ...] = ()
    lineage_status: str = "unresolved"
    origin: str = ""
    outcome_dimensions: tuple[str, ...] = ()
    gold_id: str = ""
    predicted_id: str = ""
    raw: Mapping[str, Any] | None = None


_ALIASES: dict[str, FailureType] = {
    "query_analysis_error": FailureType.QUERY_ANALYSIS_ERROR,
    "analysis_error": FailureType.QUERY_ANALYSIS_ERROR,
    "routing_error": FailureType.ROUTING_ERROR,
    "decomposition_error": FailureType.DECOMPOSITION_ERROR,
    "decompose_error": FailureType.DECOMPOSITION_ERROR,
    "structured_lookup_miss": FailureType.STRUCTURED_LOOKUP_MISS,
    "structured_miss": FailureType.STRUCTURED_LOOKUP_MISS,
    "retrieval_miss": FailureType.RETRIEVAL_MISS,
    "rerank_error": FailureType.RERANK_ERROR,
    "evidence_selection_error": FailureType.EVIDENCE_SELECTION_ERROR,
    "selection_error": FailureType.EVIDENCE_SELECTION_ERROR,
    "tool_selection_error": FailureType.TOOL_SELECTION_ERROR,
    "synthesis_error": FailureType.SYNTHESIS_ERROR,
    "citation_error": FailureType.CITATION_ERROR,
    "claim_extraction_error": FailureType.CLAIM_EXTRACTION_ERROR,
    "verification_false_positive": FailureType.VERIFICATION_FALSE_POSITIVE,
    "verification_false_negative": FailureType.VERIFICATION_FALSE_NEGATIVE,
    "abstain_false_positive": FailureType.ABSTAIN_FALSE_POSITIVE,
    "abstain_false_negative": FailureType.ABSTAIN_FALSE_NEGATIVE,
    "no_evidence": FailureType.NO_EVIDENCE,
    "evidence_missing": FailureType.NO_EVIDENCE,
    "numeric_missing": FailureType.NUMERIC_MISSING,
    "numeric_constraints_incomplete": FailureType.NUMERIC_MISSING,
    "source_diversity_missing": FailureType.SOURCE_DIVERSITY_MISSING,
    "unsupported": FailureType.UNSUPPORTED,
    "unsupported_answer": FailureType.UNSUPPORTED,
    "retrieval_error": FailureType.RETRIEVAL_ERROR,
    "retrieval_failed": FailureType.RETRIEVAL_ERROR,
    "tool_execution_failed": FailureType.TOOL_EXECUTION_FAILED,
    "tool_failed": FailureType.TOOL_EXECUTION_FAILED,
    "calculation_failed": FailureType.CALCULATION_FAILED,
    "calculation_error": FailureType.CALCULATION_FAILED,
    "validation_failed": FailureType.VALIDATION_FAILED,
    "support_validation_failed": FailureType.VALIDATION_FAILED,
    "parse_error": FailureType.PARSE_ERROR,
    "json_error": FailureType.PARSE_ERROR,
    "llm_error": FailureType.LLM_ERROR,
    "provider_error": FailureType.LLM_ERROR,
    "budget": FailureType.BUDGET_EXHAUSTED,
    "budget_exhausted": FailureType.BUDGET_EXHAUSTED,
    "max_llm_calls": FailureType.BUDGET_EXHAUSTED,
    "max_tool_calls": FailureType.BUDGET_EXHAUSTED,
    "no_improvement": FailureType.NO_IMPROVEMENT,
    "duplicate_tool_call": FailureType.NO_IMPROVEMENT,
    "unresolved": FailureType.UNRESOLVED,
}


def normalize_failure_type(value: Any, *, default: str = FailureType.UNKNOWN.value) -> str:
    """Return a canonical taxonomy value while preserving unknown labels."""
    text = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    if text in _ALIASES:
        return _ALIASES[text].value
    if text in FAILURE_TAXONOMY:
        return text
    return default


def _event_failure_type(event: Mapping[str, Any]) -> str | None:
    status = str(event.get("status") or "").upper()
    explicit = event.get("failure_type") or event.get("error_type") or event.get("reason")
    normalized = normalize_failure_type(explicit, default="")
    if normalized:
        return normalized
    action = str(event.get("action") or "").lower()
    node = str(event.get("node") or "").lower()
    if status in {"FAILED", "ERROR", "EXCEPTION"}:
        if "retriev" in action or "retriev" in node:
            return FailureType.RETRIEVAL_ERROR.value
        if "calculat" in action or "calculat" in node:
            return FailureType.CALCULATION_FAILED.value
        if "tool" in action or "tool" in node:
            return FailureType.TOOL_EXECUTION_FAILED.value
        if "llm" in action or "llm" in node:
            return FailureType.LLM_ERROR.value
        return FailureType.UNKNOWN.value
    if status in {"SKIPPED", "BUDGET_EXHAUSTED"} or "budget_exhausted" in str(explicit or ""):
        return FailureType.BUDGET_EXHAUSTED.value
    if status in {"NO_NEW_EVIDENCE", "NO_IMPROVEMENT"}:
        return FailureType.NO_IMPROVEMENT.value
    return None


def _gold_outcome_events(
    source: Mapping[str, Any],
    gold: Mapping[str, Any] | None,
    *,
    start_step: int,
    additional_outcomes: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(gold, Mapping):
        return []
    events: list[dict[str, Any]] = []
    predicted_claims = source.get("claims")
    gold_claims = gold.get("gold_claims")
    if isinstance(gold_claims, list):
        comparison = evaluate_claim_verification(
            predicted_claims if isinstance(predicted_claims, list) else [],
            gold_claims,
        )
        for item in comparison["comparisons"]:
            gold_status = item.get("gold_status")
            predicted_status = item.get("predicted_status")
            failure_type = None
            if predicted_status == "ENTAILED" and gold_status != "ENTAILED":
                failure_type = FailureType.VERIFICATION_FALSE_POSITIVE.value
            elif gold_status == "ENTAILED" and predicted_status != "ENTAILED":
                failure_type = FailureType.VERIFICATION_FALSE_NEGATIVE.value
            if failure_type:
                events.append(
                    {
                        "event_id": f"reviewed-gold-{start_step + len(events):04d}",
                        # Reviewed outcomes are independent observations.  They
                        # are not a causal continuation of runtime order or of
                        # one another unless a producer supplies real lineage.
                        "dependencies": [],
                        "recovery_of": [],
                        "step": start_step + len(events),
                        "node": "reviewed_gold_claim_verification",
                        "status": "FAILED",
                        "error_type": failure_type,
                        "claim_id": item.get("gold_claim_id") or item.get("predicted_claim_id"),
                        "origin": "reviewed_gold",
                        "outcome_dimensions": ["claim.verification_status"],
                        "gold_id": item.get("gold_claim_id"),
                        "predicted_id": item.get("predicted_claim_id"),
                    }
                )
            if item.get("status_match") is True and item.get("evidence_match") is False:
                events.append(
                    {
                        "event_id": f"reviewed-gold-{start_step + len(events):04d}",
                        "dependencies": [],
                        "recovery_of": [],
                        "step": start_step + len(events),
                        "node": "reviewed_gold_claim_evidence",
                        "status": "FAILED",
                        "error_type": FailureType.CITATION_ERROR.value,
                        "claim_id": item.get("gold_claim_id")
                        or item.get("predicted_claim_id"),
                        "origin": "reviewed_gold",
                        "outcome_dimensions": ["claim.evidence"],
                        "gold_id": item.get("gold_claim_id"),
                        "predicted_id": item.get("predicted_claim_id"),
                        "output_summary": (
                            "claim status matches reviewed gold but evidence IDs differ"
                        ),
                    }
                )
    if "must_abstain" in gold:
        expected_abstain = bool(gold.get("must_abstain"))
        actual_abstain = bool(source.get("abstained", False))
        if actual_abstain != expected_abstain:
            failure_type = (
                FailureType.ABSTAIN_FALSE_POSITIVE.value
                if actual_abstain
                else FailureType.ABSTAIN_FALSE_NEGATIVE.value
            )
            events.append(
                {
                    "event_id": f"reviewed-gold-{start_step + len(events):04d}",
                    "dependencies": [],
                    "recovery_of": [],
                    "step": start_step + len(events),
                    "node": "reviewed_gold_abstention",
                    "status": "FAILED",
                    "error_type": failure_type,
                    "origin": "reviewed_gold",
                    "outcome_dimensions": ["abstention"],
                }
            )
    for outcome in additional_outcomes or []:
        failure_type = normalize_failure_type(outcome.get("failure_type"), default="")
        dimensions = [
            str(value)
            for value in outcome.get("outcome_dimensions") or []
            if str(value).strip()
        ]
        if not failure_type or not dimensions:
            continue
        events.append(
            {
                "event_id": f"reviewed-gold-{start_step + len(events):04d}",
                "dependencies": [],
                "recovery_of": [],
                "step": start_step + len(events),
                "node": str(outcome.get("node") or "reviewed_gold_outcome"),
                "status": "FAILED",
                "error_type": failure_type,
                "origin": "reviewed_gold",
                "outcome_dimensions": dimensions,
                "gold_id": outcome.get("gold_id"),
                "predicted_id": outcome.get("predicted_id"),
                "output_summary": str(outcome.get("message") or ""),
            }
        )
    return events


def extract_failure_events(
    source: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
    *,
    gold: Mapping[str, Any] | None = None,
    gold_outcomes: Sequence[Mapping[str, Any]] | None = None,
) -> list[FailureEvent]:
    """Extract normalized failure events from state/trace rows or raw events."""
    if source is None:
        return []
    if isinstance(source, Mapping):
        events = canonicalize_trajectory_events(source)
        next_derived_index = len(events) + 1

        def append_derived_failure(**values: Any) -> None:
            nonlocal next_derived_index
            existing_ids = {str(event.get("event_id") or "") for event in events}
            event_id = f"derived-failure-{next_derived_index:04d}"
            while event_id in existing_ids:
                next_derived_index += 1
                event_id = f"derived-failure-{next_derived_index:04d}"
            events.append(
                {
                    "event_id": event_id,
                    # A state-level grading/termination flag is an observed
                    # failure, not proof that the preceding trace event caused
                    # it.  Keep it independent unless the source supplied an
                    # explicit trajectory edge.
                    "dependencies": [],
                    "recovery_of": [],
                    "lineage_status": "resolved",
                    "origin": "derived_state",
                    **values,
                }
            )
            next_derived_index += 1

        termination = str(source.get("termination_reason") or "")
        if termination and termination not in {"completed", ""}:
            append_derived_failure(step=source.get("step_count", 0), node="termination", status="FAILED", error_type=termination)
        if source.get("failed"):
            append_derived_failure(step=source.get("step_count", 0), node="agent", status="FAILED", error_type=source.get("error_type") or "unknown", output_summary=source.get("error_message", ""))
        # Some grading failures are state flags rather than trajectory events.
        if source.get("no_improvement") and not any(str(e.get("error_type") or "") == "no_improvement" for e in events if isinstance(e, Mapping)):
            append_derived_failure(step=source.get("step_count", 0), node="grading", status="NO_IMPROVEMENT", error_type="no_improvement")
        missing = str(source.get("missing_information") or "").strip()
        if missing and not any(str(e.get("error_type") or "") in {"no_evidence", "numeric_missing", "source_diversity_missing"} for e in events if isinstance(e, Mapping)):
            label = "numeric_missing" if "numeric" in missing.lower() or "数值" in missing else "no_evidence"
            append_derived_failure(step=source.get("step_count", 0), node="grading", status="FAILED", error_type=label, reason=missing)
        support = source.get("support_validation") or {}
        if isinstance(support, Mapping) and support.get("supported") is False:
            append_derived_failure(step=source.get("step_count", 0), node="validate_answer", status="FAILED", error_type="validation_failed")
        resolved_gold = gold
        if resolved_gold is None:
            embedded = source.get("gold") or source.get("gold_case")
            resolved_gold = embedded if isinstance(embedded, Mapping) else None
        existing_steps = [int(event["step"]) for event in events if isinstance(event, Mapping) and str(event.get("step", "")).isdigit()]
        events.extend(
            _gold_outcome_events(
                source,
                resolved_gold,
                start_step=max(existing_steps, default=-1) + 1,
                additional_outcomes=gold_outcomes,
            )
        )
    else:
        events = canonicalize_trajectory_events(source)
    events = canonicalize_trajectory_events(events)
    result: list[FailureEvent] = []
    for index, raw in enumerate(events):
        if not isinstance(raw, Mapping):
            continue
        failure_type = _event_failure_type(raw)
        if failure_type is None:
            continue
        try:
            latency = float(raw.get("latency_ms", 0.0) or 0.0)
        except (TypeError, ValueError):
            latency = 0.0
        budget_usage = raw.get("budget_usage") if isinstance(raw.get("budget_usage"), Mapping) else {}
        try:
            tokens = int(budget_usage.get("tokens", raw.get("tokens", 0)) or 0)
        except (TypeError, ValueError):
            tokens = 0
        try:
            retries = int(budget_usage.get("retries", raw.get("retries", 0)) or 0)
        except (TypeError, ValueError):
            retries = 0
        result.append(
            FailureEvent(
                index=index,
                failure_type=failure_type,
                node=str(raw.get("node") or ""),
                action=str(raw.get("action") or ""),
                step=int(raw["step"]) if str(raw.get("step", "")).isdigit() else None,
                status=str(raw.get("status") or ""),
                message=str(raw.get("error_message") or raw.get("output_summary") or raw.get("reason") or ""),
                latency_ms=latency,
                tokens=tokens,
                retries=retries,
                event_id=str(raw.get("event_id") or ""),
                dependencies=tuple(str(item) for item in raw.get("dependencies") or []),
                recovery_of=tuple(str(item) for item in raw.get("recovery_of") or []),
                lineage_status=str(raw.get("lineage_status") or "unresolved"),
                origin=str(raw.get("origin") or ""),
                outcome_dimensions=tuple(
                    str(item) for item in raw.get("outcome_dimensions") or []
                ),
                gold_id=str(raw.get("gold_id") or ""),
                predicted_id=str(raw.get("predicted_id") or ""),
                raw=raw,
            )
        )
    return result


def _event_order(event: FailureEvent) -> tuple[int, int]:
    return (event.step if event.step is not None else event.index, event.index)


def attribute_failure(
    source: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
    *,
    gold: Mapping[str, Any] | None = None,
    gold_outcomes: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Attribute failures through explicit lineage; never invent missing causality."""
    events = extract_failure_events(source, gold=gold, gold_outcomes=gold_outcomes)
    ordered_events = sorted(events, key=_event_order)
    first = ordered_events[0] if ordered_events else None
    final = ordered_events[-1] if ordered_events else None

    raw_events = canonicalize_trajectory_events(source)
    known_raw_ids = {str(item.get("event_id") or "") for item in raw_events}
    for event in ordered_events:
        if event.event_id not in known_raw_ids and isinstance(event.raw, Mapping):
            raw_events.append(dict(event.raw))
            known_raw_ids.add(event.event_id)
    raw_events = canonicalize_trajectory_events(raw_events)
    raw_by_id = {str(item["event_id"]): item for item in raw_events}
    failure_by_id = {event.event_id: event for event in ordered_events if event.event_id}
    recovered_ids = {
        event_id
        for item in raw_events
        if str(item.get("status") or "").upper() in {"SUCCESS", "COMPLETED", "ENTAILED"}
        and item.get("lineage_status") == "resolved"
        for event_id in item.get("recovery_of") or []
        if event_id in failure_by_id
    }
    unresolved_failures = [event for event in ordered_events if event.event_id not in recovered_ids]
    root_candidate = next(
        (event for event in unresolved_failures if event.failure_type not in _OUTCOME_SYMPTOMS),
        unresolved_failures[0] if unresolved_failures else None,
    )
    root: FailureEvent | None = None
    roots: list[FailureEvent] = []
    ancestry_by_failure_id: dict[str, set[str]] = {}
    root_resolution = "resolved"
    if unresolved_failures:
        lineage_valid = True

        def ancestors_for(target_id: str) -> tuple[set[str], bool]:
            ancestor_ids: set[str] = set()
            visited_ids: set[str] = set()
            visiting_ids: set[str] = set()
            valid = True

            def visit(event_id: str) -> None:
                nonlocal valid
                if event_id in visiting_ids:
                    valid = False
                    return
                if event_id in visited_ids:
                    return
                visiting_ids.add(event_id)
                ancestor_ids.add(event_id)
                raw = raw_by_id.get(event_id)
                if raw is None or raw.get("lineage_status") != "resolved":
                    valid = False
                else:
                    for dependency in raw.get("dependencies") or []:
                        dependency_id = str(dependency)
                        if dependency_id not in raw_by_id:
                            valid = False
                        else:
                            visit(dependency_id)
                visiting_ids.remove(event_id)
                visited_ids.add(event_id)

            visit(target_id)
            return ancestor_ids, valid

        roots_by_id: dict[str, FailureEvent] = {}
        for failure in unresolved_failures:
            ancestor_ids, valid = ancestors_for(failure.event_id)
            ancestry_by_failure_id[failure.event_id] = ancestor_ids
            lineage_valid = lineage_valid and valid
            causal_failures = [
                event
                for event in unresolved_failures
                if event.event_id in ancestor_ids
            ]
            if causal_failures:
                causal_root = min(causal_failures, key=_event_order)
                roots_by_id[causal_root.event_id] = causal_root
            else:
                lineage_valid = False

        if lineage_valid and roots_by_id:
            roots = sorted(roots_by_id.values(), key=_event_order)
            root = roots[0]
            if len(roots) > 1:
                root_resolution = "multiple_independent"
        else:
            root_resolution = "unresolved"
    elif ordered_events:
        root_resolution = "recovered"

    root_output = (
        root.failure_type
        if root is not None
        else FailureType.UNRESOLVED.value if unresolved_failures else None
    )
    anchor = root or root_candidate
    downstream_events = [
        event
        for event in unresolved_failures
        if root is not None
        and event.event_id != root.event_id
        and root.event_id in ancestry_by_failure_id.get(event.event_id, set())
    ]
    downstream = [event.failure_type for event in downstream_events]
    downstream_symptoms = [
        event.failure_type
        for event in downstream_events
        if event.failure_type in _OUTCOME_SYMPTOMS or event.origin == "reviewed_gold"
    ]
    downstream_ids = {event.event_id for event in downstream_events}
    independent_events = [
        event
        for event in unresolved_failures
        if event.event_id != (root.event_id if root else "")
        and event.event_id not in downstream_ids
    ]

    def _raw_order(item: Mapping[str, Any], index: int) -> tuple[int, int]:
        return (
            int(item["step"]) if str(item.get("step", "")).isdigit() else index,
            index,
        )

    def _event_cost(items: Sequence[Any]) -> dict[str, Any]:
        tokens = 0
        latency = 0.0
        retries = 0
        for item in items:
            if not isinstance(item, Mapping):
                continue
            budget = item.get("budget_usage") if isinstance(item.get("budget_usage"), Mapping) else {}
            try:
                tokens += int(budget.get("tokens", item.get("tokens", 0)) or 0)
            except (TypeError, ValueError):
                pass
            try:
                latency += float(budget.get("latency_ms", item.get("latency_ms", 0.0)) or 0.0)
            except (TypeError, ValueError):
                pass
            try:
                retries += int(budget.get("retries", item.get("retries", 0)) or 0)
            except (TypeError, ValueError):
                pass
        return {"tokens": tokens, "latency_ms": round(latency, 2), "retries": retries}

    def _cost_through(event: FailureEvent | None) -> dict[str, Any]:
        if event is None:
            return {"tokens": 0, "latency_ms": 0.0, "retries": 0}
        return _event_cost(
            [
                item
                for index, item in enumerate(raw_events)
                if _raw_order(item, index) <= _event_order(event)
            ]
        )

    cost_before_first_failure = _cost_through(first)
    cost_before_root_cause = _cost_through(anchor)
    state = source if isinstance(source, Mapping) else {}
    state_budget = state.get("budget_usage") if isinstance(state.get("budget_usage"), Mapping) else {}
    raw_cost = _event_cost(raw_events)
    cost_tokens = int(state_budget.get("tokens", state.get("total_tokens", raw_cost["tokens"])) or 0)
    cost_latency = float(
        state_budget.get(
            "latency_ms",
            state.get("end_to_end_latency_ms", raw_cost["latency_ms"]),
        )
        or 0.0
    )
    cost_retries = int(state_budget.get("retries", state.get("api_retries_total", raw_cost["retries"])) or 0)
    recovery_events = [
        item
        for item in raw_events
        if any(event_id in recovered_ids for event_id in item.get("recovery_of") or [])
    ]
    first_recovery = recovery_events[0] if recovery_events else None
    recovered = bool(ordered_events) and not unresolved_failures

    def _failure_descriptor(event: FailureEvent) -> dict[str, Any]:
        return {
            "event_id": event.event_id,
            "failure_type": event.failure_type,
            "node": event.node,
            "step": event.step,
            "origin": event.origin or None,
            "outcome_dimensions": list(event.outcome_dimensions),
            "gold_id": event.gold_id or None,
            "predicted_id": event.predicted_id or None,
        }

    result = {
        "failure_count": len(ordered_events),
        "has_failure": bool(ordered_events),
        "first_failure": first.failure_type if first else None,
        "first_failure_type": first.failure_type if first else None,
        "first_failure_event_id": first.event_id if first else None,
        "root_cause": root_output,
        "root_cause_candidate": root_candidate.failure_type if root_candidate else None,
        "root_cause_resolution": root_resolution,
        "root_cause_event_id": root.event_id if root else None,
        "root_causes": [_failure_descriptor(event) for event in roots],
        "root_cause_node": anchor.node if anchor else None,
        "root_cause_step": anchor.step if anchor else None,
        "root_cause_index": anchor.index if anchor else None,
        "downstream_failures": downstream,
        "downstream_symptoms": downstream_symptoms,
        "independent_failures": [
            _failure_descriptor(event) for event in independent_events
        ],
        "final_failure": final.failure_type if final else None,
        "final_failure_type": final.failure_type if final else None,
        "final_failure_event_id": final.event_id if final else None,
        "expected_first_failure": state.get("expected_first_failure") if isinstance(state, Mapping) else None,
        "first_failure_matches_expected": (
            normalize_failure_type(state.get("expected_first_failure"), default="") == first.failure_type
            if isinstance(state, Mapping) and state.get("expected_first_failure") and first else None
        ),
        "recovered": bool(recovered),
        "recovery": bool(recovered),
        "recovered_failure_event_ids": sorted(recovered_ids),
        "recovery_step": first_recovery.get("step") if first_recovery else None,
        "recovery_latency_ms": float(first_recovery.get("latency_ms", 0.0) or 0.0) if first_recovery else None,
        "cost": {"tokens": cost_tokens, "latency_ms": round(cost_latency, 2), "retries": cost_retries},
        "cost_before_root_cause": cost_before_root_cause,
        "cost_before_first_failure": cost_before_first_failure,
        "failure_events": [
            {
                "index": event.index,
                "event_id": event.event_id,
                "dependencies": list(event.dependencies),
                "recovery_of": list(event.recovery_of),
                "lineage_status": event.lineage_status,
                "origin": event.origin or None,
                "outcome_dimensions": list(event.outcome_dimensions),
                "gold_id": event.gold_id or None,
                "predicted_id": event.predicted_id or None,
                "failure_type": event.failure_type,
                "node": event.node,
                "action": event.action,
                "step": event.step,
                "status": event.status,
                "message": event.message,
                "latency_ms": event.latency_ms,
                "tokens": event.tokens,
                "retries": event.retries,
            }
            for event in ordered_events
        ],
    }
    return result


def build_failure_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    attributions = []
    for row in rows:
        existing = row.get("failure_attribution")
        if isinstance(existing, Mapping) and "root_cause" in existing and "failure_events" in existing:
            attributions.append(dict(existing))
        else:
            attributions.append(attribute_failure(row))
    def counts_for(field: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in attributions:
            value = item.get(field)
            if value:
                counts[str(value)] = counts.get(str(value), 0) + 1
        return counts

    symptom_counts: dict[str, int] = {}
    for item in attributions:
        for value in item.get("downstream_symptoms", []):
            symptom_counts[str(value)] = symptom_counts.get(str(value), 0) + 1
    before_root = [item.get("cost_before_root_cause") or item.get("cost_before_first_failure") or {} for item in attributions]
    before_first = [item.get("cost_before_first_failure") or {} for item in attributions]
    failed_count = sum(bool(item["has_failure"]) for item in attributions)
    expected_abstention_count = sum(
        bool(item.get("expected_abstention")) for item in attributions
    )
    observed_process_attributions = [
        item.get("observed_process_attribution")
        for item in attributions
        if isinstance(item.get("observed_process_attribution"), Mapping)
    ]
    observed_process_failure_count = sum(
        bool(item.get("has_failure")) for item in observed_process_attributions
    )
    failed_before_first = [
        item.get("cost_before_first_failure") or {}
        for item in attributions
        if item.get("has_failure")
    ]
    total_before_first = {
        "tokens": sum(int(item.get("tokens", 0) or 0) for item in before_first),
        "latency_ms": round(
            sum(float(item.get("latency_ms", 0.0) or 0.0) for item in before_first), 2
        ),
        "retries": sum(int(item.get("retries", 0) or 0) for item in before_first),
    }
    failed_before_first_total = {
        "tokens": sum(int(item.get("tokens", 0) or 0) for item in failed_before_first),
        "latency_ms": round(
            sum(
                float(item.get("latency_ms", 0.0) or 0.0)
                for item in failed_before_first
            ),
            2,
        ),
        "retries": sum(int(item.get("retries", 0) or 0) for item in failed_before_first),
    }
    return {
        "query_count": len(rows),
        "failed_query_count": failed_count,
        "failure_rate": round(failed_count / len(rows), 4) if rows else 0.0,
        "expected_abstention_count": expected_abstention_count,
        "observed_process_failure_query_count": observed_process_failure_count,
        "recovered_query_count": sum(bool(item["recovered"]) for item in attributions),
        "recovery_success_rate": (
            sum(bool(item["recovered"]) for item in attributions)
            / max(1, sum(bool(item["has_failure"]) for item in attributions))
        ),
        "first_failure_counts": counts_for("first_failure"),
        "root_cause_counts": counts_for("root_cause"),
        "final_failure_counts": counts_for("final_failure"),
        "downstream_symptom_counts": symptom_counts,
        "failure_type_counts": {
            key: sum(1 for item in attributions for event in item.get("failure_events", []) if event.get("failure_type") == key)
            for key in sorted({event.get("failure_type") for item in attributions for event in item.get("failure_events", []) if event.get("failure_type")})
        },
        "total_failure_events": sum(int(item["failure_count"]) for item in attributions),
        "total_cost_before_root_cause": {
            "tokens": sum(int(item.get("tokens", 0) or 0) for item in before_root),
            "latency_ms": round(sum(float(item.get("latency_ms", 0.0) or 0.0) for item in before_root), 2),
            "retries": sum(int(item.get("retries", 0) or 0) for item in before_root),
        },
        "total_cost_before_first_failure": total_before_first,
        "average_cost_before_first_failure": {
            "tokens": (
                round(failed_before_first_total["tokens"] / failed_count, 4)
                if failed_count
                else None
            ),
            "latency_ms": (
                round(failed_before_first_total["latency_ms"] / failed_count, 2)
                if failed_count
                else None
            ),
            "retries": (
                round(failed_before_first_total["retries"] / failed_count, 4)
                if failed_count
                else None
            ),
        },
        "total_cost": {"tokens": sum(int(item["cost"]["tokens"]) for item in attributions), "latency_ms": round(sum(float(item["cost"]["latency_ms"]) for item in attributions), 2), "retries": sum(int(item["cost"]["retries"]) for item in attributions)},
    }


__all__ = ["FAILURE_TAXONOMY", "FailureEvent", "FailureType", "attribute_failure", "build_failure_summary", "extract_failure_events", "normalize_failure_type"]
