"""Execute one dependency-ready reasoning step."""

from __future__ import annotations

from decimal import Decimal
from time import perf_counter
from typing import Any

from src.agent.calculation_operand_resolver import (
    AmbiguousOperand,
    CalculationOperandResolver,
    MissingOperand,
    OperandRequirement,
    ResolvedOperand,
)
from src.agent.calculation_provenance import CalculationStatus, execute_calculation
from src.agent.config import AgentConfig
from src.agent.policies import begin_node
from src.agent.reasoning_plan import ReasoningStep, ReasoningStepKind, StepResult, StepStatus
from src.agent.tool_orchestration import tool_call_key
from src.retrieval.domain_priority import apply_retrieval_domain_priority
from src.structured.agent_bridge import financial_fact_to_row
from src.structured.schema import Period
from src.structured.structured_query import build_structured_query, execute_structured_query


def _row_ids(rows: list[dict[str, Any]], *keys: str) -> tuple[str, ...]:
    values: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in keys:
            value = str(row.get(key) or "").strip()
            if value and value not in seen:
                seen.add(value)
                values.append(value)
                break
    return tuple(values)


def _is_single_step_plan(state: dict[str, Any]) -> bool:
    return len(list((state.get("reasoning_plan") or {}).get("steps") or [])) == 1


def _tool_call_was_attempted(
    state: dict[str, Any], tool_name: str, arguments: dict[str, Any]
) -> bool:
    """Share the controlled-tool dedup contract across rebuilt reasoning plans."""

    expected = tool_call_key(tool_name, arguments)
    for raw_call in state.get("tool_calls") or []:
        if not isinstance(raw_call, dict):
            continue
        if str(raw_call.get("tool_name") or "") != tool_name:
            continue
        raw_arguments = raw_call.get("arguments")
        if not isinstance(raw_arguments, dict):
            continue
        if tool_call_key(tool_name, raw_arguments) == expected:
            return True
    return False


def _stop_duplicate_tool_call(
    state: dict[str, Any],
    step: ReasoningStep,
    *,
    tool_name: str,
    arguments: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    state["termination_reason"] = state.get("termination_reason") or "duplicate_tool_call"
    state["no_improvement"] = True
    result = StepResult(
        step.step_id,
        StepStatus.SKIPPED,
        error_type="NO_IMPROVEMENT",
        latency_ms=(perf_counter() - started) * 1000,
    )
    state["pending_step_observation"] = {
        "result": result.to_dict(),
        "artifact": {
            "source": "duplicate_guard",
            "duplicate_tool_name": tool_name,
            "duplicate_arguments": dict(arguments),
            "rows": [],
        },
        # The attempted duplicate remains observable through the step event,
        # but is not recorded as an executed tool call.
        "tool_calls": [],
    }
    return state


def _effective_search_arguments(
    state: dict[str, Any], step: ReasoningStep
) -> dict[str, Any]:
    """Return the exact arguments that will be passed to report search.

    A recovery rewrite may rebuild a one-step plan from an older serialized
    step.  In that case ``active_query`` is the execution authority.  Multi-step
    plans keep each step's explicit query because their searches are distinct
    dependencies rather than retries of the top-level query.
    """

    planned_query = str(step.arguments.get("query") or state.get("query") or "")
    active_query = str(state.get("active_query") or "")
    query = (
        active_query
        if _is_single_step_plan(state) and int(state.get("rewrite_count", 0)) > 0
        else planned_query
    )
    return {
        "query": query or planned_query,
        "domain_hint": str(
            state.get("query_domain_bucket") or state.get("domain_hint") or ""
        ),
    }


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _operand_requirement(step: ReasoningStep, arguments: dict[str, Any], period: Period) -> OperandRequirement:
    return OperandRequirement(
        name=str(arguments.get("operand_name") or step.step_id),
        entity=str(arguments.get("company") or "").strip(),
        metric=str(arguments.get("metric") or "").strip(),
        period=period.label,
        value_type=str(arguments.get("value_type") or "actual").strip(),
        unit=_optional_text(arguments.get("unit")),
        period_basis=_optional_text(arguments.get("period_basis")),
        accounting_scope=_optional_text(
            arguments.get("accounting_scope") or arguments.get("scope")
        ),
        revision_status=_optional_text(arguments.get("revision_status")),
    )


def _resolved_operand_fact(
    resolution: ResolvedOperand,
    requirement: OperandRequirement,
    period: Period,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    operand = resolution.operand
    matching_row = next(
        (
            row
            for row in rows
            if str(row.get("evidence_id") or row.get("chunk_id") or "").strip()
            in {str(operand.evidence_id or "").strip(), str(operand.chunk_id or "").strip()}
        ),
        {},
    )
    source_span = str(
        matching_row.get("support_span")
        or matching_row.get("source_span")
        or matching_row.get("text")
        or operand.raw_text
        or ""
    )
    return {
        "fact_id": operand.fact_id,
        "company": requirement.entity,
        "metric": requirement.metric,
        "period": period.to_dict(),
        "value_type": requirement.value_type,
        "value": str(operand.value),
        "unit": operand.unit,
        "doc_id": operand.doc_id,
        "page": matching_row.get("page") or matching_row.get("page_start"),
        "evidence_id": operand.evidence_id,
        "chunk_id": operand.chunk_id,
        "raw_value": operand.raw_text or str(operand.value),
        "source_span": source_span,
        "accounting_scope": requirement.accounting_scope,
        "period_basis": requirement.period_basis,
        "source_date": None,
        "revision_status": requirement.revision_status,
        "resolution_source": resolution.source,
    }


def _materialize_operand_resolution(
    resolution: ResolvedOperand | AmbiguousOperand | MissingOperand,
    *,
    requirement: OperandRequirement,
    period: Period,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], StepStatus, str | None, dict[str, Any]]:
    """Convert a fail-closed resolver result into one step observation."""

    if isinstance(resolution, ResolvedOperand):
        return (
            [_resolved_operand_fact(resolution, requirement, period, rows)],
            StepStatus.SUCCESS,
            None,
            {"operand_resolution": resolution.source},
        )
    if isinstance(resolution, AmbiguousOperand):
        return (
            [],
            StepStatus.INSUFFICIENT,
            "AMBIGUOUS_OPERAND",
            {
                "candidate_ids": list(resolution.candidate_ids),
                "operand_resolution_reason": resolution.reason,
            },
        )
    return (
        [],
        StepStatus.INSUFFICIENT,
        "MISSING_OPERAND",
        {
            "operand_resolution_reason": resolution.reason,
            "operand_retrieval_attempted": resolution.retrieval_attempted,
        },
    )


def make_execute_step(runtime, fact_store, company_aliases, config: AgentConfig):
    del company_aliases  # lookup arguments already carry canonical coordinates
    capabilities = config.resolved_execution_capabilities()

    def execute_step(state: dict[str, Any]) -> dict[str, Any]:
        raw_step = state.get("pending_reasoning_step")
        if not raw_step:
            state["termination_reason"] = state.get("termination_reason") or "missing_pending_reasoning_step"
            return state
        step = ReasoningStep.from_mapping(raw_step)
        started = perf_counter()

        if not begin_node(state, config):
            result = StepResult(
                step.step_id,
                StepStatus.SKIPPED,
                error_type=str(state.get("termination_reason") or "BUDGET_EXHAUSTED").upper(),
            )
            state["pending_step_observation"] = {
                "result": result.to_dict(),
                "artifact": {},
                "tool_calls": [],
            }
            return state

        prior = {
            result.step_id: result
            for result in (
                StepResult.from_mapping(value)
                for value in (state.get("reasoning_step_results") or [])
            )
        }
        blocked = [
            dependency
            for dependency in step.depends_on
            if dependency not in prior or prior[dependency].status is not StepStatus.SUCCESS
        ]
        if blocked:
            result = StepResult(
                step.step_id,
                StepStatus.SKIPPED,
                error_type="DEPENDENCY_UNSATISFIED",
                latency_ms=(perf_counter() - started) * 1000,
            )
            state["pending_step_observation"] = {
                "result": result.to_dict(),
                "artifact": {"blocked_by": blocked},
                "tool_calls": [],
            }
            return state

        if int(state.get("tool_call_count", 0)) >= config.max_tool_calls:
            state["termination_reason"] = state.get("termination_reason") or "max_tool_calls"
            result = StepResult(
                step.step_id,
                StepStatus.SKIPPED,
                error_type="BUDGET_EXHAUSTED",
                latency_ms=(perf_counter() - started) * 1000,
            )
            state["pending_step_observation"] = {
                "result": result.to_dict(),
                "artifact": {},
                "tool_calls": [],
            }
            return state

        if step.kind is ReasoningStepKind.LOOKUP:
            arguments = dict(step.arguments)
            query = str(arguments.get("query") or "")
            requires_fact = bool(arguments.get("requires_fact"))
            retrieval_result: dict[str, Any] | None = None
            source = "structured"
            tool_calls: list[dict[str, Any]] = []
            fact_dicts: list[dict[str, Any]] = []
            resolution_artifact: dict[str, Any] = {}
            structured_fail_closed_reason: str | None = None
            try:
                period_value = arguments.get("period")
                period = Period.from_dict(period_value) if isinstance(period_value, dict) else None
                if fact_store is not None and period is not None:
                    structured_query = build_structured_query(
                        entities=[str(arguments.get("company") or "")],
                        metrics=[str(arguments.get("metric") or "")],
                        periods=[period],
                        operation="lookup",
                        value_type=(
                            arguments.get("value_type")
                            if "value_type" in arguments
                            else "actual"
                        ),
                        period_basis=arguments.get("period_basis"),
                        accounting_scope=arguments.get("accounting_scope"),
                        revision_status=arguments.get("revision_status"),
                    )
                    structured_result = execute_structured_query(fact_store, structured_query)
                    facts = list(structured_result.facts)
                    if structured_result.status == "fail_closed":
                        structured_fail_closed_reason = str(
                            structured_result.reason or "structured_query_fail_closed"
                        )
                else:
                    facts = []
                fact_dicts = [fact.to_dict() for fact in facts]
                rows = [financial_fact_to_row(fact) for fact in facts]
                structured_latency_ms = (perf_counter() - started) * 1000
                tool_calls.append(
                    {
                        "tool_name": "structured_lookup",
                        "arguments": arguments,
                        "reason": "reasoning_lookup_step",
                        "status": (
                            "INSUFFICIENT" if structured_fail_closed_reason else "SUCCESS"
                        ),
                        "result_summary": f"rows={len(rows)}",
                        "latency_ms": structured_latency_ms,
                        "error_type": (
                            structured_fail_closed_reason.upper()
                            if structured_fail_closed_reason
                            else None
                        ),
                        "reasoning_step_id": step.step_id,
                    }
                )
                if structured_fail_closed_reason:
                    status = StepStatus.INSUFFICIENT
                    error_type = structured_fail_closed_reason.upper()
                    resolution_artifact = {
                        "structured_query_status": "fail_closed",
                        "structured_query_reason": structured_fail_closed_reason,
                    }
                elif len(facts) == 1:
                    status = StepStatus.SUCCESS
                    error_type = None
                elif len(facts) > 1:
                    status = StepStatus.INSUFFICIENT
                    error_type = "AMBIGUOUS_FACT"
                else:
                    requirement = (
                        _operand_requirement(step, arguments, period)
                        if requires_fact and period is not None
                        else None
                    )
                    pool_rows = [
                        dict(row)
                        for row in (state.get("evidence_pool") or {}).values()
                        if isinstance(row, dict)
                    ]
                    pool_resolution = (
                        CalculationOperandResolver().resolve(
                            requirement,
                            structured_facts=(),
                            evidence_rows=pool_rows,
                        )
                        if requirement is not None and pool_rows
                        else None
                    )
                    if isinstance(pool_resolution, (ResolvedOperand, AmbiguousOperand)):
                        # Reuse already-observed evidence before spending the one
                        # bounded requirement-specific retrieval for this operand.
                        source = "evidence_pool"
                        rows = pool_rows
                        fact_dicts, status, error_type, resolution_artifact = (
                            _materialize_operand_resolution(
                                pool_resolution,
                                requirement=requirement,
                                period=period,
                                rows=rows,
                            )
                        )
                        resolution_artifact["operand_resolution_stage"] = "evidence_pool"
                    elif fact_store is not None and not capabilities.controlled_tools:
                        # The structured-only treatment must not silently take
                        # the full-agent report-search recovery path.
                        status = StepStatus.INSUFFICIENT
                        error_type = "CONTROLLED_TOOLS_DISABLED"
                    elif not capabilities.retrieval:
                        status = StepStatus.INSUFFICIENT
                        error_type = "RETRIEVAL_DISABLED"
                    elif (
                        config.max_tool_calls - int(state.get("tool_call_count", 0))
                    ) < 2:
                        state["termination_reason"] = (
                            state.get("termination_reason") or "max_tool_calls"
                        )
                        status = StepStatus.INSUFFICIENT
                        error_type = "BUDGET_EXHAUSTED"
                    elif (
                        _is_single_step_plan(state)
                        and int(state.get("retrieval_count", 0)) >= config.max_retrieval_rounds
                    ):
                        state["termination_reason"] = (
                            state.get("termination_reason") or "max_retrieval_rounds"
                        )
                        status = StepStatus.INSUFFICIENT
                        error_type = "BUDGET_EXHAUSTED"
                    else:
                        source = "report_search"
                        domain_hint = str(
                            state.get("query_domain_bucket") or state.get("domain_hint") or ""
                        )
                        retrieval_started = perf_counter()
                        retrieval_result = runtime.search(query)
                        retrieval_result = apply_retrieval_domain_priority(
                            retrieval_result,
                            domain_hint,
                        )
                        rows = [
                            row
                            for row in (
                                retrieval_result.get("rerank_rows")
                                or retrieval_result.get("hybrid_rows")
                                or []
                            )
                            if isinstance(row, dict)
                        ]
                        retrieval_latency_ms = (perf_counter() - retrieval_started) * 1000
                        if rows and not requires_fact:
                            status = StepStatus.SUCCESS
                            error_type = None
                        elif rows:
                            if period is None:
                                status = StepStatus.INSUFFICIENT
                                error_type = "MISSING_OPERAND_COORDINATES"
                            else:
                                requirement = requirement or _operand_requirement(
                                    step, arguments, period
                                )
                                resolution = CalculationOperandResolver().resolve(
                                    requirement,
                                    structured_facts=(),
                                    evidence_rows=rows,
                                )
                                fact_dicts, status, error_type, resolution_artifact = (
                                    _materialize_operand_resolution(
                                        resolution,
                                        requirement=requirement,
                                        period=period,
                                        rows=rows,
                                    )
                                )
                                resolution_artifact["operand_resolution_stage"] = (
                                    "bounded_retrieval"
                                )
                        else:
                            status = StepStatus.INSUFFICIENT
                            error_type = "FACT_NOT_FOUND"
                        tool_calls.append(
                            {
                                "tool_name": "report_search",
                                "arguments": {"query": query, "domain_hint": domain_hint},
                                "reason": "reasoning_lookup_miss_fallback",
                                "status": "SUCCESS",
                                "result_summary": f"rows={len(rows)}",
                                "latency_ms": retrieval_latency_ms,
                                "error_type": None,
                                "reasoning_step_id": step.step_id,
                            }
                        )
            except Exception as exc:
                if not config.enable_tool_orchestration and _is_single_step_plan(state):
                    raise
                facts = []
                fact_dicts = []
                rows = []
                status = StepStatus.FAILED
                error_type = type(exc).__name__
                if _is_single_step_plan(state):
                    state["termination_reason"] = (
                        state.get("termination_reason") or "tool_execution_failed"
                    )
                if source == "report_search":
                    tool_calls.append(
                        {
                            "tool_name": "report_search",
                            "arguments": {"query": query},
                            "reason": "reasoning_lookup_miss_fallback",
                            "status": "FAILED",
                            "result_summary": str(exc)[:200],
                            "latency_ms": (perf_counter() - started) * 1000,
                            "error_type": error_type,
                            "reasoning_step_id": step.step_id,
                        }
                    )
                elif not tool_calls:
                    tool_calls.append(
                        {
                            "tool_name": "structured_lookup",
                            "arguments": arguments,
                            "reason": "reasoning_lookup_step",
                            "status": "FAILED",
                            "result_summary": str(exc)[:200],
                            "latency_ms": (perf_counter() - started) * 1000,
                            "error_type": error_type,
                            "reasoning_step_id": step.step_id,
                        }
                    )
            latency_ms = (perf_counter() - started) * 1000
            result = StepResult(
                step.step_id,
                status,
                fact_ids=tuple(
                    str(fact.get("fact_id")) for fact in fact_dicts if fact.get("fact_id")
                ),
                evidence_ids=(
                    tuple(
                        str(fact.get("evidence_id"))
                        for fact in fact_dicts
                        if fact.get("evidence_id")
                    )
                    if fact_dicts
                    else _row_ids(rows, "chunk_id", "evidence_id")
                ),
                error_type=error_type,
                latency_ms=latency_ms,
                budget_usage={
                    "structured_lookups": 1,
                    "retrieval_calls": 1 if source == "report_search" else 0,
                    "tool_calls": len(tool_calls),
                },
            )
            state["pending_step_observation"] = {
                "result": result.to_dict(),
                "artifact": {
                    "source": source,
                    "query": query,
                    "facts": fact_dicts,
                    "rows": rows,
                    "retrieval_result": retrieval_result,
                    **resolution_artifact,
                },
                "tool_calls": tool_calls,
            }
            return state

        if step.kind is ReasoningStepKind.CALCULATE:
            if not capabilities.deterministic_calculation:
                result = StepResult(
                    step.step_id,
                    StepStatus.SKIPPED,
                    error_type="DETERMINISTIC_CALCULATION_DISABLED",
                    latency_ms=(perf_counter() - started) * 1000,
                )
                state["pending_step_observation"] = {
                    "result": result.to_dict(),
                    "artifact": {},
                    "tool_calls": [],
                }
                return state
            artifacts = state.get("reasoning_step_artifacts") or {}
            operand_steps = dict(step.arguments.get("operand_steps") or {})
            inputs: list[dict[str, Any]] = []
            missing: list[str] = []
            for name, source_step_id in operand_steps.items():
                source = artifacts.get(str(source_step_id)) or {}
                facts = [fact for fact in (source.get("facts") or []) if isinstance(fact, dict)]
                if len(facts) != 1:
                    missing.append(str(source_step_id))
                    continue
                fact = facts[0]
                period = fact.get("period")
                if isinstance(period, dict):
                    period = f"{period.get('kind', 'FY')}{period.get('year', '')}"
                inputs.append(
                    {
                        "name": str(name),
                        "value": fact.get("value"),
                        "unit": fact.get("unit"),
                        "evidence_id": fact.get("evidence_id"),
                        "fact_id": fact.get("fact_id"),
                        "doc_id": fact.get("doc_id"),
                        "raw_text": fact.get("raw_value") or fact.get("source_span"),
                        "period": period,
                        "source_step_id": str(source_step_id),
                    }
                )
            if missing or len(inputs) != len(operand_steps):
                latency_ms = (perf_counter() - started) * 1000
                result = StepResult(
                    step.step_id,
                    StepStatus.INSUFFICIENT,
                    error_type="MISSING_CALCULATION_OPERAND",
                    latency_ms=latency_ms,
                )
                state["pending_step_observation"] = {
                    "result": result.to_dict(),
                    "artifact": {"missing_operand_steps": missing},
                    "tool_calls": [],
                }
                return state

            calculation = execute_calculation(
                str(step.arguments.get("operation") or ""),
                inputs,
                precision=int(step.arguments.get("precision") or 4),
                years=step.arguments.get("years"),
            )
            calculation_dict = calculation.to_dict()
            success = calculation.status is CalculationStatus.SUCCESS
            latency_ms = (perf_counter() - started) * 1000
            result = StepResult(
                step.step_id,
                StepStatus.SUCCESS if success else StepStatus.FAILED,
                fact_ids=tuple(str(item.get("fact_id")) for item in inputs if item.get("fact_id")),
                evidence_ids=tuple(
                    str(item.get("evidence_id")) for item in inputs if item.get("evidence_id")
                ),
                calculation_id=calculation.calculation_id,
                error_type=None if success else calculation.status.value,
                latency_ms=latency_ms,
                budget_usage={"calculation_calls": 1, "tool_calls": 1},
            )
            state["pending_step_observation"] = {
                "result": result.to_dict(),
                "artifact": {
                    "source": "calculator",
                    "calculation": calculation_dict,
                    "operand_steps": operand_steps,
                },
                "tool_calls": [
                    {
                        "tool_name": "calculator",
                        "arguments": {
                            "operation": step.arguments.get("operation"),
                            "inputs": inputs,
                            "precision": int(step.arguments.get("precision") or 4),
                        },
                        "reason": "reasoning_calculate_step",
                        "status": result.status.value,
                        "result_summary": f"calculation_status={calculation.status.value}",
                        "latency_ms": latency_ms,
                        "error_type": result.error_type,
                        "reasoning_step_id": step.step_id,
                        "calculation_id": calculation.calculation_id,
                    }
                ],
            }
            return state

        if step.kind is ReasoningStepKind.COMPARE:
            if not capabilities.deterministic_calculation:
                result = StepResult(
                    step.step_id,
                    StepStatus.SKIPPED,
                    error_type="DETERMINISTIC_CALCULATION_DISABLED",
                    latency_ms=(perf_counter() - started) * 1000,
                )
                state["pending_step_observation"] = {
                    "result": result.to_dict(),
                    "artifact": {},
                    "tool_calls": [],
                }
                return state
            artifacts = state.get("reasoning_step_artifacts") or {}
            operand_steps = dict(step.arguments.get("operand_steps") or {})
            calculation_inputs: list[dict[str, Any]] = []
            evidence_ids: list[str] = []
            missing: list[str] = []
            for name, source_step_id in operand_steps.items():
                source = artifacts.get(str(source_step_id)) or {}
                calculation = source.get("calculation")
                if (
                    not isinstance(calculation, dict)
                    or calculation.get("status") != CalculationStatus.SUCCESS.value
                    or not isinstance(calculation.get("result"), dict)
                ):
                    missing.append(str(source_step_id))
                    continue
                calculation_inputs.append(
                    {
                        "name": str(name),
                        "value": calculation["result"].get("value"),
                        "unit": None,
                        "upstream_calculation_id": calculation.get("calculation_id"),
                        "source_step_id": str(source_step_id),
                        "upstream_evidence_ids": list(
                            calculation.get("evidence_ids")
                            or dict.fromkeys(
                                str(item.get("evidence_id") or "")
                                for item in (calculation.get("inputs") or [])
                                if isinstance(item, dict) and item.get("evidence_id")
                            )
                        ),
                        "raw_text": calculation["result"].get("formatted"),
                    }
                )
                for item in calculation.get("inputs") or []:
                    if isinstance(item, dict) and item.get("evidence_id"):
                        evidence_id = str(item["evidence_id"])
                        if evidence_id not in evidence_ids:
                            evidence_ids.append(evidence_id)
            if missing or len(calculation_inputs) != 2:
                latency_ms = (perf_counter() - started) * 1000
                result = StepResult(
                    step.step_id,
                    StepStatus.SKIPPED,
                    evidence_ids=tuple(evidence_ids),
                    error_type="MISSING_COMPARISON_INPUT",
                    latency_ms=latency_ms,
                )
                state["pending_step_observation"] = {
                    "result": result.to_dict(),
                    "artifact": {"missing_operand_steps": missing},
                    "tool_calls": [],
                }
                return state

            calculation = execute_calculation(
                "difference",
                calculation_inputs,
                precision=int(step.arguments.get("precision") or 4),
            )
            success = calculation.status is CalculationStatus.SUCCESS
            labels = dict(step.arguments.get("labels") or {})
            conclusion: dict[str, Any] | None = None
            if success and calculation.result is not None:
                difference = Decimal(calculation.result.value)
                if difference > 0:
                    relation = "greater"
                    winner = str(labels.get("left") or "left")
                    loser = str(labels.get("right") or "right")
                elif difference < 0:
                    relation = "less"
                    winner = str(labels.get("right") or "right")
                    loser = str(labels.get("left") or "left")
                else:
                    relation = "equal"
                    winner = None
                    loser = None
                conclusion = {
                    "relation": relation,
                    "winner": winner,
                    "loser": loser,
                    "left_label": labels.get("left"),
                    "right_label": labels.get("right"),
                    "left_value": calculation_inputs[0]["value"],
                    "right_value": calculation_inputs[1]["value"],
                    "calculation_id": calculation.calculation_id,
                    "answer": (
                        f"{winner}增长更快。" if winner else f"{labels.get('left')}与{labels.get('right')}增长率相同。"
                    ),
                }
            latency_ms = (perf_counter() - started) * 1000
            result = StepResult(
                step.step_id,
                StepStatus.SUCCESS if success else StepStatus.FAILED,
                evidence_ids=tuple(evidence_ids),
                calculation_id=calculation.calculation_id,
                error_type=None if success else calculation.status.value,
                latency_ms=latency_ms,
                budget_usage={"calculation_calls": 1, "tool_calls": 1},
            )
            calculation_dict = calculation.to_dict()
            calculation_dict["evidence_ids"] = list(evidence_ids)
            calculation_dict["comparison"] = conclusion
            state["pending_step_observation"] = {
                "result": result.to_dict(),
                "artifact": {
                    "source": "calculator",
                    "calculation": calculation_dict,
                    "operand_steps": operand_steps,
                    "conclusion": conclusion,
                },
                "tool_calls": [
                    {
                        "tool_name": "calculator",
                        "arguments": {"operation": "difference", "inputs": calculation_inputs},
                        "reason": "reasoning_compare_step",
                        "status": result.status.value,
                        "result_summary": f"calculation_status={calculation.status.value}",
                        "latency_ms": latency_ms,
                        "error_type": result.error_type,
                        "reasoning_step_id": step.step_id,
                        "calculation_id": calculation.calculation_id,
                    }
                ],
            }
            return state

        if step.kind is not ReasoningStepKind.SEARCH:
            result = StepResult(
                step.step_id,
                StepStatus.FAILED,
                error_type="UNSUPPORTED_STEP_KIND",
                latency_ms=(perf_counter() - started) * 1000,
            )
            state["pending_step_observation"] = {
                "result": result.to_dict(),
                "artifact": {},
                "tool_calls": [],
            }
            return state

        report_arguments = _effective_search_arguments(state, step)
        query = str(report_arguments["query"])
        domain_hint = str(report_arguments["domain_hint"])
        if not capabilities.retrieval:
            result = StepResult(
                step.step_id,
                StepStatus.SKIPPED,
                error_type="RETRIEVAL_DISABLED",
                latency_ms=(perf_counter() - started) * 1000,
            )
            state["pending_step_observation"] = {
                "result": result.to_dict(),
                "artifact": {"source": "capability_guard", "query": query, "rows": []},
                "tool_calls": [],
            }
            return state
        if _tool_call_was_attempted(state, "report_search", report_arguments):
            return _stop_duplicate_tool_call(
                state,
                step,
                tool_name="report_search",
                arguments=report_arguments,
                started=started,
            )
        # ``max_retrieval_rounds`` bounds single-query retry/rewrite recovery.
        # A multi-step plan is instead bounded by max_sub_questions,
        # max_tool_calls and max_steps; applying the retry cap here would skip
        # required, independent SEARCH steps and manufacture a partial answer.
        if (
            _is_single_step_plan(state)
            and int(state.get("retrieval_count", 0)) >= config.max_retrieval_rounds
        ):
            state["termination_reason"] = (
                state.get("termination_reason") or "max_retrieval_rounds"
            )
            result = StepResult(
                step.step_id,
                StepStatus.SKIPPED,
                error_type="BUDGET_EXHAUSTED",
                latency_ms=(perf_counter() - started) * 1000,
            )
            state["pending_step_observation"] = {
                "result": result.to_dict(),
                "artifact": {"source": "budget_guard", "query": query, "rows": []},
                "tool_calls": [],
            }
            return state
        try:
            retrieval_result = runtime.search(query)
            retrieval_result = apply_retrieval_domain_priority(retrieval_result, domain_hint)
            rows = [
                row
                for row in (
                    retrieval_result.get("rerank_rows")
                    or retrieval_result.get("hybrid_rows")
                    or []
                )
                if isinstance(row, dict)
            ]
            status = StepStatus.SUCCESS if rows else StepStatus.INSUFFICIENT
            error_type = None if rows else "NO_EVIDENCE"
        except Exception as exc:  # controlled-tool failures are trace data
            if not config.enable_tool_orchestration and _is_single_step_plan(state):
                raise
            retrieval_result = {
                "query_mode": "reasoning_search",
                "numeric_query": False,
                "hybrid_rows": [],
                "rerank_rows": [],
                "timings": {},
            }
            rows = []
            status = StepStatus.FAILED
            error_type = type(exc).__name__
            if _is_single_step_plan(state):
                state["termination_reason"] = (
                    state.get("termination_reason") or "tool_execution_failed"
                )

        latency_ms = (perf_counter() - started) * 1000
        result = StepResult(
            step.step_id,
            status,
            evidence_ids=_row_ids(rows, "chunk_id", "evidence_id"),
            error_type=error_type,
            latency_ms=latency_ms,
            budget_usage={"retrieval_calls": 1, "tool_calls": 1},
        )
        state["pending_step_observation"] = {
            "result": result.to_dict(),
            "artifact": {
                "source": "report_search",
                "query": query,
                "rows": rows,
                "retrieval_result": retrieval_result,
            },
            "tool_calls": [
                {
                    "tool_name": "report_search",
                    "arguments": report_arguments,
                    "reason": "reasoning_search_step",
                    "status": "SUCCESS" if status is StepStatus.SUCCESS else status.value,
                    "result_summary": f"rows={len(rows)}",
                    "latency_ms": latency_ms,
                    "error_type": error_type,
                    "reasoning_step_id": step.step_id,
                }
            ],
        }
        return state

    return execute_step


__all__ = ["make_execute_step"]
