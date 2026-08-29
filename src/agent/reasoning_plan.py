"""Stable reasoning-plan interface shared by single-hop and multi-hop flows.

The module owns plan construction and validation.  Callers only need the
three immutable records below; graph nodes and compatibility adapters may
change without changing the interface.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from src.structured.metric_registry import extract_metrics, metric_label
from src.structured.schema import Metric, Period
from src.structured.structured_query import StructuredQuery, extract_companies, parse_structured_query
from src.agent.tool_orchestration import detect_calculation_intent


class ReasoningStepKind(str, Enum):
    LOOKUP = "LOOKUP"
    SEARCH = "SEARCH"
    CALCULATE = "CALCULATE"
    COMPARE = "COMPARE"


class StepStatus(str, Enum):
    SUCCESS = "SUCCESS"
    INSUFFICIENT = "INSUFFICIENT"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


def _clean_ids(values: Iterable[Any]) -> tuple[str, ...]:
    return tuple(str(value).strip() for value in values if str(value).strip())


@dataclass(frozen=True)
class ReasoningStep:
    step_id: str
    kind: ReasoningStepKind
    depends_on: tuple[str, ...] = ()
    required: bool = True
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        step_id = str(self.step_id).strip()
        if not step_id:
            raise ValueError("reasoning step_id must be non-empty")
        object.__setattr__(self, "step_id", step_id)
        try:
            object.__setattr__(self, "kind", ReasoningStepKind(self.kind))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid reasoning step kind {self.kind!r}") from exc
        object.__setattr__(self, "depends_on", _clean_ids(self.depends_on))
        if self.step_id in self.depends_on:
            raise ValueError(f"reasoning step {self.step_id!r} cannot depend on itself")
        if not isinstance(self.arguments, Mapping):
            raise ValueError("reasoning step arguments must be a mapping")
        object.__setattr__(self, "arguments", dict(self.arguments))

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "kind": self.kind.value,
            "depends_on": list(self.depends_on),
            "required": self.required,
            "arguments": dict(self.arguments),
        }

    @classmethod
    def from_mapping(cls, value: "ReasoningStep | Mapping[str, Any]") -> "ReasoningStep":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("reasoning step must be a mapping")
        return cls(
            step_id=str(value.get("step_id") or value.get("id") or ""),
            kind=value.get("kind", ReasoningStepKind.SEARCH),
            depends_on=tuple(value.get("depends_on") or ()),
            required=bool(value.get("required", True)),
            arguments=dict(value.get("arguments") or {}),
        )


@dataclass(frozen=True)
class ReasoningPlan:
    plan_id: str
    steps: tuple[ReasoningStep, ...]
    answer_requirement_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        plan_id = str(self.plan_id).strip()
        if not plan_id:
            raise ValueError("reasoning plan_id must be non-empty")
        object.__setattr__(self, "plan_id", plan_id)
        steps = tuple(ReasoningStep.from_mapping(step) for step in self.steps)
        if not steps:
            raise ValueError("reasoning plan requires at least one step")
        object.__setattr__(self, "steps", steps)
        answer_ids = _clean_ids(self.answer_requirement_ids)
        object.__setattr__(self, "answer_requirement_ids", answer_ids)
        self._validate_graph()

    def _validate_graph(self) -> None:
        step_ids = [step.step_id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("reasoning plan step IDs must be unique")
        known = set(step_ids)
        for step in self.steps:
            unknown = set(step.depends_on) - known
            if unknown:
                raise ValueError(f"reasoning step {step.step_id!r} has unknown dependencies: {sorted(unknown)}")
        unknown_answers = set(self.answer_requirement_ids) - known
        if unknown_answers:
            raise ValueError(f"unknown answer requirements: {sorted(unknown_answers)}")
        if not self.answer_requirement_ids:
            raise ValueError("reasoning plan requires at least one answer requirement")

        dependencies = {step.step_id: set(step.depends_on) for step in self.steps}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(step_id: str) -> None:
            if step_id in visiting:
                raise ValueError("reasoning plan dependencies must be acyclic")
            if step_id in visited:
                return
            visiting.add(step_id)
            for dependency in dependencies[step_id]:
                visit(dependency)
            visiting.remove(step_id)
            visited.add(step_id)

        for step_id in step_ids:
            visit(step_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "steps": [step.to_dict() for step in self.steps],
            "answer_requirement_ids": list(self.answer_requirement_ids),
        }

    @classmethod
    def from_mapping(cls, value: "ReasoningPlan | Mapping[str, Any]") -> "ReasoningPlan":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("reasoning plan must be a mapping")
        return cls(
            plan_id=str(value.get("plan_id") or ""),
            steps=tuple(ReasoningStep.from_mapping(step) for step in (value.get("steps") or ())),
            answer_requirement_ids=tuple(value.get("answer_requirement_ids") or ()),
        )


@dataclass(frozen=True)
class StepResult:
    step_id: str
    status: StepStatus
    fact_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    calculation_id: str | None = None
    error_type: str | None = None
    latency_ms: float = 0.0
    budget_usage: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        step_id = str(self.step_id).strip()
        if not step_id:
            raise ValueError("step result step_id must be non-empty")
        object.__setattr__(self, "step_id", step_id)
        try:
            object.__setattr__(self, "status", StepStatus(self.status))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid step status {self.status!r}") from exc
        object.__setattr__(self, "fact_ids", _clean_ids(self.fact_ids))
        object.__setattr__(self, "evidence_ids", _clean_ids(self.evidence_ids))
        object.__setattr__(self, "calculation_id", str(self.calculation_id).strip() if self.calculation_id else None)
        object.__setattr__(self, "error_type", str(self.error_type).strip() if self.error_type else None)
        object.__setattr__(self, "latency_ms", max(0.0, float(self.latency_ms or 0.0)))
        if not isinstance(self.budget_usage, Mapping):
            raise ValueError("step result budget_usage must be a mapping")
        object.__setattr__(
            self,
            "budget_usage",
            {str(key): int(value) for key, value in self.budget_usage.items()},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "status": self.status.value,
            "fact_ids": list(self.fact_ids),
            "evidence_ids": list(self.evidence_ids),
            "calculation_id": self.calculation_id,
            "error_type": self.error_type,
            "latency_ms": self.latency_ms,
            "budget_usage": dict(self.budget_usage),
        }

    @classmethod
    def from_mapping(cls, value: "StepResult | Mapping[str, Any]") -> "StepResult":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("step result must be a mapping")
        return cls(
            step_id=str(value.get("step_id") or value.get("id") or ""),
            status=value.get("status", StepStatus.INSUFFICIENT),
            fact_ids=tuple(value.get("fact_ids") or ()),
            evidence_ids=tuple(value.get("evidence_ids") or value.get("new_chunk_ids") or ()),
            calculation_id=value.get("calculation_id"),
            error_type=value.get("error_type"),
            latency_ms=float(value.get("latency_ms") or 0.0),
            budget_usage=dict(value.get("budget_usage") or {}),
        )


class LegacySubQuestionAdapter:
    """Translate the retired sub-question result shape at the state seam."""

    @staticmethod
    def from_legacy(value: Any, *, fallback_id: str = "") -> StepResult:
        from src.agent.dependency_reasoning import SubQuestionResult

        legacy = SubQuestionResult.from_mapping(value, fallback_id=fallback_id)
        status_text = legacy.status.strip().lower()
        if legacy.completed:
            status = StepStatus.SUCCESS
        elif status_text in {"failed", "error"}:
            status = StepStatus.FAILED
        elif status_text in {"skipped", "skipped_budget"}:
            status = StepStatus.SKIPPED
        else:
            status = StepStatus.INSUFFICIENT
        fact_ids = tuple(
            str(fact.get("fact_id")).strip()
            for fact in legacy.facts
            if isinstance(fact, Mapping) and fact.get("fact_id")
        )
        return StepResult(
            step_id=legacy.id,
            status=status,
            fact_ids=fact_ids,
            evidence_ids=legacy.evidence_ids,
            error_type=legacy.metadata.get("error_type"),
            latency_ms=float(legacy.metadata.get("latency_ms") or 0.0),
            budget_usage=dict(legacy.metadata.get("budget_usage") or {}),
        )

    @staticmethod
    def to_legacy(
        value: StepResult | Mapping[str, Any],
        *,
        answer: str = "",
        required_fields: Iterable[str] = (),
        facts: Iterable[Mapping[str, Any]] = (),
    ) -> Any:
        from src.agent.dependency_reasoning import SubQuestionResult

        result = StepResult.from_mapping(value)
        legacy_status = {
            StepStatus.SUCCESS: "completed",
            StepStatus.INSUFFICIENT: "insufficient",
            StepStatus.FAILED: "failed",
            StepStatus.SKIPPED: "skipped",
        }[result.status]
        required = _clean_ids(required_fields)
        normalized_facts = tuple(dict(fact) for fact in facts if isinstance(fact, Mapping))
        return SubQuestionResult(
            id=result.step_id,
            status=legacy_status,
            answer=str(answer or ""),
            evidence_ids=result.evidence_ids,
            required_fields=required,
            missing_fields=() if result.status is StepStatus.SUCCESS else required,
            facts=normalized_facts,
            metadata={
                "step_result": result.to_dict(),
                "latency_ms": result.latency_ms,
                "budget_usage": dict(result.budget_usage),
                "error_type": result.error_type,
            },
        )


_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_GROWTH_MARKERS = ("同比", "增长率", "增速", "growth")
_COMPARISON_MARKERS = ("比较", "对比", "谁", "哪家", "更快", "高于", "低于")


def _companies_in_query_order(
    query: str,
    company_aliases: Mapping[str, Sequence[str]],
) -> tuple[str, ...]:
    found: list[tuple[int, int, str]] = []
    for canonical, aliases in company_aliases.items():
        candidates = [str(canonical), *(str(alias) for alias in aliases)]
        matches = [
            (query.find(alias), -len(alias))
            for alias in candidates
            if alias and query.find(alias) >= 0
        ]
        if matches:
            position, negative_length = min(matches)
            found.append((position, negative_length, str(canonical)))
    return tuple(company for _position, _negative_length, company in sorted(found))


def _plan_id(query: str, steps: Sequence[ReasoningStep], answer_requirement_ids: Sequence[str]) -> str:
    payload = {
        "query": query,
        "steps": [step.to_dict() for step in steps],
        "answer_requirement_ids": list(answer_requirement_ids),
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return "PLAN-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _lookup_step(
    *,
    step_id: str,
    company: str,
    metric: Any,
    year: int,
    period_kind: str = "FY",
    requires_fact: bool = False,
    structured_query: StructuredQuery | None = None,
    query_text: str | None = None,
) -> ReasoningStep:
    label = metric_label(metric)
    value_type = (
        structured_query.value_type.value
        if structured_query is not None and structured_query.value_type is not None
        else (None if structured_query is not None else "actual")
    )
    period_basis = (
        structured_query.period_basis.value
        if structured_query is not None and structured_query.period_basis is not None
        else None
    )
    return ReasoningStep(
        step_id=step_id,
        kind=ReasoningStepKind.LOOKUP,
        arguments={
            "query": query_text or f"{company}{year}年{label}是多少？",
            "company": company,
            "metric": metric.value,
            "period": {"kind": period_kind, "year": year},
            "value_type": value_type,
            "period_basis": period_basis,
            "accounting_scope": (
                structured_query.accounting_scope if structured_query is not None else None
            ),
            "revision_status": (
                structured_query.revision_status if structured_query is not None else None
            ),
            "requires_fact": requires_fact,
        },
    )


def _single_lookup_coordinates(
    query: str,
    *,
    company_aliases: Mapping[str, Sequence[str]],
) -> tuple[StructuredQuery, str, Any, Period] | None:
    structured = parse_structured_query(query, company_aliases=company_aliases)
    if structured is None or structured.operation != "lookup":
        return None
    if len(structured.entities) != 1 or len(structured.metrics) != 1 or len(structured.periods) != 1:
        return None
    period = structured.periods[0]
    if not isinstance(period, Period):
        return None
    return structured, structured.entities[0], structured.metrics[0], period


def _growth_comparison_plan(
    query: str,
    *,
    company_aliases: Mapping[str, Sequence[str]],
) -> ReasoningPlan | None:
    lowered = query.lower()
    if not any(marker in lowered for marker in _GROWTH_MARKERS):
        return None
    if not any(marker in lowered for marker in _COMPARISON_MARKERS):
        return None
    companies = _companies_in_query_order(query, company_aliases) or extract_companies(query, company_aliases)
    metrics = extract_metrics(query)
    years = list(dict.fromkeys(int(value) for value in _YEAR_RE.findall(query)))
    if len(companies) != 2 or len(metrics) != 1 or not years:
        return None
    start_year, end_year = (years[0] - 1, years[0]) if len(years) == 1 else (min(years), max(years))
    if start_year >= end_year:
        return None

    metric = metrics[0]
    structured = parse_structured_query(query, company_aliases=company_aliases)
    lookup_steps: list[ReasoningStep] = []
    calculation_steps: list[ReasoningStep] = []
    for company_index, company in enumerate(companies, start=1):
        prior_id = f"lookup_{company_index}_prior"
        current_id = f"lookup_{company_index}_current"
        lookup_steps.extend(
            [
                _lookup_step(
                    step_id=prior_id,
                    company=company,
                    metric=metric,
                    year=start_year,
                    requires_fact=True,
                    structured_query=structured,
                ),
                _lookup_step(
                    step_id=current_id,
                    company=company,
                    metric=metric,
                    year=end_year,
                    requires_fact=True,
                    structured_query=structured,
                ),
            ]
        )
        calculation_steps.append(
            ReasoningStep(
                step_id=f"calculate_{company_index}_yoy",
                kind=ReasoningStepKind.CALCULATE,
                depends_on=(prior_id, current_id),
                arguments={
                    "operation": "yoy",
                    "company": company,
                    "operand_steps": {"prior": prior_id, "current": current_id},
                    "precision": 4,
                },
            )
        )
    compare = ReasoningStep(
        step_id="compare_growth",
        kind=ReasoningStepKind.COMPARE,
        depends_on=tuple(step.step_id for step in calculation_steps),
        arguments={
            "operand_steps": {
                "left": calculation_steps[0].step_id,
                "right": calculation_steps[1].step_id,
            },
            "labels": {"left": companies[0], "right": companies[1]},
            "comparison": "greater",
        },
    )
    steps = tuple([*lookup_steps, *calculation_steps, compare])
    answer_ids = (compare.step_id,)
    return ReasoningPlan(_plan_id(query, steps, answer_ids), steps, answer_ids)


def _single_company_growth_plan(
    query: str,
    *,
    company_aliases: Mapping[str, Sequence[str]],
) -> ReasoningPlan | None:
    lowered = query.lower()
    if not any(marker in lowered for marker in _GROWTH_MARKERS):
        return None
    companies = _companies_in_query_order(query, company_aliases) or extract_companies(
        query, company_aliases
    )
    metrics = extract_metrics(query)
    years = list(dict.fromkeys(int(value) for value in _YEAR_RE.findall(query)))
    if len(companies) != 1 or len(metrics) != 1 or not years:
        return None
    start_year, end_year = (
        (years[0] - 1, years[0]) if len(years) == 1 else (min(years), max(years))
    )
    if start_year >= end_year:
        return None

    company = companies[0]
    metric = metrics[0]
    structured = parse_structured_query(query, company_aliases=company_aliases)
    prior = _lookup_step(
        step_id="lookup_prior",
        company=company,
        metric=metric,
        year=start_year,
        requires_fact=True,
        structured_query=structured,
    )
    current = _lookup_step(
        step_id="lookup_current",
        company=company,
        metric=metric,
        year=end_year,
        requires_fact=True,
        structured_query=structured,
    )
    calculation = ReasoningStep(
        step_id="calculate_yoy",
        kind=ReasoningStepKind.CALCULATE,
        depends_on=(prior.step_id, current.step_id),
        arguments={
            "operation": "yoy",
            "company": company,
            "operand_steps": {"prior": prior.step_id, "current": current.step_id},
            "precision": 4,
        },
    )
    steps = (prior, current, calculation)
    return ReasoningPlan(
        _plan_id(query, steps, (calculation.step_id,)),
        steps,
        (calculation.step_id,),
    )


def _general_calculation_plan(
    query: str,
    *,
    company_aliases: Mapping[str, Sequence[str]],
) -> ReasoningPlan | None:
    """Translate the bounded legacy intent parser into public reasoning steps.

    The intent parser only names operations and fully coordinated operands; it
    does not execute retrieval or arithmetic.  This adapter keeps that useful
    parsing logic while ensuring every operation runs through the same
    LOOKUP -> CALCULATE graph and provenance records as YoY.
    """

    intent = detect_calculation_intent(query, company_aliases=company_aliases)
    if not isinstance(intent, Mapping):
        return None
    operation = str(intent.get("operation") or "").strip().lower()
    # Existing YoY/growth plans carry the same contract and stable ids.
    if operation in {"yoy", "growth_rate"}:
        return None
    operands = [item for item in (intent.get("operands") or []) if isinstance(item, Mapping)]
    if len(operands) != 2 or any(
        not operand.get("name")
        or not operand.get("company")
        or not operand.get("metric")
        or not isinstance(operand.get("period"), Mapping)
        for operand in operands
    ):
        # Literal-only requests carry no evidence/fact provenance and are not
        # eligible for a verified calculation path.
        return None

    structured = parse_structured_query(query, company_aliases=company_aliases)
    structured_periods = (
        [period for period in structured.periods if isinstance(period, Period)]
        if structured is not None
        else []
    )
    if operation == "qoq" and len(structured_periods) == 2:
        operands = [
            {**dict(operand), "period": structured_periods[index].to_dict()}
            for index, operand in enumerate(operands)
        ]

    lookup_steps: list[ReasoningStep] = []
    operand_steps: dict[str, str] = {}
    for index, operand in enumerate(operands, start=1):
        period = Period.from_dict(operand["period"])
        name = str(operand["name"])
        step_id = f"lookup_{name}"
        if step_id in operand_steps.values():
            step_id = f"lookup_{index}_{name}"
        step = _lookup_step(
            step_id=step_id,
            company=str(operand["company"]),
            metric=Metric(str(operand["metric"])),
            year=period.year,
            period_kind=period.kind.value,
            requires_fact=True,
            structured_query=structured,
            query_text=str(operand.get("query") or query),
        )
        step_arguments = dict(step.arguments)
        step_arguments["operand_name"] = name
        step = ReasoningStep(
            step_id=step.step_id,
            kind=step.kind,
            depends_on=step.depends_on,
            required=step.required,
            arguments=step_arguments,
        )
        lookup_steps.append(step)
        operand_steps[name] = step_id

    calculation = ReasoningStep(
        step_id=f"calculate_{operation}",
        kind=ReasoningStepKind.CALCULATE,
        depends_on=tuple(step.step_id for step in lookup_steps),
        arguments={
            "operation": operation,
            "operand_steps": operand_steps,
            "precision": int(intent.get("precision") or 4),
            "years": intent.get("years"),
        },
    )
    steps = tuple([*lookup_steps, calculation])
    return ReasoningPlan(
        _plan_id(query, steps, (calculation.step_id,)),
        steps,
        (calculation.step_id,),
    )


def build_reasoning_plan(
    query: str,
    *,
    fact_store_available: bool = False,
    company_aliases: Mapping[str, Sequence[str]] | None = None,
    sub_questions: Iterable[Mapping[str, Any]] | None = None,
) -> ReasoningPlan:
    """Build the shortest valid plan for a query.

    A deterministic A/B growth comparison is expanded into explicit lookup,
    calculation and comparison dependencies.  Legacy decompositions are
    accepted as input; every other query becomes exactly one lookup or search.
    """
    cleaned_query = str(query or "").strip()
    if not cleaned_query:
        raise ValueError("reasoning query must be non-empty")
    aliases = company_aliases or {}
    comparison = _growth_comparison_plan(cleaned_query, company_aliases=aliases)
    if comparison is not None:
        return comparison
    calculation = _general_calculation_plan(cleaned_query, company_aliases=aliases)
    if calculation is not None:
        return calculation
    growth = _single_company_growth_plan(cleaned_query, company_aliases=aliases)
    if growth is not None:
        return growth

    legacy_steps: list[ReasoningStep] = []
    for index, sub_question in enumerate(sub_questions or (), start=1):
        if not isinstance(sub_question, Mapping):
            continue
        sub_query = str(sub_question.get("query") or "").strip()
        if not sub_query:
            continue
        coordinates = (
            _single_lookup_coordinates(sub_query, company_aliases=aliases)
            if fact_store_available
            else None
        )
        arguments: dict[str, Any] = {
            "query": sub_query,
            "required_fields": list(sub_question.get("required_fields") or ()),
        }
        if coordinates is not None:
            structured, company, metric, period = coordinates
            arguments = dict(
                _lookup_step(
                    step_id=str(sub_question.get("id") or f"q{index}"),
                    company=company,
                    metric=metric,
                    year=period.year,
                    period_kind=period.kind.value,
                    structured_query=structured,
                    query_text=sub_query,
                ).arguments
            ) | {"required_fields": list(sub_question.get("required_fields") or ())}
        legacy_steps.append(
            ReasoningStep(
                step_id=str(sub_question.get("id") or f"q{index}"),
                kind=ReasoningStepKind.LOOKUP if coordinates else ReasoningStepKind.SEARCH,
                depends_on=tuple(sub_question.get("depends_on") or ()),
                required=bool(sub_question.get("required", True)),
                arguments=arguments,
            )
        )
    if legacy_steps:
        answer_ids = tuple(step.step_id for step in legacy_steps if step.required)
        return ReasoningPlan(_plan_id(cleaned_query, legacy_steps, answer_ids), tuple(legacy_steps), answer_ids)

    coordinates = (
        _single_lookup_coordinates(cleaned_query, company_aliases=aliases)
        if fact_store_available
        else None
    )
    if coordinates is not None:
        structured, company, metric, period = coordinates
        step = _lookup_step(
            step_id="lookup_answer",
            company=company,
            metric=metric,
            year=period.year,
            period_kind=period.kind.value,
            structured_query=structured,
            query_text=cleaned_query,
        )
    else:
        step = ReasoningStep(
            step_id="search_answer",
            kind=ReasoningStepKind.SEARCH,
            arguments={"query": cleaned_query},
        )
    return ReasoningPlan(_plan_id(cleaned_query, (step,), (step.step_id,)), (step,), (step.step_id,))


def assess_reasoning_coverage(
    plan: ReasoningPlan | Mapping[str, Any],
    results: Iterable[StepResult | Mapping[str, Any]],
) -> dict[str, Any]:
    """Assess required-step and answer coverage without inferring missing work.

    A successful downstream result is ignored when any dependency is not
    successful.  Partial is reserved for an independently completed answer
    requirement; a missing core conclusion therefore abstains even when some
    prerequisite lookups succeeded.
    """
    normalized_plan = ReasoningPlan.from_mapping(plan)
    normalized_results = [StepResult.from_mapping(result) for result in results]
    by_id = {result.step_id: result for result in normalized_results}
    step_by_id = {step.step_id: step for step in normalized_plan.steps}

    completed = {
        step_id
        for step_id, result in by_id.items()
        if step_id in step_by_id and result.status is StepStatus.SUCCESS
    }
    blocked_by: dict[str, list[str]] = {}
    changed = True
    while changed:
        changed = False
        for step in normalized_plan.steps:
            if step.step_id not in completed:
                continue
            missing_dependencies = [dependency for dependency in step.depends_on if dependency not in completed]
            if missing_dependencies:
                completed.remove(step.step_id)
                blocked_by[step.step_id] = missing_dependencies
                changed = True

    for step in normalized_plan.steps:
        missing_dependencies = [dependency for dependency in step.depends_on if dependency not in completed]
        if missing_dependencies:
            blocked_by[step.step_id] = missing_dependencies

    required = [step.step_id for step in normalized_plan.steps if step.required]
    missing = [step_id for step_id in required if step_id not in completed]
    answered = [step_id for step_id in normalized_plan.answer_requirement_ids if step_id in completed]
    missing_answers = [
        step_id for step_id in normalized_plan.answer_requirement_ids if step_id not in completed
    ]
    if not missing and not missing_answers:
        decision = "complete"
    elif answered:
        decision = "partial"
    else:
        decision = "abstain"

    failures = {
        step_id: {
            "status": result.status.value,
            "error_type": result.error_type,
        }
        for step_id, result in by_id.items()
        if step_id in step_by_id and result.status is not StepStatus.SUCCESS
    }
    return {
        "plan_id": normalized_plan.plan_id,
        "required": required,
        "completed": [step.step_id for step in normalized_plan.steps if step.step_id in completed],
        "missing": missing,
        "answer_requirements": list(normalized_plan.answer_requirement_ids),
        "completed_answer_requirements": answered,
        "missing_answer_requirements": missing_answers,
        "blocked_by": blocked_by,
        "failures": failures,
        "coverage_ratio": (len(required) - len(missing)) / len(required) if required else 0.0,
        "decision": decision,
        "partial": decision == "partial",
        "abstain": decision == "abstain",
        "deterministic_conclusion_allowed": decision == "complete",
    }


__all__ = [
    "LegacySubQuestionAdapter",
    "ReasoningPlan",
    "ReasoningStep",
    "ReasoningStepKind",
    "StepResult",
    "StepStatus",
    "assess_reasoning_coverage",
    "build_reasoning_plan",
]
