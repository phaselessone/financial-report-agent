"""Public-interface tests for the unified single/multi-hop reasoning plan."""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from src.agent.config import AgentConfig
from src.agent.graph import run_agentic_rag
from src.agent.reasoning_plan import (
    LegacySubQuestionAdapter,
    ReasoningPlan,
    ReasoningStep,
    ReasoningStepKind,
    StepResult,
    StepStatus,
    assess_reasoning_coverage,
    build_reasoning_plan,
)
from src.agent.dependency_reasoning import SubQuestionResult
from src.llm.types import LLMResponse
from src.structured.fact_store import FactStore
from src.structured.schema import FinancialFact, Period
from tests.test_agent_flow import (
    FakeAnswerer,
    FakeLLM,
    FakeRuntime,
    make_result,
    make_row,
    supported_draft,
)


ALIASES = {
    "甲公司": ["甲公司", "甲"],
    "乙公司": ["乙公司", "乙"],
    "贵州茅台": ["贵州茅台", "茅台"],
    "宁德时代": ["宁德时代"],
}


def _revenue_fact(company: str, year: int, value: str, evidence_id: str) -> FinancialFact:
    return FinancialFact(
        company=company,
        metric="revenue",
        period=Period("FY", year),
        value_type="actual",
        value=Decimal(value),
        unit="元",
        doc_id=f"doc-{company}-{year}",
        page=1,
        evidence_id=evidence_id,
        raw_value=value,
        source_span=f"{company}{year}年营业收入为{value}元",
    )


def _coordinated_revenue_fact(
    *,
    value: str,
    evidence_id: str,
    value_type: str,
    period_basis: str,
    accounting_scope: str,
    revision_status: str,
) -> FinancialFact:
    return FinancialFact(
        company="甲公司",
        metric="revenue",
        period=Period("Q2", 2025),
        value_type=value_type,
        value=Decimal(value),
        unit="元",
        doc_id=f"doc-{evidence_id}",
        page=2,
        evidence_id=evidence_id,
        raw_value=value,
        source_span=f"甲公司2025年二季度营业收入为{value}元",
        accounting_scope=accounting_scope,
        period_basis=period_basis,
        revision_status=revision_status,
    )


def _calculation_fact(
    *,
    metric: str,
    period: Period,
    value: str,
    unit: str,
    evidence_id: str,
    period_basis: str | None = None,
) -> FinancialFact:
    return FinancialFact(
        company="甲公司",
        metric=metric,
        period=period,
        value_type="actual",
        value=Decimal(value),
        unit=unit,
        doc_id=f"doc-{evidence_id}",
        page=3,
        evidence_id=evidence_id,
        raw_value=f"{value}{unit}",
        source_span=f"甲公司{period.label}{metric}为{value}{unit}",
        period_basis=period_basis,
    )


def _run_calculation_graph(query: str, facts: list[FinancialFact]) -> dict:
    store = FactStore(":memory:")
    try:
        store.upsert(facts)
        return run_agentic_rag(
            runtime=FakeRuntime([]),
            answerer=FakeAnswerer([supported_draft("计算结果。")]),
            llm=FakeLLM([]),
            config=AgentConfig(strict_claim_verification=True, max_claim_retrievals=0),
            query=query,
            fact_store=store,
            company_aliases=ALIASES,
        )
    finally:
        store.close()


def _assert_strict_calculation_published(state: dict, formatted: str) -> None:
    calculation = next(iter(state["calculations"].values()))
    assert calculation["result"]["formatted"] == formatted
    assert state["final_answer"]["abstained"] is False
    assert state["final_answer"]["partial_answer"] is False
    assert len(state["final_answer"]["claims"]) == 1
    claim = state["final_answer"]["claims"][0]
    assert claim["claim_type"] == "DERIVED"
    assert claim["calculation_id"] == calculation["calculation_id"]
    assert claim["verification"]["status"] == "ENTAILED"
    assert claim["source_step_ids"]
    assert all(item["source_step_id"] for item in calculation["inputs"])


def test_simple_structured_fact_uses_one_lookup_step() -> None:
    plan = build_reasoning_plan(
        "贵州茅台2025年营业收入是多少？",
        fact_store_available=True,
        company_aliases=ALIASES,
    )

    assert len(plan.steps) == 1
    assert plan.steps[0].kind is ReasoningStepKind.LOOKUP
    assert plan.steps[0].depends_on == ()
    assert plan.answer_requirement_ids == (plan.steps[0].step_id,)


def test_structured_lookup_plan_preserves_full_financial_coordinates() -> None:
    plan = build_reasoning_plan(
        "甲公司2025年二季度单季合并口径重述后调整后营业收入是多少？",
        fact_store_available=True,
        company_aliases=ALIASES,
    )

    assert len(plan.steps) == 1
    arguments = plan.steps[0].arguments
    assert arguments["period"] == {"kind": "Q2", "year": 2025}
    assert arguments["value_type"] == "adjusted"
    assert arguments["period_basis"] == "standalone"
    assert arguments["accounting_scope"] == "consolidated"
    assert arguments["revision_status"] == "restated"


def test_graph_structured_lookup_uses_full_coordinates_to_select_one_fact() -> None:
    store = FactStore(":memory:")
    runtime = FakeRuntime([])
    try:
        store.upsert(
            [
                _coordinated_revenue_fact(
                    value="123",
                    evidence_id="E-match",
                    value_type="adjusted",
                    period_basis="standalone",
                    accounting_scope="consolidated",
                    revision_status="restated",
                ),
                _coordinated_revenue_fact(
                    value="100",
                    evidence_id="E-actual",
                    value_type="actual",
                    period_basis="standalone",
                    accounting_scope="consolidated",
                    revision_status="restated",
                ),
                _coordinated_revenue_fact(
                    value="110",
                    evidence_id="E-parent",
                    value_type="adjusted",
                    period_basis="standalone",
                    accounting_scope="parent",
                    revision_status="restated",
                ),
                _coordinated_revenue_fact(
                    value="115",
                    evidence_id="E-original",
                    value_type="adjusted",
                    period_basis="standalone",
                    accounting_scope="consolidated",
                    revision_status="original",
                ),
                _coordinated_revenue_fact(
                    value="220",
                    evidence_id="E-cumulative",
                    value_type="adjusted",
                    period_basis="cumulative",
                    accounting_scope="consolidated",
                    revision_status="restated",
                ),
            ]
        )
        state = run_agentic_rag(
            runtime=runtime,
            answerer=FakeAnswerer(
                [
                    supported_draft(
                        "甲公司2025年二季度单季合并口径重述后调整后营业收入为123元。"
                    )
                ]
            ),
            llm=FakeLLM([]),
            config=AgentConfig(strict_claim_verification=True, max_claim_retrievals=0),
            query="甲公司2025年二季度单季合并口径重述后调整后营业收入是多少？",
            fact_store=store,
            company_aliases=ALIASES,
        )
    finally:
        store.close()

    result = state["reasoning_step_results"][0]
    facts = state["reasoning_step_artifacts"]["lookup_answer"]["facts"]
    assert result["status"] == "SUCCESS"
    assert result["evidence_ids"] == ["E-match"]
    assert len(facts) == 1
    assert facts[0]["value"] == "123"
    assert facts[0]["value_type"] == "adjusted"
    assert runtime.calls == []
    assert state["final_answer"]["abstained"] is False
    assert state["final_answer"]["used_evidence_ids"] == ["E-match"]
    assert state["final_answer"]["claims"][0]["verification"]["status"] == "ENTAILED"


@pytest.mark.parametrize(
    ("query", "facts"),
    [
        (
            "甲公司2025年二季度单季重述后调整后营业收入是多少？",
            [
                _coordinated_revenue_fact(
                    value="123",
                    evidence_id="E-consolidated",
                    value_type="adjusted",
                    period_basis="standalone",
                    accounting_scope="consolidated",
                    revision_status="restated",
                ),
                _coordinated_revenue_fact(
                    value="456",
                    evidence_id="E-parent",
                    value_type="adjusted",
                    period_basis="standalone",
                    accounting_scope="parent",
                    revision_status="restated",
                ),
            ],
        ),
        (
            "甲公司2025年二季度单季合并口径调整后营业收入是多少？",
            [
                _coordinated_revenue_fact(
                    value="123",
                    evidence_id="E-original",
                    value_type="adjusted",
                    period_basis="standalone",
                    accounting_scope="consolidated",
                    revision_status="original",
                ),
                _coordinated_revenue_fact(
                    value="456",
                    evidence_id="E-restated",
                    value_type="adjusted",
                    period_basis="standalone",
                    accounting_scope="consolidated",
                    revision_status="restated",
                ),
            ],
        ),
    ],
    ids=["missing-accounting-scope", "missing-revision-status"],
)
def test_strict_graph_fails_closed_when_conflicting_facts_lack_a_required_coordinate(
    query: str,
    facts: list[FinancialFact],
) -> None:
    store = FactStore(":memory:")
    runtime = FakeRuntime([])
    answerer = FakeAnswerer([supported_draft("不应生成")])
    try:
        store.upsert(facts)
        state = run_agentic_rag(
            runtime=runtime,
            answerer=answerer,
            llm=FakeLLM([]),
            config=AgentConfig(strict_claim_verification=True, max_claim_retrievals=0),
            query=query,
            fact_store=store,
            company_aliases=ALIASES,
        )
    finally:
        store.close()

    result = state["reasoning_step_results"][0]
    assert result["status"] == "INSUFFICIENT"
    assert result["error_type"] == "AMBIGUOUS_FACT"
    assert state["final_answer"]["abstained"] is True
    assert state["final_answer"]["used_evidence_ids"] == []
    assert state["final_answer"]["citations"] == []
    assert all(
        claim["verification"]["status"] != "ENTAILED"
        for claim in state["final_answer"]["claims"]
    )
    assert answerer.answer_calls == []
    assert runtime.calls == []


def test_plain_q2_graph_fails_closed_before_report_search() -> None:
    store = FactStore(":memory:")
    runtime = FakeRuntime([])
    try:
        store.upsert(
            [
                _coordinated_revenue_fact(
                    value="100",
                    evidence_id="E-standalone",
                    value_type="actual",
                    period_basis="standalone",
                    accounting_scope="consolidated",
                    revision_status="original",
                ),
                _coordinated_revenue_fact(
                    value="200",
                    evidence_id="E-cumulative",
                    value_type="actual",
                    period_basis="cumulative",
                    accounting_scope="consolidated",
                    revision_status="original",
                ),
            ]
        )
        state = run_agentic_rag(
            runtime=runtime,
            answerer=FakeAnswerer([]),
            llm=FakeLLM([]),
            config=AgentConfig(max_claim_retrievals=0),
            query="甲公司2025年二季度营业收入是多少？",
            fact_store=store,
            company_aliases=ALIASES,
        )
    finally:
        store.close()

    result = state["reasoning_step_results"][0]
    assert result["status"] == "INSUFFICIENT"
    assert result["error_type"] == "AMBIGUOUS_PERIOD_BASIS"
    assert runtime.calls == []
    assert state["final_answer"]["abstained"] is True


def test_cagr_query_runs_two_lookups_then_deterministic_calculation() -> None:
    state = _run_calculation_graph(
        "甲公司2022年至2025年营业收入复合年增长率是多少？",
        [
            _calculation_fact(
                metric="revenue",
                period=Period("FY", 2022),
                value="100",
                unit="元",
                evidence_id="E-2022",
            ),
            _calculation_fact(
                metric="revenue",
                period=Period("FY", 2025),
                value="133.1",
                unit="元",
                evidence_id="E-2025",
            ),
        ],
    )

    assert [step["kind"] for step in state["reasoning_plan"]["steps"]] == [
        "LOOKUP",
        "LOOKUP",
        "CALCULATE",
    ]
    calculation = next(iter(state["calculations"].values()))
    assert calculation["operation"] == "cagr"
    assert calculation["status"] == "SUCCESS"
    assert calculation["result"]["formatted"] == "10.0000%"
    assert calculation["years"] == "3"
    _assert_strict_calculation_published(state, "10.0000%")


def test_strict_negative_yoy_publishes_verified_calculation_claim() -> None:
    state = _run_calculation_graph(
        "甲公司2025年营业收入同比增长多少？",
        [
            _calculation_fact(
                metric="revenue",
                period=Period("FY", 2024),
                value="200",
                unit="元",
                evidence_id="E-2024",
            ),
            _calculation_fact(
                metric="revenue",
                period=Period("FY", 2025),
                value="170",
                unit="元",
                evidence_id="E-2025",
            ),
        ],
    )

    _assert_strict_calculation_published(state, "-15.0000%")


def test_strict_yoy_normalizes_currency_units_end_to_end() -> None:
    store = FactStore(":memory:")
    try:
        state = run_agentic_rag(
            runtime=FakeRuntime(
                [
                    make_result(
                        [
                            make_row(
                                chunk_id="E-2024",
                                doc_id="annual-report",
                                text="甲公司2024年营业收入1亿元。",
                            )
                        ]
                    ),
                    make_result(
                        [
                            make_row(
                                chunk_id="E-2025",
                                doc_id="annual-report",
                                text="甲公司2025年营业收入12000万元。",
                            )
                        ]
                    ),
                ]
            ),
            answerer=FakeAnswerer([supported_draft("计算结果。")]),
            llm=FakeLLM([]),
            config=AgentConfig(strict_claim_verification=True, max_claim_retrievals=0),
            query="甲公司2025年营业收入同比增长多少？",
            fact_store=store,
            company_aliases=ALIASES,
        )
    finally:
        store.close()

    _assert_strict_calculation_published(state, "20.0000%")


def test_strict_yoy_zero_base_abstains_without_a_derived_claim() -> None:
    state = _run_calculation_graph(
        "甲公司2025年营业收入同比增长多少？",
        [
            _calculation_fact(
                metric="revenue",
                period=Period("FY", 2024),
                value="0",
                unit="元",
                evidence_id="E-2024",
            ),
            _calculation_fact(
                metric="revenue",
                period=Period("FY", 2025),
                value="10",
                unit="元",
                evidence_id="E-2025",
            ),
        ],
    )

    calculation = next(iter(state["calculations"].values()))
    assert calculation["status"] != "SUCCESS"
    assert calculation["result"] is None
    assert state["final_answer"]["abstained"] is True
    assert state["claims"] == []


def test_qoq_query_uses_prior_quarter_not_prior_year() -> None:
    state = _run_calculation_graph(
        "甲公司2025年二季度单季营业收入环比增长多少？",
        [
            _calculation_fact(
                metric="revenue",
                period=Period("Q1", 2025),
                value="100",
                unit="元",
                evidence_id="E-Q1",
                period_basis="standalone",
            ),
            _calculation_fact(
                metric="revenue",
                period=Period("Q2", 2025),
                value="120",
                unit="元",
                evidence_id="E-Q2",
                period_basis="standalone",
            ),
        ],
    )

    lookup_periods = [
        step["arguments"]["period"]
        for step in state["reasoning_plan"]["steps"]
        if step["kind"] == "LOOKUP"
    ]
    assert lookup_periods == [
        {"kind": "Q1", "year": 2025},
        {"kind": "Q2", "year": 2025},
    ]
    calculation = next(iter(state["calculations"].values()))
    assert calculation["operation"] == "qoq"
    assert calculation["result"]["formatted"] == "20.0000%"


def test_ratio_query_uses_two_metric_fact_dependencies() -> None:
    state = _run_calculation_graph(
        "甲公司2025年研发费用占营业收入比例是多少？",
        [
            _calculation_fact(
                metric="rd_expense",
                period=Period("FY", 2025),
                value="10",
                unit="元",
                evidence_id="E-RD",
            ),
            _calculation_fact(
                metric="revenue",
                period=Period("FY", 2025),
                value="100",
                unit="元",
                evidence_id="E-REV",
            ),
        ],
    )

    calculation = next(iter(state["calculations"].values()))
    assert calculation["operation"] == "ratio"
    assert calculation["result"]["formatted"] == "10.0000%"
    assert {item["evidence_id"] for item in calculation["inputs"]} == {"E-RD", "E-REV"}


def test_percentage_point_change_and_formula_margin_run_in_main_graph() -> None:
    pp_state = _run_calculation_graph(
        "甲公司2024年至2025年毛利率变化了多少个百分点？",
        [
            _calculation_fact(
                metric="gross_margin",
                period=Period("FY", 2024),
                value="20",
                unit="%",
                evidence_id="E-M24",
            ),
            _calculation_fact(
                metric="gross_margin",
                period=Period("FY", 2025),
                value="25",
                unit="%",
                evidence_id="E-M25",
            ),
        ],
    )
    pp_calculation = next(iter(pp_state["calculations"].values()))
    assert pp_calculation["operation"] == "percentage_point_change"
    assert pp_calculation["result"]["formatted"] == "5.0000pp"
    _assert_strict_calculation_published(pp_state, "5.0000pp")

    margin_state = _run_calculation_graph(
        "甲公司2025年毛利/营业收入是多少？",
        [
            _calculation_fact(
                metric="gross_profit",
                period=Period("FY", 2025),
                value="20",
                unit="元",
                evidence_id="E-GP",
            ),
            _calculation_fact(
                metric="revenue",
                period=Period("FY", 2025),
                value="100",
                unit="元",
                evidence_id="E-R",
            ),
        ],
    )
    margin_calculation = next(iter(margin_state["calculations"].values()))
    assert margin_calculation["operation"] == "gross_margin"
    assert margin_calculation["result"]["formatted"] == "20.0000%"


def test_single_hop_graph_executes_the_public_reasoning_step_and_artifact() -> None:
    runtime = FakeRuntime(
        [make_result([make_row(chunk_id="c1", doc_id="d1", text="半导体行业景气度持续回升。")])]
    )
    state = run_agentic_rag(
        runtime=runtime,
        answerer=FakeAnswerer([supported_draft("半导体行业景气度持续回升。")]),
        llm=FakeLLM([]),
        config=AgentConfig(max_claim_retrievals=0, strict_claim_verification=False),
        query="半导体行业景气度如何？",
    )

    assert [step["kind"] for step in state["reasoning_plan"]["steps"]] == ["SEARCH"]
    assert state["reasoning_step_results"][0]["status"] == "SUCCESS"
    assert state["reasoning_step_artifacts"]["search_answer"]["source"] == "report_search"
    assert state["reasoning_step_artifacts"]["search_answer"]["rows"][0]["chunk_id"] == "c1"
    assert state["reasoning_coverage"]["decision"] == "complete"
    assert state["final_answer"]["final_answer"] == "半导体行业景气度持续回升。"


def test_company_growth_comparison_builds_four_lookups_two_yoy_and_compare() -> None:
    plan = build_reasoning_plan(
        "甲公司和乙公司2025年营业收入谁同比增长更快？",
        fact_store_available=True,
        company_aliases=ALIASES,
    )

    lookups = [step for step in plan.steps if step.kind is ReasoningStepKind.LOOKUP]
    calculations = [step for step in plan.steps if step.kind is ReasoningStepKind.CALCULATE]
    comparisons = [step for step in plan.steps if step.kind is ReasoningStepKind.COMPARE]

    assert len(lookups) == 4
    assert {(step.arguments["company"], step.arguments["period"]["year"]) for step in lookups} == {
        ("甲公司", 2024),
        ("甲公司", 2025),
        ("乙公司", 2024),
        ("乙公司", 2025),
    }
    assert len(calculations) == 2
    assert all(step.arguments["operation"] == "yoy" for step in calculations)
    assert all(len(step.depends_on) == 2 for step in calculations)
    assert len(comparisons) == 1
    assert set(comparisons[0].depends_on) == {step.step_id for step in calculations}
    assert plan.answer_requirement_ids == (comparisons[0].step_id,)


def test_legacy_subquestion_result_roundtrips_through_step_result() -> None:
    legacy = SubQuestionResult(
        id="q1",
        status="completed",
        evidence_ids=("E1",),
        facts=({"fact_id": "F1", "value": "120"},),
    )

    current = LegacySubQuestionAdapter.from_legacy(legacy)
    restored = LegacySubQuestionAdapter.to_legacy(current, facts=legacy.facts)

    assert current.status is StepStatus.SUCCESS
    assert current.fact_ids == ("F1",)
    assert current.evidence_ids == ("E1",)
    assert restored.id == "q1"
    assert restored.completed
    assert restored.facts == legacy.facts


def test_independent_answer_requirement_can_be_returned_as_partial() -> None:
    plan = ReasoningPlan(
        plan_id="PLAN-independent",
        steps=(
            ReasoningStep("revenue", ReasoningStepKind.LOOKUP),
            ReasoningStep("outlook", ReasoningStepKind.SEARCH),
        ),
        answer_requirement_ids=("revenue", "outlook"),
    )
    report = assess_reasoning_coverage(
        plan,
        (
            StepResult("revenue", StepStatus.SUCCESS, fact_ids=("F1",), evidence_ids=("E1",)),
            StepResult("outlook", StepStatus.INSUFFICIENT, error_type="NO_EVIDENCE"),
        ),
    )

    assert report["decision"] == "partial"
    assert report["completed_answer_requirements"] == ["revenue"]
    assert report["missing"] == ["outlook"]
    assert not report["deterministic_conclusion_allowed"]


def test_multi_hop_graph_executes_every_subquestion_through_reasoning_results() -> None:
    query = "宁德时代2025年营业收入是多少？市场对其增长逻辑怎么看？"
    llm = FakeLLM(
        [
            LLMResponse(
                content=json.dumps(
                    {
                        "sub_questions": [
                            {"id": "revenue", "query": "宁德时代2025年营业收入是多少"},
                            {"id": "outlook", "query": "市场对宁德时代增长逻辑的看法"},
                        ]
                    },
                    ensure_ascii=False,
                ),
                provider="fake",
                model="fake",
            )
        ]
    )
    state = run_agentic_rag(
        runtime=FakeRuntime(
            [
                make_result([make_row(chunk_id="c1", doc_id="d1", text="宁德时代营收1234亿元。")]),
                make_result([make_row(chunk_id="c2", doc_id="d2", text="机构看好其增长逻辑。")]),
            ]
        ),
        answerer=FakeAnswerer([supported_draft("宁德时代营收1234亿元，机构看好增长逻辑。")]),
        llm=llm,
        config=AgentConfig(),
        query=query,
    )

    assert [step["step_id"] for step in state["reasoning_plan"]["steps"]] == ["revenue", "outlook"]
    assert [result["status"] for result in state["reasoning_step_results"]] == ["SUCCESS", "SUCCESS"]
    assert [result["id"] for result in state["sub_question_results"]] == ["revenue", "outlook"]
    assert state["reasoning_coverage"]["decision"] == "complete"


def test_growth_comparison_lookup_steps_retain_fact_and_evidence_provenance() -> None:
    store = FactStore(":memory:")
    try:
        store.upsert(
            [
                _revenue_fact("甲公司", 2024, "100", "E-A-2024"),
                _revenue_fact("甲公司", 2025, "120", "E-A-2025"),
                _revenue_fact("乙公司", 2024, "200", "E-B-2024"),
                _revenue_fact("乙公司", 2025, "210", "E-B-2025"),
            ]
        )
        state = run_agentic_rag(
            runtime=FakeRuntime([]),
            answerer=FakeAnswerer([]),
            llm=FakeLLM([]),
            config=AgentConfig(),
            query="甲公司和乙公司2025年营业收入谁同比增长更快？",
            fact_store=store,
            company_aliases=ALIASES,
        )
    finally:
        store.close()

    lookup_results = state["reasoning_step_results"][:4]
    assert [result["status"] for result in lookup_results] == ["SUCCESS"] * 4
    assert {result["evidence_ids"][0] for result in lookup_results} == {
        "E-A-2024",
        "E-A-2025",
        "E-B-2024",
        "E-B-2025",
    }
    assert all(result["fact_ids"] for result in lookup_results)


def test_growth_comparison_calculates_both_yoy_results_from_lookup_dependencies() -> None:
    store = FactStore(":memory:")
    try:
        store.upsert(
            [
                _revenue_fact("甲公司", 2024, "100", "E-A-2024"),
                _revenue_fact("甲公司", 2025, "120", "E-A-2025"),
                _revenue_fact("乙公司", 2024, "200", "E-B-2024"),
                _revenue_fact("乙公司", 2025, "210", "E-B-2025"),
            ]
        )
        state = run_agentic_rag(
            runtime=FakeRuntime([]),
            answerer=FakeAnswerer([]),
            llm=FakeLLM([]),
            config=AgentConfig(),
            query="甲公司和乙公司2025年营业收入谁同比增长更快？",
            fact_store=store,
            company_aliases=ALIASES,
        )
    finally:
        store.close()

    calculation_results = state["reasoning_step_results"][4:6]
    assert [result["status"] for result in calculation_results] == ["SUCCESS", "SUCCESS"]
    assert all(result["calculation_id"] in state["calculations"] for result in calculation_results)
    calculations = [state["calculations"][result["calculation_id"]] for result in calculation_results]
    assert [calculation["operation"] for calculation in calculations] == ["yoy", "yoy"]
    assert {calculation["result"]["formatted"] for calculation in calculations} == {
        "20.0000%",
        "5.0000%",
    }


def test_growth_comparison_requires_successful_compare_for_core_coverage() -> None:
    store = FactStore(":memory:")
    try:
        store.upsert(
            [
                _revenue_fact("甲公司", 2024, "100", "E-A-2024"),
                _revenue_fact("甲公司", 2025, "120", "E-A-2025"),
                _revenue_fact("乙公司", 2024, "200", "E-B-2024"),
                _revenue_fact("乙公司", 2025, "210", "E-B-2025"),
            ]
        )
        state = run_agentic_rag(
            runtime=FakeRuntime([]),
            answerer=FakeAnswerer([supported_draft("甲公司同比增速更快。")]),
            llm=FakeLLM([]),
            config=AgentConfig(),
            query="甲公司和乙公司2025年营业收入谁同比增长更快？",
            fact_store=store,
            company_aliases=ALIASES,
        )
    finally:
        store.close()

    compare_result = state["reasoning_step_results"][-1]
    assert compare_result["step_id"] == "compare_growth"
    assert compare_result["status"] == "SUCCESS"
    assert compare_result["calculation_id"] in state["calculations"]
    comparison_record = state["calculations"][compare_result["calculation_id"]]
    parent_calculation_ids = {
        result["calculation_id"]
        for result in state["reasoning_step_results"]
        if result["step_id"].startswith("calculate_")
    }
    assert set(comparison_record["input_calculation_ids"]) == parent_calculation_ids
    assert set(comparison_record["source_step_ids"]) == {
        "calculate_1_yoy",
        "calculate_2_yoy",
    }
    assert all(item["fact_id"] is None for item in comparison_record["inputs"])
    assert all(item["upstream_calculation_id"] in parent_calculation_ids for item in comparison_record["inputs"])
    assert set(comparison_record["evidence_ids"]) == {
        "E-A-2024",
        "E-A-2025",
        "E-B-2024",
        "E-B-2025",
    }
    assert comparison_record["evidence_ids"] == [
        "E-A-2024",
        "E-A-2025",
        "E-B-2024",
        "E-B-2025",
    ]
    assert comparison_record["comparison"]["winner"] == "甲公司"
    assert comparison_record["comparison"]["relation"] == "greater"
    assert state["reasoning_conclusion"]["winner"] == "甲公司"
    assert state["reasoning_conclusion"]["relation"] == "greater"
    assert state["reasoning_coverage"]["decision"] == "complete"
    assert state["reasoning_coverage"]["deterministic_conclusion_allowed"]


def test_missing_growth_operand_blocks_core_conclusion_and_abstains() -> None:
    store = FactStore(":memory:")
    try:
        store.upsert(
            [
                _revenue_fact("甲公司", 2024, "100", "E-A-2024"),
                _revenue_fact("甲公司", 2025, "120", "E-A-2025"),
                # 乙公司 2024 is deliberately absent.
                _revenue_fact("乙公司", 2025, "210", "E-B-2025"),
            ]
        )
        answerer = FakeAnswerer([supported_draft("不应生成")])
        state = run_agentic_rag(
            runtime=FakeRuntime([]),
            answerer=answerer,
            llm=FakeLLM([]),
            config=AgentConfig(),
            query="甲公司和乙公司2025年营业收入谁同比增长更快？",
            fact_store=store,
            company_aliases=ALIASES,
        )
    finally:
        store.close()

    assert state["reasoning_coverage"]["decision"] == "abstain"
    assert state["reasoning_coverage"]["missing_answer_requirements"] == ["compare_growth"]
    assert state["reasoning_coverage"]["blocked_by"]["compare_growth"] == ["calculate_2_yoy"]
    assert state["termination_reason"] == "abstain_core_dependency_missing"
    assert state["final_answer"]["abstained"]
    assert answerer.answer_calls == []


def test_independent_multi_hop_success_is_returned_as_verified_partial_answer() -> None:
    query = "宁德时代2025年营业收入是多少？市场对其增长逻辑怎么看？"
    llm = FakeLLM(
        [
            LLMResponse(
                content=json.dumps(
                    {
                        "sub_questions": [
                            {"id": "revenue", "query": "宁德时代2025年营业收入是多少"},
                            {"id": "outlook", "query": "市场对宁德时代增长逻辑的看法"},
                        ]
                    },
                    ensure_ascii=False,
                ),
                provider="fake",
                model="fake",
            )
        ]
    )
    answerer = FakeAnswerer([supported_draft("宁德时代2025年营业收入1234亿元。")])
    state = run_agentic_rag(
        runtime=FakeRuntime(
            [
                make_result(
                    [make_row(chunk_id="c1", doc_id="d1", text="宁德时代2025年营业收入1234亿元。")]
                ),
                make_result([]),
            ]
        ),
        answerer=answerer,
        llm=llm,
        config=AgentConfig(),
        query=query,
    )

    assert state["reasoning_coverage"]["decision"] == "partial"
    assert state["reasoning_coverage"]["completed_answer_requirements"] == ["revenue"]
    assert state["reasoning_coverage"]["missing_answer_requirements"] == ["outlook"]
    assert state["final_answer"]["partial_answer"]
    assert not state["final_answer"]["abstained"]
    assert state["final_answer"]["final_answer"] == "宁德时代2025年营业收入1234亿元。"
    assert state["final_answer"]["used_evidence_ids"] == ["c1"]
    assert len(state["final_answer"]["citations"]) == 1
    assert state["final_answer"]["citations"][0]["doc_id"] == "d1"
    assert state["final_answer"]["citations"][0]["page_start"] == 1
    assert state["final_answer"]["claims"]
    assert {
        claim["verification"]["status"]
        for claim in state["final_answer"]["claims"]
    } == {"ENTAILED"}
    assert {
        evidence_id
        for claim in state["final_answer"]["claims"]
        for evidence_id in claim["evidence_ids"]
    } == {"c1"}
    assert "增长逻辑" not in state["final_answer"]["final_answer"]
    assert len(answerer.answer_calls) == 1


def test_multi_hop_lookup_miss_uses_one_bounded_report_search_fallback() -> None:
    query = "宁德时代2025年营业收入是多少？市场对其增长逻辑怎么看？"
    llm = FakeLLM(
        [
            LLMResponse(
                content=json.dumps(
                    {
                        "sub_questions": [
                            {"id": "revenue", "query": "宁德时代2025年营业收入是多少"},
                            {"id": "outlook", "query": "市场对宁德时代增长逻辑的看法"},
                        ]
                    },
                    ensure_ascii=False,
                ),
                provider="fake",
                model="fake",
            )
        ]
    )
    runtime = FakeRuntime(
        [
            make_result([make_row(chunk_id="c1", doc_id="d1", text="宁德时代营业收入1234亿元。")]),
            make_result([make_row(chunk_id="c2", doc_id="d2", text="机构看好其增长逻辑。")]),
        ]
    )
    store = FactStore(":memory:")
    try:
        state = run_agentic_rag(
            runtime=runtime,
            answerer=FakeAnswerer([supported_draft("宁德时代营业收入1234亿元，机构看好其增长逻辑。")]),
            llm=llm,
            config=AgentConfig(),
            query=query,
            fact_store=store,
            company_aliases=ALIASES,
        )
    finally:
        store.close()

    assert [result["status"] for result in state["reasoning_step_results"]] == ["SUCCESS", "SUCCESS"]
    assert runtime.calls == ["宁德时代2025年营业收入是多少", "市场对宁德时代增长逻辑的看法"]
    assert [call["tool_name"] for call in state["tool_calls"][:2]] == [
        "structured_lookup",
        "report_search",
    ]
    assert state["reasoning_coverage"]["decision"] == "complete"


def test_growth_operands_resolve_from_one_report_search_per_lookup() -> None:
    runtime = FakeRuntime(
        [
            make_result(
                [make_row(chunk_id="E-A-2024", doc_id="d-a", text="甲公司2024年营业收入100元。")]
            ),
            make_result(
                [make_row(chunk_id="E-A-2025", doc_id="d-a", text="甲公司2025年营业收入120元。")]
            ),
            make_result(
                [make_row(chunk_id="E-B-2024", doc_id="d-b", text="乙公司2024年营业收入200元。")]
            ),
            make_result(
                [make_row(chunk_id="E-B-2025", doc_id="d-b", text="乙公司2025年营业收入210元。")]
            ),
        ]
    )
    store = FactStore(":memory:")
    try:
        state = run_agentic_rag(
            runtime=runtime,
            answerer=FakeAnswerer([supported_draft("乙公司增长更快。")]),
            llm=FakeLLM([]),
            config=AgentConfig(
                max_claim_retrievals=0,
                strict_claim_verification=False,
            ),
            query="甲公司和乙公司2025年营业收入谁同比增长更快？",
            fact_store=store,
            company_aliases=ALIASES,
        )
    finally:
        store.close()

    assert runtime.calls == [
        "甲公司2024年营业收入是多少？",
        "甲公司2025年营业收入是多少？",
        "乙公司2024年营业收入是多少？",
        "乙公司2025年营业收入是多少？",
    ]
    assert [result["status"] for result in state["reasoning_step_results"]] == [
        "SUCCESS",
        "SUCCESS",
        "SUCCESS",
        "SUCCESS",
        "SUCCESS",
        "SUCCESS",
        "SUCCESS",
    ]
    assert state["reasoning_step_artifacts"]["lookup_1_prior"]["facts"][0]["value"] == "100"
    assert state["reasoning_step_artifacts"]["lookup_2_current"]["facts"][0]["evidence_id"] == "E-B-2025"
    assert state["reasoning_conclusion"]["winner"] == "甲公司"
    assert state["reasoning_coverage"]["decision"] == "complete"


def test_growth_operands_reuse_all_matching_values_from_existing_evidence_pool() -> None:
    all_operands = [
        make_row(chunk_id="E-A-2024", doc_id="d-a", text="甲公司2024年营业收入100元。"),
        make_row(chunk_id="E-A-2025", doc_id="d-a", text="甲公司2025年营业收入120元。"),
        make_row(chunk_id="E-B-2024", doc_id="d-b", text="乙公司2024年营业收入200元。"),
        make_row(chunk_id="E-B-2025", doc_id="d-b", text="乙公司2025年营业收入210元。"),
    ]
    runtime = FakeRuntime([make_result(all_operands)])
    store = FactStore(":memory:")
    try:
        state = run_agentic_rag(
            runtime=runtime,
            answerer=FakeAnswerer([supported_draft("甲公司增长更快。")]),
            llm=FakeLLM([]),
            config=AgentConfig(
                max_claim_retrievals=0,
                strict_claim_verification=False,
            ),
            query="甲公司和乙公司2025年营业收入谁同比增长更快？",
            fact_store=store,
            company_aliases=ALIASES,
        )
    finally:
        store.close()

    assert runtime.calls == ["甲公司2024年营业收入是多少？"]
    assert state["retrieval_count"] == 1
    assert set(state["evidence_pool"]) == {
        "E-A-2024",
        "E-A-2025",
        "E-B-2024",
        "E-B-2025",
    }
    assert [result["status"] for result in state["reasoning_step_results"]] == [
        "SUCCESS",
        "SUCCESS",
        "SUCCESS",
        "SUCCESS",
        "SUCCESS",
        "SUCCESS",
        "SUCCESS",
    ]
    assert state["reasoning_step_artifacts"]["lookup_1_current"]["source"] == "evidence_pool"
    assert state["reasoning_step_artifacts"]["lookup_2_current"]["facts"][0]["evidence_id"] == "E-B-2025"
    assert state["reasoning_coverage"]["decision"] == "complete"


def test_ambiguous_report_operands_fail_closed_without_extra_retrieval() -> None:
    runtime = FakeRuntime(
        [
            make_result(
                [
                    make_row(chunk_id="E-A-2024-1", doc_id="d-a", text="甲公司2024年营业收入100元。"),
                    make_row(chunk_id="E-A-2024-2", doc_id="d-a", text="甲公司2024年营业收入101元。"),
                ]
            ),
            make_result(
                [make_row(chunk_id="E-A-2025", doc_id="d-a", text="甲公司2025年营业收入120元。")]
            ),
            make_result(
                [make_row(chunk_id="E-B-2024", doc_id="d-b", text="乙公司2024年营业收入200元。")]
            ),
            make_result(
                [make_row(chunk_id="E-B-2025", doc_id="d-b", text="乙公司2025年营业收入210元。")]
            ),
        ]
    )
    store = FactStore(":memory:")
    try:
        answerer = FakeAnswerer([supported_draft("不应生成")])
        state = run_agentic_rag(
            runtime=runtime,
            answerer=answerer,
            llm=FakeLLM([]),
            config=AgentConfig(max_tool_calls=12, strict_claim_verification=False),
            query="甲公司和乙公司2025年营业收入谁同比增长更快？",
            fact_store=store,
            company_aliases=ALIASES,
        )
    finally:
        store.close()

    first = state["reasoning_step_results"][0]
    assert first["status"] == "INSUFFICIENT"
    assert first["error_type"] == "AMBIGUOUS_OPERAND"
    assert state["reasoning_step_artifacts"]["lookup_1_prior"]["candidate_ids"] == [
        "E-A-2024-1",
        "E-A-2024-2",
    ]
    assert len(runtime.calls) == 4
    assert state["reasoning_coverage"]["decision"] == "abstain"
    assert state["termination_reason"] == "abstain_core_dependency_missing"
    assert answerer.answer_calls == []


def test_deterministic_compare_conclusion_overrides_conflicting_model_draft() -> None:
    store = FactStore(":memory:")
    try:
        store.upsert(
            [
                _revenue_fact("甲公司", 2024, "100", "E-A-2024"),
                _revenue_fact("甲公司", 2025, "120", "E-A-2025"),
                _revenue_fact("乙公司", 2024, "200", "E-B-2024"),
                _revenue_fact("乙公司", 2025, "210", "E-B-2025"),
            ]
        )
        state = run_agentic_rag(
            runtime=FakeRuntime([]),
            answerer=FakeAnswerer([supported_draft("乙公司增长更快。")]),
            llm=FakeLLM([]),
            config=AgentConfig(),
            query="甲公司和乙公司2025年营业收入谁同比增长更快？",
            fact_store=store,
            company_aliases=ALIASES,
        )
    finally:
        store.close()

    assert state["draft_answer"]["final_answer"] == "甲公司增长更快。"
    assert state["draft_answer"]["reasoning_conclusion"]["winner"] == "甲公司"
    assert set(state["draft_answer"]["used_evidence_ids"]) == {
        "E-A-2024",
        "E-A-2025",
        "E-B-2024",
        "E-B-2025",
    }
    final_answer = state["final_answer"]
    assert final_answer["final_answer"] == "甲公司增长更快。"
    assert not final_answer["abstained"]
    assert final_answer["support_validation"] == {
        "supported": True,
        "reasoning_gate": "deterministic_compare",
        "claim_gate": "all_entailed",
    }
    assert set(final_answer["used_evidence_ids"]) == {
        "E-A-2024",
        "E-A-2025",
        "E-B-2024",
        "E-B-2025",
    }

    claims = state["claims"]
    assert len(claims) == 3
    assert all(claim["verification"]["status"] == "ENTAILED" for claim in claims)
    core_claim = next(claim for claim in claims if claim["is_core"])
    compare_result = next(
        result for result in state["reasoning_step_results"] if result["step_id"] == "compare_growth"
    )
    assert core_claim["claim_type"] == "SYNTHESIZED"
    assert core_claim["calculation_id"] == compare_result["calculation_id"]
    assert core_claim["source_step_ids"] == ["compare_growth"]
    assert set(core_claim["evidence_ids"]) == {
        "E-A-2024",
        "E-A-2025",
        "E-B-2024",
        "E-B-2025",
    }
