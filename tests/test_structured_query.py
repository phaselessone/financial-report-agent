from __future__ import annotations

import unittest
from decimal import Decimal

from src.structured.fact_store import FactStore
from src.structured.schema import FinancialFact, Metric, Period, PeriodBasis, ValueType
from src.structured.structured_query import (
    build_structured_query,
    execute_structured_query,
    parse_structured_query,
)

ALIASES = {"贵州茅台": ["贵州茅台", "茅台"], "五粮液": ["五粮液"]}


def make_fact(
    company: str = "贵州茅台",
    metric: str = "revenue",
    year: int = 2025,
    value: str = "123450000000",
    value_type: str = "actual",
    unit: str = "元",
    **overrides,
) -> FinancialFact:
    base = dict(
        company=company,
        metric=metric,
        period=Period(kind="FY", year=year),
        value_type=value_type,
        value=Decimal(value),
        unit=unit,
        doc_id=f"doc-{company}-{year}",
        page=3,
        evidence_id=f"E-{company}-{year}-{metric}",
        raw_value="1234.5亿元" if unit == "元" else "25%",
        source_span="2025年公司实现营业收入1234.5亿元" if unit == "元" else "2025年毛利率25%",
    )
    base.update(overrides)
    return FinancialFact(**base)


class TestParseStructuredQuery(unittest.TestCase):
    def test_parse_lookup(self) -> None:
        query = parse_structured_query("贵州茅台2025年营业收入是多少", company_aliases=ALIASES)
        self.assertIsNotNone(query)
        assert query is not None
        self.assertEqual(query.operation, "lookup")
        self.assertEqual(query.entities, ("贵州茅台",))
        self.assertEqual(query.metrics, (Metric.REVENUE,))
        self.assertEqual(query.value_type, ValueType.ACTUAL)

    def test_parse_compare(self) -> None:
        query = parse_structured_query("比较贵州茅台和五粮液2025年营业收入", company_aliases=ALIASES)
        self.assertIsNotNone(query)
        assert query is not None
        self.assertEqual(query.operation, "compare")
        self.assertEqual(query.entities, ("贵州茅台", "五粮液"))

    def test_parse_trend_relative_range(self) -> None:
        query = parse_structured_query("贵州茅台近三年营业收入趋势", company_aliases=ALIASES, anchor_year=2025)
        self.assertIsNotNone(query)
        assert query is not None
        self.assertEqual(query.operation, "trend")
        self.assertEqual(query.periods, ((Period("FY", 2023), Period("FY", 2025)),))

    def test_parse_forecast_isolated(self) -> None:
        query = parse_structured_query("预计贵州茅台2025年营业收入", company_aliases=ALIASES)
        self.assertIsNotNone(query)
        assert query is not None
        self.assertEqual(query.value_type, ValueType.FORECAST)

    def test_parse_adjusted_isolated(self) -> None:
        query = parse_structured_query("贵州茅台2025年经调整营业收入", company_aliases=ALIASES)
        self.assertIsNotNone(query)
        assert query is not None
        self.assertEqual(query.value_type, ValueType.ADJUSTED)

    def test_parse_q2_standalone_basis(self) -> None:
        query = parse_structured_query("贵州茅台2025年二季度单季营业收入", company_aliases=ALIASES)
        self.assertIsNotNone(query)
        assert query is not None
        self.assertEqual(query.periods, (Period("Q2", 2025),))
        self.assertEqual(query.period_basis, PeriodBasis.STANDALONE)

    def test_parse_english_q2_standalone_basis(self) -> None:
        query = parse_structured_query("贵州茅台2025Q2单季营业收入", company_aliases=ALIASES)
        self.assertIsNotNone(query)
        assert query is not None
        self.assertEqual(query.periods, (Period("Q2", 2025),))
        self.assertEqual(query.period_basis, PeriodBasis.STANDALONE)

    def test_parse_yoy_expands_required_periods(self) -> None:
        query = parse_structured_query("贵州茅台2025年营业收入同比增长多少", company_aliases=ALIASES)
        self.assertIsNotNone(query)
        assert query is not None
        self.assertEqual(query.operation, "calculate")
        self.assertEqual(query.calculation_kind, "yoy")
        self.assertEqual(query.periods, (Period("FY", 2024), Period("FY", 2025)))

    def test_parse_plain_q2_calculation_does_not_infer_period_basis(self) -> None:
        cases = (
            (
                "贵州茅台2025年二季度营业收入同比增长多少",
                "yoy",
                (Period("Q2", 2024), Period("Q2", 2025)),
            ),
            (
                "贵州茅台2025年二季度营业收入环比增长多少",
                "qoq",
                (Period("Q1", 2025), Period("Q2", 2025)),
            ),
        )
        for text, calculation_kind, periods in cases:
            with self.subTest(calculation_kind=calculation_kind):
                query = parse_structured_query(text, company_aliases=ALIASES)
                self.assertIsNotNone(query)
                assert query is not None
                self.assertEqual(query.operation, "calculate")
                self.assertEqual(query.calculation_kind, calculation_kind)
                self.assertEqual(query.periods, periods)
                self.assertIsNone(query.period_basis)

    def test_parse_explicit_standalone_q2_calculation_keeps_basis(self) -> None:
        query = parse_structured_query(
            "贵州茅台2025年二季度单季营业收入同比增长多少",
            company_aliases=ALIASES,
        )
        self.assertIsNotNone(query)
        assert query is not None
        self.assertEqual(query.periods, (Period("Q2", 2024), Period("Q2", 2025)))
        self.assertEqual(query.period_basis, PeriodBasis.STANDALONE)

    def test_parse_ratio_generates_metric_operands(self) -> None:
        query = parse_structured_query("贵州茅台2025年研发费用占营业收入比例", company_aliases=ALIASES)
        self.assertIsNotNone(query)
        assert query is not None
        self.assertEqual(query.operation, "calculate")
        self.assertEqual(query.calculation_kind, "ratio")
        result = execute_structured_query(None, query)  # type: ignore[arg-type]
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.facts, ())
        self.assertEqual(
            {item.metric for item in result.operand_requirements},
            {Metric.RD_EXPENSE, Metric.REVENUE},
        )


class TestExecuteStructuredQuery(unittest.TestCase):
    def setUp(self) -> None:
        self.store = FactStore(":memory:")
        self.addCleanup(self.store.close)

    def test_execute_compare(self) -> None:
        self.store.upsert([
            make_fact(company="贵州茅台", value="123450000000"),
            make_fact(company="五粮液", value="98700000000", doc_id="doc-wly-2025"),
        ])
        query = build_structured_query(
            entities=["贵州茅台", "五粮液"],
            metrics=[Metric.REVENUE],
            periods=[Period("FY", 2025)],
            operation="compare",
        )
        result = execute_structured_query(self.store, query)
        self.assertEqual(result.status, "ok")
        self.assertEqual({fact.company for fact in result.facts}, {"贵州茅台", "五粮液"})

    def test_execute_trend_range(self) -> None:
        self.store.upsert([
            make_fact(year=2023, value="100000000000", doc_id="doc-2023"),
            make_fact(year=2024, value="110000000000", doc_id="doc-2024"),
            make_fact(year=2025, value="123450000000", doc_id="doc-2025"),
        ])
        query = build_structured_query(
            entities=["贵州茅台"],
            metrics=[Metric.REVENUE],
            periods=[(Period("FY", 2023), Period("FY", 2025))],
            operation="trend",
        )
        result = execute_structured_query(self.store, query)
        self.assertEqual(result.status, "ok")
        self.assertEqual([fact.period.year for fact in result.facts], [2023, 2024, 2025])

    def test_execute_forecast_isolation(self) -> None:
        self.store.upsert([
            make_fact(year=2025, value_type="actual", doc_id="doc-actual"),
            make_fact(year=2025, value_type="forecast", doc_id="doc-forecast", evidence_id="E-forecast"),
        ])
        query = build_structured_query(
            entities=["贵州茅台"],
            metrics=[Metric.REVENUE],
            periods=[Period("FY", 2025)],
            operation="lookup",
            value_type=ValueType.FORECAST,
        )
        result = execute_structured_query(self.store, query)
        self.assertEqual(result.status, "ok")
        self.assertEqual(len(result.facts), 1)
        self.assertEqual(result.facts[0].value_type, ValueType.FORECAST)

    def test_default_lookup_does_not_mix_value_types(self) -> None:
        self.store.upsert([
            make_fact(value_type="actual", doc_id="doc-actual", evidence_id="E-actual"),
            make_fact(value_type="forecast", doc_id="doc-forecast", evidence_id="E-forecast"),
            make_fact(value_type="adjusted", doc_id="doc-adjusted", evidence_id="E-adjusted"),
        ])
        query = build_structured_query(
            entities=["贵州茅台"],
            metrics=[Metric.REVENUE],
            periods=[Period("FY", 2025)],
            operation="lookup",
        )
        result = execute_structured_query(self.store, query)
        self.assertEqual([fact.value_type for fact in result.facts], [ValueType.ACTUAL])

    def test_q2_standalone_only_matches_standalone_facts(self) -> None:
        self.store.upsert([
            make_fact(
                period=Period("Q2", 2025),
                period_basis="standalone",
                doc_id="doc-q2-single",
                evidence_id="E-q2-single",
            ),
            make_fact(
                period=Period("Q2", 2025),
                period_basis="cumulative",
                doc_id="doc-q2-cumulative",
                evidence_id="E-q2-cumulative",
            ),
        ])
        query = parse_structured_query(
            "贵州茅台2025年二季度单季营业收入",
            company_aliases=ALIASES,
        )
        assert query is not None
        result = execute_structured_query(self.store, query)
        self.assertEqual(result.status, "ok")
        self.assertEqual(len(result.facts), 1)
        self.assertEqual(result.facts[0].period_basis, PeriodBasis.STANDALONE)

    def test_lookup_filters_accounting_scope_and_revision(self) -> None:
        self.store.upsert([
            make_fact(
                accounting_scope="parent",
                revision_status="original",
                doc_id="doc-parent",
                evidence_id="E-parent",
            ),
            make_fact(
                accounting_scope="consolidated",
                revision_status="restated",
                doc_id="doc-consolidated",
                evidence_id="E-consolidated",
            ),
        ])
        query = build_structured_query(
            entities=["贵州茅台"],
            metrics=[Metric.REVENUE],
            periods=[Period("FY", 2025)],
            operation="lookup",
            accounting_scope="consolidated",
            revision_status="restated",
        )
        result = execute_structured_query(self.store, query)
        self.assertEqual(result.status, "ok")
        self.assertEqual(len(result.facts), 1)
        self.assertEqual(result.facts[0].doc_id, "doc-consolidated")

    def test_plain_q2_fails_closed_when_period_basis_is_ambiguous(self) -> None:
        query = parse_structured_query("贵州茅台2025年二季度营业收入", company_aliases=ALIASES)
        self.assertIsNotNone(query)
        assert query is not None
        result = execute_structured_query(self.store, query)
        self.assertEqual(result.status, "fail_closed")
        self.assertEqual(result.reason, "ambiguous_period_basis")
        self.assertEqual(result.facts, ())

    def test_plain_q2_calculations_fail_closed_when_period_basis_is_ambiguous(self) -> None:
        for text in (
            "贵州茅台2025年二季度营业收入同比增长多少",
            "贵州茅台2025年二季度营业收入环比增长多少",
        ):
            with self.subTest(text=text):
                query = parse_structured_query(text, company_aliases=ALIASES)
                self.assertIsNotNone(query)
                assert query is not None
                result = execute_structured_query(self.store, query)
                self.assertEqual(result.status, "fail_closed")
                self.assertEqual(result.reason, "ambiguous_period_basis")
                self.assertEqual(result.facts, ())
                self.assertEqual(result.operand_requirements, ())

    def test_explicit_standalone_q2_calculation_returns_requirements(self) -> None:
        query = parse_structured_query(
            "贵州茅台2025年二季度单季营业收入同比增长多少",
            company_aliases=ALIASES,
        )
        self.assertIsNotNone(query)
        assert query is not None
        result = execute_structured_query(None, query)  # type: ignore[arg-type]
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.reason, None)
        self.assertEqual([item.name for item in result.operand_requirements], ["prior", "current"])
        self.assertTrue(
            all(item.period_basis is PeriodBasis.STANDALONE for item in result.operand_requirements)
        )

    def test_q2_range_without_basis_also_fails_closed(self) -> None:
        query = build_structured_query(
            entities=["贵州茅台"],
            metrics=[Metric.REVENUE],
            periods=[(Period("Q2", 2024), Period("Q2", 2025))],
            operation="trend",
        )
        result = execute_structured_query(None, query)  # type: ignore[arg-type]
        self.assertEqual(result.status, "fail_closed")
        self.assertEqual(result.reason, "ambiguous_period_basis")

    def test_lookup_allows_multi_dimension_metric_selection(self) -> None:
        query = build_structured_query(
            entities=["贵州茅台"],
            metrics=[Metric.REVENUE, Metric.GROSS_MARGIN],
            periods=[Period("FY", 2025)],
            operation="lookup",
        )
        self.assertEqual(query.metrics, (Metric.REVENUE, Metric.GROSS_MARGIN))

    def test_calculation_rejects_incompatible_metric_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            build_structured_query(
                entities=["贵州茅台"],
                metrics=[Metric.REVENUE, Metric.GROSS_MARGIN],
                periods=[Period("FY", 2025)],
                operation="calculate",
            )

    def test_calculate_returns_requirements_without_reading_store(self) -> None:
        class StoreMustNotBeRead:
            def query_many(self, **kwargs):
                raise AssertionError("calculate must not query or return facts")

        query = build_structured_query(
            entities=["贵州茅台"],
            metrics=[Metric.REVENUE],
            periods=[Period("FY", 2024), Period("FY", 2025)],
            operation="calculate",
            calculation_kind="yoy",
        )
        result = execute_structured_query(StoreMustNotBeRead(), query)  # type: ignore[arg-type]
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.facts, ())
        self.assertEqual([item.name for item in result.operand_requirements], ["prior", "current"])
        self.assertEqual(
            [item.period for item in result.operand_requirements],
            [Period("FY", 2024), Period("FY", 2025)],
        )
        self.assertTrue(all(item.value_type is ValueType.ACTUAL for item in result.operand_requirements))

    def test_calculate_with_one_coordinate_fails_closed_but_returns_requirement(self) -> None:
        query = build_structured_query(
            entities=["贵州茅台"],
            metrics=[Metric.REVENUE],
            periods=[Period("FY", 2025)],
            operation="calculate",
        )
        result = execute_structured_query(None, query)  # type: ignore[arg-type]
        self.assertEqual(result.status, "fail_closed")
        self.assertEqual(result.reason, "insufficient_operand_requirements")
        self.assertEqual(len(result.operand_requirements), 1)
        self.assertEqual(result.facts, ())

    def test_query_many_sort_and_filter(self) -> None:
        self.store.upsert([
            make_fact(year=2024, value="100000000000", doc_id="doc-2024"),
            make_fact(year=2025, value="123450000000", doc_id="doc-2025"),
            make_fact(company="五粮液", value="98000000000", doc_id="doc-wly"),
        ])
        facts = self.store.query_many(
            companies=["贵州茅台"],
            metrics=[Metric.REVENUE],
            periods=[(Period("FY", 2024), Period("FY", 2025))],
            value_types=[ValueType.ACTUAL],
            order_by="value",
            descending=True,
        )
        self.assertEqual([fact.period.year for fact in facts], [2025, 2024])


if __name__ == "__main__":
    unittest.main()
