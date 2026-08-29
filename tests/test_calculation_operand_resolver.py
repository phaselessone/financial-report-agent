from __future__ import annotations

from decimal import Decimal

from src.agent.calculation_operand_resolver import (
    AmbiguousOperand,
    CalculationOperandResolver,
    OperandRequirement,
    ResolvedOperand,
)
from src.structured.schema import FinancialFact, Period


def _fact(*, value: str = "12000000000", year: int = 2025) -> FinancialFact:
    return FinancialFact(
        company="甲公司",
        metric="revenue",
        period=Period("FY", year),
        value_type="actual",
        value=Decimal(value),
        unit="元",
        doc_id=f"doc-{year}",
        page=8,
        evidence_id=f"E-{year}",
        raw_value="120亿元",
        source_span="甲公司2025年营业收入为120亿元。",
    )


def test_structured_fact_exact_match_resolves_with_full_provenance() -> None:
    requirement = OperandRequirement(
        name="current",
        entity="甲公司",
        metric="revenue",
        period="FY2025",
        value_type="actual",
        unit="元",
    )

    result = CalculationOperandResolver().resolve(
        requirement,
        structured_facts=[_fact()],
        evidence_rows=[],
    )

    assert isinstance(result, ResolvedOperand)
    assert result.source == "structured_fact"
    assert result.operand.name == "current"
    assert result.operand.value == Decimal("12000000000")
    assert result.operand.unit == "元"
    assert result.operand.fact_id == _fact().fact_id
    assert result.operand.evidence_id == "E-2025"


def test_evidence_row_with_multiple_numbers_uses_coordinate_value_not_first_number() -> None:
    requirement = OperandRequirement(
        name="current",
        entity="甲公司",
        metric="revenue",
        period="FY2025",
        value_type="actual",
        unit="亿元",
    )
    row = {
        "entity": "甲公司",
        "metric": "revenue",
        "period": "FY2025",
        "value_type": "actual",
        "value": "120",
        "unit": "亿元",
        "raw_value": "120亿元",
        "evidence_id": "E-current",
        "chunk_id": "chunk-current",
        "doc_id": "annual-report",
        "text": "甲公司2024年营收100亿元，2025年营收120亿元。",
    }

    result = CalculationOperandResolver().resolve(
        requirement,
        structured_facts=[],
        evidence_rows=[row],
    )

    assert isinstance(result, ResolvedOperand)
    assert result.source == "evidence"
    assert result.operand.value == Decimal("120")
    assert result.operand.raw_text == "120亿元"
    assert result.operand.evidence_id == "E-current"
    assert result.operand.chunk_id == "chunk-current"


def test_text_only_evidence_with_original_and_restated_values_is_ambiguous() -> None:
    requirement = OperandRequirement(
        name="current",
        entity="甲公司",
        metric="revenue",
        period="FY2025",
        value_type="actual",
        unit="亿元",
    )
    row = {
        "evidence_id": "E-conflict",
        "chunk_id": "chunk-conflict",
        "doc_id": "annual-report",
        "text": "甲公司2025年营业收入100亿元，重述后营业收入120亿元。",
    }

    result = CalculationOperandResolver().resolve(
        requirement,
        structured_facts=[],
        evidence_rows=[row],
    )

    assert isinstance(result, AmbiguousOperand)
    assert result.candidate_ids == ("E-conflict",)
    assert result.reason == "multiple_numeric_candidates"


def test_missing_local_operand_uses_one_bounded_requirement_specific_retrieval() -> None:
    calls: list[OperandRequirement] = []

    def retrieve(requirement: OperandRequirement):
        calls.append(requirement)
        return [
            {
                "entity": "甲公司",
                "metric": "revenue",
                "period": "FY2024",
                "value_type": "actual",
                "value": "100",
                "unit": "亿元",
                "raw_value": "100亿元",
                "evidence_id": "E-retrieved",
            }
        ]

    requirement = OperandRequirement(
        name="prior",
        entity="甲公司",
        metric="revenue",
        period="FY2024",
        unit="亿元",
    )
    result = CalculationOperandResolver(retrieve=retrieve).resolve(
        requirement,
        structured_facts=[],
        evidence_rows=[],
    )

    assert isinstance(result, ResolvedOperand)
    assert result.source == "retrieved_evidence"
    assert result.operand.evidence_id == "E-retrieved"
    assert calls == [requirement]


def test_text_only_evidence_selects_number_anchored_to_requested_period() -> None:
    requirement = OperandRequirement(
        name="current",
        entity="甲公司",
        metric="revenue",
        period="FY2025",
        unit="亿元",
    )
    result = CalculationOperandResolver().resolve(
        requirement,
        structured_facts=[],
        evidence_rows=[
            {
                "evidence_id": "E-table-row",
                "chunk_id": "chunk-table-row",
                "doc_id": "annual-report",
                "text": "甲公司2024年营业收入100亿元；2025年营业收入120亿元。",
            }
        ],
    )

    assert isinstance(result, ResolvedOperand)
    assert result.operand.value == Decimal("120")
    assert result.operand.unit == "亿元"
    assert result.operand.raw_text == "120亿元"


def test_operand_resolution_filters_period_basis_scope_and_revision() -> None:
    requirement = OperandRequirement(
        name="current",
        entity="甲公司",
        metric="revenue",
        period="Q22025",
        value_type="adjusted",
        unit="元",
        period_basis="standalone",
        accounting_scope="consolidated",
        revision_status="restated",
    )

    def coordinated_fact(
        value: str,
        evidence_id: str,
        *,
        period_basis: str,
        revision_status: str,
    ) -> FinancialFact:
        return FinancialFact(
            company="甲公司",
            metric="revenue",
            period=Period("Q2", 2025),
            value_type="adjusted",
            value=Decimal(value),
            unit="元",
            doc_id="report",
            page=2,
            evidence_id=evidence_id,
            raw_value=value,
            source_span="甲公司2025年二季度调整后营业收入",
            period_basis=period_basis,
            accounting_scope="consolidated",
            revision_status=revision_status,
        )

    result = CalculationOperandResolver().resolve(
        requirement,
        structured_facts=[
            coordinated_fact(
                "123", "E-match", period_basis="standalone", revision_status="restated"
            ),
            coordinated_fact(
                "115", "E-original", period_basis="standalone", revision_status="original"
            ),
            coordinated_fact(
                "220", "E-cumulative", period_basis="cumulative", revision_status="restated"
            ),
        ],
        evidence_rows=[],
    )

    assert isinstance(result, ResolvedOperand)
    assert result.operand.evidence_id == "E-match"


def test_evidence_operand_requires_explicit_full_coordinates_when_requested() -> None:
    requirement = OperandRequirement(
        name="current",
        entity="甲公司",
        metric="revenue",
        period="Q22025",
        value_type="adjusted",
        unit="元",
        period_basis="standalone",
        accounting_scope="consolidated",
        revision_status="restated",
    )
    rows = [
        {
            "entity": "甲公司",
            "metric": "revenue",
            "period": "Q22025",
            "value_type": "adjusted",
            "value": "123",
            "unit": "元",
            "period_basis": "standalone",
            "accounting_scope": "consolidated",
            "revision_status": "restated",
            "evidence_id": "E-match",
        },
        {
            "entity": "甲公司",
            "metric": "revenue",
            "period": "Q22025",
            "value_type": "adjusted",
            "value": "220",
            "unit": "元",
            "period_basis": "cumulative",
            "accounting_scope": "consolidated",
            "revision_status": "restated",
            "evidence_id": "E-cumulative",
        },
    ]

    result = CalculationOperandResolver().resolve(
        requirement,
        structured_facts=[],
        evidence_rows=rows,
    )

    assert isinstance(result, ResolvedOperand)
    assert result.operand.evidence_id == "E-match"
