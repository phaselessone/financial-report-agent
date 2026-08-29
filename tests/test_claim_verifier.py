"""Regression tests for the claim verifier's public seam."""

from __future__ import annotations

import json

import pytest

from src.agent.claim_verifier import (
    CONTRADICTED,
    ENTAILED,
    INSUFFICIENT,
    verify_claim,
)


def numeric_claim(
    *,
    entity: str = "宁德时代",
    metric: str = "revenue",
    period: str = "2025",
    value: object = "100亿元",
    unit: str | None = None,
    value_type: str = "actual",
    text: str | None = None,
    **extra: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "claim_id": "claim-1",
        "text": text or f"{entity}{period}年{metric}为{value}。",
        "claim_type": "EXTRACTED",
        "entity": entity,
        "metric": metric,
        "period": period,
        "value": value,
        "value_type": value_type,
    }
    if unit is not None:
        payload["unit"] = unit
    payload.update(extra)
    return payload


def numeric_row(
    evidence_id: str = "E1",
    *,
    entity: str = "宁德时代",
    metric: str = "revenue",
    period: str = "2025",
    value: object = "100亿元",
    unit: str | None = None,
    value_type: str = "actual",
    text: str | None = None,
    **extra: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "evidence_id": evidence_id,
        "doc_id": f"doc-{evidence_id}",
        "text": text or f"{entity}{period}年{metric}为{value}。",
        "entity": entity,
        "metric": metric,
        "period": period,
        "value": value,
        "value_type": value_type,
    }
    if unit is not None:
        payload["unit"] = unit
    payload.update(extra)
    return payload


def row(evidence_id: str, text: str, **metadata: object) -> dict[str, object]:
    return {"evidence_id": evidence_id, "doc_id": f"doc-{evidence_id}", "text": text, **metadata}


def status(result: dict[str, object]) -> str:
    return result["verification"]["status"]  # type: ignore[index]


@pytest.mark.parametrize(
    ("claim_overrides", "row_overrides", "expected", "reason"),
    [
        ({}, {}, ENTAILED, "exact structured numeric fact"),
        ({"value": "1亿元"}, {"value": "10000万元"}, ENTAILED, "currency unit conversion"),
        ({"value": "100000000元"}, {"value": "1亿元"}, ENTAILED, "base currency conversion"),
        ({"value": "１００亿元"}, {"value": "100亿元"}, ENTAILED, "full-width digits"),
        (
            {"metric": "gross_margin", "value": "20%"},
            {"metric": "毛利率", "value": "20%"},
            ENTAILED,
            "metric alias",
        ),
        (
            {"metric": "gross_margin", "value": "2个百分点"},
            {"metric": "gross_margin", "value": "2pp"},
            ENTAILED,
            "percentage-point aliases",
        ),
        (
            {
                "value": "101亿元",
                "accounting_scope": "consolidated",
                "revision_status": "original",
            },
            {"accounting_scope": "consolidated", "revision_status": "original"},
            CONTRADICTED,
            "same complete coordinates, different value",
        ),
        ({"entity": "比亚迪"}, {}, INSUFFICIENT, "same number, different company"),
        ({"metric": "net_profit"}, {}, INSUFFICIENT, "same number, different metric"),
        ({"period": "2024"}, {}, INSUFFICIENT, "same number, different period"),
        ({"value_type": "forecast"}, {}, INSUFFICIENT, "forecast claim versus actual evidence"),
        ({}, {"value_type": "forecast"}, INSUFFICIENT, "actual claim versus forecast evidence"),
        (
            {
                "value_type": "adjusted",
                "text": "宁德时代2025年经调整营收为100亿元。",
            },
            {},
            INSUFFICIENT,
            "adjusted claim versus actual evidence",
        ),
        (
            {
                "value_type": "adjusted",
                "text": "宁德时代2025年经调整营收为100亿元。",
            },
            {
                "value_type": "adjusted",
                "text": "宁德时代2025年调整后营收为100亿元。",
            },
            ENTAILED,
            "adjusted-to-adjusted",
        ),
        (
            {"metric": "gross_margin", "value": "2%"},
            {"metric": "gross_margin", "value": "2pp"},
            INSUFFICIENT,
            "percent is not percentage points",
        ),
        (
            {"metric": "gross_margin", "value": "2个百分点"},
            {"metric": "gross_margin", "value": "2%"},
            INSUFFICIENT,
            "percentage points are not percent",
        ),
        ({"value": "100亿元"}, {"value": "100%"}, INSUFFICIENT, "currency is not percent"),
        (
            {"period": "2025Q1"},
            {"period": "2025年第一季度"},
            ENTAILED,
            "quarter aliases",
        ),
        (
            {"period": "2025H1"},
            {"period": "2025年上半年"},
            ENTAILED,
            "half-year aliases",
        ),
        (
            {"value_type": "forecast", "text": "预计宁德时代2025年营收为100亿元。"},
            {"value_type": "forecast", "text": "宁德时代2025年预测营收为100亿元。"},
            ENTAILED,
            "forecast-to-forecast",
        ),
    ],
)
def test_numeric_constraint_regressions(
    claim_overrides: dict[str, object],
    row_overrides: dict[str, object],
    expected: str,
    reason: str,
) -> None:
    claim = numeric_claim(**claim_overrides)
    evidence = numeric_row(**row_overrides)
    result = verify_claim(claim, [evidence])

    assert status(result) == expected, reason
    assert result["verification"]["method"] == "deterministic"


def test_numeric_claim_maps_only_matching_evidence() -> None:
    result = verify_claim(
        numeric_claim(),
        [
            numeric_row("E-wrong-company", entity="比亚迪"),
            numeric_row("E-support"),
            numeric_row("E-wrong-period", period="2024"),
        ],
    )

    assert status(result) == ENTAILED
    assert result["evidence_ids"] == ["E-support"]


def test_support_and_conflict_is_contradicted_and_keeps_both_sources() -> None:
    result = verify_claim(
        {
            **numeric_claim(),
            "accounting_scope": "consolidated",
            "revision_status": "original",
        },
        [
            {
                **numeric_row("E-support"),
                "accounting_scope": "consolidated",
                "revision_status": "original",
            },
            {
                **numeric_row("E-conflict", value="120亿元"),
                "accounting_scope": "consolidated",
                "revision_status": "original",
            },
        ],
    )

    assert status(result) == CONTRADICTED
    assert result["evidence_ids"] == ["E-support", "E-conflict"]
    assert "support_and_conflict" in result["verification"]["reasons"]


def test_numeric_conflict_with_missing_scope_or_revision_is_insufficient() -> None:
    result = verify_claim(
        numeric_claim(),
        [numeric_row("E-support"), numeric_row("E-conflict", value="120亿元")],
    )

    assert status(result) == INSUFFICIENT
    assert result["evidence_ids"] == ["E-support", "E-conflict"]
    assert result["verification"]["reasons"] == [
        "numeric_conflict_dimensions_incomplete"
    ]


def test_conflict_for_other_company_does_not_override_support() -> None:
    result = verify_claim(
        numeric_claim(),
        [numeric_row("E-support"), numeric_row("E-other", entity="比亚迪", value="120亿元")],
    )

    assert status(result) == ENTAILED
    assert result["evidence_ids"] == ["E-support"]


def test_claim_and_row_can_parse_constraints_from_text() -> None:
    claim = {
        "claim_id": "claim-text",
        "claim_type": "EXTRACTED",
        "entity": "宁德时代",
        "text": "宁德时代2025年营收为1亿元。",
    }
    evidence = row(
        "E1",
        "宁德时代2025年营业收入为10000万元。",
        entity="宁德时代",
    )

    result = verify_claim(claim, [evidence])

    assert status(result) == ENTAILED
    assert result["evidence_ids"] == ["E1"]


def test_missing_numeric_entity_fails_closed_even_when_text_is_contained() -> None:
    claim = {"text": "2025年营收为100亿元。", "claim_type": "EXTRACTED"}
    evidence = row("E1", "2025年营收为100亿元。")

    result = verify_claim(claim, [evidence])

    assert status(result) == INSUFFICIENT
    assert any("numeric_constraints_incomplete" in item for item in result["verification"]["reasons"])


def test_empty_claim_is_insufficient() -> None:
    result = verify_claim({"text": ""}, [row("E1", "任意证据")])
    assert status(result) == INSUFFICIENT
    assert result["verification"]["reasons"] == ["empty_claim_text"]


def test_no_evidence_is_insufficient() -> None:
    result = verify_claim({"text": "公司增加研发投入。"}, [])
    assert status(result) == INSUFFICIENT
    assert result["evidence_ids"] == []


def test_qualitative_exact_containment_is_entailed_and_claim_specific() -> None:
    result = verify_claim(
        {"text": "公司增加研发投入。", "claim_type": "EXTRACTED"},
        [row("E1", "报告显示，公司增加研发投入。"), row("E2", "公司削减资本开支。")],
    )

    assert status(result) == ENTAILED
    assert result["evidence_ids"] == ["E1"]


def test_qualitative_negation_without_complete_conflict_coordinates_is_insufficient() -> None:
    result = verify_claim(
        {"text": "公司未增加研发投入。"},
        [row("E1", "公司增加研发投入。")],
    )
    assert status(result) == INSUFFICIENT
    assert result["evidence_ids"] == ["E1"]
    assert result["verification"]["reasons"] == [
        "qualitative_conflict_dimensions_incomplete"
    ]


def test_qualitative_support_and_uncoordinated_conflict_is_insufficient() -> None:
    result = verify_claim(
        {"text": "公司增加研发投入。"},
        [row("E1", "公司增加研发投入。"), row("E2", "公司未增加研发投入。")],
    )
    assert status(result) == INSUFFICIENT
    assert result["evidence_ids"] == ["E1", "E2"]


def test_qualitative_conflict_requires_matching_complete_scope_and_revision() -> None:
    coordinates = {
        "entity": "甲公司",
        "metric": "research_and_development",
        "period": "FY2025",
        "value_type": "actual",
        "accounting_scope": "consolidated",
        "revision_status": "original",
    }
    contradicted = verify_claim(
        {"text": "甲公司2025年增加研发投入。", **coordinates},
        [row("E1", "甲公司2025年未增加研发投入。", **coordinates)],
    )
    assert status(contradicted) == CONTRADICTED

    revised_conflict = verify_claim(
        {"text": "甲公司2025年增加研发投入。", **coordinates},
        [
            row("E-support", "甲公司2025年增加研发投入。", **coordinates),
            row(
                "E-restated",
                "甲公司2025年未增加研发投入。",
                **{**coordinates, "revision_status": "restated"},
            ),
        ],
    )
    assert status(revised_conflict) == ENTAILED
    assert revised_conflict["evidence_ids"] == ["E-support"]


def test_unrelated_evidence_is_insufficient_and_unmapped() -> None:
    result = verify_claim(
        {"text": "公司增加研发投入。"},
        [row("E1", "行业库存处于低位。")],
    )
    assert status(result) == INSUFFICIENT
    assert result["evidence_ids"] == []


def test_bare_semantic_similarity_only_ranks_and_cannot_entail() -> None:
    calls: list[tuple[str, str]] = []

    def scorer(claim_text: str, evidence_text: str) -> float:
        calls.append((claim_text, evidence_text))
        return 0.91

    result = verify_claim(
        {"text": "人工智能推动增长。"},
        [row("E1", "AI成为业绩增长动力。")],
        semantic_scorer=scorer,
    )

    assert status(result) == INSUFFICIENT
    assert result["verification"]["method"] == "semantic"
    assert result["evidence_ids"] == ["E1"]
    assert "semantic_similarity_not_directional" in result["verification"]["reasons"]
    assert len(calls) == 1


def test_semantic_scorer_can_report_contradiction() -> None:
    result = verify_claim(
        {"text": "人工智能推动增长。"},
        [row("E1", "AI与业绩增长之间的关系不明确。")],
        semantic_scorer=lambda _claim, _evidence: {"status": CONTRADICTED, "score": 0.95},
    )
    assert status(result) == CONTRADICTED
    assert result["verification"]["method"] == "semantic"


def test_semantic_score_below_threshold_is_insufficient() -> None:
    result = verify_claim(
        {"text": "人工智能推动增长。"},
        [row("E1", "AI成为业绩增长动力。")],
        semantic_scorer=lambda _claim, _evidence: 0.84,
        semantic_threshold=0.85,
    )
    assert status(result) == INSUFFICIENT
    assert result["verification"]["method"] == "semantic"


def test_semantic_scorer_error_fails_closed() -> None:
    def broken(_claim: str, _evidence: str) -> float:
        raise RuntimeError("offline")

    result = verify_claim(
        {"text": "人工智能推动增长。"},
        [row("E1", "AI成为业绩增长动力。")],
        semantic_scorer=broken,
    )
    assert status(result) == INSUFFICIENT
    assert "semantic_scorer_error" in result["verification"]["reasons"]


def test_deterministic_decision_does_not_call_semantic_or_llm() -> None:
    calls = {"semantic": 0, "llm": 0}

    def scorer(_claim: str, _evidence: str) -> float:
        calls["semantic"] += 1
        return 1.0

    def judge(_payload: object) -> dict[str, object]:
        calls["llm"] += 1
        return {"status": ENTAILED, "evidence_ids": ["E1"], "score": 1.0, "reasons": ["ok"]}

    result = verify_claim(
        {"text": "公司增加研发投入。"},
        [row("E1", "公司增加研发投入。")],
        semantic_scorer=scorer,
        llm_judge=judge,
    )
    assert status(result) == ENTAILED
    assert calls == {"semantic": 0, "llm": 0}


def test_llm_judge_receives_numbered_evidence_and_can_entail_two_documents() -> None:
    seen: list[object] = []

    def judge(payload: object) -> dict[str, object]:
        seen.append(payload)
        return {
            "status": ENTAILED,
            "evidence_ids": ["E1", "E2"],
            "score": 0.9,
            "reasons": ["two reports jointly support the synthesis"],
        }

    result = verify_claim(
        {"claim_id": "claim-7", "claim_type": "SYNTHESIZED", "text": "两家公司都强调增长质量。"},
        [row("E1", "甲公司强调高质量增长。"), row("E2", "乙公司重视增长质量。")],
        llm_judge=judge,
    )

    assert status(result) == ENTAILED
    assert result["evidence_ids"] == ["E1", "E2"]
    payload = seen[0]
    assert [item["evidence_id"] for item in payload["evidence"]] == ["E1", "E2"]  # type: ignore[index]


def test_llm_judge_accepts_structured_json_string() -> None:
    output = json.dumps(
        {"status": CONTRADICTED, "evidence_ids": ["E1"], "score": 0.88, "reasons": ["opposite meaning"]}
    )
    result = verify_claim(
        {"text": "人工智能推动增长。"},
        [row("E1", "AI与增长之间的关系尚不明确。")],
        llm_judge=lambda _payload: output,
    )
    assert status(result) == CONTRADICTED
    assert result["verification"]["method"] == "llm"


@pytest.mark.parametrize(
    "malformed",
    [
        "not json",
        {"status": ENTAILED},
        {"status": "SUPPORTED", "evidence_ids": ["E1"], "score": 1.0, "reasons": ["x"]},
        {"status": ENTAILED, "evidence_ids": ["unknown"], "score": 1.0, "reasons": ["x"]},
        {"status": ENTAILED, "evidence_ids": [], "score": 1.0, "reasons": ["x"]},
        {"status": ENTAILED, "evidence_ids": ["E1"], "score": 1.5, "reasons": ["x"]},
        {"status": ENTAILED, "evidence_ids": ["E1"], "score": 1.0, "reasons": []},
    ],
)
def test_malformed_llm_output_fails_closed(malformed: object) -> None:
    result = verify_claim(
        {"text": "人工智能推动增长。"},
        [row("E1", "AI成为业绩增长动力。")],
        llm_judge=lambda _payload: malformed,  # type: ignore[return-value]
    )
    assert status(result) == INSUFFICIENT
    assert result["verification"]["reasons"] == ["llm_judge_malformed"]


def test_llm_exception_fails_closed() -> None:
    def broken(_payload: object) -> dict[str, object]:
        raise RuntimeError("provider unavailable")

    result = verify_claim(
        {"text": "人工智能推动增长。"},
        [row("E1", "AI成为业绩增长动力。")],
        llm_judge=broken,
    )
    assert status(result) == INSUFFICIENT
    assert result["verification"]["reasons"] == ["llm_judge_error"]


def test_zero_llm_budget_stops_before_judge() -> None:
    calls = 0

    def judge(_payload: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"status": ENTAILED, "evidence_ids": ["E1"], "score": 1.0, "reasons": ["x"]}

    result = verify_claim(
        {"text": "人工智能推动增长。"},
        [row("E1", "AI成为业绩增长动力。")],
        llm_judge=judge,
        llm_budget=0,
    )
    assert status(result) == INSUFFICIENT
    assert result["verification"]["method"] == "budget"
    assert calls == 0


def test_llm_cannot_entail_numeric_claim_with_missing_entity() -> None:
    result = verify_claim(
        {"text": "2025年营收为100亿元。", "claim_type": "EXTRACTED"},
        [row("E1", "2025年营收大约为100亿元。")],
        llm_judge=lambda _payload: {
            "status": ENTAILED, "evidence_ids": ["E1"], "score": 0.99, "reasons": ["looks supported"]
        },
    )
    assert status(result) == INSUFFICIENT
    assert "numeric_entailment_blocked_by_incomplete_constraints" in result["verification"]["reasons"]


def test_semantic_scorer_cannot_entail_numeric_claim_with_missing_entity() -> None:
    result = verify_claim(
        {"text": "2025年营收为100亿元。"},
        [row("E1", "2025年营业收入约为100亿元。")],
        semantic_scorer=lambda _claim, _evidence: 0.99,
    )
    assert status(result) == INSUFFICIENT


def calculation_claim(**extra: object) -> dict[str, object]:
    return numeric_claim(
        metric="yoy_growth",
        value="20%",
        text="宁德时代2025年营收同比增长20%。",
        claim_type="DERIVED",
        calculation_id="calc-1",
        **extra,
    )


def calculation_rows() -> list[dict[str, object]]:
    return [
        numeric_row("E-prior", period="2024", value="100亿元"),
        numeric_row("E-current", period="2025", value="120亿元"),
    ]


def test_derived_claim_without_calculation_never_entails_from_reported_result_alone() -> None:
    claim = numeric_claim(
        metric="yoy_growth",
        value="20%",
        text="宁德时代2025年营收同比增长20%。",
        claim_type="DERIVED",
    )
    result = verify_claim(
        claim,
        [
            numeric_row(
                "E-yoy",
                metric="yoy_growth",
                value="20%",
                text="宁德时代2025年营收同比增长20%。",
            )
        ],
        calculation_lookup={},
    )

    assert status(result) == INSUFFICIENT
    assert result["verification"]["reasons"] == ["derived_claim_missing_calculation"]


def test_matching_calculation_and_provenance_entails_derived_claim() -> None:
    calculations = {
        "calc-1": {
            "status": "success",
            "formatted": "20%",
            "inputs": [{"evidence_id": "E-prior"}, {"evidence_id": "E-current"}],
        }
    }
    result = verify_claim(calculation_claim(), calculation_rows(), calculation_lookup=calculations)
    assert status(result) == ENTAILED
    assert result["verification"]["method"] == "calculation"
    assert result["evidence_ids"] == ["E-prior", "E-current"]


def test_fact_id_calculation_provenance_entails_when_rows_expose_fact_aliases() -> None:
    calculations = {
        "calc-1": {
            "status": "success",
            "formatted": "20%",
            "inputs": [{"fact_id": "F-prior"}, {"fact_id": "F-current"}],
        }
    }
    rows = [
        {**numeric_row("structured:F-prior", period="2024", value="100亿元"), "fact_id": "F-prior"},
        {**numeric_row("structured:F-current", period="2025", value="120亿元"), "fact_id": "F-current"},
    ]
    result = verify_claim(calculation_claim(), rows, calculation_lookup=calculations)

    assert status(result) == ENTAILED
    assert result["evidence_ids"] == ["F-prior", "F-current"]


def test_claim_value_different_from_calculation_is_contradicted() -> None:
    calculations = {
        "calc-1": {
            "status": "success",
            "formatted": "18%",
            "evidence_ids": ["E-prior", "E-current"],
        }
    }
    result = verify_claim(calculation_claim(), calculation_rows(), calculation_lookup=calculations)
    assert status(result) == CONTRADICTED
    assert result["verification"]["reasons"] == ["claim_value_differs_from_calculation"]


def test_calculation_missing_input_provenance_is_insufficient() -> None:
    calculations = {
        "calc-1": {
            "status": "success",
            "formatted": "20%",
            "inputs": [{"evidence_id": "E-prior"}, {"raw_text": "120亿元"}],
        }
    }
    result = verify_claim(calculation_claim(), calculation_rows(), calculation_lookup=calculations)
    assert status(result) == INSUFFICIENT
    assert result["verification"]["reasons"] == ["calculation_provenance_incomplete"]


def test_calculation_unknown_evidence_is_insufficient() -> None:
    calculations = {
        "calc-1": {
            "status": "success",
            "formatted": "20%",
            "evidence_ids": ["E-prior", "E-ghost"],
        }
    }
    result = verify_claim(calculation_claim(), calculation_rows(), calculation_lookup=calculations)
    assert status(result) == INSUFFICIENT
    assert result["evidence_ids"] == ["E-prior"]


def test_failed_calculation_is_insufficient() -> None:
    calculations = {"calc-1": {"status": "error", "error_type": "zero_denominator"}}
    result = verify_claim(calculation_claim(), calculation_rows(), calculation_lookup=calculations)
    assert status(result) == INSUFFICIENT
    assert result["verification"]["reasons"] == ["calculation_not_successful"]


def test_missing_calculation_is_insufficient() -> None:
    result = verify_claim(calculation_claim(), calculation_rows(), calculation_lookup={})
    assert status(result) == INSUFFICIENT
    assert result["verification"]["reasons"] == ["calculation_not_found"]


def test_callable_calculation_lookup_is_supported() -> None:
    def lookup(calculation_id: str) -> dict[str, object] | None:
        assert calculation_id == "calc-1"
        return {
            "status": "ok",
            "result": "20%",
            "provenance": [{"evidence_id": "E-prior"}, {"evidence_id": "E-current"}],
        }

    result = verify_claim(calculation_claim(), calculation_rows(), calculation_lookup=lookup)
    assert status(result) == ENTAILED


def test_calculator_backed_comparison_entails_exact_conclusion() -> None:
    rows = [row("E1", "甲公司同比增长20%。"), row("E2", "乙公司同比增长5%。")]
    calculations = {
        "calc-compare": {
            "status": "success",
            "evidence_ids": ["E1", "E2"],
            "comparison": {
                "answer": "甲公司增长更快。",
                "winner": "甲公司",
                "loser": "乙公司",
            },
        }
    }

    result = verify_claim(
        {
            "text": "甲公司增长更快。",
            "claim_type": "SYNTHESIZED",
            "calculation_id": "calc-compare",
        },
        rows,
        calculation_lookup=calculations,
    )

    assert status(result) == ENTAILED
    assert result["verification"]["reasons"] == [
        "comparison_conclusion_and_provenance_match"
    ]


def test_calculator_backed_comparison_contradicts_wrong_winner() -> None:
    rows = [row("E1", "甲公司同比增长20%。"), row("E2", "乙公司同比增长5%。")]
    calculations = {
        "calc-compare": {
            "status": "success",
            "evidence_ids": ["E1", "E2"],
            "comparison": {
                "answer": "甲公司增长更快。",
                "winner": "甲公司",
                "loser": "乙公司",
            },
        }
    }

    result = verify_claim(
        {
            "text": "乙公司增长更快。",
            "claim_type": "SYNTHESIZED",
            "calculation_id": "calc-compare",
        },
        rows,
        calculation_lookup=calculations,
    )

    assert status(result) == CONTRADICTED
    assert result["verification"]["reasons"] == [
        "claim_contradicts_comparison_result"
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"semantic_threshold": -0.1}, "semantic_threshold"),
        ({"semantic_threshold": 1.1}, "semantic_threshold"),
        ({"llm_budget": -1}, "llm_budget"),
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        verify_claim({"text": "公司增加研发投入。"}, [row("E1", "公司增加研发投入。")], **kwargs)


def test_non_mapping_claim_is_rejected() -> None:
    with pytest.raises(TypeError, match="claim must be a mapping"):
        verify_claim("not a claim", [])  # type: ignore[arg-type]
