"""Strict claim-chain coverage for deterministic multi-hop comparisons."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import patch

from src.agent.config import AgentConfig
from src.agent.claims import stable_claim_id
from src.agent.graph import run_agentic_rag
from src.agent.nodes.extract_claims import _deterministic_comparison_claims
from src.structured.fact_store import FactStore
from src.structured.schema import FinancialFact, Period
from tests.test_agent_flow import FakeAnswerer, FakeLLM, FakeRuntime


ALIASES = {"甲公司": ["甲公司", "甲"], "乙公司": ["乙公司", "乙"]}


def _fact(company: str, year: int, value: str, evidence_id: str) -> FinancialFact:
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


def test_strict_growth_comparison_emits_verified_parent_claim_chain() -> None:
    store = FactStore(":memory:")
    try:
        store.upsert(
            [
                _fact("甲公司", 2024, "100", "E-A-2024"),
                _fact("甲公司", 2025, "120", "E-A-2025"),
                _fact("乙公司", 2024, "200", "E-B-2024"),
                _fact("乙公司", 2025, "210", "E-B-2025"),
            ]
        )
        state = run_agentic_rag(
            runtime=FakeRuntime([]),
            answerer=FakeAnswerer([]),
            llm=FakeLLM([]),
            config=AgentConfig(strict_claim_verification=True),
            query="甲公司和乙公司2025年营业收入谁同比增长更快？",
            fact_store=store,
            company_aliases=ALIASES,
        )
    finally:
        store.close()

    final = state["final_answer"]
    assert not final["abstained"]
    assert final["final_answer"] == "甲公司增长更快。"

    claims = final["claims"]
    parents = [claim for claim in claims if not claim["is_core"]]
    core = [claim for claim in claims if claim["is_core"]]
    assert len(parents) == 2
    assert {claim["claim_type"] for claim in parents} == {"DERIVED"}
    assert all(claim["calculation_id"] for claim in parents)
    assert len(core) == 1
    assert core[0]["claim_type"] == "SYNTHESIZED"
    assert set(core[0]["parent_claim_ids"]) == {claim["claim_id"] for claim in parents}
    assert {claim["verification"]["status"] for claim in claims} == {"ENTAILED"}
    assert set(final["used_evidence_ids"]) == {
        "E-A-2024",
        "E-A-2025",
        "E-B-2024",
        "E-B-2025",
    }


def test_strict_growth_comparison_nonstandard_claim_cannot_bypass_calculation_lineage() -> None:
    judge_calls: list[dict[str, object]] = []

    def judge(payload):
        judge_calls.append(dict(payload))
        return {
            "status": "ENTAILED",
            "evidence_ids": ["E-A-2024", "E-A-2025", "E-B-2024", "E-B-2025"],
            "score": 0.99,
            "reasons": ["model inferred the comparison"],
        }

    def malformed_comparison_claims(state, *, answer_text):
        claims = _deterministic_comparison_claims(state, answer_text=answer_text)
        assert claims is not None
        core = next(claim for claim in claims if claim["is_core"])
        core["text"] = "甲公司的扩张动能领先乙公司。"
        core["claim_id"] = stable_claim_id(core["text"])
        core["calculation_id"] = None
        return claims

    store = FactStore(":memory:")
    try:
        store.upsert(
            [
                _fact("甲公司", 2024, "100", "E-A-2024"),
                _fact("甲公司", 2025, "120", "E-A-2025"),
                _fact("乙公司", 2024, "200", "E-B-2024"),
                _fact("乙公司", 2025, "210", "E-B-2025"),
            ]
        )
        with patch(
            "src.agent.nodes.extract_claims._deterministic_comparison_claims",
            side_effect=malformed_comparison_claims,
        ):
            state = run_agentic_rag(
                runtime=FakeRuntime([]),
                answerer=FakeAnswerer([]),
                llm=FakeLLM([]),
                config=AgentConfig(strict_claim_verification=True, claim_llm_budget=1),
                query="甲公司和乙公司2025年营业收入谁同比增长更快？",
                fact_store=store,
                company_aliases=ALIASES,
                llm_judge=judge,
            )
    finally:
        store.close()

    core = next(claim for claim in state["claims"] if claim["is_core"])
    assert core["verification"]["status"] == "INSUFFICIENT"
    assert "calculation_backed_claim_missing_calculation" in core["verification"]["reasons"]
    assert judge_calls == []
    assert state["final_answer"]["abstained"] is True
    assert state["final_answer"]["citations"] == []
    assert state["final_answer"]["used_evidence_ids"] == []
