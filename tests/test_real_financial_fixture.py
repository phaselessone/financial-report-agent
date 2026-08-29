"""Public-safe financial-business acceptance fixture for clean-checkout CI."""

from __future__ import annotations

import json
from pathlib import Path

from src.agent.config import AgentConfig
from src.agent.graph import run_agentic_rag
from src.structured.fact_store import FactStore
from src.structured.schema import FinancialFact
from tests.test_agent_flow import FakeAnswerer, FakeLLM, FakeRuntime, supported_draft


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures"
FIXTURE_PATH = FIXTURE_ROOT / "controlled_financial_business_case.json"


def _load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_controlled_financial_fixture_is_in_public_tracked_scope() -> None:
    relative_path = FIXTURE_PATH.relative_to(REPO_ROOT)
    assert relative_path.parts[:2] == ("tests", "fixtures")
    assert "data" not in relative_path.parts

    fixture = _load_fixture()
    contract = fixture["fixture_contract"]
    assert contract["kind"] == "controlled_synthetic_financial_business_record"
    assert contract["public_safe"] is True
    assert contract["contains_private_or_raw_pdf_content"] is False
    assert "not copied from a real report" in contract["provenance_note"]


def test_controlled_financial_fixture_runs_without_ignored_data_tree(
    monkeypatch,
) -> None:
    monkeypatch.chdir(FIXTURE_ROOT)
    assert not Path("data").exists()

    fixture = _load_fixture()
    fact_row = fixture["fact"]
    chunk = fixture["chunk"]
    expected = fixture["expected"]
    evidence_id = fact_row["evidence_id"]

    assert chunk["chunk_id"] == evidence_id
    assert chunk["doc_id"] == fact_row["doc_id"]
    assert chunk["page_start"] == chunk["page_end"] == fact_row["page"]
    assert fact_row["source_span"] in chunk["support_span"]

    store = FactStore(":memory:")
    runtime = FakeRuntime([])
    try:
        store.upsert([FinancialFact.from_dict(fact_row)])
        state = run_agentic_rag(
            runtime=runtime,
            answerer=FakeAnswerer(
                [supported_draft(expected["answer"])]
            ),
            llm=FakeLLM([]),
            config=AgentConfig(strict_claim_verification=True, max_claim_retrievals=0),
            query=fixture["query"],
            fact_store=store,
            company_aliases=fixture["company_aliases"],
        )
    finally:
        store.close()

    final = state["final_answer"]
    assert runtime.calls == []
    assert final["abstained"] is False
    assert final["used_evidence_ids"] == [evidence_id]
    assert final["claims"][0]["verification"]["status"] == "ENTAILED"
    assert final["citations"][0]["doc_id"] == fact_row["doc_id"]
    assert final["citations"][0]["page_start"] == expected["citation_page"]
