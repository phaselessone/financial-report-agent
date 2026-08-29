"""Provider-free execution contract for the four canonical profiles."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from run_profile_ablation import main as run_profile_ablation_main
from src.evaluation.offline_contract_runtime import (
    OFFLINE_CONTRACT_MODEL,
    OFFLINE_CONTRACT_PROVIDER,
    OfflineContractAnswerer,
    OfflineContractLLM,
)


def test_offline_answerer_uses_retrieval_evidence_not_gold_fields() -> None:
    answerer = OfflineContractAnswerer()
    result = answerer.answer(
        query="ignored gold test",
        question_type="simple_factual",
        gold_answer="must never be visible",
        retrieval_result={
            "rerank_rows": [
                {
                    "chunk_id": "contract-evidence-1",
                    "evidence_id": "contract-evidence-1",
                    "doc_id": "synthetic-report",
                    "page_start": 3,
                    "text": "Only retrieved evidence may become the answer.",
                }
            ]
        },
    )

    assert result["final_answer"] == "Only retrieved evidence may become the answer."
    assert "must never be visible" not in result["final_answer"]
    assert result["used_evidence_ids"] == ["contract-evidence-1"]
    assert answerer.llm_provider == OFFLINE_CONTRACT_PROVIDER
    assert answerer.llm_model == OFFLINE_CONTRACT_MODEL


def test_offline_llm_is_deterministic_and_fail_closed() -> None:
    llm = OfflineContractLLM()
    first = llm.generate(messages=[])
    second = llm.generate(messages=[{"role": "user", "content": "anything"}])

    assert json.loads(first.content) == {}
    assert first.content == second.content
    assert first.provider == OFFLINE_CONTRACT_PROVIDER
    assert first.model == OFFLINE_CONTRACT_MODEL


def test_offline_contract_requires_both_synthetic_opt_ins(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="--offline-contract requires"):
        run_profile_ablation_main(
            [
                "--cases-path",
                "benchmarks/hard_cases/cases.jsonl",
                "--allow-synthetic-contract",
                "--offline-contract",
                "--output-root",
                str(tmp_path),
            ]
        )
