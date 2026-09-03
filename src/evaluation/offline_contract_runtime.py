"""Provider-free runtime pieces for the synthetic four-profile contract.

This module is deliberately narrow.  It may only be selected by the profile
ablation CLI for the fixed synthetic contract oracle.  It reads the rows
returned by that oracle, never reads benchmark gold answers, and never makes a
network or model-download request.  Consequently its outputs prove execution
and bundle plumbing only; they are not quality or performance evidence.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from src.llm.types import LLMResponse


OFFLINE_CONTRACT_PROVIDER = "offline-deterministic"
OFFLINE_CONTRACT_MODEL = "synthetic-contract-evidence-echo"
OFFLINE_CONTRACT_REVISION = "offline-contract-v1"


def _evidence_rows(retrieval_result: Mapping[str, Any]) -> list[dict[str, Any]]:
    for key in ("rerank_rows", "hybrid_rows", "bm25_rows", "dense_rows"):
        rows = retrieval_result.get(key)
        if isinstance(rows, list) and rows:
            return [dict(row) for row in rows if isinstance(row, Mapping)]
    return []


def _citation(row: Mapping[str, Any]) -> dict[str, Any]:
    evidence_id = str(row.get("evidence_id") or row.get("chunk_id") or "")
    doc_id = str(row.get("doc_id") or row.get("source") or "synthetic-contract")
    page = int(row.get("page") or row.get("page_start") or 1)
    text = str(row.get("support_span") or row.get("text") or row.get("child_text") or "")
    return {
        "evidence_id": evidence_id,
        "chunk_id": str(row.get("chunk_id") or evidence_id),
        "doc_id": doc_id,
        "file_name": str(row.get("file_name") or doc_id),
        "page": page,
        "page_start": int(row.get("page_start") or page),
        "page_end": int(row.get("page_end") or page),
        "text": str(row.get("text") or text),
        "support_span": text,
    }


class OfflineContractAnswerer:
    """Echo retrieved synthetic evidence through the production answer seam.

    ``question_type`` is used only to honor the explicit ``must_abstain``
    contract category.  No gold answer, required claim, or expected evidence ID
    is visible to this class.
    """

    llm_provider = OFFLINE_CONTRACT_PROVIDER
    llm_model = OFFLINE_CONTRACT_MODEL

    def __init__(self) -> None:
        self.llm_calls: list[LLMResponse] = []

    def answer(self, **kwargs: Any) -> dict[str, Any]:
        query = str(kwargs.get("query") or "")
        question_type = str(kwargs.get("question_type") or "")
        retrieval_result = kwargs.get("retrieval_result")
        rows = _evidence_rows(retrieval_result) if isinstance(retrieval_result, Mapping) else []
        must_abstain = question_type == "must_abstain"

        if must_abstain or not rows:
            final_answer = "信息不足，暂时无法给出可靠答案。"
            citations: list[dict[str, Any]] = []
            used_ids: list[str] = []
            abstained = True
        else:
            citations = [_citation(row) for row in rows]
            citations = [item for item in citations if item["evidence_id"]]
            first = citations[0] if citations else None
            if first is None or not str(first.get("support_span") or "").strip():
                final_answer = "信息不足，暂时无法给出可靠答案。"
                citations = []
                used_ids = []
                abstained = True
            else:
                final_answer = str(first["support_span"]).strip()
                used_ids = [str(item["evidence_id"]) for item in citations]
                abstained = False

        response = LLMResponse(
            content=final_answer,
            provider=OFFLINE_CONTRACT_PROVIDER,
            model=OFFLINE_CONTRACT_MODEL,
            prompt_tokens=0,
            completion_tokens=0,
        )
        self.llm_calls.append(response)
        return {
            "query": query,
            "question_type": question_type,
            "answer_mode": "synthetic_contract_evidence_echo",
            "final_answer": final_answer,
            "abstained": abstained,
            "abstain_reason": "synthetic_contract_must_abstain"
            if must_abstain
            else ("no_contract_evidence" if abstained else None),
            "used_evidence_ids": used_ids,
            "citations": citations,
            "support_validation": {
                "supported": not abstained,
                "method": "synthetic_contract_exact_evidence_echo",
                "matched_numeric_tokens": [],
            },
            "selected_doc_ids": [str(item["doc_id"]) for item in citations],
        }


class OfflineContractLLM:
    """Fail-closed deterministic LLM seam used by graph-only helper nodes."""

    provider_name = OFFLINE_CONTRACT_PROVIDER
    model_name = OFFLINE_CONTRACT_MODEL

    def generate(self, **_: Any) -> LLMResponse:
        # An empty object intentionally makes query rewriting, decomposition,
        # claim extraction and qualitative judging use their deterministic,
        # fail-closed fallbacks.
        return LLMResponse(
            content=json.dumps({}, ensure_ascii=False),
            provider=OFFLINE_CONTRACT_PROVIDER,
            model=OFFLINE_CONTRACT_MODEL,
            prompt_tokens=0,
            completion_tokens=0,
        )


__all__ = [
    "OFFLINE_CONTRACT_MODEL",
    "OFFLINE_CONTRACT_PROVIDER",
    "OFFLINE_CONTRACT_REVISION",
    "OfflineContractAnswerer",
    "OfflineContractLLM",
]
