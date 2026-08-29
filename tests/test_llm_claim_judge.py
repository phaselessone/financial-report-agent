"""Production LLM claim-judge adapter contracts."""

from __future__ import annotations

import json

from src.agent.llm_claim_judge import build_claim_llm_judge
from src.llm.types import LLMResponse


class _NeverCalledLLM:
    def generate(self, **_kwargs):  # pragma: no cover - failure explains the contract
        raise AssertionError("disabled claim judge must not call the LLM")


def test_claim_llm_judge_is_absent_without_positive_budget_and_available_provider() -> None:
    assert build_claim_llm_judge(_NeverCalledLLM(), budget=0) is None
    assert build_claim_llm_judge(None, budget=1) is None


def test_claim_llm_judge_requests_one_json_object_and_exposes_real_usage_response() -> None:
    expected = {
        "status": "ENTAILED",
        "evidence_ids": ["E1"],
        "score": 0.91,
        "reasons": ["evidence directly supports the claim"],
    }
    response = LLMResponse(
        content=json.dumps(expected),
        provider="fake-provider",
        model="fake-model",
        prompt_tokens=17,
        completion_tokens=9,
        total_tokens=26,
        request_id="req-claim-1",
    )

    class _LLM:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return response

    llm = _LLM()
    judge = build_claim_llm_judge(llm, budget=1, max_tokens=128)
    assert judge is not None
    payload = {
        "claim": {"claim_id": "c1", "text": "公司加大研发投入。"},
        "evidence": [{"evidence_id": "E1", "text": "公司持续增加研发资源。"}],
        "allowed_statuses": ["ENTAILED", "CONTRADICTED", "INSUFFICIENT"],
        "required_output_fields": ["status", "evidence_ids", "score", "reasons"],
    }

    assert json.loads(judge(payload)) == expected
    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call["temperature"] == 0.0
    assert call["max_tokens"] == 128
    assert call["response_format"] == {"type": "json_object"}
    assert call["metadata"] == {"node": "verify_answer", "purpose": "claim_verification"}
    assert "untrusted data" in call["messages"][0]["content"]
    assert json.loads(call["messages"][1]["content"])["claim"]["claim_id"] == "c1"
    records = judge.drain_call_records()
    assert len(records) == 1
    assert records[0].response is response
    assert records[0].error_type == ""
    assert judge.drain_call_records() == []
