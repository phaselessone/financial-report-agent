from __future__ import annotations

from src.agent.config import AgentConfig
from src.agent.graph import run_agentic_rag
from src.agent.nodes.finalize import make_finalize
from src.generation.routing import FALLBACK_ANSWER
from tests.test_agent_flow import (
    FakeAnswerer,
    FakeLLM,
    FakeRuntime,
    make_result,
    make_row,
    supported_draft,
)


def _claim(
    claim_id: str,
    text: str,
    *,
    is_core: bool,
    status: str,
    evidence_ids: list[str] | None = None,
) -> dict[str, object]:
    return {
        "claim_id": claim_id,
        "text": text,
        "claim_type": "EXTRACTED",
        "is_core": is_core,
        "source_step_ids": [],
        "evidence_ids": list(evidence_ids or []),
        "calculation_id": None,
        "parent_claim_ids": [],
        "verification": {
            "status": status,
            "method": "deterministic",
            "score": 1.0 if status == "ENTAILED" else 0.0,
            "reasons": [],
        },
    }


def _state(claims: list[dict[str, object]]) -> dict[str, object]:
    return {
        "query": "问题",
        "step_count": 0,
        "termination_reason": "",
        "claims": claims,
        "calculations": {},
        "claim_verification_summary": {},
        "evidence_pool": {
            "E-core": {
                "evidence_id": "E-core",
                "chunk_id": "E-core",
                "doc_id": "doc-core",
                "page_start": 2,
            }
        },
        "draft_answer": {
            "final_answer": "非核心数字为999。核心结论成立。",
            "answer": "非核心数字为999。核心结论成立。",
            "abstained": False,
            "citations": [],
            "support_validation": {"supported": False},
        },
    }


def test_finalize_uses_explicit_core_flag_and_keeps_verified_core_answer() -> None:
    state = _state(
        [
            _claim("C-noncore", "非核心数字为999。", is_core=False, status="INSUFFICIENT"),
            _claim("C-core", "核心结论成立。", is_core=True, status="ENTAILED", evidence_ids=["E-core"]),
        ]
    )

    result = make_finalize(AgentConfig(strict_claim_verification=True))(state)

    final = result["final_answer"]
    assert final["abstained"] is False
    assert final["partial_answer"] is True
    assert final["final_answer"] == "核心结论成立。"
    assert final["used_evidence_ids"] == ["E-core"]


def test_strict_finalize_rejects_legacy_whole_answer_fallback() -> None:
    state = _state(
        [_claim("C-core", "核心结论成立。", is_core=True, status="INSUFFICIENT")]
    )
    state["legacy_support_fallback"] = True

    result = make_finalize(AgentConfig(strict_claim_verification=True))(state)

    assert result["final_answer"]["abstained"] is True
    assert result["final_answer"]["abstain_reason"] == "claim_verification_failed"


def test_strict_finalize_rebuilds_even_fully_supported_answer_from_claims_only() -> None:
    state = _state(
        [_claim("C-core", "核心结论成立。", is_core=True, status="ENTAILED", evidence_ids=["E-core"])]
    )
    state["draft_answer"]["final_answer"] = "核心结论成立。未抽取的模型附言。"
    state["draft_answer"]["answer"] = "核心结论成立。未抽取的模型附言。"
    state["draft_answer"]["support_validation"] = {"supported": True}

    result = make_finalize(AgentConfig(strict_claim_verification=True))(state)

    assert result["final_answer"]["final_answer"] == "核心结论成立。"
    assert result["final_answer"]["answer"] == "核心结论成立。"
    assert result["final_answer"]["partial_answer"] is False


def test_three_claim_answer_keeps_two_entailed_and_drops_one_noncore_insufficient() -> None:
    state = _state(
        [
            _claim("C-core", "核心结论成立。", is_core=True, status="ENTAILED", evidence_ids=["E-core"]),
            _claim("C-detail", "已验证补充。", is_core=False, status="ENTAILED", evidence_ids=["E-detail"]),
            _claim("C-weak", "未验证补充。", is_core=False, status="INSUFFICIENT"),
        ]
    )
    state["evidence_pool"]["E-detail"] = {
        "evidence_id": "E-detail",
        "chunk_id": "E-detail",
        "doc_id": "doc-detail",
    }
    state["draft_answer"]["final_answer"] = "核心结论成立。已验证补充。未验证补充。"
    state["draft_answer"]["answer"] = state["draft_answer"]["final_answer"]

    result = make_finalize(AgentConfig(strict_claim_verification=True))(state)

    final = result["final_answer"]
    assert final["final_answer"] == "核心结论成立。 已验证补充。"
    assert final["partial_answer"] is True
    assert final["used_evidence_ids"] == ["E-core", "E-detail"]


def test_strict_finalize_never_publishes_supported_draft_without_claims() -> None:
    state = _state([])
    state["draft_answer"]["support_validation"] = {"supported": True}

    result = make_finalize(AgentConfig(strict_claim_verification=True))(state)

    assert result["final_answer"]["abstained"] is True
    assert result["final_answer"]["used_evidence_ids"] == []
    assert result["termination_reason"] == "claim_extraction_missing"


def test_strict_finalize_normalizes_abstained_draft_without_publishing_evidence() -> None:
    state = _state([])
    state["draft_answer"].update(
        {
            "abstained": True,
            "abstain_reason": "insufficient_evidence",
            "support_validation": {"supported": True},
            "used_evidence_ids": ["E-core"],
            "citations": [{"evidence_id": "E-core", "doc_id": "doc-core"}],
            "selected_doc_ids": ["doc-core"],
        }
    )

    result = make_finalize(AgentConfig(strict_claim_verification=True))(state)

    assert result["final_answer"]["abstained"] is True
    assert result["final_answer"]["used_evidence_ids"] == []
    assert result["final_answer"]["citations"] == []
    assert result["final_answer"]["final_answer"] == FALLBACK_ANSWER
    assert result["final_answer"]["support_validation"] == {
        "supported": False,
        "claim_gate": "abstained",
    }
    assert result["draft_answer"]["used_evidence_ids"] == ["E-core"]
    assert result["draft_answer"]["citations"] == [
        {"evidence_id": "E-core", "doc_id": "doc-core"}
    ]


def test_legacy_nonstrict_finalize_can_still_return_supported_draft_without_claims() -> None:
    state = _state([])
    state["draft_answer"]["support_validation"] = {"supported": True}

    result = make_finalize(AgentConfig(strict_claim_verification=False))(state)

    assert result["final_answer"]["abstained"] is False


def test_claim_extraction_budget_exhaustion_abstains_in_strict_graph() -> None:
    state = run_agentic_rag(
        runtime=FakeRuntime(
            [
                make_result(
                    [
                        make_row(
                            chunk_id="E1",
                            doc_id="D1",
                            text="公司持续增加研发投入。",
                        )
                    ]
                )
            ]
        ),
        answerer=FakeAnswerer([supported_draft("公司持续增加研发投入。")]),
        llm=FakeLLM([]),
        config=AgentConfig(
            max_llm_calls=1,
            max_claim_retrievals=0,
            strict_claim_verification=True,
        ),
        query="公司的研发投入如何？",
    )

    assert state["claims"] == []
    assert state["termination_reason"] == "max_llm_calls"
    assert state["final_answer"]["abstained"] is True
