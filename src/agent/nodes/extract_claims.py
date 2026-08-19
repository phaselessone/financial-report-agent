"""extract_claims node: split the draft answer into claims (checklist v3.0 §P7).

Runs once after synthesis, before verification. The LLM splits the draft
``answer`` into claim candidates; the node classifies each claim's type with
deterministic local heuristics and stamps ``supported`` from the draft's own
whole-answer ``support_validation`` (v1: whole-answer granularity, not per-claim
independent evidence). Abstained drafts yield no claims and no LLM call.
"""

from __future__ import annotations

from typing import Any

from src.agent.claims import classify_claim_type
from src.agent.config import AgentConfig
from src.agent.policies import begin_node, record_llm_response, token_budget_exceeded
from src.generation.payload_parser import parse_model_json

_EXTRACT_CLAIMS_SYSTEM_PROMPT = (
    "You are a financial research claim extractor. Split the given answer into "
    "the smallest independent factual claims. Each claim should be one verifiable "
    "statement. Do not invent facts and do not add anything not present in the answer. "
    'Output one JSON object only: {"claims": [{"text": "...", "claim_type": "EXTRACTED|DERIVED|SYNTHESIZED"}]}. '
    "claim_type is advisory only and may be left as EXTRACTED."
)


def parse_claims(payload: dict[str, Any] | None, *, fallback_text: str) -> list[dict[str, Any]] | None:
    """Validate/normalize an extraction payload.

    Returns ``None`` when no usable claims exist; otherwise a non-empty list of
    ``{"text": ...}`` dicts. Non-dict entries and empty texts are dropped.
    """
    if not isinstance(payload, dict):
        return None
    raw = payload.get("claims")
    if not isinstance(raw, list) or not raw:
        return None
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append({"text": text})
    return out or None


def make_extract_claims(llm, config: AgentConfig):
    def extract_claims(state: dict[str, Any]) -> dict[str, Any]:
        if not begin_node(state, config):
            return state

        draft = state.get("draft_answer") or {}
        if draft.get("abstained"):
            # No verifiable claims from an abstention; no LLM call.
            state.setdefault("claims", [])
            return state
        answer_text = str(draft.get("final_answer") or draft.get("answer") or "").strip()
        if not answer_text:
            state.setdefault("claims", [])
            return state

        if int(state.get("llm_call_count", 0)) >= config.max_llm_calls:
            state["termination_reason"] = "max_llm_calls"
            return state
        if token_budget_exceeded(state, config):
            return state

        used_evidence_ids = [str(eid) for eid in (draft.get("used_evidence_ids") or [])]
        doc_ids = [
            str(c.get("doc_id") or "")
            for c in (draft.get("citations") or [])
            if isinstance(c, dict)
        ]
        supported = bool((draft.get("support_validation") or {}).get("supported", False))

        response = llm.generate(
            messages=[
                {"role": "system", "content": _EXTRACT_CLAIMS_SYSTEM_PROMPT},
                {"role": "user", "content": f"Answer:\n{answer_text}"},
            ],
            temperature=0.0,
            max_tokens=int(getattr(config, "claim_max_tokens", 512) or 512),
        )
        record_llm_response(state, response, node="extract_claims")

        payload = parse_model_json(response.content) or {}
        raw_claims = parse_claims(payload, fallback_text=answer_text)
        if not raw_claims:
            # Fallback: treat the whole answer as a single EXTRACTED claim.
            raw_claims = [{"text": answer_text}]

        claims: list[dict[str, Any]] = []
        for index, item in enumerate(raw_claims, start=1):
            text = item["text"]
            claim_type = classify_claim_type(text, doc_ids)
            claims.append(
                {
                    "claim_id": f"claim-{index}",
                    "text": text,
                    "claim_type": claim_type,
                    "evidence_ids": list(used_evidence_ids),
                    "calculation_id": None,
                    "parent_claim_ids": [],
                    "supported": supported,
                }
            )
        state["claims"] = claims
        return state

    return extract_claims
