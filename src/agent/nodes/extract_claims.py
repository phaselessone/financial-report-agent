"""extract_claims node: split the draft answer into stable claim records.

Runs once after synthesis, before verification. The node only extracts claim
boundaries and stable ids/types. Claim-specific evidence mapping and three-state
verification happen later in ``verify_answer``.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

from src.agent.claims import (
    ClaimRecord,
    classify_claim_type,
    normalize_id_list,
    stable_claim_id,
    valid_claim_type,
)
from src.agent.config import AgentConfig
from src.agent.policies import begin_node, record_llm_response, token_budget_exceeded
from src.generation.payload_parser import parse_model_json

_EXTRACT_CLAIMS_SYSTEM_PROMPT = (
    "You are a financial research claim extractor. Split the given answer into "
    "the smallest independent factual claims. Each claim should be one verifiable "
    "statement. Do not invent facts and do not add anything not present in the answer. "
    'Output one JSON object only: {"claims": [{"id": "local-id", "text": "...", '
    '"claim_type": "EXTRACTED|DERIVED|SYNTHESIZED", "is_core": true, '
    '"source_step_ids": [], "parent_claim_ids": []}]}. '
    "Use parent_claim_ids only for claims whose truth depends on earlier claims."
)


def _calculation_id_for_claim(text: str, calculations: Any) -> str | None:
    if not isinstance(calculations, dict):
        return None
    compact = re.sub(r"\s+", "", str(text or ""))
    claim_numbers = _claim_numeric_values(text)
    for calculation_id, record in calculations.items():
        if not isinstance(record, dict):
            continue
        result = record.get("result") or {}
        candidates = [record.get("formatted"), record.get("value"), result.get("formatted"), result.get("value")]
        for candidate in candidates:
            token = str(candidate or "").strip()
            if token and (token in text or re.sub(r"\s+", "", token) in compact):
                return str(calculation_id)
        # Presentation often rounds ``20.0000%`` to ``20%``.  Compare the
        # rendered percentage numerically instead of requiring exact text.
        result_value = result.get("value")
        result_unit = str(result.get("unit") or "")
        if claim_numbers and result_value is not None:
            try:
                calculated = Decimal(str(result_value))
                if result_unit in {"", "%", "pp"}:
                    calculated_percent = calculated * 100 if result_unit == "" else calculated
                    if any(abs(value - calculated_percent) <= Decimal("0.0001") for value in claim_numbers):
                        return str(calculation_id)
            except (InvalidOperation, ValueError, TypeError):
                pass
    # If there is exactly one successful calculation and the claim is marked
    # derived, it is the only safe candidate even when the model paraphrases
    # the operation/result wording.
    successful = [
        str(calculation_id)
        for calculation_id, record in calculations.items()
        if isinstance(record, dict) and str(record.get("status") or "").upper() == "SUCCESS"
    ]
    if len(successful) == 1 and any(marker in compact.lower() for marker in ("同比", "环比", "增速", "增长率", "占比", "毛利率", "净利率", "cagr", "yoy")):
        return successful[0]
    return None


def _calculation_step_ids(
    calculation_id: str | None,
    state: dict[str, Any],
) -> tuple[str, ...]:
    """Return the reasoning step that produced a calculation proof object."""

    if not calculation_id:
        return ()
    return tuple(
        dict.fromkeys(
            str(result.get("step_id") or "").strip()
            for result in (state.get("reasoning_step_results") or [])
            if isinstance(result, dict)
            and str(result.get("calculation_id") or "") == calculation_id
            and str(result.get("step_id") or "").strip()
        )
    )


def _claim_numeric_values(text: str) -> list[Decimal]:
    values: list[Decimal] = []
    for token in re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", str(text or "")):
        try:
            value = Decimal(token.replace(",", ""))
        except InvalidOperation:
            continue
        # Years are coordinates, not result values.
        if value == value.to_integral_value() and 1900 <= value <= 2100:
            continue
        values.append(value)
    return values


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
        local_id = str(item.get("id") or item.get("claim_id") or "").strip()
        out.append(
            {
                "local_id": local_id,
                "text": text,
                "claim_type": valid_claim_type(item.get("claim_type")),
                # Fail closed: if extraction cannot reliably say otherwise,
                # the claim is core and cannot be dropped as decoration.
                "is_core": item.get("is_core") if isinstance(item.get("is_core"), bool) else True,
                "source_step_ids": normalize_id_list(item.get("source_step_ids")),
                "parent_claim_ids": normalize_id_list(item.get("parent_claim_ids")),
            }
        )
    return out or None


_ASSERTION_MARKERS = (
    "为",
    "是",
    "达到",
    "增长",
    "下降",
    "同比",
    "环比",
    "收入",
    "营收",
    "利润",
    "率",
    "指出",
    "认为",
    "预计",
    "高于",
    "低于",
    "超过",
)


def _looks_assertive_clause(clause: str) -> bool:
    stripped = clause.strip()
    if stripped.startswith(("针对", "关于", "就“", '就"')) and stripped.endswith(
        ("”", '"', "’", "'")
    ):
        return False
    return any(marker in stripped for marker in _ASSERTION_MARKERS)


def _split_claim_sentences(text: str) -> list[str]:
    """Split terminal punctuation while ignoring punctuation inside quotes."""

    quote_pairs = {"“": "”", "「": "」", "『": "』", '"': '"', "'": "'"}
    expected_closers: list[str] = []
    buffer: list[str] = []
    out: list[str] = []
    for char in text:
        if expected_closers and char == expected_closers[-1]:
            expected_closers.pop()
        elif char in quote_pairs:
            expected_closers.append(quote_pairs[char])
        if char not in "\r\n":
            buffer.append(char)
        if (char in "。！？!?；;\r\n") and not expected_closers:
            sentence = "".join(buffer).strip()
            if sentence:
                out.append(sentence)
            buffer = []
    tail = "".join(buffer).strip()
    if tail:
        out.append(tail)
    return out


def deterministic_atomic_claims(text: str) -> list[dict[str, Any]]:
    """Conservative local fallback when the extractor emits invalid JSON.

    Sentence boundaries are always safe split points.  Commas are used only
    when every resulting clause independently looks assertive; this separates
    common financial constructions such as ``营收100亿元，同比增长20%`` while
    preserving coordinate prefixes such as ``截至2025年，营收...``.
    """

    normalized = str(text or "").strip()
    if not normalized:
        return []
    sentences = _split_claim_sentences(normalized)
    claims: list[dict[str, Any]] = []
    for sentence in sentences:
        clauses = [part.strip() for part in re.split(r"[，,]", sentence) if part.strip()]
        split_clauses = (
            len(clauses) > 1
            and all(_looks_assertive_clause(clause) for clause in clauses)
        )
        pieces = clauses if split_clauses else [sentence]
        for piece in pieces:
            if piece[-1:] not in "。！？!?；;":
                piece = f"{piece}。"
            claims.append({"text": piece})
    return claims


def _deterministic_comparison_claims(
    state: dict[str, Any], *, answer_text: str
) -> list[dict[str, Any]] | None:
    """Build the auditable claim DAG for a calculator-backed comparison.

    A deterministic reasoning conclusion must not be sent back through an LLM
    extractor: doing so can erase its calculation and dependency provenance.
    The plan already contains the exact two growth calculations and compare
    step, so materialize that trace directly as two non-core DERIVED parents
    and one core SYNTHESIZED conclusion.
    """

    draft = state.get("draft_answer") or {}
    support = draft.get("support_validation") or {}
    conclusion = state.get("reasoning_conclusion") or {}
    coverage = state.get("reasoning_coverage") or {}
    if (
        support.get("reasoning_gate") != "deterministic_compare"
        or not coverage.get("deterministic_conclusion_allowed")
        or not isinstance(conclusion, dict)
        or str(conclusion.get("answer") or "").strip() != answer_text
    ):
        return None

    raw_steps = (state.get("reasoning_plan") or {}).get("steps") or []
    steps = {
        str(step.get("step_id") or ""): step
        for step in raw_steps
        if isinstance(step, dict) and step.get("step_id")
    }
    compare_step = next(
        (
            step
            for step in steps.values()
            if str(step.get("kind") or "").upper() == "COMPARE"
            and str((state.get("reasoning_step_artifacts") or {}).get(
                str(step.get("step_id") or ""), {}
            ).get("calculation", {}).get("calculation_id") or "")
            == str(conclusion.get("calculation_id") or "")
        ),
        None,
    )
    if not isinstance(compare_step, dict):
        return None

    artifacts = state.get("reasoning_step_artifacts") or {}
    parent_claims: list[ClaimRecord] = []
    for dependency_id in compare_step.get("depends_on") or []:
        dependency_id = str(dependency_id)
        step = steps.get(dependency_id) or {}
        artifact = artifacts.get(dependency_id) or {}
        calculation = artifact.get("calculation") or {}
        result = calculation.get("result") or {}
        company = str((step.get("arguments") or {}).get("company") or "").strip()
        formatted = str(result.get("formatted") or "").strip()
        calculation_id = str(calculation.get("calculation_id") or "").strip()
        if (
            str(step.get("kind") or "").upper() != "CALCULATE"
            or not company
            or not formatted
            or not calculation_id
            or str(calculation.get("status") or "").upper() != "SUCCESS"
        ):
            return None
        text = f"{company}营业收入同比增长率为{formatted}。"
        parent_claims.append(
            ClaimRecord(
                claim_id=stable_claim_id(text),
                text=text,
                claim_type="DERIVED",
                is_core=False,
                source_step_ids=(dependency_id,),
                calculation_id=calculation_id,
            )
        )
    if len(parent_claims) != 2:
        return None

    compare_step_id = str(compare_step.get("step_id") or "")
    comparison_id = str(conclusion.get("calculation_id") or "").strip()
    if not comparison_id:
        return None
    core = ClaimRecord(
        claim_id=stable_claim_id(answer_text),
        text=answer_text,
        claim_type="SYNTHESIZED",
        is_core=True,
        source_step_ids=(compare_step_id,),
        calculation_id=comparison_id,
        parent_claim_ids=tuple(claim.claim_id for claim in parent_claims),
    )
    return [claim.to_dict() for claim in (*parent_claims, core)]


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

        deterministic_claims = _deterministic_comparison_claims(
            state, answer_text=answer_text
        )
        if deterministic_claims is not None:
            state["claims"] = deterministic_claims
            return state

        if int(state.get("llm_call_count", 0)) >= config.max_llm_calls:
            state["termination_reason"] = "max_llm_calls"
            return state
        if token_budget_exceeded(state, config):
            return state

        doc_ids = [
            str(c.get("doc_id") or "")
            for c in (draft.get("citations") or [])
            if isinstance(c, dict)
        ]
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
            # Invalid extractor output must not turn a multi-assertion answer
            # into one pseudo-atomic claim.  Split conservatively and let the
            # normal verifier fail closed on every resulting assertion.
            raw_claims = deterministic_atomic_claims(answer_text)
            state["claim_extraction_fallback"] = "deterministic_atomic_split"
            if not raw_claims:
                state["termination_reason"] = (
                    state.get("termination_reason") or "claim_extraction_failed"
                )
                state["claims"] = []
                return state

        local_to_stable = {
            str(item.get("local_id") or stable_claim_id(item["text"])): stable_claim_id(item["text"])
            for item in raw_claims
        }
        claims: list[dict[str, Any]] = []
        for item in raw_claims:
            text = item["text"]
            # Whole-answer citations cannot safely classify an atomic claim as
            # synthesized.  Use them only after claim-specific mapping; here
            # the deterministic heuristic is intentionally limited to derived
            # markers and the model's explicit atomic type/parents.
            heuristic_type = classify_claim_type(text, [])
            advisory_type = valid_claim_type(item.get("claim_type"))
            claim_type = heuristic_type if heuristic_type != "EXTRACTED" else (advisory_type or heuristic_type)
            if item.get("parent_claim_ids") and claim_type == "EXTRACTED":
                claim_type = "SYNTHESIZED"
            calculation_id = _calculation_id_for_claim(text, state.get("calculations")) if claim_type == "DERIVED" else None
            parents = tuple(local_to_stable.get(parent_id, parent_id) for parent_id in item.get("parent_claim_ids") or ())
            source_step_ids = tuple(item.get("source_step_ids") or ())
            if claim_type == "DERIVED" and calculation_id and not source_step_ids:
                source_step_ids = _calculation_step_ids(calculation_id, state)
            claims.append(
                ClaimRecord(
                    claim_id=stable_claim_id(text),
                    text=text,
                    claim_type=claim_type,
                    is_core=bool(item.get("is_core", True)),
                    source_step_ids=source_step_ids,
                    calculation_id=calculation_id,
                    parent_claim_ids=parents,
                ).to_dict()
            )
        state["claims"] = claims
        return state

    return extract_claims
