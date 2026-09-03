"""verify_answer node: reuse the existing validate_answer_support verdict.

M1 adds claim-specific verification before the legacy whole-answer retry policy.
Each claim gets claim-specific evidence ids and a three-state verification
record. To preserve compatibility, the existing whole-answer support verdict
remains the retry/finalization gate unless a claim is explicitly contradicted.
"""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

from src.agent.claim_verifier import ENTAILED, verify_claim
from src.agent.claims import claim_structure_errors
from src.agent.config import AgentConfig
from src.agent.policies import begin_node, record_llm_failure, record_llm_response


def _evidence_rows(state: dict[str, Any]) -> list[dict[str, Any]]:
    pool = state.get("evidence_pool") or {}
    if isinstance(pool, Mapping) and pool:
        rows = [row for row in pool.values() if isinstance(row, dict)]
    else:
        result = state.get("last_retrieval_result") or {}
        rows = list(result.get("rerank_rows") or result.get("hybrid_rows") or [])
        rows = [row for row in rows if isinstance(row, dict)]
    normalized: list[dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        copied.setdefault("evidence_id", copied.get("chunk_id"))
        normalized.append(copied)
    return normalized


def _merge_retrieval_rows(state: dict[str, Any], result: Mapping[str, Any] | None) -> int:
    if not isinstance(result, Mapping):
        return 0
    rows = [row for row in (result.get("rerank_rows") or result.get("hybrid_rows") or []) if isinstance(row, dict)]
    pool = state.setdefault("evidence_pool", {})
    seen = state.setdefault("seen_chunk_ids", set())
    added = 0
    for row in rows:
        row_id = str(row.get("chunk_id") or row.get("evidence_id") or "").strip()
        if not row_id:
            continue
        row = dict(row)
        row.setdefault("evidence_id", row_id)
        if row_id not in pool:
            added += 1
        pool[row_id] = row
        seen.add(row_id)
    if rows:
        state["last_retrieval_result"] = dict(result)
    return added


def _verification_claim(
    claim: Mapping[str, Any],
    *,
    state: Mapping[str, Any],
    rows: list[dict[str, Any]],
    company_aliases: Mapping[str, list[str]] | None,
) -> dict[str, Any]:
    """Resolve a claim pronoun only when exactly one known issuer is grounded.

    The stored/output claim text is never rewritten.  This adapter enriches a
    copy used by the verifier, and only when a canonical issuer alias is present
    in the original query and current evidence.  Ambiguous or ungrounded cases
    remain unchanged and therefore fail closed.
    """
    enriched = dict(claim)
    text = str(claim.get("text") or "")
    if not any(token in text for token in ("其", "该公司", "公司")):
        return enriched
    aliases = company_aliases or {}
    query = str(state.get("query") or "")
    evidence_text = "\n".join(
        str(row.get("support_span") or row.get("text") or row.get("child_text") or "") for row in rows
    )
    grounded: list[str] = []
    for canonical, raw_aliases in aliases.items():
        names = [str(canonical), *(str(alias) for alias in (raw_aliases or []))]
        if any(name and name in query for name in names) and any(name and name in evidence_text for name in names):
            grounded.append(str(canonical))
    if not grounded:
        generic = {
            "市场", "行业", "逻辑", "看法", "增长", "问题", "营业收入", "营收", "收入", "净利润",
            "毛利率", "净利率", "同比", "环比", "是多少", "如何", "哪些", "公司", "报告",
        }
        spans = re.findall(r"[\u4e00-\u9fff]{2,12}", query)
        for span in spans:
            if span in generic or len(span) < 2:
                continue
            if span in evidence_text:
                grounded.append(span)
    grounded = list(dict.fromkeys(grounded))
    if len(grounded) != 1:
        return enriched
    canonical = grounded[0]
    resolved = text.replace("该公司", canonical).replace("其", canonical, 1)
    if resolved == text and "公司" in text:
        resolved = text.replace("公司", canonical, 1)
    enriched["text"] = resolved
    enriched["entity"] = canonical
    return enriched


def _claim_verification_order(claims: list[Any]) -> list[Mapping[str, Any]]:
    """Return parents before children while preserving stable input order."""
    mappings = [claim for claim in claims if isinstance(claim, Mapping)]
    by_id = {str(claim.get("claim_id") or ""): claim for claim in mappings}
    ordered: list[Mapping[str, Any]] = []
    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(claim: Mapping[str, Any]) -> None:
        claim_id = str(claim.get("claim_id") or "")
        if claim_id in visited or claim_id in visiting:
            return
        visiting.add(claim_id)
        for parent_id in claim.get("parent_claim_ids") or []:
            parent = by_id.get(str(parent_id))
            if parent is not None:
                visit(parent)
        visiting.discard(claim_id)
        visited.add(claim_id)
        ordered.append(claim)

    for claim in mappings:
        visit(claim)
    return ordered


_CALCULATION_STEP_KINDS = frozenset({"CALCULATE", "COMPARE"})


def _step_index(state: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    plan = state.get("reasoning_plan") or {}
    if not isinstance(plan, Mapping):
        return {}
    return {
        str(step.get("step_id") or ""): step
        for step in (plan.get("steps") or [])
        if isinstance(step, Mapping) and str(step.get("step_id") or "")
    }


def _step_result_index(state: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(result.get("step_id") or ""): result
        for result in (state.get("reasoning_step_results") or [])
        if isinstance(result, Mapping) and str(result.get("step_id") or "")
    }


def _calculation_record(
    calculations: Any, calculation_id: str
) -> Mapping[str, Any] | None:
    if not isinstance(calculations, Mapping):
        return None
    record = calculations.get(calculation_id)
    return record if isinstance(record, Mapping) else None


def _calculation_integrity_reasons(
    calculations: Any,
    calculation_id: str,
    *,
    prefix: str,
) -> list[str]:
    record = _calculation_record(calculations, calculation_id)
    if record is None:
        return [f"{prefix}_calculation_not_found:{calculation_id}"]
    reasons: list[str] = []
    if str(record.get("status") or "").upper() != "SUCCESS":
        reasons.append(f"{prefix}_calculation_not_successful:{calculation_id}")
    if record.get("verified") is not True:
        reasons.append(f"{prefix}_calculation_not_verified:{calculation_id}")
    if record.get("provenance_complete") is not True:
        reasons.append(f"{prefix}_calculation_provenance_incomplete:{calculation_id}")
    return reasons


def _calculation_backed_synthesis_reasons(
    claim: Mapping[str, Any],
    *,
    state: Mapping[str, Any],
    verified_by_id: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Fail closed when a planned calculation conclusion loses its proof.

    Text is deliberately irrelevant here.  A core SYNTHESIZED claim is
    calculation-backed when its source step is CALCULATE/COMPARE, or when its
    parent claims already carry calculation lineage.  Such a claim must bind
    to the successful step result and to an explicitly verified,
    provenance-complete CalculationRecord.  COMPARE additionally binds the
    record's upstream calculation ids to the parent claims and plan edges.
    """
    if str(claim.get("claim_type") or "").upper() != "SYNTHESIZED":
        return []
    if claim.get("is_core") is False:
        return []

    steps = _step_index(state)
    results = _step_result_index(state)
    source_step_ids = [
        str(step_id)
        for step_id in (claim.get("source_step_ids") or [])
        if str(step_id)
    ]
    source_steps = [steps[step_id] for step_id in source_step_ids if step_id in steps]
    calculation_source_steps = [
        step
        for step in source_steps
        if str(step.get("kind") or "").upper() in _CALCULATION_STEP_KINDS
    ]

    parent_ids = [
        str(parent_id)
        for parent_id in (claim.get("parent_claim_ids") or [])
        if str(parent_id)
    ]
    parent_claims = [verified_by_id[parent_id] for parent_id in parent_ids if parent_id in verified_by_id]
    parent_calculation_pairs = [
        (parent, str(parent.get("calculation_id") or ""))
        for parent in parent_claims
        if str(parent.get("calculation_id") or "")
    ]
    parent_calculation_ids = [
        calculation_id for _, calculation_id in parent_calculation_pairs
    ]
    parent_calculation_steps = [
        steps[str(step_id)]
        for parent in parent_claims
        for step_id in (parent.get("source_step_ids") or [])
        if str(step_id) in steps
        and str(steps[str(step_id)].get("kind") or "").upper()
        in _CALCULATION_STEP_KINDS
    ]

    if not calculation_source_steps and not parent_calculation_ids and not parent_calculation_steps:
        return []
    if parent_calculation_ids and not calculation_source_steps:
        return ["calculation_backed_claim_missing_source_step"]

    calculation_id = str(claim.get("calculation_id") or "").strip()
    if not calculation_id:
        return ["calculation_backed_claim_missing_calculation"]

    calculations = state.get("calculations")
    reasons = _calculation_integrity_reasons(
        calculations,
        calculation_id,
        prefix="claim",
    )
    record = _calculation_record(calculations, calculation_id)
    if record is None:
        return reasons

    # The claim's direct source step must be the step that produced this exact
    # calculation.  This prevents a valid but unrelated record from being
    # attached after extraction.
    for step in calculation_source_steps:
        step_id = str(step.get("step_id") or "")
        result = results.get(step_id) or {}
        if str(result.get("status") or "").upper() != "SUCCESS":
            reasons.append(f"claim_source_step_not_successful:{step_id}")
        if str(result.get("calculation_id") or "") != calculation_id:
            reasons.append(f"claim_calculation_not_bound_to_source_step:{step_id}")

    compare_steps = [
        step
        for step in calculation_source_steps
        if str(step.get("kind") or "").upper() == "COMPARE"
    ]
    if compare_steps:
        if len(parent_claims) != len(parent_ids) or len(parent_calculation_ids) != len(parent_ids):
            reasons.append("comparison_parent_calculation_lineage_incomplete")
        for parent, parent_calculation_id in parent_calculation_pairs:
            reasons.extend(
                _calculation_integrity_reasons(
                    calculations,
                    parent_calculation_id,
                    prefix="parent",
                )
            )
            for parent_step_id in parent.get("source_step_ids") or []:
                parent_step_id = str(parent_step_id)
                parent_step = steps.get(parent_step_id) or {}
                if str(parent_step.get("kind") or "").upper() not in _CALCULATION_STEP_KINDS:
                    continue
                parent_result = results.get(parent_step_id) or {}
                if (
                    str(parent_result.get("status") or "").upper() != "SUCCESS"
                    or str(parent_result.get("calculation_id") or "")
                    != parent_calculation_id
                ):
                    reasons.append(
                        f"parent_calculation_not_bound_to_source_step:{parent_step_id}"
                    )

        expected_parent_ids = set(parent_calculation_ids)
        actual_parent_ids = {
            str(item)
            for item in (record.get("input_calculation_ids") or [])
            if str(item)
        }
        if not expected_parent_ids or actual_parent_ids != expected_parent_ids:
            reasons.append("comparison_parent_calculation_lineage_mismatch")

        for compare_step in compare_steps:
            expected_source_steps = {
                str(item)
                for item in (compare_step.get("depends_on") or [])
                if str(item)
            }
            actual_source_steps = {
                str(item)
                for item in (record.get("source_step_ids") or [])
                if str(item)
            }
            parent_source_steps = {
                str(item)
                for parent in parent_claims
                for item in (parent.get("source_step_ids") or [])
                if str(item)
            }
            if (
                not expected_source_steps
                or actual_source_steps != expected_source_steps
                or parent_source_steps != expected_source_steps
            ):
                reasons.append("comparison_plan_lineage_mismatch")

    return list(dict.fromkeys(reasons))


class _JudgeInvocationTracker:
    """Count actual judge calls without changing the verifier interface."""

    def __init__(self, judge: Any) -> None:
        self._judge = judge
        self.calls = 0

    def __call__(self, payload: Mapping[str, Any]) -> Any:
        self.calls += 1
        return self._judge(payload)


def _remaining_llm_judge_budget(
    state: Mapping[str, Any], config: AgentConfig | None, llm_judge: Any
) -> int:
    if config is None or llm_judge is None:
        return 0
    remaining = max(
        0,
        int(getattr(config, "claim_llm_budget", 0) or 0)
        - int(state.get("claim_llm_judge_count", 0)),
    )
    if int(state.get("llm_call_count", 0)) >= int(getattr(config, "max_llm_calls", 0) or 0):
        return 0
    token_budget = int(getattr(config, "max_total_tokens", 0) or 0)
    if token_budget > 0 and int(state.get("total_tokens", 0)) >= token_budget:
        return 0
    return remaining


def _verify_claim_once(
    state: dict[str, Any],
    candidate: Mapping[str, Any],
    rows: list[dict[str, Any]],
    *,
    calculations: Any,
    semantic_scorer: Any,
    llm_judge: Any,
    semantic_threshold: float,
    config: AgentConfig | None,
) -> dict[str, Any]:
    """Verify once and account every judge invocation before any retry."""
    llm_budget = _remaining_llm_judge_budget(state, config, llm_judge)
    tracked_judge = _JudgeInvocationTracker(llm_judge) if llm_judge is not None else None
    try:
        return verify_claim(
            candidate,
            rows,
            calculation_lookup=calculations,
            semantic_scorer=semantic_scorer,
            llm_judge=tracked_judge,
            semantic_threshold=semantic_threshold,
            llm_budget=llm_budget,
        )
    finally:
        invocation_count = tracked_judge.calls if tracked_judge is not None else 0
        if invocation_count:
            state["claim_llm_judge_count"] = int(
                state.get("claim_llm_judge_count", 0)
            ) + invocation_count
        drain_records = getattr(llm_judge, "drain_call_records", None)
        if callable(drain_records):
            for record in drain_records():
                response = getattr(record, "response", None)
                if response is not None:
                    record_llm_response(state, response, node="verify_answer")
                else:
                    record_llm_failure(
                        state,
                        node="verify_answer",
                        provider=str(getattr(record, "provider", "") or "unknown"),
                        model=str(getattr(record, "model", "") or "unknown"),
                        error_type=str(getattr(record, "error_type", "") or "LLMProviderError"),
                        latency_ms=float(getattr(record, "latency_ms", 0.0) or 0.0),
                        retries=int(getattr(record, "retries", 0) or 0),
                    )


def _verify_claims(
    state: dict[str, Any],
    *,
    runtime: Any = None,
    config: AgentConfig | None = None,
    company_aliases: Mapping[str, list[str]] | None = None,
    semantic_scorer: Any = None,
    llm_judge: Any = None,
    allow_claim_retrieval: bool = True,
) -> dict[str, int]:
    claims = list(state.get("claims") or [])
    rows = _evidence_rows(state)
    calculations = state.get("calculations")
    structure_errors = claim_structure_errors(
        [claim for claim in claims if isinstance(claim, Mapping)]
    )
    state["claim_structure_errors"] = {
        claim_id: reasons for claim_id, reasons in structure_errors.items() if reasons
    }
    # Multi-hop retrieval is deliberately fixed by the decomposition plan: each
    # RAG sub-question gets one bounded retrieval.  A claim-specific lookup at
    # this point would consume another runtime call, change the trajectory, and
    # make the dependency graph non-reproducible.  Single-hop runs retain the
    # M1 one-shot claim retrieval budget.
    allow_claim_retrieval = bool(allow_claim_retrieval) and not bool(state.get("is_multi_hop"))
    verified_by_id: dict[str, dict[str, Any]] = {}
    summary = {
        "claim_count": len(claims),
        "entailed_count": 0,
        "contradicted_count": 0,
        "insufficient_count": 0,
    }
    for claim in _claim_verification_order(claims):
        candidate = _verification_claim(claim, state=state, rows=rows, company_aliases=company_aliases)
        semantic_threshold = float(
            getattr(semantic_scorer, "threshold", None)
            or (getattr(config, "claim_semantic_threshold", 0.85) if config is not None else 0.85)
        )
        claim_errors = structure_errors.get(str(claim.get("claim_id") or ""), [])
        strict = bool(getattr(config, "strict_claim_verification", True) if config is not None else True)
        parent_ids = [str(item) for item in (claim.get("parent_claim_ids") or []) if str(item)]
        parent_failures = [
            parent_id
            for parent_id in parent_ids
            if str(
                ((verified_by_id.get(parent_id) or {}).get("verification") or {}).get("status")
                or "INSUFFICIENT"
            )
            != ENTAILED
        ]
        dependency_reasons: list[str] = []
        if strict and str(claim.get("claim_type") or "").upper() == "SYNTHESIZED":
            if not parent_ids:
                dependency_reasons.append("synthesized_claim_missing_parents")
            dependency_reasons.extend(
                f"parent_claim_not_entailed:{parent_id}" for parent_id in parent_failures
            )
            dependency_reasons.extend(
                _calculation_backed_synthesis_reasons(
                    claim,
                    state=state,
                    verified_by_id=verified_by_id,
                )
            )
        if claim_errors or dependency_reasons:
            result = {
                "evidence_ids": [],
                "verification": {
                    "status": "INSUFFICIENT",
                    "method": "deterministic",
                    "score": 0.0,
                    "reasons": list(dict.fromkeys([*claim_errors, *dependency_reasons])),
                },
            }
        else:
            result = _verify_claim_once(
                state,
                candidate,
                rows,
                calculations=calculations,
                semantic_scorer=semantic_scorer,
                llm_judge=llm_judge,
                semantic_threshold=semantic_threshold,
                config=config,
            )
        if (
            result["verification"]["status"] == "INSUFFICIENT"
            and not claim_errors
            and not dependency_reasons
            and allow_claim_retrieval
            and runtime is not None
            and config is not None
            and int(state.get("claim_retrieval_count", 0)) < int(getattr(config, "max_claim_retrievals", 0) or 0)
        ):
            query = str(claim.get("text") or "").strip()
            if query:
                try:
                    retrieval = runtime.search(query)
                    added = _merge_retrieval_rows(state, retrieval)
                    state["claim_retrieval_count"] = int(state.get("claim_retrieval_count", 0)) + 1
                    state.setdefault("trajectory_events", []).append(
                        {
                            "step": int(state.get("step_count", 0)),
                            "node": "verify_answer",
                            "action": "claim_specific_retrieval",
                            "claim_id": str(claim.get("claim_id") or ""),
                            "status": "SUCCESS" if added else "NO_NEW_EVIDENCE",
                            "new_evidence_count": added,
                            "tokens": 0,
                        }
                    )
                    rows = _evidence_rows(state)
                    candidate = _verification_claim(claim, state=state, rows=rows, company_aliases=company_aliases)
                    result = _verify_claim_once(
                        state,
                        candidate,
                        rows,
                        calculations=calculations,
                        semantic_scorer=semantic_scorer,
                        llm_judge=llm_judge,
                        semantic_threshold=semantic_threshold,
                        config=config,
                    )
                except Exception as exc:  # retrieval failure is observable, not fatal
                    state.setdefault("trajectory_events", []).append(
                        {
                            "step": int(state.get("step_count", 0)),
                            "node": "verify_answer",
                            "action": "claim_specific_retrieval",
                            "claim_id": str(claim.get("claim_id") or ""),
                            "status": "FAILED",
                            "error_type": type(exc).__name__,
                            "tokens": 0,
                        }
                    )
        status = result["verification"]["status"]
        merged = dict(claim)
        merged["evidence_ids"] = list(result["evidence_ids"])
        merged["verification"] = dict(result["verification"])
        merged["supported"] = status == ENTAILED
        verified_by_id[str(claim.get("claim_id") or "")] = merged
        if status == ENTAILED:
            summary["entailed_count"] += 1
        elif status == "CONTRADICTED":
            summary["contradicted_count"] += 1
        else:
            summary["insufficient_count"] += 1
    verified_claims = [
        verified_by_id[str(claim.get("claim_id") or "")]
        for claim in claims
        if isinstance(claim, Mapping) and str(claim.get("claim_id") or "") in verified_by_id
    ]
    state["claims"] = verified_claims
    state["claim_verifications"] = [dict(c.get("verification") or {}) for c in verified_claims]
    state["claim_verification_summary"] = summary
    return summary


def make_verify_answer(
    config: AgentConfig,
    *,
    runtime: Any = None,
    company_aliases: Mapping[str, list[str]] | None = None,
    semantic_scorer: Any = None,
    llm_judge: Any = None,
):
    def verify_answer(state: dict[str, Any]) -> dict[str, Any]:
        if not begin_node(state, config):
            return state

        draft = state.get("draft_answer")
        if draft is None:
            state["termination_reason"] = "no_draft"
            return state

        support_validation = draft.get("support_validation") or {}
        claim_summary = _verify_claims(
            state,
            runtime=runtime,
            config=config,
            company_aliases=company_aliases,
            semantic_scorer=semantic_scorer,
            llm_judge=llm_judge,
            # An unsupported whole-answer draft already has a bounded
            # rewrite/retrieve remediation path.  Do not spend an additional
            # claim lookup before that path runs; this preserves legacy
            # retrieval counts and its explicit insufficient-evidence gate.
            allow_claim_retrieval=bool(support_validation.get("supported", False)),
        )
        # Compatibility bridge for pre-M1 answerers.  Older deterministic
        # answerers expose a whole-answer `supported=True` verdict, while the
        # extractor may conservatively collapse a multi-sentence answer into a
        # single claim that cannot be mapped to one evidence row.  Preserve that
        # verdict only when no claim is contradicted and every claim is merely
        # INSUFFICIENT; explicit contradiction must always block finalization.
        state["legacy_support_fallback"] = bool(
            not bool(getattr(config, "strict_claim_verification", True))
            and bool(support_validation.get("supported", False))
            and claim_summary["claim_count"] > 0
            and claim_summary["contradicted_count"] == 0
            and claim_summary["insufficient_count"] == claim_summary["claim_count"]
        )
        if claim_summary["contradicted_count"] > 0:
            support_validation = {
                **support_validation,
                "supported": False,
                "claim_contradicted": True,
            }
            state["missing_information"] = state.get("missing_information") or "claim_contradicted"
        elif claim_summary["insufficient_count"] > 0:
            support_validation = {**support_validation, "claim_partial_support": True}
        draft["support_validation"] = support_validation
        state["support_validation"] = support_validation
        supported = bool(support_validation.get("supported", False))
        abstained = bool(draft.get("abstained", False))
        if abstained:
            # Retry policy (gate remediation): abstaining despite having seen
            # evidence is worth one rewrite + re-retrieval attempt; without any
            # evidence the abstention is final (P2 semantics preserved).
            has_evidence = bool(state.get("evidence_pool"))
            if (
                has_evidence
                and int(state.get("rewrite_count", 0)) < config.max_query_rewrites
                and int(state.get("retrieval_count", 0)) < config.max_retrieval_rounds
                and int(state.get("llm_call_count", 0)) < config.max_llm_calls
            ):
                state["missing_information"] = state.get("missing_information") or "abstained_despite_evidence"
                state["unsupported_retry"] = True
                return state  # rewrite_query
            state["unsupported_retry"] = False
            return state  # finalize
        low_confidence = str(draft.get("confidence_label", "high") or "high").strip().lower() == "low"
        if supported and not low_confidence:
            state["unsupported_retry"] = False
            return state  # finalize

        # Retry policy (gate remediation): an unsupported OR low-confidence draft
        # first tries to rewrite + re-retrieve fresher evidence; only when the
        # retrieval budget is exhausted does it fall back to regenerating on the
        # same evidence.
        if (
            int(state.get("rewrite_count", 0)) < config.max_query_rewrites
            and int(state.get("retrieval_count", 0)) < config.max_retrieval_rounds
            and int(state.get("llm_call_count", 0)) < config.max_llm_calls
        ):
            state["missing_information"] = state.get("missing_information") or (
                "unsupported_answer" if not supported else "low_confidence_answer"
            )
            state["unsupported_retry"] = True
            return state  # rewrite_query
        state["unsupported_retry"] = False
        if (
            int(state.get("generation_count", 0)) < config.max_generation_attempts
            and int(state.get("llm_call_count", 0)) < config.max_llm_calls
        ):
            state["missing_information"] = state.get("missing_information") or (
                "unsupported_answer" if not supported else "low_confidence_answer"
            )
            return state  # back to synthesize
        state["termination_reason"] = "max_generation_attempts"
        return state

    return verify_answer
