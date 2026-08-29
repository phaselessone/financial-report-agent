"""Concrete baseline and unified-agent executors for profile ablations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any

from src.agent.config import AgentConfig, ExecutionCapabilities
from src.agent.graph import run_agentic_rag


@dataclass(frozen=True)
class ProfileSpec:
    """One executable ablation profile, including graph-entry policy."""

    profile_id: str
    capabilities: ExecutionCapabilities
    enters_graph: bool

    @property
    def effective_treatments(self) -> dict[str, Any]:
        return {
            "profile": self.profile_id,
            "pipeline": "agentic" if self.enters_graph else "baseline",
            "strict_claim_provenance": self.enters_graph,
            **self.capabilities.to_dict(),
        }


# Sole execution registry for ablations.  The legacy evaluator label table is
# reporting compatibility only and must never select an execution path.
PROFILE_SPECS: dict[str, ProfileSpec] = {
    "baseline-rag": ProfileSpec(
        "baseline-rag",
        ExecutionCapabilities(
            retrieval=True,
            structured_facts=False,
            controlled_tools=False,
            deterministic_calculation=False,
            multi_hop_reasoning=False,
            claim_verification=False,
        ),
        enters_graph=False,
    ),
    "agentic-rag": ProfileSpec(
        "agentic-rag",
        ExecutionCapabilities(
            retrieval=True,
            structured_facts=False,
            controlled_tools=True,
            deterministic_calculation=True,
            multi_hop_reasoning=True,
            claim_verification=True,
        ),
        enters_graph=True,
    ),
    "structured-agent": ProfileSpec(
        "structured-agent",
        ExecutionCapabilities(
            retrieval=True,
            structured_facts=True,
            controlled_tools=False,
            deterministic_calculation=True,
            multi_hop_reasoning=True,
            claim_verification=False,
        ),
        enters_graph=True,
    ),
    "full-agent": ProfileSpec(
        "full-agent",
        ExecutionCapabilities(),
        enters_graph=True,
    ),
}
PROFILE_IDS = tuple(PROFILE_SPECS)
_PROFILE_ALIASES = {
    "Baseline RAG": "baseline-rag",
    "Agentic RAG": "agentic-rag",
    "Structured Agent": "structured-agent",
    "Structured+Tool+Claim Verification": "full-agent",
}


def get_profile_spec(profile: str | ProfileSpec) -> ProfileSpec:
    if isinstance(profile, ProfileSpec):
        canonical = PROFILE_SPECS.get(profile.profile_id)
        if canonical != profile:
            raise ValueError(f"profile spec is not canonical: {profile.profile_id}")
        return profile
    text = str(profile or "").strip()
    profile_id = _PROFILE_ALIASES.get(text, text.lower().replace("_", "-"))
    try:
        return PROFILE_SPECS[profile_id]
    except KeyError as exc:
        raise ValueError(f"unknown profile: {profile}") from exc


class BenchmarkEvidenceRuntime:
    """Expose one benchmark case's reviewed evidence through the retrieval seam."""

    def __init__(self, evidence: list[Mapping[str, Any]]) -> None:
        self._rows = [self._row(item, index) for index, item in enumerate(evidence)]
        self.calls: list[str] = []

    @staticmethod
    def _row(item: Mapping[str, Any], index: int) -> dict[str, Any]:
        evidence_id = str(item.get("evidence_id") or f"evidence-{index + 1}")
        text = str(item.get("text") or "")
        source = str(item.get("source") or item.get("document_id") or "benchmark")
        page = int(item.get("page") or item.get("page_start") or 1)
        return {
            **dict(item),
            "chunk_id": evidence_id,
            "evidence_id": evidence_id,
            "doc_id": source,
            "file_name": source,
            "page": page,
            "page_start": page,
            "page_end": page,
            "text": text,
            "child_text": text,
            "support_span": text,
            "score": float(item.get("score") or 3.0),
            "rerank_score": float(item.get("rerank_score") or item.get("score") or 3.0),
        }

    def search(self, query: str) -> dict[str, Any]:
        self.calls.append(str(query))
        rows = [dict(row) for row in self._rows]
        return {
            "query_mode": "benchmark_evidence",
            "numeric_query": False,
            "dense_rows": [],
            "bm25_rows": [],
            "hybrid_rows": rows,
            "rerank_rows": rows,
            "timings": {},
        }


def _baseline_prediction(
    *,
    case: Mapping[str, Any],
    runtime: Any,
    answerer: Any,
    effective_treatments: Mapping[str, Any],
) -> dict[str, Any]:
    started = perf_counter()
    retrieval = runtime.search(str(case["question"]))
    calls_before = len(getattr(answerer, "llm_calls", []))
    draft = dict(
        answerer.answer(
            query=str(case["question"]),
            question_type=str(case.get("question_type") or case.get("category") or ""),
            retrieval_result=retrieval,
            query_domain_hint=str(case.get("domain_hint") or ""),
        )
    )
    new_calls = list(getattr(answerer, "llm_calls", [])[calls_before:])
    used_ids = [str(item) for item in draft.get("used_evidence_ids") or []]
    trace_events = [
        {
            "event_type": "node",
            "node": "baseline_rag",
            "action": "retrieve_then_answer",
            "status": "SUCCESS",
            "latency_ms": (perf_counter() - started) * 1000,
            "effective_treatments": dict(effective_treatments),
        }
    ]
    capabilities = {
        key: value
        for key, value in effective_treatments.items()
        if key not in {"profile", "pipeline", "strict_claim_provenance"}
    }
    return {
        **draft,
        "case_id": str(case["case_id"]),
        "answer": str(draft.get("final_answer") or draft.get("answer") or ""),
        "cited_evidence_ids": used_ids,
        "tool_calls": [
            {
                "tool_name": "report_search",
                "status": "SUCCESS",
                "result_count": len(retrieval.get("rerank_rows") or []),
            }
        ],
        "trace_events": trace_events,
        "llm_call_count": len(new_calls),
        "total_tokens": sum(call.usage_total_tokens() for call in new_calls),
        "end_to_end_latency_ms": (perf_counter() - started) * 1000,
        "effective_treatments": dict(effective_treatments),
        "applied_treatments": capabilities,
    }


def _agent_prediction(
    *,
    spec: ProfileSpec,
    case: Mapping[str, Any],
    runtime: Any,
    answerer: Any,
    llm: Any,
    base_config: AgentConfig,
    fact_store: Any,
    company_aliases: Mapping[str, list[str]] | None,
    semantic_scorer: Any,
    llm_judge: Any,
) -> dict[str, Any]:
    capabilities = spec.capabilities
    effective_treatments = spec.effective_treatments
    if capabilities.structured_facts and fact_store is None:
        raise ValueError(f"{spec.profile_id} requires a structured fact store")
    config = replace(
        base_config,
        enable_tool_orchestration=capabilities.controlled_tools,
        # Strict deterministic claim provenance is an invariant of every graph
        # profile.  claim_verification controls enhanced validation/recovery.
        strict_claim_verification=True,
        execution_capabilities=capabilities,
    )
    started = perf_counter()
    state = run_agentic_rag(
        runtime=runtime,
        answerer=answerer,
        llm=llm,
        config=config,
        query=str(case["question"]),
        domain_hint=str(case.get("domain_hint") or ""),
        question_type=str(case.get("question_type") or case.get("category") or ""),
        fact_store=fact_store if capabilities.structured_facts else None,
        company_aliases=company_aliases,
        semantic_scorer=semantic_scorer if capabilities.claim_verification else None,
        llm_judge=llm_judge if capabilities.claim_verification else None,
    )
    final = dict(state.get("final_answer") or {})
    used_ids = [str(item) for item in final.get("used_evidence_ids") or []]
    trace_events = [
        {**dict(event), "effective_treatments": dict(effective_treatments)}
        for event in (state.get("trajectory_events") or [])
        if isinstance(event, Mapping)
    ]
    return {
        **final,
        "case_id": str(case["case_id"]),
        "answer": str(final.get("final_answer") or final.get("answer") or ""),
        "cited_evidence_ids": used_ids,
        "claims": list(state.get("claims") or []),
        "calculations": dict(state.get("calculations") or {}),
        "reasoning_plan": dict(state.get("reasoning_plan") or {}),
        "reasoning_step_results": list(state.get("reasoning_step_results") or []),
        "reasoning_conclusion": dict(state.get("reasoning_conclusion") or {}),
        "tool_calls": list(state.get("tool_calls") or []),
        "trace_events": trace_events,
        "llm_call_count": int(state.get("llm_call_count", 0)),
        "tool_call_count": int(state.get("tool_call_count", 0)),
        "total_tokens": int(state.get("total_tokens", 0)),
        "end_to_end_latency_ms": (perf_counter() - started) * 1000,
        "termination_reason": state.get("termination_reason"),
        "effective_treatments": dict(effective_treatments),
        "applied_treatments": capabilities.to_dict(),
    }


def build_profile_executor(
    *,
    answerer: Any,
    llm: Any,
    base_config: AgentConfig,
    fact_store: Any = None,
    company_aliases: Mapping[str, list[str]] | None = None,
    semantic_scorer: Any = None,
    llm_judge: Any = None,
    retrieval_runtime: Any = None,
    allow_contract_oracle: bool = False,
):
    """Return the single executor used by all four profile treatments.

    Reviewed evaluation must inject one shared corpus-backed retrieval runtime.
    Benchmark gold evidence can only be exposed to retrieval when the caller
    explicitly selects the synthetic contract oracle; it is never a reviewed
    performance path.
    """

    if retrieval_runtime is None and not allow_contract_oracle:
        raise ValueError(
            "profile executor requires a shared retrieval runtime; "
            "synthetic fixtures may explicitly enable the contract oracle"
        )
    if retrieval_runtime is not None and allow_contract_oracle:
        raise ValueError(
            "profile executor cannot combine a shared retrieval runtime with the contract oracle"
        )

    def execute(
        profile: str | ProfileSpec,
        treatments_or_case: Mapping[str, Any],
        legacy_case: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Execute through the ProfileSpec seam.

        ``execute(spec, case)`` is canonical. The three-argument form remains
        as one-release compatibility for callers of the initial evaluator.
        """

        spec = get_profile_spec(profile)
        if legacy_case is None:
            case = treatments_or_case
        else:
            supplied = dict(treatments_or_case)
            expected_legacy = spec.capabilities.to_dict()
            if supplied != expected_legacy:
                raise ValueError(
                    f"treatment mapping does not match canonical profile {spec.profile_id}"
                )
            case = legacy_case
        if allow_contract_oracle:
            if case.get("synthetic") is not True:
                raise ValueError(
                    "contract oracle is restricted to explicitly synthetic contract cases"
                )
            evidence = case.get("evidence")
            if not isinstance(evidence, list):
                raise ValueError("synthetic benchmark case evidence must be a list")
            runtime = BenchmarkEvidenceRuntime(
                [item for item in evidence if isinstance(item, Mapping)]
            )
        else:
            runtime = retrieval_runtime
        if not spec.enters_graph:
            return _baseline_prediction(
                case=case,
                runtime=runtime,
                answerer=answerer,
                effective_treatments=spec.effective_treatments,
            )
        return _agent_prediction(
            spec=spec,
            case=case,
            runtime=runtime,
            answerer=answerer,
            llm=llm,
            base_config=base_config,
            fact_store=fact_store,
            company_aliases=company_aliases,
            semantic_scorer=semantic_scorer,
            llm_judge=llm_judge,
        )

    return execute


__all__ = [
    "BenchmarkEvidenceRuntime",
    "PROFILE_IDS",
    "PROFILE_SPECS",
    "ProfileSpec",
    "build_profile_executor",
    "get_profile_spec",
]
