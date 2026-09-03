"""Agent configuration and hard budgets (checklist v3.0 §P2 Budget)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionCapabilities:
    """Features that may materially change one agent execution path.

    The default is the current full graph so existing callers retain their
    behaviour.  Evaluation profiles pass an explicit instance and persist the
    exact mapping in their RunIdentity treatment namespace.
    """

    retrieval: bool = True
    structured_facts: bool = True
    controlled_tools: bool = True
    deterministic_calculation: bool = True
    multi_hop_reasoning: bool = True
    claim_verification: bool = True

    def __post_init__(self) -> None:
        for name, value in self.to_dict().items():
            if not isinstance(value, bool):
                raise TypeError(f"execution capability {name} must be boolean")

    def to_dict(self) -> dict[str, bool]:
        return {
            "retrieval": self.retrieval,
            "structured_facts": self.structured_facts,
            "controlled_tools": self.controlled_tools,
            "deterministic_calculation": self.deterministic_calculation,
            "multi_hop_reasoning": self.multi_hop_reasoning,
            "claim_verification": self.claim_verification,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ExecutionCapabilities":
        expected = set(cls().to_dict())
        actual = set(value)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                f"invalid execution capabilities; missing={missing}, extra={extra}"
            )
        if any(not isinstance(item, bool) for item in value.values()):
            raise TypeError("execution capabilities must contain only booleans")
        return cls(**{name: bool(value[name]) for name in expected})


@dataclass(frozen=True)
class AgentConfig:
    # M3 adds plan/execute/observe nodes.  Twenty steps leave room for one
    # bounded rewrite/retrieval recovery plus claim verification without
    # weakening the explicit hard stop.
    max_steps: int = 20
    # A/B YoY comparison has four independent report-backed operands when the
    # structured store misses.  Four is therefore the smallest default that
    # can execute the required reasoning plan without silently truncating it.
    max_retrieval_rounds: int = 4
    max_query_rewrites: int = 2
    max_generation_attempts: int = 2
    max_llm_calls: int = 6
    max_total_tokens: int = 0  # 0 = token budget disabled (checklist §P3 API cost knob)
    # abstain when a post-rewrite grading round is still insufficient this many
    # times in a row (checklist §P3 gate remediation R1: termination policy)
    max_failed_rewrite_rounds: int = 0
    # P6: hard cap on sub-questions a multi-hop query may decompose into.
    max_sub_questions: int = 6
    # P7: max tokens for the claim-extraction LLM call (one per non-abstained draft).
    claim_max_tokens: int = 512
    # M3: compatibility switch.  The CLI/evaluation entrypoints enable the
    # controlled tool graph; direct legacy callers retain the old graph unless
    # they opt in explicitly.
    enable_tool_orchestration: bool = False
    # Four report-backed comparison operands require 4 structured misses,
    # 4 bounded report searches, 2 calculations, and 1 comparison.  Twelve
    # keeps every real call auditable while leaving one-call recovery headroom.
    max_tool_calls: int = 12
    max_empty_tool_results: int = 2
    # M1 bounded claim-specific recovery and M2 trace retention.
    max_claim_retrievals: int = 1
    claim_semantic_threshold: float = 0.85
    claim_llm_budget: int = 0
    # Strict profiles fail closed on malformed claim dependencies and never
    # reuse a whole-answer support verdict as a substitute for claim evidence.
    strict_claim_verification: bool = True
    # None preserves the pre-ablation full graph. Evaluation profiles set an
    # explicit capability object so a label cannot silently leave behaviour
    # unchanged.
    execution_capabilities: ExecutionCapabilities | None = None

    # deterministic-only grading in v1 (checklist: prefer deterministic; API only when undecidable)
    grade_mode: str = "deterministic"
    rewrite_temperature: float = 0.0
    rewrite_max_tokens: int = 256

    def resolved_execution_capabilities(self) -> ExecutionCapabilities:
        return self.execution_capabilities or ExecutionCapabilities()


__all__ = ["AgentConfig", "ExecutionCapabilities"]
