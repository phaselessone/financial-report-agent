"""Agent configuration and hard budgets (checklist v3.0 §P2 Budget)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    # max_steps was 8 in P2; raised to 12 (gate remediation: the verification
    # retry policy needs room for one rewrite + re-retrieval cycle).
    max_steps: int = 12
    max_retrieval_rounds: int = 3
    max_query_rewrites: int = 2
    max_generation_attempts: int = 2
    max_llm_calls: int = 6
    max_total_tokens: int = 0  # 0 = token budget disabled (checklist §P3 API cost knob)
    # abstain when a post-rewrite grading round is still insufficient this many
    # times in a row (checklist §P3 gate remediation R1: termination policy)
    max_failed_rewrite_rounds: int = 0
    # P6: hard cap on sub-questions a multi-hop query may decompose into.
    max_sub_questions: int = 6

    # deterministic-only grading in v1 (checklist: prefer deterministic; API only when undecidable)
    grade_mode: str = "deterministic"
    rewrite_temperature: float = 0.0
    rewrite_max_tokens: int = 256
