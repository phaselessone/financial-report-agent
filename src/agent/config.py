"""Agent configuration and hard budgets (checklist v3.0 §P2 Budget)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    max_steps: int = 8
    max_retrieval_rounds: int = 3
    max_query_rewrites: int = 2
    max_generation_attempts: int = 2
    max_llm_calls: int = 6
    max_total_tokens: int = 0  # 0 = token budget disabled (checklist §P3 API cost knob)

    # deterministic-only grading in v1 (checklist: prefer deterministic; API only when undecidable)
    grade_mode: str = "deterministic"
    rewrite_temperature: float = 0.0
    rewrite_max_tokens: int = 256
