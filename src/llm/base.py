"""Provider protocol for the remote-only LLM layer."""

from __future__ import annotations

from typing import Any, Protocol

from src.llm.types import LLMResponse


class LLMProvider(Protocol):
    """Minimal remote-LLM contract (checklist v3.0 §P1.5)."""

    provider_name: str

    def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        response_format: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LLMResponse: ...
