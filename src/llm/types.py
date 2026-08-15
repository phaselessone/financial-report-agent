"""Shared LLM layer types: normalized responses and errors."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass(frozen=True)
class LLMResponse:
    """Unified response of one remote LLM call (checklist v3.0 §P1.5)."""

    content: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    retries: int = 0
    request_id: str = ""
    finish_reason: str = ""
    extra: dict[str, object] = field(default_factory=dict)

    def usage_total_tokens(self) -> int:
        return self.total_tokens if self.total_tokens else self.prompt_tokens + self.completion_tokens


class LLMErrorKind(str, Enum):
    """Normalized failure categories across providers."""

    TIMEOUT = "timeout"
    NETWORK = "network"
    HTTP = "http"
    JSON = "json"
    RESPONSE = "response"


class LLMProviderError(Exception):
    """Normalized provider error.

    Never carries API keys or other credentials; ``response_body`` is truncated
    server text used for diagnostics only.
    """

    def __init__(
        self,
        message: str,
        *,
        kind: LLMErrorKind,
        provider: str = "",
        model: str = "",
        http_status: int | None = None,
        retries: int = 0,
        response_body: str = "",
    ) -> None:
        super().__init__(message)
        self.message = message
        self.kind = kind
        self.provider = provider
        self.model = model
        self.http_status = http_status
        self.retries = retries
        self.response_body = response_body
