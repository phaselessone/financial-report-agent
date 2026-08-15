"""Generic OpenAI-compatible chat-completions provider.

The shared HTTP/retry/usage/normalization implementation lives here; provider
specifics (defaults, naming) are expressed by subclasses such as
:class:`src.llm.providers.deepseek.DeepSeekProvider`.
"""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

from src.llm.base import LLMProvider
from src.llm.config import LLMConfig
from src.llm.retry import RetryPolicy
from src.llm.types import LLMErrorKind, LLMProviderError, LLMResponse
from src.llm.usage import parse_usage

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(LLMProvider):
    """Chat-completions client against any OpenAI-compatible endpoint."""

    provider_name = "openai_compatible"

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.retry_policy = RetryPolicy(max_retries=config.max_retries)

    def _error(
        self,
        *,
        kind: LLMErrorKind,
        message: str,
        http_status: int | None = None,
        retries: int = 0,
        response_body: str = "",
    ) -> LLMProviderError:
        return LLMProviderError(
            message,
            kind=kind,
            provider=self.provider_name,
            model=self.config.model,
            http_status=http_status,
            retries=retries,
            response_body=response_body[:200],
        )

    def _post(self, payload: dict[str, Any]) -> Any:
        import httpx

        endpoint = f"{self.config.base_url}/chat/completions"
        with httpx.Client(timeout=self.config.timeout_seconds) as client:
            return client.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )

    def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        response_format: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        top_p: float | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if top_p is not None:
            payload["top_p"] = top_p
        if response_format is not None:
            payload["response_format"] = response_format

        start = perf_counter()
        last_error: LLMProviderError | None = None
        for attempt in range(self.config.max_retries + 1):
            if attempt > 0:
                logger.warning(
                    "%s attempt %d/%d after %s; retrying",
                    self.provider_name,
                    attempt,
                    self.config.max_retries,
                    last_error.kind.value if last_error else "unknown",
                )
                self.retry_policy.sleep(attempt - 1)
            try:
                response = self._post(payload)
                last_error = None
            except Exception as exc:
                kind = LLMErrorKind.TIMEOUT if "timeout" in type(exc).__name__.lower() else LLMErrorKind.NETWORK
                last_error = self._error(kind=kind, message=f"{self.provider_name} request failed: {exc}", retries=attempt)
                continue

            status_code = int(getattr(response, "status_code", 0) or 0)
            if self.retry_policy.is_retryable_status(status_code):
                body = str(getattr(response, "text", "") or "")
                last_error = self._error(
                    kind=LLMErrorKind.HTTP,
                    message=f"{self.provider_name} API returned status {status_code}",
                    http_status=status_code,
                    retries=attempt,
                    response_body=body,
                )
                if attempt < self.config.max_retries:
                    logger.warning(
                        "%s API returned status %s (attempt %d/%d); retrying",
                        self.provider_name,
                        status_code,
                        attempt + 1,
                        self.config.max_retries,
                    )
                    continue
                raise last_error
            if 400 <= status_code < 500:
                body = str(getattr(response, "text", "") or "")
                raise self._error(
                    kind=LLMErrorKind.HTTP,
                    message=f"{self.provider_name} API rejected the request with status {status_code}",
                    http_status=status_code,
                    retries=attempt,
                    response_body=body,
                )
            if status_code != 200:
                last_error = self._error(
                    kind=LLMErrorKind.HTTP,
                    message=f"{self.provider_name} API returned unexpected status {status_code}",
                    http_status=status_code,
                    retries=attempt,
                )
                continue

            try:
                data = response.json()
            except Exception as exc:
                last_error = self._error(
                    kind=LLMErrorKind.JSON,
                    message=f"{self.provider_name} returned invalid JSON: {exc}",
                    retries=attempt,
                )
                continue
            if not isinstance(data, dict) or not isinstance(data.get("choices"), list) or not data["choices"]:
                last_error = self._error(
                    kind=LLMErrorKind.RESPONSE,
                    message=f"{self.provider_name} returned an empty or malformed response body",
                    retries=attempt,
                )
                continue

            latency_ms = round((perf_counter() - start) * 1000, 2)
            first_choice = data["choices"][0]
            message = first_choice.get("message") if isinstance(first_choice, dict) else None
            content = message.get("content") if isinstance(message, dict) else None
            prompt_tokens, completion_tokens, total_tokens = parse_usage(data)
            request_id = str(data.get("id") or "")
            if not request_id:
                headers = getattr(response, "headers", None) or {}
                request_id = str(headers.get("x-request-id") or headers.get("X-Request-Id") or "")
            return LLMResponse(
                content=str(content).strip() if content is not None else "",
                provider=self.provider_name,
                model=str(data.get("model") or self.config.model),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                retries=attempt,
                request_id=request_id,
                finish_reason=str(first_choice.get("finish_reason") or ""),
                extra={"metadata": metadata} if metadata else {},
            )

        raise self._error(
            kind=last_error.kind if last_error else LLMErrorKind.NETWORK,
            message=last_error.message if last_error else f"{self.provider_name} request failed after retries",
            http_status=last_error.http_status if last_error else None,
            retries=self.config.max_retries,
            response_body=last_error.response_body if last_error else "",
        )
