from __future__ import annotations

import os
from pathlib import Path

from src.generation.answerer import LocalEvidenceAnswerer
from src.llm.base import LLMProvider
from src.llm.config import resolve_llm_config
from src.llm.providers import DeepSeekProvider, OpenAICompatibleProvider
from src.llm.types import LLMErrorKind, LLMProviderError, LLMResponse
from src.llm.usage import summarize_usage

# Backward-compatible answerer built on the generic remote LLM layer.
# All provider-specific HTTP lives in src/llm/providers; this class only
# converts prompts into messages and maps normalized errors back to the
# pre-P1.5 semantics (4xx non-429 raises; exhausted retries yield "").


class DeepSeekEvidenceAnswerer(LocalEvidenceAnswerer):
    def __init__(
        self,
        model_name: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.8,
        llm: LLMProvider | None = None,
    ) -> None:
        if llm is not None:
            config = llm.config
        else:
            config = resolve_llm_config(
                provider="deepseek",
                base_url=base_url,
                api_key=api_key,
                model=model_name or None,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                max_tokens=max_new_tokens,
            )
        self._llm = llm or DeepSeekProvider(config)
        self.llm_provider = self._llm.provider_name
        self.llm_model = config.model
        self.model_name = self.llm_model
        self.model_path = Path(f"{self.llm_provider}://chat-completions")
        self.device = None
        self.max_new_tokens = config.max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.base_url = config.base_url
        self.api_key = config.api_key
        self.timeout_seconds = config.timeout_seconds
        self.max_retries = config.max_retries
        self.llm_calls: list[LLMResponse] = []

    @property
    def llm(self) -> LLMProvider:
        """The underlying generic LLM provider (for agent orchestration)."""
        return self._llm

    def _generate(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are a financial research assistant. Answer with one JSON object only. Do not output markdown, code fences, or any extra text."},
            {"role": "user", "content": prompt},
        ]
        try:
            response = self._llm.generate(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
                top_p=self.top_p,
            )
        except LLMProviderError as exc:
            if exc.kind == LLMErrorKind.HTTP and exc.http_status is not None and 400 <= exc.http_status < 500 and exc.http_status != 429:
                raise RuntimeError(
                    f"{exc.provider or 'LLM'} API rejected the request with status {exc.http_status}: {exc.response_body}"
                ) from exc
            return ""
        self.llm_calls.append(response)
        return response.content

    def usage_summary(self) -> dict[str, float | int]:
        """Aggregate usage/latency/retry stats for the calls made by this answerer."""
        return summarize_usage(self.llm_calls)


def build_generation_answerer(
    *,
    provider: str,
    local_model_name: str,
    remote_model_name: str,
    cache_dir: Path,
    device: str = "cuda",
    max_new_tokens: int | None = None,
    temperature: float = 0.1,
    top_p: float = 0.8,
) -> LocalEvidenceAnswerer:
    normalized_provider = (provider or os.environ.get("LLM_PROVIDER", "deepseek")).strip().lower()
    if normalized_provider == "local":
        answerer = LocalEvidenceAnswerer(
            local_model_name,
            cache_dir=cache_dir,
            device=device,
            max_new_tokens=max_new_tokens if max_new_tokens is not None else 512,
            temperature=temperature,
            top_p=top_p,
        )
        answerer.llm_provider = "local"
        answerer.llm_model = local_model_name
        return answerer
    if normalized_provider in ("deepseek", "openai_compatible"):
        config = resolve_llm_config(
            provider=normalized_provider,
            model=remote_model_name or None,
            max_tokens=max_new_tokens,
        )
        llm: LLMProvider
        if config.provider == "deepseek":
            llm = DeepSeekProvider(config)
        else:
            llm = OpenAICompatibleProvider(config)
        return DeepSeekEvidenceAnswerer(
            config.model,
            max_new_tokens=config.max_tokens,
            temperature=temperature,
            top_p=top_p,
            llm=llm,
        )
    raise ValueError(f"Unsupported llm provider: {provider}")
