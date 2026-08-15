"""Unified LLM configuration (checklist v3.0 §P1.5).

Environment resolution precedence (highest wins):

1. explicit keyword arguments
2. unified ``LLM_*`` variables
3. legacy ``DEEPSEEK_*`` variables (compat fallback)
4. built-in defaults

Unified variables: ``LLM_PROVIDER``, ``LLM_BASE_URL``, ``LLM_API_KEY``,
``LLM_MODEL``, ``LLM_TIMEOUT_SECONDS``, ``LLM_MAX_RETRIES``, ``LLM_MAX_TOKENS``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# unified var -> (legacy fallback var, default)
_VAR_TABLE: dict[str, tuple[str, str]] = {
    "LLM_PROVIDER": ("DEEPSEEK_PROVIDER", "deepseek"),
    "LLM_BASE_URL": ("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    "LLM_API_KEY": ("DEEPSEEK_API_KEY", ""),
    "LLM_MODEL": ("DEEPSEEK_MODEL", "deepseek-chat"),
    "LLM_TIMEOUT_SECONDS": ("DEEPSEEK_TIMEOUT_SECONDS", "60"),
    "LLM_MAX_RETRIES": ("DEEPSEEK_MAX_RETRIES", "3"),
    "LLM_MAX_TOKENS": ("DEEPSEEK_MAX_TOKENS", "512"),
}


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    base_url: str
    api_key: str
    model: str
    timeout_seconds: float
    max_retries: int
    max_tokens: int


def _env_first(unified: str, legacy: str, default: str) -> str:
    value = os.environ.get(unified, "")
    if value:
        return value
    value = os.environ.get(legacy, "")
    if value:
        return value
    return default


def _parse_float(raw: str, var: str, default: float) -> float:
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid {var} value {raw!r}: expected a number.") from exc


def _parse_int(raw: str, var: str, default: int) -> int:
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid {var} value {raw!r}: expected an integer.") from exc


def resolve_llm_config(
    *,
    provider: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    timeout_seconds: float | None = None,
    max_retries: int | None = None,
    max_tokens: int | None = None,
) -> LLMConfig:
    """Resolve the LLM configuration from explicit kwargs, ``LLM_*`` and ``DEEPSEEK_*`` env."""

    def pick(explicit, unified, legacy, default):
        if explicit is not None and str(explicit).strip():
            return str(explicit).strip()
        return _env_first(unified, legacy, default)

    provider_value = pick(provider, "LLM_PROVIDER", "DEEPSEEK_PROVIDER", "deepseek")
    base_url_value = pick(base_url, "LLM_BASE_URL", "DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    api_key_value = pick(api_key, "LLM_API_KEY", "DEEPSEEK_API_KEY", "")
    model_value = pick(model, "LLM_MODEL", "DEEPSEEK_MODEL", "deepseek-chat")
    if not model_value:
        model_value = "deepseek-chat"
    if not api_key_value:
        raise RuntimeError("Missing required configuration: LLM_API_KEY (or DEEPSEEK_API_KEY)")

    timeout_raw = (
        str(timeout_seconds)
        if timeout_seconds is not None and str(timeout_seconds).strip()
        else _env_first("LLM_TIMEOUT_SECONDS", "DEEPSEEK_TIMEOUT_SECONDS", "60")
    )
    max_retries_raw = (
        str(max_retries)
        if max_retries is not None and str(max_retries).strip()
        else _env_first("LLM_MAX_RETRIES", "DEEPSEEK_MAX_RETRIES", "3")
    )
    max_tokens_raw = (
        str(max_tokens)
        if max_tokens is not None and str(max_tokens).strip()
        else _env_first("LLM_MAX_TOKENS", "DEEPSEEK_MAX_TOKENS", "512")
    )
    return LLMConfig(
        provider=provider_value.strip().lower(),
        base_url=base_url_value,
        api_key=api_key_value,
        model=model_value,
        timeout_seconds=_parse_float(timeout_raw, "LLM_TIMEOUT_SECONDS", 60.0),
        max_retries=_parse_int(max_retries_raw, "LLM_MAX_RETRIES", 3),
        max_tokens=_parse_int(max_tokens_raw, "LLM_MAX_TOKENS", 512),
    )
