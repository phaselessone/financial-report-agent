from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from src.generation.answerer import LocalEvidenceAnswerer

logger = logging.getLogger(__name__)


def _env_or_value(value: str | None, env_name: str, *, required: bool = False) -> str:
    resolved = value or os.environ.get(env_name, "")
    if required and not resolved:
        raise RuntimeError(f"Missing required configuration: {env_name}")
    return resolved


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
    ) -> None:
        self.llm_provider = "deepseek"
        self.llm_model = model_name or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        self.model_name = self.llm_model
        self.model_path = Path("deepseek://chat-completions")
        self.device = None
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.base_url = _env_or_value(base_url, "DEEPSEEK_BASE_URL", required=True).rstrip("/")
        self.api_key = _env_or_value(api_key, "DEEPSEEK_API_KEY", required=True)
        self.timeout_seconds = float(
            _env_or_value(str(timeout_seconds) if timeout_seconds is not None else "", "DEEPSEEK_TIMEOUT_SECONDS") or "60"
        )
        self.max_retries = int(
            _env_or_value(str(max_retries) if max_retries is not None else "", "DEEPSEEK_MAX_RETRIES") or "3"
        )

    def _generate(self, prompt: str) -> str:
        import httpx

        messages = [
            {"role": "system", "content": "You are a financial research assistant. Answer with one JSON object only. Do not output markdown, code fences, or any extra text."},
            {"role": "user", "content": prompt},
        ]
        payload = {
            "model": self.llm_model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_new_tokens,
            "stream": False,
        }
        endpoint = f"{self.base_url}/chat/completions"
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    response = client.post(
                        endpoint,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
                status_code = response.status_code
                if 400 <= status_code < 500 and status_code != 429:
                    logger.warning(
                        "DeepSeek API rejected the request with status %s (attempt %d); not retrying",
                        status_code,
                        attempt + 1,
                    )
                    raise RuntimeError(
                        f"DeepSeek API rejected the request with status {status_code}: {response.text[:200]}"
                    )
                if status_code == 429 or status_code >= 500:
                    if attempt < self.max_retries:
                        logger.warning(
                            "DeepSeek API returned status %s (attempt %d/%d); retrying",
                            status_code,
                            attempt + 1,
                            self.max_retries,
                        )
                        time.sleep(min(2**attempt, 8))
                        continue
                    logger.warning(
                        "DeepSeek API returned status %s after %d retries; giving up",
                        status_code,
                        self.max_retries,
                    )
                    return ""
                response.raise_for_status()
                data = response.json()
                return str(data["choices"][0]["message"]["content"]).strip()
            except RuntimeError:
                raise
            except Exception as exc:
                if attempt < self.max_retries:
                    logger.warning(
                        "DeepSeek API request failed on attempt %d/%d: %s; retrying",
                        attempt + 1,
                        self.max_retries,
                        exc,
                    )
                    time.sleep(min(2**attempt, 8))
                    continue
                logger.warning(
                    "DeepSeek API request failed after %d retries: %s",
                    self.max_retries,
                    exc,
                )
                return ""


def build_generation_answerer(
    *,
    provider: str,
    local_model_name: str,
    remote_model_name: str,
    cache_dir: Path,
    device: str = "cuda",
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    top_p: float = 0.8,
) -> LocalEvidenceAnswerer:
    normalized_provider = (provider or os.environ.get("LLM_PROVIDER", "deepseek")).strip().lower()
    if normalized_provider == "local":
        answerer = LocalEvidenceAnswerer(
            local_model_name,
            cache_dir=cache_dir,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        answerer.llm_provider = "local"
        answerer.llm_model = local_model_name
        return answerer
    if normalized_provider == "deepseek":
        return DeepSeekEvidenceAnswerer(
            remote_model_name or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    raise ValueError(f"Unsupported llm provider: {provider}")
