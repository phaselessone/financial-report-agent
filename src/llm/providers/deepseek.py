"""DeepSeek chat-completions provider (default remote LLM)."""

from __future__ import annotations

from src.llm.config import LLMConfig
from src.llm.providers.openai_compatible import OpenAICompatibleProvider


class DeepSeekProvider(OpenAICompatibleProvider):
    """DeepSeek API provider built on the generic OpenAI-compatible transport."""

    provider_name = "deepseek"

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
