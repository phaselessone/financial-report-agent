"""Remote LLM layer (checklist v3.0 §P1.5): LLM = Remote API only.

Public surface: ``LLMProvider`` protocol, ``LLMResponse``, ``LLMConfig`` /
``resolve_llm_config``, ``RetryPolicy``, usage helpers and the bundled
providers.
"""

from src.llm.base import LLMProvider
from src.llm.config import LLMConfig, resolve_llm_config
from src.llm.providers import DeepSeekProvider, OpenAICompatibleProvider
from src.llm.retry import RetryPolicy
from src.llm.types import LLMErrorKind, LLMProviderError, LLMResponse
from src.llm.usage import parse_usage, summarize_usage

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "LLMErrorKind",
    "LLMProviderError",
    "LLMConfig",
    "resolve_llm_config",
    "RetryPolicy",
    "parse_usage",
    "summarize_usage",
    "DeepSeekProvider",
    "OpenAICompatibleProvider",
]
