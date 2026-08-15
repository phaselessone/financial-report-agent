"""P1.5 LLM provider layer tests — mock HTTP only, never real API calls.

Covers: 200 success, 429 retry, 5xx retry, timeout, invalid JSON, empty
response, usage parsing, retry exhaustion, error normalization, config
resolution (LLM_* vs DEEPSEEK_*), API-key hygiene, and the answerer wiring.
"""

from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from src.generation.provider import DeepSeekEvidenceAnswerer, build_generation_answerer
from src.llm.config import LLMConfig, resolve_llm_config
from src.llm.providers import DeepSeekProvider, OpenAICompatibleProvider
from src.llm.types import LLMErrorKind, LLMProviderError

LLM_ENV_KEYS = (
    "LLM_PROVIDER", "LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL",
    "LLM_TIMEOUT_SECONDS", "LLM_MAX_RETRIES", "LLM_MAX_TOKENS",
    "DEEPSEEK_PROVIDER", "DEEPSEEK_BASE_URL", "DEEPSEEK_API_KEY",
    "DEEPSEEK_MODEL", "DEEPSEEK_TIMEOUT_SECONDS", "DEEPSEEK_MAX_RETRIES",
    "DEEPSEEK_MAX_TOKENS",
)


def ok_payload(content: str) -> dict:
    return {
        "id": "req-123",
        "model": "deepseek-chat",
        "choices": [
            {"message": {"content": content}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 22, "total_tokens": 33},
    }


class FakeResponse:
    def __init__(self, *, status_code: int = 200, payload: dict | None = None, text: str = "", headers: dict | None = None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text
        self.headers = headers or {}

    def json(self) -> dict:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


class FakeClient:
    def __init__(self, script):
        self._script = script  # shared queue: retries consume successive items
        self.attempts: list[tuple[str, dict]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def post(self, url, *, headers, json):
        self.attempts.append((url, {"headers": headers, "json": json}))
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeTimeoutException(Exception):
    pass


def make_httpx_module(script) -> types.SimpleNamespace:
    module = types.SimpleNamespace()
    module.clients = []

    def client_factory(**kwargs):
        client = FakeClient(script)
        module.clients.append(client)
        return client

    module.Client = client_factory
    module.TimeoutException = FakeTimeoutException
    return module


def install_httpx(script):
    sys.modules["httpx"] = make_httpx_module(script)
    return sys.modules["httpx"]


def sample_config(**overrides) -> LLMConfig:
    values = dict(
        provider="deepseek",
        base_url="https://api.deepseek.com",
        api_key="sk-secret-123",
        model="deepseek-chat",
        timeout_seconds=5.0,
        max_retries=1,
        max_tokens=512,
    )
    values.update(overrides)
    return LLMConfig(**values)


class CleanLLMEnvTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = {key: os.environ[key] for key in LLM_ENV_KEYS if key in os.environ}
        for key in LLM_ENV_KEYS:
            os.environ.pop(key, None)

    def tearDown(self) -> None:
        for key in LLM_ENV_KEYS:
            os.environ.pop(key, None)
        os.environ.update(self._saved)


class ProviderSuccessTests(CleanLLMEnvTestCase):
    def test_200_success_parses_content_usage_request_id_finish_reason(self) -> None:
        with patch.dict(sys.modules, {"httpx": install_httpx([FakeResponse(payload=ok_payload('{"final_answer":"ok"}'))])}):
            response = DeepSeekProvider(sample_config()).generate(messages=[{"role": "user", "content": "hi"}])
        self.assertEqual(response.content, '{"final_answer":"ok"}')
        self.assertEqual(response.provider, "deepseek")
        self.assertEqual(response.model, "deepseek-chat")
        self.assertEqual((response.prompt_tokens, response.completion_tokens, response.total_tokens), (11, 22, 33))
        self.assertEqual(response.retries, 0)
        self.assertEqual(response.request_id, "req-123")
        self.assertEqual(response.finish_reason, "stop")
        self.assertGreaterEqual(response.latency_ms, 0.0)

    def test_429_retries_then_succeeds_and_counts_retries(self) -> None:
        script = [
            FakeResponse(status_code=429, text="rate limited"),
            FakeResponse(payload=ok_payload("answer")),
        ]
        with patch.dict(sys.modules, {"httpx": install_httpx(script)}), patch("time.sleep", return_value=None):
            response = DeepSeekProvider(sample_config()).generate(messages=[{"role": "user", "content": "hi"}])
        self.assertEqual(response.content, "answer")
        self.assertEqual(response.retries, 1)

    def test_5xx_retries_then_succeeds(self) -> None:
        script = [FakeResponse(status_code=503, text="unavailable"), FakeResponse(payload=ok_payload("answer"))]
        with patch.dict(sys.modules, {"httpx": install_httpx(script)}), patch("time.sleep", return_value=None):
            response = DeepSeekProvider(sample_config()).generate(messages=[{"role": "user", "content": "hi"}])
        self.assertEqual(response.content, "answer")
        self.assertEqual(response.retries, 1)

    def test_timeout_then_succeeds(self) -> None:
        script = [FakeTimeoutException("timeout"), FakeResponse(payload=ok_payload("answer"))]
        with patch.dict(sys.modules, {"httpx": install_httpx(script)}), patch("time.sleep", return_value=None):
            response = DeepSeekProvider(sample_config()).generate(messages=[{"role": "user", "content": "hi"}])
        self.assertEqual(response.content, "answer")
        self.assertEqual(response.retries, 1)

    def test_invalid_json_then_succeeds(self) -> None:
        script = [FakeResponse(payload=ValueError("bad json")), FakeResponse(payload=ok_payload("answer"))]
        with patch.dict(sys.modules, {"httpx": install_httpx(script)}), patch("time.sleep", return_value=None):
            response = DeepSeekProvider(sample_config()).generate(messages=[{"role": "user", "content": "hi"}])
        self.assertEqual(response.content, "answer")
        self.assertEqual(response.retries, 1)

    def test_missing_usage_defaults_to_zero(self) -> None:
        payload = ok_payload("answer")
        payload.pop("usage")
        with patch.dict(sys.modules, {"httpx": install_httpx([FakeResponse(payload=payload)])}):
            response = DeepSeekProvider(sample_config()).generate(messages=[{"role": "user", "content": "hi"}])
        self.assertEqual(response.prompt_tokens, 0)
        self.assertEqual(response.completion_tokens, 0)
        self.assertEqual(response.total_tokens, 0)

    def test_empty_content_returns_empty_string(self) -> None:
        with patch.dict(sys.modules, {"httpx": install_httpx([FakeResponse(payload=ok_payload(""))])}):
            response = DeepSeekProvider(sample_config()).generate(messages=[{"role": "user", "content": "hi"}])
        self.assertEqual(response.content, "")

    def test_response_format_and_top_p_are_forwarded_when_provided(self) -> None:
        client_module = install_httpx([FakeResponse(payload=ok_payload("answer"))])
        with patch.dict(sys.modules, {"httpx": client_module}):
            DeepSeekProvider(sample_config()).generate(
                messages=[{"role": "user", "content": "hi"}],
                response_format={"type": "json_object"},
                top_p=0.8,
            )
        _, call = client_module.clients[0].attempts[0]
        self.assertEqual(call["json"]["top_p"], 0.8)
        self.assertEqual(call["json"]["response_format"], {"type": "json_object"})


class ProviderErrorTests(CleanLLMEnvTestCase):
    def test_4xx_non_429_raises_immediately_without_retry(self) -> None:
        client_module = install_httpx([FakeResponse(status_code=400, text="bad request")])
        with patch.dict(sys.modules, {"httpx": client_module}):
            with self.assertRaises(LLMProviderError) as ctx:
                DeepSeekProvider(sample_config()).generate(messages=[{"role": "user", "content": "hi"}])
        self.assertEqual(ctx.exception.kind, LLMErrorKind.HTTP)
        self.assertEqual(ctx.exception.http_status, 400)
        self.assertEqual(ctx.exception.retries, 0)
        self.assertEqual(len(client_module.clients[0].attempts), 1)

    def test_retry_exhausted_on_5xx_raises_http_error(self) -> None:
        script = [FakeResponse(status_code=500, text="boom"), FakeResponse(status_code=500, text="boom")]
        with patch.dict(sys.modules, {"httpx": install_httpx(script)}), patch("time.sleep", return_value=None):
            with self.assertRaises(LLMProviderError) as ctx:
                DeepSeekProvider(sample_config()).generate(messages=[{"role": "user", "content": "hi"}])
        self.assertEqual(ctx.exception.kind, LLMErrorKind.HTTP)
        self.assertEqual(ctx.exception.http_status, 500)
        self.assertEqual(ctx.exception.retries, 1)

    def test_timeout_exhausted_raises_timeout_error(self) -> None:
        script = [FakeTimeoutException("timeout"), FakeTimeoutException("timeout")]
        with patch.dict(sys.modules, {"httpx": install_httpx(script)}), patch("time.sleep", return_value=None):
            with self.assertRaises(LLMProviderError) as ctx:
                DeepSeekProvider(sample_config()).generate(messages=[{"role": "user", "content": "hi"}])
        self.assertEqual(ctx.exception.kind, LLMErrorKind.TIMEOUT)
        self.assertEqual(ctx.exception.retries, 1)

    def test_empty_choices_raises_response_error(self) -> None:
        payload = {"id": "req-1", "choices": []}
        with patch.dict(sys.modules, {"httpx": install_httpx([FakeResponse(payload=payload)])}):
            with self.assertRaises(LLMProviderError) as ctx:
                DeepSeekProvider(sample_config(max_retries=0)).generate(messages=[{"role": "user", "content": "hi"}])
        self.assertEqual(ctx.exception.kind, LLMErrorKind.RESPONSE)

    def test_api_key_never_appears_in_error_messages(self) -> None:
        script = [FakeResponse(status_code=400, text="bad request"), FakeResponse(status_code=400, text="bad request")]
        with patch.dict(sys.modules, {"httpx": install_httpx(script)}):
            with self.assertRaises(LLMProviderError) as ctx:
                DeepSeekProvider(sample_config(max_retries=1)).generate(messages=[{"role": "user", "content": "hi"}])
            self.assertNotIn("sk-secret-123", str(ctx.exception))
            self.assertNotIn("sk-secret-123", ctx.exception.response_body)


class ProviderSelectionTests(CleanLLMEnvTestCase):
    def test_openai_compatible_uses_custom_base_url(self) -> None:
        client_module = install_httpx([FakeResponse(payload=ok_payload("answer"))])
        config = sample_config(provider="openai_compatible", base_url="https://example.com/v1", api_key="k")
        with patch.dict(sys.modules, {"httpx": client_module}):
            response = OpenAICompatibleProvider(config).generate(messages=[{"role": "user", "content": "hi"}])
        self.assertEqual(response.provider, "openai_compatible")
        url, _ = client_module.clients[0].attempts[0]
        self.assertEqual(url, "https://example.com/v1/chat/completions")


class LLMConfigTests(CleanLLMEnvTestCase):
    def test_unified_env_wins_over_legacy_fallback(self) -> None:
        os.environ.update({
            "LLM_MODEL": "llm-model",
            "DEEPSEEK_MODEL": "legacy-model",
            "LLM_API_KEY": "unified-key",
            "DEEPSEEK_API_KEY": "legacy-key",
            "LLM_MAX_TOKENS": "1024",
            "DEEPSEEK_MAX_TOKENS": "512",
        })
        config = resolve_llm_config()
        self.assertEqual(config.model, "llm-model")
        self.assertEqual(config.api_key, "unified-key")
        self.assertEqual(config.max_tokens, 1024)

    def test_legacy_env_used_when_unified_missing(self) -> None:
        os.environ.update({"DEEPSEEK_MODEL": "legacy-model", "DEEPSEEK_API_KEY": "legacy-key"})
        config = resolve_llm_config()
        self.assertEqual(config.model, "legacy-model")
        self.assertEqual(config.api_key, "legacy-key")
        self.assertEqual(config.max_tokens, 512)

    def test_explicit_kwargs_win_over_env(self) -> None:
        os.environ.update({"LLM_MODEL": "env-model", "LLM_API_KEY": "env-key", "LLM_MAX_TOKENS": "777"})
        config = resolve_llm_config(model="explicit-model", api_key="explicit-key", max_tokens=999)
        self.assertEqual(config.model, "explicit-model")
        self.assertEqual(config.api_key, "explicit-key")
        self.assertEqual(config.max_tokens, 999)

    def test_missing_api_key_raises(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            resolve_llm_config()
        self.assertIn("LLM_API_KEY", str(ctx.exception))

    def test_defaults_match_pre_p15_wire_values(self) -> None:
        os.environ["LLM_API_KEY"] = "k"
        config = resolve_llm_config()
        self.assertEqual(config.provider, "deepseek")
        self.assertEqual(config.base_url, "https://api.deepseek.com")
        self.assertEqual(config.timeout_seconds, 60.0)
        self.assertEqual(config.max_retries, 3)
        self.assertEqual(config.max_tokens, 512)


class AnswererWiringTests(CleanLLMEnvTestCase):
    def _answerer(self, **kwargs) -> DeepSeekEvidenceAnswerer:
        values = dict(
            model_name="deepseek-chat",
            base_url="https://api.deepseek.com",
            api_key="sk-secret-123",
            max_retries=1,
        )
        values.update(kwargs)
        return DeepSeekEvidenceAnswerer(**values)

    def test_generate_records_usage_and_latency(self) -> None:
        with patch.dict(sys.modules, {"httpx": install_httpx([FakeResponse(payload=ok_payload('{"final_answer":"ok"}'))])}):
            answerer = self._answerer()
            content = answerer._generate("prompt")
        self.assertEqual(content, '{"final_answer":"ok"}')
        self.assertEqual(len(answerer.llm_calls), 1)
        summary = answerer.usage_summary()
        self.assertEqual(summary["llm_calls"], 1)
        self.assertEqual(summary["prompt_tokens"], 11)
        self.assertEqual(summary["completion_tokens"], 22)
        self.assertEqual(summary["total_tokens"], 33)
        self.assertGreaterEqual(summary["total_latency_ms"], 0.0)

    def test_timeout_exhausted_returns_empty_string(self) -> None:
        script = [FakeTimeoutException("timeout"), FakeTimeoutException("timeout")]
        with patch.dict(sys.modules, {"httpx": install_httpx(script)}), patch("time.sleep", return_value=None):
            content = self._answerer()._generate("prompt")
        self.assertEqual(content, "")

    def test_4xx_non_429_raises_runtime_error_without_key(self) -> None:
        script = [FakeResponse(status_code=401, text="unauthorized")]
        with patch.dict(sys.modules, {"httpx": install_httpx(script)}):
            with self.assertRaises(RuntimeError) as ctx:
                self._answerer()._generate("prompt")
        self.assertIn("401", str(ctx.exception))
        self.assertNotIn("sk-secret-123", str(ctx.exception))

    def test_build_generation_answerer_selects_openai_compatible(self) -> None:
        os.environ.update({"LLM_API_KEY": "k", "LLM_BASE_URL": "https://example.com/v1"})
        answerer = build_generation_answerer(
            provider="openai_compatible",
            local_model_name="Qwen/Qwen2.5-7B-Instruct",
            remote_model_name="some-model",
            cache_dir=Path("models"),
            device="cpu",
        )
        self.assertEqual(answerer.llm_provider, "openai_compatible")
        self.assertEqual(answerer.llm_model, "some-model")

    def test_build_generation_answerer_honors_llm_max_tokens(self) -> None:
        os.environ.update({"LLM_API_KEY": "k", "LLM_MAX_TOKENS": "256"})
        answerer = build_generation_answerer(
            provider="deepseek",
            local_model_name="Qwen/Qwen2.5-7B-Instruct",
            remote_model_name="deepseek-chat",
            cache_dir=Path("models"),
            device="cpu",
        )
        self.assertEqual(answerer.max_new_tokens, 256)


if __name__ == "__main__":
    unittest.main()
