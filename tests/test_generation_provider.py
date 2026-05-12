import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.generation.provider import DeepSeekEvidenceAnswerer, build_generation_answerer


class FakeResponse:
    def __init__(self, *, status_code: int = 200, payload: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


class FakeClient:
    def __init__(self, responses_or_exc):
        self._responses_or_exc = responses_or_exc

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def post(self, *_args, **_kwargs):
        item = self._responses_or_exc.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeTimeoutException(Exception):
    pass


def make_httpx_module(responses_or_exc) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        Client=lambda **_kwargs: FakeClient(responses_or_exc),
        TimeoutException=FakeTimeoutException,
    )


class DeepSeekProviderTests(unittest.TestCase):
    def test_deepseek_generate_returns_content(self) -> None:
        responses = [
            FakeResponse(
                payload={
                    "choices": [
                        {
                            "message": {
                                "content": '{"final_answer":"ok","evidence_summary":"","uncertainty_note":"","used_evidence_ids":["E1"]}'
                            }
                        }
                    ]
                }
            )
        ]
        with patch.dict(sys.modules, {"httpx": make_httpx_module(responses)}):
            answerer = DeepSeekEvidenceAnswerer(
                "deepseek-chat",
                base_url="https://api.deepseek.com",
                api_key="test-key",
                max_retries=0,
            )
            content = answerer._generate("prompt")
        self.assertIn('"final_answer":"ok"', content)

    def test_deepseek_generate_retries_on_429_then_succeeds(self) -> None:
        responses = [
            FakeResponse(status_code=429, text="rate limited"),
            FakeResponse(
                payload={
                    "choices": [
                        {
                            "message": {
                                "content": '{"final_answer":"ok","evidence_summary":"","uncertainty_note":"","used_evidence_ids":["E1"]}'
                            }
                        }
                    ]
                }
            ),
        ]
        with patch.dict(sys.modules, {"httpx": make_httpx_module(responses)}), patch("time.sleep", return_value=None):
            answerer = DeepSeekEvidenceAnswerer(
                "deepseek-chat",
                base_url="https://api.deepseek.com",
                api_key="test-key",
                max_retries=1,
            )
            content = answerer._generate("prompt")
        self.assertIn('"final_answer":"ok"', content)

    def test_deepseek_generate_returns_empty_after_timeout(self) -> None:
        responses = [FakeTimeoutException("timeout"), FakeTimeoutException("timeout")]
        with patch.dict(sys.modules, {"httpx": make_httpx_module(responses)}), patch("time.sleep", return_value=None):
            answerer = DeepSeekEvidenceAnswerer(
                "deepseek-chat",
                base_url="https://api.deepseek.com",
                api_key="test-key",
                max_retries=1,
            )
            content = answerer._generate("prompt")
        self.assertEqual(content, "")

    def test_build_generation_answerer_selects_deepseek(self) -> None:
        with patch.dict(
            os.environ,
            {"DEEPSEEK_API_KEY": "test-key", "DEEPSEEK_BASE_URL": "https://api.deepseek.com"},
            clear=False,
        ):
            answerer = build_generation_answerer(
                provider="deepseek",
                local_model_name="Qwen/Qwen2.5-7B-Instruct",
                remote_model_name="deepseek-chat",
                cache_dir=Path("models"),
                device="cpu",
            )
        self.assertEqual(answerer.llm_provider, "deepseek")
        self.assertEqual(answerer.llm_model, "deepseek-chat")

    def test_build_generation_answerer_selects_local(self) -> None:
        stub_instance = MagicMock()
        stub_instance.llm_provider = "local"
        stub_instance.llm_model = "Qwen/Qwen2.5-7B-Instruct"
        with patch("src.generation.provider.LocalEvidenceAnswerer", return_value=stub_instance) as local_cls:
            answerer = build_generation_answerer(
                provider="local",
                local_model_name="Qwen/Qwen2.5-7B-Instruct",
                remote_model_name="deepseek-chat",
                cache_dir=Path("models"),
                device="cpu",
            )
        self.assertIs(answerer, stub_instance)
        local_cls.assert_called_once()


if __name__ == "__main__":
    unittest.main()
