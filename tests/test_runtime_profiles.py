"""P1 device-decoupling tests: runtime profiles, env precedence, and that
build_retrieval_runtime assigns embedding and reranker devices independently.

All constructor calls are mocked — no model loading, no API, no GPU required.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.retrieval import runtime_profiles
from src.retrieval.runtime import build_retrieval_runtime

RELEVANT_ENV = (
    "RUNTIME_PROFILE",
    "EMBEDDING_DEVICE",
    "RERANKER_DEVICE",
    "EMBEDDING_BATCH_SIZE",
    "RERANKER_BATCH_SIZE",
)


class CleanEnvTestCase(unittest.TestCase):
    def setUp(self) -> None:
        for key in RELEVANT_ENV:
            os.environ.pop(key, None)

    def tearDown(self) -> None:
        for key in RELEVANT_ENV:
            os.environ.pop(key, None)


class RuntimeProfileParsingTests(CleanEnvTestCase):
    def test_all_profiles_parse_and_match_checklist_values(self) -> None:
        expected = {
            "low_vram": {"embedding_device": "cuda", "embedding_batch_size": 4, "reranker_device": "cpu", "reranker_batch_size": 2},
            "cpu": {"embedding_device": "cpu", "embedding_batch_size": 4, "reranker_device": "cpu", "reranker_batch_size": 2},
            "standard_gpu": {"embedding_device": "cuda", "embedding_batch_size": 16, "reranker_device": "cuda", "reranker_batch_size": 8},
        }
        self.assertEqual(runtime_profiles.available_profiles(), ["cpu", "low_vram", "standard_gpu"])
        for name, fields in expected.items():
            toml_file = runtime_profiles.RUNTIME_PROFILES_DIR / f"{name}.toml"
            self.assertTrue(toml_file.is_file(), f"missing profile file {toml_file}")
            values = runtime_profiles.load_runtime_profile(name)
            for key, want in fields.items():
                self.assertEqual(values[key], want, f"{name}.toml {key}")

    def test_default_profile_is_low_vram(self) -> None:
        settings = runtime_profiles.resolve_runtime_settings()
        self.assertEqual(settings.profile, "low_vram")
        self.assertEqual(settings.embedding_device, "cuda")
        self.assertEqual(settings.reranker_device, "cpu")
        self.assertEqual(settings.embedding_batch_size, 4)
        self.assertEqual(settings.reranker_batch_size, 2)

    def test_unknown_profile_raises_with_options(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            runtime_profiles.resolve_runtime_settings(profile="nope")
        self.assertIn("low_vram", str(ctx.exception))
        self.assertIn("standard_gpu", str(ctx.exception))


class RuntimeSettingsPrecedenceTests(CleanEnvTestCase):
    def test_explicit_arguments_win_over_profile_and_env(self) -> None:
        os.environ["EMBEDDING_DEVICE"] = "cuda"
        settings = runtime_profiles.resolve_runtime_settings(
            profile="cpu",
            embedding_device="auto",
            reranker_device="cuda",
            embedding_batch_size=7,
            reranker_batch_size=9,
        )
        self.assertEqual(settings.profile, "cpu")
        self.assertEqual(settings.embedding_device, "auto")
        self.assertEqual(settings.reranker_device, "cuda")
        self.assertEqual(settings.embedding_batch_size, 7)
        self.assertEqual(settings.reranker_batch_size, 9)

    def test_env_vars_win_over_profile(self) -> None:
        os.environ["EMBEDDING_DEVICE"] = "auto"
        os.environ["RERANKER_DEVICE"] = "cuda"
        os.environ["EMBEDDING_BATCH_SIZE"] = "3"
        os.environ["RERANKER_BATCH_SIZE"] = "5"
        settings = runtime_profiles.resolve_runtime_settings(profile="cpu")
        self.assertEqual(settings.embedding_device, "auto")
        self.assertEqual(settings.reranker_device, "cuda")
        self.assertEqual(settings.embedding_batch_size, 3)
        self.assertEqual(settings.reranker_batch_size, 5)

    def test_runtime_profile_env_selects_profile(self) -> None:
        os.environ["RUNTIME_PROFILE"] = "standard_gpu"
        settings = runtime_profiles.resolve_runtime_settings()
        self.assertEqual(settings.profile, "standard_gpu")
        self.assertEqual(settings.reranker_device, "cuda")

    def test_legacy_device_alias_sets_both(self) -> None:
        settings = runtime_profiles.resolve_runtime_settings(device="cpu")
        self.assertEqual(settings.embedding_device, "cpu")
        self.assertEqual(settings.reranker_device, "cpu")

    def test_legacy_device_alias_loses_to_explicit_knob(self) -> None:
        settings = runtime_profiles.resolve_runtime_settings(device="cpu", embedding_device="cuda")
        self.assertEqual(settings.embedding_device, "cuda")
        self.assertEqual(settings.reranker_device, "cpu")

    def test_invalid_device_value_raises(self) -> None:
        with self.assertRaises(ValueError):
            runtime_profiles.resolve_runtime_settings(profile="cpu", embedding_device="tpu")

    def test_invalid_batch_size_raises(self) -> None:
        with self.assertRaises(ValueError):
            runtime_profiles.resolve_runtime_settings(profile="cpu", embedding_batch_size=0)
        with self.assertRaises(ValueError):
            runtime_profiles.resolve_runtime_settings(profile="cpu", reranker_batch_size=-3)


class BuildRuntimeDeviceTests(CleanEnvTestCase):
    def _chunks(self):
        return [
            {"chunk_id": "c1", "doc_id": "d1", "file_name": "a.pdf", "text": "文本", "page_start": 1, "page_end": 1}
        ]

    def test_low_vram_puts_embedding_on_cuda_and_reranker_on_cpu(self) -> None:
        with patch("src.retrieval.runtime.resolve_device") as resolve_device_mock, patch(
            "src.retrieval.runtime.DenseEmbedder"
        ) as embedder_mock, patch("src.retrieval.runtime.build_or_load_embeddings", return_value=(MagicMock(), ["c1"])) as embeddings_mock, patch(
            "src.retrieval.runtime.build_or_load_faiss_index", return_value=MagicMock()
        ), patch("src.retrieval.runtime.DenseRetriever"), patch(
            "src.retrieval.runtime.build_or_load_bm25_index", return_value=MagicMock()
        ), patch("src.retrieval.runtime.BM25Retriever"), patch("src.retrieval.runtime.HybridRetriever"), patch(
            "src.retrieval.runtime.CrossEncoderReranker"
        ) as reranker_mock:
            resolve_device_mock.side_effect = lambda value: value
            runtime = build_retrieval_runtime(
                chunks=self._chunks(),
                output_dir=Path("outputs"),
                model_cache_dir=Path("models"),
                embedding_model="BAAI/bge-m3",
                reranker_model="BAAI/bge-reranker-v2-m3",
                runtime_profile="low_vram",
            )
            self.assertEqual(embedder_mock.call_args.kwargs["device"], "cuda")
            self.assertEqual(reranker_mock.call_args.kwargs["device"], "cpu")
            self.assertEqual(embeddings_mock.call_args.kwargs["batch_size"], 4)
            self.assertEqual(runtime.rerank_batch_size, 2)

    def test_cpu_profile_initializes_both_on_cpu(self) -> None:
        with patch("src.retrieval.runtime.resolve_device") as resolve_device_mock, patch(
            "src.retrieval.runtime.DenseEmbedder"
        ) as embedder_mock, patch("src.retrieval.runtime.build_or_load_embeddings", return_value=(MagicMock(), ["c1"])), patch(
            "src.retrieval.runtime.build_or_load_faiss_index", return_value=MagicMock()
        ), patch("src.retrieval.runtime.DenseRetriever"), patch(
            "src.retrieval.runtime.build_or_load_bm25_index", return_value=MagicMock()
        ), patch("src.retrieval.runtime.BM25Retriever"), patch("src.retrieval.runtime.HybridRetriever"), patch(
            "src.retrieval.runtime.CrossEncoderReranker"
        ) as reranker_mock:
            resolve_device_mock.side_effect = lambda value: value
            build_retrieval_runtime(
                chunks=self._chunks(),
                output_dir=Path("outputs"),
                model_cache_dir=Path("models"),
                embedding_model="BAAI/bge-m3",
                reranker_model="BAAI/bge-reranker-v2-m3",
                runtime_profile="cpu",
            )
            self.assertEqual(embedder_mock.call_args.kwargs["device"], "cpu")
            self.assertEqual(reranker_mock.call_args.kwargs["device"], "cpu")

    def test_legacy_device_argument_sets_both_devices(self) -> None:
        with patch("src.retrieval.runtime.resolve_device") as resolve_device_mock, patch(
            "src.retrieval.runtime.DenseEmbedder"
        ) as embedder_mock, patch("src.retrieval.runtime.build_or_load_embeddings", return_value=(MagicMock(), ["c1"])), patch(
            "src.retrieval.runtime.build_or_load_faiss_index", return_value=MagicMock()
        ), patch("src.retrieval.runtime.DenseRetriever"), patch(
            "src.retrieval.runtime.build_or_load_bm25_index", return_value=MagicMock()
        ), patch("src.retrieval.runtime.BM25Retriever"), patch("src.retrieval.runtime.HybridRetriever"), patch(
            "src.retrieval.runtime.CrossEncoderReranker"
        ) as reranker_mock:
            resolve_device_mock.side_effect = lambda value: value
            build_retrieval_runtime(
                chunks=self._chunks(),
                output_dir=Path("outputs"),
                model_cache_dir=Path("models"),
                embedding_model="BAAI/bge-m3",
                reranker_model="BAAI/bge-reranker-v2-m3",
                device="auto",
            )
            self.assertEqual(embedder_mock.call_args.kwargs["device"], "auto")
            self.assertEqual(reranker_mock.call_args.kwargs["device"], "auto")


if __name__ == "__main__":
    unittest.main()
