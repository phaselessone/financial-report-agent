import argparse
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    import pymupdf  # noqa: F401
except ModuleNotFoundError:
    HAS_FITZ = False
    resolve_answer_seed_paths = None
    resolve_eval_limit = None
else:
    HAS_FITZ = True
    from run_answer_eval import (
        apply_retrieval_domain_priority,
        resolve_answer_seed_paths,
        resolve_benchmark_profile,
        resolve_eval_limit,
        resolve_requested_materialized_splits,
        validate_benchmark_profile,
    )


@unittest.skipUnless(HAS_FITZ, "fitz is required for run_answer_eval imports")
class RunAnswerEvalTests(unittest.TestCase):
    def test_resolve_eval_limit_prefers_new_flag(self) -> None:
        args = argparse.Namespace(eval_limit=7, limit=3)
        self.assertEqual(resolve_eval_limit(args), 7)

    def test_resolve_eval_limit_uses_deprecated_flag_as_fallback(self) -> None:
        args = argparse.Namespace(eval_limit=0, limit=5)
        self.assertEqual(resolve_eval_limit(args), 5)

    def test_full_split_requires_full_seed(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dev_seed = tmp_path / "answer_eval_seed_dev.jsonl"
            full_seed = tmp_path / "answer_eval_seed_full.jsonl"
            dev_seed.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                resolve_answer_seed_paths(split="full", dev_seed_path=dev_seed, full_seed_path=full_seed)

    def test_dev_split_allows_missing_full_seed(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dev_seed = tmp_path / "answer_eval_seed_dev.jsonl"
            full_seed = tmp_path / "answer_eval_seed_full.jsonl"
            dev_seed.write_text("{}\n", encoding="utf-8")
            seed_paths = resolve_answer_seed_paths(split="dev", dev_seed_path=dev_seed, full_seed_path=full_seed)
            self.assertEqual(seed_paths, {"dev": dev_seed})

    def test_default_profile_follows_split(self) -> None:
        dev_args = argparse.Namespace(split="dev", benchmark_profile="")
        full_args = argparse.Namespace(split="full", benchmark_profile="")
        self.assertEqual(resolve_benchmark_profile(dev_args), "current-dev")
        self.assertEqual(resolve_benchmark_profile(full_args), "historical-full-core")

    def test_historical_full_alias_normalizes_to_raw(self) -> None:
        alias_args = argparse.Namespace(split="full", benchmark_profile="historical-full")
        self.assertEqual(resolve_benchmark_profile(alias_args), "historical-full-raw")

    def test_dev_split_materializes_only_dev_by_default(self) -> None:
        args = argparse.Namespace(split="dev", materialize_full=False)
        self.assertEqual(resolve_requested_materialized_splits(args, "current-dev"), ("dev",))

    def test_dev_split_can_materialize_full_only_when_explicit(self) -> None:
        args = argparse.Namespace(split="dev", materialize_full=True)
        self.assertEqual(resolve_requested_materialized_splits(args, "current-dev"), ("dev", "full"))

    def test_historical_core_profile_requires_historical_corpus(self) -> None:
        with self.assertRaises(ValueError):
            validate_benchmark_profile(
                profile="historical-full-core",
                split="full",
                chunks_path=Path("data/chunks/chunks.jsonl"),
                corpus_label="current",
            )

    def test_historical_raw_profile_requires_historical_corpus(self) -> None:
        with self.assertRaises(ValueError):
            validate_benchmark_profile(
                profile="historical-full-raw",
                split="full",
                chunks_path=Path("data/chunks/chunks.jsonl"),
                corpus_label="current",
            )

    def test_historical_full_profile_requires_historical_corpus(self) -> None:
        with self.assertRaises(ValueError):
            validate_benchmark_profile(
                profile="historical-full",
                split="full",
                chunks_path=Path("data/chunks/chunks.jsonl"),
                corpus_label="current",
            )

    def test_apply_retrieval_domain_priority_moves_matching_rows_first(self) -> None:
        retrieval_result = {
            "dense_rows": [
                {"chunk_id": "c1", "file_name": "消费_2026-04-01_REF001_Report.pdf"},
                {"chunk_id": "c2", "file_name": "半导体_2026-04-01_REF002_Report.pdf"},
            ],
            "bm25_rows": [
                {"chunk_id": "c3", "file_name": "半导体_2026-04-01_REF003_Report.pdf"},
            ],
            "hybrid_rows": [
                {"chunk_id": "c4", "file_name": "消费_2026-04-01_REF004_Report.pdf"},
                {"chunk_id": "c5", "file_name": "半导体_2026-04-01_REF005_Report.pdf"},
            ],
            "rerank_rows": [
                {"chunk_id": "c6", "file_name": "消费_2026-04-01_REF006_Report.pdf"},
                {"chunk_id": "c7", "file_name": "半导体_2026-04-01_REF007_Report.pdf"},
            ],
        }
        prioritized = apply_retrieval_domain_priority(retrieval_result, "semiconductor")
        self.assertEqual([row["chunk_id"] for row in prioritized["dense_rows"]], ["c2", "c1"])
        self.assertEqual([row["chunk_id"] for row in prioritized["bm25_rows"]], ["c3"])
        self.assertEqual([row["chunk_id"] for row in prioritized["hybrid_rows"]], ["c5", "c4"])
        self.assertEqual([row["chunk_id"] for row in prioritized["rerank_rows"]], ["c7", "c6"])

    def test_apply_retrieval_domain_priority_treats_liquor_as_consumer_compatible(self) -> None:
        retrieval_result = {
            "dense_rows": [
                {"chunk_id": "c1", "file_name": "消费_2026-04-01_REF001_Report.pdf"},
                {"chunk_id": "c2", "file_name": "白酒_2026-04-01_REF002_Report.pdf"},
            ],
        }
        prioritized = apply_retrieval_domain_priority(retrieval_result, "liquor")
        self.assertEqual([row["chunk_id"] for row in prioritized["dense_rows"]], ["c2", "c1"])


if __name__ == "__main__":
    unittest.main()
