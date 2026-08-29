import argparse
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

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
        build_answer_eval_run_identity,
        enforce_historical_evidence_gate,
        resolve_answer_seed_paths,
        resolve_benchmark_profile,
        resolve_eval_limit,
        resolve_profile_output_dir,
        resolve_profile_run_output_dir,
        resolve_requested_materialized_splits,
        reserve_profile_run_output_dir,
        validate_benchmark_profile,
        validate_model_revision_for_profile,
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
                resolve_answer_seed_paths(
                    split="full", dev_seed_path=dev_seed, full_seed_path=full_seed
                )

    def test_dev_split_allows_missing_full_seed(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dev_seed = tmp_path / "answer_eval_seed_dev.jsonl"
            full_seed = tmp_path / "answer_eval_seed_full.jsonl"
            dev_seed.write_text("{}\n", encoding="utf-8")
            seed_paths = resolve_answer_seed_paths(
                split="dev", dev_seed_path=dev_seed, full_seed_path=full_seed
            )
            self.assertEqual(seed_paths, {"dev": dev_seed})

    def test_default_profile_follows_split(self) -> None:
        dev_args = argparse.Namespace(split="dev", benchmark_profile="")
        full_args = argparse.Namespace(split="full", benchmark_profile="")
        self.assertEqual(resolve_benchmark_profile(dev_args), "current-dev")
        self.assertEqual(resolve_benchmark_profile(full_args), "historical-full-core")

    def test_historical_full_alias_normalizes_to_raw(self) -> None:
        alias_args = argparse.Namespace(split="full", benchmark_profile="historical-full")
        self.assertEqual(resolve_benchmark_profile(alias_args), "historical-full-raw")

    def test_historical_core_and_raw_outputs_are_isolated(self) -> None:
        root = Path("outputs")
        self.assertEqual(resolve_profile_output_dir(root, "current-dev"), root)
        self.assertEqual(
            resolve_profile_output_dir(root, "historical-full-core"),
            root / "eval_profiles" / "historical-full-core",
        )
        self.assertEqual(
            resolve_profile_output_dir(root, "historical-full-raw"),
            root / "eval_profiles" / "historical-full-raw",
        )
        self.assertNotEqual(
            resolve_profile_output_dir(root, "historical-full-core"),
            resolve_profile_output_dir(root, "historical-full-raw"),
        )

    def test_historical_runs_are_reserved_without_overwriting_prior_outputs(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "outputs"
            core_run = reserve_profile_run_output_dir(root, "historical-full-core", "core-run-001")
            (core_run / "sentinel.txt").write_text("core", encoding="utf-8")
            raw_run = reserve_profile_run_output_dir(root, "historical-full-raw", "raw-run-001")

            self.assertEqual(
                core_run,
                root / "eval_profiles" / "historical-full-core" / "runs" / "core-run-001",
            )
            self.assertEqual(
                raw_run,
                root / "eval_profiles" / "historical-full-raw" / "runs" / "raw-run-001",
            )
            self.assertEqual((core_run / "sentinel.txt").read_text(encoding="utf-8"), "core")
            self.assertEqual(resolve_profile_run_output_dir(root, "current-dev", "ignored"), root)
            with self.assertRaises(FileExistsError):
                reserve_profile_run_output_dir(root, "historical-full-core", "core-run-001")

    def test_answer_eval_runner_builds_complete_canonical_run_identity(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            chunks_path = Path(tmp_dir) / "chunks.jsonl"
            chunks_path.write_text('{"chunk_id":"c1"}\n', encoding="utf-8")
            args = argparse.Namespace(
                benchmark_profile="historical-full",
                benchmark_version="history-v1",
                chunks_path=chunks_path,
                llm_provider="provider-x",
                llm_model="model-x",
                generation_model="local-model",
                model_revision="revision-1",
                temperature=0.0,
                prompt_version="answer-eval-v2",
                eval_limit=0,
                dense_top_k=20,
                bm25_top_k=20,
                rerank_top_k=5,
                rerank_candidates_k=0,
                embedding_model="embed-x",
                reranker_model="rerank-x",
                historical_evidence_gate_status="READY",
                historical_evidence_gate_proof_level="CONTENT_ANCHORED",
                historical_evidence_gate_report_sha256="a" * 64,
                historical_stage6_corpus_sha256="b" * 64,
            )
            rows = [{"question_id": "full_q1", "query": "q"}]

            identity = build_answer_eval_run_identity(
                args,
                rows,
                run_metadata={"git_sha": "commit", "source_manifest_hash": "manifest-sha256"},
                provider="provider-x",
                model="model-x",
            )

            self.assertEqual(identity.benchmark_profile, "historical-full-raw")
            self.assertEqual(identity.benchmark_version, "history-v1")
            self.assertEqual(identity.case_count, 1)
            self.assertEqual(len(identity.to_dict()), 15)
            self.assertEqual(
                identity.feature_flags["runtime"]["historical_evidence_gate"],
                {
                    "status": "READY",
                    "proof_level": "CONTENT_ANCHORED",
                    "report_sha256": "a" * 64,
                    "stage6_corpus_sha256": "b" * 64,
                },
            )

    def test_historical_profile_requires_versioned_model_revision(self) -> None:
        for revision in ("", "unversioned", "default", "latest", "unknown"):
            with self.subTest(revision=revision):
                with self.assertRaisesRegex(ValueError, "versioned --model-revision"):
                    validate_model_revision_for_profile(
                        profile="historical-full-core",
                        model_revision=revision,
                    )

        self.assertEqual(
            validate_model_revision_for_profile(
                profile="historical-full-raw",
                model_revision="provider-snapshot-2026-08-24",
            ),
            "provider-snapshot-2026-08-24",
        )

    def test_current_dev_may_retain_unversioned_non_publishable_identity(self) -> None:
        self.assertEqual(
            validate_model_revision_for_profile(
                profile="current-dev",
                model_revision="unversioned",
            ),
            "unversioned",
        )

    def test_historical_execution_invokes_formal_stage6_gate(self) -> None:
        ready_report = {
            "status": "READY",
            "proof_level": "CONTENT_ANCHORED",
            "attestation": {"validated": True},
            "corpus_attestation": {"validated": True},
            "candidate_corpus": {"hash_pinned": True},
        }
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            with patch(
                "scripts.historical_full_evidence_gate.run",
                return_value=ready_report,
            ) as gate:
                observed = enforce_historical_evidence_gate(
                    profile="historical-full-core",
                    seed_path=root / "seed.jsonl",
                    historical_results_path=root / "results.jsonl",
                    benchmark_attestation_path=root / "benchmark.attestation.json",
                    chunks_path=root / "stage6-chunks.jsonl",
                    corpus_attestation_path=root / "stage6.attestation.json",
                    output_path=root / "gate.json",
                )

        self.assertEqual(observed, ready_report)
        gate.assert_called_once()
        gate_args = gate.call_args.args[0]
        self.assertEqual(gate_args.chunks_path.name, "stage6-chunks.jsonl")
        self.assertEqual(gate_args.corpus_attestation_path.name, "stage6.attestation.json")
        self.assertIsNone(gate_args.expected_chunks_sha256)

    def test_historical_execution_rejects_incomplete_or_unattested_gate_report(self) -> None:
        malformed_ready = {
            "status": "READY",
            "proof_level": "CONTENT_ANCHORED",
            "attestation": {"validated": True},
            "corpus_attestation": {"validated": False},
            "candidate_corpus": {"hash_pinned": True},
        }
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            with patch(
                "scripts.historical_full_evidence_gate.run",
                return_value=malformed_ready,
            ):
                with self.assertRaisesRegex(ValueError, "is not READY"):
                    enforce_historical_evidence_gate(
                        profile="historical-full-raw",
                        seed_path=root / "seed.jsonl",
                        historical_results_path=root / "results.jsonl",
                        benchmark_attestation_path=root / "benchmark.attestation.json",
                        chunks_path=root / "stage6-chunks.jsonl",
                        corpus_attestation_path=root / "stage6.attestation.json",
                        output_path=root / "gate.json",
                    )

    def test_non_historical_execution_does_not_invoke_stage6_gate(self) -> None:
        with patch("scripts.historical_full_evidence_gate.run") as gate:
            observed = enforce_historical_evidence_gate(
                profile="current-dev",
                seed_path=Path("unused-seed.jsonl"),
                historical_results_path=Path("unused-results.jsonl"),
                benchmark_attestation_path=Path("unused-benchmark-attestation.json"),
                chunks_path=Path("unused-chunks.jsonl"),
                corpus_attestation_path=Path("unused-corpus-attestation.json"),
                output_path=Path("unused-output.json"),
            )

        self.assertIsNone(observed)
        gate.assert_not_called()

    def test_dev_split_materializes_only_dev_by_default(self) -> None:
        args = argparse.Namespace(split="dev", materialize_full=False)
        self.assertEqual(resolve_requested_materialized_splits(args, "current-dev"), ("dev",))

    def test_dev_split_can_materialize_full_only_when_explicit(self) -> None:
        args = argparse.Namespace(split="dev", materialize_full=True)
        self.assertEqual(
            resolve_requested_materialized_splits(args, "current-dev"), ("dev", "full")
        )

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
