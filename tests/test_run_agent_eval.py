"""run_agent_eval CLI helper tests (checklist v3.0 §P3 entry point)."""

from __future__ import annotations

import argparse
import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

try:
    import pymupdf  # noqa: F401
except ModuleNotFoundError:
    HAS_FITZ = False
    run_agent_eval = None
else:
    HAS_FITZ = True
    import run_agent_eval as run_agent_eval_module
    from run_agent_eval import (
        build_agent_config,
        build_eval_bundle_config,
        build_eval_run_identity,
        resolve_baseline_results_path,
        resolve_report_paths,
        run_compare_mode,
        run_agentic_eval_rows,
        run_baseline_rows,
        write_validated_eval_bundle,
    )

from src.agent.config import AgentConfig
from src.agent.llm_claim_judge import build_claim_llm_judge
from src.evaluation.eval_bundle import (
    EvalBundleIntegrityError,
    build_corpus_asset_manifest,
    read_eval_bundle,
    require_ready_eval_bundle_integrity,
    write_eval_bundle,
)
from src.evaluation.run_identity import IncompatibleRunIdentityError
from src.evaluation.trajectory_eval import build_failed_agent_trace_row
from src.llm.types import LLMResponse
from tests.test_agent_flow import FakeAnswerer, FakeLLM, FakeRuntime, make_result, make_row, supported_draft


def make_eval_row(question_id: str = "q1") -> dict:
    return {
        "question_id": question_id,
        "query": "半导体行业景气度如何？",
        "question_type": "fact",
        "intent": "numeric_fact",
        "industry": "semiconductor",
        "category": "agent_numeric_missing",
        "must_recover": True,
    }


@unittest.skipUnless(HAS_FITZ, "fitz is required for run_agent_eval imports")
class RunAgentEvalArgTests(unittest.TestCase):
    @staticmethod
    def _identity_args(tmp_path: Path, *, mode: str) -> argparse.Namespace:
        chunks_path = tmp_path / "chunks.jsonl"
        facts_path = tmp_path / "facts.jsonl"
        aliases_path = tmp_path / "aliases.json"
        chunks_path.write_text('{"chunk_id":"c1"}\n', encoding="utf-8")
        facts_path.write_text('{"fact_id":"f1"}\n', encoding="utf-8")
        aliases_path.write_text('{"A公司":["A"]}', encoding="utf-8")
        return argparse.Namespace(
            mode=mode,
            benchmark_profile="contract-v1",
            benchmark_version="1.0",
            chunks_path=chunks_path,
            facts_path=facts_path,
            company_aliases_path=aliases_path,
            llm_provider="provider-x",
            llm_model="model-x",
            llm_model_revision="rev-1",
            temperature=0.0,
            prompt_version="prompt-v2",
            max_steps=12,
            max_retrieval_rounds=3,
            max_query_rewrites=2,
            max_generation_attempts=2,
            max_llm_calls=6,
            claim_llm_budget=2,
            max_total_tokens=0,
            embedding_model="embed-x",
            reranker_model="rerank-x",
            runtime_profile="cpu",
            embedding_device="cpu",
            reranker_device="cpu",
            device=None,
            embedding_batch_size=8,
            rerank_batch_size=4,
            dense_top_k=20,
            bm25_top_k=20,
            rerank_top_k=5,
            rerank_candidates_k=0,
            rebuild_indexes=False,
            enable_tool_orchestration=True,
        )

    def test_eval_identity_separates_treatment_from_shared_config(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            rows = [make_eval_row("q1"), make_eval_row("q2")]
            metadata = {"git_sha": "abc123", "source_manifest_id": "source:sha256"}
            baseline_args = self._identity_args(tmp_path, mode="baseline")
            baseline = build_eval_run_identity(baseline_args, rows, metadata=metadata)
            agentic_args = self._identity_args(tmp_path, mode="agentic")
            agentic = build_eval_run_identity(agentic_args, rows, metadata=metadata)

            baseline.assert_comparable(agentic)
            self.assertNotEqual(baseline.run_id, agentic.run_id)
            self.assertEqual(baseline.case_count, 2)
            self.assertEqual(
                build_eval_bundle_config(baseline_args, baseline)["runtime"],
                build_eval_bundle_config(agentic_args, agentic)["runtime"],
            )
            self.assertNotEqual(
                build_eval_bundle_config(baseline_args, baseline)["treatments"],
                build_eval_bundle_config(agentic_args, agentic)["treatments"],
            )
            self.assertIn("max_tool_calls", baseline.to_dict()["budgets"])
            self.assertEqual(baseline.to_dict()["budgets"]["claim_llm_budget"], 2)

    def test_eval_identity_binds_explicit_three_asset_manifest_hash(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            args = self._identity_args(tmp_path, mode="agentic")
            manifest = build_corpus_asset_manifest(
                chunks_path=args.chunks_path,
                facts_path=args.facts_path,
                aliases_path=args.company_aliases_path,
            )

            identity = build_eval_run_identity(
                args,
                [make_eval_row("q1")],
                metadata={"git_sha": "abc123", "source_manifest_hash": "source-sha256"},
            )

            self.assertEqual(identity.corpus_hash, manifest["corpus_hash"])
            self.assertEqual(
                identity.feature_flags["runtime"]["corpus_asset_manifest_hash"],
                manifest["manifest_hash"],
            )

    def test_eval_identity_rejects_retrieval_runtime_config_mismatch(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            rows = [make_eval_row("q1")]
            metadata = {"git_sha": "abc123", "source_manifest_hash": "source-sha256"}
            baseline_args = self._identity_args(tmp_path, mode="baseline")
            baseline = build_eval_run_identity(baseline_args, rows, metadata=metadata)
            agentic_args = self._identity_args(tmp_path, mode="agentic")
            agentic_args.embedding_batch_size = 32
            agentic = build_eval_run_identity(agentic_args, rows, metadata=metadata)

            with self.assertRaises(IncompatibleRunIdentityError):
                baseline.assert_comparable(agentic)

    def test_build_agent_config_maps_budgets(self) -> None:
        args = argparse.Namespace(
            max_steps=3,
            max_retrieval_rounds=2,
            max_query_rewrites=1,
            max_generation_attempts=2,
            max_llm_calls=4,
            claim_llm_budget=2,
            max_total_tokens=500,
        )
        config = build_agent_config(args)
        self.assertIsInstance(config, AgentConfig)
        self.assertEqual(config.max_steps, 3)
        self.assertEqual(config.max_retrieval_rounds, 2)
        self.assertEqual(config.max_query_rewrites, 1)
        self.assertEqual(config.max_generation_attempts, 2)
        self.assertEqual(config.max_llm_calls, 4)
        self.assertEqual(config.claim_llm_budget, 2)
        self.assertEqual(config.max_total_tokens, 500)

    def test_resolve_baseline_results_path_defaults_to_reports_dir(self) -> None:
        args = argparse.Namespace(baseline_results_path=None)
        output_dir = Path("outputs")
        self.assertEqual(
            resolve_baseline_results_path(args, output_dir),
            output_dir / "reports" / "agent_baseline_results_dev.jsonl",
        )

    def test_resolve_baseline_results_path_prefers_explicit_path(self) -> None:
        args = argparse.Namespace(baseline_results_path=Path("artifacts/custom.jsonl"))
        self.assertEqual(resolve_baseline_results_path(args, Path("outputs")), Path("artifacts/custom.jsonl"))

    def test_resolve_report_paths_covers_all_artifacts(self) -> None:
        paths = resolve_report_paths(Path("outputs"))
        for key in (
            "agent_eval_results",
            "agent_traces",
            "agent_eval_summary_json",
            "agent_eval_summary_md",
            "api_usage_summary_json",
            "api_usage_summary_md",
            "baseline_results",
            "baseline_summary",
            "comparison_json",
            "comparison_md",
            "gate_verdict",
            "manifest_report",
            "agentic_bundle_ref",
            "baseline_bundle_ref",
        ):
            self.assertIn(key, paths)
        self.assertTrue(all(path.parent == Path("outputs") / "reports" for path in paths.values()))

    def test_compare_mode_rejects_bundle_source_mismatch_before_reading_benchmark(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            rows = [make_eval_row("q1")]
            metadata = {"git_sha": "abc123", "source_manifest_hash": "source-sha256"}
            baseline_args = self._identity_args(tmp_path, mode="baseline")
            baseline_identity = build_eval_run_identity(baseline_args, rows, metadata=metadata)
            agentic_args = self._identity_args(tmp_path, mode="agentic")
            agentic_identity = replace(
                build_eval_run_identity(agentic_args, rows, metadata=metadata),
                source_manifest_hash="different-source",
            )
            baseline_bundle = write_eval_bundle(
                tmp_path / "baseline",
                identity=baseline_identity,
                config=build_eval_bundle_config(baseline_args, baseline_identity),
                metrics={},
                per_case=rows,
                trajectories=[],
            )
            agentic_bundle = write_eval_bundle(
                tmp_path / "agentic",
                identity=agentic_identity,
                config=build_eval_bundle_config(agentic_args, agentic_identity),
                metrics={},
                per_case=rows,
                trajectories=rows,
            )
            compare_args = agentic_args
            compare_args.mode = "compare"
            compare_args.output_dir = tmp_path / "outputs"
            compare_args.baseline_bundle_path = baseline_bundle.path
            compare_args.agentic_bundle_path = agentic_bundle.path
            compare_args.agent_seed_path = tmp_path / "must-not-be-read.jsonl"
            compare_args.agent_eval_path = tmp_path / "must-not-be-written.jsonl"

            with self.assertRaises(IncompatibleRunIdentityError):
                run_compare_mode(compare_args, [], tmp_path / "outputs" / "reports")

            self.assertFalse(compare_args.agent_eval_path.exists())

    def test_agentic_bundle_writer_blocks_failed_evidence_integrity(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            args = self._identity_args(tmp_path, mode="agentic")
            rows = [make_eval_row("q1")]
            identity = build_eval_run_identity(
                args,
                rows,
                metadata={"git_sha": "abc123", "source_manifest_hash": "source-sha256"},
            )
            invalid_trace = {
                "question_id": "q1",
                "run_id": identity.run_id,
                "run_identity": identity.to_dict(),
                "claims": [],
                "citations": [{"evidence_id": "c1", "doc_id": "d1", "page": 1}],
                "used_evidence_ids": ["c1"],
                "abstained": False,
                "calculations": {},
            }

            with self.assertRaisesRegex(EvalBundleIntegrityError, "evidence integrity BLOCKED"):
                write_validated_eval_bundle(
                    tmp_path / "eval",
                    identity=identity,
                    config=build_eval_bundle_config(args, identity),
                    metrics={},
                    per_case=[invalid_trace],
                    trajectories=[invalid_trace],
                    chunks_path=args.chunks_path,
                    metadata={"mode": "agentic"},
                )

            self.assertFalse((tmp_path / "eval" / identity.run_id).exists())

    def test_agentic_bundle_writer_cannot_skip_integrity_when_mode_is_omitted(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            args = self._identity_args(tmp_path, mode="agentic")
            rows = [make_eval_row("q1")]
            identity = build_eval_run_identity(
                args,
                rows,
                metadata={"git_sha": "abc123", "source_manifest_hash": "source-sha256"},
            )
            invalid_trace = {
                "question_id": "q1",
                "run_id": identity.run_id,
                "run_identity": identity.to_dict(),
                "claims": [],
                "citations": [{"evidence_id": "c1", "doc_id": "d1", "page": 1}],
                "used_evidence_ids": ["c1"],
                "abstained": False,
                "calculations": {},
            }

            with self.assertRaisesRegex(EvalBundleIntegrityError, "evidence integrity BLOCKED"):
                write_validated_eval_bundle(
                    tmp_path / "eval",
                    identity=identity,
                    config=build_eval_bundle_config(args, identity),
                    metrics={},
                    per_case=[invalid_trace],
                    trajectories=[invalid_trace],
                    chunks_path=args.chunks_path,
                    metadata={},
                )

    def test_agentic_bundle_writer_records_ready_integrity_verdict(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            args = self._identity_args(tmp_path, mode="agentic")
            rows = [make_eval_row("q1")]
            identity = build_eval_run_identity(
                args,
                rows,
                metadata={"git_sha": "abc123", "source_manifest_hash": "source-sha256"},
            )
            valid_trace = {
                "question_id": "q1",
                "run_id": identity.run_id,
                "run_identity": identity.to_dict(),
                "claims": [],
                "citations": [],
                "used_evidence_ids": [],
                "abstained": True,
                "calculations": {},
                "tool_calls": [],
                "trajectory_events": [
                    {"event_id": "event-1", "node": "finalize", "status": "SUCCESS"}
                ],
                "dependency_coverage": {},
                "failure_attribution": {"has_failure": False, "root_cause": None},
            }

            bundle = write_validated_eval_bundle(
                tmp_path / "eval",
                identity=identity,
                config=build_eval_bundle_config(args, identity),
                metrics={},
                per_case=[valid_trace],
                trajectories=[valid_trace],
                chunks_path=args.chunks_path,
                metadata={"mode": "agentic"},
            )

            require_ready_eval_bundle_integrity(bundle)
            self.assertEqual(bundle.metadata["integrity_verdict"]["status"], "READY")

    def test_agentic_bundle_preserves_one_runtime_failure_as_an_auditable_row(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            args = self._identity_args(tmp_path, mode="agentic")
            rows = [make_eval_row("q1"), make_eval_row("q2")]
            identity = build_eval_run_identity(
                args,
                rows,
                metadata={"git_sha": "abc123", "source_manifest_hash": "source-sha256"},
            )
            success = {
                "question_id": "q1",
                "run_id": identity.run_id,
                "run_identity": identity.to_dict(),
                "claims": [],
                "citations": [],
                "used_evidence_ids": [],
                "abstained": True,
                "failed": False,
                "calculations": {},
                "tool_calls": [],
                "trajectory_events": [
                    {"event_id": "event-1", "node": "finalize", "status": "SUCCESS"}
                ],
                "dependency_coverage": {},
                "failure_attribution": {"has_failure": False, "root_cause": None},
            }
            failed = build_failed_agent_trace_row(
                eval_row=rows[1],
                error=RuntimeError("provider unavailable"),
                end_to_end_latency_ms=3.0,
                run_identity=identity,
            )

            bundle = write_validated_eval_bundle(
                tmp_path / "eval",
                identity=identity,
                config=build_eval_bundle_config(args, identity),
                metrics={"failed_query_count": 1},
                per_case=[success, failed],
                trajectories=[success, failed],
                chunks_path=args.chunks_path,
                metadata={"mode": "agentic"},
            )

            require_ready_eval_bundle_integrity(bundle)
            self.assertEqual(bundle.metadata["integrity_verdict"]["status"], "READY")
            self.assertTrue(bundle.trajectories[1]["failed"])
            self.assertTrue(bundle.trajectories[1]["failure_attribution"]["has_failure"])

    def test_failed_validated_overwrite_preserves_previous_ready_bundle(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            args = self._identity_args(tmp_path, mode="agentic")
            rows = [make_eval_row("q1")]
            identity = build_eval_run_identity(
                args,
                rows,
                metadata={"git_sha": "abc123", "source_manifest_hash": "source-sha256"},
            )
            valid_trace = {
                "question_id": "q1",
                "run_id": identity.run_id,
                "run_identity": identity.to_dict(),
                "claims": [],
                "citations": [],
                "used_evidence_ids": [],
                "abstained": True,
                "failed": False,
                "calculations": {},
                "tool_calls": [],
                "trajectory_events": [
                    {"event_id": "event-1", "node": "finalize", "status": "SUCCESS"}
                ],
                "dependency_coverage": {},
                "failure_attribution": {"has_failure": False, "root_cause": None},
            }
            original = write_validated_eval_bundle(
                tmp_path / "eval",
                identity=identity,
                config=build_eval_bundle_config(args, identity),
                metrics={"sentinel": "original"},
                per_case=[valid_trace],
                trajectories=[valid_trace],
                chunks_path=args.chunks_path,
                metadata={"mode": "agentic"},
            )
            invalid_trace = {
                **valid_trace,
                "abstained": False,
                "claims": [],
                "trajectory_events": [],
            }

            with self.assertRaisesRegex(EvalBundleIntegrityError, "evidence integrity BLOCKED"):
                write_validated_eval_bundle(
                    tmp_path / "eval",
                    identity=identity,
                    config=build_eval_bundle_config(args, identity),
                    metrics={"sentinel": "replacement"},
                    per_case=[invalid_trace],
                    trajectories=[invalid_trace],
                    chunks_path=args.chunks_path,
                    metadata={"mode": "agentic"},
                    overwrite=True,
                )

            preserved = read_eval_bundle(original.path)
            require_ready_eval_bundle_integrity(preserved)
            self.assertEqual(preserved.metrics["sentinel"], "original")

    def test_agentic_bundle_writer_blocks_missing_strict_trace_sections(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            args = self._identity_args(tmp_path, mode="agentic")
            rows = [make_eval_row("q1")]
            identity = build_eval_run_identity(
                args,
                rows,
                metadata={"git_sha": "abc123", "source_manifest_hash": "source-sha256"},
            )
            incomplete_trace = {
                "question_id": "q1",
                "run_id": identity.run_id,
                "run_identity": identity.to_dict(),
                "claims": [],
                "citations": [],
                "used_evidence_ids": [],
                "abstained": True,
                "calculations": {},
            }

            with self.assertRaisesRegex(EvalBundleIntegrityError, "evidence integrity BLOCKED"):
                write_validated_eval_bundle(
                    tmp_path / "eval",
                    identity=identity,
                    config=build_eval_bundle_config(args, identity),
                    metrics={},
                    per_case=[incomplete_trace],
                    trajectories=[incomplete_trace],
                    chunks_path=args.chunks_path,
                    metadata={"mode": "agentic"},
                )

    def test_main_does_not_publish_legacy_reports_before_bundle_gate_passes(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            args = self._identity_args(tmp_path, mode="baseline")
            args.output_dir = tmp_path / "outputs"
            args.agent_seed_path = tmp_path / "seed.jsonl"
            args.agent_eval_path = tmp_path / "materialized.jsonl"
            args.retrieval_seed_path = tmp_path / "retrieval-seed.jsonl"
            args.split = "dev"
            args.prepare_agent_benchmark = False
            args.eval_limit = 0
            args.limit = 0
            args.skip_artifact_archive = True
            args.artifact_root = tmp_path / "artifacts"
            args.artifact_label = ""
            args.corpus_label = "fixture"
            args.overwrite_eval_bundle = False
            rows = [make_eval_row("q1")]
            result_rows = [
                {
                    "question_id": "q1",
                    "final_answer": "answer",
                    "abstained": False,
                    "citations": [],
                    "support_validation": {},
                }
            ]

            class _Answerer:
                llm_provider = "provider-x"
                llm_model = "model-x"

            observed_bundle_call: dict[str, object] = {}

            def block_bundle_publish(*_args, **kwargs):
                observed_bundle_call.update(kwargs)
                raise EvalBundleIntegrityError("gate blocked")

            def materialize_to_requested_paths(*, output_path, report_output_path, **_kwargs):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text('{"question_id":"q1"}\n', encoding="utf-8")
                report_output_path.parent.mkdir(parents=True, exist_ok=True)
                report_output_path.write_text('{"status":"READY"}', encoding="utf-8")
                return rows

            def build_metadata_at_requested_path(_args, _rows, report_dir):
                report_dir.mkdir(parents=True, exist_ok=True)
                (report_dir / "source_manifest.json").write_text(
                    '{"source_manifest_hash":"source-sha256"}', encoding="utf-8"
                )
                return {"git_sha": "abc123", "source_manifest_hash": "source-sha256"}

            with (
                patch.object(run_agent_eval_module, "parse_args", return_value=args),
                patch.object(
                    run_agent_eval_module,
                    "materialize_agent_eval_set",
                    side_effect=materialize_to_requested_paths,
                ),
                patch.object(run_agent_eval_module, "_build_runtime", return_value=object()),
                patch.object(run_agent_eval_module, "_build_answerer", return_value=_Answerer()),
                patch.object(
                    run_agent_eval_module,
                    "_build_metadata",
                    side_effect=build_metadata_at_requested_path,
                ),
                patch.object(
                    run_agent_eval_module,
                    "run_baseline_rows",
                    return_value=(result_rows, [[]]),
                ),
                patch.object(run_agent_eval_module, "evaluate_agent_quality", return_value={}),
                patch.object(
                    run_agent_eval_module,
                    "build_api_usage_summary_from_calls",
                    return_value={},
                ),
                patch.object(
                    run_agent_eval_module,
                    "write_validated_eval_bundle",
                    side_effect=block_bundle_publish,
                ),
            ):
                with self.assertRaisesRegex(EvalBundleIntegrityError, "gate blocked"):
                    run_agent_eval_module.main()

            report_paths = resolve_report_paths(args.output_dir)
            self.assertFalse(report_paths["baseline_results"].exists())
            self.assertFalse(report_paths["baseline_summary"].exists())
            self.assertFalse(report_paths["baseline_bundle_ref"].exists())
            self.assertFalse(report_paths["manifest_report"].exists())
            self.assertFalse((args.output_dir / "reports" / "source_manifest.json").exists())
            self.assertFalse(args.agent_eval_path.exists())
            self.assertEqual(
                observed_bundle_call["corpus_asset_manifest"]["corpus_hash"],
                observed_bundle_call["identity"].corpus_hash,
            )


@unittest.skipUnless(HAS_FITZ, "fitz is required for run_agent_eval imports")
class RunAgentEvalLoopTests(unittest.TestCase):
    def test_run_agentic_eval_rows_produces_trace_rows(self) -> None:
        runtime = FakeRuntime([make_result([make_row(chunk_id="c1", doc_id="d1", text="半导体行业景气度持续回升。")])])
        llm = FakeLLM([])
        answerer = FakeAnswerer([supported_draft("半导体行业景气度持续回升。")])
        trace_rows = run_agentic_eval_rows(
            eval_rows=[make_eval_row()],
            runtime=runtime,
            answerer=answerer,
            llm=llm,
            config=AgentConfig(),
        )
        self.assertEqual(len(trace_rows), 1)
        row = trace_rows[0]
        self.assertEqual(row["question_id"], "q1")
        self.assertFalse(row["failed"])
        self.assertEqual(row["termination_reason"], "completed")

    def test_run_agentic_eval_rows_captures_failures_per_row(self) -> None:
        class BoomRuntime:
            def search(self, query: str):
                raise RuntimeError("index broken")

        llm = FakeLLM([])
        answerer = FakeAnswerer([])
        trace_rows = run_agentic_eval_rows(
            eval_rows=[make_eval_row(), make_eval_row("q2")],
            runtime=BoomRuntime(),
            answerer=answerer,
            llm=llm,
            config=AgentConfig(),
        )
        self.assertEqual(len(trace_rows), 2)
        self.assertTrue(all(row["failed"] for row in trace_rows))
        self.assertEqual(trace_rows[0]["error_type"], "RuntimeError")

    def test_run_agentic_eval_rows_passes_enabled_claim_judge_and_retains_usage(self) -> None:
        eval_row = make_eval_row()
        eval_row["query"] = "公司的研发投入如何？"
        extractor = LLMResponse(
            content=json.dumps(
                {
                    "claims": [
                        {
                            "id": "c1",
                            "text": "公司加大研发投入。",
                            "claim_type": "EXTRACTED",
                            "is_core": True,
                            "source_step_ids": ["search_1"],
                            "parent_claim_ids": [],
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            provider="fake",
            model="fake",
        )
        judgment = LLMResponse(
            content=json.dumps(
                {
                    "status": "ENTAILED",
                    "evidence_ids": ["E1"],
                    "score": 0.95,
                    "reasons": ["directional qualitative judgment"],
                }
            ),
            provider="fake",
            model="fake",
            prompt_tokens=7,
            completion_tokens=3,
        )
        llm = FakeLLM([extractor, judgment])
        judge = build_claim_llm_judge(llm, budget=1)

        rows = run_agentic_eval_rows(
            eval_rows=[eval_row],
            runtime=FakeRuntime(
                [make_result([make_row(chunk_id="E1", doc_id="D1", text="公司持续增加研发资源。")])]
            ),
            answerer=FakeAnswerer([supported_draft("公司加大研发投入。")]),
            llm=llm,
            config=AgentConfig(
                strict_claim_verification=True,
                claim_llm_budget=1,
                max_claim_retrievals=0,
            ),
            llm_judge=judge,
        )

        assert rows[0]["claims"][0]["verification"]["method"] == "llm"
        judge_logs = [entry for entry in rows[0]["llm_calls_log"] if entry["node"] == "verify_answer"]
        assert len(judge_logs) == 1
        assert judge_logs[0]["total_tokens"] == 10

    def test_run_baseline_rows_records_answers_and_per_query_calls(self) -> None:
        runtime = FakeRuntime([make_result([make_row(chunk_id="c1", doc_id="d1", text="半导体行业景气度持续回升。")])])
        answerer = FakeAnswerer([supported_draft("半导体行业景气度持续回升。")], llm_calls_per_answer=2)
        result_rows, calls_by_query = run_baseline_rows(
            eval_rows=[make_eval_row()],
            runtime=runtime,
            answerer=answerer,
        )
        self.assertEqual(len(result_rows), 1)
        self.assertEqual(result_rows[0]["question_id"], "q1")
        self.assertEqual(len(calls_by_query), 1)
        self.assertEqual(len(calls_by_query[0]), 2)

    def test_baseline_row_feeds_agent_trace_quality_hits(self) -> None:
        runtime = FakeRuntime([make_result([make_row(chunk_id="c1", doc_id="d1", text="半导体行业景气度持续回升。")])])
        answerer = FakeAnswerer([supported_draft("半导体行业景气度持续回升。")], llm_calls_per_answer=1)
        result_rows, _ = run_baseline_rows(
            eval_rows=[make_eval_row()],
            runtime=runtime,
            answerer=answerer,
        )
        # The answer row carries the fields the quality hit functions need.
        row = result_rows[0]
        for key in ("final_answer", "abstained", "citations", "support_validation"):
            self.assertIn(key, row)


if __name__ == "__main__":
    unittest.main()
