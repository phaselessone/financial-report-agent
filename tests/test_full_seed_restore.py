import unittest
from collections import Counter
from pathlib import Path

from src.evaluation.answer_eval import materialize_answer_eval_sets
from src.utils.io import read_json
from src.utils.io import read_jsonl
from run_answer_eval import enforce_historical_evidence_gate
from scripts.phase0_gate import DEFAULT_FULL_ATTESTATION


class FullSeedRestoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.seed_path = Path("data/eval_set/answer_eval_seed_full.jsonl")
        self.artifact_results_path = Path("artifacts/remote_20260401/outputs_reports/answer_eval_results_full.jsonl")

    def test_full_seed_matches_historical_question_ids(self) -> None:
        self.assertTrue(self.seed_path.exists(), f"missing full seed: {self.seed_path}")
        seed_rows = read_jsonl(self.seed_path)
        artifact_rows = read_jsonl(self.artifact_results_path)
        self.assertEqual(len(seed_rows), 50)
        self.assertEqual(
            {row["question_id"] for row in seed_rows},
            {row["question_id"] for row in artifact_rows},
        )

    def test_full_seed_keeps_expected_distribution(self) -> None:
        seed_rows = read_jsonl(self.seed_path)
        self.assertEqual(Counter(row["question_type"] for row in seed_rows), {"fact": 20, "comparison": 15, "inductive": 15})
        abstain_ids = sorted(row["question_id"] for row in seed_rows if row["must_abstain"])
        self.assertEqual(
            abstain_ids,
            [
                "compare_abstain_01",
                "compare_abstain_02",
                "fact_abstain_01",
                "fact_abstain_02",
                "inductive_abstain_01",
            ],
        )
        self.assertTrue(all(not row.get("manual_review_required", False) for row in seed_rows))

    def test_full_seed_materializes_without_missing_rows_when_stage6_chunks_exist(self) -> None:
        chunks_path = Path("tmp_stage6_data/chunks/chunks.jsonl")
        if not chunks_path.exists():
            self.skipTest(f"stage6 chunks not found: {chunks_path}")

        evidence_audit = enforce_historical_evidence_gate(
            profile="historical-full-raw",
            seed_path=self.seed_path,
            historical_results_path=self.artifact_results_path,
            benchmark_attestation_path=DEFAULT_FULL_ATTESTATION,
            chunks_path=chunks_path,
            corpus_attestation_path=Path(
                "benchmarks/full/historical-stage6-corpus.attestation.json"
            ),
            output_path=Path(
                "tmp_stage6_data/eval_set/historical_full_evidence_audit.json"
            ),
        )
        self.assertIsNotNone(evidence_audit)
        assert evidence_audit is not None
        self.assertEqual(
            evidence_audit["status"],
            "READY",
            f"stage6 content audit failed: {evidence_audit.get('blocking_reasons')}",
        )
        self.assertTrue(evidence_audit["formal_release_ready"])

        report_output_path = Path("tmp_stage6_data/eval_set/full_seed_materialization_report.json")
        dev_output_path = Path("tmp_stage6_data/eval_set/full_seed_materialization_dev.jsonl")
        full_output_path = Path("tmp_stage6_data/eval_set/full_seed_materialization_full.jsonl")
        chunks = read_jsonl(chunks_path)
        materialize_answer_eval_sets(
            dev_seed_path=self.seed_path,
            full_seed_path=self.seed_path,
            chunks=chunks,
            dev_output_path=dev_output_path,
            full_output_path=full_output_path,
            report_output_path=report_output_path,
        )
        report = read_json(report_output_path)
        self.assertEqual(report["splits"]["full"]["resolved_row_count"], 50)
        self.assertEqual(report["splits"]["full"]["missing_row_count"], 0)


if __name__ == "__main__":
    unittest.main()
