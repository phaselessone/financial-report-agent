import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.evaluation.answer_eval import evaluate_answer_results, materialize_answer_eval_sets
from src.evaluation.benchmark_assets import build_doc_manifest


class AnswerEvalTests(unittest.TestCase):
    def _sample_chunks(self) -> list[dict[str, object]]:
        return [
            {
                "chunk_id": "doc-a-c1",
                "doc_id": "doc-a",
                "file_name": "半导体_2026-04-01_REF001_晶圆代工周报.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "晶圆代工行业景气上行。",
                "child_text": "晶圆代工行业景气上行。",
                "industry": "semiconductor",
                "element_type": "paragraph",
                "chunk_type": "text",
            },
            {
                "chunk_id": "doc-a-c2",
                "doc_id": "doc-a",
                "file_name": "半导体_2026-04-01_REF001_晶圆代工周报.pdf",
                "page_start": 3,
                "page_end": 3,
                "text": "附录页提到晶圆代工行业景气上行。",
                "child_text": "附录页提到晶圆代工行业景气上行。",
                "industry": "semiconductor",
                "element_type": "paragraph",
                "chunk_type": "text",
            },
            {
                "chunk_id": "doc-b-c1",
                "doc_id": "doc-b",
                "file_name": "新能源_2026-04-01_REF002_储能行业观察.pdf",
                "page_start": 2,
                "page_end": 2,
                "text": "储能需求持续增长。",
                "child_text": "储能需求持续增长。",
                "industry": "new_energy",
                "element_type": "paragraph",
                "chunk_type": "text",
            },
        ]

    def _write_seed_rows(self, path: Path, rows: list[dict[str, object]]) -> None:
        path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )

    def test_materialize_answer_eval_sets_keeps_dev_and_full_separate(self) -> None:
        chunks = self._sample_chunks()
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dev_seed = tmp_path / "answer_eval_seed_dev.jsonl"
            full_seed = tmp_path / "answer_eval_seed_full.jsonl"
            dev_output = tmp_path / "answer_eval_dev.jsonl"
            full_output = tmp_path / "answer_eval_full.jsonl"
            report_output = tmp_path / "answer_eval_manifest_report.json"
            self._write_seed_rows(
                dev_seed,
                [
                    {
                        "question_id": "dev_q1",
                        "query": "哪份报告讨论了晶圆代工？",
                        "question_type": "fact",
                        "intent": "report_lookup",
                        "industry": "semiconductor",
                        "target_doc_keys": ["晶圆代工周报"],
                        "gold_answer": "晶圆代工周报",
                        "gold_chunk_ids": ["doc-a-c1"],
                    }
                ],
            )
            self._write_seed_rows(
                full_seed,
                [
                    {
                        "question_id": "full_q1",
                        "query": "哪份报告讨论了晶圆代工？",
                        "question_type": "fact",
                        "intent": "report_lookup",
                        "industry": "semiconductor",
                        "target_doc_keys": ["晶圆代工周报"],
                        "gold_answer": "晶圆代工周报",
                        "gold_chunk_ids": ["doc-a-c1"],
                    },
                    {
                        "question_id": "full_q2",
                        "query": "哪份报告讨论了储能？",
                        "question_type": "fact",
                        "intent": "report_lookup",
                        "industry": "new_energy",
                        "target_doc_keys": ["储能行业观察"],
                        "gold_answer": "储能行业观察",
                        "gold_chunk_ids": ["doc-b-c1"],
                    },
                ],
            )

            answer_eval_sets = materialize_answer_eval_sets(
                dev_seed_path=dev_seed,
                full_seed_path=full_seed,
                chunks=chunks,
                dev_output_path=dev_output,
                full_output_path=full_output,
                report_output_path=report_output,
            )

            self.assertEqual(len(answer_eval_sets["dev"]), 1)
            self.assertEqual(len(answer_eval_sets["full"]), 2)
            self.assertNotEqual(answer_eval_sets["dev"], answer_eval_sets["full"])
            report = json.loads(report_output.read_text(encoding="utf-8"))
            self.assertEqual(report["available_splits"], ["dev", "full"])

    def test_historical_full_core_drops_conflicting_query_groups(self) -> None:
        chunks = self._sample_chunks()
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dev_seed = tmp_path / "answer_eval_seed_dev.jsonl"
            full_seed = tmp_path / "answer_eval_seed_full.jsonl"
            dev_output = tmp_path / "answer_eval_dev.jsonl"
            full_output = tmp_path / "answer_eval_full.jsonl"
            report_output = tmp_path / "answer_eval_manifest_report.json"
            self._write_seed_rows(
                dev_seed,
                [
                    {
                        "question_id": "dev_q1",
                        "query": "哪份报告讨论了晶圆代工？",
                        "question_type": "fact",
                        "intent": "report_lookup",
                        "industry": "semiconductor",
                        "target_doc_keys": ["晶圆代工周报"],
                        "gold_answer": "晶圆代工周报",
                        "gold_chunk_ids": ["doc-a-c1"],
                    }
                ],
            )
            self._write_seed_rows(
                full_seed,
                [
                    {
                        "question_id": "full_conflict_1",
                        "query": "近期农林牧渔报告共同强调了哪些主题？",
                        "question_type": "inductive",
                        "intent": "inductive",
                        "industry": "agriculture",
                        "target_doc_keys": ["晶圆代工周报"],
                        "gold_answer": "主题一",
                        "gold_chunk_ids": ["doc-a-c1"],
                    },
                    {
                        "question_id": "full_conflict_2",
                        "query": "近期农林牧渔报告共同强调了哪些主题？",
                        "question_type": "inductive",
                        "intent": "inductive",
                        "industry": "agriculture",
                        "target_doc_keys": ["储能行业观察"],
                        "gold_answer": "主题二",
                        "gold_chunk_ids": ["doc-b-c1"],
                    },
                    {
                        "question_id": "full_keep_1",
                        "query": "哪份报告讨论了晶圆代工？",
                        "question_type": "fact",
                        "intent": "report_lookup",
                        "industry": "semiconductor",
                        "target_doc_keys": ["晶圆代工周报"],
                        "gold_answer": "晶圆代工周报",
                        "gold_chunk_ids": ["doc-a-c1"],
                    },
                ],
            )

            answer_eval_sets = materialize_answer_eval_sets(
                dev_seed_path=dev_seed,
                full_seed_path=full_seed,
                chunks=chunks,
                dev_output_path=dev_output,
                full_output_path=full_output,
                report_output_path=report_output,
                requested_splits=("full",),
                benchmark_profile="historical-full-core",
            )

            self.assertEqual([row["question_id"] for row in answer_eval_sets["full"]], ["full_keep_1"])
            report = json.loads(report_output.read_text(encoding="utf-8"))
            full_report = report["splits"]["full"]
            self.assertEqual(full_report["resolved_row_count"], 3)
            self.assertEqual(full_report["materialized_row_count"], 1)
            self.assertEqual(full_report["benchmark_profile"], "historical-full-core")
            self.assertEqual(full_report["profile_filter"]["dropped_row_count"], 2)
            self.assertEqual(full_report["profile_filter"]["dropped_conflict_group_count"], 1)
            self.assertEqual(
                full_report["profile_filter"]["dropped_question_ids"],
                ["full_conflict_1", "full_conflict_2"],
            )

    def test_historical_full_raw_keeps_conflicting_query_groups(self) -> None:
        chunks = self._sample_chunks()
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dev_seed = tmp_path / "answer_eval_seed_dev.jsonl"
            full_seed = tmp_path / "answer_eval_seed_full.jsonl"
            dev_output = tmp_path / "answer_eval_dev.jsonl"
            full_output = tmp_path / "answer_eval_full.jsonl"
            report_output = tmp_path / "answer_eval_manifest_report.json"
            self._write_seed_rows(
                dev_seed,
                [
                    {
                        "question_id": "dev_q1",
                        "query": "哪份报告讨论了晶圆代工？",
                        "question_type": "fact",
                        "intent": "report_lookup",
                        "industry": "semiconductor",
                        "target_doc_keys": ["晶圆代工周报"],
                        "gold_answer": "晶圆代工周报",
                        "gold_chunk_ids": ["doc-a-c1"],
                    }
                ],
            )
            self._write_seed_rows(
                full_seed,
                [
                    {
                        "question_id": "full_conflict_1",
                        "query": "近期农林牧渔报告共同强调了哪些主题？",
                        "question_type": "inductive",
                        "intent": "inductive",
                        "industry": "agriculture",
                        "target_doc_keys": ["晶圆代工周报"],
                        "gold_answer": "主题一",
                        "gold_chunk_ids": ["doc-a-c1"],
                    },
                    {
                        "question_id": "full_conflict_2",
                        "query": "近期农林牧渔报告共同强调了哪些主题？",
                        "question_type": "inductive",
                        "intent": "inductive",
                        "industry": "agriculture",
                        "target_doc_keys": ["储能行业观察"],
                        "gold_answer": "主题二",
                        "gold_chunk_ids": ["doc-b-c1"],
                    },
                    {
                        "question_id": "full_keep_1",
                        "query": "哪份报告讨论了晶圆代工？",
                        "question_type": "fact",
                        "intent": "report_lookup",
                        "industry": "semiconductor",
                        "target_doc_keys": ["晶圆代工周报"],
                        "gold_answer": "晶圆代工周报",
                        "gold_chunk_ids": ["doc-a-c1"],
                    },
                ],
            )

            answer_eval_sets = materialize_answer_eval_sets(
                dev_seed_path=dev_seed,
                full_seed_path=full_seed,
                chunks=chunks,
                dev_output_path=dev_output,
                full_output_path=full_output,
                report_output_path=report_output,
                requested_splits=("full",),
                benchmark_profile="historical-full-raw",
            )

            self.assertEqual(
                [row["question_id"] for row in answer_eval_sets["full"]],
                ["full_conflict_1", "full_conflict_2", "full_keep_1"],
            )
            report = json.loads(report_output.read_text(encoding="utf-8"))
            full_report = report["splits"]["full"]
            self.assertEqual(full_report["resolved_row_count"], 3)
            self.assertEqual(full_report["materialized_row_count"], 3)
            self.assertEqual(full_report["benchmark_profile"], "historical-full-raw")
            self.assertEqual(full_report["profile_filter"]["dropped_row_count"], 0)

    def test_historical_full_alias_is_reported_as_raw_profile(self) -> None:
        chunks = self._sample_chunks()
        doc_key = build_doc_manifest(chunks)[0]["doc_key"]
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dev_seed = tmp_path / "answer_eval_seed_dev.jsonl"
            full_seed = tmp_path / "answer_eval_seed_full.jsonl"
            dev_output = tmp_path / "answer_eval_dev.jsonl"
            full_output = tmp_path / "answer_eval_full.jsonl"
            report_output = tmp_path / "answer_eval_manifest_report.json"
            self._write_seed_rows(
                dev_seed,
                [
                    {
                        "question_id": "dev_q1",
                        "query": "Which report discusses foundry?",
                        "question_type": "fact",
                        "intent": "report_lookup",
                        "industry": "semiconductor",
                        "target_doc_keys": [doc_key],
                        "gold_answer": "Foundry Weekly",
                        "gold_chunk_ids": ["doc-a-c1"],
                    }
                ],
            )
            self._write_seed_rows(
                full_seed,
                [
                    {
                        "question_id": "full_q1",
                        "query": "Which report discusses foundry?",
                        "question_type": "fact",
                        "intent": "report_lookup",
                        "industry": "semiconductor",
                        "target_doc_keys": [doc_key],
                        "gold_answer": "Foundry Weekly",
                        "gold_chunk_ids": ["doc-a-c1"],
                    }
                ],
            )

            answer_eval_sets = materialize_answer_eval_sets(
                dev_seed_path=dev_seed,
                full_seed_path=full_seed,
                chunks=chunks,
                dev_output_path=dev_output,
                full_output_path=full_output,
                report_output_path=report_output,
                requested_splits=("full",),
                benchmark_profile="historical-full",
            )

            self.assertEqual([row["question_id"] for row in answer_eval_sets["full"]], ["full_q1"])
            report = json.loads(report_output.read_text(encoding="utf-8"))
            full_report = report["splits"]["full"]
            self.assertEqual(full_report["benchmark_profile"], "historical-full-raw")
            self.assertEqual(full_report["profile_filter"]["benchmark_profile"], "historical-full-raw")
            self.assertEqual(full_report["profile_filter"]["dropped_conflict_group_count"], 0)
            self.assertEqual(full_report["profile_filter"]["dropped_question_ids"], [])

    def test_failed_result_rows_are_counted_in_summary(self) -> None:
        chunks = self._sample_chunks()
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            chunks_path = tmp_path / "chunks.jsonl"
            chunks_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in chunks) + "\n", encoding="utf-8")
            results_output = tmp_path / "answer_eval_results.jsonl"
            summary_output = tmp_path / "answer_eval_summary.json"
            markdown_output = tmp_path / "answer_eval_summary.md"
            latency_output = tmp_path / "latency_summary.json"
            answer_badcase_output = tmp_path / "answer_badcases.jsonl"
            abstain_badcase_output = tmp_path / "abstain_badcases.jsonl"

            eval_rows = [
                {
                    "question_id": "dev_q1",
                    "query": "哪份报告讨论了晶圆代工？",
                    "question_type": "fact",
                    "intent": "report_lookup",
                    "industry": "semiconductor",
                    "gold_answer": "晶圆代工周报",
                    "gold_doc_ids": ["doc-a"],
                    "gold_page_nums": [1, 2, 3],
                    "gold_chunk_ids": ["doc-a-c1"],
                    "must_abstain": False,
                    "target_doc_keys": ["晶圆代工周报"],
                    "target_titles": ["晶圆代工周报"],
                }
            ]
            result_rows = [
                {
                    "question_id": "dev_q1",
                    "failed": True,
                    "error_type": "RuntimeError",
                    "error_message": "boom",
                    "abstained": False,
                    "abstain_reason": None,
                    "abstain_gate": "error",
                    "final_answer": "",
                    "support_validation": {"supported": False},
                    "citations": [],
                    "retrieval_scores": {},
                    "generation_latency_ms": 0.0,
                    "timings": {},
                }
            ]

            summary = evaluate_answer_results(
                eval_rows=eval_rows,
                result_rows=result_rows,
                chunks_path=chunks_path,
                results_output_path=results_output,
                summary_output_path=summary_output,
                markdown_output_path=markdown_output,
                latency_output_path=latency_output,
                answer_badcase_output_path=answer_badcase_output,
                abstain_badcase_output_path=abstain_badcase_output,
            )

            self.assertEqual(summary["Failed Query Count"], 1)
            answer_badcases = [json.loads(line) for line in answer_badcase_output.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertTrue(answer_badcases[0]["failed"])
            self.assertEqual(answer_badcases[0]["error_type"], "RuntimeError")

    def test_answer_metrics_are_decoupled_from_citation_span(self) -> None:
        chunks = self._sample_chunks()
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            chunks_path = tmp_path / "chunks.jsonl"
            chunks_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in chunks) + "\n", encoding="utf-8")
            results_output = tmp_path / "answer_eval_results.jsonl"
            summary_output = tmp_path / "answer_eval_summary.json"
            markdown_output = tmp_path / "answer_eval_summary.md"
            latency_output = tmp_path / "latency_summary.json"
            answer_badcase_output = tmp_path / "answer_badcases.jsonl"
            abstain_badcase_output = tmp_path / "abstain_badcases.jsonl"

            eval_rows = [
                {
                    "question_id": "dev_q1",
                    "query": "哪份报告讨论了晶圆代工？",
                    "question_type": "fact",
                    "intent": "report_lookup",
                    "industry": "semiconductor",
                    "gold_answer": "晶圆代工周报",
                    "gold_doc_ids": ["doc-a"],
                    "gold_page_nums": [1],
                    "gold_chunk_ids": ["doc-a-c1"],
                    "must_abstain": False,
                    "target_doc_keys": ["晶圆代工周报"],
                    "target_titles": ["晶圆代工周报"],
                }
            ]
            result_rows = [
                {
                    "question_id": "dev_q1",
                    "failed": False,
                    "abstained": False,
                    "abstain_reason": None,
                    "abstain_gate": "none",
                    "final_answer": "晶圆代工周报",
                    "evidence_summary": "《晶圆代工周报》P3 提到晶圆代工行业景气上行。",
                    "fact_subtype": "report_lookup",
                    "answer_source": "llm",
                    "fallback_used": False,
                    "fallback_reason": None,
                    "support_validation": {"supported": True},
                    "citations": [
                        {
                            "evidence_id": "E1",
                            "chunk_id": "doc-a-c2",
                            "doc_id": "doc-a",
                            "file_name": "半导体_2026-04-01_REF001_晶圆代工周报.pdf",
                            "page_start": 3,
                            "page_end": 3,
                            "snippet": "附录页提到晶圆代工行业景气上行。",
                        }
                    ],
                    "retrieval_scores": {},
                    "generation_latency_ms": 0.0,
                    "timings": {},
                }
            ]

            summary = evaluate_answer_results(
                eval_rows=eval_rows,
                result_rows=result_rows,
                chunks_path=chunks_path,
                results_output_path=results_output,
                summary_output_path=summary_output,
                markdown_output_path=markdown_output,
                latency_output_path=latency_output,
                answer_badcase_output_path=answer_badcase_output,
                abstain_badcase_output_path=abstain_badcase_output,
            )

            self.assertEqual(summary["Answer Semantic Hit Rate"], 1.0)
            self.assertEqual(summary["Citation Doc Hit Rate"], 1.0)
            self.assertEqual(summary["Citation Span Hit Rate"], 0.0)
            answer_badcases = [json.loads(line) for line in answer_badcase_output.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertTrue(answer_badcases[0]["answer_semantic_hit"])
            self.assertTrue(answer_badcases[0]["citation_doc_hit"])
            self.assertFalse(answer_badcases[0]["citation_span_hit"])

    def test_report_lookup_support_hit_is_true_when_answer_and_doc_hit_are_true(self) -> None:
        chunks = self._sample_chunks()
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            chunks_path = tmp_path / "chunks.jsonl"
            chunks_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in chunks) + "\n", encoding="utf-8")
            results_output = tmp_path / "answer_eval_results.jsonl"
            summary_output = tmp_path / "answer_eval_summary.json"
            markdown_output = tmp_path / "answer_eval_summary.md"
            latency_output = tmp_path / "latency_summary.json"
            answer_badcase_output = tmp_path / "answer_badcases.jsonl"
            abstain_badcase_output = tmp_path / "abstain_badcases.jsonl"

            eval_rows = [
                {
                    "question_id": "dev_q1",
                    "query": "哪份报告讨论了晶圆代工？",
                    "question_type": "fact",
                    "intent": "report_lookup",
                    "industry": "semiconductor",
                    "gold_answer": "晶圆代工周报",
                    "gold_doc_ids": ["doc-a"],
                    "gold_page_nums": [1],
                    "gold_chunk_ids": ["doc-a-c1"],
                    "must_abstain": False,
                    "target_doc_keys": ["晶圆代工周报"],
                    "target_titles": ["晶圆代工周报"],
                }
            ]
            result_rows = [
                {
                    "question_id": "dev_q1",
                    "failed": False,
                    "abstained": False,
                    "abstain_reason": None,
                    "abstain_gate": "none",
                    "final_answer": "晶圆代工周报",
                    "evidence_summary": "《晶圆代工周报》P3 提到晶圆代工行业景气上行。",
                    "fact_subtype": "report_lookup",
                    "answer_source": "llm",
                    "fallback_used": False,
                    "fallback_reason": None,
                    "support_validation": {"supported": False, "domain_mismatch": False},
                    "citations": [
                        {
                            "evidence_id": "E1",
                            "chunk_id": "doc-a-c2",
                            "doc_id": "doc-a",
                            "file_name": "半导体_2026-04-01_REF001_晶圆代工周报.pdf",
                            "page_start": 3,
                            "page_end": 3,
                            "snippet": "附录页提到晶圆代工行业景气上行。",
                        }
                    ],
                    "retrieval_scores": {},
                    "generation_latency_ms": 0.0,
                    "timings": {},
                }
            ]

            summary = evaluate_answer_results(
                eval_rows=eval_rows,
                result_rows=result_rows,
                chunks_path=chunks_path,
                results_output_path=results_output,
                summary_output_path=summary_output,
                markdown_output_path=markdown_output,
                latency_output_path=latency_output,
                answer_badcase_output_path=answer_badcase_output,
                abstain_badcase_output_path=abstain_badcase_output,
            )

            self.assertEqual(summary["Support Hit Rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
