"""Agent eval tests (checklist v3.0 §P3 quality metrics, comparison, gate)."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.evaluation.agent_eval import (
    build_baseline_agentic_comparison,
    evaluate_agent_quality,
    evaluate_p3_gate,
    materialize_agent_eval_set,
)
from src.evaluation.benchmark_assets import build_agent_seed_draft, build_doc_manifest, write_agent_seed_draft
from tests.test_agent_seed import INDUSTRIES, INDUSTRY_NAMES, REPO_ROOT

SCRATCH_DIR = REPO_ROOT / "outputs" / "test_scratch_agent_eval"


def make_chunk_lookup() -> dict:
    return {
        "c1": {
            "chunk_id": "c1",
            "doc_id": "d1",
            "text": "晶圆代工价格同比上涨20%。",
            "page_start": 1,
            "page_end": 1,
            "section_title": "行业观点",
            "section_path": "行业观点",
        }
    }


def make_answerable_eval_row() -> dict:
    return {
        "question_id": "q1",
        "query": "晶圆代工价格同比涨幅是多少？",
        "question_type": "fact",
        "intent": "numeric_fact",
        "industry": "semiconductor",
        "category": "agent_numeric_missing",
        "must_recover": True,
        "expected_first_failure": "numeric_missing",
        "gold_answer": "晶圆代工价格同比上涨20%。",
        "gold_doc_ids": ["d1"],
        "gold_page_nums": [1],
        "gold_chunk_ids": ["c1"],
        "must_abstain": False,
        "target_doc_keys": [],
        "target_titles": [],
        "target_files": [],
    }


def make_abstain_eval_row() -> dict:
    return {
        "question_id": "q2",
        "query": "半导体行业2027年度资本开支总额是多少？",
        "question_type": "fact",
        "intent": "numeric_fact",
        "industry": "semiconductor",
        "category": "agent_abstain",
        "must_recover": False,
        "gold_answer": "",
        "gold_doc_ids": [],
        "gold_page_nums": [],
        "gold_chunk_ids": [],
        "must_abstain": True,
        "target_doc_keys": [],
        "target_titles": [],
        "target_files": [],
    }


def make_supported_trace_row(*, question_id: str = "q1", rewrite_count: int = 1) -> dict:
    return {
        "question_id": question_id,
        "query": "晶圆代工价格同比涨幅是多少？",
        "question_type": "fact",
        "intent": "numeric_fact",
        "industry": "semiconductor",
        "category": "agent_numeric_missing",
        "rewrite_count": rewrite_count,
        "final_answer": "晶圆代工价格同比上涨20%。",
        "evidence_summary": "",
        "fact_subtype": "value_fact",
        "abstained": False,
        "abstain_reason": None,
        "citations": [
            {
                "doc_id": "d1",
                "chunk_id": "c1",
                "page_start": 1,
                "page_end": 1,
                "section_title": "",
                "snippet": "晶圆代工价格同比上涨20%。",
            }
        ],
        "used_evidence_ids": [],
        "support_validation": {"supported": True},
        "matched_numeric_tokens": ["20%"],
        "failed": False,
    }


def make_abstained_trace_row(*, question_id: str = "q2") -> dict:
    row = make_supported_trace_row(question_id=question_id, rewrite_count=0)
    row.update(
        {
            "category": "agent_abstain",
            "final_answer": "信息不足，暂时无法给出可靠答案。",
            "fact_subtype": "semantic_fact",
            "abstained": True,
            "abstain_reason": "no_evidence",
            "citations": [],
            "support_validation": {"supported": False},
        }
    )
    return row


class AgentQualityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.eval_rows = [make_answerable_eval_row(), make_abstain_eval_row()]
        self.chunk_lookup = make_chunk_lookup()

    def test_quality_summary_all_hits(self) -> None:
        trace_rows = [make_supported_trace_row(), make_abstained_trace_row()]
        summary = evaluate_agent_quality(self.eval_rows, trace_rows, self.chunk_lookup)
        self.assertEqual(summary["query_count"], 2)
        self.assertEqual(summary["answerable_query_count"], 1)
        self.assertEqual(summary["must_abstain_query_count"], 1)
        self.assertEqual(summary["Answer Accuracy"], 1.0)
        self.assertEqual(summary["Citation Accuracy"], 1.0)
        self.assertEqual(summary["Citation Span Accuracy"], 1.0)
        self.assertEqual(summary["Abstain Accuracy"], 1.0)
        self.assertEqual(summary["Abstain Precision"], 1.0)
        self.assertEqual(summary["Recovery Success Rate"], 1.0)
        self.assertEqual(summary["Recovery Trigger Rate"], 1.0)
        self.assertEqual(summary["Failed Query Count"], 0)

    def test_recovery_failure_when_answerable_row_abstains(self) -> None:
        trace_rows = [make_abstained_trace_row(question_id="q1"), make_abstained_trace_row()]
        summary = evaluate_agent_quality(self.eval_rows, trace_rows, self.chunk_lookup)
        self.assertEqual(summary["Answer Accuracy"], 0.0)
        self.assertEqual(summary["Recovery Success Rate"], 0.0)
        # q1 abstained on an answerable question -> abstain precision drops
        self.assertEqual(summary["Abstain Precision"], 0.5)

    def test_missing_result_rows_count_as_failed(self) -> None:
        trace_rows = [make_supported_trace_row()]  # q2 result missing
        summary = evaluate_agent_quality(self.eval_rows, trace_rows, self.chunk_lookup)
        self.assertEqual(summary["Failed Query Count"], 1)
        self.assertEqual(summary["Abstain Accuracy"], 0.0)

    def test_no_recovery_rows_yields_none_recovery_metrics(self) -> None:
        eval_rows = [make_abstain_eval_row()]
        trace_rows = [make_abstained_trace_row()]
        summary = evaluate_agent_quality(eval_rows, trace_rows, self.chunk_lookup)
        self.assertIsNone(summary["Recovery Success Rate"])
        self.assertIsNone(summary["Recovery Trigger Rate"])


class P3GateTests(unittest.TestCase):
    def test_pass_via_quality_gain(self) -> None:
        verdict = evaluate_p3_gate(
            baseline={"Answer Accuracy": 0.5, "Citation Accuracy": 0.5, "Abstain Accuracy": 0.4, "Recovery Success Rate": 0.0},
            agentic={"Answer Accuracy": 0.6, "Citation Accuracy": 0.5, "Abstain Accuracy": 0.4, "Recovery Success Rate": 0.0},
        )
        self.assertTrue(verdict["gate_passed"])
        self.assertTrue(verdict["gain_quality"])
        self.assertFalse(verdict["gain_recovery"])
        self.assertTrue(verdict["no_regression"])

    def test_pass_via_recovery_gain(self) -> None:
        verdict = evaluate_p3_gate(
            baseline={"Answer Accuracy": 0.5, "Citation Accuracy": 0.5, "Abstain Accuracy": 0.4, "Recovery Success Rate": 0.2},
            agentic={"Answer Accuracy": 0.5, "Citation Accuracy": 0.5, "Abstain Accuracy": 0.4, "Recovery Success Rate": 0.4},
        )
        self.assertTrue(verdict["gate_passed"])
        self.assertFalse(verdict["gain_quality"])
        self.assertTrue(verdict["gain_recovery"])

    def test_fail_when_no_gain(self) -> None:
        verdict = evaluate_p3_gate(
            baseline={"Answer Accuracy": 0.5, "Citation Accuracy": 0.5, "Abstain Accuracy": 0.4, "Recovery Success Rate": 0.2},
            agentic={"Answer Accuracy": 0.5, "Citation Accuracy": 0.5, "Abstain Accuracy": 0.4, "Recovery Success Rate": 0.2},
        )
        self.assertFalse(verdict["gate_passed"])
        self.assertFalse(verdict["gain_quality"])
        self.assertFalse(verdict["gain_recovery"])

    def test_fail_when_quality_regresses(self) -> None:
        verdict = evaluate_p3_gate(
            baseline={"Answer Accuracy": 0.5, "Citation Accuracy": 0.5, "Abstain Accuracy": 0.4, "Recovery Success Rate": 0.2},
            agentic={"Answer Accuracy": 0.4, "Citation Accuracy": 0.5, "Abstain Accuracy": 0.4, "Recovery Success Rate": 0.4},
        )
        self.assertFalse(verdict["gate_passed"])
        self.assertTrue(verdict["gain_recovery"])
        self.assertFalse(verdict["no_regression"])

    def test_none_recovery_metrics_treated_as_no_recovery_gain(self) -> None:
        verdict = evaluate_p3_gate(
            baseline={"Answer Accuracy": 0.5, "Citation Accuracy": 0.5, "Abstain Accuracy": None, "Recovery Success Rate": None},
            agentic={"Answer Accuracy": 0.5, "Citation Accuracy": 0.5, "Abstain Accuracy": None, "Recovery Success Rate": None},
        )
        self.assertFalse(verdict["gate_passed"])
        self.assertFalse(verdict["gain_recovery"])


class BaselineComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.eval_rows = [make_answerable_eval_row()]
        self.chunk_lookup = make_chunk_lookup()

    def test_comparison_aligns_rows_and_reports_deltas(self) -> None:
        baseline_rows = [make_supported_trace_row(rewrite_count=0)]
        agentic_rows = [make_supported_trace_row(rewrite_count=1)]
        comparison = build_baseline_agentic_comparison(self.eval_rows, baseline_rows, agentic_rows, self.chunk_lookup)
        self.assertEqual(comparison["question_count"], 1)
        self.assertEqual(len(comparison["per_question"]), 1)
        per_question = comparison["per_question"][0]
        self.assertEqual(per_question["question_id"], "q1")
        self.assertTrue(per_question["baseline"]["answer_semantic_hit"])
        self.assertTrue(per_question["agentic"]["answer_semantic_hit"])
        self.assertEqual(comparison["deltas"]["Answer Accuracy"], 0.0)
        self.assertEqual(comparison["deltas"]["Citation Accuracy"], 0.0)


class AgentEvalSetMaterializationTests(unittest.TestCase):
    @staticmethod
    def _materialization_corpus() -> tuple[list[dict], list[dict]]:
        """Chunks with realistic file names so build_doc_manifest derives industry/doc_key."""
        chunks: list[dict] = []
        for industry in INDUSTRIES:
            for ordinal in (1, 2):
                title = f"{INDUSTRY_NAMES[industry]}行业周报{ordinal}"
                doc_id = f"{industry}-doc{ordinal}"
                file_name = f"{INDUSTRY_NAMES[industry]}_2026-01-0{ordinal}_AP0000{ordinal}_{title}.pdf"
                chunks.append(
                    {
                        "chunk_id": f"{doc_id}-c1",
                        "doc_id": doc_id,
                        "file_name": file_name,
                        "page_start": 1,
                        "page_end": 1,
                        "text": f"{title}显示，{INDUSTRY_NAMES[industry]}景气度上行{ordinal}0%。",
                        "child_text": f"{title}显示，{INDUSTRY_NAMES[industry]}景气度上行{ordinal}0%。",
                        "support_span": "",
                        "section_title": "",
                        "section_path": "",
                        "chunk_type": "text",
                        "element_type": "paragraph",
                    }
                )
        return chunks, build_doc_manifest(chunks)

    @staticmethod
    def _seed_rows_from_manifest(manifest: list[dict]) -> list[dict]:
        by_industry: dict[str, list[dict]] = {}
        for row in manifest:
            by_industry.setdefault(row["industry"], []).append(row)
        rows: list[dict] = []
        for industry in INDUSTRIES:
            docs = by_industry[industry]
            first, second = docs[0], docs[1]
            rows.append(
                {
                    "question_id": f"{industry}_numeric_fact_01",
                    "query": f"{INDUSTRY_NAMES[industry]}行业周报中的关键数据是怎样的？",
                    "question_type": "fact",
                    "intent": "numeric_fact",
                    "industry": industry,
                    "target_doc_keys": [first["doc_key"]],
                }
            )
            for ordinal, pair in ((1, (first, second)), (2, (first, second))):
                rows.append(
                    {
                        "question_id": f"{industry}_comparison_{ordinal:02d}",
                        "query": f"《{pair[0]['short_title']}》与《{pair[1]['short_title']}》分别关注哪些重点？",
                        "question_type": "comparison",
                        "intent": "comparison",
                        "industry": industry,
                        "target_doc_keys": [pair[0]["doc_key"], pair[1]["doc_key"]],
                    }
                )
            rows.append(
                {
                    "question_id": f"{industry}_inductive_01",
                    "query": f"近期{INDUSTRY_NAMES[industry]}行业报告共同强调了哪些主题？",
                    "question_type": "inductive",
                    "intent": "inductive",
                    "industry": industry,
                    "target_doc_keys": [first["doc_key"], second["doc_key"]],
                }
            )
        return rows

    def test_materialize_agent_eval_set_writes_rows_and_report(self) -> None:
        import shutil

        shutil.rmtree(SCRATCH_DIR, ignore_errors=True)
        SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
        try:
            chunks, manifest = self._materialization_corpus()
            seed_rows = self._seed_rows_from_manifest(manifest)
            agent_rows = build_agent_seed_draft(retrieval_seed_rows=seed_rows, chunks=chunks, manifest=manifest)
            seed_path = SCRATCH_DIR / "agent_seed.jsonl"
            write_agent_seed_draft(agent_rows, seed_path)
            output_path = SCRATCH_DIR / "agent_eval_dev.jsonl"
            report_path = SCRATCH_DIR / "agent_eval_manifest_report.json"
            rows = materialize_agent_eval_set(
                seed_path=seed_path,
                chunks=chunks,
                output_path=output_path,
                report_output_path=report_path,
            )
            self.assertEqual(len(rows), 20)
            self.assertTrue(output_path.exists())
            self.assertTrue(report_path.exists())
            for row in rows:
                self.assertIn("category", row)
                self.assertIn("must_recover", row)
        finally:
            shutil.rmtree(SCRATCH_DIR, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
