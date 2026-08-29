"""Tests for the dependency-free trace observability renderer."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.observability import (
    build_observability_view,
    load_trace_rows,
    render_observability_html,
    select_trace_row,
    write_observability_demo,
)


def trace_row(question_id: str = "q1") -> dict:
    return {
        "question_id": question_id,
        "query": "What is the reported margin? <script>alert(1)</script>",
        "question_type": "fact",
        "industry": "semiconductor",
        "final_answer": "The margin is 20%.",
        "termination_reason": "completed",
        "step_count": 7,
        "retrieval_count": 2,
        "tool_call_count": 1,
        "total_tokens": 123,
        "end_to_end_latency_ms": 45.6,
        "claims": [
            {
                "claim_id": "C1",
                "text": "The margin is 20%.",
                "claim_type": "EXTRACTED",
                "evidence_ids": ["E1"],
                "verification": {"status": "ENTAILED", "method": "deterministic"},
            }
        ],
        "citations": [
            {"evidence_id": "E1", "doc_id": "report.pdf", "page": 18, "text": "Margin 20%"}
        ],
        "calculations": {"calc-1": {"status": "SUCCESS", "result": {"formatted": "20%"}}},
        "trajectory_events": [{"node": "retrieve", "status": "SUCCESS", "latency_ms": 3.2}],
    }


class ObservabilityTests(unittest.TestCase):
    def test_load_and_select_trace_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text(json.dumps(trace_row()) + "\n" + json.dumps(trace_row("q2")) + "\n", encoding="utf-8")
            rows = load_trace_rows(path)
            self.assertEqual(select_trace_row(rows, question_id="q2")["question_id"], "q2")
            self.assertEqual(select_trace_row(rows)["question_id"], "q1")
            with self.assertRaises(KeyError):
                select_trace_row(rows, question_id="missing")

    def test_view_normalizes_claims_evidence_and_metrics(self) -> None:
        view = build_observability_view(trace_row())
        self.assertEqual(view["claims"][0]["status"], "ENTAILED")
        self.assertEqual(view["evidence"][0]["doc_id"], "report.pdf")
        self.assertEqual(view["evidence"][0]["page"], 18)
        self.assertEqual(view["calculations"]["calc-1"]["status"], "SUCCESS")
        self.assertEqual(view["metrics"]["total_tokens"], 123)
        self.assertEqual(view["steps"][0]["node"], "retrieve")

    def test_render_escapes_user_text_and_includes_sections(self) -> None:
        html = render_observability_html(build_observability_view(trace_row()))
        self.assertIn("Agent trace q1", html)
        self.assertIn("Claims", html)
        self.assertIn("Evidence and pages", html)
        self.assertIn("Calculation trace", html)
        self.assertIn("Agent steps", html)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", html)
        self.assertNotIn("<script>alert(1)</script>", html)

    def test_write_demo_creates_parent_and_html(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace_path = root / "trace.jsonl"
            output_path = root / "nested" / "demo.html"
            trace_path.write_text(json.dumps(trace_row()) + "\n", encoding="utf-8")
            self.assertEqual(write_observability_demo(trace_path, output_path), output_path)
            self.assertTrue(output_path.is_file())
            self.assertIn("The margin is 20%.", output_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
