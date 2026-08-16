"""Citation verify tool tests: checklist v3.0 §P4 ('citation verify reuses existing validation')."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.tools.citation_verify import verify_citations


def make_row(
    *,
    chunk_id: str,
    doc_id: str,
    evidence_id: str,
    text: str = "",
) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "evidence_id": evidence_id,
        "file_name": f"{doc_id}.pdf",
        "page_start": 1,
        "page_end": 1,
        "text": text,
        "child_text": text,
        "support_span": text,
        "section_title": "",
        "section_path": "",
        "chunk_type": "text",
        "element_type": "paragraph",
        "score": 3.0,
    }


def make_answer(answer: str) -> dict:
    return {
        "query": "半导体行业景气度如何？",
        "question_type": "fact",
        "final_answer": answer,
        "evidence_summary": answer,
    }


class VerifyCitationsTests(unittest.TestCase):
    def test_all_ids_present_and_supported_is_valid(self) -> None:
        rows = [
            make_row(chunk_id="c1", doc_id="d1", evidence_id="e1", text="半导体行业景气度持续回升。"),
        ]
        answer = make_answer("半导体行业景气度持续回升。")
        result = verify_citations(
            query=answer["query"],
            question_type="fact",
            final_answer=answer["final_answer"],
            evidence_summary=answer["evidence_summary"],
            used_evidence_ids=["e1"],
            evidence_rows=rows,
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["missing_evidence_ids"], [])
        self.assertEqual(result["malformed_citations"], [])
        self.assertEqual(len(result["citations"]), 1)

    def test_missing_id_lists_exactly_it(self) -> None:
        rows = [
            make_row(chunk_id="c1", doc_id="d1", evidence_id="e1", text="半导体行业景气度持续回升。"),
        ]
        result = verify_citations(
            query="半导体行业景气度如何？",
            question_type="fact",
            final_answer="半导体行业景气度持续回升。",
            evidence_summary="半导体行业景气度持续回升。",
            used_evidence_ids=["e1", "e-ghost"],
            evidence_rows=rows,
        )
        self.assertFalse(result["valid"])
        self.assertEqual(result["missing_evidence_ids"], ["e-ghost"])

    def test_number_not_in_evidence_is_unsupported(self) -> None:
        rows = [
            make_row(chunk_id="c1", doc_id="d1", evidence_id="e1", text="半导体行业景气度持续回升。"),
        ]
        result = verify_citations(
            query="晶圆代工价格是多少？",
            question_type="fact",
            final_answer="晶圆代工价格同比上涨25%。",
            evidence_summary="晶圆代工价格同比上涨25%。",
            used_evidence_ids=["e1"],
            evidence_rows=rows,
        )
        self.assertFalse(result["valid"])
        self.assertIs(result["support_validation"]["supported"], False)
        self.assertTrue(result["support_validation"]["missing_numeric_tokens"])

    def test_empty_doc_id_emits_malformed_citation(self) -> None:
        rows = [
            make_row(chunk_id="c1", doc_id="", evidence_id="e1", text="半导体行业景气度持续回升。"),
        ]
        result = verify_citations(
            query="半导体行业景气度如何？",
            question_type="fact",
            final_answer="半导体行业景气度持续回升。",
            evidence_summary="半导体行业景气度持续回升。",
            used_evidence_ids=["e1"],
            evidence_rows=rows,
        )
        self.assertTrue(result["malformed_citations"])
        self.assertFalse(result["valid"])

    def test_empty_rows_and_ids_does_not_raise(self) -> None:
        result = verify_citations(
            query="晶圆代工价格是多少？",
            question_type="fact",
            final_answer="晶圆代工价格同比上涨25%。",
            evidence_summary="晶圆代工价格同比上涨25%。",
            used_evidence_ids=[],
            evidence_rows=[],
        )
        self.assertFalse(result["valid"])
        self.assertEqual(result["missing_evidence_ids"], [])
        self.assertEqual(result["citations"], [])

    def test_source_guard(self) -> None:
        # tests/ -> parents[0]; repo root (financial-report-agent) -> parents[1]
        source_path = Path(__file__).resolve().parents[1] / "src" / "tools" / "citation_verify.py"
        source = source_path.read_text(encoding="utf-8")
        self.assertIn(
            "from src.generation.support_validator import validate_answer_support",
            source,
        )
        for forbidden in ("eval(", "exec(", "subprocess", "httpx", "src.llm"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
