import unittest
from unittest.mock import patch

from src.generation.answerer import FALLBACK_ANSWER, LocalEvidenceAnswerer


class AnswererFallbackTests(unittest.TestCase):
    def test_invalid_model_json_fallback_becomes_abstain(self) -> None:
        answerer = LocalEvidenceAnswerer.__new__(LocalEvidenceAnswerer)
        answerer._generate = lambda prompt: 'not-json'
        evidence_row = {
            "evidence_id": "E1",
            "chunk_id": "chunk-1",
            "doc_id": "doc-1",
            "file_name": "半导体_2026-04-01_REF001_晶圆代工周报.pdf",
            "page_start": 1,
            "page_end": 1,
            "text": "晶圆代工景气上行。",
            "child_text": "晶圆代工景气上行。",
            "support_span": "晶圆代工景气上行。",
            "chunk_type": "text",
            "element_type": "paragraph",
        }
        retrieval_result = {
            "dense_rows": [],
            "bm25_rows": [],
            "hybrid_rows": [],
            "rerank_rows": [],
            "timings": {"retrieval_latency_ms": 0.0, "rerank_latency_ms": 0.0},
        }
        with patch('src.generation.answerer._prepare_evidence', return_value=([evidence_row], [], ['doc-1'], None, False)), patch(
            'src.generation.answerer._fallback_payload',
            return_value={
                'final_answer': FALLBACK_ANSWER,
                'evidence_summary': '',
                'uncertainty_note': '',
                'used_evidence_ids': [],
            },
        ), patch('src.generation.answerer.build_answer_prompt', return_value='prompt'):
            result = LocalEvidenceAnswerer.answer(
                answerer,
                query='测试问题',
                question_type='fact',
                retrieval_result=retrieval_result,
            )

        self.assertTrue(result['abstained'])
        self.assertEqual(result['abstain_reason'], 'invalid_model_json')
        self.assertEqual(result['final_answer'], FALLBACK_ANSWER)
        self.assertEqual(result['citations'], [])


if __name__ == '__main__':
    unittest.main()
