import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.evaluation.retrieval_eval import materialize_eval_set


class RetrievalEvalMaterializationTests(unittest.TestCase):
    def test_gold_page_nums_cover_full_chunk_range(self) -> None:
        chunks = [
            {
                'chunk_id': 'chunk-1',
                'doc_id': 'doc-1',
                'file_name': '半导体_2026-04-01_REF001_晶圆代工周报.pdf',
                'page_start': 1,
                'page_end': 3,
                'text': '晶圆代工景气上行。',
                'section_title': '行业观点',
                'section_path': '行业观点',
                'chunk_type': 'text',
                'element_type': 'paragraph',
            }
        ]
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            chunks_path = tmp_path / 'chunks.jsonl'
            seed_path = tmp_path / 'retrieval_eval_seed.jsonl'
            output_path = tmp_path / 'retrieval_eval.jsonl'
            report_path = tmp_path / 'retrieval_eval_report.json'
            chunks_path.write_text(json.dumps(chunks[0], ensure_ascii=False) + '\n', encoding='utf-8')
            seed_path.write_text(
                json.dumps(
                    {
                        'question_id': 'q1',
                        'query': '哪份报告讨论了晶圆代工？',
                        'question_type': 'fact',
                        'intent': 'report_lookup',
                        'industry': 'semiconductor',
                        'target_doc_keys': ['晶圆代工周报'],
                        'page_hints': [1],
                        'keyword_hints': ['晶圆代工'],
                    },
                    ensure_ascii=False,
                )
                + '\n',
                encoding='utf-8',
            )

            rows = materialize_eval_set(
                seed_path=seed_path,
                chunks_path=chunks_path,
                output_path=output_path,
                report_output_path=report_path,
            )

            self.assertEqual(rows[0]['gold_page_nums'], [1, 2, 3])


if __name__ == '__main__':
    unittest.main()
