import unittest

from src.generation.answerer import _prepare_evidence


class AnswererRoutingTests(unittest.TestCase):
    def _row(
        self,
        *,
        chunk_id: str,
        doc_id: str,
        file_name: str,
        text: str,
        score: float,
        rerank_score: float,
        page_start: int = 1,
        section_title: str = "",
        chunk_type: str = "text",
        element_type: str = "paragraph",
    ) -> dict[str, object]:
        return {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "file_name": file_name,
            "page_start": page_start,
            "page_end": page_start,
            "text": text,
            "child_text": text,
            "support_span": text,
            "section_title": section_title,
            "section_path": section_title,
            "chunk_type": chunk_type,
            "element_type": element_type,
            "score": score,
            "rerank_score": rerank_score,
        }

    def test_comparison_picks_two_distinct_docs(self) -> None:
        retrieval_result = {
            "rerank_rows": [
                self._row(
                    chunk_id="doc-a-c1",
                    doc_id="doc-a",
                    file_name="消费_2026-04-01_REF001_报告A.pdf",
                    text="报告A强调家居宠物新品。",
                    score=1.0,
                    rerank_score=5.0,
                    section_title="核心观点",
                ),
                self._row(
                    chunk_id="doc-a-c2",
                    doc_id="doc-a",
                    file_name="消费_2026-04-01_REF001_报告A.pdf",
                    text="报告A继续讨论家居宠物新品标准。",
                    score=0.9,
                    rerank_score=4.8,
                    page_start=2,
                    section_title="正文",
                ),
                self._row(
                    chunk_id="doc-b-c1",
                    doc_id="doc-b",
                    file_name="消费_2026-04-01_REF002_报告B.pdf",
                    text="报告B强调新银发消费活力。",
                    score=0.8,
                    rerank_score=4.7,
                    section_title="核心观点",
                ),
            ],
            "hybrid_rows": [],
        }
        selected_evidence, _doc_candidates, selected_doc_ids, forced_reason, _guard = _prepare_evidence(
            query="《报告A》与《报告B》分别关注哪些重点？",
            question_type="comparison",
            answer_mode="comparison",
            fact_subtype="semantic_fact",
            query_domain_bucket="consumer",
            query_domain_buckets=["consumer"],
            retrieval_result=retrieval_result,
        )
        self.assertIsNone(forced_reason)
        self.assertEqual(selected_doc_ids, ["doc-a", "doc-b"])
        self.assertEqual(len({row["doc_id"] for row in selected_evidence}), 2)

    def test_inductive_backfills_third_doc_when_guard_triggers(self) -> None:
        retrieval_result = {
            "rerank_rows": [
                self._row(
                    chunk_id="doc-a-c1",
                    doc_id="doc-a",
                    file_name="半导体_2026-04-01_REF001_报告A.pdf",
                    text="报告A强调AI算力与终端创新。",
                    score=1.0,
                    rerank_score=5.0,
                    section_title="核心观点",
                ),
                self._row(
                    chunk_id="doc-b-c1",
                    doc_id="doc-b",
                    file_name="半导体_2026-04-01_REF002_报告B.pdf",
                    text="报告B强调AI算力与存储景气。",
                    score=0.9,
                    rerank_score=4.9,
                    section_title="核心观点",
                ),
                self._row(
                    chunk_id="doc-c-c1",
                    doc_id="doc-c",
                    file_name="消费_2026-04-01_REF003_报告C.pdf",
                    text="报告C强调消费电子端的新产品渗透。",
                    score=0.8,
                    rerank_score=4.8,
                    section_title="核心观点",
                ),
            ],
            "hybrid_rows": [],
        }
        selected_evidence, _doc_candidates, selected_doc_ids, forced_reason, doc_guard_triggered = _prepare_evidence(
            query="近期半导体报告共同强调了哪些主题？",
            question_type="inductive",
            answer_mode="inductive",
            fact_subtype="semantic_fact",
            query_domain_bucket="semiconductor",
            query_domain_buckets=["semiconductor"],
            retrieval_result=retrieval_result,
        )
        self.assertIsNone(forced_reason)
        self.assertTrue(doc_guard_triggered)
        self.assertEqual(selected_doc_ids, ["doc-a", "doc-b"])
        self.assertEqual(len({row["doc_id"] for row in selected_evidence}), 2)

    def test_comparison_prefers_query_named_reports(self) -> None:
        retrieval_result = {
            "rerank_rows": [
                self._row(
                    chunk_id="doc-a-c1",
                    doc_id="doc-a",
                    file_name="SourceA_2026-04-01_REF001_Alpha Focus.pdf",
                    text="Alpha Focus emphasizes hog price weakness.",
                    score=1.0,
                    rerank_score=5.0,
                    section_title="overview",
                ),
                self._row(
                    chunk_id="doc-c-c1",
                    doc_id="doc-c",
                    file_name="SourceA_2026-04-08_REF002_Another Hog Weekly.pdf",
                    text="Another Hog Weekly emphasizes downstream pressure.",
                    score=0.95,
                    rerank_score=4.95,
                    section_title="overview",
                ),
                self._row(
                    chunk_id="doc-b-c1",
                    doc_id="doc-b",
                    file_name="SourceB_2026-04-01_REF003_Beta Seed.pdf",
                    text="Beta Seed emphasizes seed revitalization.",
                    score=0.78,
                    rerank_score=4.2,
                    section_title="overview",
                ),
            ],
            "hybrid_rows": [],
        }
        selected_evidence, _doc_candidates, selected_doc_ids, forced_reason, _guard = _prepare_evidence(
            query="\u300aAlpha Focus\u300b vs \u300aBeta Seed\u300b",
            question_type="comparison",
            answer_mode="comparison",
            fact_subtype="semantic_fact",
            query_domain_bucket="",
            query_domain_buckets=[],
            retrieval_result=retrieval_result,
        )
        self.assertIsNone(forced_reason)
        self.assertEqual(selected_doc_ids, ["doc-a", "doc-b"])
        self.assertEqual(len({row["doc_id"] for row in selected_evidence}), 2)

    def test_inductive_prefers_mixed_sources_for_generic_query(self) -> None:
        retrieval_result = {
            "rerank_rows": [
                self._row(
                    chunk_id="doc-a-c1",
                    doc_id="doc-a",
                    file_name="SourceA_2026-04-01_REF001_Hog Weekly A.pdf",
                    text="Hog Weekly A emphasizes weak prices and supply pressure.",
                    score=1.0,
                    rerank_score=5.0,
                    section_title="overview",
                ),
                self._row(
                    chunk_id="doc-b-c1",
                    doc_id="doc-b",
                    file_name="SourceA_2026-04-08_REF002_Hog Weekly B.pdf",
                    text="Hog Weekly B emphasizes weak prices and destocking.",
                    score=0.97,
                    rerank_score=4.9,
                    section_title="overview",
                ),
                self._row(
                    chunk_id="doc-c-c1",
                    doc_id="doc-c",
                    file_name="SourceA_2026-04-15_REF003_Hog Weekly C.pdf",
                    text="Hog Weekly C emphasizes low prices and waiting for inflection.",
                    score=0.96,
                    rerank_score=4.85,
                    section_title="overview",
                ),
                self._row(
                    chunk_id="doc-d-c1",
                    doc_id="doc-d",
                    file_name="SourceB_2026-04-01_REF004_Seed Revitalization.pdf",
                    text="Seed Revitalization emphasizes breeding innovation.",
                    score=0.82,
                    rerank_score=4.1,
                    section_title="overview",
                ),
                self._row(
                    chunk_id="doc-e-c1",
                    doc_id="doc-e",
                    file_name="SourceC_2026-04-01_REF005_Vegetable Prices.pdf",
                    text="Vegetable Prices emphasizes weather-driven price rebound.",
                    score=0.8,
                    rerank_score=4.0,
                    section_title="overview",
                ),
            ],
            "hybrid_rows": [],
        }
        selected_evidence, _doc_candidates, selected_doc_ids, forced_reason, _guard = _prepare_evidence(
            query="recent agriculture common themes",
            question_type="inductive",
            answer_mode="inductive",
            fact_subtype="semantic_fact",
            query_domain_bucket="",
            query_domain_buckets=[],
            retrieval_result=retrieval_result,
        )
        self.assertIsNone(forced_reason)
        self.assertEqual(len(selected_doc_ids), 3)
        selected_sources = {row["file_name"].split("_", 1)[0] for row in selected_evidence}
        self.assertGreaterEqual(len(selected_sources), 2)

    def test_broad_inductive_prefers_multiple_topic_clusters(self) -> None:
        retrieval_result = {
            "rerank_rows": [
                self._row(
                    chunk_id="doc-a-c1",
                    doc_id="doc-a",
                    file_name="agriculture_2026-04-01_REF001_Hog Destocking Weekly.pdf",
                    text="Hog price weakness and destocking remain the central theme.",
                    score=1.0,
                    rerank_score=5.0,
                    section_title="overview",
                ),
                self._row(
                    chunk_id="doc-b-c1",
                    doc_id="doc-b",
                    file_name="agriculture_2026-04-08_REF002_Pig Cycle Weekly.pdf",
                    text="Pig prices hit a new low and the market waits for faster destocking.",
                    score=0.98,
                    rerank_score=4.95,
                    section_title="overview",
                ),
                self._row(
                    chunk_id="doc-c-c1",
                    doc_id="doc-c",
                    file_name="agriculture_2026-04-15_REF003_Hog Supply Weekly.pdf",
                    text="Hog supply pressure is accelerating and piglet demand is weakening.",
                    score=0.97,
                    rerank_score=4.9,
                    section_title="overview",
                ),
                self._row(
                    chunk_id="doc-d-c1",
                    doc_id="doc-d",
                    file_name="agriculture_2026-04-22_REF004_Seed Revitalization Weekly.pdf",
                    text="The report emphasizes seed revitalization and agricultural technology.",
                    score=0.82,
                    rerank_score=4.1,
                    section_title="overview",
                ),
                self._row(
                    chunk_id="doc-e-c1",
                    doc_id="doc-e",
                    file_name="agriculture_2026-04-29_REF005_Oil Driven Commodity Weekly.pdf",
                    text="Oil strength is lifting agricultural commodity price expectations.",
                    score=0.8,
                    rerank_score=4.0,
                    section_title="overview",
                ),
            ],
            "hybrid_rows": [],
        }
        selected_evidence, _doc_candidates, selected_doc_ids, forced_reason, _guard = _prepare_evidence(
            query="recent agriculture common themes",
            question_type="inductive",
            answer_mode="inductive",
            fact_subtype="semantic_fact",
            query_domain_bucket="agriculture",
            query_domain_buckets=["agriculture"],
            retrieval_result=retrieval_result,
        )
        self.assertIsNone(forced_reason)
        self.assertEqual(len(selected_doc_ids), 3)
        self.assertTrue(any(doc_id in {"doc-d", "doc-e"} for doc_id in selected_doc_ids))

    def test_non_broad_inductive_query_stays_in_domain_bucket(self) -> None:
        retrieval_result = {
            "rerank_rows": [
                self._row(
                    chunk_id="doc-a-c1",
                    doc_id="doc-a",
                    file_name="semicon_2026-04-01_REF001_ReportA.pdf",
                    text="ReportA emphasizes AI compute expansion and HBM demand.",
                    score=1.0,
                    rerank_score=5.0,
                    section_title="core",
                ),
                self._row(
                    chunk_id="doc-b-c1",
                    doc_id="doc-b",
                    file_name="semicon_2026-04-08_REF002_ReportB.pdf",
                    text="ReportB emphasizes AI compute links and advanced packaging.",
                    score=0.96,
                    rerank_score=4.85,
                    section_title="core",
                ),
                self._row(
                    chunk_id="doc-c-c1",
                    doc_id="doc-c",
                    file_name="consumer_2026-04-01_REF003_ReportC.pdf",
                    text="ReportC emphasizes consumer brand trends.",
                    score=0.9,
                    rerank_score=4.8,
                    section_title="core",
                ),
            ],
            "hybrid_rows": [],
        }
        selected_evidence, _doc_candidates, selected_doc_ids, forced_reason, doc_guard_triggered = _prepare_evidence(
            query="recent semiconductor reports common AI compute themes",
            question_type="inductive",
            answer_mode="inductive",
            fact_subtype="semantic_fact",
            query_domain_bucket="semiconductor",
            query_domain_buckets=["semiconductor"],
            retrieval_result=retrieval_result,
        )
        self.assertIsNone(forced_reason)
        self.assertTrue(doc_guard_triggered)
        self.assertEqual(selected_doc_ids, ["doc-a", "doc-b"])
        self.assertEqual({row["doc_id"] for row in selected_evidence}, {"doc-a", "doc-b"})

    def test_report_lookup_rejects_uncovered_cross_domain_query(self) -> None:
        retrieval_result = {
            "rerank_rows": [
                self._row(
                    chunk_id="doc-a-c1",
                    doc_id="doc-a",
                    file_name="农林牧渔_2026-04-01_REF001_Hog Weekly.pdf",
                    text="Hog Weekly emphasizes hog price weakness and destocking.",
                    score=1.0,
                    rerank_score=5.0,
                    section_title="core",
                ),
            ],
            "hybrid_rows": [],
        }
        selected_evidence, _doc_candidates, selected_doc_ids, forced_reason, _guard = _prepare_evidence(
            query="Which agriculture weekly mainly analyzes GPU accelerator pricing?",
            question_type="fact",
            answer_mode="report_lookup",
            fact_subtype="report_lookup",
            query_domain_bucket="agriculture",
            query_domain_buckets=["agriculture", "semiconductor"],
            retrieval_result=retrieval_result,
        )
        self.assertEqual(selected_evidence, [])
        self.assertEqual(selected_doc_ids, ["doc-a"])
        self.assertEqual(forced_reason, "domain_mismatch")

    def test_comparison_rejects_when_query_mentions_uncovered_domain(self) -> None:
        retrieval_result = {
            "rerank_rows": [
                self._row(
                    chunk_id="doc-a-c1",
                    doc_id="doc-a",
                    file_name="农林牧渔_2026-04-01_REF001_Hog Weekly.pdf",
                    text="Hog Weekly emphasizes hog price weakness and destocking.",
                    score=1.0,
                    rerank_score=5.0,
                    section_title="core",
                ),
                self._row(
                    chunk_id="doc-b-c1",
                    doc_id="doc-b",
                    file_name="农林牧渔_2026-04-08_REF002_Seed Weekly.pdf",
                    text="Seed Weekly emphasizes breeding innovation and seed policy.",
                    score=0.95,
                    rerank_score=4.9,
                    section_title="core",
                ),
            ],
            "hybrid_rows": [],
        }
        selected_evidence, _doc_candidates, selected_doc_ids, forced_reason, _guard = _prepare_evidence(
            query="Which wafer foundry opportunities are jointly highlighted by WHO risk reports and hog breeding weeklies?",
            question_type="comparison",
            answer_mode="comparison",
            fact_subtype="semantic_fact",
            query_domain_bucket="agriculture",
            query_domain_buckets=["healthcare", "agriculture", "semiconductor"],
            retrieval_result=retrieval_result,
        )
        self.assertEqual(selected_evidence, [])
        self.assertEqual(selected_doc_ids, ["doc-a", "doc-b"])
        self.assertEqual(forced_reason, "domain_mismatch")



if __name__ == "__main__":
    unittest.main()
