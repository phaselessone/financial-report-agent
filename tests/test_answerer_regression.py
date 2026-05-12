import unittest

from src.generation.answerer import (
    FALLBACK_ANSWER,
    _compose_inductive_answer,
    _shared_logic_comparison_answer,
    _prepare_evidence,
    _repair_answer_payload,
    _select_rows_for_doc,
    _selected_domain_mismatch,
)


class AnswererRegressionTests(unittest.TestCase):
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

    def test_other_is_not_a_hard_domain_mismatch(self) -> None:
        self.assertFalse(_selected_domain_mismatch(["other"], ["agriculture"], answer_mode="report_lookup"))

    def test_liquor_consumer_compatibility_does_not_force_mismatch(self) -> None:
        retrieval_result = {
            "rerank_rows": [
                self._row(
                    chunk_id="doc-a-c1",
                    doc_id="doc-a",
                    file_name="消费_2026-04-01_REF001_Report.pdf",
                    text="食品饮料行业修复趋势确立，板块布局继续优化。",
                    score=1.0,
                    rerank_score=5.0,
                    section_title="核心观点",
                ),
            ],
            "hybrid_rows": [],
        }
        selected_evidence, _doc_candidates, selected_doc_ids, forced_reason, _guard = _prepare_evidence(
            query="报告中关于“章节、路径、提升”的关键数据或变化是怎样的？",
            question_type="fact",
            answer_mode="numeric_fact",
            fact_subtype="semantic_fact",
            query_domain_bucket="liquor",
            query_domain_buckets=["liquor", "consumer"],
            retrieval_result=retrieval_result,
        )
        self.assertIsNone(forced_reason)
        self.assertEqual(selected_doc_ids, ["doc-a"])
        self.assertTrue(selected_evidence)

    def test_exact_domain_is_preferred_over_only_compatible_domain(self) -> None:
        retrieval_result = {
            "rerank_rows": [
                self._row(
                    chunk_id="doc-liquor-c1",
                    doc_id="doc-liquor",
                    file_name="白酒_2026-04-01_REF001_Liquor Weekly.pdf",
                    text="白酒板块修复趋势明朗。",
                    score=1.0,
                    rerank_score=5.0,
                    section_title="核心观点",
                ),
                self._row(
                    chunk_id="doc-consumer-c1",
                    doc_id="doc-consumer",
                    file_name="消费_2026-04-01_REF002_Food Beverage Review.pdf",
                    text="食品饮料行业2025年业绩预告整体承压，静待需求回暖。",
                    score=0.95,
                    rerank_score=4.85,
                    section_title="核心观点",
                ),
            ],
            "hybrid_rows": [],
        }
        selected_evidence, _doc_candidates, selected_doc_ids, forced_reason, doc_guard_triggered = _prepare_evidence(
            query="报告中关于“业绩、食品、饮料”的关键数据或变化是怎样的？",
            question_type="fact",
            answer_mode="numeric_fact",
            fact_subtype="semantic_fact",
            query_domain_bucket="consumer",
            query_domain_buckets=["consumer"],
            retrieval_result=retrieval_result,
        )
        self.assertIsNone(forced_reason)
        self.assertTrue(doc_guard_triggered)
        self.assertEqual(selected_doc_ids, ["doc-consumer"])
        self.assertEqual({row["doc_id"] for row in selected_evidence}, {"doc-consumer"})

    def test_generic_numeric_semantic_prefers_paragraph_anchor(self) -> None:
        doc_candidate = {
            "rows": [
                self._row(
                    chunk_id="doc-a-table",
                    doc_id="doc-a",
                    file_name="半导体_2026-04-01_REF001_Report.pdf",
                    text="表头：现货均价 120，周环比 +3%，月环比 +8%。",
                    score=1.0,
                    rerank_score=5.0,
                    page_start=6,
                    section_title="表格数据",
                    chunk_type="table_like",
                    element_type="table",
                ),
                self._row(
                    chunk_id="doc-a-para",
                    doc_id="doc-a",
                    file_name="半导体_2026-04-01_REF001_Report.pdf",
                    text="报告指出现货均价持续抬升，涨价正在向下游环节传导，行业景气度延续改善。",
                    score=0.8,
                    rerank_score=4.8,
                    page_start=2,
                    section_title="核心观点",
                ),
            ]
        }
        selected_rows = _select_rows_for_doc(
            doc_candidate,
            query="报告中关于“现货、平均、均价”的关键数据或变化是怎样的？",
            answer_mode="numeric_fact",
            fact_subtype="semantic_fact",
            numeric_query=True,
            max_rows=2,
        )
        self.assertEqual(selected_rows[0]["chunk_id"], "doc-a-para")

    def test_comparison_uses_structured_fields_for_final_answer(self) -> None:
        selected_evidence = [
            {
                **self._row(
                    chunk_id="doc-a-c1",
                    doc_id="doc-a",
                    file_name="消费_2026-03-22_REF001_耐用消费产业行业研究：家居宠物新品频发定义行业标准，中烟换帅期待新变革.pdf",
                    text="家居宠物新品频发，正在重塑行业标准。",
                    score=1.0,
                    rerank_score=5.0,
                ),
                "evidence_id": "E1",
            },
            {
                **self._row(
                    chunk_id="doc-b-c1",
                    doc_id="doc-b",
                    file_name="消费_2026-03-25_REF002_银发行业中国消费者洞察：新银发的活力人生.pdf",
                    text="新银发消费者更关注活力生活方式与消费体验。",
                    score=0.9,
                    rerank_score=4.9,
                ),
                "evidence_id": "E2",
            },
        ]
        payload, fallback_reason = _repair_answer_payload(
            query="《耐用消费产业行业研究：家居宠物新品频发定义行业标准，中烟换帅期待新变革》与《银发行业中国消费者洞察：新银发的活力人生》分别关注哪些重点？",
            answer_mode="comparison",
            fact_subtype="semantic_fact",
            payload={
                "doc_focus_map": {
                    "doc-a": "家居宠物新品如何定义行业标准",
                    "doc-b": "新银发消费者的人群洞察与消费活力",
                },
                "difference_dimension": "研究对象",
                "difference_detail": "前者聚焦产业与新品标准，后者聚焦银发消费者画像和消费行为。",
                "shared_points": ["都关注消费升级"],
                "used_evidence_ids": ["E1", "E2"],
            },
            selected_evidence=selected_evidence,
        )
        self.assertIsNone(fallback_reason)
        self.assertIn("研究对象", payload["final_answer"])
        self.assertIn("耐用消费产业行业研究", payload["final_answer"])
        self.assertIn("银发行业中国消费者洞察", payload["final_answer"])

    def test_shared_logic_comparison_handles_same_title_reports(self) -> None:
        selected_evidence = [
            {
                **self._row(
                    chunk_id="doc-a-c1",
                    doc_id="doc-a",
                    file_name="农林牧渔_2026-03-17_REF001_建议关注养殖股产能去化逻辑的回归和演绎.pdf",
                    text="报告强调养殖股产能去化逻辑的回归与演绎。",
                    score=1.0,
                    rerank_score=5.0,
                ),
                "evidence_id": "E1",
            },
            {
                **self._row(
                    chunk_id="doc-b-c1",
                    doc_id="doc-b",
                    file_name="农林牧渔_2026-03-24_REF002_建议关注养殖股产能去化逻辑的回归和演绎.pdf",
                    text="报告继续围绕养殖股产能去化主线展开跟踪。",
                    score=0.95,
                    rerank_score=4.9,
                ),
                "evidence_id": "E2",
            },
        ]
        final_answer = _shared_logic_comparison_answer(
            required_doc_ids=["doc-a", "doc-b"],
            selected_evidence=selected_evidence,
            doc_focus_map={
                "doc-a": "养殖股产能去化逻辑的回归与演绎",
                "doc-b": "养殖股产能去化主线的持续跟踪",
            },
            shared_points=["养殖股产能去化逻辑"],
            conclusion="两份周报主线一致。",
        )
        self.assertIn("0317", final_answer)
        self.assertIn("0324", final_answer)
        self.assertIn("养殖股产能去化逻辑", final_answer)

    def test_shared_logic_repair_allows_missing_difference_detail(self) -> None:
        selected_evidence = [
            {
                **self._row(
                    chunk_id="doc-a-c1",
                    doc_id="doc-a",
                    file_name="农林牧渔_2026-03-17_REF001_建议关注养殖股产能去化逻辑的回归和演绎.pdf",
                    text="报告强调养殖股产能去化逻辑的回归与演绎。",
                    score=1.0,
                    rerank_score=5.0,
                ),
                "evidence_id": "E1",
            },
            {
                **self._row(
                    chunk_id="doc-b-c1",
                    doc_id="doc-b",
                    file_name="农林牧渔_2026-03-24_REF002_建议关注养殖股产能去化逻辑的回归和演绎.pdf",
                    text="报告继续围绕养殖股产能去化主线展开跟踪。",
                    score=0.95,
                    rerank_score=4.9,
                ),
                "evidence_id": "E2",
            },
        ]
        payload, fallback_reason = _repair_answer_payload(
            query="两份山西证券农业行业周报（0317 与 0324）共同关注什么逻辑？",
            answer_mode="comparison",
            fact_subtype="semantic_fact",
            payload={
                "doc_focus_map": {
                    "doc-a": "养殖股产能去化逻辑的回归与演绎",
                    "doc-b": "养殖股产能去化主线的持续跟踪",
                },
                "shared_points": ["养殖股产能去化逻辑"],
                "used_evidence_ids": ["E1", "E2"],
            },
            selected_evidence=selected_evidence,
        )
        self.assertIsNone(fallback_reason)
        self.assertIn("养殖股产能去化逻辑", payload["final_answer"])
        self.assertIn("0317", payload["final_answer"])
        self.assertIn("0324", payload["final_answer"])

    def test_inductive_compose_accepts_single_strong_theme_with_basis(self) -> None:
        selected_evidence = [
            {
                **self._row(
                    chunk_id="doc-a-c1",
                    doc_id="doc-a",
                    file_name="半导体_2026-03-24_REF001_年度策略报告A.pdf",
                    text="报告A强调AI算力扩张正在带动硬件投资和散热需求改善。",
                    score=1.0,
                    rerank_score=5.0,
                ),
                "evidence_id": "E1",
            },
            {
                **self._row(
                    chunk_id="doc-b-c1",
                    doc_id="doc-b",
                    file_name="半导体_2026-03-09_REF002_行业周报B.pdf",
                    text="报告B强调AI需求带动存储景气和算力链条持续受益。",
                    score=0.9,
                    rerank_score=4.8,
                ),
                "evidence_id": "E2",
            },
        ]
        final_answer = _compose_inductive_answer(
            query="近期电子/半导体报告共同强调了哪些主题？",
            observed_doc_ids=["doc-a", "doc-b"],
            selected_evidence=selected_evidence,
            per_doc_observation={
                "doc-a": "AI算力扩张带动硬件投资和散热需求改善",
                "doc-b": "AI需求带动存储景气和算力链条持续受益",
            },
            shared_themes=["AI算力景气"],
            synthesis_basis="两篇报告都把AI算力扩张视为拉动产业链景气的主线。",
            synthesis="",
        )
        self.assertNotEqual(final_answer, FALLBACK_ANSWER)
        self.assertIn("AI算力景气", final_answer)
        self.assertIn("年度策略报告A", final_answer)
        self.assertIn("行业周报B", final_answer)


if __name__ == "__main__":
    unittest.main()
