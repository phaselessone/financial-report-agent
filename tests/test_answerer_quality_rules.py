import unittest

from src.generation.answerer import (
    FALLBACK_ANSWER,
    _common_terms_for_rows,
    _fallback_payload,
    _finalize_used_evidence_ids,
    _refine_citation_evidence_ids,
    _repair_answer_payload,
    infer_answer_mode,
    infer_query_domain_buckets,
    resolve_query_domain_buckets,
)


class AnswererQualityRuleTests(unittest.TestCase):
    def _row(self, *, doc_id: str, file_name: str, text: str, evidence_id: str) -> dict[str, object]:
        return {
            'evidence_id': evidence_id,
            'chunk_id': f'{doc_id}-{evidence_id}',
            'doc_id': doc_id,
            'file_name': file_name,
            'page_start': 1,
            'page_end': 1,
            'text': text,
            'child_text': text,
            'support_span': text,
            'chunk_type': 'text',
            'element_type': 'paragraph',
            'section_title': '',
            'section_path': '',
        }

    def test_report_lookup_patterns_cover_broader_fact_queries(self) -> None:
        cases = [
            '哪份年度策略报告提出 AI 开启新一轮硬件通胀、国产算力加速突围？',
            '哪份周报写到猪价持续低迷、重视板块去产能投资机会？',
            '哪份白皮书讨论了新冠肺炎全球风险评估第9版？',
            '哪份医药健康行业研究强调业绩与临床数据催化密集、看好创新药回暖？',
        ]
        for query in cases:
            with self.subTest(query=query):
                self.assertEqual(infer_answer_mode(query, 'fact'), 'report_lookup')

    def test_query_domain_buckets_include_agriculture_and_healthcare(self) -> None:
        buckets = infer_query_domain_buckets('WHO 风险评估报告与生猪养殖周报共同强调了哪些晶圆代工机会？')
        self.assertIn('healthcare', buckets)
        self.assertIn('agriculture', buckets)
        self.assertIn('semiconductor', buckets)

    def test_query_domain_hint_precedes_inferred_buckets(self) -> None:
        buckets = resolve_query_domain_buckets(
            '报告中关于“现货、平均、均价”的关键数据或变化是怎样的？',
            domain_hint='semiconductor',
        )
        self.assertEqual(buckets[0], 'semiconductor')

    def test_comparison_fallback_avoids_date_only_topic(self) -> None:
        rows = [
            self._row(
                doc_id='doc-a',
                evidence_id='E1',
                file_name='东海证券-电子行业周报：中国晶圆产能占比望超30%，小米2025年四大业务协同增长-260330.pdf',
                text='AIOT 需求持续增长，乐鑫科技和恒玄科技受益明显。',
            ),
            self._row(
                doc_id='doc-b',
                evidence_id='E2',
                file_name='中山证券-电子行业周报：机构预测2026年PC出货量将下滑12%-260312.pdf',
                text='PC 出货量承压，存储芯片与功率半导体细分赛道出现分化。',
            ),
        ]
        payload = _fallback_payload('comparison', rows, fact_subtype='semantic_fact')
        self.assertIn('主要强调', payload['final_answer'])
        self.assertNotIn('260330', payload['final_answer'])
        self.assertNotIn('260312', payload['final_answer'])

    def test_inductive_fallback_abstains_when_only_noise_terms_exist(self) -> None:
        rows = [
            self._row(
                doc_id='doc-a',
                evidence_id='E1',
                file_name='????-??????-260317.pdf',
                text='农林牧渔行业近一年市场表现。本周沪深300指数上涨。',
            ),
            self._row(
                doc_id='doc-b',
                evidence_id='E2',
                file_name='????-??????-260324.pdf',
                text='农林牧渔行业近一年市场表现。本周指数震荡。',
            ),
        ]
        payload = _fallback_payload('inductive', rows, fact_subtype='semantic_fact')
        self.assertEqual(payload['final_answer'], FALLBACK_ANSWER)

    def test_common_terms_filter_domain_and_report_noise(self) -> None:
        rows = [
            self._row(
                doc_id='doc-a',
                evidence_id='E1',
                file_name='国金证券-农林牧渔行业周报：猪价持续低迷，重视板块去产能投资机会-260329.pdf',
                text='生猪价格持续下跌，行业去产能有望加速。',
            ),
            self._row(
                doc_id='doc-b',
                evidence_id='E2',
                file_name='开源证券-农林牧渔行业周报：猪价创五年新低，重压之下静待产能去化提速-260314.pdf',
                text='猪价继续下探，产能去化节奏加快，供给压力仍存。',
            ),
        ]
        terms = _common_terms_for_rows(rows)
        self.assertTrue(any(term in terms for term in ['生猪', '猪价', '产能', '供给']))
        self.assertNotIn('农林牧渔', terms)
        self.assertNotIn('周报', terms)

    def test_comparison_payload_requires_focus_for_each_doc(self) -> None:
        rows = [
            self._row(
                doc_id='doc-a',
                evidence_id='E1',
                file_name='国金证券-农林牧渔行业周报：猪价持续低迷，重视板块去产能投资机会-260329.pdf',
                text='生猪价格持续下跌，行业去产能有望加速。',
            ),
            self._row(
                doc_id='doc-b',
                evidence_id='E2',
                file_name='国金证券-农林牧渔行业周报：生猪价格持续下跌，牛价有望开启上行-260322.pdf',
                text='牛价景气改善，产业逻辑切换。',
            ),
        ]
        payload, fallback_reason = _repair_answer_payload(
            answer_mode='comparison',
            fact_subtype='semantic_fact',
            payload={
                'doc_focus_map': {'doc-a': '去产能逻辑'},
                'final_answer': '两份报告都关注农业。',
                'used_evidence_ids': ['E1', 'E2'],
            },
            selected_evidence=rows,
        )
        self.assertEqual(fallback_reason, 'comparison_validation')
        self.assertIn('主要强调', payload['final_answer'])

    def test_inductive_payload_requires_multiple_shared_themes(self) -> None:
        rows = [
            self._row(
                doc_id='doc-a',
                evidence_id='E1',
                file_name='消费_2026-03-10_AP202603101820454018_商社美护行业周报：政府工作报告加码促内需.pdf',
                text='政府工作报告加码促内需，支持新型消费。',
            ),
            self._row(
                doc_id='doc-b',
                evidence_id='E2',
                file_name='消费_2026-03-23_AP202603231820694394_商贸零售行业跟踪周报：重视高低切消费板块投资机会.pdf',
                text='消费分层明显，高低切机会并存。',
            ),
        ]
        payload, fallback_reason = _repair_answer_payload(
            answer_mode='inductive',
            fact_subtype='semantic_fact',
            payload={
                'per_doc_observation': {'doc-a': '促内需政策加码', 'doc-b': '消费分层明显'},
                'shared_themes': ['消费'],
                'final_answer': '共同主题包括：消费。',
                'used_evidence_ids': ['E1', 'E2'],
            },
            selected_evidence=rows,
        )
        self.assertEqual(payload['final_answer'], FALLBACK_ANSWER)
        self.assertEqual(fallback_reason, 'inductive_validation')

    def test_comparison_payload_rebuilds_answer_from_focus_map(self) -> None:
        rows = [
            self._row(
                doc_id='doc-a',
                evidence_id='E1',
                file_name='SourceA_2026-04-01_REF001_Alpha Focus.pdf',
                text='Alpha Focus emphasizes hog price weakness and destocking.',
            ),
            self._row(
                doc_id='doc-b',
                evidence_id='E2',
                file_name='SourceB_2026-04-01_REF002_Beta Seed.pdf',
                text='Beta Seed emphasizes breeding innovation and seed supply.',
            ),
        ]
        payload, fallback_reason = _repair_answer_payload(
            answer_mode='comparison',
            fact_subtype='semantic_fact',
            payload={
                'doc_focus_map': {'doc-a': 'hog price weakness', 'doc-b': 'seed revitalization'},
                'differences': ['hog cycle versus seed policy'],
                'conclusion': 'Their focus differs clearly.',
                'final_answer': 'They are different.',
                'used_evidence_ids': ['E1', 'E2'],
            },
            selected_evidence=rows,
        )
        self.assertIsNone(fallback_reason)
        self.assertIn('Alpha Focus', payload['final_answer'])
        self.assertIn('Beta Seed', payload['final_answer'])

    def test_inductive_payload_builds_synthesis_from_observations(self) -> None:
        rows = [
            self._row(
                doc_id='doc-a',
                evidence_id='E1',
                file_name='SourceA_2026-04-01_REF001_Alpha Focus.pdf',
                text='Alpha Focus emphasizes weak prices and supply pressure.',
            ),
            self._row(
                doc_id='doc-b',
                evidence_id='E2',
                file_name='SourceB_2026-04-01_REF002_Beta Seed.pdf',
                text='Beta Seed emphasizes breeding innovation and seed support.',
            ),
        ]
        payload, fallback_reason = _repair_answer_payload(
            answer_mode='inductive',
            fact_subtype='semantic_fact',
            payload={
                'per_doc_observation': {'doc-a': 'weak prices and supply pressure', 'doc-b': 'breeding innovation and seed support'},
                'shared_themes': ['supply', 'innovation'],
                'final_answer': 'Themes: supply, innovation.',
                'used_evidence_ids': ['E1', 'E2'],
            },
            selected_evidence=rows,
        )
        self.assertIsNone(fallback_reason)
        self.assertNotEqual(payload['final_answer'], FALLBACK_ANSWER)

    def test_multidoc_used_evidence_ids_keep_support_rows(self) -> None:
        rows = [
            self._row(doc_id='doc-a', evidence_id='E1', file_name='SourceA_2026-04-01_REF001_Alpha Focus.pdf', text='Alpha overview.'),
            self._row(doc_id='doc-a', evidence_id='E2', file_name='SourceA_2026-04-01_REF001_Alpha Focus.pdf', text='Alpha support paragraph.'),
            self._row(doc_id='doc-b', evidence_id='E3', file_name='SourceB_2026-04-01_REF002_Beta Seed.pdf', text='Beta overview.'),
            self._row(doc_id='doc-b', evidence_id='E4', file_name='SourceB_2026-04-01_REF002_Beta Seed.pdf', text='Beta support paragraph.'),
        ]
        used_ids = _finalize_used_evidence_ids(
            parsed_ids=['E1', 'E3'],
            selected_evidence=rows,
            question_type='comparison',
            answer_mode='comparison',
        )
        self.assertEqual(used_ids[:2], ['E1', 'E3'])
        self.assertIn('E2', used_ids)
        self.assertIn('E4', used_ids)

    def test_citation_refinement_prefers_same_section_paragraph(self) -> None:
        rows = [
            {
                **self._row(doc_id='doc-a', evidence_id='E1', file_name='SourceA_2026-04-01_REF001_Alpha Focus.pdf', text='Alpha title anchor.'),
                'page_start': 1,
                'page_end': 1,
                'section_title': '??',
                'section_path': '??',
            },
            {
                **self._row(doc_id='doc-a', evidence_id='E2', file_name='SourceA_2026-04-01_REF001_Alpha Focus.pdf', text='Alpha support paragraph discusses hog price weakness and destocking.'),
                'page_start': 3,
                'page_end': 3,
                'section_title': '????',
                'section_path': '????',
            },
            {
                **self._row(doc_id='doc-b', evidence_id='E3', file_name='SourceB_2026-04-01_REF002_Beta Seed.pdf', text='Beta title anchor.'),
                'page_start': 1,
                'page_end': 1,
                'section_title': '??',
                'section_path': '??',
            },
            {
                **self._row(doc_id='doc-b', evidence_id='E4', file_name='SourceB_2026-04-01_REF002_Beta Seed.pdf', text='Beta support paragraph discusses seed revitalization and breeding innovation.'),
                'page_start': 4,
                'page_end': 4,
                'section_title': '????',
                'section_path': '????',
            },
        ]
        refined_ids = _refine_citation_evidence_ids(
            answer_mode='comparison',
            payload={
                'doc_focus_map': {'doc-a': 'hog price weakness', 'doc-b': 'seed revitalization'},
                'differences': ['hog cycle versus seed policy'],
                'conclusion': 'Their focus differs clearly.',
                'final_answer': 'Alpha Focus differs from Beta Seed.',
            },
            selected_evidence=rows,
            used_evidence_ids=['E1', 'E3'],
        )
        self.assertLess(refined_ids.index('E2'), refined_ids.index('E1'))
        self.assertLess(refined_ids.index('E4'), refined_ids.index('E3'))

    def test_inductive_citation_refinement_prefers_body_support_over_title_anchor(self) -> None:
        rows = [
            {
                **self._row(doc_id='doc-a', evidence_id='E1', file_name='SourceA_2026-04-01_REF001_Alpha Focus.pdf', text='Alpha title anchor.'),
                'page_start': 1,
                'page_end': 1,
                'section_title': '??',
                'section_path': '??',
            },
            {
                **self._row(doc_id='doc-a', evidence_id='E2', file_name='SourceA_2026-04-01_REF001_Alpha Focus.pdf', text='Alpha support paragraph discusses hog price weakness and supply pressure.'),
                'page_start': 4,
                'page_end': 4,
                'section_title': '????',
                'section_path': '????',
            },
            {
                **self._row(doc_id='doc-b', evidence_id='E3', file_name='SourceB_2026-04-01_REF002_Beta Seed.pdf', text='Beta title anchor.'),
                'page_start': 1,
                'page_end': 1,
                'section_title': '??',
                'section_path': '??',
            },
            {
                **self._row(doc_id='doc-b', evidence_id='E4', file_name='SourceB_2026-04-01_REF002_Beta Seed.pdf', text='Beta support paragraph discusses breeding innovation and seed revitalization.'),
                'page_start': 5,
                'page_end': 5,
                'section_title': '????',
                'section_path': '????',
            },
        ]
        refined_ids = _refine_citation_evidence_ids(
            answer_mode='inductive',
            payload={
                'per_doc_observation': {'doc-a': 'hog price weakness and supply pressure', 'doc-b': 'breeding innovation and seed revitalization'},
                'shared_themes': ['supply', 'innovation'],
                'synthesis': 'Recent reports highlighted supply pressure and breeding innovation.',
                'final_answer': 'Common themes include supply pressure and innovation.',
            },
            selected_evidence=rows,
            used_evidence_ids=['E1', 'E3'],
        )
        self.assertLess(refined_ids.index('E2'), refined_ids.index('E1'))
        self.assertLess(refined_ids.index('E4'), refined_ids.index('E3'))
        self.assertEqual({refined_ids[0], refined_ids[1]}, {'E1', 'E2'})
        self.assertEqual({refined_ids[2], refined_ids[3]}, {'E3', 'E4'})



if __name__ == '__main__':
    unittest.main()
