"""Local claim_type classification tests (checklist v3.0 §P7).

The claim extractor trusts the LLM for claim splitting/text but classifies
claim_type with deterministic local heuristics so the three types are stable:
  DERIVED      - text expresses a calculation/derived figure (growth, ratio, diff...)
  SYNTHESIZED  - draws on 2+ distinct source docs (cross-report synthesis), not a calc
  EXTRACTED    - single-source directly quoted/restated figure or fact

Priority: DERIVED > SYNTHESIZED > EXTRACTED.
"""

from __future__ import annotations

import unittest

from src.agent.claims import classify_claim_type


class ClaimTypeTests(unittest.TestCase):
    def test_derived_growth_keyword(self) -> None:
        self.assertEqual(classify_claim_type("营收同比增长31.1%。", ["d1"]), "DERIVED")

    def test_derived_ratio_verb(self) -> None:
        self.assertEqual(classify_claim_type("毛利率占营收比为45.2%。", ["d1"]), "DERIVED")

    def test_derived_diff_keyword(self) -> None:
        self.assertEqual(classify_claim_type("全年营收较上年增加120亿元。", ["d1"]), "DERIVED")

    def test_derived_wins_over_synthesis(self) -> None:
        # A calculation statement that also spans two docs is still DERIVED.
        self.assertEqual(classify_claim_type("两家公司合计营收增速为18%。", ["d1", "d2"]), "DERIVED")

    def test_synthesized_two_docs_non_numeric(self) -> None:
        self.assertEqual(classify_claim_type("两家公司都强调AI驱动的增长逻辑。", ["d1", "d2"]), "SYNTHESIZED")

    def test_synthesized_inductive_cross_report(self) -> None:
        self.assertEqual(classify_claim_type("多份报告共同关注涨价主线。", ["d1", "d2", "d3"]), "SYNTHESIZED")

    def test_extracted_single_source(self) -> None:
        self.assertEqual(classify_claim_type("宁德时代2025年营收达到1234亿元。", ["d1"]), "EXTRACTED")

    def test_extracted_single_doc_multi_evidence(self) -> None:
        # Two evidence ids but the same doc - still EXTRACTED.
        self.assertEqual(classify_claim_type("公司公告称将加大研发投入。", ["d1", "d1"]), "EXTRACTED")

    def test_empty_text_defaults_extracted(self) -> None:
        self.assertEqual(classify_claim_type("", ["d1"]), "EXTRACTED")


if __name__ == "__main__":
    unittest.main()
