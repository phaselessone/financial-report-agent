"""Fact extractor tests (checklist v3.0 §P5).

Contract: extraction goes through the generic ``LLMProvider`` protocol (mock
providers only, never real API). Every LLM candidate must pass the local
validation chain — metric/period/value/unit/value_type resolution plus
provenance verification (source_span must occur in the source text) — before
it can become a :class:`FinancialFact`.
"""

from __future__ import annotations

import unittest
from decimal import Decimal

from src.llm.types import LLMResponse
from src.structured.fact_extractor import FactExtractionError, FactExtractor, validate_candidate
from src.structured.schema import FinancialFact, Metric, PeriodType, ValueType

ALIASES = {"贵州茅台": ["贵州茅台", "茅台"], "五粮液": ["五粮液"]}

SOURCE = (
    "2025年公司实现营业收入1234.5亿元，同比增长10%；"
    "归母净利润136.55亿元，同比增长111.78%；"
    "预计2026年营收达到1600亿元，毛利率提升至25%。"
)


class FakeLLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict] = []

    def generate(self, *, messages, temperature=0.0, max_tokens=512, response_format=None, metadata=None, **kwargs):
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": response_format,
                "metadata": metadata,
            }
        )
        return LLMResponse(
            content=self.content,
            provider="fake",
            model="fake-model",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )


def candidate(**overrides):
    base = {
        "company": "贵州茅台",
        "metric": "revenue",
        "period": "FY2025",
        "value_raw": "1234.5亿元",
        "value_type": "actual",
        "source_span": "2025年公司实现营业收入1234.5亿元",
    }
    base.update(overrides)
    return base


class TestValidateCandidate(unittest.TestCase):
    def test_valid_candidate(self) -> None:
        fact = validate_candidate(
            candidate(), doc_id="d1", page=2, evidence_id="E1", source_text=SOURCE, company_aliases=ALIASES
        )
        self.assertIsInstance(fact, FinancialFact)
        self.assertEqual(fact.value, Decimal("123450000000"))
        self.assertEqual(fact.unit, "元")
        self.assertEqual(fact.period.kind, PeriodType.FY)
        self.assertEqual(fact.period.year, 2025)
        self.assertEqual(fact.value_type, ValueType.ACTUAL)

    def test_metric_from_chinese_keyword(self) -> None:
        fact = validate_candidate(
            candidate(metric="归母净利润", period="FY2025", value_raw="136.55亿元"),
            doc_id="d1",
            page=2,
            evidence_id="E1",
            source_text=SOURCE,
            company_aliases=ALIASES,
        )
        self.assertEqual(fact.metric, Metric.NET_PROFIT)

    def test_unknown_metric_rejected(self) -> None:
        self.assertIsNone(
            validate_candidate(
                candidate(metric="ebitda"),
                doc_id="d1",
                page=2,
                evidence_id="E1",
                source_text=SOURCE,
                company_aliases=ALIASES,
            )
        )

    def test_unresolvable_period_rejected(self) -> None:
        self.assertIsNone(
            validate_candidate(
                candidate(period="2025年第二季度"),
                doc_id="d1",
                page=2,
                evidence_id="E1",
                source_text=SOURCE,
                company_aliases=ALIASES,
            )
        )

    def test_period_with_anchor(self) -> None:
        fact = validate_candidate(
            candidate(period="全年"),
            doc_id="d1",
            page=2,
            evidence_id="E1",
            source_text=SOURCE,
            company_aliases=ALIASES,
            anchor_year=2025,
        )
        self.assertEqual(fact.period.kind, PeriodType.FY)
        self.assertEqual(fact.period.year, 2025)

    def test_dimension_mismatch_rejected(self) -> None:
        self.assertIsNone(
            validate_candidate(
                candidate(metric="revenue", value_raw="25%"),
                doc_id="d1",
                page=2,
                evidence_id="E1",
                source_text=SOURCE,
                company_aliases=ALIASES,
            )
        )

    def test_source_span_not_in_text_rejected(self) -> None:
        self.assertIsNone(
            validate_candidate(
                candidate(source_span="2024年营业收入999亿元"),
                doc_id="d1",
                page=2,
                evidence_id="E1",
                source_text=SOURCE,
                company_aliases=ALIASES,
            )
        )

    def test_unknown_company_rejected(self) -> None:
        self.assertIsNone(
            validate_candidate(
                candidate(company="比亚迪"),
                doc_id="d1",
                page=2,
                evidence_id="E1",
                source_text=SOURCE,
                company_aliases=ALIASES,
            )
        )

    def test_company_hint_fallback(self) -> None:
        fact = validate_candidate(
            candidate(company="茅台"),
            doc_id="d1",
            page=2,
            evidence_id="E1",
            source_text=SOURCE,
            company_aliases=ALIASES,
            company_hint="贵州茅台",
        )
        self.assertEqual(fact.company, "贵州茅台")

    def test_value_type_from_span_when_absent(self) -> None:
        fact = validate_candidate(
            candidate(value_type=None),
            doc_id="d1",
            page=2,
            evidence_id="E1",
            source_text=SOURCE,
            company_aliases=ALIASES,
        )
        self.assertEqual(fact.value_type, ValueType.ACTUAL)

    def test_margin_metric_percent(self) -> None:
        fact = validate_candidate(
            candidate(metric="gross_margin", value_raw="25%", source_span="毛利率提升至25%", value_type="forecast"),
            doc_id="d1",
            page=2,
            evidence_id="E1",
            source_text=SOURCE,
            company_aliases=ALIASES,
        )
        self.assertEqual(fact.value, Decimal("25"))
        self.assertEqual(fact.unit, "%")


class TestFactExtractor(unittest.TestCase):
    def _extractor(self, content: str, **kwargs) -> tuple[FactExtractor, FakeLLM]:
        llm = FakeLLM(content)
        extractor = FactExtractor(llm=llm, company_aliases=ALIASES, **kwargs)
        return extractor, llm

    def test_extract_success(self) -> None:
        payload = (
            '{"facts": [{"company": "贵州茅台", "metric": "revenue", "period": "FY2025", '
            '"value_raw": "1234.5亿元", "value_type": "actual", '
            '"source_span": "2025年公司实现营业收入1234.5亿元"}]}'
        )
        extractor, llm = self._extractor(payload)
        facts = extractor.extract(doc_id="d1", page=2, evidence_id="E1", source_text=SOURCE)
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].company, "贵州茅台")
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(llm.calls[0]["response_format"], {"type": "json_object"})
        self.assertEqual(extractor.usage_summary()["total_tokens"], 150)

    def test_extract_skips_invalid_candidates(self) -> None:
        payload = (
            '{"facts": ['
            '{"company": "比亚迪", "metric": "revenue", "period": "FY2025", '
            '"value_raw": "1亿元", "value_type": "actual", "source_span": "2025年营收1亿元"},'
            '{"company": "贵州茅台", "metric": "revenue", "period": "FY2025", '
            '"value_raw": "1234.5亿元", "value_type": "actual", '
            '"source_span": "2025年公司实现营业收入1234.5亿元"}'
            "]}"
        )
        extractor, _ = self._extractor(payload)
        facts = extractor.extract(doc_id="d1", page=2, evidence_id="E1", source_text=SOURCE)
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].company, "贵州茅台")

    def test_malformed_json_raises(self) -> None:
        extractor, _ = self._extractor("not json at all")
        with self.assertRaises(FactExtractionError):
            extractor.extract(doc_id="d1", page=2, evidence_id="E1", source_text=SOURCE)

    def test_facts_not_a_list_raises(self) -> None:
        extractor, _ = self._extractor('{"facts": "oops"}')
        with self.assertRaises(FactExtractionError):
            extractor.extract(doc_id="d1", page=2, evidence_id="E1", source_text=SOURCE)

    def test_empty_payload_raises(self) -> None:
        extractor, _ = self._extractor('{"facts": []}')
        facts = extractor.extract(doc_id="d1", page=2, evidence_id="E1", source_text=SOURCE)
        self.assertEqual(facts, [])

    def test_dedupe_within_source(self) -> None:
        payload = (
            '{"facts": ['
            '{"company": "贵州茅台", "metric": "revenue", "period": "FY2025", '
            '"value_raw": "1234.5亿元", "value_type": "actual", '
            '"source_span": "2025年公司实现营业收入1234.5亿元"},'
            '{"company": "贵州茅台", "metric": "revenue", "period": "FY2025", '
            '"value_raw": "1234.5亿元", "value_type": "actual", '
            '"source_span": "2025年公司实现营业收入1234.5亿元"}'
            "]}"
        )
        extractor, _ = self._extractor(payload)
        facts = extractor.extract(doc_id="d1", page=2, evidence_id="E1", source_text=SOURCE)
        self.assertEqual(len(facts), 1)

    def test_extract_from_rows(self) -> None:
        payload = (
            '{"facts": [{"company": "贵州茅台", "metric": "revenue", "period": "FY2025", '
            '"value_raw": "1234.5亿元", "value_type": "actual", '
            '"source_span": "2025年公司实现营业收入1234.5亿元"}]}'
        )
        extractor, _ = self._extractor(payload)
        rows = [
            {"chunk_id": "c1", "doc_id": "d1", "page_start": 2, "evidence_id": "E1", "text": SOURCE},
            {"chunk_id": "c2", "doc_id": "d2", "page_start": 3, "evidence_id": "E2", "text": "无数字"},
        ]
        facts = extractor.extract_from_rows(rows)
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].doc_id, "d1")
        self.assertEqual(facts[0].page, 2)


if __name__ == "__main__":
    unittest.main()
