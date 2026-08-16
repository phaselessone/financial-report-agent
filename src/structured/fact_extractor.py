"""Fact extraction via remote LLM + local validation (checklist v3.0 §P5).

The extractor sends source text to the generic :class:`~src.llm.base.LLMProvider`
(remote API only — no local generative LLM) and validates every candidate
locally before it can become a :class:`~src.structured.schema.FinancialFact`.
Candidates that fail metric/period/value/unit/value_type normalization or
provenance verification (source_span must occur in the source text) are
dropped: they never become official facts.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from src.generation.payload_parser import parse_model_json
from src.llm.base import LLMProvider
from src.llm.types import LLMProviderError, LLMResponse
from src.llm.usage import summarize_usage
from src.structured.fact_normalizer import match_metric, normalize_company_name
from src.structured.period_normalizer import normalize_period
from src.structured.schema import FinancialFact, Metric, ValueType
from src.structured.unit_normalizer import normalize_fact_value
from src.structured.value_type import classify_value_type
from src.utils.text_utils import normalize_text

SYSTEM_PROMPT = (
    "You are a financial fact extraction assistant. Extract structured financial "
    "facts from the provided Chinese financial report text. Answer with one JSON "
    "object only. Do not output markdown, code fences, or any extra text."
)

_USER_INSTRUCTIONS = """Extract financial facts as a JSON object {"facts": [...]}.
Each fact must have these fields:
- "company": Chinese company name
- "metric": one of revenue | net_profit | gross_margin | rd_expense | operating_cash_flow
  (营收/营业收入 -> revenue, 净利润/归母净利润 -> net_profit, 毛利率 -> gross_margin,
   研发费用/研发投入 -> rd_expense, 经营性现金流 -> operating_cash_flow)
- "period": the period exactly as written (e.g. "2025年一季度", "2025年全年", "FY2025")
- "value_raw": the exact numeric string WITH its unit (e.g. "123.4亿元", "25%", "5,678万元")
- "value_type": actual | forecast | unknown (预计/预期/目标/指引 -> forecast; 实现/同比增长 -> actual)
- "source_span": the exact verbatim substring of the source text that contains the value

Rules:
- Only extract V1 metrics listed above. Skip other metrics (净利率, 毛利, 增速 etc).
- gross_margin must use a percentage value (毛利率 25%).
- Currency metrics (revenue/net_profit/rd_expense/operating_cash_flow) must quote the unit (亿元/万元 etc).
- source_span must be copied verbatim from the source text.
- Skip facts whose value or period cannot be quoted verbatim.
- Return {"facts": []} when nothing qualifies."""


class FactExtractionError(Exception):
    """Raised when the LLM payload cannot be parsed into fact candidates."""


def _clean_value_raw(raw: str) -> str:
    cleaned = re.sub(r"\s+", "", str(raw))
    cleaned = re.sub(r"^(?:大约|约|近|逾|超)(?=\d)", "", cleaned)
    return cleaned


def _resolve_metric(raw: Any) -> Metric | None:
    if isinstance(raw, str) and raw.strip():
        try:
            return Metric(raw.strip().lower())
        except ValueError:
            pass
        return match_metric(raw)
    if isinstance(raw, Metric):
        return raw
    return None


def _resolve_value_type(raw: Any, source_span: str) -> ValueType:
    if isinstance(raw, str) and raw.strip():
        try:
            return ValueType(raw.strip().lower())
        except ValueError:
            pass
    return classify_value_type(source_span)


def _resolve_company(candidate_company: Any, aliases: Mapping[str, Sequence[str]], company_hint: str) -> str | None:
    company = normalize_company_name(candidate_company, aliases)
    if company:
        return company
    hint = normalize_company_name(company_hint, aliases)
    if hint:
        return hint
    return None


def validate_candidate(
    candidate: dict[str, Any],
    *,
    doc_id: str,
    page: int,
    evidence_id: str,
    source_text: str,
    company_aliases: Mapping[str, Sequence[str]],
    anchor_year: int | None = None,
    company_hint: str = "",
) -> FinancialFact | None:
    """Validate one LLM fact candidate; returns a FinancialFact or None.

    Local validation chain: metric resolution, period normalization, value/unit
    normalization with dimension checks, value-type resolution, company
    resolution, and provenance verification (source_span must be a verbatim
    substring of the source text). Any failure drops the candidate.
    """
    if not isinstance(candidate, dict):
        return None
    metric = _resolve_metric(candidate.get("metric"))
    if metric is None:
        return None
    period = normalize_period(candidate.get("period"), anchor_year=anchor_year)
    if period is None:
        return None
    raw_value = _clean_value_raw(candidate.get("value_raw", ""))
    normalized = normalize_fact_value(raw_value, metric)
    if normalized is None:
        return None
    value, unit = normalized
    source_span = str(candidate.get("source_span") or "").strip()
    if not source_span:
        return None
    normalized_source = normalize_text(source_text)
    normalized_span = normalize_text(source_span)
    if not normalized_span or normalized_span not in normalized_source:
        return None
    company = _resolve_company(candidate.get("company"), company_aliases, company_hint)
    if company is None:
        return None
    value_type = _resolve_value_type(candidate.get("value_type"), source_span)
    try:
        return FinancialFact(
            company=company,
            metric=metric,
            period=period,
            value_type=value_type,
            value=value,
            unit=unit,
            doc_id=doc_id,
            page=page,
            evidence_id=evidence_id,
            raw_value=raw_value,
            source_span=source_span,
        )
    except ValueError:
        return None


def build_user_prompt(source_text: str) -> str:
    return f"{_USER_INSTRUCTIONS}\n\nSOURCE TEXT:\n{source_text}"


class FactExtractor:
    """Extract validated financial facts from source text via a remote LLM."""

    def __init__(
        self,
        *,
        llm: LLMProvider,
        company_aliases: Mapping[str, Sequence[str]],
        anchor_year: int | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        use_response_format: bool = True,
    ) -> None:
        self._llm = llm
        self.company_aliases = dict(company_aliases)
        self.anchor_year = anchor_year
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_response_format = use_response_format
        self.llm_calls: list[LLMResponse] = []

    def _generate_payload(self, source_text: str, *, doc_id: str) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(source_text)},
        ]
        response_format = {"type": "json_object"} if self.use_response_format else None
        try:
            response = self._llm.generate(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format=response_format,
                metadata={"task": "fact_extraction", "doc_id": doc_id},
            )
        except LLMProviderError as exc:
            raise FactExtractionError(f"fact extraction LLM call failed: {exc.message}") from exc
        self.llm_calls.append(response)
        payload = parse_model_json(response.content)
        if payload is None:
            raise FactExtractionError("fact extraction returned unparseable JSON")
        if not isinstance(payload.get("facts"), list):
            raise FactExtractionError("fact extraction payload missing a 'facts' list")
        return payload

    def extract(
        self,
        *,
        doc_id: str,
        page: int,
        evidence_id: str,
        source_text: str,
        company_hint: str = "",
    ) -> list[FinancialFact]:
        """Extract validated facts from one source text (raises on malformed payload)."""
        payload = self._generate_payload(source_text, doc_id=doc_id)
        facts: list[FinancialFact] = []
        seen: set[str] = set()
        for candidate in payload["facts"]:
            fact = validate_candidate(
                candidate,
                doc_id=doc_id,
                page=page,
                evidence_id=evidence_id,
                source_text=source_text,
                company_aliases=self.company_aliases,
                anchor_year=self.anchor_year,
                company_hint=company_hint,
            )
            if fact is not None and fact.fact_id not in seen:
                seen.add(fact.fact_id)
                facts.append(fact)
        return facts

    def extract_from_rows(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        company_hint: str = "",
        tolerant: bool = True,
    ) -> list[FinancialFact]:
        """Extract facts from evidence rows.

        Each row supplies doc_id / page (page_start) / evidence_id (falling
        back to chunk_id) / text. In tolerant mode a malformed payload for one
        row is skipped; otherwise it raises.
        """
        facts: list[FinancialFact] = []
        seen: set[str] = set()
        for index, row in enumerate(rows):
            doc_id = str(row.get("doc_id") or "")
            page = int(row.get("page_start") or 0)
            evidence_id = str(row.get("evidence_id") or row.get("chunk_id") or f"E{index + 1}")
            source_text = str(row.get("text") or row.get("child_text") or "")
            if not doc_id or not source_text:
                continue
            try:
                batch = self.extract(
                    doc_id=doc_id,
                    page=page,
                    evidence_id=evidence_id,
                    source_text=source_text,
                    company_hint=company_hint,
                )
            except FactExtractionError:
                if not tolerant:
                    raise
                continue
            for fact in batch:
                if fact.fact_id not in seen:
                    seen.add(fact.fact_id)
                    facts.append(fact)
        return facts

    def usage_summary(self) -> dict[str, float | int]:
        """Aggregate usage/latency/retry stats for the calls made by this extractor."""
        return summarize_usage(self.llm_calls)
