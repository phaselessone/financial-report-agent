"""Financial tool layer (checklist v3.0 §P4).

Local deterministic tools only: no LLM calls, no dynamic code execution.
"""

from __future__ import annotations

from src.tools.citation_verify import verify_citations
from src.tools.evidence_search import evidence_rows_for_doc, lookup_evidence, lookup_evidence_in_pool
from src.tools.financial_calculator import (
    CalculationError,
    CalculationInput,
    CalculationResult,
    cagr,
    difference,
    gross_margin,
    growth_rate,
    input_from_row,
    net_margin,
    parse_numeric_value,
    percentage_point_change,
    ratio,
    yoy,
)
from src.tools.report_lookup import search_within_report, search_within_reports
from src.tools.report_search import ReportSearchTool, top_rows

__all__ = [
    "ReportSearchTool",
    "top_rows",
    "search_within_report",
    "search_within_reports",
    "lookup_evidence",
    "lookup_evidence_in_pool",
    "evidence_rows_for_doc",
    "verify_citations",
    "CalculationError",
    "CalculationInput",
    "CalculationResult",
    "parse_numeric_value",
    "input_from_row",
    "growth_rate",
    "yoy",
    "cagr",
    "gross_margin",
    "net_margin",
    "ratio",
    "difference",
    "percentage_point_change",
]
