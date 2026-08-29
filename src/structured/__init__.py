"""Structured financial facts layer (checklist v3.0 §P5).

Schema + normalizers + routing are local deterministic functions; extraction
may use the remote LLM provider and is validated locally before persistence.
"""

from __future__ import annotations

from src.structured.schema import (
    FinancialFact,
    Metric,
    Period,
    PeriodBasis,
    PeriodType,
    PROVENANCE_FIELDS,
    ValueType,
    missing_provenance_fields,
)
from src.structured.period_normalizer import normalize_period
from src.structured.unit_normalizer import is_currency_metric, is_margin_metric, normalize_fact_value
from src.structured.value_type import classify_value_type
from src.structured.fact_normalizer import match_metric, normalize_company_name
from src.structured.fact_extractor import FactExtractionError, FactExtractor, validate_candidate
from src.structured.fact_store import FactStore
from src.structured.fact_query import FactQuery, build_structured_answer, detect_query_signature, route
from src.structured.metric_registry import (
    MetricSpec,
    all_metric_specs,
    extract_metrics,
    match_metric,
    metric_allowed_units,
    metric_dimension,
    metric_label,
    metrics_compatible,
)
from src.structured.structured_query import (
    OperandRequirement,
    StructuredQuery,
    StructuredQueryResult,
    build_structured_query,
    execute_structured_query,
    parse_structured_query,
    summarize_structured_result,
)

__all__ = [
    "FinancialFact",
    "Metric",
    "Period",
    "PeriodBasis",
    "PeriodType",
    "PROVENANCE_FIELDS",
    "ValueType",
    "missing_provenance_fields",
    "normalize_period",
    "normalize_fact_value",
    "is_currency_metric",
    "is_margin_metric",
    "classify_value_type",
    "match_metric",
    "normalize_company_name",
    "FactExtractor",
    "FactExtractionError",
    "validate_candidate",
    "FactStore",
    "FactQuery",
    "detect_query_signature",
    "route",
    "build_structured_answer",
    "MetricSpec",
    "all_metric_specs",
    "extract_metrics",
    "match_metric",
    "metric_allowed_units",
    "metric_dimension",
    "metric_label",
    "metrics_compatible",
    "StructuredQuery",
    "OperandRequirement",
    "StructuredQueryResult",
    "build_structured_query",
    "execute_structured_query",
    "parse_structured_query",
    "summarize_structured_result",
]
