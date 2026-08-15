"""Compatibility shim for src.generation.answerer (P1).

All symbols previously defined or imported in this module now live in the
dedicated P1 modules; this module re-exports them so existing imports keep
working unchanged.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from src.generation.answer_service import LocalEvidenceAnswerer
from src.generation.evidence_selector import CLUSTER_DATE_FRAGMENT_RE, CLUSTER_ENGLISH_NOISE_RE, CLUSTER_PERIOD_FRAGMENT_RE, CLUSTER_REPORT_NOISE_RE, ENGLISH_CLUSTER_NOISE_TERMS, GENERIC_THEME_TERMS, NOISY_SECTION_PATTERNS, NORMALIZED_CLUSTER_NOISE_TERMS, REPORT_PERIOD_RE, SINGLE_LETTER_TOKEN_RE, TOPIC_DATE_RE, TOPIC_NOISE_PATTERNS, TOPIC_NOISE_TERMS, TOPIC_NUMERIC_RE, _append_candidate_selection, _apply_domain_guard, _assign_evidence_ids, _best_companion, _best_support_row, _build_doc_candidates, _candidate_cluster_key, _candidate_diversity_key, _candidate_domain_bucket, _candidate_source_key, _candidate_title_key, _candidate_title_match_score, _candidate_title_match_sort_key, _candidate_topic_family_key, _candidate_topic_key, _citation_context_map, _citation_row_sort_key, _cluster_stem_terms, _coerce_doc_summary_map, _common_terms_for_rows, _compact_summary_text, _compose_comparison_answer, _compose_inductive_answer, _compose_inductive_synthesis, _context_overlap_score, _default_doc_used_evidence_ids, _default_evidence_summary, _derive_shared_themes, _doc_theme_summary, _doc_title_variants, _duplicate_short_titles, _ensure_doc_focus_map, _ensure_per_doc_observation, _evidence_line, _finalize_used_evidence_ids, _has_focus_overlap, _is_noise_evidence_row, _is_noise_theme_term, _is_noise_topic_text, _is_paragraph_like, _is_single_evidence_rephrase, _is_structured_row, _iter_candidate_pool, _normalize_theme_candidates, _normalize_theme_list, _ordered_pool, _pick_ranked_candidate, _preferred_numeric_semantic_answer, _prepare_evidence, _query_bucket_tokens, _refine_citation_evidence_ids, _report_display_label, _report_period_label, _report_source_key, _required_doc_ids, _row_match_text, _row_query_overlap, _row_score, _row_text, _row_topic_hint, _rows_by_doc, _same_context, _select_broad_inductive_doc_candidates, _select_comparison_doc_candidates, _select_conservative_inductive_doc_candidates, _select_diverse_doc_candidates, _select_doc_anchor, _select_inductive_doc_candidates, _select_multidoc_rows_for_doc, _select_report_lookup_rows, _select_rows_for_doc, _selected_domain_mismatch, _shared_logic_comparison_answer, _shared_logic_focus_text, _short_report_title, _theme_terms, _title_overlap_score, _title_page_candidates, _title_topic_hint, _uncovered_query_domains, _unique_rows, clean_report_stem, clean_topic_stem, cluster_key_for_candidate, summarize_doc_candidates, summarize_retrieval_scores, summarize_selected_evidence
from src.generation.payload_parser import _strip_code_fences, parse_model_json
from src.generation.payload_repair import _fallback_payload, _repair_answer_payload
from src.generation.routing import BROAD_INDUCTIVE_GENERIC_TERMS, BROAD_INDUCTIVE_MARKER_PHRASES, COARSE_DOMAIN_QUERY_TERMS, DOMAIN_COMPATIBILITY_GROUPS, FALLBACK_ANSWER, GENERIC_NUMERIC_QUERY_TERMS, NORMALIZED_BROAD_INDUCTIVE_GENERIC_TERMS, NORMALIZED_BROAD_INDUCTIVE_MARKER_PHRASES, NORMALIZED_COARSE_DOMAIN_QUERY_TERMS, NORMALIZED_FINE_GRAIN_QUERY_HINTS, NORMALIZED_GENERIC_NUMERIC_QUERY_TERMS, NUMERIC_QUERY_HINTS, QUERY_DOMAIN_HINTS, QUERY_REPORT_TITLE_RE, QUESTION_TYPE_KEYWORDS, REPORT_LOOKUP_PATTERN, SHARED_LOGIC_COMPARISON_MARKERS, VALUE_QUERY_HINTS, _compatible_domain_buckets, _domains_compatible, _extract_query_report_titles, _fallback_abstain_reason, _is_generic_numeric_semantic_query, _is_weak_domain_bucket, _meaningful_numeric_query_terms, _normalize_domain_bucket, _query_terms, has_strong_entity_terms, infer_answer_mode, infer_comparison_subtype, infer_fact_subtype, infer_query_domain_bucket, infer_query_domain_buckets, infer_question_type, is_broad_inductive_query, is_fallback_answer, is_numeric_or_table_query, resolve_query_domain_buckets

from src.generation.abstain import compute_confidence_label, decide_abstention, select_used_evidence_ids, validate_answer_support
from src.generation.citation_builder import build_citations
from src.generation.prompt import (
    SYSTEM_PROMPT,
    build_answer_prompt,
    build_comparison_focus_prompt,
    build_comparison_synthesis_prompt,
    build_inductive_observation_prompt,
    build_inductive_synthesis_prompt,
)
from src.retrieval.model_store import ensure_model_downloaded
from src.utils.text_utils import (
    extract_terms,
    first_sentence,
    industry_from_file_name,
    normalize_for_match,
    normalize_text,
    short_title_from_file_name,
    strip_file_extension,
    topic_hint_from_title,
)
