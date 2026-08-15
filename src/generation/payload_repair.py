"""P1 module extracted from src/generation/answerer.py (behavior preserved)."""
from __future__ import annotations

from collections import defaultdict
from src.generation.evidence_selector import _candidate_diversity_key, _coerce_doc_summary_map, _common_terms_for_rows, _compose_comparison_answer, _compose_inductive_answer, _compose_inductive_synthesis, _default_doc_used_evidence_ids, _default_evidence_summary, _derive_shared_themes, _doc_theme_summary, _doc_title_variants, _duplicate_short_titles, _ensure_doc_focus_map, _ensure_per_doc_observation, _has_focus_overlap, _is_noise_topic_text, _is_single_evidence_rephrase, _normalize_theme_candidates, _preferred_numeric_semantic_answer, _report_source_key, _required_doc_ids, _row_text, _rows_by_doc, _shared_logic_comparison_answer, _short_report_title, _title_topic_hint, clean_report_stem, clean_topic_stem
from src.generation.routing import FALLBACK_ANSWER, _is_generic_numeric_semantic_query, infer_comparison_subtype, is_broad_inductive_query
from src.utils.text_utils import first_sentence, normalize_for_match, normalize_text
from typing import Any


def _fallback_payload(answer_mode: str, selected_evidence: list[dict[str, Any]], *, fact_subtype: str, query: str = "") -> dict[str, Any]:
    conservative_note = "Conservative fallback based on current evidence."
    if not selected_evidence:
        return {
            "final_answer": FALLBACK_ANSWER,
            "evidence_summary": "",
            "uncertainty_note": conservative_note,
            "used_evidence_ids": [],
        }

    if answer_mode == "report_lookup":
        title = _short_report_title(selected_evidence[0]["file_name"])
        return {
            "final_answer": title,
            "evidence_summary": _default_evidence_summary(answer_mode, selected_evidence),
            "uncertainty_note": "",
            "used_evidence_ids": [selected_evidence[0]["evidence_id"]],
        }

    if answer_mode == "numeric_fact":
        anchor = selected_evidence[0]
        answer_text = first_sentence(_row_text(anchor, prefer_exact=True), max_chars=96) or FALLBACK_ANSWER
        if fact_subtype == "semantic_fact":
            preferred_answer = _preferred_numeric_semantic_answer(selected_evidence)
            if preferred_answer:
                answer_text = preferred_answer
            if _is_generic_numeric_semantic_query(query, fact_subtype=fact_subtype, answer_mode=answer_mode):
                answer_text = preferred_answer or answer_text
        return {
            "final_answer": answer_text,
            "evidence_summary": _default_evidence_summary(answer_mode, selected_evidence),
            "uncertainty_note": "",
            "used_evidence_ids": [row["evidence_id"] for row in selected_evidence[:2]],
        }

    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_evidence:
        by_doc[row["doc_id"]].append(row)
    ordered_docs: list[str] = []
    seen_docs: set[str] = set()
    for row in selected_evidence:
        if row["doc_id"] in seen_docs:
            continue
        ordered_docs.append(row["doc_id"])
        seen_docs.add(row["doc_id"])

    if answer_mode == "comparison":
        if len(ordered_docs) < 2:
            return {
                "final_answer": FALLBACK_ANSWER,
                "evidence_summary": _default_evidence_summary(answer_mode, selected_evidence),
                "uncertainty_note": conservative_note,
                "used_evidence_ids": [],
            }
        doc_summaries: list[tuple[dict[str, Any], str]] = []
        for doc_id in ordered_docs[:2]:
            rows = by_doc[doc_id]
            summary = _doc_theme_summary(rows)
            if not summary or _is_noise_topic_text(summary):
                return {
                    "final_answer": FALLBACK_ANSWER,
                    "evidence_summary": _default_evidence_summary(answer_mode, selected_evidence),
                    "uncertainty_note": conservative_note,
                    "used_evidence_ids": [],
                }
            doc_summaries.append((rows[0], summary))
        common_terms = _common_terms_for_rows(selected_evidence, min_doc_frequency=2, top_k=2)
        tail = (
            "二者共同点在于" + "、".join(common_terms) + "。"
            if common_terms
            else "二者关注点存在明显差异。"
        )
        final_answer = normalize_text(
            f"《{_short_report_title(doc_summaries[0][0]['file_name'])}》主要强调{doc_summaries[0][1]}；"
            f"《{_short_report_title(doc_summaries[1][0]['file_name'])}》主要强调{doc_summaries[1][1]}。{tail}"
        )
        return {
            "final_answer": final_answer,
            "evidence_summary": _default_evidence_summary(answer_mode, selected_evidence),
            "uncertainty_note": "",
            "used_evidence_ids": [by_doc[doc_id][0]["evidence_id"] for doc_id in ordered_docs[:2]],
        }

    common_terms = _common_terms_for_rows(selected_evidence)
    if len(common_terms) < 2:
        return {
            "final_answer": FALLBACK_ANSWER,
            "evidence_summary": _default_evidence_summary(answer_mode, selected_evidence),
            "uncertainty_note": conservative_note,
            "used_evidence_ids": [],
        }
    common_text = "、".join(common_terms[:3])
    return {
        "final_answer": normalize_text(f"共同主题包括：{common_text}。"),
        "evidence_summary": _default_evidence_summary(answer_mode, selected_evidence),
        "uncertainty_note": "",
        "used_evidence_ids": [by_doc[doc_id][0]["evidence_id"] for doc_id in ordered_docs[:3]],
    }


def _repair_answer_payload(
    *,
    query: str = "",
    answer_mode: str,
    fact_subtype: str,
    payload: dict[str, Any],
    selected_evidence: list[dict[str, Any]],
) -> tuple[dict[str, Any], str | None]:
    fallback = _fallback_payload(answer_mode, selected_evidence, fact_subtype=fact_subtype, query=query)
    final_answer = normalize_text(payload.get("final_answer", "")) or fallback["final_answer"]
    evidence_summary = normalize_text(payload.get("evidence_summary", "")) or fallback["evidence_summary"]
    uncertainty_note = normalize_text(payload.get("uncertainty_note", ""))
    used_evidence_ids = payload.get("used_evidence_ids", [])

    if answer_mode == "report_lookup":
        title_variants = _doc_title_variants(selected_evidence)
        canonical_titles: list[str] = []
        matched_title = ""
        normalized_answer = normalize_for_match(final_answer)
        for variants in title_variants.values():
            canonical_title = variants[0]
            canonical_titles.append(canonical_title)
            if any(normalize_for_match(variant) and normalize_for_match(variant) in normalized_answer for variant in variants):
                matched_title = canonical_title
                break
        if matched_title:
            final_answer = matched_title
        else:
            cleaned_answer = normalize_text(final_answer).strip("《》“”\"' ")
            if cleaned_answer in canonical_titles:
                final_answer = cleaned_answer
            else:
                return fallback, "report_lookup_validation"
        return {
            "final_answer": final_answer,
            "evidence_summary": evidence_summary,
            "uncertainty_note": uncertainty_note,
            "used_evidence_ids": used_evidence_ids or [row["evidence_id"] for row in selected_evidence[:2]],
        }, None

    if answer_mode == "numeric_fact" and fact_subtype == "semantic_fact":
        preferred_answer = _preferred_numeric_semantic_answer(selected_evidence)
        if preferred_answer and _is_generic_numeric_semantic_query(query, fact_subtype=fact_subtype, answer_mode=answer_mode):
            final_answer = preferred_answer

    if answer_mode == "comparison":
        doc_focus_map = _coerce_doc_summary_map(payload.get("doc_focus_map", {}) or {}, field_name="focus")
        required_doc_ids = _required_doc_ids(selected_evidence)[:2]
        doc_focus_map = _ensure_doc_focus_map(required_doc_ids, selected_evidence, doc_focus_map)
        if any(not normalize_text(str(doc_focus_map.get(doc_id, ""))) for doc_id in required_doc_ids):
            return fallback, "comparison_validation"
        used_evidence_ids = list(dict.fromkeys(list(used_evidence_ids) + _default_doc_used_evidence_ids(required_doc_ids, selected_evidence)))
        comparison_subtype = infer_comparison_subtype(query)
        if _duplicate_short_titles(required_doc_ids, selected_evidence):
            comparison_subtype = "shared_logic_comparison"
        difference_dimension = normalize_text(payload.get("difference_dimension", ""))
        difference_detail = normalize_text(payload.get("difference_detail", ""))
        differences = [normalize_text(item) for item in payload.get("differences", []) if normalize_text(item)]
        if not difference_detail and differences:
            difference_detail = "；".join(differences[:2])
        if not difference_dimension and differences:
            difference_dimension = "关注重点"
        shared_points = [normalize_text(item) for item in payload.get("shared_points", []) if normalize_text(item)]
        if not shared_points:
            shared_points = [normalize_text(item) for item in payload.get("common_points", []) if normalize_text(item)]
        if not shared_points:
            shared_points = _common_terms_for_rows(selected_evidence, min_doc_frequency=2, top_k=3)
        conclusion = normalize_text(payload.get("conclusion", ""))
        if comparison_subtype == "shared_logic_comparison":
            candidate_answer = _shared_logic_comparison_answer(
                required_doc_ids=required_doc_ids,
                selected_evidence=selected_evidence,
                doc_focus_map=doc_focus_map,
                shared_points=shared_points,
                conclusion=conclusion,
            )
            combined_text = normalize_text("\n".join([candidate_answer, *shared_points, conclusion]))
        else:
            if not difference_detail:
                return fallback, "comparison_validation"
            candidate_answer = _compose_comparison_answer(
                query=query,
                required_doc_ids=required_doc_ids,
                selected_evidence=selected_evidence,
                doc_focus_map=doc_focus_map,
                difference_dimension=difference_dimension,
                difference_detail=difference_detail,
                shared_points=shared_points,
                conclusion=conclusion,
            )
            combined_text = normalize_text("\n".join([candidate_answer, difference_dimension, difference_detail, conclusion, *shared_points]))
        if not candidate_answer or not combined_text:
            return fallback, "comparison_validation"
        if not all(_has_focus_overlap(combined_text, doc_focus_map[doc_id]) for doc_id in required_doc_ids):
            return fallback, "comparison_validation"
        final_answer = candidate_answer

    if answer_mode == "inductive":
        required_doc_ids = _required_doc_ids(selected_evidence)
        per_doc_observation = _coerce_doc_summary_map(payload.get("per_doc_observation", {}) or {}, field_name="observation")
        per_doc_observation = _ensure_per_doc_observation(required_doc_ids, selected_evidence, per_doc_observation)
        observed_doc_ids = [doc_id for doc_id in required_doc_ids if normalize_text(str(per_doc_observation.get(doc_id, "")))]
        if len(observed_doc_ids) < 2:
            return fallback, "inductive_validation"
        theme_candidates = _normalize_theme_candidates(payload.get("theme_candidates", {}) or {})
        shared_themes = [normalize_text(item) for item in payload.get("shared_themes", []) if normalize_text(item)]
        if not shared_themes:
            shared_themes = _derive_shared_themes(
                observed_doc_ids=observed_doc_ids,
                theme_candidates=theme_candidates,
                selected_evidence=selected_evidence,
            )
        synthesis_basis = normalize_text(payload.get("synthesis_basis", ""))
        synthesis = normalize_text(payload.get("evidence_backed_synthesis", "") or payload.get("synthesis", ""))
        evidence_by_id = {row["evidence_id"]: row for row in selected_evidence}
        used_evidence_ids = list(dict.fromkeys(list(used_evidence_ids) + _default_doc_used_evidence_ids(observed_doc_ids, selected_evidence)))
        covered_docs = {
            evidence_by_id[evidence_id]["doc_id"]
            for evidence_id in used_evidence_ids
            if evidence_id in evidence_by_id
        }
        if len(covered_docs & set(observed_doc_ids)) < 2:
            return fallback, "inductive_validation"
        if len(shared_themes) < 1:
            return fallback, "inductive_validation"
        if not synthesis:
            synthesis = _compose_inductive_synthesis(
                observed_doc_ids=observed_doc_ids,
                per_doc_observation=per_doc_observation,
            )
        if not synthesis and not synthesis_basis:
            return fallback, "inductive_validation"
        if is_broad_inductive_query(query):
            doc_rows = _rows_by_doc(selected_evidence)
            diversity_keys = set()
            for doc_id in observed_doc_ids:
                rows = doc_rows.get(doc_id, [])
                if not rows:
                    continue
                diversity_keys.add(
                    _candidate_diversity_key(
                        {
                            "doc_id": doc_id,
                            "file_name": rows[0]["file_name"],
                            "short_title": _short_report_title(rows[0]["file_name"]),
                            "cleaned_topic_stem": clean_topic_stem(_title_topic_hint(rows[0]["file_name"])),
                            "cleaned_title_stem": clean_report_stem(_short_report_title(rows[0]["file_name"])),
                            "cluster_key": "",
                            "source_key": _report_source_key(rows[0]["file_name"]),
                            "topic_key": _title_topic_hint(rows[0]["file_name"]),
                        }
                    )
                )
            diversity_keys = {key for key in diversity_keys if key}
            if len(diversity_keys) < 2:
                return fallback, "inductive_validation"
        final_answer = _compose_inductive_answer(
            query=query,
            observed_doc_ids=observed_doc_ids,
            selected_evidence=selected_evidence,
            per_doc_observation=per_doc_observation,
            shared_themes=shared_themes,
            synthesis_basis=synthesis_basis,
            synthesis=synthesis,
        )
        normalized_final = normalize_for_match(final_answer)
        if _is_noise_topic_text(final_answer) or len(normalized_final) < 8:
            return fallback, "inductive_validation"
        combined_inductive_text = normalize_text("\n".join([final_answer, synthesis_basis, synthesis, *shared_themes]))
        observation_hits = sum(
            1
            for doc_id in observed_doc_ids
            if _has_focus_overlap(combined_inductive_text, normalize_text(str(per_doc_observation[doc_id])))
        )
        if observation_hits < 1:
            return fallback, "inductive_validation"
        if _is_single_evidence_rephrase(final_answer, selected_evidence):
            return fallback, "inductive_validation"

    return {
        "final_answer": final_answer,
        "evidence_summary": evidence_summary,
        "uncertainty_note": uncertainty_note,
        "used_evidence_ids": used_evidence_ids,
    }, None
