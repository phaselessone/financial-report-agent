from __future__ import annotations

from typing import Any

from src.utils.text_utils import first_sentence, normalize_text

SYSTEM_PROMPT = (
    "You are a financial research assistant. "
    "Answer with one JSON object only. "
    "Do not output markdown, code fences, or any extra text."
)

STRICT_JSON_SUFFIX = "Return valid JSON only."


def build_answer_prompt(
    *,
    query: str,
    question_type: str,
    answer_mode: str,
    evidence_rows: list[dict[str, Any]],
    numeric_query: bool = False,
    strict_json: bool = False,
) -> str:
    evidence_blocks = [format_evidence_block(row) for row in evidence_rows]
    candidate_titles = unique_report_titles(evidence_rows)
    schema = """{
  "final_answer": "Use concise Chinese.",
  "evidence_summary": "Briefly explain which evidence supports the answer in Chinese.",
  "uncertainty_note": "If there is uncertainty, explain it briefly in Chinese. Otherwise use an empty string.",
  "used_evidence_ids": ["E1", "E2"]
}"""

    instructions = _mode_instructions(answer_mode=answer_mode, question_type=question_type, numeric_query=numeric_query)
    prompt = f"""Question type: {question_type}
Answer mode: {answer_mode}
Question: {normalize_text(query)}

Candidate report titles:
{chr(10).join(f"- {title}" for title in candidate_titles) if candidate_titles else "- N/A"}

Evidence:
{chr(10).join(evidence_blocks)}

Instructions:
{instructions}

JSON Schema:
{schema}
"""
    if strict_json:
        prompt = f"{prompt}\n{STRICT_JSON_SUFFIX}"
    return prompt.strip()


def build_comparison_focus_prompt(*, query: str, evidence_rows: list[dict[str, Any]], strict_json: bool = False) -> str:
    evidence_blocks = [format_evidence_block(row) for row in evidence_rows]
    doc_registry = _doc_registry(evidence_rows)
    schema = """{
  "doc_focus_map": {
    "doc_id_1": "One concise focus summary for this report.",
    "doc_id_2": "One concise focus summary for this report."
  },
  "used_evidence_ids": ["E1", "E2"]
}"""
    prompt = f"""Question type: comparison
Question: {normalize_text(query)}

Document registry:
{chr(10).join(doc_registry)}

Evidence:
{chr(10).join(evidence_blocks)}

Instructions:
- Use Chinese.
- Fill `doc_focus_map` for every document id in the registry.
- Each focus should summarize the report's main emphasis in one sentence.
- `used_evidence_ids` must cover evidence from both reports.
- Do not output `final_answer` in this step.

JSON Schema:
{schema}
"""
    if strict_json:
        prompt = f"{prompt}\n{STRICT_JSON_SUFFIX}"
    return prompt.strip()


def build_comparison_synthesis_prompt(
    *,
    query: str,
    evidence_rows: list[dict[str, Any]],
    doc_focus_map: dict[str, str],
    strict_json: bool = False,
) -> str:
    evidence_blocks = [format_evidence_block(row) for row in evidence_rows]
    focus_lines = [f"- {doc_id}: {normalize_text(summary)}" for doc_id, summary in doc_focus_map.items()]
    schema = """{
  "shared_points": ["Point A", "Point B"],
  "difference_dimension": "One clear comparison dimension in Chinese.",
  "difference_detail": "A concrete difference sentence in Chinese that covers both reports.",
  "conclusion": "Short conclusion in Chinese.",
  "used_evidence_ids": ["E1", "E2"]
}"""
    prompt = f"""Question type: comparison
Question: {normalize_text(query)}

Document focus map:
{chr(10).join(focus_lines)}

Evidence:
{chr(10).join(evidence_blocks)}

Instructions:
- Use Chinese.
- `shared_points` should contain shared themes across the reports.
- `difference_dimension` must be one concrete comparison dimension, such as demand, pricing, policy, product focus, clinical theme, or industry logic.
- `difference_detail` must describe how the two reports differ on that dimension, not just say they are different.
- Do not output a free-form `final_answer`; return the structured fields above only.
- `used_evidence_ids` must cover at least one evidence row per report.

JSON Schema:
{schema}
"""
    if strict_json:
        prompt = f"{prompt}\n{STRICT_JSON_SUFFIX}"
    return prompt.strip()


def build_inductive_observation_prompt(*, query: str, evidence_rows: list[dict[str, Any]], strict_json: bool = False) -> str:
    evidence_blocks = [format_evidence_block(row) for row in evidence_rows]
    doc_registry = _doc_registry(evidence_rows)
    schema = """{
  "per_doc_observation": {
    "doc_id_1": "One concise observation for this report.",
    "doc_id_2": "One concise observation for this report."
  },
  "theme_candidates": {
    "doc_id_1": ["Theme A", "Theme B"],
    "doc_id_2": ["Theme A", "Theme C"]
  },
  "used_evidence_ids": ["E1", "E2", "E3"]
}"""
    prompt = f"""Question type: inductive
Question: {normalize_text(query)}

Document registry:
{chr(10).join(doc_registry)}

Evidence:
{chr(10).join(evidence_blocks)}

Instructions:
- Use Chinese.
- Fill `per_doc_observation` for every document id in the registry.
- Each observation should summarize that report's main observation in one sentence.
- Fill `theme_candidates` for every document id in the registry with 1-3 concrete theme phrases.
- `used_evidence_ids` should cover evidence from at least two reports.
- Do not output `final_answer` in this step.

JSON Schema:
{schema}
"""
    if strict_json:
        prompt = f"{prompt}\n{STRICT_JSON_SUFFIX}"
    return prompt.strip()


def build_inductive_synthesis_prompt(
    *,
    query: str,
    evidence_rows: list[dict[str, Any]],
    per_doc_observation: dict[str, str],
    theme_candidates: dict[str, list[str]] | None = None,
    strict_json: bool = False,
) -> str:
    evidence_blocks = [format_evidence_block(row) for row in evidence_rows]
    observation_lines = [f"- {doc_id}: {normalize_text(summary)}" for doc_id, summary in per_doc_observation.items()]
    theme_candidate_lines = [
        f"- {doc_id}: {', '.join(normalize_text(item) for item in items if normalize_text(item))}"
        for doc_id, items in (theme_candidates or {}).items()
        if any(normalize_text(item) for item in items)
    ]
    schema = """{
  "shared_themes": ["Theme A", "Theme B"],
  "synthesis_basis": "Explain in Chinese which observations support the synthesis.",
  "evidence_backed_synthesis": "Short synthesis in Chinese grounded in multiple reports.",
  "used_evidence_ids": ["E1", "E2", "E3"]
}"""
    prompt = f"""Question type: inductive
Question: {normalize_text(query)}

Per-document observations:
{chr(10).join(observation_lines)}

Theme candidates:
{chr(10).join(theme_candidate_lines) if theme_candidate_lines else "- N/A"}

Evidence:
{chr(10).join(evidence_blocks)}

Instructions:
- Use Chinese.
- `shared_themes` should contain the strongest shared themes supported by multiple reports; one strong theme is acceptable if it is clearly grounded.
- `synthesis_basis` should explain which observations support the synthesis.
- `evidence_backed_synthesis` should summarize the shared direction without copying one evidence block verbatim.
- Do not output a free-form `final_answer`; return the structured fields above only.
- `used_evidence_ids` should cover evidence from at least two reports.

JSON Schema:
{schema}
"""
    if strict_json:
        prompt = f"{prompt}\n{STRICT_JSON_SUFFIX}"
    return prompt.strip()


def _mode_instructions(*, answer_mode: str, question_type: str, numeric_query: bool) -> str:
    shared = [
        "- Use Chinese in `final_answer` and `evidence_summary`.",
        "- Only rely on the provided evidence.",
        "- `used_evidence_ids` must reference the evidence ids that directly support the answer.",
        "- If the evidence is incomplete, explain the uncertainty in `uncertainty_note` instead of inventing facts.",
    ]
    if answer_mode == "report_lookup":
        shared.extend(
            [
                "- `final_answer` should contain only one report name and nothing else.",
                "- Prefer copying the exact report title from the candidate report titles above.",
                "- `evidence_summary` should mention the report name and page number, for example: 《报告名》P1 提到……",
                "- Do not compare across reports in this mode.",
            ]
        )
    elif answer_mode == "comparison":
        shared.extend(
            [
                "- Explicitly mention the comparison objects, the comparison dimension, and the comparison direction.",
                "- Use evidence from at least two different reports.",
                "- Do not introduce numbers unless they can be found in the evidence.",
            ]
        )
    elif answer_mode == "inductive":
        shared.extend(
            [
                "- First state the shared themes or common points, then give a short synthesis.",
                "- Mention at least two report names in the answer or evidence summary.",
                "- The answer must be a synthesis, not a verbatim copy of one evidence block.",
                "- Keep the answer grounded in the provided evidence.",
            ]
        )
    else:
        shared.extend(
            [
                "- Answer the question directly and keep the wording concise.",
                "- Prefer the strongest 1-2 evidence sources from the same report.",
            ]
        )
    if question_type == "fact":
        shared.append("- For fact questions, avoid broad summaries and focus on the target report or target statement.")
    if numeric_query:
        shared.append("- If you mention a number, that number must appear in the evidence or be an equivalent normalized form.")
    return "\n".join(shared)


def format_evidence_block(row: dict[str, Any]) -> str:
    section_title = row.get("section_title") or "N/A"
    score = row.get("rerank_score", row.get("score", 0.0))
    snippet_source = row.get("support_span") or row.get("child_text") or row.get("text", "")
    snippet = normalize_text(snippet_source)
    if len(snippet) > 320:
        snippet = snippet[:320].rstrip() + "..."
    summary = first_sentence(snippet, max_chars=120)
    bundle_id = row.get("bundle_id") or row.get("chunk_id")
    support_type = row.get("support_type") or row.get("chunk_type")
    return (
        f"{row['evidence_id']} | file: {row.get('file_name')} | pages: {row.get('page_start')}-{row.get('page_end')} "
        f"| section: {section_title} | chunk_type: {row.get('chunk_type')} | element_type: {row.get('element_type')} "
        f"| bundle_id: {bundle_id} | support_type: {support_type} | score: {float(score):.4f}\n"
        f"summary: {summary}\n"
        f"snippet:\n{snippet}"
    )


def unique_report_titles(evidence_rows: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    titles: list[str] = []
    for row in evidence_rows:
        file_name = normalize_text(str(row.get("file_name", "")))
        if not file_name or file_name in seen:
            continue
        seen.add(file_name)
        titles.append(file_name)
    return titles


def _doc_registry(evidence_rows: list[dict[str, Any]]) -> list[str]:
    rows_by_doc: dict[str, str] = {}
    for row in evidence_rows:
        rows_by_doc.setdefault(str(row.get("doc_id", "")), normalize_text(str(row.get("file_name", ""))))
    return [f"- {doc_id}: {title}" for doc_id, title in rows_by_doc.items() if doc_id and title]
