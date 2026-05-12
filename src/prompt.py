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
                "- `final_answer` should contain only the report name.",
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
