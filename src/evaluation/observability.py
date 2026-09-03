"""Build a small, dependency-free HTML view for one agent trace row.

The renderer is intentionally read-only: it consumes an existing JSONL trace and
does not invoke a model, retrieval runtime, browser, or external service.  This
keeps the demo useful on a laptop while making the same process, provenance, and
cost fields visible during a review.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_TRACE_PATH = Path("outputs/reports/agent_traces_dev.jsonl")
DEFAULT_OUTPUT_PATH = Path("outputs/observability/agent_trace_demo.html")


def load_trace_rows(path: Path) -> list[dict[str, Any]]:
    """Read JSONL trace rows, rejecting malformed or non-object records."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def select_trace_row(rows: Iterable[Mapping[str, Any]], question_id: str | None = None) -> dict[str, Any]:
    """Select one trace row, preserving source order for deterministic output."""
    materialized = [dict(row) for row in rows if isinstance(row, Mapping)]
    if not materialized:
        raise ValueError("trace contains no rows")
    if question_id:
        for row in materialized:
            if str(row.get("question_id") or "") == question_id:
                return row
        raise KeyError(f"question_id not found in trace: {question_id}")
    return materialized[0]


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _text(value: Any, default: str = "") -> str:
    return str(value).strip() if value is not None else default


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def _evidence_rows(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize citation-shaped evidence without inventing missing metadata."""
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    candidates = _as_list(row.get("citations"))
    candidates.extend(_as_list(row.get("evidence_rows")))
    for item in candidates:
        if not isinstance(item, Mapping):
            continue
        evidence_id = _text(item.get("evidence_id") or item.get("chunk_id"))
        dedupe_key = evidence_id or _text(item.get("doc_id")) + ":" + _text(item.get("page"))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        result.append(
            {
                "evidence_id": evidence_id,
                "doc_id": _text(item.get("doc_id") or item.get("file_name") or item.get("source"), "-"),
                "page": item.get("page", item.get("page_number", "-")),
                "section": _text(item.get("section_title") or item.get("section")),
                "text": _text(item.get("text") or item.get("content") or item.get("support_span") or item.get("child_text")),
                "score": item.get("score", item.get("rerank_score")),
            }
        )
    return result


def build_observability_view(row: Mapping[str, Any]) -> dict[str, Any]:
    """Project a raw trace row into the stable fields shown by the demo."""
    claims: list[dict[str, Any]] = []
    for item in _as_list(row.get("claims")):
        if not isinstance(item, Mapping):
            continue
        verification = item.get("verification") if isinstance(item.get("verification"), Mapping) else {}
        claims.append(
            {
                "claim_id": _text(item.get("claim_id"), "-"),
                "text": _text(item.get("text")),
                "claim_type": _text(item.get("claim_type"), "EXTRACTED"),
                "status": _text(verification.get("status") or ("ENTAILED" if item.get("supported") else "INSUFFICIENT")),
                "method": _text(verification.get("method"), "-"),
                "evidence_ids": [str(value) for value in _as_list(item.get("evidence_ids"))],
                "calculation_id": _text(item.get("calculation_id")),
            }
        )

    calculations = row.get("calculations")
    if not isinstance(calculations, Mapping):
        calculations = {}
    steps = _as_list(row.get("trajectory_events"))
    if not steps:
        steps = _as_list(row.get("tool_calls"))
    return {
        "question_id": _text(row.get("question_id"), "-"),
        "query": _text(row.get("query")),
        "answer": _text(row.get("final_answer") or row.get("answer")),
        "evidence_summary": _text(row.get("evidence_summary")),
        "abstained": bool(row.get("abstained", False)),
        "abstain_reason": _text(row.get("abstain_reason")),
        "termination_reason": _text(row.get("termination_reason"), "-"),
        "failed": bool(row.get("failed", False)),
        "error": _text(row.get("error_message")),
        "question_type": _text(row.get("question_type"), "-"),
        "industry": _text(row.get("industry"), "-"),
        "claims": claims,
        "evidence": _evidence_rows(row),
        "calculations": dict(calculations),
        "steps": steps,
        "metrics": {
            "steps": row.get("step_count", 0),
            "retrieval_calls": row.get("retrieval_count", 0),
            "rewrite_calls": row.get("rewrite_count", 0),
            "generation_calls": row.get("generation_count", 0),
            "tool_calls": row.get("tool_call_count", 0),
            "llm_calls": row.get("llm_call_count", 0),
            "prompt_tokens": row.get("prompt_tokens", 0),
            "completion_tokens": row.get("completion_tokens", 0),
            "total_tokens": row.get("total_tokens", 0),
            "latency_ms": row.get("end_to_end_latency_ms", 0),
            "api_latency_ms": row.get("api_latency_ms_total", 0),
        },
    }


def _esc(value: Any) -> str:
    return html.escape(_text(value), quote=True)


def _badge(label: str, value: Any, tone: str = "") -> str:
    class_name = "badge" + (f" {tone}" if tone else "")
    return f'<div class="{class_name}"><span>{_esc(label)}</span><strong>{_esc(value)}</strong></div>'


def _table(headers: list[str], rows: list[list[Any]], empty: str) -> str:
    if not rows:
        return f'<p class="muted">{_esc(empty)}</p>'
    head = "".join(f"<th>{_esc(value)}</th>" for value in headers)
    body = "".join("<tr>" + "".join(f"<td>{_esc(value) or '&mdash;'}</td>" for value in values) + "</tr>" for values in rows)
    return f"<div class=\"table-wrap\"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def render_observability_html(view: Mapping[str, Any]) -> str:
    """Render a complete standalone HTML document from ``build_observability_view``."""
    claims = list(view.get("claims") or [])
    evidence = list(view.get("evidence") or [])
    calculations = view.get("calculations") if isinstance(view.get("calculations"), Mapping) else {}
    steps = list(view.get("steps") or [])
    metrics = view.get("metrics") if isinstance(view.get("metrics"), Mapping) else {}
    status = "FAILED" if view.get("failed") else ("ABSTAINED" if view.get("abstained") else "COMPLETED")
    status_tone = "bad" if status in {"FAILED", "ABSTAINED"} else "good"
    metric_badges = "".join(
        _badge(label.replace("_", " "), metrics.get(key, 0))
        for key, label in (
            ("steps", "Steps"),
            ("retrieval_calls", "Retrieval"),
            ("tool_calls", "Tools"),
            ("total_tokens", "Tokens"),
            ("latency_ms", "Latency ms"),
        )
    )
    claim_rows = [
        [item.get("claim_id"), item.get("status"), item.get("claim_type"), item.get("text"), ", ".join(item.get("evidence_ids") or [])]
        for item in claims
    ]
    evidence_rows = [[item.get("evidence_id"), item.get("doc_id"), item.get("page"), item.get("section"), item.get("text")] for item in evidence]
    step_rows = []
    for index, step in enumerate(steps, start=1):
        if isinstance(step, Mapping):
            step_rows.append([index, step.get("node") or step.get("action") or step.get("name"), step.get("status"), step.get("claim_id") or step.get("tool_call_id"), step.get("latency_ms", "")])
        else:
            step_rows.append([index, step, "", "", ""])
    calculation_block = _esc(_json(calculations)) if calculations else "No calculation provenance recorded."
    error_block = f'<div class="notice bad">{_esc(view.get("error"))}</div>' if view.get("error") else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Agent trace {_esc(view.get('question_id'))}</title>
  <style>
    :root {{ color-scheme: light; --ink:#17202a; --muted:#64707d; --line:#d9e0e6; --panel:#fff; --wash:#f4f7f9; --good:#16734b; --bad:#a83232; --accent:#1f5f8b; }}
    * {{ box-sizing:border-box; }} body {{ margin:0; background:var(--wash); color:var(--ink); font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }}
    main {{ max-width:1180px; margin:0 auto; padding:28px 20px 56px; }} h1,h2 {{ margin:0 0 12px; line-height:1.2; }} h1 {{ font-size:26px; }} h2 {{ font-size:17px; border-bottom:1px solid var(--line); padding-bottom:8px; }}
    .eyebrow {{ color:var(--muted); font-size:12px; letter-spacing:.04em; text-transform:uppercase; }} .header {{ display:flex; align-items:flex-start; justify-content:space-between; gap:20px; margin-bottom:18px; }}
    .status {{ border:1px solid var(--line); padding:7px 11px; font-weight:700; font-size:12px; color:var(--good); background:#edf8f2; }} .status.bad {{ color:var(--bad); background:#fff0f0; }}
    .question {{ background:var(--panel); border:1px solid var(--line); padding:16px; margin-bottom:16px; }} .question p {{ margin:5px 0 0; font-size:18px; }}
    .badges {{ display:flex; flex-wrap:wrap; gap:8px; margin:14px 0 20px; }} .badge {{ min-width:104px; border:1px solid var(--line); background:var(--panel); padding:8px 10px; }} .badge span {{ display:block; color:var(--muted); font-size:11px; text-transform:uppercase; }} .badge strong {{ display:block; margin-top:2px; font-size:17px; }}
    .grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }} section {{ background:var(--panel); border:1px solid var(--line); padding:16px; min-width:0; margin-bottom:16px; }} section.full {{ grid-column:1/-1; }}
    .answer {{ white-space:pre-wrap; font-size:16px; }} .muted {{ color:var(--muted); }} .notice {{ padding:10px 12px; border-left:3px solid var(--bad); background:#fff4f4; margin:10px 0; }}
    .table-wrap {{ overflow:auto; }} table {{ width:100%; border-collapse:collapse; min-width:560px; }} th,td {{ border-bottom:1px solid var(--line); padding:8px; text-align:left; vertical-align:top; }} th {{ color:var(--muted); font-size:11px; text-transform:uppercase; white-space:nowrap; }} td {{ max-width:420px; white-space:pre-wrap; }}
    pre {{ margin:0; padding:12px; overflow:auto; background:#f7f9fb; border:1px solid var(--line); font:12px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace; }}
    @media (max-width:760px) {{ main {{ padding:20px 12px 42px; }} .header {{ display:block; }} .status {{ display:inline-block; margin-top:10px; }} .grid {{ grid-template-columns:1fr; }} section.full {{ grid-column:auto; }} }}
  </style>
</head>
<body><main>
  <div class="header"><div><div class="eyebrow">Financial RAG observability</div><h1>Agent trace {_esc(view.get('question_id'))}</h1><div class="muted">{_esc(view.get('question_type'))} · {_esc(view.get('industry'))}</div></div><div class="status {status_tone}">{_esc(status)}</div></div>
  <div class="question"><div class="eyebrow">Question</div><p>{_esc(view.get('query'))}</p></div>
  <div class="badges">{metric_badges}</div>
  {error_block}
  <div class="grid">
    <section><h2>Answer</h2><div class="answer">{_esc(view.get('answer')) or '<span class="muted">No answer recorded.</span>'}</div>{f'<p class="muted">Abstain reason: {_esc(view.get("abstain_reason"))}</p>' if view.get('abstain_reason') else ''}<p class="muted">Termination: {_esc(view.get('termination_reason'))}</p></section>
    <section><h2>Claims</h2>{_table(['ID','Status','Type','Claim','Evidence'], claim_rows, 'No claims recorded.')}</section>
    <section class="full"><h2>Evidence and pages</h2>{_table(['Evidence ID','Document','Page','Section','Text'], evidence_rows, 'No citation evidence recorded.')}</section>
    <section><h2>Calculation trace</h2><pre>{calculation_block}</pre></section>
    <section><h2>Agent steps</h2>{_table(['#','Node / action','Status','ID','Latency ms'], step_rows, 'No trajectory steps recorded.')}</section>
  </div>
</main></body></html>
"""


def write_observability_demo(
    trace_path: Path = DEFAULT_TRACE_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    *,
    question_id: str | None = None,
) -> Path:
    """Render one trace row to ``output_path`` and return the written path."""
    row = select_trace_row(load_trace_rows(trace_path), question_id=question_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_observability_html(build_observability_view(row)), encoding="utf-8")
    return output_path
