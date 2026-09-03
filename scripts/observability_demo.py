"""Render one strict agent eval bundle row as a standalone review page.

The default source is the immutable eval-bundle reference emitted by
``run_agent_eval.py``.  A raw legacy trace can still be inspected explicitly,
but it is always labelled unverified and exits with the blocked status code.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.observability import (  # noqa: E402
    DEFAULT_OUTPUT_PATH,
    build_observability_view,
    load_trace_rows,
    render_observability_html,
    select_trace_row,
)
from src.evaluation.eval_bundle import (  # noqa: E402
    EvalBundleIntegrityError,
    read_eval_bundle,
    read_eval_bundle_reference,
    require_ready_eval_bundle_integrity,
)
from scripts.evidence_integrity_gate import validate_bundle_evidence_integrity  # noqa: E402


DEFAULT_BUNDLE_PATH = Path("outputs/reports/agent_eval_bundle_dev.json")
DEFAULT_CHUNKS_PATH = Path("data/chunks/chunks.jsonl")


def _load_bundle(path: Path):
    return read_eval_bundle(path) if path.is_dir() else read_eval_bundle_reference(path)


def _evidence_id(item: Mapping[str, Any]) -> str:
    return str(item.get("evidence_id") or item.get("chunk_id") or "").strip()


def _entailed_evidence_ids(row: Mapping[str, Any]) -> set[str]:
    evidence_ids: set[str] = set()
    for claim in row.get("claims") or []:
        if not isinstance(claim, Mapping):
            continue
        verification = claim.get("verification") if isinstance(claim.get("verification"), Mapping) else {}
        if str(verification.get("status") or "").upper() != "ENTAILED":
            continue
        values = claim.get("evidence_ids") or []
        if isinstance(values, str):
            values = [values]
        evidence_ids.update(str(value).strip() for value in values if str(value).strip())
    return evidence_ids


def _review_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on displayed citations and expose every execution layer."""

    reviewed = dict(row)
    allowed = _entailed_evidence_ids(row)
    verified_evidence_rows = [
        dict(item)
        for item in (row.get("evidence_rows") or [])
        if isinstance(item, Mapping) and _evidence_id(item) in allowed
    ]
    evidence_by_id = {_evidence_id(item): item for item in verified_evidence_rows}
    reviewed["citations"] = [
        {**evidence_by_id.get(_evidence_id(item), {}), **dict(item)}
        for item in (row.get("citations") or [])
        if isinstance(item, Mapping) and _evidence_id(item) in allowed
    ]
    reviewed["evidence_rows"] = verified_evidence_rows

    result_by_id = {
        str(item.get("step_id") or ""): item
        for item in (row.get("reasoning_step_results") or [])
        if isinstance(item, Mapping)
    }
    steps: list[dict[str, Any]] = []
    reasoning_plan = row.get("reasoning_plan") if isinstance(row.get("reasoning_plan"), Mapping) else {}
    for step in reasoning_plan.get("steps") or []:
        if not isinstance(step, Mapping):
            continue
        step_id = str(step.get("step_id") or "")
        result = result_by_id.get(step_id, {})
        steps.append(
            {
                "node": f"Reasoning {step.get('kind') or 'step'}",
                "status": result.get("status") or "PLANNED",
                "tool_call_id": step_id,
                "latency_ms": result.get("latency_ms", ""),
            }
        )
    for call in row.get("tool_calls") or []:
        if not isinstance(call, Mapping):
            continue
        steps.append(
            {
                "node": f"Tool {call.get('tool_name') or call.get('name') or 'call'}",
                "status": call.get("status"),
                "tool_call_id": call.get("tool_call_id") or call.get("id"),
                "latency_ms": call.get("latency_ms", ""),
            }
        )
    coverage = row.get("reasoning_coverage") or row.get("dependency_coverage")
    if isinstance(coverage, Mapping) and coverage:
        steps.append(
            {
                "node": "Dependency coverage",
                "status": str(coverage.get("decision") or coverage.get("status") or "RECORDED").upper(),
                "tool_call_id": reasoning_plan.get("plan_id") or "",
                "latency_ms": "",
            }
        )
    steps.extend(dict(event) for event in (row.get("trajectory_events") or []) if isinstance(event, Mapping))
    reviewed["trajectory_events"] = steps
    return reviewed


def _json_block(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
    return f"<pre>{html.escape(encoded, quote=True)}</pre>"


def _supplement_html(
    *,
    identity: Mapping[str, Any] | None,
    run_id: str,
    identity_hash: str,
    source_type: str,
    integrity_report: Mapping[str, Any],
    row: Mapping[str, Any],
) -> str:
    status = str(integrity_report.get("status") or "BLOCKED")
    source = {
        "source_type": source_type,
        "integrity_status": status,
        "run_id": run_id,
        "identity_hash": identity_hash,
        "issues": integrity_report.get("issues") or [],
    }
    token_latency = {
        "prompt_tokens": row.get("prompt_tokens", 0),
        "completion_tokens": row.get("completion_tokens", 0),
        "total_tokens": row.get("total_tokens", 0),
        "api_latency_ms_total": row.get("api_latency_ms_total", 0),
        "end_to_end_latency_ms": row.get("end_to_end_latency_ms", 0),
    }
    return (
        '<div class="grid">'
        f'<section class="full"><h2>Trace source and integrity</h2>{_json_block(source)}</section>'
        f'<section class="full"><h2>RunIdentity</h2>{_json_block(identity or {})}</section>'
        f'<section><h2>Failure attribution</h2>{_json_block(row.get("failure_attribution") or {})}</section>'
        f'<section><h2>Dependency detail</h2>{_json_block(row.get("reasoning_coverage") or row.get("dependency_coverage") or {})}</section>'
        f'<section class="full"><h2>Token and latency detail</h2>{_json_block(token_latency)}</section>'
        "</div>"
    )


def _write_review_page(
    row: Mapping[str, Any],
    output: Path,
    *,
    identity: Mapping[str, Any] | None,
    run_id: str,
    identity_hash: str,
    source_type: str,
    integrity_report: Mapping[str, Any],
) -> Path:
    reviewed = _review_row(row)
    document = render_observability_html(build_observability_view(reviewed))
    supplement = _supplement_html(
        identity=identity,
        run_id=run_id,
        identity_hash=identity_hash,
        source_type=source_type,
        integrity_report=integrity_report,
        row=reviewed,
    )
    document = document.replace("</main></body></html>", f"{supplement}</main></body></html>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render one agent trace row as a local HTML observability demo.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--bundle-path", type=Path, help="Strict eval bundle directory or bundle-reference JSON.")
    source.add_argument("--trace-path", type=Path, help="Legacy/raw trace JSONL; always rendered as BLOCKED/unverified.")
    parser.add_argument("--chunks-path", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--question-id", default=None)
    args = parser.parse_args(argv)
    try:
        if args.trace_path is not None:
            row = select_trace_row(load_trace_rows(args.trace_path), question_id=args.question_id)
            integrity_report: dict[str, Any] = {
                "status": "BLOCKED",
                "issues": [
                    {
                        "code": "LEGACY_TRACE_UNVERIFIED",
                        "message": "raw trace is not protected by a strict eval bundle identity/hash",
                    }
                ],
            }
            output = _write_review_page(
                row,
                args.output,
                identity=None,
                run_id=str(row.get("run_id") or ""),
                identity_hash="",
                source_type="legacy_trace_unverified",
                integrity_report=integrity_report,
            )
            print(output)
            print("OBSERVABILITY_BLOCKED: legacy trace is unverified", file=sys.stderr)
            return 2

        bundle = _load_bundle(args.bundle_path or DEFAULT_BUNDLE_PATH)
        row = select_trace_row(bundle.trajectories, question_id=args.question_id)
        if row.get("run_id") != bundle.identity.run_id or row.get("run_identity") != bundle.identity.to_dict():
            raise ValueError("selected trajectory RunIdentity does not match the strict eval bundle")
        try:
            require_ready_eval_bundle_integrity(bundle)
        except EvalBundleIntegrityError as exc:
            integrity_report = {
                "status": "BLOCKED",
                "issues": [
                    {
                        "code": "PERSISTED_INTEGRITY_VERDICT_REQUIRED",
                        "message": str(exc),
                    }
                ],
            }
        else:
            # Recheck the live chunk asset as well as requiring the persisted,
            # bundle-bound verdict.  A transient recheck can invalidate READY,
            # but it can never manufacture READY for an unvalidated bundle.
            integrity_report = validate_bundle_evidence_integrity(bundle, args.chunks_path)
        output = _write_review_page(
            row,
            args.output,
            identity=bundle.identity.to_dict(),
            run_id=bundle.identity.run_id,
            identity_hash=bundle.identity.identity_hash,
            source_type="strict_eval_bundle",
            integrity_report=integrity_report,
        )
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(output)
    if integrity_report.get("status") != "READY":
        print("OBSERVABILITY_BLOCKED: evidence integrity errors found", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
