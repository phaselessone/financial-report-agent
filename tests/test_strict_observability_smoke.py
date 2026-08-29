"""End-to-end contract tests for the offline strict observability smoke."""

from __future__ import annotations

import shutil

from scripts.strict_observability_smoke import run_strict_observability_smoke
from src.evaluation.eval_bundle import (
    read_eval_bundle_reference,
    require_ready_eval_bundle_integrity,
)
from src.evaluation.run_metadata import build_source_manifest


def test_strict_smoke_runs_real_graph_and_writes_movable_ready_bundle(tmp_path) -> None:
    output_root = tmp_path / "strict-smoke"

    summary = run_strict_observability_smoke(output_root)

    assert summary["status"] == "READY"
    assert summary["evidence_scope"] == "SYNTHETIC_CONTRACT_ONLY"
    assert summary["publishable"] is False
    assert summary["external_network_used"] is False
    assert summary["model_download_used"] is False

    reference_path = output_root / summary["bundle_reference"]
    html_path = output_root / summary["observability_html"]
    summary_path = output_root / "strict_smoke_summary.json"
    assert reference_path.is_file()
    assert html_path.is_file()
    assert summary_path.is_file()

    bundle = read_eval_bundle_reference(reference_path)
    require_ready_eval_bundle_integrity(bundle)
    assert bundle.identity.case_count == 1
    assert bundle.identity.source_manifest_hash == build_source_manifest()["source_manifest_hash"]
    assert bundle.metadata["evidence_scope"] == "SYNTHETIC_CONTRACT_ONLY"
    assert bundle.metadata["publishable"] is False

    trace = bundle.trajectories[0]
    assert trace["failed"] is False
    assert trace["abstained"] is False
    assert trace["claims"]
    assert all(claim["verification"]["status"] == "ENTAILED" for claim in trace["claims"])
    assert trace["calculations"]
    calculation = next(iter(trace["calculations"].values()))
    assert calculation["operation"] == "yoy"
    assert calculation["status"] == "SUCCESS"
    assert calculation["result"]["formatted"] == "20.0000%"
    assert {item["evidence_id"] for item in calculation["inputs"]} == {
        "strict-smoke-revenue-2024",
        "strict-smoke-revenue-2025",
    }
    assert trace["end_to_end_latency_ms"] > 0
    assert trace["used_evidence_ids"]
    assert trace["citations"]
    assert set(trace["used_evidence_ids"]) == {
        evidence_id
        for claim in trace["claims"]
        for evidence_id in claim["evidence_ids"]
    }

    node_names = {event["node"] for event in trace["trajectory_events"]}
    assert {
        "analyze_query",
        "build_reasoning_plan",
        "execute_step",
        "observe_step_result",
        "dependency_gate",
        "synthesize",
        "extract_claims",
        "verify_answer",
        "finalize",
    }.issubset(node_names)

    html = html_path.read_text(encoding="utf-8")
    assert "Trace source and integrity" in html
    assert "RunIdentity" in html
    assert "Failure attribution" in html
    assert "Dependency detail" in html
    assert "Token and latency detail" in html

    moved_root = tmp_path / "moved-strict-smoke"
    shutil.move(str(output_root), moved_root)
    moved_bundle = read_eval_bundle_reference(moved_root / summary["bundle_reference"])
    require_ready_eval_bundle_integrity(moved_bundle)
    assert moved_bundle.identity == bundle.identity


def test_strict_smoke_can_explicitly_refresh_its_deterministic_artifact(tmp_path) -> None:
    output_root = tmp_path / "strict-smoke"
    first = run_strict_observability_smoke(output_root)
    second = run_strict_observability_smoke(output_root, overwrite=True)

    assert second["status"] == "READY"
    assert second["run_id"] == first["run_id"]
    assert second["identity_hash"] == first["identity_hash"]
