"""Current-dev strict trace built from real corpus-shaped assets."""

from __future__ import annotations

import shutil
from decimal import Decimal

from scripts.regenerate_dev_trace import (
    DEV_EVIDENCE_REPORT_NAME,
    DEV_REFERENCE_NAME,
    DEV_SCOPE,
    run_current_dev_trace,
)
from src.evaluation.eval_bundle import (
    read_eval_bundle_reference,
    require_ready_eval_bundle_integrity,
)
from src.structured.fact_store import FactStore
from src.structured.schema import FinancialFact, Period
from src.utils.io import read_json, write_json, write_jsonl


def _fact(metric: str, value: str, evidence_id: str) -> FinancialFact:
    return FinancialFact(
        company="示例制造",
        metric=metric,
        period=Period("FY", 2025),
        value_type="actual",
        value=Decimal(value),
        unit="元",
        doc_id=f"report-{evidence_id}",
        page=8,
        evidence_id=evidence_id,
        raw_value=f"{value}元",
        source_span=f"示例制造2025年{metric}为{value}元。",
    )


def test_current_dev_trace_is_real_asset_bound_strict_and_movable(tmp_path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    facts_path = tmp_path / "facts.duckdb"
    aliases_path = tmp_path / "aliases.json"
    output_root = tmp_path / "outputs"
    facts = [
        _fact("net_profit", "20", "E-NP"),
        _fact("revenue", "100", "E-REV"),
    ]
    write_jsonl(
        chunks_path,
        [
            {
                "chunk_id": fact.evidence_id,
                "doc_id": fact.doc_id,
                "page_start": fact.page,
                "page_end": fact.page,
                "text": fact.source_span,
                "support_span": fact.source_span,
            }
            for fact in facts
        ],
    )
    store = FactStore(facts_path)
    try:
        store.upsert(facts)
    finally:
        store.close()
    write_json(aliases_path, {"示例制造": ["示例制造"]})

    legacy_path = output_root / "reports" / "agent_traces_dev.jsonl"
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_text("legacy-blocked-trace\n", encoding="utf-8")

    summary = run_current_dev_trace(
        output_root,
        chunks_path=chunks_path,
        facts_path=facts_path,
        aliases_path=aliases_path,
    )

    assert summary["status"] == "READY"
    assert summary["evidence_scope"] == DEV_SCOPE
    assert summary["publishable"] is False
    assert summary["quality_claims_allowed"] is False
    assert summary["retrieval_calls"] == 0
    assert legacy_path.read_text(encoding="utf-8") == "legacy-blocked-trace\n"

    reference_path = output_root / "reports" / DEV_REFERENCE_NAME
    bundle = read_eval_bundle_reference(reference_path)
    require_ready_eval_bundle_integrity(bundle)
    assert bundle.identity.benchmark_profile == "current-dev"
    assert bundle.identity.corpus_hash == bundle.metadata["corpus_asset_manifest"]["corpus_hash"]
    assert bundle.metadata["evidence_scope"] == DEV_SCOPE
    assert bundle.metadata["publishable"] is False
    trace = bundle.trajectories[0]
    assert trace["failed"] is False
    assert trace["abstained"] is False
    assert trace["end_to_end_latency_ms"] > 0
    assert trace["claims"]
    assert all(item["verification"]["status"] == "ENTAILED" for item in trace["claims"])
    calculation = next(iter(trace["calculations"].values()))
    assert calculation["operation"] == "net_margin"
    assert calculation["status"] == "SUCCESS"
    assert calculation["result"]["formatted"] == "20.0000%"
    assert {item["evidence_id"] for item in calculation["inputs"]} == {"E-NP", "E-REV"}
    assert not any(
        item.get("tool_name") == "report_search" for item in trace["tool_calls"]
    )

    report = read_json(output_root / "reports" / DEV_EVIDENCE_REPORT_NAME)
    assert report["status"] == "READY"
    assert report["bundle"]["run_id"] == bundle.identity.run_id

    moved_root = tmp_path / "moved-outputs"
    shutil.move(str(output_root), moved_root)
    moved = read_eval_bundle_reference(moved_root / "reports" / DEV_REFERENCE_NAME)
    require_ready_eval_bundle_integrity(moved)
    assert moved.identity == bundle.identity
