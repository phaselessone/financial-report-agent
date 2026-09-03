"""Run one offline graph case through the strict evidence/observability stack.

This is a deterministic contract smoke, not a quality benchmark.  It uses no
network, model download, private case, or reviewed label, and its artifacts are
therefore permanently marked ``SYNTHETIC_CONTRACT_ONLY`` and non-publishable.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_agent_eval import write_validated_eval_bundle  # noqa: E402
from scripts.observability_demo import main as render_observability_main  # noqa: E402
from src.agent.config import AgentConfig  # noqa: E402
from src.agent.graph import run_agentic_rag  # noqa: E402
from src.evaluation.eval_bundle import (  # noqa: E402
    require_ready_eval_bundle_integrity,
    write_eval_bundle_reference,
)
from src.evaluation.run_identity import (  # noqa: E402
    RunIdentity,
    canonical_sha256,
    hash_benchmark_rows,
    hash_case_ids,
)
from src.evaluation.run_metadata import build_source_manifest  # noqa: E402
from src.evaluation.trajectory_eval import (  # noqa: E402
    build_agent_process_summary,
    build_agent_trace_row,
)
from src.llm.types import LLMResponse  # noqa: E402
from src.structured.fact_store import FactStore  # noqa: E402
from src.structured.schema import FinancialFact, Period  # noqa: E402
from src.utils.io import ensure_dir, write_json, write_jsonl  # noqa: E402


SMOKE_SCOPE = "SYNTHETIC_CONTRACT_ONLY"
SMOKE_CASE_ID = "strict-observability-smoke-001"
SMOKE_QUERY = "甲公司2025年营业收入同比增长多少？"
SMOKE_CLAIM = "甲公司2025年营业收入同比增长20.0000%。"


def _smoke_chunks() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for year, value in ((2024, "100"), (2025, "120")):
        evidence_id = f"strict-smoke-revenue-{year}"
        text = f"甲公司{year}年营业收入为{value}元。"
        rows.append(
            {
                "chunk_id": evidence_id,
                "evidence_id": evidence_id,
                "doc_id": f"strict-smoke-report-{year}",
                "file_name": f"strict-smoke-report-{year}.pdf",
                "page": 1,
                "page_start": 1,
                "page_end": 1,
                "company": "甲公司",
                "metric": "revenue",
                "period": f"FY{year}",
                "value_type": "actual",
                "unit": "元",
                "text": text,
                "child_text": text,
                "support_span": text,
                "section_title": "营业收入",
                "section_path": "财务报表/营业收入",
                "chunk_type": "text",
                "element_type": "paragraph",
                "score": 3.0,
                "rerank_score": 3.0,
            }
        )
    return rows


def _smoke_facts(chunks: list[dict[str, Any]]) -> list[FinancialFact]:
    facts: list[FinancialFact] = []
    for chunk, year, value in zip(chunks, (2024, 2025), ("100", "120")):
        facts.append(
            FinancialFact(
                company="甲公司",
                metric="revenue",
                period=Period("FY", year),
                value_type="actual",
                value=Decimal(value),
                unit="元",
                doc_id=str(chunk["doc_id"]),
                page=1,
                evidence_id=str(chunk["evidence_id"]),
                raw_value=f"{value}元",
                source_span=str(chunk["support_span"]),
            )
        )
    return facts


def _retrieval_result(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [dict(chunk) for chunk in chunks]
    return {
        "query_mode": "fact",
        "numeric_query": False,
        "dense_rows": [],
        "bm25_rows": [],
        "hybrid_rows": rows,
        "rerank_rows": rows,
        "timings": {"offline_smoke_ms": 0.0},
    }


class _OfflineRuntime:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self._chunks = [dict(chunk) for chunk in chunks]
        self.calls: list[str] = []

    def search(self, query: str) -> dict[str, Any]:
        self.calls.append(query)
        return _retrieval_result(self._chunks)


class _OfflineAnswerer:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self._chunks = [dict(chunk) for chunk in chunks]
        self.llm_calls: list[LLMResponse] = []

    def answer(self, **kwargs: Any) -> dict[str, Any]:
        self.llm_calls.append(
            LLMResponse(
                content=SMOKE_CLAIM,
                provider="offline-deterministic",
                model="strict-smoke-answerer",
                prompt_tokens=4,
                completion_tokens=4,
            )
        )
        citations = [
            {
                "evidence_id": chunk["evidence_id"],
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "file_name": chunk["file_name"],
                "page": chunk["page"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "text": chunk["text"],
                "support_span": chunk["support_span"],
            }
            for chunk in self._chunks
        ]
        evidence_ids = [str(chunk["evidence_id"]) for chunk in self._chunks]
        return {
            "query": str(kwargs.get("query") or SMOKE_QUERY),
            "question_type": "fact",
            "answer_mode": "fact",
            "final_answer": SMOKE_CLAIM,
            "abstained": False,
            "abstain_reason": None,
            "used_evidence_ids": evidence_ids,
            "citations": citations,
            "support_validation": {
                "supported": True,
                "method": "offline_exact_match",
                "matched_numeric_tokens": [],
            },
            "selected_doc_ids": [str(chunk["doc_id"]) for chunk in self._chunks],
        }


class _OfflineClaimExtractor:
    def generate(self, **_: Any) -> LLMResponse:
        payload = {
            "claims": [
                {
                    "id": "claim-1",
                    "text": SMOKE_CLAIM,
                    "claim_type": "DERIVED",
                    "is_core": True,
                    "source_step_ids": ["calculate_yoy"],
                    "parent_claim_ids": [],
                }
            ]
        }
        return LLMResponse(
            content=json.dumps(payload, ensure_ascii=False),
            provider="offline-deterministic",
            model="strict-smoke-claim-extractor",
            prompt_tokens=5,
            completion_tokens=5,
        )


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else "git-unavailable"


def _source_manifest_hash() -> str:
    """Bind the smoke run to the same canonical source scope as every bundle."""

    return str(build_source_manifest(cwd=REPO_ROOT)["source_manifest_hash"])


def _budgets(config: AgentConfig) -> dict[str, int]:
    return {
        "max_steps": config.max_steps,
        "max_retrieval_rounds": config.max_retrieval_rounds,
        "max_query_rewrites": config.max_query_rewrites,
        "max_generation_attempts": config.max_generation_attempts,
        "max_llm_calls": config.max_llm_calls,
        "max_total_tokens": config.max_total_tokens,
        "max_failed_rewrite_rounds": config.max_failed_rewrite_rounds,
        "max_sub_questions": config.max_sub_questions,
        "claim_max_tokens": config.claim_max_tokens,
        "max_tool_calls": config.max_tool_calls,
        "max_empty_tool_results": config.max_empty_tool_results,
        "max_claim_retrievals": config.max_claim_retrievals,
        "rewrite_max_tokens": config.rewrite_max_tokens,
    }


def _identity(
    *,
    eval_row: dict[str, Any],
    chunks: list[dict[str, Any]],
    config: AgentConfig,
) -> RunIdentity:
    capabilities = config.resolved_execution_capabilities().to_dict()
    return RunIdentity(
        benchmark_profile="current-dev",
        benchmark_version="strict-observability-smoke-v1",
        benchmark_hash=hash_benchmark_rows([eval_row]),
        case_ids_hash=hash_case_ids([SMOKE_CASE_ID]),
        case_count=1,
        corpus_hash=canonical_sha256(chunks),
        model="strict-smoke-composite",
        provider="offline-deterministic",
        model_revision="strict-smoke-v1",
        temperature=0.0,
        prompt_version="strict-observability-smoke-v1",
        budgets=_budgets(config),
        feature_flags={
            "runtime": {
                "strict_trace": True,
                "external_network": False,
                "model_downloads": False,
                "synthetic_contract": True,
            },
            "treatments": {
                "pipeline": "agentic",
                "profile": "full-agent",
                **capabilities,
            },
        },
        git_commit=_git_commit(),
        source_manifest_hash=_source_manifest_hash(),
    )


def _bundle_config(identity: RunIdentity) -> dict[str, Any]:
    row = identity.to_dict()
    return {
        "benchmark_profile": identity.benchmark_profile,
        "benchmark_version": identity.benchmark_version,
        "provider": identity.provider,
        "model": identity.model,
        "model_revision": identity.model_revision,
        "temperature": identity.temperature,
        "prompt_version": identity.prompt_version,
        "budgets": row["budgets"],
        "runtime": row["feature_flags"]["runtime"],
        "treatments": row["feature_flags"]["treatments"],
        "evidence_scope": SMOKE_SCOPE,
        "publishable": False,
    }


def run_strict_observability_smoke(
    output_root: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Execute the real graph and emit one strict, movable review artifact."""

    output_root = Path(output_root).resolve()
    chunks = _smoke_chunks()
    eval_row = {
        "question_id": SMOKE_CASE_ID,
        "case_id": SMOKE_CASE_ID,
        "query": SMOKE_QUERY,
        "question_type": "fact",
        "intent": "derived_calculation",
        "industry": "other",
        "category": "strict_observability_contract",
    }
    config = AgentConfig(enable_tool_orchestration=True)
    identity = _identity(eval_row=eval_row, chunks=chunks, config=config)
    bundle_path = output_root / "eval" / identity.run_id
    if bundle_path.exists() and not overwrite:
        raise FileExistsError(f"strict smoke bundle already exists: {bundle_path}")

    ensure_dir(output_root)
    chunks_path = output_root / "strict_smoke_chunks.jsonl"
    write_jsonl(chunks_path, chunks)

    fact_store = FactStore(":memory:")
    fact_store.upsert(_smoke_facts(chunks))
    started = perf_counter()
    try:
        state = run_agentic_rag(
            runtime=_OfflineRuntime(chunks),
            answerer=_OfflineAnswerer(chunks),
            llm=_OfflineClaimExtractor(),
            config=config,
            query=SMOKE_QUERY,
            question_type="calculation",
            fact_store=fact_store,
            company_aliases={"甲公司": ["甲公司", "甲"]},
        )
    finally:
        fact_store.close()
    latency_ms = (perf_counter() - started) * 1000
    trace = build_agent_trace_row(
        state,
        eval_row,
        end_to_end_latency_ms=latency_ms,
        run_identity=identity,
        strict=True,
    )
    metrics = build_agent_process_summary([trace])
    metrics.update(
        {
            "evidence_scope": SMOKE_SCOPE,
            "publishable": False,
            "quality_metrics_emitted": False,
        }
    )
    bundle = write_validated_eval_bundle(
        output_root / "eval",
        identity=identity,
        config=_bundle_config(identity),
        metrics=metrics,
        per_case=[trace],
        trajectories=[trace],
        chunks_path=chunks_path,
        metadata={
            "mode": "agentic",
            "evidence_scope": SMOKE_SCOPE,
            "publishable": False,
            "purpose": "offline strict evidence and observability contract smoke",
        },
        overwrite=overwrite,
    )
    require_ready_eval_bundle_integrity(bundle)

    reference_path = output_root / "strict_smoke_bundle.json"
    write_eval_bundle_reference(reference_path, bundle)
    html_path = output_root / "strict_observability_demo.html"
    render_status = render_observability_main(
        [
            "--bundle-path",
            str(reference_path),
            "--chunks-path",
            str(chunks_path),
            "--output",
            str(html_path),
            "--question-id",
            SMOKE_CASE_ID,
        ]
    )
    if render_status != 0:
        raise RuntimeError(f"strict observability renderer exited with status {render_status}")

    summary = {
        "status": "READY",
        "evidence_scope": SMOKE_SCOPE,
        "publishable": False,
        "quality_claims_allowed": False,
        "external_network_used": False,
        "model_download_used": False,
        "run_id": identity.run_id,
        "identity_hash": identity.identity_hash,
        "bundle_reference": reference_path.relative_to(output_root).as_posix(),
        "chunks": chunks_path.relative_to(output_root).as_posix(),
        "observability_html": html_path.relative_to(output_root).as_posix(),
        "integrity_verdict": bundle.metadata["integrity_verdict"],
    }
    write_json(output_root / "strict_smoke_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the offline strict observability contract smoke."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/strict_observability_smoke"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Refresh the deterministic smoke bundle with the same RunIdentity.",
    )
    args = parser.parse_args(argv)
    summary = run_strict_observability_smoke(args.output_root, overwrite=args.overwrite)
    print(args.output_root / "strict_smoke_summary.json")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
