"""Regenerate one current-dev strict trace from current-local real corpus assets.

This is a reproducibility and evidence-integrity contract, not a reviewed
quality benchmark.  It selects one deterministic net-margin case from the
actual structured-fact snapshot, resolves its evidence in the actual chunks
asset, runs the production graph offline, and publishes an identity-bound
strict bundle.  The legacy BLOCKED trace is deliberately left untouched.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agent.config import AgentConfig  # noqa: E402
from src.agent.graph import run_agentic_rag  # noqa: E402
from src.evaluation.eval_bundle import (  # noqa: E402
    build_corpus_asset_manifest,
    require_ready_eval_bundle_integrity,
    write_eval_bundle_reference,
    write_validated_eval_bundle,
)
from src.evaluation.evidence_integrity import (  # noqa: E402
    validate_bundle_evidence_integrity,
)
from src.evaluation.run_identity import (  # noqa: E402
    RunIdentity,
    hash_benchmark_rows,
    hash_case_ids,
)
from src.evaluation.run_metadata import (  # noqa: E402
    build_source_manifest,
    detect_git_sha,
)
from src.evaluation.trajectory_eval import (  # noqa: E402
    build_agent_process_summary,
    build_agent_trace_row,
)
from src.llm.types import LLMResponse  # noqa: E402
from src.structured.fact_store import FactStore  # noqa: E402
from src.structured.schema import FinancialFact  # noqa: E402
from src.tools.financial_calculator import net_margin  # noqa: E402
from src.utils.io import ensure_dir, read_json, write_json  # noqa: E402


DEV_SCOPE = "REAL_CORPUS_DEV_CONTRACT_ONLY"
DEV_CASE_ID = "current-dev-real-corpus-net-margin-001"
DEV_REFERENCE_NAME = "agent_eval_bundle_dev.json"
DEV_EVIDENCE_REPORT_NAME = "evidence_integrity_dev_current.json"
DEV_SUMMARY_NAME = "current_dev_trace_summary.json"
DEV_SOURCE_MANIFEST_NAME = "current_dev_source_manifest.json"
_REQUIRED_FACT_COLUMNS = {
    "fact_id",
    "company",
    "metric",
    "period_kind",
    "year",
    "value_type",
    "value",
    "unit",
    "doc_id",
    "page",
    "evidence_id",
    "raw_value",
    "source_span",
}
_OPTIONAL_FACT_COLUMNS = (
    "accounting_scope",
    "period_basis",
    "source_date",
    "revision_status",
)


class DevTraceAssetError(ValueError):
    """Raised when the real corpus cannot provide one unambiguous dev case."""


@dataclass(frozen=True)
class SelectedDevCase:
    company: str
    year: int
    net_profit: FinancialFact
    revenue: FinancialFact
    chunks: tuple[dict[str, Any], ...]
    aliases: dict[str, list[str]]
    expected_answer: str

    @property
    def query(self) -> str:
        return f"{self.company}{self.year}年净利润/营业收入是多少？"


def _load_aliases(path: Path) -> dict[str, list[str]]:
    raw = read_json(path)
    if not isinstance(raw, Mapping):
        raise DevTraceAssetError("company aliases asset must be a JSON object")
    aliases: dict[str, list[str]] = {}
    for company, values in raw.items():
        canonical = str(company or "").strip()
        if not canonical or not isinstance(values, list):
            continue
        normalized = [str(item).strip() for item in values if str(item).strip()]
        if canonical not in normalized:
            normalized.insert(0, canonical)
        aliases[canonical] = list(dict.fromkeys(normalized))
    if not aliases:
        raise DevTraceAssetError("company aliases asset has no usable entries")
    return aliases


def _load_candidate_facts(path: Path) -> list[FinancialFact]:
    connection = duckdb.connect(str(path), read_only=True)
    try:
        tables = {str(row[0]) for row in connection.execute("SHOW TABLES").fetchall()}
        if "facts" not in tables:
            raise DevTraceAssetError("structured facts snapshot is missing table 'facts'")
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info('facts')").fetchall()
        }
        missing = sorted(_REQUIRED_FACT_COLUMNS - columns)
        if missing:
            raise DevTraceAssetError(
                "structured facts snapshot is missing columns: " + ", ".join(missing)
            )
        selected_columns = [*sorted(_REQUIRED_FACT_COLUMNS), *_OPTIONAL_FACT_COLUMNS]
        expressions = [
            name if name in columns else f"NULL AS {name}"
            for name in selected_columns
        ]
        cursor = connection.execute(
            "SELECT "
            + ", ".join(expressions)
            + " FROM facts "
            + "WHERE metric IN ('net_profit', 'revenue') "
            + "AND period_kind = 'FY' AND value_type = 'actual' "
            + "ORDER BY company, year, metric, fact_id"
        )
        names = [str(item[0]) for item in cursor.description]
        rows = [dict(zip(names, values)) for values in cursor.fetchall()]
    finally:
        connection.close()
    facts: list[FinancialFact] = []
    for row in rows:
        facts.append(
            FinancialFact.from_dict(
                {
                    **row,
                    "period": {
                        "kind": str(row.pop("period_kind")),
                        "year": int(row.pop("year")),
                    },
                }
            )
        )
    return facts


def _chunk_identifier(row: Mapping[str, Any]) -> str:
    return str(row.get("chunk_id") or row.get("evidence_id") or row.get("id") or "").strip()


def _load_matching_chunks(path: Path, evidence_ids: set[str]) -> dict[str, dict[str, Any]]:
    matches: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DevTraceAssetError(
                    f"chunks asset contains invalid JSON at line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(row, Mapping):
                raise DevTraceAssetError(
                    f"chunks asset row {line_number} must be a JSON object"
                )
            chunk_id = _chunk_identifier(row)
            if chunk_id in evidence_ids and chunk_id not in matches:
                matches[chunk_id] = dict(row)
    return matches


def _select_dev_case(
    *,
    chunks_path: Path,
    facts_path: Path,
    aliases_path: Path,
) -> SelectedDevCase:
    aliases = _load_aliases(aliases_path)
    grouped: dict[tuple[str, int], dict[str, list[FinancialFact]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for fact in _load_candidate_facts(facts_path):
        if fact.company in aliases:
            grouped[(fact.company, fact.period.year)][fact.metric.value].append(fact)

    candidates: list[tuple[FinancialFact, FinancialFact]] = []
    for key in sorted(grouped):
        metrics = grouped[key]
        net_profits = metrics.get("net_profit", [])
        revenues = metrics.get("revenue", [])
        # The graph must resolve each structured lookup to exactly one fact.
        if len(net_profits) != 1 or len(revenues) != 1:
            continue
        try:
            revenue_value = Decimal(str(revenues[0].value))
        except (InvalidOperation, ValueError):
            continue
        if revenue_value > 0:
            candidates.append((net_profits[0], revenues[0]))
    if not candidates:
        raise DevTraceAssetError(
            "no unambiguous actual FY net_profit/revenue pair exists in structured facts"
        )

    evidence_ids = {
        fact.evidence_id for pair in candidates for fact in pair if fact.evidence_id
    }
    chunks = _load_matching_chunks(chunks_path, evidence_ids)
    selected = next(
        (
            pair
            for pair in candidates
            if all(fact.evidence_id in chunks for fact in pair)
        ),
        None,
    )
    if selected is None:
        raise DevTraceAssetError(
            "no unambiguous structured fact pair has resolvable real chunk evidence"
        )
    net_profit_fact, revenue_fact = selected
    result = net_margin(net_profit_fact.value, revenue_fact.value, precision=4)
    answer = (
        f"{net_profit_fact.company}{net_profit_fact.period.year}年"
        f"净利率为{result.formatted}。"
    )
    ordered_chunks: list[dict[str, Any]] = []
    for evidence_id in dict.fromkeys(
        (net_profit_fact.evidence_id, revenue_fact.evidence_id)
    ):
        row = dict(chunks[evidence_id])
        row.setdefault("chunk_id", evidence_id)
        row.setdefault("evidence_id", evidence_id)
        if row.get("page") in (None, ""):
            row["page"] = row.get("page_start") or row.get("page_end")
        ordered_chunks.append(row)
    return SelectedDevCase(
        company=net_profit_fact.company,
        year=net_profit_fact.period.year,
        net_profit=net_profit_fact,
        revenue=revenue_fact,
        chunks=tuple(ordered_chunks),
        aliases=aliases,
        expected_answer=answer,
    )


class _NoRetrievalRuntime:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def search(self, query: str) -> dict[str, Any]:
        self.calls.append(str(query))
        raise AssertionError(
            "current-dev structured case unexpectedly entered report-search retrieval"
        )


class _DeterministicDevAnswerer:
    def __init__(self, selected: SelectedDevCase) -> None:
        self._selected = selected
        self.llm_calls: list[LLMResponse] = []

    def answer(self, **kwargs: Any) -> dict[str, Any]:
        self.llm_calls.append(
            LLMResponse(
                content=self._selected.expected_answer,
                provider="offline-deterministic",
                model="real-corpus-dev-presenter",
                prompt_tokens=4,
                completion_tokens=4,
            )
        )
        evidence_ids = [
            self._selected.net_profit.evidence_id,
            self._selected.revenue.evidence_id,
        ]
        evidence_ids = list(dict.fromkeys(evidence_ids))
        return {
            "query": str(kwargs.get("query") or self._selected.query),
            "question_type": "calculation",
            "answer_mode": "numeric_fact",
            "final_answer": self._selected.expected_answer,
            "answer": self._selected.expected_answer,
            "abstained": False,
            "abstain_reason": None,
            "used_evidence_ids": evidence_ids,
            "citations": [dict(row) for row in self._selected.chunks],
            "selected_doc_ids": list(
                dict.fromkeys(str(row.get("doc_id") or "") for row in self._selected.chunks)
            ),
            "support_validation": {
                "supported": True,
                "method": "deterministic_calculation_presentation",
            },
        }


class _DeterministicDevClaimExtractor:
    def __init__(self, selected: SelectedDevCase) -> None:
        self._selected = selected

    def generate(self, **_: Any) -> LLMResponse:
        payload = {
            "claims": [
                {
                    "id": "dev-derived-claim",
                    "text": self._selected.expected_answer,
                    "claim_type": "DERIVED",
                    "is_core": True,
                    "source_step_ids": ["calculate_net_margin"],
                    "parent_claim_ids": [],
                }
            ]
        }
        return LLMResponse(
            content=json.dumps(payload, ensure_ascii=False),
            provider="offline-deterministic",
            model="real-corpus-dev-claim-extractor",
            prompt_tokens=5,
            completion_tokens=5,
        )


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
        "claim_llm_budget": config.claim_llm_budget,
        "rewrite_max_tokens": config.rewrite_max_tokens,
    }


def _bundle_config(identity: RunIdentity) -> dict[str, Any]:
    value = identity.to_dict()
    return {
        "benchmark_profile": identity.benchmark_profile,
        "benchmark_version": identity.benchmark_version,
        "provider": identity.provider,
        "model": identity.model,
        "model_revision": identity.model_revision,
        "temperature": identity.temperature,
        "prompt_version": identity.prompt_version,
        "budgets": value["budgets"],
        "runtime": value["feature_flags"]["runtime"],
        "treatments": value["feature_flags"]["treatments"],
        "evidence_scope": DEV_SCOPE,
        "publishable": False,
    }


def run_current_dev_trace(
    output_root: Path,
    *,
    chunks_path: Path,
    facts_path: Path,
    aliases_path: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run and publish the current real-corpus dev evidence contract."""

    output_root = Path(output_root).resolve()
    chunks_path = Path(chunks_path).resolve()
    facts_path = Path(facts_path).resolve()
    aliases_path = Path(aliases_path).resolve()
    for label, path in (
        ("chunks", chunks_path),
        ("structured facts", facts_path),
        ("company aliases", aliases_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} asset does not exist: {path}")

    reports_dir = ensure_dir(output_root / "reports")
    source_manifest = build_source_manifest(cwd=REPO_ROOT)
    source_manifest_path = reports_dir / DEV_SOURCE_MANIFEST_NAME
    write_json(source_manifest_path, source_manifest)
    corpus_manifest = build_corpus_asset_manifest(
        chunks_path=chunks_path,
        facts_path=facts_path,
        aliases_path=aliases_path,
    )
    selected = _select_dev_case(
        chunks_path=chunks_path,
        facts_path=facts_path,
        aliases_path=aliases_path,
    )
    eval_row = {
        "question_id": DEV_CASE_ID,
        "case_id": DEV_CASE_ID,
        "query": selected.query,
        "question_type": "calculation",
        "intent": "derived_calculation",
        "industry": "real_corpus_dev",
        "category": "real_corpus_net_margin_contract",
    }
    config = AgentConfig(
        enable_tool_orchestration=True,
        max_query_rewrites=0,
        max_generation_attempts=1,
        max_claim_retrievals=0,
        claim_llm_budget=0,
    )
    capabilities = config.resolved_execution_capabilities().to_dict()
    identity = RunIdentity(
        benchmark_profile="current-dev",
        benchmark_version="real-corpus-dev-contract-v1",
        benchmark_hash=hash_benchmark_rows([eval_row]),
        case_ids_hash=hash_case_ids([DEV_CASE_ID]),
        case_count=1,
        corpus_hash=str(corpus_manifest["corpus_hash"]),
        model="real-corpus-dev-deterministic-composite",
        provider="offline-deterministic",
        model_revision="real-corpus-dev-v1",
        temperature=0.0,
        prompt_version="real-corpus-net-margin-v1",
        budgets=_budgets(config),
        feature_flags={
            "runtime": {
                "strict_trace": True,
                "external_network": False,
                "model_downloads": False,
                "dev_contract": True,
                "evidence_scope": DEV_SCOPE,
                "retrieval_source": "current-local-structured-facts",
                "corpus_asset_manifest_hash": corpus_manifest["manifest_hash"],
            },
            "treatments": {
                "pipeline": "agentic",
                "profile": "full-agent",
                **capabilities,
            },
        },
        git_commit=detect_git_sha(cwd=REPO_ROOT) or "git-unavailable",
        source_manifest_hash=str(source_manifest["source_manifest_hash"]),
    )

    runtime = _NoRetrievalRuntime()
    store = FactStore(":memory:")
    store.upsert((selected.net_profit, selected.revenue))
    started = perf_counter()
    try:
        state = run_agentic_rag(
            runtime=runtime,
            answerer=_DeterministicDevAnswerer(selected),
            llm=_DeterministicDevClaimExtractor(selected),
            config=config,
            query=selected.query,
            question_type="calculation",
            fact_store=store,
            company_aliases=selected.aliases,
        )
    finally:
        store.close()
    latency_ms = (perf_counter() - started) * 1000
    if runtime.calls:
        raise RuntimeError(
            "current-dev contract must stay on structured facts; "
            f"report search was called {len(runtime.calls)} time(s)"
        )
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
            "evidence_scope": DEV_SCOPE,
            "publishable": False,
            "quality_metrics_emitted": False,
            "retrieval_calls": 0,
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
        corpus_asset_manifest=corpus_manifest,
        metadata={
            "mode": "agentic",
            "evidence_scope": DEV_SCOPE,
            "publishable": False,
            "quality_claims_allowed": False,
            "purpose": "current-code real-corpus strict evidence contract",
            "legacy_trace_policy": "preserve outputs/reports/agent_traces_dev.jsonl as migration fixture",
            "selected_case": {
                "case_id": DEV_CASE_ID,
                "company": selected.company,
                "year": selected.year,
                "operation": "net_margin",
                "evidence_ids": list(
                    dict.fromkeys(
                        (
                            selected.net_profit.evidence_id,
                            selected.revenue.evidence_id,
                        )
                    )
                ),
            },
        },
        overwrite=overwrite,
    )
    require_ready_eval_bundle_integrity(bundle)

    integrity_report = validate_bundle_evidence_integrity(bundle, chunks_path)
    if integrity_report.get("status") != "READY":
        raise RuntimeError("persisted current-dev bundle failed its independent integrity recheck")
    evidence_report_path = reports_dir / DEV_EVIDENCE_REPORT_NAME
    write_json(evidence_report_path, integrity_report)
    reference_path = reports_dir / DEV_REFERENCE_NAME
    write_eval_bundle_reference(reference_path, bundle)
    summary = {
        "status": "READY",
        "evidence_scope": DEV_SCOPE,
        "publishable": False,
        "quality_claims_allowed": False,
        "external_network_used": False,
        "model_download_used": False,
        "retrieval_calls": 0,
        "run_id": identity.run_id,
        "identity_hash": identity.identity_hash,
        "source_manifest_hash": identity.source_manifest_hash,
        "corpus_hash": identity.corpus_hash,
        "corpus_asset_manifest_hash": corpus_manifest["manifest_hash"],
        "case_id": DEV_CASE_ID,
        "company": selected.company,
        "year": selected.year,
        "operation": "net_margin",
        "bundle_reference": reference_path.relative_to(output_root).as_posix(),
        "evidence_report": evidence_report_path.relative_to(output_root).as_posix(),
        "source_manifest": source_manifest_path.relative_to(output_root).as_posix(),
        "integrity_verdict": bundle.metadata["integrity_verdict"],
        "legacy_trace_preserved": "reports/agent_traces_dev.jsonl",
    }
    write_json(reports_dir / DEV_SUMMARY_NAME, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate the current-dev strict trace from current-local corpus assets."
    )
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=REPO_ROOT / "data" / "chunks" / "chunks.jsonl",
    )
    parser.add_argument(
        "--facts-path",
        type=Path,
        default=REPO_ROOT / "data" / "structured" / "facts.duckdb",
    )
    parser.add_argument(
        "--aliases-path",
        type=Path,
        default=REPO_ROOT / "data" / "structured" / "company_aliases.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    summary = run_current_dev_trace(
        args.output_root,
        chunks_path=args.chunks_path,
        facts_path=args.facts_path,
        aliases_path=args.aliases_path,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
