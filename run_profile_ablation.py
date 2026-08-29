"""Run the strict four-profile hard-case ablation matrix.

Reviewed runs retrieve from one explicit shared chunk corpus. Benchmark gold
evidence remains evaluator-only.  A benchmark-evidence oracle exists solely
behind the explicit synthetic contract flag and can never support a
publishable performance claim.
"""

from __future__ import annotations

from src.utils.env import load_env_files

load_env_files()

import argparse
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.agent.config import AgentConfig
from src.agent.llm_claim_judge import build_claim_llm_judge
from src.agent.semantic_policy import activate_semantic_scorer
from src.evaluation.eval_bundle import (
    build_corpus_asset_manifest,
    validate_corpus_asset_manifest,
)
from src.evaluation.hard_case_benchmark import (
    DEFAULT_REVIEWED_CASES_PATH,
    DEFAULT_REVIEWED_MANIFEST_PATH,
    canonical_reviewed_cases_hash,
    load_cases,
    load_reviewed_cases,
)
from src.evaluation.profile_ablation import run_profile_matrix
from src.evaluation.profile_runtime import build_profile_executor
from src.evaluation.run_identity import (
    RunIdentity,
    canonical_sha256,
    file_sha256,
    hash_benchmark_rows,
    hash_case_ids,
)
from src.evaluation.run_metadata import detect_git_sha, write_source_manifest
from src.utils.io import ensure_dir, read_jsonl, write_json, write_jsonl


_OFFLINE_CONTRACT_FACT = {
    "fact_id": "offline-contract-controlled-revenue-fy2025",
    "company": "合约示例制造",
    "metric": "revenue",
    "period": {"kind": "FY", "year": 2025},
    "value_type": "actual",
    "value": "31010000000.0",
    "unit": "元",
    "doc_id": "offline-contract-controlled-annual-report-v1",
    "page": 17,
    "evidence_id": "offline-contract-controlled-revenue-fy2025-p17",
    "raw_value": "310.1亿元",
    "source_span": "合约示例制造2025年度营业收入为310.1亿元。",
}
_OFFLINE_CONTRACT_ALIASES = {"合约示例制造": ["合约示例制造"]}
_OFFLINE_CONTRACT_CHUNK = {
    "chunk_id": _OFFLINE_CONTRACT_FACT["evidence_id"],
    "evidence_id": _OFFLINE_CONTRACT_FACT["evidence_id"],
    "doc_id": _OFFLINE_CONTRACT_FACT["doc_id"],
    "file_name": f"{_OFFLINE_CONTRACT_FACT['doc_id']}.pdf",
    "page": _OFFLINE_CONTRACT_FACT["page"],
    "page_start": _OFFLINE_CONTRACT_FACT["page"],
    "page_end": _OFFLINE_CONTRACT_FACT["page"],
    "text": _OFFLINE_CONTRACT_FACT["source_span"],
    "child_text": _OFFLINE_CONTRACT_FACT["source_span"],
    "support_span": _OFFLINE_CONTRACT_FACT["source_span"],
    "section_title": "主要会计数据",
    "section_path": "受控合成年度报告/主要会计数据",
    "chunk_type": "structured_fact_fixture",
    "element_type": "table_row",
}


def build_parser() -> argparse.ArgumentParser:
    defaults = AgentConfig()
    parser = argparse.ArgumentParser(
        description="Run baseline-rag, agentic-rag, structured-agent and full-agent under one RunIdentity context."
    )
    parser.add_argument("--cases-path", type=Path, default=DEFAULT_REVIEWED_CASES_PATH)
    parser.add_argument(
        "--reviewed-manifest-path",
        type=Path,
        default=DEFAULT_REVIEWED_MANIFEST_PATH,
        help="Manifest required whenever --cases-path contains reviewed cases.",
    )
    parser.add_argument(
        "--allow-synthetic-contract",
        action="store_true",
        help="Explicitly permit the synthetic contract fixture; its outputs never support performance claims.",
    )
    parser.add_argument(
        "--contract-oracle",
        action="store_true",
        help="Use case evidence as retrieval only for explicit synthetic contract checks; never publishable.",
    )
    parser.add_argument(
        "--offline-contract",
        action="store_true",
        help=(
            "Run the synthetic contract with a deterministic provider-free answerer/LLM. "
            "Requires --allow-synthetic-contract and --contract-oracle; never publishable."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/profile_ablation"))
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=Path("data/chunks/chunks.jsonl"),
        help="Shared retrieval corpus required for reviewed runs and normal synthetic runs.",
    )
    parser.add_argument("--facts-path", type=Path, default=Path("data/structured/facts_dev.jsonl"))
    parser.add_argument(
        "--company-aliases-path",
        type=Path,
        default=Path("data/structured/company_aliases.json"),
    )
    parser.add_argument("--model-cache-dir", type=Path, default=Path("models"))
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument(
        "--runtime-profile", choices=("low_vram", "cpu", "standard_gpu"), default=None
    )
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--reranker-device", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=None)
    parser.add_argument("--rerank-batch-size", type=int, default=None)
    parser.add_argument("--dense-top-k", type=int, default=20)
    parser.add_argument("--bm25-top-k", type=int, default=20)
    parser.add_argument("--rerank-top-k", type=int, default=5)
    parser.add_argument("--rerank-candidates-k", type=int, default=0)
    parser.add_argument("--rebuild-indexes", action="store_true")
    parser.add_argument(
        "--llm-provider", choices=("deepseek", "openai_compatible"), default="deepseek"
    )
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--llm-model-revision", default="unversioned")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prompt-version", default="profile-ablation-v1")
    parser.add_argument("--benchmark-profile", default="")
    parser.add_argument("--reviewed-target-count", type=int, default=0)
    parser.add_argument("--answer-max-tokens", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=defaults.max_steps)
    parser.add_argument("--max-retrieval-rounds", type=int, default=defaults.max_retrieval_rounds)
    parser.add_argument("--max-query-rewrites", type=int, default=defaults.max_query_rewrites)
    parser.add_argument(
        "--max-generation-attempts", type=int, default=defaults.max_generation_attempts
    )
    parser.add_argument("--max-llm-calls", type=int, default=defaults.max_llm_calls)
    parser.add_argument("--max-total-tokens", type=int, default=defaults.max_total_tokens)
    parser.add_argument("--max-sub-questions", type=int, default=defaults.max_sub_questions)
    parser.add_argument("--max-tool-calls", type=int, default=defaults.max_tool_calls)
    parser.add_argument(
        "--max-empty-tool-results", type=int, default=defaults.max_empty_tool_results
    )
    parser.add_argument("--max-claim-retrievals", type=int, default=defaults.max_claim_retrievals)
    parser.add_argument("--claim-max-tokens", type=int, default=defaults.claim_max_tokens)
    parser.add_argument("--claim-llm-budget", type=int, default=defaults.claim_llm_budget)
    parser.add_argument(
        "--semantic-calibration-report",
        type=Path,
        default=None,
        help="Reviewed precision-first calibration report for an exact local directional-NLI scorer.",
    )
    parser.add_argument(
        "--semantic-scorer-config",
        type=Path,
        default=None,
        help="Hashed local-only directional-NLI runtime config; requires --semantic-calibration-report.",
    )
    parser.add_argument(
        "--semantic-labels-path",
        type=Path,
        default=None,
        help=(
            "Exact reviewed semantic-label JSONL used for calibration; required for semantic "
            "activation and verified against both calibration and readiness hashes."
        ),
    )
    parser.add_argument(
        "--readiness-report",
        type=Path,
        default=None,
        help=(
            "Top-level READY Phase 0 report binding the reviewed labels and calibration; "
            "required for semantic activation."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def build_agent_config(args: argparse.Namespace) -> AgentConfig:
    return AgentConfig(
        max_steps=args.max_steps,
        max_retrieval_rounds=args.max_retrieval_rounds,
        max_query_rewrites=args.max_query_rewrites,
        max_generation_attempts=args.max_generation_attempts,
        max_llm_calls=args.max_llm_calls,
        max_total_tokens=args.max_total_tokens,
        max_sub_questions=args.max_sub_questions,
        claim_max_tokens=args.claim_max_tokens,
        enable_tool_orchestration=True,
        max_tool_calls=args.max_tool_calls,
        max_empty_tool_results=args.max_empty_tool_results,
        max_claim_retrievals=args.max_claim_retrievals,
        claim_llm_budget=args.claim_llm_budget,
        strict_claim_verification=True,
    )


def _budget_identity(config: AgentConfig, *, answer_max_tokens: int) -> dict[str, int]:
    return {
        "max_steps": config.max_steps,
        "max_retrieval_rounds": config.max_retrieval_rounds,
        "max_query_rewrites": config.max_query_rewrites,
        "max_generation_attempts": config.max_generation_attempts,
        "max_llm_calls": config.max_llm_calls,
        "max_total_tokens": config.max_total_tokens,
        "max_sub_questions": config.max_sub_questions,
        "max_tool_calls": config.max_tool_calls,
        "max_empty_tool_results": config.max_empty_tool_results,
        "max_claim_retrievals": config.max_claim_retrievals,
        "claim_max_tokens": config.claim_max_tokens,
        "claim_llm_budget": config.claim_llm_budget,
        "answer_max_tokens": int(answer_max_tokens),
    }


def _benchmark_kind(cases: Sequence[Mapping[str, Any]]) -> str:
    kinds = {"synthetic" if case.get("synthetic") is True else "reviewed" for case in cases}
    if len(kinds) != 1:
        raise ValueError("profile matrix requires exactly one benchmark kind")
    return next(iter(kinds))


def _benchmark_hash(cases: Sequence[Mapping[str, Any]]) -> str:
    if _benchmark_kind(cases) == "reviewed":
        return canonical_reviewed_cases_hash(cases)
    return hash_benchmark_rows(cases)


_UNVERSIONED_MODEL_REVISIONS = frozenset(
    {"", "default", "latest", "n/a", "na", "none", "unknown", "unspecified", "unversioned"}
)


def _validate_model_revision_for_benchmark(
    model_revision: Any,
    *,
    benchmark_kind: str,
) -> str:
    """Require an explicit immutable model revision for publishable reviewed runs."""

    revision = str(model_revision or "").strip()
    if benchmark_kind == "reviewed" and revision.lower() in _UNVERSIONED_MODEL_REVISIONS:
        raise ValueError(
            "reviewed profile runs require a non-default, versioned --llm-model-revision"
        )
    return revision


def build_base_identity(
    *,
    cases: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    provider: str,
    model: str,
    benchmark_version: str,
    corpus_hash: str,
    git_commit: str,
    source_manifest_hash: str,
    reviewed_manifest: Mapping[str, Any] | None = None,
    corpus_asset_manifest: Mapping[str, Any] | None = None,
    semantic_activation: Mapping[str, Any] | None = None,
) -> RunIdentity:
    """Build the common identity before profile-specific treatments exist."""
    kind = _benchmark_kind(cases)
    model_revision = _validate_model_revision_for_benchmark(
        args.llm_model_revision,
        benchmark_kind=kind,
    )
    if kind == "reviewed" and reviewed_manifest is None:
        raise ValueError("reviewed RunIdentity requires a validated benchmark manifest")
    if kind == "reviewed" and corpus_asset_manifest is None:
        raise ValueError("reviewed RunIdentity requires an explicit corpus asset manifest")
    if kind == "synthetic" and reviewed_manifest is not None:
        raise ValueError("synthetic RunIdentity must not include a reviewed manifest")
    config = build_agent_config(args)
    identity = RunIdentity(
        benchmark_profile=args.benchmark_profile or f"hard-cases-{kind}",
        benchmark_version=benchmark_version,
        benchmark_hash=_benchmark_hash(cases),
        case_ids_hash=hash_case_ids(str(case["case_id"]) for case in cases),
        case_count=len(cases),
        corpus_hash=corpus_hash,
        model=model,
        provider=provider,
        model_revision=model_revision,
        temperature=args.temperature,
        prompt_version=args.prompt_version,
        budgets=_budget_identity(config, answer_max_tokens=args.answer_max_tokens),
        feature_flags={
            "runtime": {
                "profile_execution_seam": "profile-spec-v1",
                "retrieval_source": (
                    "synthetic-contract-oracle"
                    if bool(getattr(args, "contract_oracle", False))
                    else "shared-corpus"
                ),
                "contract_oracle_non_publishable": bool(getattr(args, "contract_oracle", False)),
                "offline_contract": bool(getattr(args, "offline_contract", False)),
                "structured_context_source": (
                    "materialized-offline-contract-fixture"
                    if bool(getattr(args, "offline_contract", False))
                    else "explicit-corpus-assets"
                ),
                "external_network": False
                if bool(getattr(args, "offline_contract", False))
                else None,
                "model_downloads": False
                if bool(getattr(args, "offline_contract", False))
                else None,
                "strict_treatment_integrity": True,
                "reviewed_manifest_validated": kind == "reviewed",
                "reviewed_manifest_hash": (
                    canonical_sha256(reviewed_manifest) if reviewed_manifest is not None else None
                ),
                "reviewed_target_count": args.reviewed_target_count or None,
                "corpus_asset_manifest_hash": (
                    str(corpus_asset_manifest.get("manifest_hash") or "")
                    if corpus_asset_manifest is not None
                    else None
                ),
                "semantic_activation": dict(semantic_activation or {"enabled": False}),
            },
            "treatments": {},
        },
        git_commit=git_commit or "unavailable",
        source_manifest_hash=source_manifest_hash,
    )
    if corpus_asset_manifest is not None:
        validate_corpus_asset_manifest(identity, corpus_asset_manifest, verify_files=True)
    return identity


def _load_benchmark(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    cases_path = args.cases_path.resolve()
    if not cases_path.is_file():
        raise FileNotFoundError(
            f"benchmark cases not found: {cases_path}. Provide a reviewed asset with --cases-path, "
            "or explicitly select the synthetic fixture with --allow-synthetic-contract."
        )
    cases = load_cases(cases_path)
    kind = _benchmark_kind(cases)
    if kind == "synthetic":
        if not args.allow_synthetic_contract:
            raise ValueError(
                "synthetic hard cases require --allow-synthetic-contract and cannot support performance claims"
            )
        return cases, None
    manifest_path = args.reviewed_manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"reviewed benchmark manifest not found: {manifest_path}")
    return load_reviewed_cases(cases_path, manifest_path)


def _corpus_hash(
    *,
    chunks_path: Path,
    facts_path: Path,
    aliases_path: Path,
) -> str:
    return str(
        build_corpus_asset_manifest(
            chunks_path=chunks_path,
            facts_path=facts_path,
            aliases_path=aliases_path,
        )["corpus_hash"]
    )


def _materialize_contract_oracle_chunks(
    cases: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    include_offline_structured_fixture: bool = False,
) -> Path:
    rows: list[dict[str, Any]] = []
    for case in cases:
        if case.get("synthetic") is not True:
            raise ValueError("--contract-oracle is restricted to synthetic contract cases")
        evidence = case.get("evidence")
        if not isinstance(evidence, list):
            raise ValueError("synthetic contract case evidence must be a list")
        for item in evidence:
            if not isinstance(item, Mapping):
                raise ValueError("synthetic contract evidence rows must be objects")
            evidence_id = str(item.get("evidence_id") or "").strip()
            if not evidence_id:
                raise ValueError("synthetic contract evidence requires evidence_id")
            source = str(item.get("source") or "synthetic-contract").strip()
            rows.append(
                {
                    **dict(item),
                    "chunk_id": evidence_id,
                    "evidence_id": evidence_id,
                    "doc_id": source,
                    "page": int(item.get("page") or 1),
                }
            )
    if not rows:
        raise ValueError("synthetic contract oracle requires non-empty evidence")
    if include_offline_structured_fixture:
        rows.append(dict(_OFFLINE_CONTRACT_CHUNK))
    write_jsonl(path, rows)
    return path


def _materialize_offline_contract_structured_context(
    output_root: Path,
) -> tuple[Path, Path]:
    """Write one public-safe structured seam fixture inside the run directory."""

    facts_path = output_root / "offline_contract_structured_facts.jsonl"
    aliases_path = output_root / "offline_contract_company_aliases.json"
    write_jsonl(facts_path, [dict(_OFFLINE_CONTRACT_FACT)])
    write_json(aliases_path, dict(_OFFLINE_CONTRACT_ALIASES))
    return facts_path, aliases_path


def _resolve_structured_context_paths(
    args: argparse.Namespace,
    output_root: Path,
) -> tuple[Path, Path]:
    """Select real corpus assets or materialize the provider-free contract fixture."""

    if args.offline_contract and args.contract_oracle:
        return _materialize_offline_contract_structured_context(output_root)
    return args.facts_path.resolve(), args.company_aliases_path.resolve()


def _load_shared_chunks(path: Path) -> list[dict[str, Any]]:
    source = path.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"shared chunks asset not found: {source}")
    try:
        rows = read_jsonl(source)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"shared chunks asset is invalid: {source}: {exc}") from exc
    if not rows:
        raise ValueError(f"shared chunks asset is empty: {source}")
    seen_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping) or not str(row.get("chunk_id") or "").strip():
            raise ValueError(f"shared chunks row {index} requires chunk_id")
        chunk_id = str(row["chunk_id"]).strip()
        if chunk_id in seen_ids:
            raise ValueError(f"shared chunks row {index} duplicates chunk_id={chunk_id}")
        seen_ids.add(chunk_id)
        document = str(
            row.get("doc_id") or row.get("document_id") or row.get("document") or ""
        ).strip()
        page = row.get("page") or row.get("page_num") or row.get("page_number")
        if page is None:
            page = row.get("page_start") or row.get("page_end")
        if not document or page in (None, ""):
            raise ValueError(f"shared chunks row {index} requires document and page metadata")
    return [dict(row) for row in rows]


def _build_retrieval_runtime(
    args: argparse.Namespace,
    chunks: list[dict[str, Any]],
    output_root: Path,
) -> Any:
    from src.retrieval.runtime import build_retrieval_runtime

    return build_retrieval_runtime(
        chunks=chunks,
        output_dir=output_root,
        model_cache_dir=args.model_cache_dir.resolve(),
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
        runtime_profile=args.runtime_profile,
        embedding_device=args.embedding_device,
        reranker_device=args.reranker_device,
        embedding_batch_size=args.embedding_batch_size,
        rerank_batch_size=args.rerank_batch_size,
        dense_top_k=args.dense_top_k,
        bm25_top_k=args.bm25_top_k,
        rerank_top_k=args.rerank_top_k,
        rerank_candidates_k=args.rerank_candidates_k or None,
        rebuild_indexes=args.rebuild_indexes,
    )


def _require_structured_context(
    facts_path: Path,
    aliases_path: Path,
) -> tuple[Any, Mapping[str, list[str]]]:
    from src.structured.agent_bridge import load_structured_context

    facts_path = facts_path.resolve()
    aliases_path = aliases_path.resolve()
    fact_store, aliases = load_structured_context(facts_path, aliases_path)
    if fact_store is None or not aliases:
        if fact_store is not None and hasattr(fact_store, "close"):
            fact_store.close()
        raise FileNotFoundError(
            "structured-agent and full-agent require non-empty facts and company aliases: "
            f"facts={facts_path}, aliases={aliases_path}"
        )
    if hasattr(fact_store, "count") and int(fact_store.count()) <= 0:
        fact_store.close()
        raise ValueError(f"structured fact store is empty: {facts_path}")
    return fact_store, aliases


def _build_profile_executor(
    *,
    args: argparse.Namespace,
    answerer: Any,
    llm: Any,
    fact_store: Any,
    aliases: Mapping[str, list[str]],
    retrieval_runtime: Any,
    semantic_scorer: Any = None,
):
    """Build the shared executor and bind its judge to the recorded budget."""

    config = build_agent_config(args)
    llm_judge = build_claim_llm_judge(
        llm,
        budget=config.claim_llm_budget,
        max_tokens=config.claim_max_tokens,
    )
    return build_profile_executor(
        answerer=answerer,
        llm=llm,
        base_config=config,
        fact_store=fact_store,
        company_aliases=aliases,
        retrieval_runtime=retrieval_runtime,
        allow_contract_oracle=args.contract_oracle,
        semantic_scorer=semantic_scorer,
        llm_judge=llm_judge,
    )


def _validate_semantic_cli_request(args: argparse.Namespace, *, benchmark_kind: str) -> None:
    report = getattr(args, "semantic_calibration_report", None)
    config = getattr(args, "semantic_scorer_config", None)
    labels = getattr(args, "semantic_labels_path", None)
    readiness = getattr(args, "readiness_report", None)
    requested = (report, config, labels, readiness)
    if any(requested) and not all(requested):
        raise ValueError(
            "semantic activation requires --semantic-calibration-report, "
            "--semantic-scorer-config, --semantic-labels-path, and --readiness-report"
        )
    if report and benchmark_kind != "reviewed":
        raise ValueError("semantic activation is restricted to reviewed profile runs")
    if report and getattr(args, "offline_contract", False):
        raise ValueError("--offline-contract cannot enable semantic activation")


def _load_optional_semantic_scorer(
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any]]:
    report_path = getattr(args, "semantic_calibration_report", None)
    config_path = getattr(args, "semantic_scorer_config", None)
    labels_path = getattr(args, "semantic_labels_path", None)
    readiness_path = getattr(args, "readiness_report", None)
    if (
        report_path is None
        and config_path is None
        and labels_path is None
        and readiness_path is None
    ):
        return None, {"enabled": False}
    if not all((report_path, config_path, labels_path, readiness_path)):
        raise ValueError(
            "semantic activation requires calibration, scorer config, reviewed labels, and READY readiness"
        )
    assert report_path is not None and config_path is not None
    assert labels_path is not None and readiness_path is not None
    report_path = Path(report_path).resolve()
    config_path = Path(config_path).resolve()
    labels_path = Path(labels_path).resolve()
    readiness_path = Path(readiness_path).resolve()
    if not report_path.is_file():
        raise FileNotFoundError(f"semantic calibration report does not exist: {report_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"semantic scorer config does not exist: {config_path}")
    if not labels_path.is_file():
        raise FileNotFoundError(f"reviewed semantic labels do not exist: {labels_path}")
    if not readiness_path.is_file():
        raise FileNotFoundError(f"readiness report does not exist: {readiness_path}")
    try:
        calibration_document = json.loads(report_path.read_text(encoding="utf-8"))
        scorer_config = json.loads(config_path.read_text(encoding="utf-8"))
        readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"semantic activation JSON is malformed: {exc.msg}") from exc
    if not all(
        isinstance(document, Mapping)
        for document in (calibration_document, scorer_config, readiness)
    ):
        raise ValueError(
            "semantic calibration report, scorer config, and readiness report must be JSON objects"
        )
    nested_report = calibration_document.get("semantic_calibration")
    calibration_report = (
        nested_report if isinstance(nested_report, Mapping) else calibration_document
    )

    from src.agent.directional_nli import (
        build_local_directional_nli_scorer,
        directional_nli_identity,
    )
    from src.evaluation.semantic_calibration import load_reviewed_semantic_labels

    scorer_identity = directional_nli_identity(scorer_config)
    label_rows = load_reviewed_semantic_labels(labels_path)
    if not label_rows:
        raise ValueError("reviewed semantic labels must not be empty")
    actual_labels_sha256 = file_sha256(labels_path)
    label_identity = {
        "kind": str(label_rows[0].get("scorer_kind") or "").strip(),
        "model": str(label_rows[0].get("scorer_model") or "").strip(),
        "revision": str(label_rows[0].get("scorer_revision") or "").strip(),
        "config_sha256": str(label_rows[0].get("scorer_config_sha256") or "").strip().lower(),
    }
    runtime_identity = {
        field: str(scorer_identity.get(field) or "").strip()
        for field in ("kind", "model", "revision", "config_sha256")
    }
    runtime_identity["config_sha256"] = runtime_identity["config_sha256"].lower()
    if label_identity != runtime_identity:
        raise ValueError("actual reviewed semantic labels do not match the runtime scorer identity")

    if (
        readiness.get("status") != "READY"
        or readiness.get("readiness_status") != "READY"
        or readiness.get("blocked") is not False
    ):
        raise ValueError("semantic activation requires a top-level READY readiness report")
    readiness_assets = readiness.get("assets")
    readiness_labels = (
        readiness_assets.get("semantic_labels") if isinstance(readiness_assets, Mapping) else None
    )
    if not isinstance(readiness_labels, Mapping) or readiness_labels.get("status") != "READY":
        raise ValueError("readiness report does not mark the reviewed semantic labels READY")
    readiness_labels_sha256 = str(readiness_labels.get("sha256") or "").strip().lower()
    if readiness_labels_sha256 != actual_labels_sha256:
        raise ValueError("readiness semantic-label SHA-256 does not match the actual labels file")
    readiness_labels_path = str(readiness_labels.get("path") or "").strip()
    if not readiness_labels_path or os.path.normcase(
        str(Path(readiness_labels_path).resolve())
    ) != os.path.normcase(str(labels_path)):
        raise ValueError("readiness semantic-label path does not match --semantic-labels-path")
    readiness_calibration = readiness.get("semantic_calibration")
    if not isinstance(readiness_calibration, Mapping):
        raise ValueError("readiness report has no semantic_calibration contract")
    readiness_calibration_hash = (
        str(
            readiness_calibration.get("labels_sha256")
            or readiness_calibration.get("dataset_sha256")
            or ""
        )
        .strip()
        .lower()
    )
    if readiness_calibration_hash != actual_labels_sha256:
        raise ValueError("readiness calibration is not bound to the actual reviewed labels")
    for field in ("calibrated", "precision_constraint_satisfied", "coverage_constraint_satisfied"):
        if readiness_calibration.get(field) is not True:
            raise ValueError(f"readiness semantic calibration field {field!r} is not true")

    report_labels_sha256 = (
        str(
            calibration_report.get("labels_sha256")
            or calibration_report.get("dataset_sha256")
            or ""
        )
        .strip()
        .lower()
    )
    if not re.fullmatch(r"[0-9a-f]{64}", report_labels_sha256):
        raise ValueError("calibration report has no valid reviewed semantic-label SHA-256")
    if report_labels_sha256 != actual_labels_sha256:
        raise ValueError("calibration report SHA-256 does not match the actual reviewed labels")
    readiness_identity = readiness_calibration.get("scorer_identity")
    calibration_identity = calibration_report.get("scorer_identity")
    if (
        not isinstance(readiness_identity, Mapping)
        or not isinstance(calibration_identity, Mapping)
        or dict(readiness_identity) != dict(calibration_identity)
    ):
        raise ValueError("readiness and standalone calibration scorer identities do not match")
    if readiness_calibration.get("threshold") != calibration_report.get("threshold"):
        raise ValueError("readiness and standalone calibration thresholds do not match")
    raw_scorer = build_local_directional_nli_scorer(scorer_config)
    if raw_scorer.identity != scorer_identity:
        raise ValueError("loaded semantic scorer identity does not match its validated config")
    activated = activate_semantic_scorer(
        raw_scorer,
        readiness_calibration,
        scorer_identity=scorer_identity,
    )
    metadata = {
        "enabled": True,
        "profiles": ["agentic-rag", "full-agent"],
        "calibration_report_sha256": file_sha256(report_path),
        "readiness_report_sha256": file_sha256(readiness_path),
        "semantic_labels_path": str(labels_path),
        "labels_sha256": activated.labels_sha256,
        "threshold": activated.threshold,
        "calibrated_statuses": list(activated.calibrated_statuses),
        "scorer_identity": scorer_identity,
        "local_files_only": True,
        "trust_remote_code": False,
    }
    return activated, metadata


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.reviewed_target_count < 0:
        raise ValueError("--reviewed-target-count must be non-negative")
    if args.answer_max_tokens <= 0:
        raise ValueError("--answer-max-tokens must be positive")

    cases, reviewed_manifest = _load_benchmark(args)
    benchmark_kind = _benchmark_kind(cases)
    _validate_model_revision_for_benchmark(
        args.llm_model_revision,
        benchmark_kind=benchmark_kind,
    )
    _validate_semantic_cli_request(args, benchmark_kind=benchmark_kind)
    semantic_scorer, semantic_activation = _load_optional_semantic_scorer(args)
    if args.offline_contract and (
        not args.allow_synthetic_contract
        or not args.contract_oracle
        or benchmark_kind != "synthetic"
    ):
        raise ValueError(
            "--offline-contract requires synthetic cases plus "
            "--allow-synthetic-contract and --contract-oracle"
        )
    output_root = ensure_dir(args.output_root.resolve())
    facts_path, aliases_path = _resolve_structured_context_paths(args, output_root)
    fact_store, aliases = _require_structured_context(facts_path, aliases_path)
    if args.contract_oracle:
        if benchmark_kind != "synthetic":
            fact_store.close()
            raise ValueError("--contract-oracle is restricted to synthetic contract cases")
        chunks_path = _materialize_contract_oracle_chunks(
            cases,
            output_root / "contract_oracle_chunks.jsonl",
            include_offline_structured_fixture=args.offline_contract,
        )
        retrieval_runtime = None
    else:
        chunks_path = args.chunks_path.resolve()
        chunks = _load_shared_chunks(chunks_path)
        retrieval_runtime = _build_retrieval_runtime(args, chunks, output_root)
    source_manifest = write_source_manifest(output_root / "source_manifest.json", cwd=Path.cwd())

    if args.offline_contract:
        from src.evaluation.offline_contract_runtime import (
            OFFLINE_CONTRACT_MODEL,
            OFFLINE_CONTRACT_PROVIDER,
            OFFLINE_CONTRACT_REVISION,
            OfflineContractAnswerer,
            OfflineContractLLM,
        )

        answerer = OfflineContractAnswerer()
        llm = OfflineContractLLM()
        provider = OFFLINE_CONTRACT_PROVIDER
        model = OFFLINE_CONTRACT_MODEL
        args.llm_model_revision = OFFLINE_CONTRACT_REVISION
    else:
        from src.generation.provider import build_generation_answerer

        answerer = build_generation_answerer(
            provider=args.llm_provider,
            local_model_name="",
            remote_model_name=args.llm_model,
            cache_dir=args.model_cache_dir.resolve(),
            max_new_tokens=args.answer_max_tokens,
            temperature=args.temperature,
        )
        llm = getattr(answerer, "llm", None)
        if llm is None:
            fact_store.close()
            raise ValueError(
                "profile ablation requires an answerer exposing the generic llm provider"
            )
        provider = str(getattr(answerer, "llm_provider", "") or args.llm_provider)
        model = str(getattr(answerer, "llm_model", "") or args.llm_model)
    versions = {str(case.get("benchmark_version") or "") for case in cases}
    if len(versions) != 1 or not next(iter(versions)):
        fact_store.close()
        raise ValueError("all benchmark cases must share one non-empty benchmark_version")
    benchmark_version = next(iter(versions))
    corpus_asset_manifest = build_corpus_asset_manifest(
        chunks_path=chunks_path,
        facts_path=facts_path,
        aliases_path=aliases_path,
    )
    base_identity = build_base_identity(
        cases=cases,
        args=args,
        provider=provider,
        model=model,
        benchmark_version=benchmark_version,
        corpus_hash=str(corpus_asset_manifest["corpus_hash"]),
        git_commit=detect_git_sha(cwd=Path.cwd()),
        source_manifest_hash=str(source_manifest["source_manifest_hash"]),
        reviewed_manifest=reviewed_manifest,
        corpus_asset_manifest=corpus_asset_manifest,
        semantic_activation=semantic_activation,
    )
    executor = _build_profile_executor(
        args=args,
        answerer=answerer,
        llm=llm,
        fact_store=fact_store,
        aliases=aliases,
        retrieval_runtime=retrieval_runtime,
        semantic_scorer=semantic_scorer,
    )
    try:
        result = run_profile_matrix(
            cases=cases,
            base_identity=base_identity,
            executor=executor,
            output_root=output_root,
            chunks_path=chunks_path,
            corpus_asset_manifest=corpus_asset_manifest,
            reviewed_manifest=reviewed_manifest,
            reviewed_target_count=args.reviewed_target_count or None,
            overwrite=args.overwrite,
        )
    finally:
        fact_store.close()

    summary_path = output_root / "profile_ablation_summary.json"
    summary = {
        **dict(result.summary),
        "base_run_identity": base_identity.to_dict(),
        "reviewed_manifest": dict(reviewed_manifest) if reviewed_manifest is not None else None,
        "source_manifest_path": str(output_root / "source_manifest.json"),
        "chunks_path": str(chunks_path),
        "facts_path": str(facts_path),
        "company_aliases_path": str(aliases_path),
        "retrieval_source": base_identity.feature_flags["runtime"]["retrieval_source"],
    }
    write_json(summary_path, summary)
    print(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "status": result.summary["status"],
                "run_ids": result.summary["run_ids"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
