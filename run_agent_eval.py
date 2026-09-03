"""Agentic RAG evaluation CLI (checklist v3.0 §P3).

Modes:
- ``--mode agentic``  run the agentic pipeline over the agent benchmark and
  emit per-query traces + quality/process/API summaries.
- ``--mode baseline``  run the single-shot baseline RAG path (same loop as
  run_answer_eval) over the same rows and emit results + API usage.
- ``--mode compare``   build the Baseline vs Agentic comparison + P3 gate verdict
  from the two artifacts above.
"""

from __future__ import annotations

from src.utils.env import load_env_files

load_env_files()

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, Mapping

from run_answer_eval import apply_retrieval_domain_priority, build_failed_result_row, resolve_eval_limit
from src.agent.config import AgentConfig
from src.agent.graph import run_agentic_rag
from src.agent.llm_claim_judge import build_claim_llm_judge
from src.evaluation.agent_eval import (
    build_baseline_agentic_comparison,
    evaluate_agent_quality,
    evaluate_p3_gate,
    materialize_agent_eval_set,
)
from src.evaluation.api_usage_eval import build_api_usage_summary, build_api_usage_summary_from_calls
from src.evaluation.eval_bundle import (
    EvalBundle,
    EvalBundleIntegrityError,
    build_corpus_asset_manifest,
    read_eval_bundle,
    read_eval_bundle_reference,
    require_comparable_eval_bundles,
    validate_corpus_asset_manifest,
    write_eval_bundle_reference,
    write_validated_eval_bundle,
)
from src.evaluation.output_archive import (
    archive_output_bundle,
    default_artifact_label,
    resolve_artifact_dir,
    write_scratch_run_policy,
)
from src.evaluation.run_identity import (
    RunIdentity,
    canonical_sha256,
    hash_benchmark_rows,
    hash_case_ids,
)
from src.evaluation.run_metadata import build_run_metadata, normalize_benchmark_profile
from src.evaluation.trajectory_eval import (
    build_agent_process_summary,
    build_agent_trace_row,
    build_failed_agent_trace_row,
)
from src.utils.io import ensure_dir, read_json, read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run agent evaluation, baseline comparison and the P3 gate.")
    parser.add_argument("--mode", choices=("agentic", "baseline", "compare"), default="agentic")
    parser.add_argument("--split", choices=("dev",), default="dev", help="Agent benchmark supports dev only (checklist scope).")
    parser.add_argument("--agent-seed-path", type=Path, default=Path("data/eval_set/agent_eval_seed_dev.jsonl"))
    parser.add_argument("--agent-eval-path", type=Path, default=Path("data/eval_set/agent_eval_dev.jsonl"))
    parser.add_argument("--retrieval-seed-path", type=Path, default=Path("data/eval_set/retrieval_eval_seed.jsonl"))
    parser.add_argument("--chunks-path", type=Path, default=Path("data/chunks/chunks.jsonl"))
    parser.add_argument("--prepare-agent-benchmark", action="store_true")
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="Deprecated alias for --eval-limit.")
    parser.add_argument("--model-cache-dir", type=Path, default=Path("models"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--runtime-profile", default=None, choices=("low_vram", "cpu", "standard_gpu"))
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--reranker-device", default=None)
    parser.add_argument("--device", default=None, help="Legacy alias: sets both retrieval devices.")
    parser.add_argument("--embedding-batch-size", type=int, default=None)
    parser.add_argument("--rerank-batch-size", type=int, default=None)
    parser.add_argument("--dense-top-k", type=int, default=20)
    parser.add_argument("--bm25-top-k", type=int, default=20)
    parser.add_argument("--rerank-top-k", type=int, default=5)
    parser.add_argument("--rerank-candidates-k", type=int, default=0)
    parser.add_argument("--rebuild-indexes", action="store_true")
    parser.add_argument("--llm-provider", default="")
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--llm-model-revision", default="unversioned")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prompt-version", default="agent-eval-v1")
    parser.add_argument("--benchmark-profile", default="current-dev")
    parser.add_argument("--benchmark-version", default="dev-v1")
    parser.add_argument("--facts-path", type=Path, default=Path("data/structured/facts_dev.jsonl"))
    parser.add_argument("--company-aliases-path", type=Path, default=Path("data/structured/company_aliases.json"))
    parser.add_argument("--max-steps", type=int, default=AgentConfig().max_steps)
    parser.add_argument(
        "--max-retrieval-rounds",
        type=int,
        default=AgentConfig().max_retrieval_rounds,
    )
    parser.add_argument("--max-query-rewrites", type=int, default=2)
    parser.add_argument("--max-generation-attempts", type=int, default=2)
    parser.add_argument("--max-llm-calls", type=int, default=6)
    parser.add_argument(
        "--claim-llm-budget",
        type=int,
        default=AgentConfig().claim_llm_budget,
        help="Maximum qualitative claim-judge calls per case; 0 keeps deterministic-only verification.",
    )
    parser.add_argument("--max-total-tokens", type=int, default=0, help="0 = token budget disabled.")
    parser.add_argument("--baseline-results-path", type=Path, default=None, help="compare mode: explicit baseline results jsonl.")
    parser.add_argument("--baseline-bundle-path", type=Path, default=None, help="compare mode: strict baseline eval bundle directory.")
    parser.add_argument("--agentic-bundle-path", type=Path, default=None, help="compare mode: strict agentic eval bundle directory.")
    parser.add_argument("--corpus-label", default="current")
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--artifact-label", default="")
    parser.add_argument("--skip-artifact-archive", action="store_true")
    parser.add_argument("--overwrite-eval-bundle", action="store_true")
    return parser.parse_args()


def build_agent_config(args: argparse.Namespace) -> AgentConfig:
    defaults = AgentConfig()
    return AgentConfig(
        max_steps=args.max_steps,
        max_retrieval_rounds=args.max_retrieval_rounds,
        max_query_rewrites=args.max_query_rewrites,
        max_generation_attempts=args.max_generation_attempts,
        max_llm_calls=args.max_llm_calls,
        max_total_tokens=args.max_total_tokens,
        max_failed_rewrite_rounds=int(
            getattr(args, "max_failed_rewrite_rounds", defaults.max_failed_rewrite_rounds)
        ),
        max_sub_questions=int(getattr(args, "max_sub_questions", defaults.max_sub_questions)),
        claim_max_tokens=int(getattr(args, "claim_max_tokens", defaults.claim_max_tokens)),
        enable_tool_orchestration=bool(getattr(args, "enable_tool_orchestration", True)),
        max_tool_calls=int(getattr(args, "max_tool_calls", defaults.max_tool_calls)),
        max_empty_tool_results=int(
            getattr(args, "max_empty_tool_results", defaults.max_empty_tool_results)
        ),
        max_claim_retrievals=int(
            getattr(args, "max_claim_retrievals", defaults.max_claim_retrievals)
        ),
        claim_semantic_threshold=float(
            getattr(args, "claim_semantic_threshold", defaults.claim_semantic_threshold)
        ),
        claim_llm_budget=int(
            getattr(args, "claim_llm_budget", defaults.claim_llm_budget)
        ),
        strict_claim_verification=bool(
            getattr(args, "strict_claim_verification", defaults.strict_claim_verification)
        ),
        grade_mode=str(getattr(args, "grade_mode", defaults.grade_mode)),
        rewrite_temperature=float(getattr(args, "rewrite_temperature", defaults.rewrite_temperature)),
        rewrite_max_tokens=int(getattr(args, "rewrite_max_tokens", defaults.rewrite_max_tokens)),
    )


def _corpus_asset_manifest(args: argparse.Namespace) -> dict[str, Any]:
    return build_corpus_asset_manifest(
        chunks_path=Path(args.chunks_path),
        facts_path=Path(args.facts_path),
        aliases_path=Path(args.company_aliases_path),
    )


def _corpus_identity_hash(args: argparse.Namespace) -> str:
    return str(_corpus_asset_manifest(args)["corpus_hash"])


def _runtime_identity_flags(
    args: argparse.Namespace,
    *,
    corpus_asset_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    agent_defaults = AgentConfig()
    return {
        "strict_trace": True,
        "corpus_asset_manifest_hash": str(corpus_asset_manifest["manifest_hash"]),
        "embedding_model": str(getattr(args, "embedding_model", "") or ""),
        "reranker_model": str(getattr(args, "reranker_model", "") or ""),
        "runtime_profile": getattr(args, "runtime_profile", None),
        "embedding_device": getattr(args, "embedding_device", None),
        "reranker_device": getattr(args, "reranker_device", None),
        "device": getattr(args, "device", None),
        "embedding_batch_size": getattr(args, "embedding_batch_size", None),
        "rerank_batch_size": getattr(args, "rerank_batch_size", None),
        "dense_top_k": int(getattr(args, "dense_top_k", 20)),
        "bm25_top_k": int(getattr(args, "bm25_top_k", 20)),
        "rerank_top_k": int(getattr(args, "rerank_top_k", 5)),
        "rerank_candidates_k": int(getattr(args, "rerank_candidates_k", 0)),
        "rebuild_indexes": bool(getattr(args, "rebuild_indexes", False)),
        "claim_semantic_threshold": float(
            getattr(args, "claim_semantic_threshold", agent_defaults.claim_semantic_threshold)
        ),
        "strict_claim_verification": bool(
            getattr(args, "strict_claim_verification", agent_defaults.strict_claim_verification)
        ),
        "grade_mode": str(getattr(args, "grade_mode", agent_defaults.grade_mode)),
        "rewrite_temperature": float(
            getattr(args, "rewrite_temperature", agent_defaults.rewrite_temperature)
        ),
    }


def _budget_identity(args: argparse.Namespace) -> dict[str, int]:
    defaults = AgentConfig()
    return {
        "max_steps": int(getattr(args, "max_steps", 12)),
        "max_retrieval_rounds": int(getattr(args, "max_retrieval_rounds", 3)),
        "max_query_rewrites": int(getattr(args, "max_query_rewrites", 2)),
        "max_generation_attempts": int(getattr(args, "max_generation_attempts", 2)),
        "max_llm_calls": int(getattr(args, "max_llm_calls", 6)),
        "claim_llm_budget": int(
            getattr(args, "claim_llm_budget", defaults.claim_llm_budget)
        ),
        "max_total_tokens": int(getattr(args, "max_total_tokens", 0)),
        "max_failed_rewrite_rounds": int(
            getattr(args, "max_failed_rewrite_rounds", defaults.max_failed_rewrite_rounds)
        ),
        "max_sub_questions": int(getattr(args, "max_sub_questions", defaults.max_sub_questions)),
        "claim_max_tokens": int(getattr(args, "claim_max_tokens", defaults.claim_max_tokens)),
        "max_tool_calls": int(getattr(args, "max_tool_calls", defaults.max_tool_calls)),
        "max_empty_tool_results": int(
            getattr(args, "max_empty_tool_results", defaults.max_empty_tool_results)
        ),
        "max_claim_retrievals": int(
            getattr(args, "max_claim_retrievals", defaults.max_claim_retrievals)
        ),
        "rewrite_max_tokens": int(getattr(args, "rewrite_max_tokens", defaults.rewrite_max_tokens)),
    }


def build_eval_run_identity(
    args: argparse.Namespace,
    eval_rows: list[dict[str, Any]],
    *,
    metadata: dict[str, Any],
    provider: str | None = None,
    model: str | None = None,
    corpus_asset_manifest: Mapping[str, Any] | None = None,
) -> RunIdentity:
    corpus_manifest = dict(corpus_asset_manifest or _corpus_asset_manifest(args))
    case_ids = [str(row.get("case_id") or row.get("question_id") or "") for row in eval_rows]
    treatments = {
        "pipeline": str(getattr(args, "mode", "") or "unknown"),
        "tool_orchestration": bool(getattr(args, "enable_tool_orchestration", True)),
    }
    identity = RunIdentity(
        benchmark_profile=normalize_benchmark_profile(
            str(getattr(args, "benchmark_profile", "current-dev") or "current-dev")
        ),
        benchmark_version=str(getattr(args, "benchmark_version", "dev-v1") or "dev-v1"),
        benchmark_hash=hash_benchmark_rows(eval_rows),
        case_ids_hash=hash_case_ids(case_ids),
        case_count=len(eval_rows),
        corpus_hash=str(corpus_manifest["corpus_hash"]),
        model=str(model or getattr(args, "llm_model", "") or "unspecified"),
        provider=str(provider or getattr(args, "llm_provider", "") or "unspecified"),
        model_revision=str(getattr(args, "llm_model_revision", "unversioned") or "unversioned"),
        temperature=float(getattr(args, "temperature", 0.0)),
        prompt_version=str(getattr(args, "prompt_version", "agent-eval-v1") or "agent-eval-v1"),
        budgets=_budget_identity(args),
        feature_flags={
            "runtime": _runtime_identity_flags(
                args, corpus_asset_manifest=corpus_manifest
            ),
            "treatments": treatments,
        },
        git_commit=str(metadata.get("git_sha") or "unavailable"),
        source_manifest_hash=str(
            metadata.get("source_manifest_hash") or metadata.get("source_manifest_id") or "unavailable"
        ),
    )
    validate_corpus_asset_manifest(identity, corpus_manifest, verify_files=True)
    return identity


def build_eval_bundle_config(args: argparse.Namespace, identity: RunIdentity) -> dict[str, Any]:
    identity_row = identity.to_dict()
    flags = identity_row["feature_flags"]
    return {
        "benchmark_profile": identity.benchmark_profile,
        "benchmark_version": identity.benchmark_version,
        "provider": identity.provider,
        "model": identity.model,
        "model_revision": identity.model_revision,
        "temperature": identity.temperature,
        "prompt_version": identity.prompt_version,
        "budgets": identity_row["budgets"],
        "runtime": flags.get("runtime", {}),
        "treatments": flags.get("treatments", {}),
    }


def resolve_report_paths(output_dir: Path) -> dict[str, Path]:
    report_dir = output_dir / "reports"
    return {
        "agent_eval_results": report_dir / "agent_eval_results_dev.jsonl",
        "agent_traces": report_dir / "agent_traces_dev.jsonl",
        "agent_eval_summary_json": report_dir / "agent_eval_summary_dev.json",
        "agent_eval_summary_md": report_dir / "agent_eval_summary_dev.md",
        "api_usage_summary_json": report_dir / "api_usage_summary_dev.json",
        "api_usage_summary_md": report_dir / "api_usage_summary_dev.md",
        "baseline_results": report_dir / "agent_baseline_results_dev.jsonl",
        "baseline_summary": report_dir / "agent_baseline_summary_dev.json",
        "comparison_json": report_dir / "agent_baseline_comparison_dev.json",
        "comparison_md": report_dir / "agent_baseline_comparison_dev.md",
        "gate_verdict": report_dir / "gate_verdict_dev.json",
        "manifest_report": report_dir / "agent_eval_manifest_report.json",
        "agentic_bundle_ref": report_dir / "agent_eval_bundle_dev.json",
        "baseline_bundle_ref": report_dir / "agent_baseline_bundle_dev.json",
    }


def resolve_baseline_results_path(args: argparse.Namespace, output_dir: Path) -> Path:
    if args.baseline_results_path is not None:
        return Path(args.baseline_results_path)
    return output_dir / "reports" / "agent_baseline_results_dev.jsonl"


def run_agentic_eval_rows(
    *,
    eval_rows: list[dict[str, Any]],
    runtime: Any,
    answerer: Any,
    llm: Any,
    config: AgentConfig,
    llm_judge: Any = None,
    fact_store: Any = None,
    company_aliases: dict[str, list[str]] | None = None,
    run_identity: RunIdentity | None = None,
    strict_trace: bool = False,
) -> list[dict[str, Any]]:
    """Run the agentic pipeline per benchmark row; failures become failed trace rows."""
    if strict_trace and not isinstance(run_identity, RunIdentity):
        raise ValueError("strict agentic eval requires run_identity")
    trace_rows: list[dict[str, Any]] = []
    for index, eval_row in enumerate(eval_rows, start=1):
        print(f"[{index}/{len(eval_rows)}] Agentic {eval_row['question_id']} ({eval_row.get('category', '')})")
        start = perf_counter()
        try:
            state = run_agentic_rag(
                runtime=runtime,
                answerer=answerer,
                llm=llm,
                config=config,
                query=eval_row["query"],
                domain_hint=str(eval_row.get("domain_hint", eval_row.get("industry", "")) or ""),
                question_type=str(eval_row.get("question_type", "") or ""),
                fact_store=fact_store,
                company_aliases=company_aliases,
                llm_judge=llm_judge,
            )
            trace_rows.append(
                build_agent_trace_row(
                    state,
                    eval_row,
                    end_to_end_latency_ms=(perf_counter() - start) * 1000,
                    run_identity=run_identity,
                    strict=strict_trace,
                )
            )
        except Exception as exc:  # noqa: BLE001 - batch must survive per-row failures
            trace_rows.append(
                build_failed_agent_trace_row(
                    eval_row=eval_row,
                    error=exc,
                    end_to_end_latency_ms=(perf_counter() - start) * 1000,
                    run_identity=run_identity,
                    strict=strict_trace,
                )
            )
    return trace_rows


def run_baseline_rows(
    *,
    eval_rows: list[dict[str, Any]],
    runtime: Any,
    answerer: Any,
) -> tuple[list[dict[str, Any]], list[list[Any]]]:
    """Run the single-shot baseline path (same loop as run_answer_eval) per row."""
    from src.generation.answerer import infer_question_type

    result_rows: list[dict[str, Any]] = []
    calls_by_query: list[list[Any]] = []
    for index, eval_row in enumerate(eval_rows, start=1):
        print(f"[{index}/{len(eval_rows)}] Baseline {eval_row['question_id']} ({eval_row.get('category', '')})")
        start = perf_counter()
        retrieval_result: dict[str, Any] | None = None
        calls_before = len(getattr(answerer, "llm_calls", []))
        try:
            retrieval_result = runtime.search(eval_row["query"])
            retrieval_result = apply_retrieval_domain_priority(
                retrieval_result,
                str(eval_row.get("domain_hint", eval_row.get("industry", "")) or ""),
            )
            answer_row = answerer.answer(
                query=eval_row["query"],
                question_type=eval_row.get("question_type") or infer_question_type(eval_row["query"]),
                retrieval_result=retrieval_result,
                query_domain_hint=eval_row.get("domain_hint", eval_row.get("industry", "")),
            )
            answer_row["question_id"] = eval_row["question_id"]
            answer_row["timings"] = {
                **(retrieval_result.get("timings") or {}),
                "end_to_end_latency_ms": round((perf_counter() - start) * 1000, 2),
            }
            result_rows.append(answer_row)
        except Exception as exc:  # noqa: BLE001 - batch must survive per-row failures
            result_rows.append(
                build_failed_result_row(
                    eval_row=eval_row,
                    error=exc,
                    retrieval_result=retrieval_result,
                    end_to_end_latency_ms=round((perf_counter() - start) * 1000, 2),
                    llm_provider=getattr(answerer, "llm_provider", ""),
                    llm_model=getattr(answerer, "llm_model", ""),
                )
            )
        calls_by_query.append(list(getattr(answerer, "llm_calls", [])[calls_before:]))
    return result_rows, calls_by_query


def _dict_table(data: dict[str, Any]) -> str:
    lines = ["| Key | Value |", "|---|---:|"]
    for key, value in data.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                lines.append(f"| {key}.{sub_key} | {sub_value} |")
        else:
            lines.append(f"| {key} | {value} |")
    return "\n".join(lines) + "\n"


def _comparison_markdown(comparison: dict[str, Any], gate: dict[str, Any]) -> str:
    lines = ["| Question ID | Category | Baseline Answer | Agentic Answer | Baseline Citation | Agentic Citation |", "|---|---|---|---|---|---|"]
    for row in comparison["per_question"]:
        lines.append(
            f"| {row['question_id']} | {row['category']} | {row['baseline']['answer_semantic_hit']} "
            f"| {row['agentic']['answer_semantic_hit']} | {row['baseline']['citation_doc_hit']} "
            f"| {row['agentic']['citation_doc_hit']} |"
        )
    lines.append("")
    lines.append("## Deltas (agentic - baseline)")
    lines.append(_dict_table(comparison["deltas"]))
    lines.append("## P3 Gate")
    lines.append(_dict_table({key: value for key, value in gate.items() if key != "deltas" and key != "thresholds"}))
    lines.append(_dict_table(gate["deltas"]))
    lines.append(_dict_table(gate["thresholds"]))
    return "\n".join(lines)


def _build_runtime(args: argparse.Namespace, chunks: list[dict[str, Any]], output_dir: Path) -> Any:
    from src.retrieval.runtime import build_retrieval_runtime

    return build_retrieval_runtime(
        chunks=chunks,
        output_dir=output_dir,
        model_cache_dir=args.model_cache_dir.resolve(),
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
        runtime_profile=args.runtime_profile,
        embedding_device=args.embedding_device,
        reranker_device=args.reranker_device,
        device=args.device,
        embedding_batch_size=args.embedding_batch_size,
        rerank_batch_size=args.rerank_batch_size,
        dense_top_k=args.dense_top_k,
        bm25_top_k=args.bm25_top_k,
        rerank_top_k=args.rerank_top_k,
        rerank_candidates_k=args.rerank_candidates_k or None,
        rebuild_indexes=args.rebuild_indexes,
    )


def _build_answerer(args: argparse.Namespace) -> Any:
    from src.generation.provider import build_generation_answerer

    return build_generation_answerer(
        provider=args.llm_provider,
        local_model_name="",
        remote_model_name=args.llm_model,
        cache_dir=args.model_cache_dir.resolve(),
        device=args.device or "cuda",
        temperature=float(getattr(args, "temperature", 0.0)),
    )


def _build_metadata(args: argparse.Namespace, eval_rows: list[dict[str, Any]], report_dir: Path) -> dict[str, Any]:
    return build_run_metadata(
        chunks_path=args.chunks_path.resolve(),
        eval_rows=eval_rows,
        benchmark_source_path=args.agent_seed_path.resolve(),
        benchmark_label=f"agent/{args.split}",
        benchmark_profile=str(getattr(args, "benchmark_profile", "current-dev") or "current-dev"),
        corpus_label=args.corpus_label,
        split=args.split,
        cwd=Path.cwd(),
        source_manifest_path=report_dir / "source_manifest.json",
        extra_fields={"benchmark_version": str(getattr(args, "benchmark_version", "dev-v1") or "dev-v1")},
    )


def _load_comparison_bundle(*, explicit_path: Path | None, reference_path: Path, label: str) -> EvalBundle:
    if explicit_path is not None:
        return read_eval_bundle(Path(explicit_path).resolve())
    if not reference_path.is_file():
        raise FileNotFoundError(
            f"Missing {label} eval bundle reference: {reference_path}. "
            f"Run --mode {label} first or pass --{label}-bundle-path."
        )
    return read_eval_bundle_reference(reference_path)


def _validate_current_inputs_for_comparison(
    identity: RunIdentity,
    *,
    args: argparse.Namespace,
    eval_rows: list[dict[str, Any]],
) -> None:
    case_ids = [str(row.get("case_id") or row.get("question_id") or "") for row in eval_rows]
    actual = {
        "case_count": len(eval_rows),
        "case_ids_hash": hash_case_ids(case_ids),
        "benchmark_hash": hash_benchmark_rows(eval_rows),
        "corpus_hash": _corpus_identity_hash(args),
    }
    expected = identity.to_dict()
    mismatches = [field for field, value in actual.items() if expected[field] != value]
    if mismatches:
        raise EvalBundleIntegrityError(
            "current evaluation inputs do not match bundle identity: " + ", ".join(sorted(mismatches))
        )


def run_compare_mode(args: argparse.Namespace, chunks: list[dict[str, Any]], report_dir: Path) -> int:
    paths = resolve_report_paths(args.output_dir.resolve())
    baseline_bundle = _load_comparison_bundle(
        explicit_path=getattr(args, "baseline_bundle_path", None),
        reference_path=paths["baseline_bundle_ref"],
        label="baseline",
    )
    agentic_bundle = _load_comparison_bundle(
        explicit_path=getattr(args, "agentic_bundle_path", None),
        reference_path=paths["agentic_bundle_ref"],
        label="agentic",
    )
    require_comparable_eval_bundles([baseline_bundle, agentic_bundle])
    eval_rows = materialize_agent_eval_set(
        seed_path=args.agent_seed_path.resolve(),
        chunks=chunks,
        output_path=args.agent_eval_path.resolve(),
        report_output_path=paths["manifest_report"],
    )
    _validate_current_inputs_for_comparison(baseline_bundle.identity, args=args, eval_rows=eval_rows)
    baseline_rows = baseline_bundle.per_case
    agentic_rows = agentic_bundle.per_case
    if getattr(args, "baseline_results_path", None) is not None:
        legacy_rows = read_jsonl(Path(args.baseline_results_path).resolve())
        if canonical_sha256(legacy_rows) != canonical_sha256(baseline_rows):
            raise EvalBundleIntegrityError("explicit baseline results do not match the strict baseline bundle")
    chunk_lookup = {chunk["chunk_id"]: chunk for chunk in chunks}
    comparison = build_baseline_agentic_comparison(eval_rows, baseline_rows, agentic_rows, chunk_lookup)
    comparison["run_identity"] = {
        "baseline_run_id": baseline_bundle.identity.run_id,
        "agentic_run_id": agentic_bundle.identity.run_id,
        "comparison_identity_hash": canonical_sha256(baseline_bundle.identity.comparison_identity()),
        "baseline_treatments": baseline_bundle.identity.to_dict()["feature_flags"].get("treatments", {}),
        "agentic_treatments": agentic_bundle.identity.to_dict()["feature_flags"].get("treatments", {}),
    }
    gate = evaluate_p3_gate(baseline=comparison["baseline"], agentic=comparison["agentic"])
    write_json(paths["comparison_json"], comparison)
    paths["comparison_md"].write_text(_comparison_markdown(comparison, gate), encoding="utf-8")
    write_json(paths["gate_verdict"], gate)
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    print(json.dumps(comparison["deltas"], ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    report_dir = ensure_dir(output_dir / "reports")
    chunks = read_jsonl(args.chunks_path.resolve())

    if args.mode == "compare":
        return run_compare_mode(args, chunks, report_dir)

    if args.prepare_agent_benchmark:
        from src.evaluation.benchmark_assets import (
            build_agent_seed_draft,
            prepare_benchmark_assets,
            write_agent_seed_draft,
        )

        manifest, retrieval_seed_rows = prepare_benchmark_assets(
            chunks=chunks,
            manifest_output_path=args.agent_seed_path.resolve().parent / "doc_manifest.jsonl",
            seed_output_path=args.retrieval_seed_path.resolve(),
        )
        agent_rows = build_agent_seed_draft(retrieval_seed_rows=retrieval_seed_rows, chunks=chunks, manifest=manifest)
        write_agent_seed_draft(agent_rows, args.agent_seed_path.resolve())
        print(f"Wrote {len(agent_rows)} agent benchmark seed rows to {args.agent_seed_path.resolve()}")

    paths = resolve_report_paths(output_dir)
    with TemporaryDirectory(prefix=".agent-eval-inputs-", dir=output_dir) as stage_dir:
        stage_root = Path(stage_dir)
        staged_eval_path = stage_root / "agent_eval.jsonl"
        staged_manifest_report_path = stage_root / "agent_eval_manifest_report.json"
        materialized_eval_rows = materialize_agent_eval_set(
            seed_path=args.agent_seed_path.resolve(),
            chunks=chunks,
            output_path=staged_eval_path,
            report_output_path=staged_manifest_report_path,
        )
        manifest_report = read_json(staged_manifest_report_path)
    eval_rows = list(materialized_eval_rows)
    eval_limit = resolve_eval_limit(args)
    if eval_limit > 0:
        eval_rows = eval_rows[:eval_limit]

    runtime = _build_runtime(args, chunks, output_dir)
    answerer = _build_answerer(args)
    chunk_lookup = {chunk["chunk_id"]: chunk for chunk in chunks}

    artifact_dir = None
    if not args.skip_artifact_archive:
        artifact_profile = normalize_benchmark_profile(
            str(getattr(args, "benchmark_profile", "current-dev") or "current-dev")
        )
        artifact_label = args.artifact_label or default_artifact_label(
            split=args.split,
            benchmark_profile=f"{artifact_profile}-{args.mode}",
        )
        artifact_dir = resolve_artifact_dir(artifact_root=args.artifact_root.resolve(), artifact_label=artifact_label)

    with TemporaryDirectory(prefix=".agent-eval-metadata-", dir=output_dir) as stage_dir:
        staged_report_dir = Path(stage_dir)
        metadata = _build_metadata(args, eval_rows, staged_report_dir)
        source_manifest = read_json(staged_report_dir / "source_manifest.json")
    metadata["source_manifest_path"] = str(report_dir / "source_manifest.json")
    corpus_asset_manifest = _corpus_asset_manifest(args)
    run_identity = build_eval_run_identity(
        args,
        eval_rows,
        metadata=metadata,
        provider=str(getattr(answerer, "llm_provider", "") or args.llm_provider or "unspecified"),
        model=str(getattr(answerer, "llm_model", "") or args.llm_model or "unspecified"),
        corpus_asset_manifest=corpus_asset_manifest,
    )
    metadata = {**metadata, "run_id": run_identity.run_id, "run_identity": run_identity.to_dict()}

    if args.mode == "agentic":
        llm = getattr(answerer, "llm", None)
        if llm is None:
            raise ValueError("agentic mode requires a remote LLM answerer exposing the generic `llm` provider.")
        config = build_agent_config(args)
        llm_judge = build_claim_llm_judge(
            llm,
            budget=config.claim_llm_budget,
            max_tokens=config.claim_max_tokens,
        )
        from src.structured.agent_bridge import load_structured_context

        fact_store, company_aliases = load_structured_context(args.facts_path, args.company_aliases_path)
        trace_rows = run_agentic_eval_rows(
            eval_rows=eval_rows,
            runtime=runtime,
            answerer=answerer,
            llm=llm,
            config=config,
            llm_judge=llm_judge,
            fact_store=fact_store,
            company_aliases=company_aliases,
            run_identity=run_identity,
            strict_trace=True,
        )
        quality = evaluate_agent_quality(eval_rows, trace_rows, chunk_lookup)
        process = build_agent_process_summary(trace_rows)
        api_usage = build_api_usage_summary(trace_rows)
        summary = {"mode": "agentic", "quality": quality, "process": process, "api_usage": api_usage, "metadata": metadata}
        summary_path = paths["agent_eval_summary_json"]
    else:  # baseline
        result_rows, calls_by_query = run_baseline_rows(eval_rows=eval_rows, runtime=runtime, answerer=answerer)
        for row in result_rows:
            row["run_id"] = run_identity.run_id
            row["run_identity"] = run_identity.to_dict()
        quality = evaluate_agent_quality(eval_rows, result_rows, chunk_lookup)
        api_usage = build_api_usage_summary_from_calls(calls_by_query)
        summary = {"mode": "baseline", "quality": quality, "api_usage": api_usage, "metadata": metadata}
        summary_path = paths["baseline_summary"]

    bundle = write_validated_eval_bundle(
        output_dir / "eval",
        identity=run_identity,
        config=build_eval_bundle_config(args, run_identity),
        metrics=summary,
        per_case=trace_rows if args.mode == "agentic" else result_rows,
        trajectories=trace_rows if args.mode == "agentic" else [],
        chunks_path=args.chunks_path.resolve(),
        metadata={"mode": args.mode, "legacy_summary_path": str(summary_path)},
        corpus_asset_manifest=corpus_asset_manifest,
        overwrite=bool(getattr(args, "overwrite_eval_bundle", False)),
    )

    # Legacy reports are compatibility views of a strict bundle.  Publish them
    # only after the staged bundle and (for agentic runs) evidence gate commit.
    write_jsonl(args.agent_eval_path.resolve(), materialized_eval_rows)
    write_json(paths["manifest_report"], manifest_report)
    write_json(report_dir / "source_manifest.json", source_manifest)
    if args.mode == "agentic":
        write_jsonl(paths["agent_eval_results"], trace_rows)
        write_jsonl(paths["agent_traces"], trace_rows)
        write_json(paths["agent_eval_summary_json"], summary)
        paths["agent_eval_summary_md"].write_text(_dict_table(summary), encoding="utf-8")
        write_json(paths["api_usage_summary_json"], {"metadata": metadata, **api_usage})
        paths["api_usage_summary_md"].write_text(
            _dict_table({"metadata": metadata, **api_usage}), encoding="utf-8"
        )
        print(_dict_table(quality))
        print(_dict_table(process))
        print(_dict_table(api_usage))
    else:
        write_jsonl(paths["baseline_results"], result_rows)
        write_json(paths["baseline_summary"], summary)
        print(_dict_table(quality))
        print(_dict_table(api_usage))
    bundle_ref_path = paths["agentic_bundle_ref"] if args.mode == "agentic" else paths["baseline_bundle_ref"]
    write_eval_bundle_reference(bundle_ref_path, bundle)
    print(f"Strict eval bundle: {bundle.path}")

    if artifact_dir is not None:
        archive_output_bundle(output_dir=output_dir, artifact_dir=artifact_dir)
    write_scratch_run_policy(
        output_dir=output_dir,
        artifact_dir=artifact_dir,
        split=args.split,
        benchmark_profile=run_identity.benchmark_profile,
        treatments=run_identity.to_dict()["feature_flags"].get("treatments", {}),
        summary_path=summary_path,
        source_manifest_path=report_dir / "source_manifest.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
