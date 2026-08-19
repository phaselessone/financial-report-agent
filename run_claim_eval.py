"""Claim-level provenance evaluation CLI (checklist v3.0 §P7).

Runs the agentic pipeline over a query list, collects ``state["claims"]`` from
each run, and reports the three claim metrics:
  Claim Support Precision / Critical Claim Support Rate /
  Derived Claim Validation Accuracy (v1 weak consistency signal).

Independent of run_agent_eval — claims are process metrics over the agent's own
claims, not a human-annotated gold set (honest scope, see src/evaluation/claim_eval.py).

Usage:
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    python run_claim_eval.py --queries data/eval_set/agent_eval_seed_dev.jsonl \
      --runtime-profile cpu --eval-limit 5
"""

from __future__ import annotations

import os

# Same thread/offline recipe as the P6 smoke script: the bge-m3 embedder and the
# reranker cannot both load at full threads on this box (handoff P4 §4), and
# offline mode forces the local snapshot so no network download races the load.
os.environ.setdefault("MODEL_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import torch  # noqa: E402

torch.set_num_threads(2)

from src.utils.env import load_env_files  # noqa: E402

load_env_files()

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from run_agent_eval import _build_runtime
from src.agent.config import AgentConfig
from src.agent.graph import run_agentic_rag
from src.evaluation.claim_eval import analyze_claims_from_state, build_claim_eval_summary
from src.utils.io import ensure_dir, read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run claim-level provenance evaluation (checklist §P7).")
    parser.add_argument("--queries", type=Path, default=Path("data/eval_set/agent_eval_seed_dev.jsonl"))
    parser.add_argument("--chunks-path", type=Path, default=Path("data/chunks/chunks.jsonl"))
    parser.add_argument("--runtime-profile", default=None, choices=("low_vram", "cpu", "standard_gpu"))
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--model-cache-dir", type=Path, default=Path("models"))
    parser.add_argument("--offline", action="store_true", help="Use offline models only (MODEL_OFFLINE=1).")
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--device", default=None, help="Legacy alias: sets both retrieval devices.")
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--reranker-device", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=None)
    parser.add_argument("--rerank-batch-size", type=int, default=None)
    parser.add_argument("--dense-top-k", type=int, default=20)
    parser.add_argument("--bm25-top-k", type=int, default=20)
    parser.add_argument("--rerank-top-k", type=int, default=5)
    parser.add_argument("--rerank-candidates-k", type=int, default=0)
    parser.add_argument("--rebuild-indexes", action="store_true")
    parser.add_argument("--llm-provider", default="")
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--facts-path", type=Path, default=Path("data/structured/facts_dev.jsonl"))
    parser.add_argument("--company-aliases-path", type=Path, default=Path("data/structured/company_aliases.json"))
    return parser.parse_args()


def _resolve_queries(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = read_jsonl(args.queries.resolve())
    if args.eval_limit > 0:
        rows = rows[: args.eval_limit]
    return rows


def _load_fact_context(args: argparse.Namespace):
    try:
        from src.structured.agent_bridge import load_structured_context

        fact_store, aliases = load_structured_context(
            facts_path=args.facts_path,
            aliases_path=args.company_aliases_path,
        )
        return fact_store, aliases
    except Exception:  # pragma: no cover - corpus optional for claim eval
        return None, None


def _run_one(state: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    metrics = analyze_claims_from_state(state)
    return {
        "query": row.get("query", ""),
        "question_type": state.get("question_type", ""),
        "answer_mode": state.get("answer_mode", ""),
        "is_multi_hop": state.get("is_multi_hop", False),
        "termination_reason": state.get("termination_reason", ""),
        "llm_call_count": state.get("llm_call_count", 0),
        "abstained": bool((state.get("draft_answer") or {}).get("abstained", False)),
        "claims": state.get("claims") or [],
        **metrics,
    }


def main() -> int:
    args = parse_args()
    queries = _resolve_queries(args)
    if not queries:
        print("no queries resolved")
        return 1

    chunks = read_jsonl(args.chunks_path.resolve())
    config = AgentConfig()  # claim eval uses default agent budgets
    # Use the shared outputs/ runtime dir so the dense/bm25 index caches built
    # by run_agent_eval are reused instead of rebuilding 23k embeddings on CPU.
    runtime = _build_runtime(args, chunks, Path("outputs"))
    from src.generation.provider import build_generation_answerer

    answerer = build_generation_answerer(
        provider=args.llm_provider,
        local_model_name="",
        remote_model_name=args.llm_model,
        cache_dir=args.model_cache_dir.resolve(),
        device=args.device or "cuda",
        temperature=0.0,  # deterministic eval (checklist §P3 reproducible benchmark)
    )
    fact_store, aliases = _load_fact_context(args)

    rows: list[dict[str, Any]] = []
    for row in queries:
        start = perf_counter()
        state = run_agentic_rag(
            runtime=runtime,
            answerer=answerer,
            llm=answerer.llm,  # the answerer owns the provider
            config=config,
            query=row.get("query", ""),
            domain_hint=row.get("domain_hint", ""),
            question_type=row.get("question_type", ""),
            fact_store=fact_store,
            company_aliases=aliases,
        )
        trace = _run_one(state, row)
        trace["end_to_end_latency_ms"] = (perf_counter() - start) * 1000
        rows.append(trace)

    report_dir = Path("outputs/claims")
    ensure_dir(report_dir)
    write_jsonl(report_dir / "claim_eval_results.jsonl", rows)
    summary = build_claim_eval_summary(rows)
    write_json(report_dir / "claim_eval_summary.json", summary)

    # markdown summary
    lines = [
        "# Claim-level Provenance Evaluation (P7)",
        "",
        f"- queries: {summary['queries']}",
        f"- total claims: {summary['total_claims']}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Claim Support Precision | {summary['claim_support_precision']} |",
        f"| Critical Claim Support Rate | {summary['critical_claim_support_rate']} |",
        f"| Derived Claim Validation Accuracy (v1 weak) | {summary['derived_consistent_rate']} |",
        "",
        "> Process metrics over the agent's own claims; v1 does not re-run the P4",
        "> calculator for DERIVED claims and has no human claim gold set.",
    ]
    (report_dir / "claim_eval_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
