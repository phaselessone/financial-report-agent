"""Agentic RAG CLI (checklist v3.0 §P2): first version supports --mode agentic-rag only."""

from __future__ import annotations

from src.utils.env import load_env_files

load_env_files()

import argparse
import json
from pathlib import Path

from src.agent.config import AgentConfig
from src.agent.graph import run_agentic_rag
from src.agent.llm_claim_judge import build_claim_llm_judge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the minimal agentic RAG pipeline.")
    parser.add_argument("--mode", choices=("agentic-rag",), default="agentic-rag")
    parser.add_argument("--query", required=True, help="Research question to answer.")
    parser.add_argument("--question-type", default="", help="Optional explicit question_type (fact/comparison/inductive).")
    parser.add_argument("--domain-hint", default="", help="Optional industry domain hint.")
    parser.add_argument("--chunks-path", type=Path, default=Path("data/chunks/chunks.jsonl"))
    parser.add_argument("--model-cache-dir", type=Path, default=Path("models"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--runtime-profile", default=None, choices=("low_vram", "cpu", "standard_gpu"))
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--reranker-device", default=None)
    parser.add_argument("--device", default=None, help="Legacy alias: sets both retrieval devices.")
    parser.add_argument("--llm-provider", default="")
    parser.add_argument("--llm-model", default="")
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
        help="Maximum qualitative claim-judge calls; 0 keeps deterministic-only verification.",
    )
    parser.add_argument(
        "--legacy-retrieval-graph",
        action="store_true",
        help="Disable controlled tool orchestration and use the pre-M3 retrieval graph.",
    )
    return parser.parse_args()


def build_agent_config(args: argparse.Namespace) -> AgentConfig:
    """Map CLI controls into the one graph configuration object."""
    return AgentConfig(
        max_steps=args.max_steps,
        max_retrieval_rounds=args.max_retrieval_rounds,
        max_query_rewrites=args.max_query_rewrites,
        max_generation_attempts=args.max_generation_attempts,
        max_llm_calls=args.max_llm_calls,
        claim_llm_budget=args.claim_llm_budget,
        enable_tool_orchestration=not args.legacy_retrieval_graph,
    )


def main() -> int:
    args = parse_args()
    from src.generation.provider import build_generation_answerer
    from src.retrieval.runtime import build_retrieval_runtime
    from src.structured.agent_bridge import load_structured_context
    from src.utils.io import read_jsonl

    chunks = read_jsonl(args.chunks_path.resolve())
    runtime = build_retrieval_runtime(
        chunks=chunks,
        output_dir=args.output_dir.resolve(),
        model_cache_dir=args.model_cache_dir.resolve(),
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
        runtime_profile=args.runtime_profile,
        embedding_device=args.embedding_device,
        reranker_device=args.reranker_device,
        device=args.device,
    )
    answerer = build_generation_answerer(
        provider=args.llm_provider,
        local_model_name="",
        remote_model_name=args.llm_model,
        cache_dir=args.model_cache_dir.resolve(),
        device=args.device or "cuda",
    )
    config = build_agent_config(args)
    llm = getattr(answerer, "llm", None)
    if llm is None:
        raise ValueError("agentic-rag mode requires an answerer exposing the generic `llm` provider.")
    llm_judge = build_claim_llm_judge(
        llm,
        budget=config.claim_llm_budget,
        max_tokens=config.claim_max_tokens,
    )
    fact_store, company_aliases = load_structured_context(args.facts_path, args.company_aliases_path)
    final_state = run_agentic_rag(
        runtime=runtime,
        answerer=answerer,
        llm=llm,
        config=config,
        query=args.query,
        domain_hint=args.domain_hint,
        question_type=args.question_type,
        fact_store=fact_store,
        company_aliases=company_aliases,
        llm_judge=llm_judge,
    )
    print(
        json.dumps(
            {
                "final_answer": final_state.get("final_answer"),
                "termination_reason": final_state.get("termination_reason"),
                "counters": {
                    "step_count": final_state.get("step_count"),
                    "retrieval_count": final_state.get("retrieval_count"),
                    "rewrite_count": final_state.get("rewrite_count"),
                    "generation_count": final_state.get("generation_count"),
                    "llm_call_count": final_state.get("llm_call_count"),
                    "prompt_tokens": final_state.get("prompt_tokens"),
                    "completion_tokens": final_state.get("completion_tokens"),
                    "total_tokens": final_state.get("total_tokens"),
                },
                "rewritten_queries": final_state.get("rewritten_queries"),
                "retrieval_history": final_state.get("retrieval_history"),
                "tool_calls": final_state.get("tool_calls"),
                "trajectory_events": final_state.get("trajectory_events"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
