"""Single-query agent CLI activation contracts."""

from __future__ import annotations

import sys
from argparse import Namespace

from run_agent import build_agent_config, parse_args


def test_single_query_cli_exposes_explicit_claim_llm_budget(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_agent.py", "--query", "公司的研发投入如何？", "--claim-llm-budget", "2"],
    )
    args = parse_args()
    assert args.claim_llm_budget == 2


def test_single_query_cli_keeps_claim_llm_judge_disabled_by_default(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_agent.py", "--query", "公司的研发投入如何？"])
    assert parse_args().claim_llm_budget == 0


def test_single_query_cli_maps_claim_llm_budget_into_agent_config() -> None:
    args = Namespace(
        max_steps=20,
        max_retrieval_rounds=4,
        max_query_rewrites=2,
        max_generation_attempts=2,
        max_llm_calls=6,
        claim_llm_budget=2,
        legacy_retrieval_graph=False,
    )
    assert build_agent_config(args).claim_llm_budget == 2
