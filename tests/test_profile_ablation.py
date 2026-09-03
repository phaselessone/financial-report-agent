"""Strict four-profile ablation orchestration."""

from __future__ import annotations

import pytest

from src.agent.config import AgentConfig
from src.agent.claims import stable_claim_id
from src.evaluation.eval_bundle import (
    EvalBundleIntegrityError,
    build_corpus_asset_manifest,
    require_comparable_eval_bundles,
)
from src.evaluation.hard_case_benchmark import (
    canonical_reviewed_cases_hash,
    load_cases,
    validate_reviewed_manifest,
)
from src.evaluation.profile_ablation import TreatmentIntegrityError, run_profile_matrix
from src.evaluation.profile_runtime import (
    PROFILE_IDS,
    PROFILE_SPECS,
    build_profile_executor,
)
from src.evaluation.run_identity import (
    RunIdentity,
    canonical_sha256,
    hash_benchmark_rows,
    hash_case_ids,
)
from src.utils.io import write_jsonl
from tests.test_agent_flow import FakeLLM, make_row, supported_draft
from tests.test_hard_case_benchmark import _reviewed_case, _reviewed_manifest
from src.structured.schema import FinancialFact


def _identity(cases) -> RunIdentity:
    return RunIdentity(
        benchmark_profile="current-dev",
        benchmark_version="contract-v1",
        benchmark_hash=hash_benchmark_rows(cases),
        case_ids_hash=hash_case_ids(case["case_id"] for case in cases),
        case_count=len(cases),
        corpus_hash="corpus-sha",
        model="answer-model",
        provider="provider",
        model_revision="revision-1",
        temperature=0.0,
        prompt_version="prompt-v1",
        budgets={"max_tokens": 1000},
        feature_flags={
            "runtime": {"offline": True, "retrieval_source": "shared-corpus"},
            "treatments": {},
        },
        git_commit="commit-sha",
        source_manifest_hash="manifest-sha",
    )


def _write_chunks(path, cases) -> None:
    write_jsonl(
        path,
        (
            {
                "chunk_id": evidence["evidence_id"],
                "evidence_id": evidence["evidence_id"],
                "doc_id": evidence["source"],
                "page": 1,
                "text": evidence["text"],
            }
            for case in cases
            for evidence in case["evidence"]
        ),
    )


def test_four_profiles_execute_all_cases_and_write_comparable_independent_bundles(tmp_path) -> None:
    cases = load_cases()
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path, cases)
    calls: list[tuple[str, str]] = []

    def executor(spec, case):
        calls.append((spec.profile_id, case["case_id"]))
        evidence_id = case["required_evidence_ids"][0]
        source = case["evidence"][0]["source"]
        abstained = bool(case["must_abstain"])
        claim_text = str(case["gold_answer"])
        return {
            "case_id": case["case_id"],
            "answer": case["gold_answer"],
            "abstained": abstained,
            "cited_evidence_ids": [] if abstained else [evidence_id],
            "used_evidence_ids": [] if abstained else [evidence_id],
            "citations": [] if abstained else [
                {"evidence_id": evidence_id, "doc_id": source, "page": 1}
            ],
            "claims": [] if abstained else [
                {
                    "claim_id": stable_claim_id(claim_text),
                    "text": claim_text,
                    "claim_type": "EXTRACTED",
                    "is_core": True,
                    "source_step_ids": [],
                    "evidence_ids": [evidence_id],
                    "calculation_id": None,
                    "parent_claim_ids": [],
                    "verification": {
                        "status": "ENTAILED",
                        "method": "deterministic",
                        "score": 1.0,
                        "reasons": ["fixture_evidence_match"],
                    },
                }
            ],
            "calculations": {},
            "trace_events": [
                {
                    "event_type": "node",
                    "node": "fixture_retrieve",
                    "status": "SUCCESS",
                    "profile": spec.profile_id,
                    "effective_treatments": spec.effective_treatments,
                },
                {
                    "event_type": "node",
                    "node": "fixture_answer",
                    "status": "SUCCESS",
                    "profile": spec.profile_id,
                    "effective_treatments": spec.effective_treatments,
                },
            ],
            "effective_treatments": spec.effective_treatments,
        }

    result = run_profile_matrix(
        cases=cases,
        base_identity=_identity(cases),
        executor=executor,
        output_root=tmp_path,
        chunks_path=chunks_path,
    )

    assert tuple(result.bundles) == PROFILE_IDS
    assert len(calls) == len(PROFILE_IDS) * len(cases)
    assert len({bundle.identity.run_id for bundle in result.bundles.values()}) == 4
    require_comparable_eval_bundles(list(result.bundles.values()))
    for profile, bundle in result.bundles.items():
        expected = PROFILE_SPECS[profile].effective_treatments
        assert bundle.path.parent == tmp_path / "eval"
        assert bundle.metadata["profile"] == profile
        assert bundle.metadata["evidence_scope"] == "shared-corpus"
        assert bundle.metadata["mode"] == (
            "agentic" if PROFILE_SPECS[profile].enters_graph else "baseline"
        )
        assert len(bundle.per_case) == 100
        assert len(bundle.trajectories) == 100
        assert dict(bundle.identity.feature_flags["treatments"]) == expected
        assert expected["pipeline"] == (
            "agentic" if PROFILE_SPECS[profile].enters_graph else "baseline"
        )
        assert bundle.config["treatments"] == expected
        assert all(row["effective_treatments"] == expected for row in bundle.per_case)
        assert all(row["effective_treatments"] == expected for row in bundle.trajectories)
        assert all(len(row["trajectory_events"]) == 2 for row in bundle.trajectories)
        assert all(
            row["claims"] if not row["abstained"] else row["claims"] == []
            for row in bundle.trajectories
        )
        assert all("citations" in row for row in bundle.trajectories)
        assert all(row["run_id"] == bundle.identity.run_id for row in bundle.trajectories)
        assert all(
            row["run_identity"] == bundle.identity.to_dict()
            for row in bundle.trajectories
        )
        if PROFILE_SPECS[profile].enters_graph:
            assert bundle.metadata["integrity_verdict"]["status"] == "READY"
        assert bundle.identity.comparison_identity() == _identity(cases).comparison_identity()
    assert result.summary["performance_claim_allowed"] is False
    assert result.summary["status"] == "SYNTHETIC_CONTRACT_ONLY"
    assert set(result.summary["bundle_paths"]) == set(PROFILE_SPECS)
    assert result.summary["evidence_integrity"]["baseline-rag"]["status"] == "NOT_APPLICABLE"
    assert all(
        result.summary["evidence_integrity"][profile]["status"] == "READY"
        for profile in ("agentic-rag", "structured-agent", "full-agent")
    )
    assert result.summary["retrieval_source"] == "shared-corpus"

    result.bundles["agentic-rag"].metadata.pop("integrity_verdict")
    with pytest.raises(EvalBundleIntegrityError, match="READY integrity verdict"):
        require_comparable_eval_bundles(list(result.bundles.values()))


@pytest.mark.parametrize("corrupt", ["prediction", "trace"])
def test_treatment_mismatch_fails_closed_before_writing_a_bundle(tmp_path, corrupt) -> None:
    cases = load_cases()
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path, cases)

    def executor(spec, case):
        expected = spec.effective_treatments
        prediction_treatments = dict(expected)
        trace_treatments = dict(expected)
        if corrupt == "prediction":
            prediction_treatments["claim_verification"] = not expected["claim_verification"]
        else:
            trace_treatments["claim_verification"] = not expected["claim_verification"]
        return {
            "case_id": case["case_id"],
            "answer": case["gold_answer"],
            "abstained": case["must_abstain"],
            "cited_evidence_ids": case["required_evidence_ids"],
            "effective_treatments": prediction_treatments,
            "trace_events": [
                {
                    "event_type": "node",
                    "node": "fixture_executor",
                    "status": "SUCCESS",
                    "effective_treatments": trace_treatments,
                }
            ],
        }

    with pytest.raises(TreatmentIntegrityError, match="effective_treatments"):
        run_profile_matrix(
            cases=cases,
            base_identity=_identity(cases),
            executor=executor,
            output_root=tmp_path,
            chunks_path=chunks_path,
        )

    assert not (tmp_path / "eval").exists()


@pytest.mark.parametrize(
    "bad_asset", ["missing", "empty", "invalid-json", "missing-id", "missing-location"]
)
def test_profile_matrix_rejects_missing_or_bad_chunks_before_execution(
    tmp_path, bad_asset
) -> None:
    cases = load_cases()
    chunks_path = tmp_path / "chunks.jsonl"
    if bad_asset == "empty":
        chunks_path.write_text("", encoding="utf-8")
    elif bad_asset == "invalid-json":
        chunks_path.write_text("{broken\n", encoding="utf-8")
    elif bad_asset == "missing-id":
        chunks_path.write_text('{"doc_id":"report","page":1}\n', encoding="utf-8")
    elif bad_asset == "missing-location":
        chunks_path.write_text('{"chunk_id":"chunk-1","text":"no locator"}\n', encoding="utf-8")
    calls = 0

    def executor(_spec, _case):
        nonlocal calls
        calls += 1
        raise AssertionError("executor must not run when chunks are invalid")

    expected_error = FileNotFoundError if bad_asset == "missing" else ValueError
    with pytest.raises(expected_error, match="chunks"):
        run_profile_matrix(
            cases=cases,
            base_identity=_identity(cases),
            executor=executor,
            output_root=tmp_path,
            chunks_path=chunks_path,
        )

    assert calls == 0
    assert not (tmp_path / "eval").exists()


def test_profile_matrix_rejects_case_set_mismatch_before_execution(tmp_path) -> None:
    cases = load_cases()
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path, cases)
    identity_row = _identity(cases).to_dict()
    identity_row["case_ids_hash"] = "0" * 64
    calls = 0

    def executor(_spec, _case):
        nonlocal calls
        calls += 1
        raise AssertionError("executor must not run for a mismatched case set")

    with pytest.raises(ValueError, match="case_ids_hash"):
        run_profile_matrix(
            cases=cases,
            base_identity=RunIdentity.from_dict(identity_row),
            executor=executor,
            output_root=tmp_path,
            chunks_path=chunks_path,
        )

    assert calls == 0
    assert not (tmp_path / "eval").exists()


def test_reviewed_profile_matrix_rejects_tampered_declared_corpus_asset_before_execution(
    tmp_path,
) -> None:
    cases = [_reviewed_case(case_id="REV-ASSET-BINDING-01", category="simple_factual")]
    reviewed_manifest = validate_reviewed_manifest(
        _reviewed_manifest(cases, category_quotas={"simple_factual": 1}),
        cases,
    )
    chunks_path = tmp_path / "chunks.jsonl"
    facts_path = tmp_path / "facts.jsonl"
    aliases_path = tmp_path / "aliases.json"
    _write_chunks(chunks_path, cases)
    facts_path.write_text('{"fact_id":"fact-1"}\n', encoding="utf-8")
    aliases_path.write_text('{"Company":["Company"]}', encoding="utf-8")
    corpus_manifest = build_corpus_asset_manifest(
        chunks_path=chunks_path,
        facts_path=facts_path,
        aliases_path=aliases_path,
    )
    base_identity = RunIdentity(
        benchmark_profile="current-dev",
        benchmark_version="reviewed-v1",
        benchmark_hash=canonical_reviewed_cases_hash(cases),
        case_ids_hash=hash_case_ids(case["case_id"] for case in cases),
        case_count=len(cases),
        corpus_hash=corpus_manifest["corpus_hash"],
        model="answer-model",
        provider="provider",
        model_revision="revision-1",
        temperature=0.0,
        prompt_version="prompt-v1",
        budgets={"max_tokens": 1000},
        feature_flags={
            "runtime": {
                "retrieval_source": "shared-corpus",
                "reviewed_manifest_hash": canonical_sha256(reviewed_manifest),
                "reviewed_target_count": None,
                "corpus_asset_manifest_hash": corpus_manifest["manifest_hash"],
            },
            "treatments": {},
        },
        git_commit="commit-sha",
        source_manifest_hash="manifest-sha",
    )
    aliases_path.write_text('{"Company":["Changed"]}', encoding="utf-8")
    calls = 0

    def executor(_spec, _case):
        nonlocal calls
        calls += 1
        raise AssertionError("executor must not run for mismatched corpus assets")

    with pytest.raises(EvalBundleIntegrityError, match="company_aliases"):
        run_profile_matrix(
            cases=cases,
            base_identity=base_identity,
            executor=executor,
            output_root=tmp_path,
            chunks_path=chunks_path,
            corpus_asset_manifest=corpus_manifest,
            reviewed_manifest=reviewed_manifest,
        )

    assert calls == 0
    assert not (tmp_path / "eval").exists()


def test_profile_executor_runtime_error_is_recorded_per_case_without_aborting_matrix(
    tmp_path,
) -> None:
    evidence_text = "Company revenue was 100."
    case = _reviewed_case(
        case_id="REV-PROFILE-ERROR-01",
        category="simple_factual",
        question="What was Company revenue?",
        evidence=[
            {
                "evidence_id": "e-runtime",
                "text": evidence_text,
                "source": "annual-report",
                "source_id": "annual-report-set",
            }
        ],
        gold_answer=evidence_text,
        gold_claims=[
            {
                "claim_id": "g-runtime",
                "text": evidence_text,
                "status": "ENTAILED",
                "evidence_ids": ["e-runtime"],
            }
        ],
        gold_calculations=[],
        required_tools=[],
        partial_answer_gold={"allowed": False, "required_claim_ids": ["g-runtime"]},
    )
    cases = [case]
    reviewed_manifest = validate_reviewed_manifest(
        _reviewed_manifest(cases, category_quotas={"simple_factual": 1}), cases
    )
    chunks_path = tmp_path / "chunks.jsonl"
    facts_path = tmp_path / "facts.jsonl"
    aliases_path = tmp_path / "aliases.json"
    _write_chunks(chunks_path, cases)
    facts_path.write_text('{"fact_id":"fact-runtime"}\n', encoding="utf-8")
    aliases_path.write_text('{"Company":["Company"]}', encoding="utf-8")
    corpus_manifest = build_corpus_asset_manifest(
        chunks_path=chunks_path,
        facts_path=facts_path,
        aliases_path=aliases_path,
    )
    base_identity = RunIdentity(
        benchmark_profile="current-dev",
        benchmark_version="reviewed-v1",
        benchmark_hash=canonical_reviewed_cases_hash(cases),
        case_ids_hash=hash_case_ids(case["case_id"] for case in cases),
        case_count=1,
        corpus_hash=corpus_manifest["corpus_hash"],
        model="answer-model",
        provider="provider",
        model_revision="revision-1",
        temperature=0.0,
        prompt_version="prompt-v1",
        budgets={"max_tokens": 1000},
        feature_flags={
            "runtime": {
                "retrieval_source": "shared-corpus",
                "reviewed_manifest_hash": canonical_sha256(reviewed_manifest),
                "reviewed_target_count": None,
                "corpus_asset_manifest_hash": corpus_manifest["manifest_hash"],
            },
            "treatments": {},
        },
        git_commit="commit-sha",
        source_manifest_hash="manifest-sha",
    )

    def executor(spec, _case):
        if spec.profile_id == "full-agent":
            raise RuntimeError("provider unavailable")
        claim_id = stable_claim_id(evidence_text)
        return {
            "case_id": case["case_id"],
            "answer": evidence_text,
            "abstained": False,
            "cited_evidence_ids": ["e-runtime"],
            "used_evidence_ids": ["e-runtime"],
            "citations": [
                {"evidence_id": "e-runtime", "doc_id": "annual-report", "page": 1}
            ],
            "claims": [
                {
                    "claim_id": claim_id,
                    "text": evidence_text,
                    "claim_type": "EXTRACTED",
                    "is_core": True,
                    "source_step_ids": [],
                    "evidence_ids": ["e-runtime"],
                    "calculation_id": None,
                    "parent_claim_ids": [],
                    "verification": {
                        "status": "ENTAILED",
                        "method": "deterministic",
                        "score": 1.0,
                        "reasons": ["fixture_evidence_match"],
                    },
                }
            ],
            "calculations": {},
            "trace_events": [
                {
                    "event_id": f"{spec.profile_id}-success",
                    "event_type": "node",
                    "node": "fixture_executor",
                    "step": 0,
                    "status": "SUCCESS",
                    "dependencies": [],
                    "recovery_of": [],
                    "effective_treatments": spec.effective_treatments,
                }
            ],
            "effective_treatments": spec.effective_treatments,
        }

    result = run_profile_matrix(
        cases=cases,
        base_identity=base_identity,
        executor=executor,
        output_root=tmp_path,
        chunks_path=chunks_path,
        corpus_asset_manifest=corpus_manifest,
        reviewed_manifest=reviewed_manifest,
    )

    failed_bundle = result.bundles["full-agent"]
    failed_trace = failed_bundle.trajectories[0]
    assert failed_trace["failed"] is True
    assert failed_trace["abstained"] is True
    assert failed_trace["error_type"] == "RuntimeError"
    assert failed_trace["failure_attribution"]["has_failure"] is True
    assert failed_bundle.metadata["integrity_verdict"]["status"] == "READY"
    assert failed_bundle.metrics["Answer Contract Match Rate"] == 0.0


class _SingleFactStore:
    def __init__(self, fact):
        self.fact = fact

    def query_many(self, **_coordinates):
        return [self.fact]


class _RepeatingAnswerer:
    def __init__(self, draft):
        self._draft = dict(draft)
        self.llm_calls = []

    def answer(self, **_kwargs):
        return {
            **self._draft,
            "citations": [dict(row) for row in self._draft.get("citations", [])],
            "used_evidence_ids": list(self._draft.get("used_evidence_ids", [])),
        }


class _StaticRuntime:
    def __init__(self, row):
        self.row = dict(row)

    def search(self, _query):
        rows = [dict(self.row)]
        return {
            "query_mode": "shared_corpus",
            "numeric_query": True,
            "dense_rows": rows,
            "bm25_rows": rows,
            "hybrid_rows": rows,
            "rerank_rows": rows,
            "timings": {},
        }


def test_real_structured_profile_non_abstain_trace_passes_strict_integrity(tmp_path) -> None:
    evidence_text = "贵州茅台2025年营业收入为100元。"
    evidence = {
        "evidence_id": "E-STRUCTURED-1",
        "text": evidence_text,
        "source": "annual-report-2025",
        "source_id": "annual-report-set",
    }
    case = _reviewed_case(
        case_id="REV-STRUCTURED-INTEGRITY-01",
        category="simple_factual",
        question=evidence_text,
        evidence=[
            evidence,
        ],
        gold_answer=evidence_text,
        gold_claims=[
            {
                "claim_id": "gold-structured-1",
                "text": evidence_text,
                "status": "ENTAILED",
                "evidence_ids": ["E-STRUCTURED-1"],
            }
        ],
        gold_calculations=[],
        required_tools=[],
        partial_answer_gold={
            "allowed": False,
            "required_claim_ids": ["gold-structured-1"],
        },
    )
    cases = [case]
    manifest = validate_reviewed_manifest(
        _reviewed_manifest(cases, category_quotas={"simple_factual": 1}),
        cases,
    )
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path, cases)
    facts_path = tmp_path / "facts.jsonl"
    aliases_path = tmp_path / "aliases.json"
    facts_path.write_text('{"fact_id":"structured-fixture"}\n', encoding="utf-8")
    aliases_path.write_text(
        '{"贵州茅台":["贵州茅台","茅台"]}', encoding="utf-8"
    )
    corpus_manifest = build_corpus_asset_manifest(
        chunks_path=chunks_path,
        facts_path=facts_path,
        aliases_path=aliases_path,
    )
    answerer = _RepeatingAnswerer(supported_draft(evidence_text))
    fact = FinancialFact.from_dict(
        {
            "company": "贵州茅台",
            "metric": "revenue",
            "period": {"kind": "FY", "year": 2025},
            "value_type": "actual",
            "value": "100",
            "unit": "元",
            "doc_id": "annual-report-2025",
            "page": 1,
            "evidence_id": "E-STRUCTURED-1",
            "raw_value": "100元",
            "source_span": evidence_text,
        }
    )
    runtime_row = make_row(
        chunk_id="E-STRUCTURED-1",
        doc_id="annual-report-2025",
        text=evidence_text,
    )
    runtime_row["evidence_id"] = "E-STRUCTURED-1"
    runtime = _StaticRuntime(runtime_row)
    executor = build_profile_executor(
        answerer=answerer,
        llm=FakeLLM([]),
        base_config=AgentConfig(
            max_query_rewrites=0,
            max_generation_attempts=1,
            strict_claim_verification=True,
        ),
        fact_store=_SingleFactStore(fact),
        company_aliases={"贵州茅台": ["贵州茅台", "茅台"]},
        retrieval_runtime=runtime,
    )
    base_identity = RunIdentity(
        benchmark_profile="current-dev",
        benchmark_version="reviewed-v1",
        benchmark_hash=canonical_reviewed_cases_hash(cases),
        case_ids_hash=hash_case_ids(case["case_id"] for case in cases),
        case_count=len(cases),
        corpus_hash=corpus_manifest["corpus_hash"],
        model="answer-model",
        provider="provider",
        model_revision="revision-1",
        temperature=0.0,
        prompt_version="prompt-v1",
        budgets={"max_tokens": 1000},
        feature_flags={
            "runtime": {
                "offline": True,
                "retrieval_source": "shared-corpus",
                "reviewed_manifest_hash": canonical_sha256(manifest),
                "reviewed_target_count": None,
                "corpus_asset_manifest_hash": corpus_manifest["manifest_hash"],
            },
            "treatments": {},
        },
        git_commit="commit-sha",
        source_manifest_hash="manifest-sha",
    )

    result = run_profile_matrix(
        cases=cases,
        base_identity=base_identity,
        executor=executor,
        output_root=tmp_path,
        chunks_path=chunks_path,
        corpus_asset_manifest=corpus_manifest,
        reviewed_manifest=manifest,
    )

    structured = result.bundles["structured-agent"]
    assert structured.metadata["integrity_verdict"]["status"] == "READY"
    assert structured.trajectories[0]["abstained"] is False
    assert structured.trajectories[0]["claims"]
    assert all(
        claim["verification"]["status"] == "ENTAILED"
        for claim in structured.trajectories[0]["claims"]
    )
