"""Behavior tests for the strict Phase G review CLIs."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from src.agent.claims import stable_claim_id
from src.evaluation.eval_bundle import (
    write_eval_bundle,
    write_eval_bundle_reference,
    write_validated_eval_bundle,
)
from src.evaluation.run_identity import RunIdentity, hash_case_ids


REPO_ROOT = Path(__file__).resolve().parents[1]


def _identity() -> RunIdentity:
    return RunIdentity(
        benchmark_profile="agentic-rag",
        benchmark_version="contract-v1",
        benchmark_hash="benchmark-hash",
        case_ids_hash=hash_case_ids(["case-01"]),
        case_count=1,
        corpus_hash="corpus-hash",
        model="fake-model",
        provider="fake",
        model_revision="fixture-v1",
        temperature=0.0,
        prompt_version="prompt-v1",
        budgets={"max_steps": 8},
        feature_flags={"runtime": {"strict_trace": True}, "treatments": {"pipeline": "agentic"}},
        git_commit="fixture-commit",
        source_manifest_hash="manifest-hash",
    )


def _trace(identity: RunIdentity) -> dict:
    claim_text = "甲公司 2025 年毛利率为 20%。"
    return {
        "case_id": "case-01",
        "question_id": "case-01",
        "query": "甲公司 2025 年毛利率是多少？",
        "final_answer": "甲公司 2025 年毛利率为 20%。",
        "run_id": identity.run_id,
        "run_identity": identity.to_dict(),
        "claims": [
            {
                "claim_id": stable_claim_id(claim_text),
                "claim_type": "EXTRACTED",
                "text": claim_text,
                "is_core": True,
                "source_step_ids": [],
                "evidence_ids": ["E1"],
                "calculation_id": None,
                "parent_claim_ids": [],
                "verification": {
                    "status": "ENTAILED",
                    "method": "deterministic",
                    "score": 1.0,
                    "reasons": [],
                },
                "supported": True,
            }
        ],
        "citations": [{"evidence_id": "E1", "doc_id": "report.pdf", "page": 18}],
        "used_evidence_ids": ["E1"],
        "abstained": False,
        "calculations": {},
        "tool_calls": [],
        "trajectory_events": [
            {"event_id": "event-1", "node": "finalize", "status": "SUCCESS"}
        ],
        "dependency_coverage": {},
        "failure_attribution": {"has_failure": False, "root_cause": None},
    }


def _write_bundle(root: Path, *, trace: dict | None = None, validated: bool = True):
    identity = _identity()
    row = trace or _trace(identity)
    chunks = root / "chunks.jsonl"
    chunks.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in (
                {"chunk_id": "E1", "doc_id": "report.pdf", "page_start": 18, "page_end": 18, "text": "毛利率 20%"},
                {"chunk_id": "E2", "doc_id": "draft.pdf", "page_start": 2, "page_end": 2, "text": "未经验证的草稿"},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    writer = write_validated_eval_bundle if validated else write_eval_bundle
    writer_kwargs = {"chunks_path": chunks} if validated else {}
    bundle = writer(
        root / "eval",
        identity=identity,
        config={"runtime": {"strict_trace": True}, "treatments": {"pipeline": "agentic"}},
        metrics={"process": {}},
        per_case=[row],
        trajectories=[row],
        metadata={"mode": "agentic"},
        **writer_kwargs,
    )
    return bundle, chunks


def test_evidence_integrity_cli_validates_strict_bundle_identity(tmp_path: Path) -> None:
    bundle, chunks = _write_bundle(tmp_path)
    reference = tmp_path / "reports" / "agent_eval_bundle.json"
    write_eval_bundle_reference(reference, bundle)
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "evidence_integrity_gate.py"),
            "--bundle-path",
            str(reference),
            "--chunks-path",
            str(chunks),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["status"] == "READY"
    assert report["bundle"]["run_id"] == bundle.identity.run_id
    assert report["bundle"]["identity_hash"] == bundle.identity.identity_hash


def test_evidence_integrity_cli_blocks_trace_profile_mismatch(tmp_path: Path) -> None:
    identity = _identity()
    trace = _trace(identity)
    trace["run_identity"] = {**identity.to_dict(), "benchmark_profile": "baseline-rag"}
    bundle, chunks = _write_bundle(tmp_path, trace=trace, validated=False)

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "evidence_integrity_gate.py"),
            "--bundle-path",
            str(bundle.path),
            "--chunks-path",
            str(chunks),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    report = json.loads(completed.stdout)
    assert any(issue["code"] == "TRACE_RUN_IDENTITY_MISMATCH" for issue in report["issues"])


def test_evidence_integrity_cli_reports_tampered_bundle_without_traceback(tmp_path: Path) -> None:
    bundle, chunks = _write_bundle(tmp_path)
    (bundle.path / "config.json").write_text('{"tampered": true}\n', encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "evidence_integrity_gate.py"),
            "--bundle-path",
            str(bundle.path),
            "--chunks-path",
            str(chunks),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    report = json.loads(completed.stdout)
    assert report["status"] == "BLOCKED"
    assert report["issues"][0]["code"] == "BUNDLE_LOAD_ERROR"
    assert "Traceback" not in completed.stderr


def test_observability_cli_renders_verified_strict_bundle_fields(tmp_path: Path) -> None:
    identity = _identity()
    trace = _trace(identity)
    trace.update(
        {
            "prompt_tokens": 100,
            "completion_tokens": 23,
            "total_tokens": 123,
            "end_to_end_latency_ms": 45.6,
            "api_latency_ms_total": 12.3,
            "claims": [
                *trace["claims"],
                {
                    "claim_id": stable_claim_id("未经充分支持的描述。"),
                    "claim_type": "EXTRACTED",
                    "text": "未经充分支持的描述。",
                    "is_core": False,
                    "source_step_ids": [],
                    "evidence_ids": ["E2"],
                    "calculation_id": None,
                    "parent_claim_ids": [],
                    "verification": {
                        "status": "INSUFFICIENT",
                        "method": "deterministic",
                        "score": 0.0,
                        "reasons": ["fixture"],
                    },
                    "supported": False,
                },
            ],
            "evidence_rows": [
                {"evidence_id": "E1", "doc_id": "report.pdf", "page": 18, "text": "verified citation text"},
                {"evidence_id": "E2", "doc_id": "draft.pdf", "page": 2, "text": "unverified evidence text"},
            ],
            "calculations": {
                    "K1": {
                        "calculation_id": "K1",
                        "status": "SUCCESS",
                        "verified": True,
                        "operation": "gross_margin",
                        "inputs": [
                            {"name": "gross_profit", "value": "20", "unit": "元", "evidence_id": "E1"},
                            {"name": "revenue", "value": "100", "unit": "元", "evidence_id": "E1"},
                        ],
                        "input_calculation_ids": [],
                        "source_step_ids": [],
                        "evidence_ids": ["E1"],
                        "result": {"value": "0.2", "formatted": "20%", "unit": ""},
                    }
            },
            "reasoning_plan": {
                "plan_id": "P1",
                "steps": [
                    {"step_id": "lookup_margin", "kind": "LOOKUP", "depends_on": [], "required": True}
                ],
                "answer_requirement_ids": ["lookup_margin"],
            },
            "reasoning_step_results": [
                {"step_id": "lookup_margin", "status": "SUCCESS", "evidence_ids": ["E1"], "latency_ms": 2.5}
            ],
            "tool_calls": [
                {
                    "tool_call_id": "T1",
                    "tool_name": "structured_lookup",
                    "status": "SUCCESS",
                    "latency_ms": 1.2,
                }
            ],
            "dependency_coverage": {"decision": "complete", "coverage_ratio": 1.0},
            "trajectory_events": [{"node": "finalize", "status": "SUCCESS", "latency_ms": 0.8}],
            "failure_attribution": {"has_failure": False, "root_cause": None, "recovered": False},
        }
    )
    bundle, chunks = _write_bundle(tmp_path, trace=trace)
    output = tmp_path / "observability.html"

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "observability_demo.py"),
            "--bundle-path",
            str(bundle.path),
            "--chunks-path",
            str(chunks),
            "--question-id",
            "case-01",
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    html = output.read_text(encoding="utf-8")
    for expected in (
        "Trace source and integrity",
        "READY",
        identity.run_id,
        "agentic-rag",
        "INSUFFICIENT",
        "verified citation text",
        "Calculation trace",
        "20%",
        "Reasoning LOOKUP",
        "Tool structured_lookup",
        "Dependency coverage",
        "Failure attribution",
        "Token and latency detail",
    ):
        assert expected in html
    assert "unverified evidence text" not in html


def test_observability_cli_never_presents_legacy_trace_as_ready(tmp_path: Path) -> None:
    identity = _identity()
    trace_path = tmp_path / "legacy.jsonl"
    trace_path.write_text(json.dumps(_trace(identity), ensure_ascii=False) + "\n", encoding="utf-8")
    output = tmp_path / "legacy.html"

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "observability_demo.py"),
            "--trace-path",
            str(trace_path),
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    html = output.read_text(encoding="utf-8")
    assert "legacy_trace_unverified" in html
    assert "BLOCKED" in html
    assert "strict_eval_bundle" not in html


def test_observability_cli_blocks_bundle_without_persisted_ready_verdict(tmp_path: Path) -> None:
    bundle, chunks = _write_bundle(tmp_path, validated=False)
    output = tmp_path / "unvalidated-bundle.html"

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "observability_demo.py"),
            "--bundle-path",
            str(bundle.path),
            "--chunks-path",
            str(chunks),
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    html = output.read_text(encoding="utf-8")
    assert "strict_eval_bundle" in html
    assert "BLOCKED" in html
    assert "PERSISTED_INTEGRITY_VERDICT_REQUIRED" in html
    assert '"integrity_status": "READY"' not in html


def test_ci_workflow_exposes_each_required_deterministic_gate() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    install = "uv pip install --system --torch-backend cpu -r requirements.lock"
    offline = "PIP_NO_INDEX=1"
    assert workflow.index(install) < workflow.index(offline)
    assert "astral-sh/setup-uv@" in workflow
    assert "ubuntu-latest" in workflow
    assert "windows-latest" in workflow
    assert "uv pip check --system" in workflow
    for command in (
        "python -m compileall -q src scripts",
        "python -m ruff check .",
        "python -m ruff format --check .",
        "python -m pytest -q --ignore=tests/test_full_seed_restore.py",
        "python scripts/phase0_gate.py smoke",
        "python scripts/hard_case_subset_gate.py --per-category 1",
        "python scripts/strict_observability_smoke.py --output-root outputs/strict_observability_smoke",
        "python -m pytest -q tests/test_eval_bundle.py tests/test_run_agent_eval.py tests/test_run_metadata.py",
        "python -m pytest -q tests/test_evidence_integrity.py tests/test_phase_g_cli.py",
    ):
        assert command in workflow


def test_ci_quality_gate_is_version_pinned_and_explicitly_scoped() -> None:
    lock = (REPO_ROOT / "requirements.lock").read_text(encoding="utf-8")
    ruff_config = (REPO_ROOT / "ruff.toml").read_text(encoding="utf-8")
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")

    assert "ruff==0.12.12" in lock
    for maintained_path in (
        "run_profile_ablation.py",
        "scripts/phase0_gate.py",
        "scripts/evidence_integrity_gate.py",
        "scripts/hard_case_subset_gate.py",
        "scripts/strict_observability_smoke.py",
        "src/evaluation/run_identity.py",
        "src/evaluation/eval_bundle.py",
        "src/evaluation/evidence_integrity.py",
        "src/evaluation/offline_contract_runtime.py",
        "src/evaluation/profile_ablation.py",
        "src/evaluation/profile_runtime.py",
    ):
        assert f'"{maintained_path}"' in ruff_config
    assert "lint:\n\t$(PYTHON) -m ruff check ." in makefile
    assert "format-check:\n\t$(PYTHON) -m ruff format --check ." in makefile


def test_dependency_lock_selects_cpu_torch_on_windows_and_linux() -> None:
    lock = (REPO_ROOT / "requirements.lock").read_text(encoding="utf-8").lower()
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    ).lower()

    assert "--extra-index-url" not in lock
    assert "--torch-backend cpu" in workflow
    assert "torch==2.13.0+cpu" in lock
    assert "hf-xet==1.6.0" in lock
    assert "sys_platform == \"linux\"" in lock
    assert "sys_platform == \"win32\"" in lock
    assert "nvidia-" not in lock
    assert "triton==" not in lock


def test_dependency_lock_covers_top_level_constraints_on_ci_platforms() -> None:
    from packaging.markers import default_environment
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    lock_rows = [
        Requirement(line)
        for line in (REPO_ROOT / "requirements.lock").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "--"))
    ]
    top_level = [
        Requirement(line)
        for line in (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert all(any(spec.operator == "==" for spec in row.specifier) for row in lock_rows)
    for platform in ("linux", "win32"):
        environment = default_environment()
        environment["sys_platform"] = platform
        active: dict[str, Requirement] = {}
        for row in lock_rows:
            if row.marker is not None and not row.marker.evaluate(environment):
                continue
            normalized = canonicalize_name(row.name)
            assert normalized not in active, f"duplicate active lock row for {normalized} on {platform}"
            active[normalized] = row

        for requirement in top_level:
            locked = active[canonicalize_name(requirement.name)]
            exact_version = next(
                spec.version for spec in locked.specifier if spec.operator == "=="
            )
            assert exact_version in requirement.specifier, (
                f"{requirement.name} lock {exact_version} violates {requirement.specifier} "
                f"on {platform}"
            )


def test_makefile_evidence_gate_uses_strict_bundle_reference() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    target = makefile.split("evidence-integrity-gate:", 1)[1].split("\n\n", 1)[0]

    assert "--bundle-path outputs/reports/agent_eval_bundle_dev.json" in target
    assert "--trace-path" not in target


def test_makefile_exposes_offline_strict_observability_smoke() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    target = makefile.split("strict-observability-smoke:", 1)[1].split("\n\n", 1)[0]

    assert "scripts/strict_observability_smoke.py" in target


def test_windows_quick_start_uses_uv_for_dependency_installation() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    windows = readme.split("Windows PowerShell:", 1)[1].split("##", 1)[0]

    assert "uv pip install --python .venv" in windows
    assert "-r requirements.lock" in windows
    assert ".venv\\Scripts\\python.exe -m pip install" not in windows


def test_readme_architecture_matches_the_reasoning_plan_graph() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    for node in (
        "build_reasoning_plan",
        "plan_next_step",
        "execute_step",
        "observe_step_result",
        "dependency_gate",
        "extract_claims",
        "verify_answer",
        "finalize",
    ):
        assert node in readme
    assert "plan_tools -> execute_tool -> observe_tool_result" not in readme
    assert "no successful hosted GitHub Actions run has been supplied" in readme


def test_technical_overview_separates_current_acceptance_from_historical_snapshot() -> None:
    overview = (REPO_ROOT / "docs" / "project_technical_overview_and_setup.md").read_text(
        encoding="utf-8"
    )

    assert "当前验收状态（2026-08-24）" in overview
    assert "Phase 0 readiness: `BLOCKED`" in overview
    assert "strict eval bundle" in overview
    assert "历史快照（非当前验收）" in overview
    assert "LangGraph" in overview
    assert "暂未引入 Milvus、Elasticsearch、LangChain 编排" not in overview


def test_observability_docs_require_a_persisted_bundle_bound_verdict() -> None:
    guide = (REPO_ROOT / "docs" / "observability_demo.md").read_text(encoding="utf-8")

    assert "persisted, bundle-bound `READY` integrity verdict" in guide
    assert "live recheck cannot promote" in guide
