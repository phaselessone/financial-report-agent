# Financial Research PDF RAG / 中文金融研报 PDF 可溯源问答系统

面向中文金融研报 PDF 的证据驱动 RAG 系统。项目覆盖从 PDF 解析、结构化切块、Hybrid+Rerank 检索，到证据约束生成、引用绑定、拒答判断和评测归档的完整闭环。

This is an evidence-grounded RAG system for Chinese financial research PDFs. It focuses on structured PDF ingestion, hybrid retrieval with reranking, citation-aware answer generation, abstention, and reproducible evaluation.

## Overview

金融研报常见多栏排版、表格、图注、页眉页脚、免责声明和跨文档比较问题。普通“向量库 + LLM”流程很容易出现证据错位、引用不准或无证据硬答。本项目的目标是把研报 PDF 转成可检索、可引用、可评测、可复现的问答系统。

核心链路：

```text
PDF ingest
  -> structured pages / cleaned pages / elements
  -> table-protected chunks + optional structured facts
  -> Dense + BM25 hybrid retrieval -> rerank
  -> analyze_query -> build_reasoning_plan (ReasoningPlan)
  -> plan_next_step -> execute_step (LOOKUP / SEARCH / CALCULATE)
  -> observe_step_result -> dependency_gate
  -> synthesize -> extract_claims -> verify_answer -> finalize
  -> calculation provenance + ENTAILED-only citations / abstain
  -> trajectories, deterministic gates, and evaluation bundles
```

## Highlights

- **结构化 PDF ingest**：基于 PyMuPDF 抽取页面块、标题、表格和正文元素，不直接把整篇 PDF 当普通文本切分。
- **表格保护切块**：生产策略为 `table_protected`，优先保护金融研报中的表格、标题层级和数字邻近上下文。
- **Hybrid+Rerank 检索**：组合 `BAAI/bge-m3`、FAISS、BM25、`BAAI/bge-reranker-v2-m3`，兼顾语义召回和关键词匹配。
- **Claim-level provenance contracts**：确定性测试覆盖稳定 claim ID、类型、独立 evidence 映射和三态验证（`ENTAILED` / `CONTRADICTED` / `INSUFFICIENT`）；strict evidence gate 只允许 ENTAILED claim 的证据进入最终引用。
- **Calculation provenance contracts**：确定性测试覆盖同比、环比、CAGR、利润率、比率、差额与百分点变化的 Decimal 计算，以及输入 evidence/fact、公式、结果、舍入与失败状态。
- **受控 Tool Agent contracts**：工具白名单、调用记录、预算停止、重试、去重和错误 trajectory 有离线回归测试；这不等于已经证明真实语料上的回答质量。
- **可复现评测工件**：strict eval bundle 固定 `RunIdentity`、配置/结果哈希、逐题结果和 trajectory；不同 benchmark/profile/runtime 身份不能静默合并。

## Architecture

| Layer | Main files | Responsibility |
| --- | --- | --- |
| Ingest | `src/ingest/*`, `run_pipeline.py` | PDF 解析、清洗、元素抽取、切块、ingest 报告 |
| Retrieval | `src/retrieval/*`, `run_retrieval_eval.py` | Dense/BM25/Hybrid/Rerank 检索运行时和评测 |
| Generation | `src/generation/*`, `run_answer_eval.py` | 证据选择、题型路由、LLM provider、citation、abstain |
| Agent | `src/agent/*`, `run_agent.py` | LangGraph 编排、claim verification、calculation trace、dependency coverage、受控工具调用 |
| Structured facts | `src/structured/*` | 指标注册表、事实标准化/查询、actual 与 forecast 隔离 |
| Evaluation | `src/evaluation/*`, `benchmarks/hard_cases/*` | benchmark、claim/calculation/trajectory 指标、failure attribution、归档与元数据 |
| Ops | `scripts/ragctl.sh`, `scripts/remote_ops.py`, `bootstrap/*` | 本地/远端执行入口、环境检查、SSH/SFTP 辅助 |

## Repository Layout

```text
.
├── bootstrap/                 # environment checks
├── docs/                      # technical review, provenance, and postmortem docs
├── scripts/                   # ragctl, remote helper, smoke scripts
├── src/
│   ├── ingest/                # PDF parser, cleaner, chunker, pipeline
│   ├── retrieval/             # embedding, FAISS, BM25, hybrid retriever, reranker
│   ├── generation/            # answerer, provider, prompt, citation, abstain
│   ├── agent/                 # graph nodes, claims, tools, calculations, dependencies
│   ├── structured/            # normalized financial facts and metric registry
│   └── evaluation/            # benchmark assets, eval metrics, metadata, archive policy
├── benchmarks/hard_cases/     # 100 synthetic, deterministic contract cases
├── tests/                     # regression tests for routing, eval, cache, metadata
├── run_pipeline.py            # PDF ingest entrypoint
├── run_prepare_benchmark.py   # benchmark materialization helper
├── run_retrieval_eval.py      # retrieval evaluation entrypoint
└── run_answer_eval.py         # answer generation/evaluation entrypoint
```

Large local data and generated artifacts are intentionally excluded from GitHub: `pdf/`, `data/`, `outputs/`, `artifacts/`, `models/`, remote sync folders, caches, backups, and zip/tgz packages.

The clean-checkout test suite keeps one minimal financial-business record at
[`tests/fixtures/controlled_financial_business_case.json`](tests/fixtures/controlled_financial_business_case.json).
It uses a fictional issuer and synthetic annual-revenue statement to exercise
structured lookup, provenance, and citation behavior without `data/`, private
PDFs, or raw report text. It is a deterministic public-safe contract fixture,
not a real-report source record or reviewed benchmark evidence.

## Quick Start

### 1. Create environment

推荐使用 [uv](https://docs.astral.sh/uv/) 创建环境并安装依赖。`requirements.lock`
是 CI 与可复现验收使用的精确 runtime/test 闭包；`requirements.txt` 和
`requirements-dev.txt` 是维护依赖时的开发输入，不是验收安装入口。

Linux (the locked CI target) and supported macOS installations:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.lock
```

The current-worktree workflow targets Ubuntu and Windows. Before deterministic-ci
run [`33259417431`](https://github.com/phaselessone/financial-report-agent/actions/runs/33259417431),
no successful hosted GitHub Actions run has been supplied as acceptance evidence;
that push run now supplies green Ubuntu and Windows evidence for commit
`48945449f0f36ae3c0ffb6590ab92f8087b5769e`. The current
`faiss-cpu` wheel requires at least macOS 14 on Apple Silicon and macOS 15 on
Intel; older macOS versions are not part of the supported lock contract.

Windows PowerShell:

```powershell
uv venv .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.lock
```

### 2. Configure runtime

Copy `.env.example` to a local ignored environment file and fill private values outside version control:

```bash
cp .env.example .env.local
```

The repository does not include API keys, private SSH keys, PDFs, indexes, models, or generated evaluation outputs.

`.env`, `.env.deepseek`, `.env.local` and `.env.runtime` are loaded automatically (lowest to highest priority, later files override earlier ones) by both `scripts/use_env.sh` and the Python entrypoints via `src/utils/env.py`.

### 3. Run pipeline

```bash
bash scripts/ragctl.sh doctor
bash scripts/ragctl.sh pipeline --input-dir pdf --output-dir data
bash scripts/ragctl.sh prepare-benchmark
bash scripts/ragctl.sh retrieval-eval
bash scripts/ragctl.sh answer-eval-dev
```

Historical full benchmark commands are separated on purpose:

```bash
bash scripts/ragctl.sh answer-eval-full --model-revision <pinned-provider-revision>
bash scripts/ragctl.sh answer-eval-full-raw --model-revision <pinned-provider-revision>
```

Do not merge these two result sets when reporting metrics.

On Windows, `bash scripts/ragctl.sh ...` requires Git Bash. You can also run the Python entrypoints directly with the venv interpreter, for example:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --input-dir pdf --output-dir data
.\.venv\Scripts\python.exe run_prepare_benchmark.py
.\.venv\Scripts\python.exe run_retrieval_eval.py
.\.venv\Scripts\python.exe run_answer_eval.py
```

## Remote And Artifact Notes

- Historical test counts are not readiness evidence. In the audited local workspace,
  the 50-row full seed, historical results, and their attestation make
  `check-full-seed` READY. Top-level Phase 0 readiness remains `BLOCKED` because
  reviewed semantic labels are absent; historical replay is separately `BLOCKED`
  because the hash-pinned Stage 6 chunks and reviewed corpus attestation are absent.
  Public clean checkouts intentionally omit these reviewed/private assets.
- The canonical remote helper is `scripts/remote_ops.py`, but large recursive artifact transfers should prefer a single remote archive plus checksum verification.
- Evidence packages and generated outputs are intentionally kept out of the public repository boundary.
- The public deterministic suite explicitly excludes `tests/test_full_seed_restore.py`; running that private-asset gate without restored reviewed assets must fail rather than skip or fabricate substitutes.
- Local `.env.*`, `.ssh/`, model caches, PDFs, generated indexes, outputs, artifacts, and backup folders must remain untracked.

## Provenance and Controlled Agent Flow

The deterministic contract suite exercises this single ReasoningPlan route:

```text
analyze query
  -> build_reasoning_plan
  -> plan_next_step -> execute_step -> observe_step_result -> repeat
  -> dependency_gate
  -> optional grade_evidence / rewrite_query recovery
  -> synthesize -> extract_claims -> verify_answer -> finalize
```

`execute_step` is the controlled seam for structured lookup, retrieval search and
provenance-aware calculation. Every query receives a `ReasoningPlan`; single-hop
and multi-hop questions share the same plan/execute/observe/dependency path.
Contract tests cover stable step/call identities, dependency ordering,
deduplication, budget exhaustion, retry limits and no-new-information
termination.

The strict trace contract carries `claim_id`, text, claim type, `is_core`, source-step IDs, claim-specific `evidence_ids`, optional `calculation_id`, parent claim IDs and verification details. The legacy `supported` field is compatibility output derived from `verification.status == "ENTAILED"`; it is not an independent verdict. The deterministic verifier tests fail closed on numeric entity, metric, period, value, unit and actual/forecast type. These are process/integrity guarantees, not reviewed semantic-accuracy results.

The evidence-integrity gate requires each published derived claim to reference a successful `CalculationRecord` with evidence/fact input provenance. Deterministic tests cover the calculator operation, ordered inputs, formula, unit, rounding, result and explicit error status for the currently supported operations; they do not establish that upstream source facts are correct.

Single-hop and multi-hop contract cases use explicit reasoning steps and dependencies. The deterministic coverage gate distinguishes complete, partial and abstain outcomes. Reviewed end-to-end multi-hop accuracy remains unverified.

## Deterministic Gates and Reproduction

All commands below are local/offline with respect to models and providers. They do not load embeddings, download models, call an API, or require a private PDF. Use the repository virtual environment on Windows:

```powershell
# Capture commit, dependency/config hashes, non-secret model configuration, and commands.
.\.venv\Scripts\python.exe scripts\phase0_gate.py baseline

# Run the deterministic Phase 0 smoke groups.
make p0-smoke

# Validate the 10-category balanced hard-case contract subset.
make hard-case-smoke

# Run one real graph execution through a strict READY bundle and HTML review page.
make strict-observability-smoke

# Rebuild the non-publishable current-dev trace from current-local real corpus assets.
make current-dev-trace

# Scan existing ingest JSONL to decide whether M10 OCR/table work should be promoted.
make corpus-quality-gate
```

`scripts/phase0_gate.py check-full-seed` is intentionally strict. Full regression requires both external artifacts—`data/eval_set/answer_eval_seed_full.jsonl` and `artifacts/remote_20260401/outputs_reports/answer_eval_results_full.jsonl`—plus `benchmarks/full/historical-full-raw.attestation.json`. The attestation preserves the historical JSONL files byte-for-byte while binding their pinned SHA-256 values to the reviewed restoration report, provenance commit, custodian, review batch/date, and canonical `historical-full-raw` profile. Explicit `FULL_SEED_SHA256` / `FULL_RESULTS_SHA256` values remain supported and must agree with the attestation when both are supplied. The shipped 48-row current seed is not a substitute. In a clone without the ignored external artifacts, run deterministic smoke or explicitly ignore `tests/test_full_seed_restore.py`; do not delete or weaken the full-seed assertions.

Seed identity is not corpus identity. Before replaying `historical-full-*`, restore
the reviewed corpus sidecar at
`benchmarks/full/historical-stage6-corpus.attestation.json` and run the
content-level gate:

```powershell
.\.venv\Scripts\python.exe scripts\historical_full_evidence_gate.py `
  --chunks-path tmp_stage6_data\chunks\chunks.jsonl
```

The required shape is documented by
`benchmarks/full/historical-stage6-corpus.attestation.schema.json`. The sidecar
must identify the historical release as `corpus:4ff7ba48fd38`, bind the canonical
path to legacy whole-file SHA-1
`4ff7ba48fd38a4400e581fac32c43c4217de1ece` and a freshly computed SHA-256,
and record count, source owner/reference/acquisition time, positive review
identity, and the exact seed/results hashes. The legacy SHA-1 is an identity
anchor recovered from the historical run metadata, not a security signature.
The gate rejects alternate paths, mutable identities, invalid review timestamps,
and unknown sidecar fields before it can report formal readiness. It then requires exact
gold IDs and checks
document, page, filename, and normalized historical snippet content. Historical
results expose text anchors for 42 of the 54 unique gold chunks; the remaining
12 reviewed badcase chunks are reported separately as
`ATTESTED_UNANCHORED` and can be accepted only after the seed attestation is
validated. They are never described as content-verified. An ID-only rebuild is
not acceptable: chunk ordinals can stay stable while their text shifts. Missing,
unattested, unpinned, or content-incompatible Stage 6 data returns
`BLOCKED`/exit code 2. `STAGE6_CHUNKS_SHA256` or
`--expected-chunks-sha256` may pin an audit-only probe, but cannot make the formal
gate READY without the reviewed sidecar. A content-compatible probe is reported
as `DIAGNOSTIC_CONTENT_COMPATIBLE` / `DIAGNOSTIC_ONLY` and still exits with code 2.

`run_answer_eval.py` now invokes this formal gate automatically for
`historical-full-core` and `historical-full-raw` before it reserves a run
directory, reads the candidate corpus, builds indexes, or calls a model. These
profiles also require an explicit immutable `--model-revision`; empty,
`default`, `latest`, `unknown`, and `unversioned` identities are rejected.

Benchmark reports use three non-interchangeable profiles: `current-dev`,
`historical-full-core`, and `historical-full-raw`. The legacy `historical-full`
label is normalized to `historical-full-raw`; metrics from different profiles
must not be merged. Run `scripts/phase0_gate.py readiness` before claiming a
full benchmark or semantic calibration is ready. It writes
`outputs/phase0/readiness.json` and returns `BLOCKED`/exit code 2 when the
reviewed full seed, historical results, their attestation, or reviewed semantic labels are absent
or invalid. `SEMANTIC_LABELS_SHA256` remains mandatory; labels must cover at
least 20 reviewed examples with both entailed and non-entailed classes, bind a
single directional-NLI scorer identity, and pass the precision-first calibration
constraint with at least five predicted positives and five true positives.
Every reviewed row must bind both claim and evidence content through inline text
or SHA-256, use a timezone-bearing review timestamp, and reject moving scorer
revisions such as `main`, `master`, `latest`, or `refs/heads/*`. It never substitutes the 48-row current seed or treats an observed
hash as a trusted expected hash; full-asset hashes come from the reviewed sidecar.

When readiness is `READY`, a reviewed four-profile run may enable the calibrated
directional-NLI layer only by supplying all four inputs:
`--semantic-calibration-report <readiness-or-calibration.json>`,
`--semantic-scorer-config <authorized-runtime-config.json>`,
`--semantic-labels-path <reviewed-labels.jsonl>`, and
`--readiness-report <ready-readiness.json>`. The config contract is documented
under `benchmarks/semantic/`. Before loading a model, the CLI hashes and parses
the actual labels, requires the top-level readiness report to be `READY`, and
cross-checks the labels SHA/path, scorer identity, threshold, calibration
identity, and runtime config. Only then does it load the exact cached model
revision (`local_files_only=true`, `trust_remote_code=false`). A partial set of
inputs, synthetic cases, failed calibration/precision/coverage, or a missing
cached revision stops the matrix before any profile executes.

For completed agent runs, validate the strict bundle and its corpus chunks:

```powershell
.\.venv\Scripts\python.exe scripts\regenerate_dev_trace.py `
  --output-root outputs

.\.venv\Scripts\python.exe scripts\evidence_integrity_gate.py `
  --bundle-path outputs\reports\agent_eval_bundle_dev.json `
  --chunks-path data\chunks\chunks.jsonl
```

The regeneration command selects one unambiguous actual-FY net-profit/revenue
pair from `data/structured/facts.duckdb`, resolves its provenance in the real
`chunks.jsonl`, and executes the production graph through deterministic
`net_margin`. It binds chunks, facts and aliases in one corpus manifest and
does not overwrite the legacy `agent_traces_dev.jsonl`. The resulting bundle
is labelled `REAL_CORPUS_DEV_CONTRACT_ONLY`, `publishable=false`: it proves the
current code can close a real-asset evidence chain, but it is not reviewed
answer-quality evidence.

The gate verifies bundle/trajectory RunIdentity equality, claim IDs and states,
evidence/citation document-page metadata, successful calculation provenance,
and final-answer references. Any mismatch returns `BLOCKED`/exit code 2. A raw
`--trace-path` can still be checked for provenance, but it has no immutable
bundle identity and is not evidence of a comparable evaluation run.

To inspect one completed agent run locally, first produce the strict bundle with
`run_agent_eval.py --mode agentic`, then render one trajectory as standalone
HTML:

```powershell
.\.venv\Scripts\python.exe scripts/observability_demo.py `
  --bundle-path outputs\reports\agent_eval_bundle_dev.json `
  --chunks-path data\chunks\chunks.jsonl `
  --output outputs\observability\agent_trace_demo.html
```

Pass `--question-id` to select a specific row. The page is a read-only review
surface for question/answer, claim status, ENTAILED-only citations, calculations,
reasoning/tool/dependency steps, failure attribution, tokens/latency and the
exact RunIdentity. A legacy `--trace-path` is visibly labelled `BLOCKED` and
returns exit code 2; it is never presented as a passing strict run.

For a provider-free end-to-end contract check, run
`scripts/strict_observability_smoke.py`. It executes the real graph with a
deterministic local runtime/answerer, writes the five-file bundle under
`outputs/strict_observability_smoke/eval/<run_id>/`, persists a READY integrity
verdict, writes a movable bundle reference, and renders the HTML page. Every
artifact is permanently marked `SYNTHETIC_CONTRACT_ONLY`, `publishable=false`;
it proves the trace/evidence plumbing only and cannot support a quality or
performance claim.

The current-worktree hard-case suite is a **synthetic contract fixture**, not a human-reviewed financial benchmark. It has exactly 100 cases: ten per category across factual, numerical disambiguation, comparison, calculation, multi-hop, rewrite, misleading retrieval, conflict, abstention, and partial-answer behavior. The executable profiles are `baseline-rag`, `agentic-rag`, `structured-agent`, and `full-agent`, with treatments derived from the single `PROFILE_SPECS` runtime registry. **BLOCKED:** no reviewed hard-case cases/manifest or identity-compatible reviewed four-profile result set is available, so no profile ranking or improvement claim is valid. The restored historical full seed/results/attestation are a separate local evidence set and do not satisfy the reviewed four-profile benchmark contract. The reviewed row/manifest schemas and publication gate are documented in [`benchmarks/hard_cases/README.md`](benchmarks/hard_cases/README.md).

To prove that all four execution paths and their independent bundles remain runnable without a provider, use `make four-profile-contract` (or the equivalent `run_profile_ablation.py` command in the Makefile). This mode requires all three explicit flags `--allow-synthetic-contract --contract-oracle --offline-contract`, reads only retrieved synthetic evidence, and records `offline-deterministic`, `external_network=false`, and `model_downloads=false` in the shared RunIdentity context. The three graph-backed bundles must have a `READY` evidence-integrity verdict; baseline integrity is explicitly `NOT_APPLICABLE` because baseline intentionally has no verified-claim layer. The matrix is always `SYNTHETIC_CONTRACT_ONLY`, `performance_claim_allowed=false`, regardless of its scores.

This 100-case matrix proves entrypoint execution, bundle persistence, identity
comparability, and integrity contracts only. It does not prove that every benchmark
category activates its named treatment: the current `derived_calculation` and
`multi_hop` contract rows execute through `SEARCH` with zero persisted calculations.
Treatment activation is covered separately by `tests/test_profile_runtime.py`;
reviewed quality conclusions still require reviewed cases and gold labels.

A formal reviewed run uses `run_profile_ablation.py --cases-path <reviewed_cases.jsonl> --reviewed-manifest-path <reviewed_manifest.json> --reviewed-source-root <reviewed_source_root> --reviewed-target-count <release_target> --chunks-path <shared_corpus_chunks.jsonl> --facts-path <facts.duckdb> --company-aliases-path <company_aliases.json> --llm-provider deepseek --llm-model <model> --llm-model-revision <pinned-revision> --output-root outputs/profile_ablation`. Add all four semantic inputs above only after their reviewed readiness gate is `READY`. All four profiles retrieve from that same shared corpus; reviewed gold evidence is evaluator-only and is never injected into search. Before model loading, every manifest source path is confined under the explicit root and its file SHA-256 is recomputed; the formal manifest also requires a timezone-bearing human release attestation. Self-reported source-verification state is rejected. The manifest hashes, provenance and quotas are recomputed; an omitted target, category/source shortfall, missing prediction, source/release verification failure, or zero reviewed claim/calculation gold denominator keeps the result `COVERAGE_ONLY`. Synthetic execution requires explicit `--allow-synthetic-contract --cases-path benchmarks/hard_cases/cases.jsonl` and remains `SYNTHETIC_CONTRACT_ONLY`; using benchmark evidence as a contract oracle additionally requires `--contract-oracle` and is permanently non-publishable.

Every graph-backed profile, including `structured-agent`, always performs atomic claim extraction and deterministic provenance verification before publishing a non-abstained answer. This is the `strict_claim_provenance` integrity invariant, not an ablation treatment. The `claim_verification` treatment controls enhanced verification, optional calibrated semantic/LLM judging, and verification-driven recovery; disabling it cannot bypass the deterministic claim gate. Each graph profile writes one complete trajectory row per case and must persist a bundle-bound `READY` evidence-integrity verdict before comparison.

The GitHub workflow in [`.github/workflows/ci.yml`](.github/workflows/ci.yml) defines clean Python 3.12 jobs for Ubuntu and Windows. It installs the exact `requirements.lock` closure through `uv`, forces the official CPU-only PyTorch backend, runs `uv pip check`, and only then enables offline flags. The workflow subsequently runs compilation; pinned Ruff lint and format checks over the explicitly maintained CI/evaluation surface in [`ruff.toml`](ruff.toml); explicit RunIdentity/profile-mismatch and evidence-integrity gates; the full deterministic suite excluding the private full-seed test; the balanced hard-case subset; and Phase 0 smoke. The Ruff scope is intentionally incremental because legacy modules still carry pre-existing style debt; expanding it is a separate, reviewable cleanup rather than an implicit mass rewrite. The job has no GPU and uses only the checked-in public-safe synthetic business fixture, not private fixtures, PDF corpora, keys, or model downloads.

A local 263-file clean-checkout simulation containing neither `data/` nor the human
attestation file passed compilation, Ruff, `uv pip check`, 59 RunIdentity/evaluation
tests, 31 evidence/Phase-G tests, the 1009-test public suite, strict observability
(`READY`), the balanced hard-case subset, the four-profile synthetic contract, and
Phase 0 smoke (`200 passed`). This remains useful local reproducibility evidence;
hosted run `33259417431` subsequently passed the same contract on both Ubuntu and
Windows. The run emitted only the GitHub runner's Node.js 20 action-deprecation
notice, not a test or gate failure.

The provider-free four-profile job also avoids ignored `data/`: in explicit
`--offline-contract --contract-oracle` mode, the runner materializes one
public-safe synthetic structured fact plus aliases under its `--output-root`.
Those exact generated paths and hashes enter the FactStore corpus manifest and
RunIdentity. Reviewed and normal synthetic runs still require explicit corpus
assets and never fall back to this non-publishable seam fixture.

## Evaluation Scope and Limitations

- No performance improvement is claimed here. The synthetic hard-case fixture checks contracts; quality claims require a separately reviewed corpus and gold benchmark run.
- **BLOCKED:** the restored full seed/results/attestation make the local seed-identity gate READY, but historical comparison still lacks the hash-pinned Stage 6 chunks and reviewed corpus attestation. Reviewed four-profile performance and precision-first semantic calibration separately remain blocked by reviewed hard-case assets, semantic labels, and an authorized identity-compatible local scorer configuration.
- Claim Support and Calculation metrics are process/evidence checks unless paired with externally reviewed labels. A successful calculation trace does not itself establish that the source fact was extracted correctly.
- Semantic/LLM judge modules are optional candidates after deterministic checks, not default proof of entailment. Similarity alone cannot establish `ENTAILED`; reviewed calibration remains `BLOCKED`, and malformed judge output must fail closed.
- Structured facts currently cover a deliberately limited metric set and isolate actual versus forecast values. Missing facts must fall back to evidence search or abstain rather than invent a value.
- The CI path uses deterministic/fake-provider compatible tests only. Full answer evaluation still needs external benchmark assets and, where applicable, locally cached models.
- M10 OCR/table work remains P2. Use [the corpus-quality gate](docs/m10_ocr_table_quality_gate.md) on a representative ingest output before promoting it; its ratios are triage signals, not OCR-quality proof.

## Current Follow-up Scope

- Expand structured coverage only through the metric registry and regression cases, keeping actual and forecast data distinct.
- Calibrate semantic verification thresholds from reviewed labels before treating semantic scores as a quality claim.
- The lightweight observability demo renders only a validated strict bundle as `READY`; legacy JSONL remains inspectable only as `BLOCKED`. It deliberately excludes login, multi-tenancy, billing, real-time market data, and automatic trading.
- Promote OCR/cross-page-table work only when the corpus-quality gate and reviewed examples show it is the primary bottleneck.

## English Summary

This repository implements a reproducible RAG pipeline for Chinese financial research PDFs. Unlike a minimal vector database demo, it includes structured PDF parsing, table-aware chunking, hybrid retrieval, reranking, evidence-constrained answer generation, citation construction, support validation, abstention, and benchmark-driven evaluation. The first GitHub release intentionally excludes raw PDFs, generated artifacts, model caches, secrets, and remote synchronization outputs.
