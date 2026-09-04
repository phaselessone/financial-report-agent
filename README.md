<p align="center">
  <a href="#zh-cn"><strong>简体中文</strong></a>
  &nbsp;·&nbsp;
  <a href="#en">English</a>
</p>

<a id="zh-cn"></a>

# 中文金融研报 PDF 可溯源问答系统

<p align="right"><a href="#en">English version →</a></p>

面向中文金融研报 PDF 的证据驱动 RAG 系统。它将 PDF 解析、结构化切块、混合检索、证据约束生成、引用绑定、拒答判断和可复现评测串成一条可审计的工作流。

> 本项目关注“回答是否能被证据支持”，而不只关注答案是否看起来合理。请勿将合约测试或合成基准结果表述为真实语料质量、投资建议或业务表现证明。

## 项目简介

金融研报常有多栏排版、表格、图注、页眉页脚、免责声明和跨文档比较。简单的“向量库 + LLM”流程容易造成证据错位、引用不准确或无证据作答。本项目把这些材料转换为可检索、可引用、可评测的问答资产。

~~~text
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
~~~

## 核心能力

- **结构化 PDF 解析**：使用 PyMuPDF 提取页面块、标题、表格和正文元素，而不是把整篇研报视为普通文本。
- **表格保护切块**：<code>table_protected</code> 策略保留表格、标题层级与数字附近的上下文。
- **混合检索与重排**：组合 <code>BAAI/bge-m3</code>、FAISS、BM25 和 <code>BAAI/bge-reranker-v2-m3</code>，兼顾语义召回与关键词匹配。
- **证据与计算溯源**：Claim、证据、计算输入、公式、舍入和结果均有可检查的契约；最终引用只接受 <code>ENTAILED</code> 的证据。
- **受控 Agent 执行**：工具白名单、预算、重试、去重、依赖覆盖和失败轨迹均有离线回归测试。
- **可复现评测**：<code>RunIdentity</code> 绑定配置、语料、基准、结果和轨迹，避免不同 profile、runtime 或 benchmark 的指标被静默混合。

## 系统架构

| 层级 | 主要位置 | 职责 |
| --- | --- | --- |
| Ingest | <code>src/ingest/*</code>、<code>run_pipeline.py</code> | PDF 解析、清洗、元素抽取、切块与 ingest 报告 |
| Retrieval | <code>src/retrieval/*</code>、<code>run_retrieval_eval.py</code> | Dense、BM25、Hybrid、Rerank 与检索评测 |
| Generation | <code>src/generation/*</code>、<code>run_answer_eval.py</code> | 证据选择、题型路由、LLM provider、引用和拒答 |
| Agent | <code>src/agent/*</code>、<code>run_agent.py</code> | LangGraph 编排、claim 校验、计算轨迹、依赖与受控工具 |
| Structured facts | <code>src/structured/*</code> | 财务指标注册、标准化事实查询、actual/forecast 隔离 |
| Evaluation | <code>src/evaluation/*</code>、<code>benchmarks/hard_cases/*</code> | 基准、指标、归档、元数据和失败归因 |
| Operations | <code>scripts/</code>、<code>bootstrap/</code> | 命令分发、环境检查、本地/远端辅助工具 |

## 快速开始

### 1. 创建环境

请先按 [uv 文档](https://docs.astral.sh/uv/) 安装 <code>uv</code>。<code>requirements.lock</code> 是 CI 与可复现验收使用的精确依赖闭包；<code>requirements.txt</code> 和 <code>requirements-dev.txt</code> 是维护依赖时的输入。

~~~bash
# Linux (CI target); macOS is best-effort only
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.lock
~~~

~~~powershell
# Windows PowerShell
uv venv .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.lock
~~~

Ubuntu 和 Windows 是受支持且由 CI 覆盖的平台。macOS 不是当前受支持或 CI 验收的平台；仅可在 Apple Silicon 的 macOS 14+ 或 Intel Mac 的 macOS 15+ 上按上述命令 best-effort 尝试安装，以满足当前 <code>faiss-cpu</code> wheel 的要求。

### 2. 配置运行时

将示例配置复制到被忽略的本地文件中，并将密钥保留在版本控制之外。

~~~bash
cp .env.example .env.local
~~~

~~~powershell
Copy-Item .env.example .env.local
~~~

<samp>scripts/use_env.sh</samp> 与 Python 入口按 <code>.env</code> → <code>.env.deepseek</code> → <code>.env.local</code> → <code>.env.runtime</code> 的顺序加载配置，后加载的值会覆盖前面的值。

### 3. 运行管线

~~~bash
bash scripts/ragctl.sh doctor
bash scripts/ragctl.sh pipeline --input-dir pdf --output-dir data
bash scripts/ragctl.sh prepare-benchmark
bash scripts/ragctl.sh retrieval-eval
bash scripts/ragctl.sh answer-eval-dev
~~~

Windows 上可在 Git Bash 中运行上述命令，或直接调用 Python 入口：

~~~powershell
.\.venv\Scripts\python.exe run_pipeline.py --input-dir pdf --output-dir data
.\.venv\Scripts\python.exe run_prepare_benchmark.py
.\.venv\Scripts\python.exe run_retrieval_eval.py
.\.venv\Scripts\python.exe run_answer_eval.py
~~~

历史完整评测必须显式提供不可变的模型版本：

~~~bash
bash scripts/ragctl.sh answer-eval-full --model-revision <pinned-provider-revision>
bash scripts/ragctl.sh answer-eval-full-raw --model-revision <pinned-provider-revision>
~~~

两类历史结果属于不同 profile，报告时不得合并。
<code>answer-eval-full</code> 对应 <code>historical-full-core</code>，而 <code>answer-eval-full-raw</code> 对应 <code>historical-full-raw</code>。

## 评测、可复现性与已知限制

| 项目 | 含义 |
| --- | --- |
| <code>current-dev</code> | 当前开发语料的可复现评测，不等同于历史全量结果。 |
| <code>historical-full-core</code> / <code>historical-full-raw</code> | 不可互换的历史 profile；必须固定 <code>--model-revision</code>，且不得合并指标。 |
| Full benchmark | 需要私有 seed、历史结果、review attestation、Stage 6 语料与 corpus attestation；缺任何一项均应 <code>BLOCKED</code>。 |
| Hard cases | 当前 100 例是合成、确定性的合约夹具；它证明执行和完整性契约，不证明真实金融问答质量。 |
| Four profiles | <code>baseline-rag</code>、<code>agentic-rag</code>、<code>structured-agent</code>、<code>full-agent</code> 的合约运行不构成质量排名或性能提升结论。 |
| Semantic calibration | 需要经审核标签、授权 scorer identity 与 readiness gate；相似度本身不能证明 <code>ENTAILED</code>。 |

ReasoningPlan、claim 级溯源、Decimal 计算溯源和严格 evidence gate 提供的是过程与证据完整性保证，不是对真实语料语义质量的独立证明。缺少经审核的 Stage 6 语料及 attestation、经审核的 hard-case 资产，或经审核的语义标注时，对应结论必须保持 <code>BLOCKED</code>。

## CI 与本地验证

以下命令不下载模型、不调用 provider，也不需要私有 PDF。<code>make</code> 命令需要具备 Make 和 Bash 的环境；Windows 上请在 Git Bash、WSL 或等效环境中运行，或直接调用相应 Python 脚本：

~~~powershell
.\.venv\Scripts\python.exe scripts\phase0_gate.py baseline
make p0-smoke
make hard-case-smoke
make strict-observability-smoke
make four-profile-contract
~~~

GitHub Actions 在 Ubuntu 和 Windows 上使用 Python 3.12、锁定依赖和 CPU-only PyTorch；它运行编译、Ruff、公开的确定性测试、证据完整性检查、hard-case 子集、四 profile 合约及 Phase 0 smoke。公开 CI 仅使用合成安全夹具，不下载模型，也不使用私有 PDF、密钥或评测资产。详见 [CI workflow](.github/workflows/ci.yml)。

## 数据、隐私与安全边界

- <code>pdf/</code>、<code>data/</code>、<code>models/</code>、<code>outputs/</code>、<code>artifacts/</code>、缓存、备份和 <code>.env.*</code> 是本地或私有状态，不得提交。
- 公共 clean checkout 只使用虚构发行人与合成财务记录的 [受控夹具](tests/fixtures/controlled_financial_business_case.json)；它不是研报原文，也不是人工审核的质量证据。
- 系统不提供登录、多租户、账单、实时市场数据或自动交易；其输出不应作为真实语料质量的证据、投资建议或业务表现的证明。
- OCR 或跨页表格工作应在代表性语料通过 [corpus-quality gate](docs/m10_ocr_table_quality_gate.md) 后再提升优先级。

## 项目结构

~~~text
.
├── bootstrap/                 # environment checks
├── docs/                      # technical, provenance, and postmortem documentation
├── scripts/                   # dispatchers, gates, and operational helpers
├── src/
│   ├── ingest/                # parsing, cleaning, chunking, pipeline
│   ├── retrieval/             # embedding, FAISS, BM25, hybrid retrieval, reranking
│   ├── generation/            # answer generation, citations, abstention
│   ├── agent/                 # graph, claims, tools, calculations, dependencies
│   ├── structured/            # normalized financial facts
│   └── evaluation/            # benchmarks, metrics, metadata, archive policy
├── benchmarks/                # public-safe contract fixtures and schemas
├── tests/                     # regression and contract tests
└── run_*.py                   # runnable entrypoints
~~~

## 延伸阅读

- [技术概览与环境配置](docs/project_technical_overview_and_setup.md)
- [完整基准溯源说明](docs/full_benchmark_provenance.md)
- [可观测性演示](docs/observability_demo.md)
- [OCR / 表格质量门禁](docs/m10_ocr_table_quality_gate.md)
- [Hard-case 合约](benchmarks/hard_cases/README.md)

---

<a id="en"></a>

# Financial Research PDF RAG

<p align="right"><a href="#zh-cn">简体中文版 →</a></p>

An evidence-grounded RAG system for Chinese financial-research PDFs. It connects PDF parsing, structured chunking, hybrid retrieval, evidence-constrained generation, citation binding, abstention, and reproducible evaluation in one auditable workflow.

> The project is designed to show whether an answer is supported by evidence, not merely whether it sounds plausible. Do not present contract-test or synthetic-benchmark results as real-corpus quality evidence, investment advice, or business-performance proof.

## Overview

Financial research PDFs frequently contain multi-column layouts, tables, captions, headers, footers, disclaimers, and cross-document comparisons. A simple “vector database + LLM” pipeline can misalign evidence, produce inaccurate citations, or answer without support. This project turns those documents into retrievable, citable, and evaluable assets.

~~~text
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
~~~

## Highlights

- **Structured PDF ingestion** — PyMuPDF extracts page blocks, headings, tables, and body elements instead of treating a report as undifferentiated text.
- **Table-protected chunking** — The <code>table_protected</code> strategy preserves tables, heading hierarchy, and numeric context.
- **Hybrid retrieval and reranking** — <code>BAAI/bge-m3</code>, FAISS, BM25, and <code>BAAI/bge-reranker-v2-m3</code> combine semantic recall with lexical matching.
- **Evidence and calculation provenance** — Claims, evidence, calculation inputs, formulas, rounding, and results have inspectable contracts; final citations admit only <code>ENTAILED</code> evidence.
- **Controlled agent execution** — Tool allowlists, budgets, retries, deduplication, dependency coverage, and failure trajectories are covered by offline regression tests.
- **Reproducible evaluation** — <code>RunIdentity</code> binds configuration, corpus, benchmark, results, and trajectories so metrics from incompatible profiles, runtimes, or benchmarks cannot be silently combined.

## Architecture

| Layer | Main locations | Responsibility |
| --- | --- | --- |
| Ingest | <code>src/ingest/*</code>, <code>run_pipeline.py</code> | PDF parsing, cleaning, element extraction, chunking, and ingest reports |
| Retrieval | <code>src/retrieval/*</code>, <code>run_retrieval_eval.py</code> | Dense, BM25, hybrid, rerank, and retrieval evaluation |
| Generation | <code>src/generation/*</code>, <code>run_answer_eval.py</code> | Evidence selection, routing, LLM providers, citations, and abstention |
| Agent | <code>src/agent/*</code>, <code>run_agent.py</code> | LangGraph orchestration, claim verification, calculation traces, dependencies, and controlled tools |
| Structured facts | <code>src/structured/*</code> | Financial metric registry, normalized fact access, and actual/forecast separation |
| Evaluation | <code>src/evaluation/*</code>, <code>benchmarks/hard_cases/*</code> | Benchmarks, metrics, archives, metadata, and failure attribution |
| Operations | <code>scripts/</code>, <code>bootstrap/</code> | Command dispatch, environment checks, and local/remote helpers |

## Quick Start

### 1. Create an environment

Install <code>uv</code> first from the [uv documentation](https://docs.astral.sh/uv/). <code>requirements.lock</code> is the exact dependency closure used by CI and reproducibility acceptance; <code>requirements.txt</code> and <code>requirements-dev.txt</code> are maintenance inputs.

~~~bash
# Linux (CI target); macOS is best-effort only
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.lock
~~~

~~~powershell
# Windows PowerShell
uv venv .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.lock
~~~

Ubuntu and Windows are the supported, CI-covered platforms. macOS is not currently a supported or CI-acceptance platform; it may only be installed on a best-effort basis with the commands above on macOS 14+ for Apple Silicon or macOS 15+ for Intel Macs, as required by the current <code>faiss-cpu</code> wheel.

### 2. Configure runtime

Copy the example configuration into an ignored local file and keep secrets outside version control.

~~~bash
cp .env.example .env.local
~~~

~~~powershell
Copy-Item .env.example .env.local
~~~

<samp>scripts/use_env.sh</samp> and the Python entrypoints load <code>.env</code> → <code>.env.deepseek</code> → <code>.env.local</code> → <code>.env.runtime</code>; later files override earlier ones.

### 3. Run the pipeline

~~~bash
bash scripts/ragctl.sh doctor
bash scripts/ragctl.sh pipeline --input-dir pdf --output-dir data
bash scripts/ragctl.sh prepare-benchmark
bash scripts/ragctl.sh retrieval-eval
bash scripts/ragctl.sh answer-eval-dev
~~~

On Windows, run the commands above in Git Bash or invoke the Python entrypoints directly:

~~~powershell
.\.venv\Scripts\python.exe run_pipeline.py --input-dir pdf --output-dir data
.\.venv\Scripts\python.exe run_prepare_benchmark.py
.\.venv\Scripts\python.exe run_retrieval_eval.py
.\.venv\Scripts\python.exe run_answer_eval.py
~~~

Historical full evaluation requires an explicit immutable model revision:

~~~bash
bash scripts/ragctl.sh answer-eval-full --model-revision <pinned-provider-revision>
bash scripts/ragctl.sh answer-eval-full-raw --model-revision <pinned-provider-revision>
~~~

The two historical result sets belong to different profiles and must never be merged: <code>answer-eval-full</code> selects <code>historical-full-core</code>, while <code>answer-eval-full-raw</code> selects <code>historical-full-raw</code>.

## Evaluation, Reproducibility, and Limitations

| Item | Meaning |
| --- | --- |
| <code>current-dev</code> | Reproducible evaluation for the current development corpus; it is not a historical full result. |
| <code>historical-full-core</code> / <code>historical-full-raw</code> | Separate, non-interchangeable historical profiles. Pin <code>--model-revision</code> and never merge their metrics. |
| Full benchmark | Requires private seed, historical results, review attestation, Stage 6 corpus, and corpus attestation. A missing dependency must produce <code>BLOCKED</code>. |
| Hard cases | The current set of 100 cases is a synthetic deterministic contract fixture. It proves execution and integrity contracts, not real financial-question-answering quality. |
| Four profiles | Contract runs of <code>baseline-rag</code>, <code>agentic-rag</code>, <code>structured-agent</code>, and <code>full-agent</code> do not support rankings or performance-improvement claims. |
| Semantic calibration | Requires reviewed labels, an authorized scorer identity, and a readiness gate; similarity alone cannot prove <code>ENTAILED</code>. |

ReasoningPlan, claim-level provenance, Decimal calculation provenance, and the strict evidence gate provide process and evidence-integrity guarantees. They are not independent proof of real-corpus semantic quality. When the reviewed Stage 6 corpus and attestation, reviewed hard-case assets, or reviewed semantic labels are absent, the related conclusion must remain <code>BLOCKED</code>.

## CI and local validation

The following commands do not download models, call a provider, or require private PDFs. The <code>make</code> commands require an environment with Make and Bash; on Windows, use Git Bash, WSL, or an equivalent environment, or invoke the corresponding Python scripts directly:

~~~powershell
.\.venv\Scripts\python.exe scripts\phase0_gate.py baseline
make p0-smoke
make hard-case-smoke
make strict-observability-smoke
make four-profile-contract
~~~

GitHub Actions uses Python 3.12, locked dependencies, and CPU-only PyTorch on Ubuntu and Windows. It runs compilation, Ruff, public deterministic tests, evidence-integrity checks, a hard-case subset, the four-profile contract, and Phase 0 smoke. Public CI uses synthetic safe fixtures only: no model downloads, private PDFs, secrets, or evaluation assets. See the [CI workflow](.github/workflows/ci.yml).

## Data, privacy, and security boundary

- <code>pdf/</code>, <code>data/</code>, <code>models/</code>, <code>outputs/</code>, <code>artifacts/</code>, caches, backups, and <code>.env.*</code> are local or private state and must not be committed.
- A public clean checkout uses only a fictional issuer and synthetic financial record in the [controlled fixture](tests/fixtures/controlled_financial_business_case.json). It is neither report text nor reviewed quality evidence.
- The system does not implement authentication, multitenancy, billing, real-time market data, or automated trading. Its output must not be used as evidence of real-corpus quality, investment advice, or proof of business performance.
- Promote OCR or cross-page-table work only after representative data passes the [corpus-quality gate](docs/m10_ocr_table_quality_gate.md).

## Repository layout

~~~text
.
├── bootstrap/                 # environment checks
├── docs/                      # technical, provenance, and postmortem documentation
├── scripts/                   # dispatchers, gates, and operational helpers
├── src/
│   ├── ingest/                # parsing, cleaning, chunking, pipeline
│   ├── retrieval/             # embedding, FAISS, BM25, hybrid retrieval, reranking
│   ├── generation/            # answer generation, citations, abstention
│   ├── agent/                 # graph, claims, tools, calculations, dependencies
│   ├── structured/            # normalized financial facts
│   └── evaluation/            # benchmarks, metrics, metadata, archive policy
├── benchmarks/                # public-safe contract fixtures and schemas
├── tests/                     # regression and contract tests
└── run_*.py                   # runnable entrypoints
~~~

## Further reading

- [Technical overview and setup](docs/project_technical_overview_and_setup.md)
- [Full benchmark provenance](docs/full_benchmark_provenance.md)
- [Observability demo](docs/observability_demo.md)
- [OCR / table-quality gate](docs/m10_ocr_table_quality_gate.md)
- [Hard-case contracts](benchmarks/hard_cases/README.md)
