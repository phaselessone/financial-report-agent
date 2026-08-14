# Financial Research PDF RAG / 中文金融研报 PDF 可溯源问答系统

面向中文金融研报 PDF 的证据驱动 RAG 系统。项目覆盖从 PDF 解析、结构化切块、Hybrid+Rerank 检索，到证据约束生成、引用绑定、拒答判断和评测归档的完整闭环。

This is an evidence-grounded RAG system for Chinese financial research PDFs. It focuses on structured PDF ingestion, hybrid retrieval with reranking, citation-aware answer generation, abstention, and reproducible evaluation.

## Overview

金融研报常见多栏排版、表格、图注、页眉页脚、免责声明和跨文档比较问题。普通“向量库 + LLM”流程很容易出现证据错位、引用不准或无证据硬答。本项目的目标是把研报 PDF 转成可检索、可引用、可评测、可复现的问答系统。

核心链路：

```text
PDF ingest
  -> structured pages / cleaned pages / elements
  -> table-protected chunks
  -> Dense + BM25 hybrid retrieval
  -> rerank
  -> evidence-constrained answer generation
  -> citation + support validation + abstain
  -> evaluation reports + badcases + artifacts
```

## Highlights

- **结构化 PDF ingest**：基于 PyMuPDF 抽取页面块、标题、表格和正文元素，不直接把整篇 PDF 当普通文本切分。
- **表格保护切块**：生产策略为 `table_protected`，优先保护金融研报中的表格、标题层级和数字邻近上下文。
- **Hybrid+Rerank 检索**：组合 `BAAI/bge-m3`、FAISS、BM25、`BAAI/bge-reranker-v2-m3`，兼顾语义召回和关键词匹配。
- **证据约束生成**：回答阶段会选择证据、构造 prompt、修复 JSON、验证支持性、生成 citation，并在证据不足时 fallback 或 abstain。
- **可复现评测闭环**：输出 retrieval/answer 指标、badcases、source manifest、corpus hash、benchmark hash 和 artifact 目录。

## Architecture

| Layer | Main files | Responsibility |
| --- | --- | --- |
| Ingest | `src/ingest/*`, `run_pipeline.py` | PDF 解析、清洗、元素抽取、切块、ingest 报告 |
| Retrieval | `src/retrieval/*`, `run_retrieval_eval.py` | Dense/BM25/Hybrid/Rerank 检索运行时和评测 |
| Generation | `src/generation/*`, `run_answer_eval.py` | 证据选择、题型路由、LLM provider、citation、abstain |
| Evaluation | `src/evaluation/*` | benchmark 构建、retrieval/answer 指标、归档与元数据 |
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
│   └── evaluation/            # benchmark assets, eval metrics, metadata, archive policy
├── tests/                     # regression tests for routing, eval, cache, metadata
├── run_pipeline.py            # PDF ingest entrypoint
├── run_prepare_benchmark.py   # benchmark materialization helper
├── run_retrieval_eval.py      # retrieval evaluation entrypoint
└── run_answer_eval.py         # answer generation/evaluation entrypoint
```

Large local data and generated artifacts are intentionally excluded from GitHub: `pdf/`, `data/`, `outputs/`, `artifacts/`, `models/`, remote sync folders, caches, backups, and zip/tgz packages.

## Quick Start

### 1. Create environment

推荐使用 [uv](https://docs.astral.sh/uv/) 创建环境并安装依赖。Runtime dependencies live in `requirements.txt`; `pytest` is only in `requirements-dev.txt`.

macOS / Linux:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt -r requirements-dev.txt
```

Windows PowerShell:

```powershell
uv venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt
```

Or install directly with `uv pip`:

```powershell
uv pip install --python .venv\Scripts\python.exe -r requirements.txt -r requirements-dev.txt
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
bash scripts/ragctl.sh answer-eval-full
bash scripts/ragctl.sh answer-eval-full-raw
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

- Remote execution during project review used `.venv/bin/python -m pytest -q` and passed with `73 passed`.
- The canonical remote helper is `scripts/remote_ops.py`, but large recursive artifact transfers should prefer a single remote archive plus checksum verification.
- Evidence packages and generated outputs are intentionally kept out of the public repository boundary.
- Local test runs with no `data/` present will report 2 full-seed cases as failing, which is expected because the full answer seed material is not shipped with the repository.
- Local `.env.*`, `.ssh/`, model caches, PDFs, generated indexes, outputs, artifacts, and backup folders must remain untracked.

## Roadmap

- Split the heavy generation answerer into smaller units: routing, evidence selection, payload repair, citation refinement, and abstain gate.
- Improve span-level citation so answers anchor to precise evidence passages, not only the correct document.
- Add stronger end-to-end tests for PDF ingest, retrieval, generation provider behavior, and CLI workflows.
- Add a lightweight demo UI that displays the answer, evidence snippets, document/page references, confidence, latency, and abstain reason.
- Support safer incremental corpus updates and report de-duplication.

## English Summary

This repository implements a reproducible RAG pipeline for Chinese financial research PDFs. Unlike a minimal vector database demo, it includes structured PDF parsing, table-aware chunking, hybrid retrieval, reranking, evidence-constrained answer generation, citation construction, support validation, abstention, and benchmark-driven evaluation. The first GitHub release intentionally excludes raw PDFs, generated artifacts, model caches, secrets, and remote synchronization outputs.
