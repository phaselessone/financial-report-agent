# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in this repository.

## Project overview

This is a Python RAG system for evidence-grounded question answering over Chinese financial-research PDFs. The main data flow is:

```text
PDF files
  -> structured page/elements and table-aware chunks
  -> dense + BM25 retrieval
  -> hybrid ranking + cross-encoder reranking
  -> evidence selection and answer generation
  -> citation binding, support validation, and abstention
  -> benchmark metrics, badcases, metadata, and archived artifacts
```

Large inputs and generated state are intentionally outside the public Git boundary. The repository ignores `pdf/`, `data/`, `models/`, `outputs/`, `artifacts/`, temporary/remote-sync folders, caches, secrets, and backup archives. Do not add those generated or private assets to commits.

## Environment and dependencies

The project uses a local Python virtual environment at `.venv`. Runtime dependencies are in `requirements.txt`; development testing depends on `requirements-dev.txt`; remote-operation helpers are in `requirements-ops.txt`.

Windows PowerShell setup:

```powershell
uv venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt
```

The equivalent `uv` install is:

```powershell
uv pip install --python .venv\Scripts\python.exe -r requirements.txt -r requirements-dev.txt
```

On macOS/Linux, use `.venv/bin/python` and `.venv/bin/python` in the corresponding commands shown in `README.md`.

Private runtime configuration is loaded from these files, in this order:

```text
.env -> .env.deepseek -> .env.local -> .env.runtime
```

Later files override earlier files, and the Python loader in `src/utils/env.py` also overwrites values already present in `os.environ`. Shell entrypoints use the same order through `scripts/use_env.sh`. Copy `.env.example` to an ignored local environment file and keep API keys and connection details out of Git.

## Tests

`pytest.ini` sets `tests/` as the test root, uses `--import-mode=importlib`, and excludes generated/cache directories. Common commands from the repository root are:

```powershell
# Full test suite
.\.venv\Scripts\python.exe -m pytest -q

# Unit/non-integration tests (the Makefile's `unit` target)
.\.venv\Scripts\python.exe -m pytest -q -m "not integration"

# Integration tests only
.\.venv\Scripts\python.exe -m pytest -q -m integration

# One test module
.\.venv\Scripts\python.exe -m pytest -q tests/test_claim_eval.py

# One test function or method
.\.venv\Scripts\python.exe -m pytest -q tests/test_claim_eval.py::ClaimEvalMetricsTests::test_build_claim_eval_summary

# Search by test name
.\.venv\Scripts\python.exe -m pytest -q -k "retrieval_cache"
```

The full suite can report two failures in `tests/test_full_seed_restore.py` when `data/eval_set/answer_eval_seed_full.jsonl` is not supplied. That is a missing private/full benchmark asset, not an implementation regression. The focused P7 claim tests can be run with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_claim_eval.py
```

No repository lint or formatting configuration/tool is defined in `pyproject.toml`, `setup.cfg`, `tox.ini`, or a pre-commit configuration, and the Makefile has no lint target. Do not assume an unavailable formatter or linter is part of the project workflow.

## Make targets

The Makefile provides these targets. Targets that invoke `scripts/ragctl.sh` require Bash; on Windows run them from Git Bash (or use the direct Python commands below).

```text
make doctor
make unit
make integration
make test
make prepare-benchmark
make retrieval-eval
make answer-eval
make answer-eval-full
make deepseek-smoke
make pipeline
make benchmark
```

`make test` is an alias for the non-integration unit suite. `make benchmark` runs benchmark preparation, retrieval evaluation, and the dev answer evaluation in sequence.

## Main commands

The canonical shell dispatcher is `bash scripts/ragctl.sh <command>`. It loads the environment, selects `.venv/bin/python` or the Windows `.venv/Scripts/python.exe`, and dispatches to the Python entrypoints. Available dispatcher commands are `doctor`, `pipeline`, `prepare-benchmark`, `retrieval-eval`, `answer-eval-dev`, `answer-eval-full`, `answer-eval-full-raw`, and `deepseek-smoke`.

Typical Git Bash workflow:

```bash
bash scripts/ragctl.sh doctor
bash scripts/ragctl.sh pipeline --input-dir pdf --output-dir data
bash scripts/ragctl.sh prepare-benchmark
bash scripts/ragctl.sh retrieval-eval
bash scripts/ragctl.sh answer-eval-dev
```

Direct Windows PowerShell equivalents:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --input-dir pdf --output-dir data
.\.venv\Scripts\python.exe run_prepare_benchmark.py
.\.venv\Scripts\python.exe run_retrieval_eval.py
.\.venv\Scripts\python.exe run_answer_eval.py
```

Important entrypoint roles:

- `run_pipeline.py` parses PDFs from `pdf/` and writes structured pages/chunks under `data/` plus badcases under `outputs/badcases/`.
- `run_prepare_benchmark.py` creates retrieval benchmark assets and answer-seed drafts from the chunk corpus.
- `run_retrieval_eval.py` builds/reuses retrieval indexes and evaluates dense, BM25, hybrid, and reranked results.
- `run_answer_eval.py` runs answer generation, citation/support validation, abstention evaluation, summaries, badcases, and artifact handling. `--split dev` is the current shipped path; historical full profiles require the separate full seed and historical corpus.
- `run_agent_eval.py` supports `--mode agentic`, `--mode baseline`, and `--mode compare` for the agentic evaluation workflow.
- `run_claim_eval.py` independently runs the agent graph and writes claim-level results to `outputs/claims/`. A representative CPU/offline invocation is:

  ```bash
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    .venv/Scripts/python.exe run_claim_eval.py \
      --queries data/eval_set/agent_eval_seed_dev.jsonl \
      --runtime-profile cpu --eval-limit 5
  ```

For PowerShell, set the three environment variables before invoking the command:

```powershell
$env:MODEL_OFFLINE = "1"
$env:OMP_NUM_THREADS = "2"
$env:OPENBLAS_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
.\.venv\Scripts\python.exe run_claim_eval.py --runtime-profile cpu --eval-limit 5
```

Do not merge the `answer-eval-full` and `answer-eval-full-raw` result sets when reporting metrics. They represent distinct historical evaluation profiles.

## Architecture and implementation map

- **Ingest (`src/ingest/`)**: `parser.py` extracts PDF structure; `cleaner.py` normalizes page content; `chunker.py` creates table-aware chunks; `pipeline.py` coordinates ingestion and reports.
- **Retrieval (`src/retrieval/`)**: `embedder.py` and `faiss_index.py` provide dense retrieval; `bm25_index.py` provides lexical retrieval; `hybrid_retriever.py` combines them; `reranker.py` applies cross-encoder reranking; `runtime.py` assembles the runtime and search result; `runtime_profiles.py` resolves device and batch settings.
- **Generation (`src/generation/`)**: `routing.py` classifies question/answer modes; `evidence_selector.py` chooses evidence; `answer_service.py` implements the answerer; `payload_parser.py`/`payload_repair.py` handle structured model output; `citation_builder.py` binds citations; `support_validator.py` and `abstain.py` enforce evidence/support behavior. `answerer.py` is a compatibility shim that re-exports the split generation modules.
- **LLM providers (`src/llm/` and `src/generation/provider.py`)**: provider configuration, retries, usage accounting, and provider adapters are separate from retrieval and answer policy.
- **Agent (`src/agent/`)**: `graph.py` wires the LangGraph flow `analyze_query -> retrieve/decompose -> grade/rewrite -> synthesize -> extract_claims -> verify_answer -> finalize`. `state.py` defines the dictionary-shaped state, `config.py` holds budgets, and node modules implement each stage. Multi-hop queries use structured subquestions/evidence; P7 claims are stored in `state["claims"]`.
- **Evaluation (`src/evaluation/`)**: benchmark materialization, retrieval/answer/agent/trajectory/API metrics, claim metrics, run metadata, source manifests, and output archive policy. Evaluation files should describe whether a metric is process-level or human/gold-set based; the current claim evaluator is a process metric and its DERIVED check is only a weak text-consistency signal, not a calculator recheck.
- **Operations (`scripts/`, `bootstrap/`)**: `scripts/ragctl.sh` is the cross-platform dispatcher, `scripts/use_env.sh` loads shell environment files, and `bootstrap/check_env.sh doctor` checks required inputs, modules, and optional remote configuration.

## Runtime conventions

The default retrieval profile is `low_vram`: embedding on CUDA and reranking on CPU with conservative batch sizes. `cpu` places both models on CPU; `standard_gpu` uses both on CUDA. Explicit CLI arguments override environment variables, which override profile TOML/default settings. Embedding and reranker devices are independently configurable.

For the CPU smoke path, keep the model-loading recipe together and configure it before model imports/loading:

```text
MODEL_OFFLINE=1
OMP_NUM_THREADS=2
OPENBLAS_NUM_THREADS=2
MKL_NUM_THREADS=2
```

`MODEL_OFFLINE=1` makes local model resolution use the existing cache. If the cache is incomplete, model loading fails rather than downloading implicitly. The retrieval runtime reuses `outputs/indexes/` when available; use `--rebuild-indexes` when the indexes must be regenerated.

Evaluation defaults are designed to be reproducible: generation evaluation uses deterministic temperature settings where applicable, records corpus/benchmark metadata, and writes reports under ignored output directories. Keep generated reports and artifacts out of commits.

## Repository guidance files

No repository-level `CLAUDE.md` existed before this file was created. No `.cursor/rules/`, `.cursorrules`, or `.github/copilot-instructions.md` files were found, so there are no additional editor-specific instructions to merge here.
