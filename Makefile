ifeq ($(OS),Windows_NT)
PYTHON ?= .venv/Scripts/python.exe
else
PYTHON ?= .venv/bin/python
SHELL := /bin/bash
endif

MODEL_REVISION ?=

.PHONY: doctor test lint format-check unit integration benchmark prepare-benchmark eval retrieval-eval answer-eval answer-eval-full pipeline deepseek-smoke p0-smoke p0-baseline p0-full-seed-check p0-readiness historical-full-evidence-gate evidence-integrity-gate hard-case-smoke four-profile-contract corpus-quality-gate observability-demo strict-observability-smoke current-dev-trace

doctor:
	bash ./scripts/ragctl.sh doctor

test: unit

lint:
	$(PYTHON) -m ruff check .

format-check:
	$(PYTHON) -m ruff format --check .

p0-smoke:
	$(PYTHON) scripts/phase0_gate.py smoke

p0-baseline:
	$(PYTHON) scripts/phase0_gate.py baseline

p0-full-seed-check:
	$(PYTHON) scripts/phase0_gate.py check-full-seed

p0-readiness:
	$(PYTHON) scripts/phase0_gate.py readiness

historical-full-evidence-gate:
	$(PYTHON) scripts/historical_full_evidence_gate.py

evidence-integrity-gate:
	$(PYTHON) scripts/evidence_integrity_gate.py --bundle-path outputs/reports/agent_eval_bundle_dev.json --chunks-path data/chunks/chunks.jsonl

hard-case-smoke:
	$(PYTHON) scripts/hard_case_subset_gate.py --per-category 1

four-profile-contract:
	$(PYTHON) run_profile_ablation.py --cases-path benchmarks/hard_cases/cases.jsonl --allow-synthetic-contract --contract-oracle --offline-contract --output-root outputs/profile_ablation_contract --overwrite

corpus-quality-gate:
	$(PYTHON) scripts/corpus_quality_gate.py

observability-demo:
	$(PYTHON) scripts/observability_demo.py

strict-observability-smoke:
	$(PYTHON) scripts/strict_observability_smoke.py --output-root outputs/strict_observability_smoke --overwrite

current-dev-trace:
	$(PYTHON) scripts/regenerate_dev_trace.py --output-root outputs

unit:
	$(PYTHON) -m pytest -q -m "not integration"

integration:
	$(PYTHON) -m pytest -q -m integration

benchmark: prepare-benchmark retrieval-eval answer-eval

prepare-benchmark:
	bash ./scripts/ragctl.sh prepare-benchmark

eval: answer-eval

retrieval-eval:
	bash ./scripts/ragctl.sh retrieval-eval

answer-eval:
	bash ./scripts/ragctl.sh answer-eval-dev

answer-eval-full:
	bash ./scripts/ragctl.sh answer-eval-full --model-revision "$(MODEL_REVISION)"

deepseek-smoke:
	bash ./scripts/ragctl.sh deepseek-smoke

pipeline:
	bash ./scripts/ragctl.sh pipeline
