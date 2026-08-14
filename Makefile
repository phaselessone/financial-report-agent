ifeq ($(OS),Windows_NT)
PYTHON ?= .venv/Scripts/python.exe
else
PYTHON ?= .venv/bin/python
SHELL := /bin/bash
endif

.PHONY: doctor test unit integration benchmark prepare-benchmark eval retrieval-eval answer-eval answer-eval-full pipeline deepseek-smoke

doctor:
	bash ./scripts/ragctl.sh doctor

test: unit

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
	bash ./scripts/ragctl.sh answer-eval-full

deepseek-smoke:
	bash ./scripts/ragctl.sh deepseek-smoke

pipeline:
	bash ./scripts/ragctl.sh pipeline
