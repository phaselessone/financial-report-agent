#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/use_env.sh"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  if [[ -x "$ROOT_DIR/.venv/Scripts/python.exe" ]]; then
    PYTHON_BIN="$ROOT_DIR/.venv/Scripts/python.exe"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "No usable Python interpreter found. Set PYTHON_BIN explicitly." >&2
    exit 1
  fi
fi

cmd="${1:-}"
if [[ -z "$cmd" ]]; then
  echo "Usage: ragctl <doctor|pipeline|prepare-benchmark|retrieval-eval|answer-eval-dev|answer-eval-full|answer-eval-full-raw|deepseek-smoke> [args...]" >&2
  exit 1
fi
shift

case "$cmd" in
  doctor)
    bash "$ROOT_DIR/bootstrap/check_env.sh" "$@"
    ;;
  pipeline)
    "$PYTHON_BIN" "$ROOT_DIR/run_pipeline.py" "$@"
    ;;
  prepare-benchmark)
    "$PYTHON_BIN" "$ROOT_DIR/run_prepare_benchmark.py" "$@"
    ;;
  retrieval-eval)
    "$PYTHON_BIN" "$ROOT_DIR/run_retrieval_eval.py" "$@"
    ;;
  answer-eval-dev)
    "$PYTHON_BIN" "$ROOT_DIR/run_answer_eval.py" --split dev --benchmark-profile current-dev "$@"
    ;;
  answer-eval-full)
    "$PYTHON_BIN" \
      "$ROOT_DIR/run_answer_eval.py" \
      --split full \
      --benchmark-profile historical-full-core \
      --chunks-path "$ROOT_DIR/tmp_stage6_data/chunks/chunks.jsonl" \
      --data-dir "$ROOT_DIR/tmp_stage6_data" \
      --corpus-label stage6-historical \
      --answer-seed-dev-path "$ROOT_DIR/data/eval_set/answer_eval_seed_full.jsonl" \
      --answer-seed-full-path "$ROOT_DIR/data/eval_set/answer_eval_seed_full.jsonl" \
      "$@"
    ;;
  answer-eval-full-raw)
    "$PYTHON_BIN" \
      "$ROOT_DIR/run_answer_eval.py" \
      --split full \
      --benchmark-profile historical-full-raw \
      --chunks-path "$ROOT_DIR/tmp_stage6_data/chunks/chunks.jsonl" \
      --data-dir "$ROOT_DIR/tmp_stage6_data" \
      --corpus-label stage6-historical \
      --answer-seed-dev-path "$ROOT_DIR/data/eval_set/answer_eval_seed_full.jsonl" \
      --answer-seed-full-path "$ROOT_DIR/data/eval_set/answer_eval_seed_full.jsonl" \
      "$@"
    ;;
  deepseek-smoke)
    "$PYTHON_BIN" "$ROOT_DIR/scripts/deepseek_smoke.py" "$@"
    ;;
  *)
    echo "Unknown ragctl command: $cmd" >&2
    exit 1
    ;;
esac
