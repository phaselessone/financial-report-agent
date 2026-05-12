#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/use_env.sh"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  if [[ -x ".venv/Scripts/python.exe" ]]; then
    PYTHON_BIN=".venv/Scripts/python.exe"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "No usable Python interpreter found. Set PYTHON_BIN explicitly." >&2
    exit 1
  fi
fi

echo "repo_root=$ROOT_DIR"
echo "python_bin=$PYTHON_BIN"
"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

repo_root = Path.cwd()

required_paths = [
    "run_pipeline.py",
    "run_retrieval_eval.py",
    "run_answer_eval.py",
    "data/eval_set/retrieval_eval_seed.jsonl",
    "data/eval_set/answer_eval_seed_dev.jsonl",
    "data/eval_set/answer_eval_seed_full.jsonl",
]
missing = [path for path in required_paths if not (repo_root / path).exists()]
if missing:
    raise SystemExit(f"missing required paths: {', '.join(missing)}")

modules = [
    "numpy",
    "httpx",
    "pytest",
    "torch",
    "transformers",
    "sentence_transformers",
    "modelscope",
]
missing_modules = [name for name in modules if importlib.util.find_spec(name) is None]
print("missing_modules=" + ",".join(missing_modules))

def count_lines(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())

summary = {
    "retrieval_seed_rows": count_lines(repo_root / "data/eval_set/retrieval_eval_seed.jsonl"),
    "answer_seed_dev_rows": count_lines(repo_root / "data/eval_set/answer_eval_seed_dev.jsonl"),
    "answer_seed_full_rows": count_lines(repo_root / "data/eval_set/answer_eval_seed_full.jsonl"),
}
print(json.dumps(summary, ensure_ascii=False))
PY

LLM_PROVIDER="${LLM_PROVIDER:-deepseek}"
echo "llm_provider=$LLM_PROVIDER"
echo "llm_model=${DEEPSEEK_MODEL:-}"
if [[ "$LLM_PROVIDER" == "deepseek" ]]; then
  if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "Missing DEEPSEEK_API_KEY for LLM_PROVIDER=deepseek" >&2
    exit 1
  fi
  if [[ -z "${DEEPSEEK_BASE_URL:-}" ]]; then
    echo "Missing DEEPSEEK_BASE_URL for LLM_PROVIDER=deepseek" >&2
    exit 1
  fi
  echo "deepseek_base_url=$DEEPSEEK_BASE_URL"
fi

if [[ -n "${REMOTE_HOST:-}" && -n "${REMOTE_USER:-}" && -n "${REMOTE_PORT:-}" ]]; then
  echo "remote_check=enabled"
  ssh_opts=(-o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20)
  if [[ -n "${REMOTE_KEY_PATH:-}" ]]; then
    ssh_opts+=(-i "$REMOTE_KEY_PATH")
  fi
  ssh "${ssh_opts[@]}" -p "$REMOTE_PORT" "${REMOTE_USER}@${REMOTE_HOST}" \
    "cd /root/autodl-tmp/financial-rag-project && test -x .venv/bin/python && .venv/bin/python -V && test -f data/eval_set/retrieval_eval_seed.jsonl && test -f data/eval_set/answer_eval_seed_full.jsonl"
fi
