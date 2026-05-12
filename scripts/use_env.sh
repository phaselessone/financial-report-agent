#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

load_env_file() {
  local path="$1"
  if [[ -f "$path" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$path"
    set +a
  fi
}

# Lowest priority first, highest priority last.
load_env_file "$ROOT_DIR/.env.deepseek"
load_env_file "$ROOT_DIR/.env.local"
load_env_file "$ROOT_DIR/.env.runtime"
