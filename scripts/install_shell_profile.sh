#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-/root/autodl-tmp/financial-rag-project}"
TARGET_RC="${HOME}/.bashrc"
START_MARKER="# >>> financial-rag >>>"
END_MARKER="# <<< financial-rag <<<"

snippet="$(cat <<EOF
$START_MARKER
export FINANCIAL_RAG_ROOT="$ROOT_DIR"
alias ragctl='bash "$ROOT_DIR/scripts/ragctl.sh"'
$END_MARKER
EOF
)"

if [[ -f "$TARGET_RC" ]] && grep -Fq "$START_MARKER" "$TARGET_RC"; then
  awk -v start="$START_MARKER" -v end="$END_MARKER" '
    BEGIN {skip=0}
    $0 == start {skip=1; next}
    $0 == end {skip=0; next}
    skip == 0 {print}
  ' "$TARGET_RC" > "${TARGET_RC}.tmp"
  mv "${TARGET_RC}.tmp" "$TARGET_RC"
fi

printf '\n%s\n' "$snippet" >> "$TARGET_RC"
echo "Updated $TARGET_RC"
