"""Validation helpers for immutable Hugging Face model identities."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


FULL_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
HF_HUB_REPO_ID_RE = re.compile(
    r"^(?=.{1,96}$)(?!.*(?:--|\.\.))(?!.*\.git$)"
    r"(?:[A-Za-z0-9_](?:[A-Za-z0-9._-]*[A-Za-z0-9_])?/)?"
    r"[A-Za-z0-9_](?:[A-Za-z0-9._-]*[A-Za-z0-9_])?$"
)


def is_full_git_commit_revision(value: Any) -> bool:
    """Return whether *value* is a canonical lowercase Git commit SHA-1."""

    return bool(FULL_GIT_COMMIT_RE.fullmatch(str(value or "").strip()))


def is_hf_hub_repo_id(value: Any) -> bool:
    """Return whether *value* is a Hub repo ID that cannot resolve as a local path."""

    if not isinstance(value, str):
        return False
    candidate = value.strip()
    if not HF_HUB_REPO_ID_RE.fullmatch(candidate):
        return False
    try:
        return not Path(candidate).exists()
    except OSError:
        return False


__all__ = [
    "FULL_GIT_COMMIT_RE",
    "HF_HUB_REPO_ID_RE",
    "is_full_git_commit_revision",
    "is_hf_hub_repo_id",
]
