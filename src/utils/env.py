"""Cross-platform ``.env`` loading that mirrors ``scripts/use_env.sh``.

Loads environment files from the repository root in priority order (lowest to
highest): ``.env``, ``.env.deepseek``, ``.env.local``, ``.env.runtime``.
Later files override earlier ones, and every value overrides a variable already
present in ``os.environ`` (mirroring ``source`` / ``set -a`` semantics).

The module is dependency-free and safe to import from any entrypoint.
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV_FILES: tuple[str, ...] = (
    ".env",
    ".env.deepseek",
    ".env.local",
    ".env.runtime",
)


def _default_base_dir() -> Path:
    # src/utils/env.py -> repo root is two levels up.
    return Path(__file__).resolve().parents[2]


def _strip_value(raw_value: str) -> str:
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    return value


def _apply_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, raw_value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        os.environ[key] = _strip_value(raw_value)


def load_env_files(base_dir: Path | None = None) -> None:
    """Load ``.env*`` files into ``os.environ``.

    ``base_dir`` defaults to the repository root (two levels above this file).
    Existing environment variables are overwritten by later files, mirroring
    the ``source`` semantics used by ``scripts/use_env.sh``.
    """
    root = base_dir or _default_base_dir()
    for filename in _ENV_FILES:
        _apply_file(root / filename)
