"""Runtime device profiles for the 4GB low-VRAM target (checklist v3.0 §0.1/0.2).

Embedding (BGE-M3) and reranker (BGE reranker) devices are independently
configurable. The default profile is ``low_vram``: embedding on CUDA, reranker
on CPU.

Resolution precedence (highest wins):

1. explicit keyword arguments
2. environment variables (``EMBEDDING_DEVICE``, ``RERANKER_DEVICE``,
   ``EMBEDDING_BATCH_SIZE``, ``RERANKER_BATCH_SIZE``)
3. profile TOML file (``configs/runtime/<profile>.toml``)
4. built-in profile defaults

The legacy single ``device=`` knob is honored as a fallback that sets both
devices, preserving the pre-P1 call convention.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

RUNTIME_PROFILES_DIR = Path(__file__).resolve().parents[2] / "configs" / "runtime"

VALID_DEVICES = ("cuda", "cpu", "auto")
DEFAULT_PROFILE = "low_vram"

BUILTIN_PROFILE_DEFAULTS: dict[str, dict[str, object]] = {
    "low_vram": {
        "embedding_device": "cuda",
        "embedding_batch_size": 4,
        "reranker_device": "cpu",
        "reranker_batch_size": 2,
    },
    "cpu": {
        "embedding_device": "cpu",
        "embedding_batch_size": 4,
        "reranker_device": "cpu",
        "reranker_batch_size": 2,
    },
    "standard_gpu": {
        "embedding_device": "cuda",
        "embedding_batch_size": 16,
        "reranker_device": "cuda",
        "reranker_batch_size": 8,
    },
}

_DEVICE_KEYS = ("embedding_device", "reranker_device")
_BATCH_KEYS = ("embedding_batch_size", "reranker_batch_size")

_ENV_DEVICE_KEYS = {"EMBEDDING_DEVICE": "embedding_device", "RERANKER_DEVICE": "reranker_device"}
_ENV_BATCH_KEYS = {"EMBEDDING_BATCH_SIZE": "embedding_batch_size", "RERANKER_BATCH_SIZE": "reranker_batch_size"}


@dataclass(frozen=True)
class RuntimeSettings:
    embedding_device: str
    reranker_device: str
    embedding_batch_size: int
    reranker_batch_size: int
    profile: str


def available_profiles() -> list[str]:
    return sorted(BUILTIN_PROFILE_DEFAULTS)


def load_runtime_profile(profile: str) -> dict[str, object]:
    """Load a profile TOML file merged over the built-in defaults.

    The TOML schema is ``[embedding] device/batch_size`` and
    ``[reranker] device/batch_size``; unknown keys are ignored.
    """
    if profile not in BUILTIN_PROFILE_DEFAULTS:
        raise ValueError(
            f"Unknown runtime profile '{profile}'. Available profiles: {', '.join(available_profiles())}"
        )
    values: dict[str, object] = dict(BUILTIN_PROFILE_DEFAULTS[profile])
    profile_path = RUNTIME_PROFILES_DIR / f"{profile}.toml"
    if not profile_path.is_file():
        return values
    with profile_path.open("rb") as handle:
        parsed = tomllib.load(handle)
    embedding = parsed.get("embedding", {}) if isinstance(parsed, dict) else {}
    reranker = parsed.get("reranker", {}) if isinstance(parsed, dict) else {}
    for section, prefix in ((embedding, "embedding"), (reranker, "reranker")):
        device = section.get("device")
        batch_size = section.get("batch_size")
        if device is not None:
            values[f"{prefix}_device"] = str(device)
        if batch_size is not None:
            values[f"{prefix}_batch_size"] = batch_size
    return values


def _validate_device(value: object, label: str) -> str:
    device = str(value)
    if device not in VALID_DEVICES:
        raise ValueError(
            f"Invalid {label} '{device}'. Allowed devices: {', '.join(VALID_DEVICES)}."
        )
    return device


def _parse_batch_size(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Invalid {label} {value!r}: expected a positive integer.")
    try:
        batch_size = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {label} {value!r}: expected a positive integer.") from exc
    if batch_size < 1:
        raise ValueError(f"Invalid {label} {value!r}: expected a positive integer.")
    return batch_size


def resolve_runtime_settings(
    *,
    profile: str | None = None,
    embedding_device: str | None = None,
    reranker_device: str | None = None,
    embedding_batch_size: int | None = None,
    reranker_batch_size: int | None = None,
    device: str | None = None,
) -> RuntimeSettings:
    """Merge profile, environment and explicit arguments into final settings."""
    profile_name = profile or os.environ.get("RUNTIME_PROFILE") or DEFAULT_PROFILE
    values = load_runtime_profile(profile_name)

    # Legacy single-device knob (pre-P1 call convention): applies when neither
    # the dedicated argument nor the environment variable provides a value.
    if device is not None:
        if embedding_device is None and "EMBEDDING_DEVICE" not in os.environ:
            values["embedding_device"] = device
        if reranker_device is None and "RERANKER_DEVICE" not in os.environ:
            values["reranker_device"] = device

    for env_name, key in _ENV_DEVICE_KEYS.items():
        if env_name in os.environ:
            values[key] = os.environ[env_name]
    for env_name, key in _ENV_BATCH_KEYS.items():
        if env_name in os.environ:
            values[key] = os.environ[env_name]

    if embedding_device is not None:
        values["embedding_device"] = embedding_device
    if reranker_device is not None:
        values["reranker_device"] = reranker_device
    if embedding_batch_size is not None:
        values["embedding_batch_size"] = embedding_batch_size
    if reranker_batch_size is not None:
        values["reranker_batch_size"] = reranker_batch_size

    return RuntimeSettings(
        embedding_device=_validate_device(values["embedding_device"], "embedding_device"),
        reranker_device=_validate_device(values["reranker_device"], "reranker_device"),
        embedding_batch_size=_parse_batch_size(values["embedding_batch_size"], "embedding_batch_size"),
        reranker_batch_size=_parse_batch_size(values["reranker_batch_size"], "reranker_batch_size"),
        profile=profile_name,
    )
