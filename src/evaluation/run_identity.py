"""Canonical identity and compatibility checks for evaluation runs.

``RunIdentity`` is the seam used by writers, readers, comparisons and future
aggregators.  Its canonical representation is deliberately independent of
mapping insertion order.  Comparison is fail-closed: every identity field and
every non-treatment feature flag must match.  The only permitted differences
live under the explicit ``feature_flags["treatments"]`` namespace.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping


RUN_IDENTITY_FIELDS = (
    "benchmark_profile",
    "benchmark_version",
    "benchmark_hash",
    "case_ids_hash",
    "case_count",
    "corpus_hash",
    "model",
    "provider",
    "model_revision",
    "temperature",
    "prompt_version",
    "budgets",
    "feature_flags",
    "git_commit",
    "source_manifest_hash",
)
TREATMENTS_KEY = "treatments"


def _normalize_json(value: Any, *, path: str = "value") -> Any:
    """Return a detached JSON value and reject ambiguous/non-canonical input."""

    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{path} keys must be non-empty strings")
            normalized[key] = _normalize_json(item, path=f"{path}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item, path=f"{path}[]") for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain NaN or infinity")
        return value
    raise TypeError(f"{path} contains a non-JSON value: {type(value).__name__}")


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    normalized = _normalize_json(_thaw_json(value))
    return json.dumps(
        normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_case_ids(case_ids: Iterable[str]) -> str:
    normalized = [str(case_id).strip() for case_id in case_ids]
    if any(not case_id for case_id in normalized):
        raise ValueError("case ids must be non-empty")
    if len(normalized) != len(set(normalized)):
        raise ValueError("duplicate case id")
    return canonical_sha256(sorted(normalized))


def hash_benchmark_rows(rows: Iterable[Mapping[str, Any]]) -> str:
    """Hash benchmark rows independently of their materialization order."""

    normalized_rows = [_normalize_json(row, path="benchmark_row") for row in rows]

    def _row_id(row: Mapping[str, Any]) -> str:
        value = row.get("case_id", row.get("question_id", ""))
        value = str(value or "").strip()
        if not value:
            raise ValueError("benchmark row requires case_id or question_id")
        return value

    ids = [_row_id(row) for row in normalized_rows]
    hash_case_ids(ids)
    return canonical_sha256(sorted(normalized_rows, key=_row_id))


class IncompatibleRunIdentityError(ValueError):
    """Raised before results from incompatible evaluation runs are combined."""

    def __init__(self, mismatched_fields: Iterable[str]) -> None:
        self.mismatched_fields = tuple(sorted(set(mismatched_fields)))
        fields = ", ".join(self.mismatched_fields)
        super().__init__(f"incompatible run identities; mismatched fields: {fields}")


@dataclass(frozen=True)
class RunIdentity:
    benchmark_profile: str
    benchmark_version: str
    benchmark_hash: str
    case_ids_hash: str
    case_count: int
    corpus_hash: str
    model: str
    provider: str
    model_revision: str
    temperature: float
    prompt_version: str
    budgets: Mapping[str, Any]
    feature_flags: Mapping[str, Any]
    git_commit: str
    source_manifest_hash: str

    def __post_init__(self) -> None:
        for field_name in (
            "benchmark_profile",
            "benchmark_version",
            "benchmark_hash",
            "case_ids_hash",
            "corpus_hash",
            "model",
            "provider",
            "model_revision",
            "prompt_version",
            "git_commit",
            "source_manifest_hash",
        ):
            value = str(getattr(self, field_name) or "").strip()
            if not value:
                raise ValueError(f"{field_name} must be non-empty")
            object.__setattr__(self, field_name, value)

        case_count = int(self.case_count)
        if case_count <= 0:
            raise ValueError("case_count must be positive")
        object.__setattr__(self, "case_count", case_count)

        temperature = float(self.temperature)
        if not math.isfinite(temperature):
            raise ValueError("temperature must be finite")
        object.__setattr__(self, "temperature", temperature)

        budgets = _normalize_json(self.budgets, path="budgets")
        feature_flags = _normalize_json(self.feature_flags, path="feature_flags")
        if not isinstance(budgets, dict):
            raise TypeError("budgets must be a mapping")
        if not isinstance(feature_flags, dict):
            raise TypeError("feature_flags must be a mapping")
        treatments = feature_flags.get(TREATMENTS_KEY, {})
        if not isinstance(treatments, dict):
            raise TypeError(f"feature_flags.{TREATMENTS_KEY} must be a mapping")
        object.__setattr__(self, "budgets", _freeze_json(budgets))
        object.__setattr__(self, "feature_flags", _freeze_json(feature_flags))

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_profile": self.benchmark_profile,
            "benchmark_version": self.benchmark_version,
            "benchmark_hash": self.benchmark_hash,
            "case_ids_hash": self.case_ids_hash,
            "case_count": self.case_count,
            "corpus_hash": self.corpus_hash,
            "model": self.model,
            "provider": self.provider,
            "model_revision": self.model_revision,
            "temperature": self.temperature,
            "prompt_version": self.prompt_version,
            "budgets": _thaw_json(self.budgets),
            "feature_flags": _thaw_json(self.feature_flags),
            "git_commit": self.git_commit,
            "source_manifest_hash": self.source_manifest_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RunIdentity":
        missing = [field for field in RUN_IDENTITY_FIELDS if field not in value]
        if missing:
            raise ValueError(f"run identity missing fields: {', '.join(missing)}")
        return cls(**{field: value[field] for field in RUN_IDENTITY_FIELDS})

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    @property
    def identity_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @property
    def run_id(self) -> str:
        return f"run-{self.identity_hash[:20]}"

    def comparison_identity(self) -> dict[str, Any]:
        """Return identity fields that must be equal before combining runs."""

        value = self.to_dict()
        flags = dict(value["feature_flags"])
        flags.pop(TREATMENTS_KEY, None)
        value["feature_flags"] = flags
        return value

    def assert_comparable(self, other: "RunIdentity") -> None:
        if not isinstance(other, RunIdentity):
            raise TypeError("other must be a RunIdentity")
        left = self.comparison_identity()
        right = other.comparison_identity()
        mismatches = [field for field in RUN_IDENTITY_FIELDS if left[field] != right[field]]
        if mismatches:
            raise IncompatibleRunIdentityError(mismatches)


__all__ = [
    "IncompatibleRunIdentityError",
    "RUN_IDENTITY_FIELDS",
    "RunIdentity",
    "canonical_json",
    "canonical_sha256",
    "file_sha256",
    "hash_benchmark_rows",
    "hash_case_ids",
]
