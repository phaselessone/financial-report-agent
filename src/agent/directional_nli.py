"""Local-only directional NLI scorer for reviewed semantic activation.

The scorer treats evidence as the NLI premise and a claim as the hypothesis.
It is intentionally unable to download a model or execute remote model code.
Its normalized configuration is hashed and must exactly match the scorer
identity recorded by a reviewed precision-first calibration report before the
graph may use it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import math
from pathlib import Path
import re
from typing import Any

from src.evaluation.run_identity import canonical_sha256
from src.utils.model_revision import is_full_git_commit_revision, is_hf_hub_repo_id


class DirectionalNLIConfigError(ValueError):
    """Raised when the local directional-NLI runtime is not reproducible."""


_ALLOWED_CONFIG_FIELDS = frozenset(
    {
        "kind",
        "model",
        "revision",
        "entailment_label_id",
        "max_length",
        "device",
        "local_files_only",
        "trust_remote_code",
    }
)
_SNAPSHOT_COMMIT_RE = re.compile(r"(?:^|[\\/])snapshots[\\/]([0-9a-f]{40})(?:[\\/]|$)")


def _is_local_model_path(model: str) -> bool:
    try:
        return Path(model).exists()
    except OSError:
        return True


def _metadata_commit_hashes(component: Any, *, include_paths: bool) -> set[str]:
    hashes: set[str] = set()

    def add(value: Any) -> None:
        if value is not None:
            hashes.add(str(value).strip())

    add(getattr(component, "_commit_hash", None))
    config = getattr(component, "config", None)
    add(getattr(config, "_commit_hash", None))
    init_kwargs = getattr(component, "init_kwargs", None)
    if isinstance(init_kwargs, Mapping):
        add(init_kwargs.get("_commit_hash"))
        if include_paths:
            for value in init_kwargs.values():
                if isinstance(value, (str, Path)):
                    match = _SNAPSHOT_COMMIT_RE.search(str(value))
                    if match:
                        hashes.add(match.group(1))
    return hashes


def _require_loaded_commit_hash(
    component: Any,
    *,
    component_name: str,
    expected_revision: str,
    include_paths: bool = False,
) -> None:
    hashes = _metadata_commit_hashes(component, include_paths=include_paths)
    if not hashes:
        raise DirectionalNLIConfigError(
            f"loaded {component_name} has no verifiable _commit_hash metadata"
        )
    if len(hashes) != 1:
        raise DirectionalNLIConfigError(
            f"loaded {component_name} has inconsistent _commit_hash metadata"
        )
    actual_revision = next(iter(hashes))
    if not is_full_git_commit_revision(actual_revision):
        raise DirectionalNLIConfigError(
            f"loaded {component_name} _commit_hash is not a canonical Git commit SHA"
        )
    if actual_revision != expected_revision:
        raise DirectionalNLIConfigError(
            f"loaded {component_name} _commit_hash does not match declared revision"
        )


def normalize_directional_nli_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize the complete scorer behavior contract."""

    if not isinstance(config, Mapping):
        raise DirectionalNLIConfigError("directional NLI config must be a mapping")
    unknown = sorted(set(config).difference(_ALLOWED_CONFIG_FIELDS))
    if unknown:
        raise DirectionalNLIConfigError(
            "directional NLI config has unknown fields: " + ", ".join(unknown)
        )
    kind = str(config.get("kind") or "directional_nli").strip()
    model = str(config.get("model") or "").strip()
    revision = str(config.get("revision") or "").strip()
    if kind != "directional_nli":
        raise DirectionalNLIConfigError("kind must equal directional_nli")
    if not is_hf_hub_repo_id(model) or _is_local_model_path(model):
        raise DirectionalNLIConfigError(
            "model must be a Hugging Face Hub repo ID, not a local or relative path"
        )
    if not is_full_git_commit_revision(revision):
        raise DirectionalNLIConfigError(
            "revision must be a lowercase full 40-character Git commit SHA"
        )

    label_id = config.get("entailment_label_id")
    if isinstance(label_id, bool):
        raise DirectionalNLIConfigError("entailment_label_id must be a non-negative integer")
    try:
        label_id = int(label_id)
    except (TypeError, ValueError) as exc:
        raise DirectionalNLIConfigError(
            "entailment_label_id must be a non-negative integer"
        ) from exc
    if label_id < 0:
        raise DirectionalNLIConfigError("entailment_label_id must be a non-negative integer")

    max_length = config.get("max_length", 512)
    if isinstance(max_length, bool):
        raise DirectionalNLIConfigError("max_length must be an integer between 8 and 4096")
    try:
        max_length = int(max_length)
    except (TypeError, ValueError) as exc:
        raise DirectionalNLIConfigError("max_length must be an integer between 8 and 4096") from exc
    if not 8 <= max_length <= 4096:
        raise DirectionalNLIConfigError("max_length must be an integer between 8 and 4096")

    device = str(config.get("device") or "cpu").strip().lower()
    if device not in {"cpu", "cuda", "mps"}:
        raise DirectionalNLIConfigError("device must be cpu, cuda, or mps")
    if config.get("local_files_only", True) is not True:
        raise DirectionalNLIConfigError("local_files_only must be true")
    if config.get("trust_remote_code", False) is not False:
        raise DirectionalNLIConfigError("trust_remote_code must be false")

    return {
        "kind": kind,
        "model": model,
        "revision": revision,
        "entailment_label_id": label_id,
        "max_length": max_length,
        "device": device,
        "local_files_only": True,
        "trust_remote_code": False,
    }


def directional_nli_identity(config: Mapping[str, Any]) -> dict[str, str]:
    normalized = normalize_directional_nli_config(config)
    return {
        "kind": normalized["kind"],
        "model": normalized["model"],
        "revision": normalized["revision"],
        "config_sha256": canonical_sha256(normalized),
    }


@dataclass(frozen=True)
class DirectionalNLIScorer:
    """Thin directional scorer whose backend returns entailment probability."""

    config: Mapping[str, Any]
    _score_entailment: Callable[[str, str], float] = field(repr=False)

    def __post_init__(self) -> None:
        normalized = normalize_directional_nli_config(self.config)
        object.__setattr__(self, "config", normalized)
        if not callable(self._score_entailment):
            raise DirectionalNLIConfigError("directional NLI backend must be callable")

    @property
    def identity(self) -> dict[str, str]:
        return directional_nli_identity(self.config)

    def __call__(self, claim_text: str, evidence_text: str) -> dict[str, Any]:
        score = float(self._score_entailment(str(evidence_text), str(claim_text)))
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("directional NLI backend returned a score outside [0, 1]")
        return {
            "status": "ENTAILED",
            "score": score,
            "direction": "evidence_to_claim",
        }


def build_local_directional_nli_scorer(config: Mapping[str, Any]) -> DirectionalNLIScorer:
    """Load an exact cached Transformers sequence-classification revision."""

    normalized = normalize_directional_nli_config(config)
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - dependency gate owns this path
        raise DirectionalNLIConfigError(
            "torch and transformers are required for directional NLI"
        ) from exc

    load_kwargs = {
        "revision": normalized["revision"],
        "local_files_only": True,
        "trust_remote_code": False,
    }
    if _is_local_model_path(str(normalized["model"])):
        raise DirectionalNLIConfigError("model resolved to a local path before loading")
    tokenizer = AutoTokenizer.from_pretrained(normalized["model"], **load_kwargs)
    if _is_local_model_path(str(normalized["model"])):
        raise DirectionalNLIConfigError("model resolved to a local path while loading tokenizer")
    _require_loaded_commit_hash(
        tokenizer,
        component_name="tokenizer",
        expected_revision=str(normalized["revision"]),
        include_paths=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(normalized["model"], **load_kwargs)
    if _is_local_model_path(str(normalized["model"])):
        raise DirectionalNLIConfigError("model resolved to a local path while loading model")
    _require_loaded_commit_hash(
        model,
        component_name="model",
        expected_revision=str(normalized["revision"]),
    )
    label_id = int(normalized["entailment_label_id"])
    num_labels = int(getattr(model.config, "num_labels", 0) or 0)
    if num_labels <= label_id:
        raise DirectionalNLIConfigError(
            f"entailment_label_id={label_id} is outside model num_labels={num_labels}"
        )
    device = str(normalized["device"])
    model.to(device)
    model.eval()

    def score_entailment(premise: str, hypothesis: str) -> float:
        encoded = tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=int(normalized["max_length"]),
        )
        encoded = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }
        with torch.no_grad():
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1)
        return float(probabilities[0, label_id].item())

    return DirectionalNLIScorer(normalized, score_entailment)


__all__ = [
    "DirectionalNLIConfigError",
    "DirectionalNLIScorer",
    "build_local_directional_nli_scorer",
    "directional_nli_identity",
    "normalize_directional_nli_config",
]
