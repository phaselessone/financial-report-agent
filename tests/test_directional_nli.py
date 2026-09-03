from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from src.agent.directional_nli import (
    DirectionalNLIConfigError,
    DirectionalNLIScorer,
    build_local_directional_nli_scorer,
    directional_nli_identity,
    normalize_directional_nli_config,
)

FULL_COMMIT_SHA = "0123456789abcdef0123456789abcdef01234567"


def _config(**overrides):
    config = {
        "kind": "directional_nli",
        "model": "local/reviewed-nli",
        "revision": FULL_COMMIT_SHA,
        "entailment_label_id": 2,
        "max_length": 384,
        "device": "cpu",
        "local_files_only": True,
        "trust_remote_code": False,
    }
    config.update(overrides)
    return config


def test_directional_nli_identity_binds_normalized_runtime_config() -> None:
    first = directional_nli_identity(_config())
    second = directional_nli_identity(_config(max_length=512))

    assert first["kind"] == "directional_nli"
    assert first["model"] == "local/reviewed-nli"
    assert first["revision"] == FULL_COMMIT_SHA
    assert len(first["config_sha256"]) == 64
    assert first["config_sha256"] != second["config_sha256"]


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"local_files_only": False}, "local_files_only"),
        ({"trust_remote_code": True}, "trust_remote_code"),
        ({"entailment_label_id": -1}, "entailment_label_id"),
        ({"unknown": "value"}, "unknown fields"),
        ({"revision": "main"}, "40-character"),
        ({"revision": "MASTER"}, "40-character"),
        ({"revision": "refs/heads/release"}, "40-character"),
        ({"revision": "latest"}, "40-character"),
        ({"revision": "feature-v1"}, "40-character"),
        ({"revision": FULL_COMMIT_SHA[:12]}, "40-character"),
        ({"revision": FULL_COMMIT_SHA.upper()}, "lowercase"),
        ({"model": ""}, "Hub repo ID"),
        ({"model": "./reviewed-nli"}, "Hub repo ID"),
        ({"model": "../reviewed-nli"}, "Hub repo ID"),
        ({"model": "C:\\models\\reviewed-nli"}, "Hub repo ID"),
        ({"model": "/models/reviewed-nli"}, "Hub repo ID"),
        ({"model": "datasets/org/reviewed-nli"}, "Hub repo ID"),
        ({"model": "reviewed--nli"}, "Hub repo ID"),
        ({"model": "reviewed..nli"}, "Hub repo ID"),
        ({"model": "reviewed-nli.git"}, "Hub repo ID"),
    ],
)
def test_directional_nli_config_fails_closed(overrides, match) -> None:
    with pytest.raises(DirectionalNLIConfigError, match=match):
        normalize_directional_nli_config(_config(**overrides))


def test_directional_scorer_preserves_evidence_to_claim_order() -> None:
    calls = []
    scorer = DirectionalNLIScorer(
        _config(),
        lambda premise, hypothesis: calls.append((premise, hypothesis)) or 0.97,
    )

    result = scorer("claim", "evidence")

    assert calls == [("evidence", "claim")]
    assert result == {
        "status": "ENTAILED",
        "score": 0.97,
        "direction": "evidence_to_claim",
    }


def test_directional_scorer_rejects_invalid_backend_probability() -> None:
    scorer = DirectionalNLIScorer(_config(), lambda _premise, _hypothesis: 1.1)
    with pytest.raises(ValueError, match="outside"):
        scorer("claim", "evidence")


def test_directional_nli_config_rejects_existing_repo_shaped_local_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_model = tmp_path / "local" / "reviewed-nli"
    local_model.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(DirectionalNLIConfigError, match="local or relative path"):
        normalize_directional_nli_config(_config(model="local/reviewed-nli"))


class _FakeModel:
    def __init__(self, commit_hash: str | None) -> None:
        self.config = SimpleNamespace(num_labels=3, _commit_hash=commit_hash)
        self.device = None
        self.evaluating = False

    def to(self, device: str) -> None:
        self.device = device

    def eval(self) -> None:
        self.evaluating = True


def _install_fake_model_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tokenizer: object,
    model: object,
) -> list[tuple[str, str, dict[str, object]]]:
    calls: list[tuple[str, str, dict[str, object]]] = []
    transformers_module = ModuleType("transformers")
    transformers_module.AutoTokenizer = SimpleNamespace(
        from_pretrained=lambda repo_id, **kwargs: calls.append(("tokenizer", repo_id, kwargs))
        or tokenizer
    )
    transformers_module.AutoModelForSequenceClassification = SimpleNamespace(
        from_pretrained=lambda repo_id, **kwargs: calls.append(("model", repo_id, kwargs)) or model
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    return calls


def test_local_builder_binds_tokenizer_and_model_to_declared_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = SimpleNamespace(
        init_kwargs={
            "vocab_file": (
                f"C:/hf/models--local--reviewed-nli/snapshots/{FULL_COMMIT_SHA}/vocab.txt"
            )
        }
    )
    model = _FakeModel(FULL_COMMIT_SHA)
    calls = _install_fake_model_runtime(monkeypatch, tokenizer=tokenizer, model=model)

    scorer = build_local_directional_nli_scorer(_config())

    expected_kwargs = {
        "revision": FULL_COMMIT_SHA,
        "local_files_only": True,
        "trust_remote_code": False,
    }
    assert calls == [
        ("tokenizer", "local/reviewed-nli", expected_kwargs),
        ("model", "local/reviewed-nli", expected_kwargs),
    ]
    assert scorer.identity["revision"] == FULL_COMMIT_SHA
    assert model.device == "cpu"
    assert model.evaluating is True


@pytest.mark.parametrize(
    "tokenizer,model,match",
    [
        (
            SimpleNamespace(init_kwargs={}),
            _FakeModel(FULL_COMMIT_SHA),
            "tokenizer has no verifiable",
        ),
        (
            SimpleNamespace(_commit_hash="b" * 40),
            _FakeModel(FULL_COMMIT_SHA),
            "tokenizer.*does not match",
        ),
        (
            SimpleNamespace(_commit_hash=FULL_COMMIT_SHA),
            _FakeModel(None),
            "model has no verifiable",
        ),
        (
            SimpleNamespace(_commit_hash=FULL_COMMIT_SHA),
            _FakeModel("b" * 40),
            "model.*does not match",
        ),
    ],
)
def test_local_builder_fails_closed_for_missing_or_drifting_commit_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tokenizer: object,
    model: object,
    match: str,
) -> None:
    _install_fake_model_runtime(monkeypatch, tokenizer=tokenizer, model=model)

    with pytest.raises(DirectionalNLIConfigError, match=match):
        build_local_directional_nli_scorer(_config())


def test_directional_nli_schema_requires_full_commit_sha() -> None:
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "semantic"
        / "directional-nli-config.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert schema["properties"]["revision"]["pattern"] == "^[0-9a-f]{40}$"
    model_pattern = schema["properties"]["model"]["pattern"]
    assert re.fullmatch(model_pattern, "local/reviewed-nli")
    assert re.fullmatch(model_pattern, "reviewed-nli")
    for invalid in ("./model", "../model", "/model", "org/model/extra", "model..v1"):
        assert re.fullmatch(model_pattern, invalid) is None
