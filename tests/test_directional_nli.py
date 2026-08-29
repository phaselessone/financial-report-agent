from __future__ import annotations

import pytest

from src.agent.directional_nli import (
    DirectionalNLIConfigError,
    DirectionalNLIScorer,
    directional_nli_identity,
    normalize_directional_nli_config,
)


def _config(**overrides):
    config = {
        "kind": "directional_nli",
        "model": "local/reviewed-nli",
        "revision": "0123456789abcdef",
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
    assert first["revision"] == "0123456789abcdef"
    assert len(first["config_sha256"]) == 64
    assert first["config_sha256"] != second["config_sha256"]


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"local_files_only": False}, "local_files_only"),
        ({"trust_remote_code": True}, "trust_remote_code"),
        ({"entailment_label_id": -1}, "entailment_label_id"),
        ({"unknown": "value"}, "unknown fields"),
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
