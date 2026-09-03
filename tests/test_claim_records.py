from __future__ import annotations

from src.agent.claims import claim_structure_errors, stable_claim_id


def _claim(text: str, *, parents: list[str] | None = None) -> dict[str, object]:
    return {
        "claim_id": stable_claim_id(text),
        "text": text,
        "claim_type": "SYNTHESIZED" if parents else "EXTRACTED",
        "is_core": True,
        "parent_claim_ids": list(parents or []),
    }


def test_claim_structure_reports_dangling_parent_and_cycle_by_claim() -> None:
    first = _claim("甲结论")
    second = _claim("乙结论")
    first["parent_claim_ids"] = [second["claim_id"], "C-missing"]
    second["parent_claim_ids"] = [first["claim_id"]]

    errors = claim_structure_errors([first, second])

    assert f"parent_claim_missing:C-missing" in errors[first["claim_id"]]
    assert "parent_claim_cycle" in errors[first["claim_id"]]
    assert "parent_claim_cycle" in errors[second["claim_id"]]
