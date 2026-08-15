"""P1 module extracted from src/generation/answerer.py (behavior preserved)."""
from __future__ import annotations

from typing import Any
import json


def _strip_code_fences(text: str) -> str:
    payload = text.strip()
    if payload.startswith("```"):
        payload = payload.strip("`").strip()
        if payload.lower().startswith("json"):
            payload = payload[4:].strip()
    if payload.endswith("```"):
        payload = payload.rstrip("`").strip()
    return payload


def parse_model_json(text: str) -> dict[str, Any] | None:
    payload = _strip_code_fences(text)
    if not payload:
        return None
    decoder = json.JSONDecoder()
    best_end = -1
    best_value: Any = None
    for index, char in enumerate(payload):
        if char not in "{[":
            continue
        try:
            value, end = decoder.raw_decode(payload, index)
        except json.JSONDecodeError:
            continue
        if end > best_end:
            best_end = end
            best_value = value
    return best_value if isinstance(best_value, dict) else None
