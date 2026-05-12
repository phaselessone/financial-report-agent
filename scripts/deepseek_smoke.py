from __future__ import annotations

import json
import os
from time import perf_counter

import httpx


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def main() -> int:
    api_key = required_env("DEEPSEEK_API_KEY")
    base_url = required_env("DEEPSEEK_BASE_URL").rstrip("/")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat").strip() or "deepseek-chat"
    timeout_seconds = float(os.environ.get("DEEPSEEK_TIMEOUT_SECONDS", "60") or "60")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a financial research assistant. Respond with one short JSON object only.",
            },
            {
                "role": "user",
                "content": 'Return {"ok": true, "provider": "deepseek"}',
            },
        ],
        "max_tokens": 32,
        "stream": False,
    }

    start = perf_counter()
    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
    latency_ms = round((perf_counter() - start) * 1000, 2)
    response.raise_for_status()
    data = response.json()
    content = str(data["choices"][0]["message"]["content"]).strip()
    print(
        json.dumps(
            {
                "provider": "deepseek",
                "model": model,
                "base_url": base_url,
                "latency_ms": latency_ms,
                "content_preview": content[:200],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
