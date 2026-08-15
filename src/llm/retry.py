"""Retry policy shared by all remote providers.

Retryable conditions: HTTP 429 / 5xx statuses and transport failures.
Other 4xx statuses fail fast without retrying (checklist v3.0 semantics,
matching the behavior introduced in commit ad86f07).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

RETRYABLE_HTTP_STATUSES = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff policy (base 2, capped at 8s)."""

    max_retries: int = 3
    base_seconds: float = 2.0
    cap_seconds: float = 8.0

    def is_retryable_status(self, status_code: int) -> bool:
        return status_code in RETRYABLE_HTTP_STATUSES

    def wait_seconds(self, attempt: int) -> float:
        return min(self.base_seconds**attempt, self.cap_seconds)

    def sleep(self, attempt: int) -> None:
        time.sleep(self.wait_seconds(attempt))
