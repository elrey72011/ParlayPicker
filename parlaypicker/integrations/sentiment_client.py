from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=128)
def fetch_sentiment(team: str) -> float:
    return 0.5
