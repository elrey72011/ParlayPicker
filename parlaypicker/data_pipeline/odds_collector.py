"""Odds API collector with caching."""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=256)
def fetch_odds(sport: str, date_key: str) -> dict:
    # Placeholder for real API call
    return {"sport": sport, "date": date_key, "games": []}
