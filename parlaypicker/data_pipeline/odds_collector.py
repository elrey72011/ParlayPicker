"""Odds API collector with in-memory and filesystem caching."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd

from core.schema.schema_utils import normalize_columns
from core.schema.schema_validator import validate_schema


CACHE_DIR = Path("data/cache")
CACHE_TTL = timedelta(minutes=15)


def _cache_path(sport: str, date_key: str) -> Path:
    safe_sport = sport.lower().replace("/", "_")
    safe_date = date_key.replace("/", "-")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"odds_{safe_sport}_{safe_date}.json"


def _read_cache(path: Path) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    ts = datetime.fromisoformat(payload["cached_at"])
    if datetime.now(timezone.utc) - ts > CACHE_TTL:
        return None
    return payload["data"]


def _write_cache(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(
            {
                "cached_at": datetime.now(timezone.utc).isoformat(),
                "data": data,
            }
        )
    )


@lru_cache(maxsize=256)
def fetch_odds(sport: str, date_key: str) -> dict:
    """Fetch odds, then cache to disk for API-heavy workflows."""
    cache_file = _cache_path(sport, date_key)
    cached = _read_cache(cache_file)
    if cached is not None:
        return cached

    # Placeholder for real API call
    data = {"sport": sport, "date": date_key, "games": []}
    _write_cache(cache_file, data)
    return data



def prepare_raw_odds_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and validate raw odds dataframe schema."""
    df = normalize_columns(df.copy())
    return validate_schema(df, "raw_odds")
