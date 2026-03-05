from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=128)
def fetch_theover_lines(date_key: str) -> dict:
    return {"date": date_key, "lines": []}
