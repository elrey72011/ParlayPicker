from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from parlaypicker.utils.config import CACHE_DIR


@lru_cache(maxsize=128)
def fetch_kalshi_markets(date_key: str) -> dict:
    cache_file = Path(CACHE_DIR) / f"kalshi_{date_key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    data = {"date": date_key, "markets": []}
    cache_file.write_text(json.dumps(data))
    return data
