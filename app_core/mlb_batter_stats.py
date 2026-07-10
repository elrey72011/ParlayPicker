"""MLB StatsAPI batting form for conservative batter-prop projections."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

import requests

_BASE = "https://statsapi.mlb.com/api/v1"
_TIMEOUT = 10


def _number(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def batter_form_from_gamelog(
    splits: list[dict], *, last_n: int = 10, as_of_date: str | None = None
) -> dict | None:
    """Blend season and recent per-game hitting rates without future leakage."""
    cutoff = None
    if as_of_date:
        try:
            cutoff = datetime.strptime(str(as_of_date), "%Y-%m-%d").date()
        except (TypeError, ValueError):
            cutoff = None

    usable = []
    for split in splits or []:
        if not isinstance(split, dict):
            continue
        if cutoff and split.get("date"):
            try:
                if datetime.strptime(str(split["date"]), "%Y-%m-%d").date() >= cutoff:
                    continue
            except (TypeError, ValueError):
                continue
        stat = split.get("stat", {}) or {}
        pa = _number(stat.get("plateAppearances") or stat.get("atBats"))
        if pa <= 0:
            continue
        usable.append(split)
    if not usable:
        return None

    recent = usable[-max(1, int(last_n)):]

    def mean(stat_key: str, rows: list[dict]) -> float:
        return sum(_number(r.get("stat", {}).get(stat_key)) for r in rows) / len(rows)

    season_hits = mean("hits", usable)
    recent_hits = mean("hits", recent)
    season_tb = mean("totalBases", usable)
    recent_tb = mean("totalBases", recent)
    # Season form is the stable anchor; recent form receives a modest 35% weight.
    expected_hits = 0.65 * season_hits + 0.35 * recent_hits
    expected_tb = 0.65 * season_tb + 0.35 * recent_tb
    avg_pa = mean("plateAppearances", usable)
    last_date = next((r.get("date") for r in reversed(usable) if r.get("date")), None)
    return {
        "hits_per_game": max(0.05, expected_hits),
        "total_bases_per_game": max(0.05, expected_tb),
        "n_games": len(usable),
        "avg_plate_appearances": avg_pa,
        "last_game_date": last_date,
    }


def resolve_batter_id(name: object, http_get: Callable = requests.get) -> int | None:
    """Resolve an active MLB player's name to a StatsAPI person id."""
    text = str(name or "").strip()
    if not text:
        return None
    try:
        response = http_get(
            f"{_BASE}/people/search",
            params={"names": text, "sportIds": 1, "active": "true"},
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
        people = response.json().get("people", [])
        exact = next(
            (p for p in people if str(p.get("fullName", "")).strip().lower() == text.lower()),
            None,
        )
        player = exact or (people[0] if people else None)
        return int(player["id"]) if player and player.get("id") is not None else None
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError):
        return None


def fetch_batter_form(
    name: object,
    season: int,
    *,
    as_of_date: str | None = None,
    last_n: int = 10,
    http_get: Callable = requests.get,
) -> dict | None:
    """Resolve a batter and return season/recent form; None on any feed failure."""
    player_id = resolve_batter_id(name, http_get=http_get)
    if player_id is None:
        return None
    try:
        response = http_get(
            f"{_BASE}/people/{player_id}/stats",
            params={"stats": "gameLog", "group": "hitting", "season": int(season)},
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
        stats = response.json().get("stats", [])
        splits = stats[0].get("splits", []) if stats else []
        return batter_form_from_gamelog(splits, last_n=last_n, as_of_date=as_of_date)
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError):
        return None

