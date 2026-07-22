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

    def per_pa(stat_key: str, rows: list[dict]) -> float:
        total_pa = sum(_number(r.get("stat", {}).get("plateAppearances")) for r in rows)
        total_stat = sum(_number(r.get("stat", {}).get(stat_key)) for r in rows)
        return total_stat / total_pa if total_pa > 0 else 0.0

    def dispersion(stat_key: str, default: float, maximum: float) -> float:
        values = [_number(r.get("stat", {}).get(stat_key)) for r in usable]
        if len(values) < 20:
            return default
        value_mean = sum(values) / len(values)
        if value_mean <= 0:
            return default
        variance = sum((value - value_mean) ** 2 for value in values) / (len(values) - 1)
        return min(maximum, max(1.05, variance / value_mean))

    season_pa = mean("plateAppearances", usable)
    recent_pa = mean("plateAppearances", recent)
    # Estimate skill per opportunity, then estimate opportunities separately.
    # This prevents shortened recent games from looking like a skills collapse.
    projected_pa = 0.65 * season_pa + 0.35 * recent_pa
    expected_hits = (
        0.65 * per_pa("hits", usable) + 0.35 * per_pa("hits", recent)
    ) * projected_pa
    expected_tb = (
        0.65 * per_pa("totalBases", usable)
        + 0.35 * per_pa("totalBases", recent)
    ) * projected_pa
    last_date = next((r.get("date") for r in reversed(usable) if r.get("date")), None)
    return {
        "hits_per_game": max(0.05, expected_hits),
        "total_bases_per_game": max(0.05, expected_tb),
        "n_games": len(usable),
        "avg_plate_appearances": season_pa,
        "expected_plate_appearances": projected_pa,
        "hits_dispersion": dispersion("hits", 1.10, 1.80),
        "total_bases_dispersion": dispersion("totalBases", 1.45, 2.20),
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

