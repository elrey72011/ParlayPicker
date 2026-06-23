"""MLB StatsAPI pitcher form feed for the strikeout-prop model (free, public API).

Provides the projection inputs ``prop_model.project_expected_strikeouts`` needs: a
starter's recent strikeout rate (K/9) and typical innings per start, plus a team's
strikeout rate for the opponent adjustment. The parsing is split from the HTTP call so the
form math is unit-tested on a fixture without network.

StatsAPI endpoints used:
  - people/{id}/stats?stats=gameLog&group=pitching&season=YYYY  (pitcher game log)
  - teams/{id}/stats?stats=season&group=hitting&season=YYYY      (team K rate)
"""
from __future__ import annotations

from typing import Any

import requests

_BASE = "https://statsapi.mlb.com/api/v1"
_TIMEOUT = 10


def _innings_to_float(ip: Any) -> float:
    """MLB innings-pitched notation -> float. '6.1' = 6 + 1/3, '6.2' = 6 + 2/3."""
    try:
        s = str(ip)
        whole, _, frac = s.partition(".")
        outs = int(frac) if frac else 0
        return int(whole) + (outs / 3.0 if outs in (1, 2) else 0.0)
    except (ValueError, TypeError):
        return 0.0


def pitcher_form_from_gamelog(splits: list[dict], last_n: int = 5) -> dict | None:
    """Compute {k_per_9, avg_innings, n_games} from a StatsAPI pitching gameLog.

    ``splits`` is the gameLog ``splits`` list (chronological). Uses the most recent
    ``last_n`` starts. Returns None if there are no usable innings.
    """
    if not splits:
        return None
    recent = splits[-last_n:]
    total_k = 0
    total_ip = 0.0
    n = 0
    for sp in recent:
        stat = sp.get("stat", {}) if isinstance(sp, dict) else {}
        ip = _innings_to_float(stat.get("inningsPitched", 0))
        if ip <= 0:
            continue
        total_k += int(stat.get("strikeOuts", 0) or 0)
        total_ip += ip
        n += 1
    if total_ip <= 0 or n == 0:
        return None
    return {
        "k_per_9": 9.0 * total_k / total_ip,
        "avg_innings": total_ip / n,
        "n_games": n,
    }


def team_k_rate_from_stats(stat: dict) -> float | None:
    """Team strikeout rate = strikeOuts / plateAppearances from a season hitting stat blob."""
    if not isinstance(stat, dict):
        return None
    k = stat.get("strikeOuts")
    pa = stat.get("plateAppearances") or stat.get("atBats")
    try:
        k = float(k)
        pa = float(pa)
    except (TypeError, ValueError):
        return None
    if pa <= 0:
        return None
    return k / pa


def parse_schedule_probables(schedule_json: dict) -> list[dict]:
    """Per-game probable starters + team ids from a StatsAPI schedule payload.

    Resolves the strikeout-prop pieces the runner can't get from the Odds API: a pitcher's
    StatsAPI id (to fetch form) and each team's id (to fetch the opposing-lineup K rate).
    Returns one row per game with both sides' team id/name and probable-pitcher id/name; a
    side with no announced starter yet carries ``pitcher_id=None``.
    """
    rows: list[dict] = []
    for day in schedule_json.get("dates", []) if isinstance(schedule_json, dict) else []:
        for game in day.get("games", []):
            teams = game.get("teams", {}) if isinstance(game, dict) else {}
            row: dict = {}
            ok = True
            for side in ("home", "away"):
                blob = teams.get(side, {}) if isinstance(teams, dict) else {}
                team = blob.get("team", {}) or {}
                pitcher = blob.get("probablePitcher", {}) or {}
                if not team.get("name"):
                    ok = False
                    break
                row[f"{side}_team"] = team.get("name")
                row[f"{side}_team_id"] = team.get("id")
                row[f"{side}_pitcher"] = pitcher.get("fullName")
                row[f"{side}_pitcher_id"] = pitcher.get("id")
            if ok:
                rows.append(row)
    return rows


def fetch_schedule_probables(date: str, sport_id: int = 1) -> list[dict]:
    """Fetch the day's MLB schedule with probable pitchers. Returns [] on any failure."""
    try:
        url = f"{_BASE}/schedule"
        params = {"sportId": sport_id, "date": date, "hydrate": "probablePitcher"}
        resp = requests.get(url, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        return parse_schedule_probables(resp.json())
    except (requests.RequestException, ValueError, KeyError, IndexError):
        return []


def fetch_pitcher_form(pitcher_id: int, season: int, last_n: int = 5) -> dict | None:
    """Fetch a pitcher's recent form from StatsAPI. Returns None on any failure."""
    try:
        url = f"{_BASE}/people/{pitcher_id}/stats"
        params = {"stats": "gameLog", "group": "pitching", "season": season}
        resp = requests.get(url, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        stats = resp.json().get("stats", [])
        splits = stats[0].get("splits", []) if stats else []
        return pitcher_form_from_gamelog(splits, last_n=last_n)
    except (requests.RequestException, ValueError, KeyError, IndexError):
        return None


def fetch_team_k_rate(team_id: int, season: int) -> float | None:
    """Fetch a team's season strikeout rate from StatsAPI. Returns None on any failure."""
    try:
        url = f"{_BASE}/teams/{team_id}/stats"
        params = {"stats": "season", "group": "hitting", "season": season}
        resp = requests.get(url, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        stats = resp.json().get("stats", [])
        splits = stats[0].get("splits", []) if stats else []
        stat = splits[0].get("stat", {}) if splits else {}
        return team_k_rate_from_stats(stat)
    except (requests.RequestException, ValueError, KeyError, IndexError):
        return None
