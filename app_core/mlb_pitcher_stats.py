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

from datetime import datetime
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


def pitcher_form_from_gamelog(
    splits: list[dict], last_n: int = 5, as_of_date: str | None = None
) -> dict | None:
    """Compute recent starter form and optional rest-gap metadata.

    ``splits`` is the gameLog ``splits`` list (chronological). Uses the most recent
    ``last_n`` usable appearances. ``as_of_date`` lets callers detect a long layoff
    before a scheduled start.
    """
    if not splits:
        return None
    usable_splits = []
    for sp in splits:
        stat = sp.get("stat", {}) if isinstance(sp, dict) else {}
        if _innings_to_float(stat.get("inningsPitched", 0)) > 0:
            usable_splits.append(sp)
    recent = usable_splits[-max(1, int(last_n)):]

    def aggregate(rows: list[dict]) -> dict[str, Any]:
        total_k = total_bb = 0.0
        total_ip = 0.0
        strikeouts: list[float] = []
        walks: list[float] = []
        for sp in rows:
            stat = sp.get("stat", {}) if isinstance(sp, dict) else {}
            ip = _innings_to_float(stat.get("inningsPitched", 0))
            if ip <= 0:
                continue
            ks = float(stat.get("strikeOuts", 0) or 0)
            bbs = float(stat.get("baseOnBalls", 0) or 0)
            total_k += ks
            total_bb += bbs
            total_ip += ip
            strikeouts.append(ks)
            walks.append(bbs)
        return {
            "k_per_9": 9.0 * total_k / total_ip if total_ip else 0.0,
            "bb_per_9": 9.0 * total_bb / total_ip if total_ip else 0.0,
            "avg_innings": total_ip / len(strikeouts) if strikeouts else 0.0,
            "n": len(strikeouts),
            "strikeouts": strikeouts,
            "walks": walks,
        }

    def dispersion(values: list[float], default: float, maximum: float) -> float:
        if len(values) < 5:
            return default
        mean = sum(values) / len(values)
        if mean <= 0:
            return default
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        return min(maximum, max(1.05, variance / mean))

    season_form = aggregate(usable_splits)
    recent_form = aggregate(recent)
    n = int(recent_form["n"])
    if season_form["avg_innings"] <= 0 or n == 0:
        return None
    season_weight, recent_weight = 0.65, 0.35
    result = {
        "k_per_9": season_weight * season_form["k_per_9"] + recent_weight * recent_form["k_per_9"],
        "bb_per_9": season_weight * season_form["bb_per_9"] + recent_weight * recent_form["bb_per_9"],
        "avg_innings": season_weight * season_form["avg_innings"] + recent_weight * recent_form["avg_innings"],
        "n_games": n,
        "season_games": int(season_form["n"]),
        "k_dispersion": dispersion(season_form["strikeouts"], 1.15, 1.75),
        "walks_dispersion": dispersion(season_form["walks"], 1.10, 1.60),
    }
    if as_of_date:
        try:
            target = datetime.strptime(str(as_of_date), "%Y-%m-%d").date()
            game_dates = [
                datetime.strptime(str(sp.get("date")), "%Y-%m-%d").date()
                for sp in usable_splits
                if sp.get("date")
            ]
            if game_dates:
                last_date = max(game_dates)
                result["last_game_date"] = last_date.isoformat()
                result["days_since_last_start"] = max(0, (target - last_date).days)
        except (TypeError, ValueError):
            pass
    return result


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


def fetch_pitcher_form(
    pitcher_id: int, season: int, last_n: int = 5, as_of_date: str | None = None
) -> dict | None:
    """Fetch a pitcher's recent form from StatsAPI. Returns None on any failure."""
    try:
        url = f"{_BASE}/people/{pitcher_id}/stats"
        params = {"stats": "gameLog", "group": "pitching", "season": season}
        resp = requests.get(url, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        stats = resp.json().get("stats", [])
        splits = stats[0].get("splits", []) if stats else []
        return pitcher_form_from_gamelog(splits, last_n=last_n, as_of_date=as_of_date)
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

