"""
External data fetcher — free APIs, no keys required.

Sources:
  - Injuries (NBA/NHL/MLB): ESPN public API (no auth)
  - Weather (MLB outdoor): wttr.in (no auth)

The pipeline calls `enrich_with_external_data(merged_df)` before probability
blending. Results flow into injury probability adjustments and weather suppression
for MLB totals.
"""
from __future__ import annotations

import logging
import re
import time
from functools import lru_cache
from typing import Any

import requests
import pandas as pd

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT = 6  # seconds per call
_CACHE_TTL = 900      # seconds — re-fetch injuries at most every 15 min

# ---------------------------------------------------------------------------
# ESPN league slugs
# ---------------------------------------------------------------------------
_ESPN_LEAGUE_SLUG: dict[str, str] = {
    "NBA": "basketball/nba",
    "NHL": "hockey/nhl",
    "MLB": "baseball/mlb",
}

# Statuses that represent a player being unavailable
_OUT_STATUSES = {"out", "doubtful", "injured reserve", "ir", "day-to-day"}

# ---------------------------------------------------------------------------
# MLB outdoor stadiums → city for wttr.in
# ---------------------------------------------------------------------------
_MLB_OUTDOOR_CITY: dict[str, str] = {
    "baltimore orioles": "Baltimore",
    "boston red sox": "Boston",
    "chicago cubs": "Chicago",
    "chicago white sox": "Chicago",
    "cincinnati reds": "Cincinnati",
    "cleveland guardians": "Cleveland",
    "colorado rockies": "Denver",
    "detroit tigers": "Detroit",
    "houston astros": "",          # Minute Maid Park is retractable roof — skip
    "kansas city royals": "Kansas+City",
    "los angeles angels": "Anaheim",
    "los angeles dodgers": "Los+Angeles",
    "miami marlins": "",           # loanDepot park has retractable roof — skip
    "milwaukee brewers": "",       # American Family Field has retractable roof — skip
    "minnesota twins": "",         # Target Field is open-air but cold — include
    "new york mets": "New+York",
    "new york yankees": "New+York",
    "oakland athletics": "Oakland",
    "philadelphia phillies": "Philadelphia",
    "pittsburgh pirates": "Pittsburgh",
    "san diego padres": "San+Diego",
    "san francisco giants": "San+Francisco",
    "seattle mariners": "",        # T-Mobile Park has retractable roof — skip
    "st. louis cardinals": "St.+Louis",
    "stl cardinals": "St.+Louis",
    "texas rangers": "",           # Globe Life Field retractable roof — skip
    "toronto blue jays": "",       # Rogers Centre retractable roof — skip
    "washington nationals": "Washington",
    "minnesota twins": "Minneapolis",
    "athletics": "Oakland",        # relocated team alias
}

# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    """Lowercase, strip punctuation/extra spaces for fuzzy matching."""
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def _teams_match(a: str, b: str) -> bool:
    """True when either name is a substring of the other after normalization."""
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return False
    # exact token overlap — covers "Lakers" matching "Los Angeles Lakers"
    tokens_a = set(na.split())
    tokens_b = set(nb.split())
    overlap = tokens_a & tokens_b
    # require at least one meaningful token (longer than 2 chars)
    return any(len(t) > 2 for t in overlap)


# ---------------------------------------------------------------------------
# ESPN injury feed — cached per league per run
# ---------------------------------------------------------------------------

_injury_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}


def _fetch_espn_injuries(league: str) -> list[dict[str, Any]]:
    """
    Fetch ESPN injury report for a league, returning a list of records:
        [{"team": "Los Angeles Lakers", "status": "out", "player": "..."}, ...]

    Results are cached for _CACHE_TTL seconds.
    """
    slug = _ESPN_LEAGUE_SLUG.get(league.upper())
    if not slug:
        return []

    now = time.time()
    cached = _injury_cache.get(league.upper())
    if cached and (now - cached[0]) < _CACHE_TTL:
        return cached[1]

    url = f"https://site.api.espn.com/apis/site/v2/sports/{slug}/injuries"
    try:
        resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"ESPN injury fetch failed for {league}: {e}")
        return []

    records: list[dict[str, Any]] = []
    # ESPN format: {"injuries": [{"team": {...}, "injuries": [{"athlete":{...}, "status":"Out"}]}]}
    for team_block in data.get("injuries", []):
        team_name = (
            team_block.get("team", {}).get("displayName")
            or team_block.get("team", {}).get("name", "")
        )
        for inj in team_block.get("injuries", []):
            status_raw = (
                inj.get("status")
                or inj.get("type", {}).get("description", "")
            ).lower().strip()
            athlete = inj.get("athlete", {}).get("displayName", "")
            records.append({"team": team_name, "status": status_raw, "player": athlete})

    _injury_cache[league.upper()] = (now, records)
    logger.info(f"ESPN injuries fetched for {league}: {len(records)} records")
    return records


def fetch_injury_counts(league: str, home_team: str, away_team: str, game_date: str) -> dict[str, int]:
    """
    Return count of key players OUT/Doubtful for home and away teams.
    Uses ESPN public API — no key required.
    """
    if league.upper() not in _ESPN_LEAGUE_SLUG:
        return {"home": 0, "away": 0}

    try:
        all_injuries = _fetch_espn_injuries(league)
        home_out = sum(
            1 for r in all_injuries
            if _teams_match(r["team"], home_team) and r["status"] in _OUT_STATUSES
        )
        away_out = sum(
            1 for r in all_injuries
            if _teams_match(r["team"], away_team) and r["status"] in _OUT_STATUSES
        )
        if home_out or away_out:
            logger.info(f"Injuries — {home_team}: {home_out} out, {away_team}: {away_out} out")
        return {"home": home_out, "away": away_out}
    except Exception as e:
        logger.warning(f"Injury count failed ({league} {home_team} vs {away_team}): {e}")
        return {"home": 0, "away": 0}


# ---------------------------------------------------------------------------
# wttr.in weather feed (MLB outdoor only)
# ---------------------------------------------------------------------------

_weather_cache: dict[str, tuple[float, float]] = {}


def fetch_weather_flag(home_team: str, game_date: str) -> float:
    """
    Return 1.0 if bad weather expected (rain, snow, wind > 15 mph) for an MLB
    outdoor game. Uses wttr.in — no key required.
    Returns 0.0 for dome/retractable-roof stadiums and on any fetch failure.
    """
    city = _MLB_OUTDOOR_CITY.get(home_team.lower().strip(), "SKIP")
    if not city:  # empty string = dome/retractable
        return 0.0

    cache_key = f"{city}:{game_date}"
    now = time.time()
    cached = _weather_cache.get(cache_key)
    if cached and (now - cached[0]) < _CACHE_TTL:
        return cached[1]

    try:
        url = f"https://wttr.in/{city}?format=j1"
        resp = requests.get(url, timeout=_REQUEST_TIMEOUT, headers={"User-Agent": "ParlayPicker/1.0"})
        resp.raise_for_status()
        data = resp.json()

        current = data.get("current_condition", [{}])[0]
        weather_code = int(current.get("weatherCode", 800))
        wind_mph = float(current.get("windspeedMiles", 0))

        # Codes below 800 = precipitation or atmosphere; 800+ = clear/cloudy
        # Specifically: 200-622 cover thunderstorm/drizzle/rain/snow
        is_precip = weather_code < 800
        is_windy = wind_mph > 15

        flag = 1.0 if (is_precip or is_windy) else 0.0
        _weather_cache[cache_key] = (now, flag)

        if flag:
            logger.info(f"Bad weather detected for {home_team} in {city}: code={weather_code}, wind={wind_mph}mph")
        return flag

    except Exception as e:
        logger.warning(f"Weather fetch failed for {home_team} ({city}): {e}")
        _weather_cache[cache_key] = (now, 0.0)
        return 0.0


# ---------------------------------------------------------------------------
# Pipeline enrichment entry point
# ---------------------------------------------------------------------------

def enrich_with_external_data(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich bet-rows DataFrame with injury counts and weather flags.
    Called once per pipeline run before probability blending.

    Adds columns:
      - injuries_home_count (int)
      - injuries_away_count (int)
      - weather_flag (float: 0.0 or 1.0, MLB outdoor only)
    """
    if merged.empty:
        merged["injuries_home_count"] = 0
        merged["injuries_away_count"] = 0
        merged["weather_flag"] = 0.0
        return merged

    merged = merged.copy()

    game_keys = merged[["league", "home_team", "away_team", "game_date"]].drop_duplicates()

    injury_cache: dict[tuple[str, str, str, str], dict[str, int]] = {}
    weather_cache_local: dict[tuple[str, str], float] = {}

    for _, row in game_keys.iterrows():
        league = str(row.get("league", ""))
        home = str(row.get("home_team", ""))
        away = str(row.get("away_team", ""))
        date = str(row.get("game_date", ""))
        key = (league, home, away, date)

        if key not in injury_cache:
            injury_cache[key] = fetch_injury_counts(league, home, away, date)

        if league.upper() == "MLB":
            wkey = (home, date)
            if wkey not in weather_cache_local:
                weather_cache_local[wkey] = fetch_weather_flag(home, date)

    def _inj_home(row: pd.Series) -> int:
        k = (str(row.get("league", "")), str(row.get("home_team", "")), str(row.get("away_team", "")), str(row.get("game_date", "")))
        return injury_cache.get(k, {}).get("home", 0)

    def _inj_away(row: pd.Series) -> int:
        k = (str(row.get("league", "")), str(row.get("home_team", "")), str(row.get("away_team", "")), str(row.get("game_date", "")))
        return injury_cache.get(k, {}).get("away", 0)

    def _wx(row: pd.Series) -> float:
        if str(row.get("league", "")).upper() != "MLB":
            return 0.0
        return weather_cache_local.get((str(row.get("home_team", "")), str(row.get("game_date", ""))), 0.0)

    merged["injuries_home_count"] = merged.apply(_inj_home, axis=1).astype(int)
    merged["injuries_away_count"] = merged.apply(_inj_away, axis=1).astype(int)
    merged["weather_flag"] = merged.apply(_wx, axis=1)

    return merged
