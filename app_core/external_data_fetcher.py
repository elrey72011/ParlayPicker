"""
External data fetcher — wire your API connections here.

Each function returns a dict keyed by canonical team name (lowercase, no punctuation).
The pipeline reads `injuries_home_count`, `injuries_away_count`, and `weather_flag`
from the merged DataFrame; populate them here and they flow into probability adjustments.

HOW TO WIRE YOUR APIS:
  1. Fill in the API base URLs and keys in the config block below.
  2. Implement the fetch logic inside each function (stubs marked with # YOUR API HERE).
  3. The pipeline calls `enrich_with_external_data(merged_df)` automatically — no other
     changes needed.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config — fill in your API credentials
# ---------------------------------------------------------------------------
INJURY_API_BASE_URL = ""        # e.g. "https://api.sportsdata.io/v3"
INJURY_API_KEY = ""             # your API key
WEATHER_API_BASE_URL = ""       # e.g. "https://api.openweathermap.org/data/2.5"
WEATHER_API_KEY = ""            # your API key

# MLB stadium city map — used to resolve weather for outdoor parks
MLB_OUTDOOR_STADIUMS: dict[str, str] = {
    "chicago cubs": "Chicago",
    "chicago white sox": "Chicago",
    "cincinnati reds": "Cincinnati",
    "cleveland guardians": "Cleveland",
    "colorado rockies": "Denver",
    "detroit tigers": "Detroit",
    "kansas city royals": "Kansas City",
    "miami marlins": "Miami",
    "milwaukee brewers": "Milwaukee",
    "new york mets": "New York",
    "new york yankees": "New York",
    "oakland athletics": "Oakland",
    "philadelphia phillies": "Philadelphia",
    "pittsburgh pirates": "Pittsburgh",
    "san diego padres": "San Diego",
    "san francisco giants": "San Francisco",
    "seattle mariners": "Seattle",
    "st. louis cardinals": "St. Louis",
    "stl cardinals": "St. Louis",
    "texas rangers": "Dallas",
    "toronto blue jays": "Toronto",
    "washington nationals": "Washington",
    "baltimore orioles": "Baltimore",
    "boston red sox": "Boston",
    "minnesota twins": "Minneapolis",
    "los angeles dodgers": "Los Angeles",
    "los angeles angels": "Anaheim",
    "tampa bay rays": "St. Petersburg",
}

# ---------------------------------------------------------------------------
# Injury fetch
# ---------------------------------------------------------------------------

def fetch_injury_counts(league: str, home_team: str, away_team: str, game_date: str) -> dict[str, int]:
    """
    Return the number of key players OUT for each team.

    Expected return format:
        {"home": 2, "away": 0}

    Replace the stub below with your injury API call.
    Key players = starters / rotation players marked OUT or Doubtful.
    """
    if not INJURY_API_KEY or not INJURY_API_BASE_URL:
        return {"home": 0, "away": 0}

    try:
        # YOUR API HERE — example structure:
        #
        # import requests
        # url = f"{INJURY_API_BASE_URL}/{league.lower()}/injuries"
        # resp = requests.get(url, params={"apiKey": INJURY_API_KEY, "date": game_date}, timeout=5)
        # resp.raise_for_status()
        # data = resp.json()
        #
        # Count players with status "Out" or "Doubtful" per team:
        # home_out = sum(1 for p in data if _normalize_team(p["team"]) == _normalize_team(home_team) and p["status"] in {"Out", "Doubtful"})
        # away_out = sum(1 for p in data if _normalize_team(p["team"]) == _normalize_team(away_team) and p["status"] in {"Out", "Doubtful"})
        # return {"home": home_out, "away": away_out}

        return {"home": 0, "away": 0}

    except Exception as e:
        logger.warning(f"Injury fetch failed ({league} {home_team} vs {away_team}): {e}")
        return {"home": 0, "away": 0}


# ---------------------------------------------------------------------------
# Weather fetch
# ---------------------------------------------------------------------------

def fetch_weather_flag(home_team: str, game_date: str) -> float:
    """
    Return 1.0 if weather conditions will materially suppress scoring (rain, snow, strong wind).
    Return 0.0 otherwise.

    Only called for MLB outdoor stadiums — indoor/dome parks are skipped automatically.
    Replace the stub below with your weather API call.
    """
    if not WEATHER_API_KEY or not WEATHER_API_BASE_URL:
        return 0.0

    city = MLB_OUTDOOR_STADIUMS.get(home_team.lower().strip())
    if not city:
        return 0.0

    try:
        # YOUR API HERE — example structure (OpenWeatherMap):
        #
        # import requests
        # url = f"{WEATHER_API_BASE_URL}/forecast"
        # resp = requests.get(url, params={"q": city, "appid": WEATHER_API_KEY, "units": "imperial"}, timeout=5)
        # resp.raise_for_status()
        # forecast = resp.json()
        #
        # Check the forecast slot matching game_date for rain/snow/wind > 15 mph:
        # for slot in forecast.get("list", []):
        #     slot_date = slot["dt_txt"][:10]
        #     if slot_date == game_date:
        #         weather_id = slot["weather"][0]["id"]
        #         wind_speed = slot["wind"]["speed"]
        #         is_precip = weather_id < 800  # < 800 = rain/snow/thunder
        #         is_windy = wind_speed > 15
        #         return 1.0 if (is_precip or is_windy) else 0.0
        # return 0.0

        return 0.0

    except Exception as e:
        logger.warning(f"Weather fetch failed ({home_team} on {game_date}): {e}")
        return 0.0


# ---------------------------------------------------------------------------
# Pipeline enrichment entry point — called by streamlit_pipeline.py
# ---------------------------------------------------------------------------

def enrich_with_external_data(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich the merged bet-rows DataFrame with injury counts and weather flags.
    Called once per pipeline run after odds merge, before probability blending.

    Adds columns:
      - injuries_home_count (int)
      - injuries_away_count (int)
      - weather_flag (float: 0.0 or 1.0, MLB only)
    """
    if merged.empty:
        return merged

    merged = merged.copy()

    # Only run if at least one API is configured
    injuries_enabled = bool(INJURY_API_KEY and INJURY_API_BASE_URL)
    weather_enabled = bool(WEATHER_API_KEY and WEATHER_API_BASE_URL)

    if not injuries_enabled and not weather_enabled:
        if "injuries_home_count" not in merged.columns:
            merged["injuries_home_count"] = 0
        if "injuries_away_count" not in merged.columns:
            merged["injuries_away_count"] = 0
        if "weather_flag" not in merged.columns:
            merged["weather_flag"] = 0.0
        return merged

    game_ids = merged[["league", "home_team", "away_team", "game_date"]].drop_duplicates()

    injury_cache: dict[tuple[Any, ...], dict[str, int]] = {}
    weather_cache: dict[tuple[Any, ...], float] = {}

    for _, row in game_ids.iterrows():
        league = str(row.get("league", ""))
        home = str(row.get("home_team", ""))
        away = str(row.get("away_team", ""))
        date = str(row.get("game_date", ""))
        key = (league, home, away, date)

        if injuries_enabled and key not in injury_cache:
            injury_cache[key] = fetch_injury_counts(league, home, away, date)

        if weather_enabled and league.upper() == "MLB":
            wkey = (home, date)
            if wkey not in weather_cache:
                weather_cache[wkey] = fetch_weather_flag(home, date)

    def _injury_home(row: pd.Series) -> int:
        k = (str(row.get("league", "")), str(row.get("home_team", "")), str(row.get("away_team", "")), str(row.get("game_date", "")))
        return injury_cache.get(k, {}).get("home", 0)

    def _injury_away(row: pd.Series) -> int:
        k = (str(row.get("league", "")), str(row.get("home_team", "")), str(row.get("away_team", "")), str(row.get("game_date", "")))
        return injury_cache.get(k, {}).get("away", 0)

    def _weather(row: pd.Series) -> float:
        if str(row.get("league", "")).upper() != "MLB":
            return 0.0
        return weather_cache.get((str(row.get("home_team", "")), str(row.get("game_date", ""))), 0.0)

    if injuries_enabled:
        merged["injuries_home_count"] = merged.apply(_injury_home, axis=1).astype(int)
        merged["injuries_away_count"] = merged.apply(_injury_away, axis=1).astype(int)
    else:
        merged.setdefault("injuries_home_count", 0)
        merged.setdefault("injuries_away_count", 0)

    if weather_enabled:
        merged["weather_flag"] = merged.apply(_weather, axis=1)
    else:
        merged.setdefault("weather_flag", 0.0)

    return merged
