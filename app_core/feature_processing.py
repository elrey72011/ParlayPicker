import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Mapping
import logging
import warnings
import os
import json
import threading
import concurrent.futures
import re
import time
import requests
import math
from collections import Counter

# -------------------------------------------------------------------------
# Library Imports with Fail-Safe Wrappers
# -------------------------------------------------------------------------

# -------------------------------------------------------------------
# DEBUG FALLBACK LOG THROTTLING
# -------------------------------------------------------------------
_FALLBACK_LOG_COUNT = 0
_FALLBACK_LOG_LIMIT = 15  # max number of fallback logs to emit

logger = logging.getLogger(__name__)

# Startup diagnostic logging
logger.info("==========================================================")
logger.info("Initializing app_core.feature_processing: Checking live stats dependencies...")

try:
    from nba_api.stats.endpoints import leaguedashteamstats, leaguestandings as nba_leaguestandings
    logger.info("✅ nba_api loaded successfully.")
except ImportError:
    leaguedashteamstats = None
    nba_leaguestandings = None
    logger.error("❌ nba_api not installed. NBA stats fetching will fail.")

try:
    import nfl_data_py as nfl
    logger.info("✅ nfl_data_py loaded successfully.")
except ImportError:
    nfl = None
    logger.error("❌ nfl_data_py not installed. NFL stats fetching will fail.")

# Removed cfbd due to dependency conflicts with google-genai.
# Will use direct requests to api.collegefootballdata.com instead.
logger.info("✅ NCAAF Adapter loaded successfully (using raw requests).")
# Additional explicit logging for CFBD check (will test token during actual fetch)
logger.info("  -> cfbd python package was completely removed from requirements.txt to fix deployment conflicts.")
logger.info("  -> NCAAF will gracefully skip to ESPN fallback if CFBD_API_KEY is missing or invalid.")

# Removed nhlpy as it uses a deprecated NHL API.
# Will use direct requests to api-web.nhle.com instead.
logger.info("✅ NHL Adapter loaded successfully (using api-web.nhle.com directly).")

# Removed CBBpy as it does not have a working stats adapter.
# Will use ESPN as the primary source for NCAAB.
logger.info("✅ NCAAB Adapter loaded successfully (using ESPN primary).")
logger.info("==========================================================")

try:
    import rapidfuzz
    from rapidfuzz import process, fuzz
except ImportError:
    rapidfuzz = None
    logger.warning("rapidfuzz not installed. Fuzzy matching will be degraded.")

# Streamlit secrets access for API keys
try:
    import streamlit as st
except ImportError:
    st = None

# Safety shim for st.cache_data if streamlit is not available (e.g. testing)
if st is None or not hasattr(st, "cache_data"):
    def _dummy_cache(**kwargs):
        def decorator(f):
            return f
        return decorator

    class _StShim:
        def cache_data(self, **kwargs):
            return _dummy_cache(**kwargs)
        @property
        def secrets(self):
            return {}

    if st is None:
        st = _StShim()
    else:
        # If st exists but cache_data is missing (old version?), shim it
        st.cache_data = _dummy_cache


from app_core.team_name_matcher import TeamNameMatcher
from app_core.prediction_engine import VERTEX_FEATURE_COLUMNS

# Config: Set to True to skip stats API calls on Free Tier plans
# (Now applies mainly if keys are missing or libs are missing)
FREE_TIER_MODE = True

_MASCOTS_TO_STRIP = None

# Define the features we want to ensure exist
LEAGUE_AVERAGES = {
    "NBA": {"ppg": 114.0, "oppg": 114.0, "win_pct": 0.5, "last5_win_pct": 0.5},
    "NFL": {"ppg": 22.0, "oppg": 22.0, "win_pct": 0.5, "last5_win_pct": 0.5},
    "NHL": {"ppg": 3.0, "oppg": 3.0, "win_pct": 0.5, "last5_win_pct": 0.5},
    "NCAAB": {"ppg": 72.0, "oppg": 72.0, "win_pct": 0.5, "last5_win_pct": 0.5},
    "NCAAF": {"ppg": 28.0, "oppg": 28.0, "win_pct": 0.5, "last5_win_pct": 0.5},
    "MLB": {"ppg": 4.5, "oppg": 4.5, "win_pct": 0.5, "last5_win_pct": 0.5},
    "default": {"ppg": 50.0, "oppg": 50.0, "win_pct": 0.5, "last5_win_pct": 0.5}
}


def _model_league_key(value: Any) -> str:
    """Return an exact model league without treating WNBA as NBA."""
    league = re.sub(r"[^A-Z0-9]+", " ", str(value).upper()).strip()
    if "WNBA" in league or "WOMEN S NATIONAL BASKETBALL" in league:
        return "WNBA"
    if "MLB" in league or "BASEBALL" in league:
        return "MLB"
    if "NCAAB" in league or "COLLEGE BASKETBALL" in league:
        return "NCAAB"
    if "NCAAF" in league or "COLLEGE FOOTBALL" in league:
        return "NCAAF"
    if "NHL" in league or "ICE HOCKEY" in league:
        return "NHL"
    if "NFL" in league:
        return "NFL"
    if "NBA" in league:
        return "NBA"
    return "default"


_NBA_STATS_RUNTIME_CACHE: Dict[int, List[Dict[str, Any]]] = {}
_NBA_STATS_RUNTIME_CACHE_DAY: Dict[int, str] = {}
_NBA_STATS_SUCCESS_ARCHIVE: Dict[int, Dict[str, Any]] = {}
_NBA_FETCH_DIAGNOSTICS: Dict[str, Any] = {
    "status": "not_started",
    "retries_used": 0,
    "source": "none",
    "last_error": "",
    "slate_day": "",
}


def _nba_cache_file_path(season_year: int, slate_day: str) -> str:
    cache_dir = os.path.join("data", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"nba_stats_{season_year}_{slate_day}.json")


def _load_nba_disk_cache(season_year: int, slate_day: str) -> List[Dict[str, Any]]:
    path = _nba_cache_file_path(season_year, slate_day)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, list) and payload:
            return payload
    except Exception:
        logger.warning("Failed reading NBA disk cache payload for season=%s slate_day=%s", season_year, slate_day)
    return []


def _save_nba_disk_cache(season_year: int, slate_day: str, stats: List[Dict[str, Any]]) -> None:
    if not stats:
        return
    path = _nba_cache_file_path(season_year, slate_day)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(stats, fh)
    except Exception:
        logger.warning("Failed writing NBA disk cache payload for season=%s slate_day=%s", season_year, slate_day)

# League-specific team alias mapping for stats resolution. Keep keys normalized/lowercase.
# Canonical aliases are full-name/team-name variants that should map exactly.
LEAGUE_TEAM_NAME_MAPPING: Dict[str, Dict[str, str]] = {
    "NBA": {
        "atlanta": "ATLANTA HAWKS",
        "new york": "NEW YORK KNICKS",
        "toronto": "TORONTO RAPTORS",
        "cleveland": "CLEVELAND CAVALIERS",
        "denver": "DENVER NUGGETS",
        "minnesota": "MINNESOTA TIMBERWOLVES",
        "phoenix": "PHOENIX SUNS",
        "golden state": "GOLDEN STATE WARRIORS",
        "sacramento": "SACRAMENTO KINGS",
        "orlando": "ORLANDO MAGIC",
        "detroit": "DETROIT PISTONS",
        "new orleans": "NEW ORLEANS PELICANS",
        "oklahoma city": "OKLAHOMA CITY THUNDER",
        "portland": "PORTLAND TRAIL BLAZERS",
        "utah": "UTAH JAZZ",
        "washington": "WASHINGTON WIZARDS",
        "indiana": "INDIANA PACERS",
        "charlotte": "CHARLOTTE HORNETS",
        "brooklyn": "BROOKLYN NETS",
        "boston": "BOSTON CELTICS",
        "milwaukee": "MILWAUKEE BUCKS",
        "miami": "MIAMI HEAT",
        "houston": "HOUSTON ROCKETS",
        "dallas": "DALLAS MAVERICKS",
        "memphis": "MEMPHIS GRIZZLIES",
        "chicago": "CHICAGO BULLS",
        "philadelphia": "PHILADELPHIA 76ERS",
        "san antonio": "SAN ANTONIO SPURS",
        "la clippers": "LOS ANGELES CLIPPERS",
        "los angeles clippers": "LOS ANGELES CLIPPERS",
        "la lakers": "LOS ANGELES LAKERS",
        "los angeles lakers": "LOS ANGELES LAKERS",
    },
    "NHL": {
        "minnesota": "MINNESOTA WILD",
        "montreal": "MONTREAL CANADIENS",
        "anaheim": "ANAHEIM DUCKS",
        "edmonton": "EDMONTON OILERS",
        "boston": "BOSTON BRUINS",
        "buffalo": "BUFFALO SABRES",
        "pittsburgh": "PITTSBURGH PENGUINS",
        "utah": "UTAH HOCKEY CLUB",
        "utah hockey club": "UTAH HOCKEY CLUB",
        "utah hc": "UTAH HOCKEY CLUB",
        "utah mammoth": "UTAH HOCKEY CLUB",
        "st louis": "ST LOUIS BLUES",
        "florida": "FLORIDA PANTHERS",
        "carolina": "CAROLINA HURRICANES",
        "tampa bay": "TAMPA BAY LIGHTNING",
        "new jersey": "NEW JERSEY DEVILS",
        "san jose": "SAN JOSE SHARKS",
        "vegas": "VEGAS GOLDEN KNIGHTS",
        "dallas": "DALLAS STARS",
        "washington": "WASHINGTON CAPITALS",
    },
    "NFL": {},
    "MLB": {
        "athletics": "OAKLAND",
        "oakland athletics": "OAKLAND",
        "arizona": "ARIZONA DIAMONDBACKS",
        "arizona diamondbacks": "ARIZONA DIAMONDBACKS",
        "houston": "HOUSTON ASTROS",
        "detroit": "DETROIT TIGERS",
        "baltimore": "BALTIMORE ORIOLES",
        "boston": "BOSTON RED SOX",
        "miami": "MIAMI MARLINS",
        "washington": "WASHINGTON NATIONALS",
        "st louis": "ST LOUIS CARDINALS",
        "saint louis": "ST LOUIS CARDINALS",
        "st. louis": "ST LOUIS CARDINALS",
        "texas": "TEXAS RANGERS",
        "milwaukee": "MILWAUKEE BREWERS",
        "pittsburgh": "PITTSBURGH PIRATES",
        "seattle": "SEATTLE MARINERS",
        "atlanta": "ATLANTA BRAVES",
    },
    "NCAAB": {},
    "NCAAF": {},
}

LEAGUE_CITY_ONLY_ALIASES: Dict[str, Dict[str, str]] = {
    "NBA": {
        "toronto": "TORONTO RAPTORS",
        "cleveland": "CLEVELAND CAVALIERS",
        "denver": "DENVER NUGGETS",
        "minnesota": "MINNESOTA TIMBERWOLVES",
        "new york": "NEW YORK KNICKS",
    },
    "NHL": {
        "colorado": "COLORADO AVALANCHE",
        "minnesota": "MINNESOTA WILD",
        "montreal": "MONTREAL CANADIENS",
        "anaheim": "ANAHEIM DUCKS",
        "edmonton": "EDMONTON OILERS",
        "boston": "BOSTON BRUINS",
        "buffalo": "BUFFALO SABRES",
        "pittsburgh": "PITTSBURGH PENGUINS",
        "utah": "UTAH HOCKEY CLUB",
        "utah hc": "UTAH HOCKEY CLUB",
        "utah mammoth": "UTAH HOCKEY CLUB",
    },
    "MLB": {
        "colorado": "COLORADO ROCKIES",
        "cleveland": "CLEVELAND GUARDIANS",
        "toronto": "TORONTO BLUE JAYS",
        "arizona": "ARIZONA DIAMONDBACKS",
        "houston": "HOUSTON ASTROS",
        "detroit": "DETROIT TIGERS",
        "baltimore": "BALTIMORE ORIOLES",
        "boston": "BOSTON RED SOX",
        "miami": "MIAMI MARLINS",
        "washington": "WASHINGTON NATIONALS",
        "st louis": "ST LOUIS CARDINALS",
        "saint louis": "ST LOUIS CARDINALS",
        "texas": "TEXAS RANGERS",
        "milwaukee": "MILWAUKEE BREWERS",
        "pittsburgh": "PITTSBURGH PIRATES",
        "seattle": "SEATTLE MARINERS",
        "atlanta": "ATLANTA BRAVES",
    },
    "NFL": {},
    "NCAAB": {},
    "NCAAF": {},
}

STATS_FUZZY_THRESHOLD_BY_LEAGUE: Dict[str, float] = {
    "NBA": 70.0,
    "NHL": 72.0,
    "MLB": 72.0,
    "NFL": 70.0,
    "NCAAB": 60.0,
    "NCAAF": 60.0,
}


def get_nba_fetch_diagnostics() -> Dict[str, Any]:
    """Expose current NBA fetch diagnostics for slate-level reporting."""
    return dict(_NBA_FETCH_DIAGNOSTICS)


def normalize_team_for_stats(team_name: str, league: Optional[str]) -> str:
    """
    League-aware stats normalization so aliases don't cross-pollute between sports.
    """
    def _simple_norm(value: str) -> str:
        value = value.upper().replace("'", "").replace(".", "")
        value = re.sub(r"[^A-Z0-9 ]", " ", value)
        value = re.sub(r"\s+", " ", value).strip()
        return value

    lg = str(league or "").upper().strip()
    raw = str(team_name or "").upper().strip()
    raw = re.sub(r"\s*[\(\[][0-9]+[\)\]]\s*", " ", raw).strip()
    raw = re.sub(r"\b(?:SAINT|ST\.?)(?!\w)", "ST", raw)
    raw = re.sub(r"\bL\.?A\.?\b", "LOS ANGELES", raw)
    raw = re.sub(r"\bN\.?Y\.?\b", "NEW YORK", raw)
    normalized = _simple_norm(raw)
    key = str(normalized).strip().lower()
    if lg == "NHL" and key in {"utah", "utah hockey club", "utah hc", "utah mammoth"}:
        return "UTAH HOCKEY CLUB"
    league_map = LEAGUE_TEAM_NAME_MAPPING.get(lg, {})
    if key in league_map:
        return _simple_norm(league_map[key])
    city_map = LEAGUE_CITY_ONLY_ALIASES.get(lg, {})
    if key in city_map:
        return _simple_norm(city_map[key])
    return normalized


def canonicalize_stats_team_index(stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Canonicalize stats team names for resolver matching using league-aware normalization.
    Ensures all downstream stats-index lookups use the same authoritative normalization path.
    """
    if stats_df is None or stats_df.empty:
        return pd.DataFrame()

    canonical = stats_df.copy()
    if "league_key" not in canonical.columns:
        canonical["league_key"] = ""
    canonical["league_key"] = canonical["league_key"].astype(str).str.upper().str.strip()
    if "team_norm" not in canonical.columns:
        canonical["team_norm"] = ""
    canonical["team_norm"] = canonical.apply(
        lambda row: normalize_team_for_stats(
            row.get("team_norm", "") or row.get("team_name_source", ""),
            row.get("league_key", ""),
        ),
        axis=1,
    )
    canonical["stats_team_key"] = canonical.apply(
        lambda row: normalize_team_for_stats(row.get("team_norm", ""), row.get("league_key", "")).strip().lower(),
        axis=1,
    )
    canonical = canonical[canonical["stats_team_key"] != ""]
    return canonical


def _build_stats_index_maps(stats_subset: pd.DataFrame, league: str) -> Dict[str, Dict[str, str]]:
    """
    Build strict, league-local matching maps for stats resolution.
    This is the authoritative mapping path for stats enrichment.
    """
    canonical_map: Dict[str, str] = {}
    for raw in stats_subset.index:
        raw_str = str(raw).strip().lower()
        if raw_str:
            canonical_map[raw_str] = raw_str

    league_key = str(league or "").upper().strip()
    canonical_alias_map: Dict[str, str] = {}
    city_only_alias_map: Dict[str, str] = {}

    for alias, canonical in LEAGUE_TEAM_NAME_MAPPING.get(league_key, {}).items():
        alias_key = normalize_team_for_stats(alias, league_key).strip().lower()
        canonical_key = normalize_team_for_stats(canonical, league_key).strip().lower()
        if alias_key and canonical_key in canonical_map:
            canonical_alias_map[alias_key] = canonical_map[canonical_key]

    for alias, canonical in LEAGUE_CITY_ONLY_ALIASES.get(league_key, {}).items():
        alias_key = normalize_team_for_stats(alias, league_key).strip().lower()
        canonical_key = normalize_team_for_stats(canonical, league_key).strip().lower()
        if alias_key and canonical_key in canonical_map:
            city_only_alias_map[alias_key] = canonical_map[canonical_key]

    return {
        "canonical": canonical_alias_map,
        "city_only": city_only_alias_map,
    }


def resolve_stats_team_match(
    team_name: str,
    league: str,
    stats_norm_map: Dict[str, str],
    canonical_alias_map: Dict[str, str],
    city_alias_map: Dict[str, str],
) -> tuple[Optional[str], str, str]:
    """Layered match strategy: normalized -> canonical alias -> city alias -> fuzzy -> unresolved."""
    if not str(team_name or "").strip():
        return None, "unresolved", "before_canonicalization"

    t_norm = normalize_team_for_stats(team_name, league)
    key = t_norm.strip().lower()
    if not key:
        return None, "unresolved", "before_canonicalization"

    if key in stats_norm_map:
        return stats_norm_map[key], "direct", "resolved"
    if key in canonical_alias_map:
        return canonical_alias_map[key], "canonical_alias", "resolved"
    if key in city_alias_map:
        return city_alias_map[key], "city_alias", "resolved"

    fuzzy_thresh = STATS_FUZZY_THRESHOLD_BY_LEAGUE.get(str(league).upper(), 70.0)
    if not stats_norm_map:
        return None, "unresolved", "stats_index_lookup"
    match_norm = fuzzy_match_team_robust(
        t_norm,
        list(stats_norm_map.keys()),
        threshold=fuzzy_thresh,
        league=league,
    )
    if match_norm:
        return stats_norm_map[match_norm], "fuzzy", "resolved"

    return None, "unresolved", "after_fuzzy"

def normalize_team(name: str) -> str:
    """
    Standardized normalization: Uppercase, alphanumeric + spaces only, single space.
    Aggressively strips mascots from professional and college teams.
    """
    if not name:
        return ""
    name = str(name).upper()

    # Replace apostrophes and dots with nothing
    name = name.replace("'", "").replace(".", "")

    # Replace all non-alphanumeric characters with a space BEFORE collapsing spaces
    name = re.sub(r"[^A-Z0-9 ]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()

    # Aggressively strip known mascots (Pro and College) if they are the last word
    # This prevents failures when one side has a mascot and the other doesn't
    global _MASCOTS_TO_STRIP
    if _MASCOTS_TO_STRIP is None:
        try:
            from app_core.team_name_mapping import TEAM_SUFFIXES
            _MASCOTS_TO_STRIP = []
            for m in TEAM_SUFFIXES:
                m_upper = m.upper().replace("'", "").replace(".", "")
                m_upper = re.sub(r"[^A-Z0-9 ]", "", m_upper)
                m_upper = re.sub(r"\s+", " ", m_upper).strip()
                if m_upper:
                    _MASCOTS_TO_STRIP.append(m_upper)
        except ImportError:
            _MASCOTS_TO_STRIP = []

    mascots_to_strip = _MASCOTS_TO_STRIP

    words = name.split()
    if len(words) >= 2:
        # Check against full multi-word mascots first, then fallback to last word.
        # But we must only strip if it leaves at least one word.
        for mascot in mascots_to_strip:
            if name.endswith(" " + mascot):
                new_name = name[:-len(mascot)-1].strip()
                if len(new_name.split()) >= 1: # ensure we don't strip it entirely
                    name = new_name
                    break
        else:
            # If no multi-word mascot matched, check if the single last word is a mascot
            last_word = words[-1]
            if last_word in mascots_to_strip:
                name = " ".join(words[:-1])

    return name

# -------------------------------------------------------------------------
# Legacy cross-league mappings (non-stats paths only).
# Stats enrichment now uses normalize_team_for_stats + resolve_stats_team_match
# as the single authoritative resolver path.
# -------------------------------------------------------------------------
TEAM_NAME_MAPPING = {
    # NBA: Usually unnecessary as stats match full names, but good for aliases
    # (Removed short city mappings to favor full names)
    
    # NFL (nfl_data_py returns codes like ARI, ATL) - MUST KEEP
    "arizona cardinals": "ARI",
    "atlanta falcons": "ATL",
    "baltimore ravens": "BAL",
    "buffalo bills": "BUF",
    "carolina panthers": "CAR",
    "chicago bears": "CHI",
    "cincinnati bengals": "CIN",
    "cleveland browns": "CLE",
    "dallas cowboys": "DAL",
    "denver broncos": "DEN",
    "detroit lions": "DET",
    "green bay packers": "GB",
    "houston texans": "HOU", # Explicitly requested
    "indianapolis colts": "IND",
    "jacksonville jaguars": "JAX",
    "kansas city chiefs": "KC",
    "las vegas raiders": "LV",
    "los angeles clippers": "LAC",
    "la clippers": "LAC", # Explicitly requested
    "los angeles chargers": "LAC",
    "los angeles rams": "LAR",
    "miami dolphins": "MIA",
    "minnesota vikings": "MIN",
    "new england patriots": "NE",
    "new orleans saints": "NO",
    "new york giants": "NYG",
    "new york jets": "NYJ",
    "philadelphia eagles": "PHI",
    "pittsburgh steelers": "PIT", # Explicitly requested
    "san francisco 49ers": "SF",
    "seattle seahawks": "SEA",
    "tampa bay buccaneers": "TB",
    "tennessee titans": "TEN",
    "washington commanders": "WAS",
    "washington football team": "WAS",

    # NBA Explicit Additions (Task 4) - Verified
    "los angeles clippers": "LAC",
    "la clippers": "LAC",
    "houston texans": "HOU",
    "pittsburgh steelers": "PIT",
    "la lakers": "LAL",
    "los angeles lakers": "LAL",  # Explicitly requested
    "los angeles rams": "LAR",    # Explicitly requested
    "la rams": "LAR",             # Explicitly requested

    # NHL: Mapping St. Louis correctly
    "st louis": "ST LOUIS BLUES",
    "st louis blues": "ST LOUIS BLUES",

    # NBA Mappings for `nba_api` (it expects full names)
    "orlando": "ORLANDO MAGIC",
    "sacramento": "SACRAMENTO KINGS",
    "detroit": "DETROIT PISTONS",
    "new orleans": "NEW ORLEANS PELICANS",

    # NHL Mappings for `nhlpy` (it typically expects full names like "Florida Panthers")
    "florida": "FLORIDA PANTHERS",
    "minnesota": "MINNESOTA WILD",
    "tampa bay": "TAMPA BAY LIGHTNING",
    "seattle": "SEATTLE KRAKEN",
    "vancouver": "VANCOUVER CANUCKS",
    "los angeles": "LOS ANGELES KINGS",
    "dallas": "DALLAS STARS",
    "boston": "BOSTON BRUINS",
    "chicago": "CHICAGO BLACKHAWKS",
    "carolina": "CAROLINA HURRICANES",
    "columbus": "COLUMBUS BLUE JACKETS",
    "new jersey": "NEW JERSEY DEVILS",
    "philadelphia": "PHILADELPHIA FLYERS",
    "montreal": "MONTREAL CANADIENS",
    "nashville": "NASHVILLE PREDATORS",

    # NHL Enhanced Location-Only Mappings (Task 1)
    "colorado": "COLORADO AVALANCHE",
    "florida": "FLORIDA PANTHERS",
    "carolina": "CAROLINA HURRICANES",
    "tampa bay": "TAMPA BAY LIGHTNING",
    "new jersey": "NEW JERSEY DEVILS",
    "san jose": "SAN JOSE SHARKS",
    "vegas": "VEGAS GOLDEN KNIGHTS",
    "dallas": "DALLAS STARS",
    "washington": "WASHINGTON CAPITALS",
}

# Manual overrides for team name normalization failures
# Keys and values should be UPPERCASE normalized forms
MANUAL_TEAM_OVERRIDES = {
    # NBA/NHL Fixes
    "PHOENIX SUNS": "PHOENIX SUNS", # Redundant but safe
    "CHICAGO BLACKHAWKS": "CHICAGO BLACKHAWKS",
    "WASHINGTON CAPITALS": "WASHINGTON CAPITALS",
    "WINNIPEG JETS": "WINNIPEG JETS",
    "LOS ANGELES KINGS": "LOS ANGELES KINGS",
    "UTAH MAMMOTH": "UTAH", # If Utah has new team name issues

    # NHL Enhanced Location-Only Mappings (Task 1)
    "COLORADO": "COLORADO AVALANCHE",
    "FLORIDA": "FLORIDA PANTHERS",
    "CAROLINA": "CAROLINA HURRICANES",
    "TAMPA BAY": "TAMPA BAY LIGHTNING",
    "NEW JERSEY": "NEW JERSEY DEVILS",
    "SAN JOSE": "SAN JOSE SHARKS",
    "VEGAS": "VEGAS GOLDEN KNIGHTS",
    "DALLAS": "DALLAS STARS",
    "WASHINGTON": "WASHINGTON CAPITALS",

    # NCAAB/NCAAF
    # Map common variations to the canonical name used in stats (usually School Name for colleges)
    # Most stats APIs (cfbd, cbbpy) return "Florida State", "Miami (OH)"
    "FLORIDA ST SEMINOLES": "FLORIDA STATE",
    "FLORIDA STATE SEMINOLES": "FLORIDA STATE",
    "NC STATE WOLFPACK": "NC STATE",
    "MIAMI HURRICANES": "MIAMI",
    "MIAMI OH REDHAWKS": "MIAMI OH",
    "MIAMI OHIO": "MIAMI OH",
    "UCONN HUSKIES": "CONNECTICUT",
    "UCONN": "CONNECTICUT",
    "OLE MISS REBELS": "OLE MISS",
    "MISSISSIPPI ST BULLDOGS": "MISSISSIPPI STATE",
    "LSU TIGERS": "LSU",
    "USC TROJANS": "USC",
    "TCU HORNED FROGS": "TCU",
    "SMU MUSTANGS": "SMU",
    "BYU COUGARS": "BYU",
    "UCF KNIGHTS": "UCF",
    "UNLV REBELS": "UNLV",
    "UTEP MINERS": "UTEP",
    "UTSA ROADRUNNERS": "UTSA",
    "VANDERBILT COMMODORES": "VANDERBILT",
    "VIRGINIA TECH HOKIES": "VIRGINIA TECH",
    "GEORGIA TECH YELLOW JACKETS": "GEORGIA TECH",
    "TEXAS AM AGGIES": "TEXAS AM", # Depending on how '&' is handled by normalize_team
    "TEXAS A&M AGGIES": "TEXAS AM",
    "TEXAS AM": "TEXAS AM",
    "TEXAS A&M": "TEXAS AM",

    # Task: Fix Missing NCAAB Stats - Extended Mappings (Issue #3)
    "TULANE GREEN WAVE": "TULANE",
    "NORTH TEXAS MEAN GREEN": "NORTH TEXAS",
    "MEMPHIS TIGERS": "MEMPHIS",
    "UTSA ROADRUNNERS": "UTSA",
    "HOUSTON COUGARS": "HOUSTON",
    "ARIZONA ST SUN DEVILS": "ARIZONA STATE",
    "ARIZONA STATE SUN DEVILS": "ARIZONA STATE",
    "SOUTH FLORIDA BULLS": "SOUTH FLORIDA",
    "USF BULLS": "SOUTH FLORIDA",
    "UAB BLAZERS": "UAB",
    "FAU OWLS": "FLORIDA ATLANTIC",
    "FLORIDA ATLANTIC OWLS": "FLORIDA ATLANTIC",
    "CHARLOTTE 49ERS": "CHARLOTTE",
    "RICE OWLS": "RICE",
    "TULSA GOLDEN HURRICANE": "TULSA",
    "ECU PIRATES": "EAST CAROLINA",
    "EAST CAROLINA PIRATES": "EAST CAROLINA",
    "SMU MUSTANGS": "SMU",
    "TEMPLE OWLS": "TEMPLE",
    "WICHITA ST SHOCKERS": "WICHITA STATE",
    "WICHITA STATE SHOCKERS": "WICHITA STATE",

    "MORGAN ST": "MORGAN STATE",
    "NORFOLK ST": "NORFOLK STATE",
    "HOWARD": "HOWARD",
    "MD EASTERN SHORE": "MARYLAND EASTERN SHORE",
    "PRAIRIE VIEW AM": "PRAIRIE VIEW",
    "AR PINE BLUFF": "ARKANSAS PINE BLUFF",
    "ARKANSAS PINE BLUFF GOLDEN LIONS": "ARKANSAS PINE BLUFF",
    "FLORIDA AM": "FLORIDA AM",
    "FLORIDA A&M": "FLORIDA AM",
    "TEXAS AM": "TEXAS A&M",
    "ALABAMA AM": "ALABAMA A&M",
    "NC AT": "NORTH CAROLINA AT",
    "NC CENTRAL": "NORTH CAROLINA CENTRAL",
    "NC STATE": "NC STATE",
    "NC WILMINGTON": "UNC WILMINGTON",
    "NC GREENSBORO": "UNC GREENSBORO",
    "NC ASHEVILLE": "UNC ASHEVILLE",
    "DELAWARE ST": "DELAWARE STATE",
    "COPPIN ST": "COPPIN STATE",
    "JACKSON ST": "JACKSON STATE",
    "JACKSONVILLE ST": "JACKSONVILLE STATE",
    "SAM HOUSTON ST": "SAM HOUSTON STATE",
    "STEPHEN F AUSTIN": "STEPHEN F AUSTIN",
    "SFA": "STEPHEN F AUSTIN",
    "MIDDLE TENNESSEE": "MIDDLE TENNESSEE",
    "MIDDLE TN": "MIDDLE TENNESSEE",
    "MTSU": "MIDDLE TENNESSEE",
    "WESTERN KENTUCKY": "WESTERN KENTUCKY",
    "WKU": "WESTERN KENTUCKY",
    "EASTERN KENTUCKY": "EASTERN KENTUCKY",
    "EKU": "EASTERN KENTUCKY",
    "UT ARLINGTON": "UT ARLINGTON",
    "UTA": "UT ARLINGTON",
    "UT MARTIN": "UT MARTIN",
    "UTM": "UT MARTIN",
    "UT RIO GRANDE VALLEY": "UT RIO GRANDE VALLEY",
    "UTRGV": "UT RIO GRANDE VALLEY",
    "SOUTHEAST MISSOURI": "SOUTHEAST MISSOURI STATE",
    "SEMO": "SOUTHEAST MISSOURI STATE",
    "SIU EDWARDSVILLE": "SIU EDWARDSVILLE",

    # MLB Pro Mascots
    "SEATTLE MARINERS": "SEATTLE",
    "CLEVELAND GUARDIANS": "CLEVELAND",
    "ATHLETICS": "OAKLAND",
    "OAKLAND ATHLETICS": "OAKLAND",

    # Vulnerable Single-Word Mascot Protections (Mapped to City/School Only)
    "HEAT": "MIAMI",
    "MAGIC": "ORLANDO",
    "JAZZ": "UTAH",
    "WILD": "MINNESOTA",
    "SUNS": "PHOENIX",
    "KINGS": "SACRAMENTO",
    "BLUES": "ST LOUIS",
    "STARS": "DALLAS",
    "DUCKS": "ANAHEIM",
    "SHARKS": "SAN JOSE",
    "KRAKEN": "SEATTLE",

    # Additional NCAAB mascot stripping (common variations that appear in logs)
    "DUKE BLUE DEVILS": "DUKE",
    "NORTH CAROLINA TAR HEELS": "NORTH CAROLINA",
    "KANSAS JAYHAWKS": "KANSAS",
    "KENTUCKY WILDCATS": "KENTUCKY",
    "GONZAGA BULLDOGS": "GONZAGA",
    "BAYLOR BEARS": "BAYLOR",
    "ARIZONA WILDCATS": "ARIZONA",
    "UCLA BRUINS": "UCLA",
    "HOUSTON COUGARS": "HOUSTON",
    "PURDUE BOILERMAKERS": "PURDUE",
    "CONNECTICUT HUSKIES": "CONNECTICUT",
    "VILLANOVA WILDCATS": "VILLANOVA",
    "MICHIGAN STATE SPARTANS": "MICHIGAN STATE",
    "TENNESSEE VOLUNTEERS": "TENNESSEE",
    "ALABAMA CRIMSON TIDE": "ALABAMA",
    "AUBURN TIGERS": "AUBURN",

    # Extended NCAAB mappings to fix missing stats issues
    "CREIGHTON BLUEJAYS": "CREIGHTON",
    "XAVIER MUSKETEERS": "XAVIER",
    "MARQUETTE GOLDEN EAGLES": "MARQUETTE",
    "PROVIDENCE FRIARS": "PROVIDENCE",
    "BUTLER BULLDOGS": "BUTLER",
    "SETON HALL PIRATES": "SETON HALL",
    "ST JOHNS RED STORM": "ST JOHNS",
    "SAINT JOHNS RED STORM": "ST JOHNS",
    "DEPAUL BLUE DEMONS": "DEPAUL",
    "GEORGETOWN HOYAS": "GEORGETOWN",
    "IOWA HAWKEYES": "IOWA",
    "IOWA STATE CYCLONES": "IOWA STATE",
    "ILLINOIS FIGHTING ILLINI": "ILLINOIS",
    "INDIANA HOOSIERS": "INDIANA",
    "MICHIGAN WOLVERINES": "MICHIGAN",
    "MICHIGAN": "MICHIGAN",
    "ALABAMA": "ALABAMA",
    "MINNESOTA GOLDEN GOPHERS": "MINNESOTA",
    "NEBRASKA CORNHUSKERS": "NEBRASKA",
    "NORTHWESTERN WILDCATS": "NORTHWESTERN",
    "OHIO STATE BUCKEYES": "OHIO STATE",
    "PENN STATE NITTANY LIONS": "PENN STATE",
    "RUTGERS SCARLET KNIGHTS": "RUTGERS",
    "WISCONSIN BADGERS": "WISCONSIN",
    "MARYLAND TERRAPINS": "MARYLAND",
    "SOUTH CAROLINA GAMECOCKS": "SOUTH CAROLINA",
    "ARKANSAS RAZORBACKS": "ARKANSAS",
    "FLORIDA GATORS": "FLORIDA",
    "GEORGIA BULLDOGS": "GEORGIA",
    "MISSOURI TIGERS": "MISSOURI",
    "MISSISSIPPI STATE BULLDOGS": "MISSISSIPPI STATE",
    "VANDERBILT COMMODORES": "VANDERBILT",
    "OREGON DUCKS": "OREGON",
    "OREGON STATE BEAVERS": "OREGON STATE",
    "WASHINGTON HUSKIES": "WASHINGTON",
    "WASHINGTON STATE COUGARS": "WASHINGTON STATE",
    "CALIFORNIA GOLDEN BEARS": "CALIFORNIA",
    "STANFORD CARDINAL": "STANFORD",
    "COLORADO BUFFALOES": "COLORADO",
    "UTAH UTES": "UTAH",
    "PITTSBURGH PANTHERS": "PITTSBURGH",
    "SYRACUSE ORANGE": "SYRACUSE",
    "BOSTON COLLEGE EAGLES": "BOSTON COLLEGE",
    "CLEMSON TIGERS": "CLEMSON",
    "FLORIDA STATE SEMINOLES": "FLORIDA STATE",
    "GEORGIA TECH YELLOW JACKETS": "GEORGIA TECH",
    "LOUISVILLE CARDINALS": "LOUISVILLE",
    "NOTRE DAME FIGHTING IRISH": "NOTRE DAME",
    "VIRGINIA CAVALIERS": "VIRGINIA",
    "VIRGINIA TECH HOKIES": "VIRGINIA TECH",
    "WAKE FOREST DEMON DEACONS": "WAKE FOREST",
    "CINCINNATI BEARCATS": "CINCINNATI",
    "KANSAS STATE WILDCATS": "KANSAS STATE",
    "OKLAHOMA SOONERS": "OKLAHOMA",
    "OKLAHOMA STATE COWBOYS": "OKLAHOMA STATE",
    "TEXAS LONGHORNS": "TEXAS",
    "TEXAS TECH RED RAIDERS": "TEXAS TECH",
    "WEST VIRGINIA MOUNTAINEERS": "WEST VIRGINIA",
    "TCU HORNED FROGS": "TCU",
    "BOISE STATE BRONCOS": "BOISE STATE",
    "COLORADO STATE RAMS": "COLORADO STATE",
    "FRESNO STATE BULLDOGS": "FRESNO STATE",
    "NEVADA WOLF PACK": "NEVADA",
    "NEW MEXICO LOBOS": "NEW MEXICO",
    "SAN DIEGO STATE AZTECS": "SAN DIEGO STATE",
    "UNLV REBELS": "UNLV",
    "UTAH STATE AGGIES": "UTAH STATE",
    "WYOMING COWBOYS": "WYOMING",
    "AIR FORCE FALCONS": "AIR FORCE",
    "SAN JOSE STATE SPARTANS": "SAN JOSE STATE",
    "TEXAS LONGHORNS": "TEXAS",
    "VIRGINIA CAVALIERS": "VIRGINIA",
    "ILLINOIS FIGHTING ILLINI": "ILLINOIS",
    "ARKANSAS RAZORBACKS": "ARKANSAS",
    "INDIANA HOOSIERS": "INDIANA",
    "MICHIGAN WOLVERINES": "MICHIGAN",
    "MICHIGAN": "MICHIGAN",
    "ALABAMA": "ALABAMA",
    "OHIO STATE BUCKEYES": "OHIO STATE",
    "FLORIDA GATORS": "FLORIDA",
    "TEXAS TECH RED RAIDERS": "TEXAS TECH",
    "WISCONSIN BADGERS": "WISCONSIN",
    "MARYLAND TERRAPINS": "MARYLAND",
    "IOWA HAWKEYES": "IOWA",
    "XAVIER MUSKETEERS": "XAVIER",
    "CREIGHTON BLUEJAYS": "CREIGHTON",
    "MARQUETTE GOLDEN EAGLES": "MARQUETTE",
    "PROVIDENCE FRIARS": "PROVIDENCE",
    "SETON HALL PIRATES": "SETON HALL",
    "ST JOHNS RED STORM": "ST JOHNS",
    "ST JOHN'S RED STORM": "ST JOHNS",
    "GEORGETOWN HOYAS": "GEORGETOWN",
    "BUTLER BULLDOGS": "BUTLER",
    "DEPAUL BLUE DEMONS": "DEPAUL",
    "MEMPHIS TIGERS": "MEMPHIS",
    "CINCINNATI BEARCATS": "CINCINNATI",
    "WICHITA STATE SHOCKERS": "WICHITA STATE",
    "TEMPLE OWLS": "TEMPLE",
    "TULANE GREEN WAVE": "TULANE",
    "DAYTON FLYERS": "DAYTON",
    "VCU RAMS": "VCU",
    "SAINT LOUIS BILLIKENS": "SAINT LOUIS",
    "ST BONAVENTURE BONNIES": "ST BONAVENTURE",
    "RICHMOND SPIDERS": "RICHMOND",
    "DAVIDSON WILDCATS": "DAVIDSON",
    "SAN DIEGO STATE AZTECS": "SAN DIEGO STATE",
    "NEVADA WOLF PACK": "NEVADA",
    "UTAH STATE AGGIES": "UTAH STATE",
    "BOISE STATE BRONCOS": "BOISE STATE",
    "NEW MEXICO LOBOS": "NEW MEXICO",
    "COLORADO STATE RAMS": "COLORADO STATE",
    "SAINT MARY'S GAELS": "SAINT MARY'S",
    "ST MARYS GAELS": "ST MARYS",
    "SAN FRANCISCO DONS": "SAN FRANCISCO",
    "BYU COUGARS": "BYU",
    "PEPPERDINE WAVES": "PEPPERDINE",
    "TCU HORNED FROGS": "TCU",
    "IOWA STATE CYCLONES": "IOWA STATE",
    "KANSAS STATE WILDCATS": "KANSAS STATE",
    "OKLAHOMA SOONERS": "OKLAHOMA",
    "OKLAHOMA STATE COWBOYS": "OKLAHOMA STATE",
    "WEST VIRGINIA MOUNTAINEERS": "WEST VIRGINIA",
    "LOUISVILLE CARDINALS": "LOUISVILLE",
    "SYRACUSE ORANGE": "SYRACUSE",
    "NOTRE DAME FIGHTING IRISH": "NOTRE DAME",
    "CLEMSON TIGERS": "CLEMSON",
    "WAKE FOREST DEMON DEACONS": "WAKE FOREST",
    "PITTSBURGH PANTHERS": "PITTSBURGH",
    "BOSTON COLLEGE EAGLES": "BOSTON COLLEGE",
    "LSU TIGERS": "LSU",
    "MISSISSIPPI STATE BULLDOGS": "MISSISSIPPI STATE",
    "MISSOURI TIGERS": "MISSOURI",
    "SOUTH CAROLINA GAMECOCKS": "SOUTH CAROLINA",
    "GEORGIA BULLDOGS": "GEORGIA",
    "VANDERBILT COMMODORES": "VANDERBILT",
    "OREGON DUCKS": "OREGON",
    "OREGON STATE BEAVERS": "OREGON STATE",
    "WASHINGTON HUSKIES": "WASHINGTON",
    "WASHINGTON STATE COUGARS": "WASHINGTON STATE",
    "COLORADO BUFFALOES": "COLORADO",
    "UTAH UTES": "UTAH",
    "ARIZONA STATE SUN DEVILS": "ARIZONA STATE",
    "CALIFORNIA GOLDEN BEARS": "CALIFORNIA",
    "STANFORD CARDINAL": "STANFORD",
    "RUTGERS SCARLET KNIGHTS": "RUTGERS",
    "PENN STATE NITTANY LIONS": "PENN STATE",
    "MINNESOTA GOLDEN GOPHERS": "MINNESOTA",
    "NORTHWESTERN WILDCATS": "NORTHWESTERN",
    "NEBRASKA CORNHUSKERS": "NEBRASKA",

    # User Request: Explicit Alias Mappings for NCAAB
    "WASHINGTON ST": "WASHINGTON STATE",
    "WASHINGTON ST COUGARS": "WASHINGTON STATE",
    "ARIZONA ST": "ARIZONA STATE",
    "ARIZONA ST SUN DEVILS": "ARIZONA STATE",
    "NORTH TEXAS": "NORTH TEXAS",
    "NORTH TEXAS MEAN GREEN": "NORTH TEXAS",
    "UTSA": "UT SAN ANTONIO",
    "UTSA ROADRUNNERS": "UT SAN ANTONIO",
    "EAST TEXAS AM": "TEXAS A&M COMMERCE",
    "EAST TEXAS A&M": "TEXAS A&M COMMERCE",
    "TEXAS A&M COMMERCE": "TEXAS A&M COMMERCE",
    "TAMU COMMERCE": "TEXAS A&M COMMERCE",
    "TAMUC": "TEXAS A&M COMMERCE",

    # Additional mascot variations with "STATE" already normalized
    "NORTH CAROLINA STATE WOLFPACK": "NC STATE",
    "MICHIGAN STATE SPARTANS": "MICHIGAN STATE",
    "FLORIDA STATE SEMINOLES": "FLORIDA STATE",
    "GEORGIA STATE PANTHERS": "GEORGIA STATE",
    "MISSISSIPPI STATE BULLDOGS": "MISSISSIPPI STATE",
    "WASHINGTON STATE COUGARS": "WASHINGTON STATE",
    "KANSAS STATE WILDCATS": "KANSAS STATE",
    "IOWA STATE CYCLONES": "IOWA STATE",
    "OHIO STATE BUCKEYES": "OHIO STATE",
    "OKLAHOMA STATE COWBOYS": "OKLAHOMA STATE",
    "OREGON STATE BEAVERS": "OREGON STATE",
    "ARIZONA STATE SUN DEVILS": "ARIZONA STATE",
    "SIUE": "SIU EDWARDSVILLE",
    "SOUTHERN ILLINOIS": "SOUTHERN ILLINOIS",
    "SIU": "SOUTHERN ILLINOIS",
    "LONG BEACH ST": "LONG BEACH STATE",

    # Additional NCAAB overrides to fix FallbackPlaceholderDetected issues
    "COLGATE RAIDERS": "COLGATE",
    "AMERICAN EAGLES": "AMERICAN",
    "WESTERN CAROLINA CATAMOUNTS": "WESTERN CAROLINA",
    "HOLY CROSS CRUSADERS": "HOLY CROSS",
    "NAVY MIDSHIPMEN": "NAVY",
    "MERCER BEARS": "MERCER",
    "QUEENS ROYALS": "QUEENS",
    "QUEENS NC": "QUEENS",
    "QUEENS UNIVERSITY": "QUEENS",
    "NORTH ALABAMA LIONS": "NORTH ALABAMA",
    "MURRAY STATE RACERS": "MURRAY STATE",
    "MURRAY ST": "MURRAY STATE",
    "ST MARYS": "SAINT MARYS",
    "SAINT MARYS": "SAINT MARYS",
    "ST MARY'S": "SAINT MARYS",
    "SAINT MARY'S": "SAINT MARYS",
    "ST LOUIS": "SAINT LOUIS",
    "ST BONAVENTURE": "ST BONAVENTURE",
    "ST JOHNS": "ST JOHNS",
    "SAINT JOHNS": "ST JOHNS",
    "ST JOHN'S": "ST JOHNS",

    # NHL common variations (for Colorado vs Anaheim type issues)
    "COLORADO AVALANCHE": "COLORADO AVALANCHE",
    "ANAHEIM DUCKS": "ANAHEIM DUCKS",
    "COLORADO": "COLORADO AVALANCHE",
    "FLORIDA": "FLORIDA PANTHERS",
    "CAROLINA": "CAROLINA HURRICANES",
    "TAMPA BAY": "TAMPA BAY LIGHTNING",
    "MINNESOTA": "MINNESOTA WILD",
    "DALLAS": "DALLAS STARS",
    "VEGAS": "VEGAS GOLDEN KNIGHTS",
    "SEATTLE": "SEATTLE KRAKEN",
    "CALGARY": "CALGARY FLAMES",
    "EDMONTON": "EDMONTON OILERS",
    "VANCOUVER": "VANCOUVER CANUCKS",
    "WINNIPEG": "WINNIPEG JETS",
    "LONG BEACH STATE": "LONG BEACH STATE",
    "CSU NORTHRIDGE": "CSU NORTHRIDGE",
    "CSUN": "CSU NORTHRIDGE",
    "CSU FULLERTON": "CSU FULLERTON",
    "CSU BAKERSFIELD": "CSU BAKERSFIELD",

    # NCAAB Fixes (Issue #2)
    "CREIGHTON BLUEJAYS": "CREIGHTON",
    "XAVIER MUSKETEERS": "XAVIER",
    "FORT WAYNE": "PURDUE FORT WAYNE",
    "IPFW": "PURDUE FORT WAYNE",
    "MILWAUKEE": "MILWAUKEE",
    "GEORGE MASON": "GEORGE MASON",
    "GW": "GEORGE WASHINGTON",
    "GEORGE WASHINGTON": "GEORGE WASHINGTON",
    "GWU": "GEORGE WASHINGTON",
    "G GARDNER WEBB": "GARDNER WEBB",
    "GARDNER WEBB": "GARDNER WEBB",
    "ST THOMAS MN": "ST THOMAS (MN)",
    "ST THOMAS (MN)": "ST THOMAS (MN)",
    "OMAHA": "NEBRASKA OMAHA",
    "NEBRASKA OMAHA": "NEBRASKA OMAHA",
    "QUEENS NC": "QUEENS (NC)",
    "QUEENS (NC)": "QUEENS (NC)",
    "SOUTHERN MISS": "SOUTHERN MISSISSIPPI",
    "SOUTHERN MISSISSIPPI": "SOUTHERN MISSISSIPPI",
    "ULL": "LOUISIANA",
    "LOUISIANA LAFAYETTE": "LOUISIANA",
    "UL MONROE": "LOUISIANA MONROE",
    "ULM": "LOUISIANA MONROE",

    # Extended NCAAB Aliases (Issue #2 Resolution)
    "ST FRANCIS PA": "SAINT FRANCIS (PA)",
    "SAINT FRANCIS PA": "SAINT FRANCIS (PA)",
    "ST FRANCIS (PA)": "SAINT FRANCIS (PA)",
    "MONMOUTH HAWKS": "MONMOUTH",
    "ALBANY GREAT DANES": "ALBANY",
    "TOWSON TIGERS": "TOWSON",
    "BINGHAMTON BEARCATS": "BINGHAMTON",
    "BROWN BEARS": "BROWN",
    "NJIT HIGHLANDERS": "NJIT",
    "MERCYHURST LAKERS": "MERCYHURST",
    "YALE BULLDOGS": "YALE",
    "SIENA SAINTS": "SIENA",
    "FAIRLEIGH DICKINSON KNIGHTS": "FAIRLEIGH DICKINSON",
    "HARVARD CRIMSON": "HARVARD",
    "SAINT PETERS PEACOCKS": "SAINT PETER'S",
    "MAINE BLACK BEARS": "MAINE",
    "QUINNIPIAC BOBCATS": "QUINNIPIAC",
    "MT ST MARYS": "MOUNT ST MARY'S",
    "MOUNT ST MARYS": "MOUNT ST MARY'S",
    "MT SAINT MARYS": "MOUNT ST MARY'S",
    "N COLORADO": "NORTHERN COLORADO",
    "NORTHERN COLORADO": "NORTHERN COLORADO",
    "UNC BEARS": "NORTHERN COLORADO",
    "ARIZONA ST": "ARIZONA STATE",
    "ASU": "ARIZONA STATE",
    "WASHINGTON ST": "WASHINGTON STATE",
    "WAZZU": "WASHINGTON STATE",
    "CHARLESTON SO": "CHARLESTON SOUTHERN",
    "CHARLESTON SOUTHERN": "CHARLESTON SOUTHERN",
    "CSU": "CHARLESTON SOUTHERN", # Context dependent, but often implies this in Big South
    "LOYOLA CHICAGO": "LOYOLA CHICAGO",
    "LOYOLA (IL)": "LOYOLA CHICAGO",
    "LOYOLA IL": "LOYOLA CHICAGO",
    "DETROIT MERCY": "DETROIT MERCY",
    "DETROIT": "DETROIT MERCY",
    "UD MERCY": "DETROIT MERCY",
    "ST PETERS": "SAINT PETER'S",
    "SAINT PETERS": "SAINT PETER'S",
    "ST PETER'S": "SAINT PETER'S",
    "LIU": "LONG ISLAND UNIVERSITY",
    "LONG ISLAND": "LONG ISLAND UNIVERSITY",
    "LIU BROOKLYN": "LONG ISLAND UNIVERSITY",
    "LIU SHARKS": "LONG ISLAND UNIVERSITY",
    "ST FRANCIS BK": "ST FRANCIS BROOKLYN", # Defunct/merged?
    "ST FRANCIS NY": "ST FRANCIS BROOKLYN",
    "SC STATE": "SOUTH CAROLINA STATE",
    "SOUTH CAROLINA ST": "SOUTH CAROLINA STATE",
    "SCSU": "SOUTH CAROLINA STATE",
    "UMASS LOWELL": "UMASS LOWELL",
    "MASS LOWELL": "UMASS LOWELL",
    "MASSACHUSETTS LOWELL": "UMASS LOWELL",
    "NJIT": "NJIT",
    "NEW JERSEY TECH": "NJIT",
    "UMBC": "UMBC",
    "MD BALTIMORE CO": "UMBC",
    "MARYLAND BALTIMORE COUNTY": "UMBC",
    "UNKW": "UNC WILMINGTON",
    "UNC WILMINGTON": "UNC WILMINGTON",
    "UNCW": "UNC WILMINGTON",
    "NC WILMINGTON": "UNC WILMINGTON",
    "UNCG": "UNC GREENSBORO",
    "UNC GREENSBORO": "UNC GREENSBORO",
    "NC GREENSBORO": "UNC GREENSBORO",
    "UNCA": "UNC ASHEVILLE",
    "UNC ASHEVILLE": "UNC ASHEVILLE",
    "NC ASHEVILLE": "UNC ASHEVILLE",
    "CAMPBELL": "CAMPBELL",
    "CAMPBELL CAMELS": "CAMPBELL",
    "HIGH POINT": "HIGH POINT",
    "HPU": "HIGH POINT",
    "RADFORD": "RADFORD",
    "WINTHROP": "WINTHROP",
    "PRESBYTERIAN": "PRESBYTERIAN",
    "PC BLUE HOSE": "PRESBYTERIAN",
    "USC UPSTATE": "USC UPSTATE",
    "SC UPSTATE": "USC UPSTATE",
    "SOUTH CAROLINA UPSTATE": "USC UPSTATE",
    "ETSU": "EAST TENNESSEE STATE",
    "EAST TENNESSEE ST": "EAST TENNESSEE STATE",
    "CHATTANOOGA": "CHATTANOOGA",
    "UTC": "CHATTANOOGA",
    "SAMFORD": "SAMFORD",
    "FURMAN": "FURMAN",
    "WOFFORD": "WOFFORD",
    "THE CITADEL": "THE CITADEL",
    "CITADEL": "THE CITADEL",
    "VMI": "VMI",
    "VIRGINIA MILITARY": "VMI",
    "MERCER": "MERCER",
    "WESTERN CAROLINA": "WESTERN CAROLINA",
    "WCU": "WESTERN CAROLINA",
    "UNCG": "UNC GREENSBORO",
    "SAM HOUSTON": "SAM HOUSTON STATE",
    "SAM HOUSTON ST": "SAM HOUSTON STATE",
    "SHSU": "SAM HOUSTON STATE",
    "STEPHEN F AUSTIN": "STEPHEN F AUSTIN",
    "SFA": "STEPHEN F AUSTIN",
    "ABILENE CHRISTIAN": "ABILENE CHRISTIAN",
    "ACU": "ABILENE CHRISTIAN",
    "LAMAR": "LAMAR",
    "LU": "LAMAR",
    "UT RIO GRANDE": "UT RIO GRANDE VALLEY",
    "UT RIO GRANDE VALLEY": "UT RIO GRANDE VALLEY",
    "UTRGV": "UT RIO GRANDE VALLEY",
    "TARLETON": "TARLETON STATE",
    "TARLETON ST": "TARLETON STATE",
    "DIXIE STATE": "UTAH TECH",
    "UTAH TECH": "UTAH TECH",
    "CAL BAPTIST": "CALIFORNIA BAPTIST",
    "CBU": "CALIFORNIA BAPTIST",
    "SEATTLE U": "SEATTLE U",
    "SEATTLE": "SEATTLE U",
    "GRAND CANYON": "GRAND CANYON",
    "GCU": "GRAND CANYON",
    "NEW MEXICO ST": "NEW MEXICO STATE",
    "NMSU": "NEW MEXICO STATE",
    "CHICAGO STATE": "CHICAGO STATE",
    "CSU": "CHICAGO STATE", # Conflict with Charleston Southern / Cleveland State - check context if possible or assume major one

    # Ticket: Investigate NCAAB stats coverage for low-tier conferences
    "WAGNER": "WAGNER",
    "STONEHILL": "STONEHILL",
    "SACRED HEART": "SACRED HEART",
    "LE MOYNE": "LE MOYNE",
    "LIU": "LONG ISLAND UNIVERSITY",
    "LONG ISLAND": "LONG ISLAND UNIVERSITY",
    "LONG ISLAND UNIV": "LONG ISLAND UNIVERSITY",
    "FDU": "FAIRLEIGH DICKINSON",
    "FAIRLEIGH DICKINSON": "FAIRLEIGH DICKINSON",
    "FAIRLEIGH DICKINSON KNIGHTS": "FAIRLEIGH DICKINSON",
    "CCSU": "CENTRAL CONNECTICUT",
    "CENTRAL CONNECTICUT ST": "CENTRAL CONNECTICUT",
    "CENTRAL CONNECTICUT STATE": "CENTRAL CONNECTICUT",
    "CHICAGO STATE": "CHICAGO STATE",
    "MVSU": "MISSISSIPPI VALLEY STATE",
    "MISSISSIPPI VALLEY STATE": "MISSISSIPPI VALLEY STATE",
    "MISSISSIPPI VALLEY ST": "MISSISSIPPI VALLEY STATE",
    "ALCORN STATE": "ALCORN STATE",
    "ALCORN ST": "ALCORN STATE",
    "JACKSON STATE": "JACKSON STATE",
    "JACKSON ST": "JACKSON STATE",
    "TEXAS SOUTHERN": "TEXAS SOUTHERN",
    "SOUTHERN": "SOUTHERN UNIVERSITY",
    "SOUTHERN UNIVERSITY": "SOUTHERN UNIVERSITY",
    "SOUTHERN UNIV": "SOUTHERN UNIVERSITY",
    "GRAMBLING STATE": "GRAMBLING",
    "GRAMBLING ST": "GRAMBLING",
    "GRAMBLING": "GRAMBLING",
    "PRAIRIE VIEW A&M": "PRAIRIE VIEW",
    "PRAIRIE VIEW AM": "PRAIRIE VIEW",
    "PRAIRIE VIEW": "PRAIRIE VIEW",
    "FLORIDA A&M": "FLORIDA A&M",
    "FLORIDA AM": "FLORIDA A&M",
    "BETHUNE COOKMAN": "BETHUNE-COOKMAN",
    "BETHUNE-COOKMAN": "BETHUNE-COOKMAN",
    "ALABAMA STATE": "ALABAMA STATE",
    "ALABAMA ST": "ALABAMA STATE",
    "ALABAMA A&M": "ALABAMA A&M",
    "ALABAMA AM": "ALABAMA A&M",
    "UAPB": "ARKANSAS PINE BLUFF",
    "ARKANSAS PINE BLUFF": "ARKANSAS PINE BLUFF",

    # User Reported Missing NCAAB Teams (Jan 21)
    "AMERICAN": "AMERICAN UNIVERSITY",
    "AMERICAN U": "AMERICAN UNIVERSITY",
    "COLGATE": "COLGATE",
    "CREIGHTON": "CREIGHTON",
    "XAVIER": "XAVIER",
    "KENTUCKY": "KENTUCKY",
    "TEXAS": "TEXAS",
    "RICE": "RICE",
    "TEMPLE": "TEMPLE",
    "SOUTH DAKOTA": "SOUTH DAKOTA",
    "NORTH ALABAMA": "NORTH ALABAMA",
    "QUEENS": "QUEENS (NC)",
    "QUEENS UNIVERSITY": "QUEENS (NC)",
    "QUEENS UNIV": "QUEENS (NC)",
    "OMAHA": "NEBRASKA OMAHA",
    "NEBRASKA OMAHA": "NEBRASKA OMAHA",
    "UNO": "NEBRASKA OMAHA",
    "ST THOMAS": "ST THOMAS (MN)",
    "ST THOMAS MN": "ST THOMAS (MN)",
    "BELLARMINE": "BELLARMINE",
    "LIPSCOMB": "LIPSCOMB",
    "STETSON": "STETSON",
    "AUSTIN PEAY": "AUSTIN PEAY",
    "CENTRAL ARKANSAS": "CENTRAL ARKANSAS",
    "EASTERN KENTUCKY": "EASTERN KENTUCKY",
    "NORTH FLORIDA": "NORTH FLORIDA",
    "JACKSONVILLE": "JACKSONVILLE",
    "FLORIDA GULF COAST": "FLORIDA GULF COAST",
    "FGCU": "FLORIDA GULF COAST",
    "KENNESAW STATE": "KENNESAW STATE",
    "KENNESAW ST": "KENNESAW STATE",

    # Fix: Explicit mappings for TheOver Data Issues
    "NEW ORLEANS SAINTS": "NEW ORLEANS PRIVATEERS",
    "NEW ORLEANS": "NEW ORLEANS PRIVATEERS",
    "MIAMI HURRICANES": "MIAMI", # Already present but ensuring
    "MIAMI HEAT": "MIAMI HEAT", # Protection against NBA confusion if needed

    # State abbreviation mappings
    "ARIZONA ST": "ARIZONA STATE",
    "WASHINGTON ST": "WASHINGTON STATE",
    "MONTANA ST": "MONTANA STATE",
    "N COLORADO": "NORTHERN COLORADO",
    "SE LOUISIANA": "SOUTHEASTERN LOUISIANA",
    "MT ST MARYS": "MOUNT ST MARY'S",
    "SAINT PETERS": "ST PETER'S",
    "GW": "GEORGE WASHINGTON",
    "VCU": "VIRGINIA COMMONWEALTH",
    "UTSA": "UT SAN ANTONIO",
    "TEXAS AM-CC": "TEXAS A&M-CORPUS CHRISTI",
    "ALABAMA AM": "ALABAMA A&M",
    "EAST TEXAS AM": "TEXAS A&M-COMMERCE",

    # NHL accent + city-only fixes
    "MONTRÉAL CANADIENS": "MONTREAL CANADIENS",
    "ST LOUIS": "ST LOUIS BLUES",
    "ST LOUIS BLUES": "ST LOUIS BLUES",

    # NFL
    "WASHINGTON FOOTBALL TEAM": "WASHINGTON COMMANDERS",
}

# NFL Aliases: City-only -> Full Canonical Name
# Used when the input is just "Carolina" or "Chicago" but stats are keyed by "Carolina Panthers"
NFL_TEAM_ALIASES = {
    "ARIZONA": "ARIZONA CARDINALS",
    "ATLANTA": "ATLANTA FALCONS",
    "BALTIMORE": "BALTIMORE RAVENS",
    "BUFFALO": "BUFFALO BILLS",
    "CAROLINA": "CAROLINA PANTHERS",
    "CHICAGO": "CHICAGO BEARS",
    "CINCINNATI": "CINCINNATI BENGALS",
    "CLEVELAND": "CLEVELAND BROWNS",
    "DALLAS": "DALLAS COWBOYS",
    "DENVER": "DENVER BRONCOS",
    "DETROIT": "DETROIT LIONS",
    "GREEN BAY": "GREEN BAY PACKERS",
    "HOUSTON": "HOUSTON TEXANS",
    "INDIANAPOLIS": "INDIANAPOLIS COLTS",
    "JACKSONVILLE": "JACKSONVILLE JAGUARS",
    "KANSAS CITY": "KANSAS CITY CHIEFS",
    "LAS VEGAS": "LAS VEGAS RAIDERS",
    "MIAMI": "MIAMI DOLPHINS",
    "MINNESOTA": "MINNESOTA VIKINGS",
    "NEW ENGLAND": "NEW ENGLAND PATRIOTS",
    "NEW ORLEANS": "NEW ORLEANS SAINTS",
    "PHILADELPHIA": "PHILADELPHIA EAGLES",
    "PITTSBURGH": "PITTSBURGH STEELERS",
    "SAN FRANCISCO": "SAN FRANCISCO 49ERS",
    "SEATTLE": "SEATTLE SEAHAWKS",
    "TAMPA BAY": "TAMPA BAY BUCCANEERS",
    "TENNESSEE": "TENNESSEE TITANS",
    "WASHINGTON": "WASHINGTON COMMANDERS",
    # "LOS ANGELES" and "NEW YORK" are intentionally excluded as they are ambiguous.
}

_NFL_ALIAS_LOG_COUNT = 0
_NFL_ALIAS_LOG_LIMIT = 10

def _get_secret(key_name: str) -> Optional[str]:
    """Helper to retrieve secrets from Streamlit secrets or env vars."""
    val = os.environ.get(key_name)
    if val: return val
    if st and hasattr(st, "secrets"):
        try:
            return st.secrets.get(key_name)
        except Exception:
            return None
    return None

def _parse_streak(streak_str: str) -> float:
    """Parse streak string (e.g., 'W3', 'L1') into numeric value."""
    if not streak_str:
        return 0.0
    try:
        val = int(streak_str[1:])
        return val if streak_str.startswith('W') else -val
    except Exception:
        return 0.0

def _parse_form(form_str: str) -> float:
    """Parse form string (e.g., 'WWLWL') into Win %."""
    if not form_str:
        return 0.5 # Default average
    wins = form_str.upper().count('W')
    total = len(form_str)
    return wins / total if total > 0 else 0.5

def resolve_nhl_ambiguity(team_name: str, opponent_name: str) -> str:
    """
    Resolves ambiguous NHL teams like "NEW YORK" based on opponent context.
    If it's too ambiguous and can't be resolved, it returns the original string.
    """
    team_name = team_name.upper().strip()
    opponent_name = opponent_name.upper().strip()

    if team_name != "NEW YORK":
        return team_name

    try:
        # Check current schedule to see who is playing the opponent
        url = "https://api-web.nhle.com/v1/schedule/now"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            for day in data.get('gameWeek', []):
                for game in day.get('games', []):
                    home = game.get('homeTeam', {})
                    away = game.get('awayTeam', {})

                    home_city = home.get('placeName', {}).get('default', '').upper()
                    home_name = home.get('commonName', {}).get('default', '').upper()

                    away_city = away.get('placeName', {}).get('default', '').upper()
                    away_name = away.get('commonName', {}).get('default', '').upper()

                    home_full = f"{home_city} {home_name}"
                    away_full = f"{away_city} {away_name}"

                    # If the opponent is in this game, check if the other team is NYR or NYI
                    # Normalize opponent to match
                    opp_norm = robust_normalize_team(opponent_name, "NHL")
                    if opp_norm in robust_normalize_team(home_full, "NHL") or opp_norm in robust_normalize_team(away_full, "NHL"):
                        # Opponent found in this game
                        if "RANGERS" in home_full or "RANGERS" in away_full:
                            return "NEW YORK RANGERS"
                        if "ISLANDERS" in home_full or "ISLANDERS" in away_full:
                            return "NEW YORK ISLANDERS"
    except Exception as e:
        logger.debug(f"Failed to resolve NHL ambiguity for {team_name} vs {opponent_name}: {e}")

    return team_name


def robust_normalize_team(name: str, league: Optional[str] = None) -> str:
    """
    Standardized team name normalization.
    Uses normalize_team() to produce UPPERCASE, clean strings.
    Applies overrides and aliases.
    """
    global _NFL_ALIAS_LOG_COUNT
    if not name:
        return ""

    # Task 3 Fix: Refined Pre-processing
    # Pre-processing: Handle punctuation and abbreviations before stripping chars
    name_upper = name.upper().strip()

    # Strip Tournament Seeds like "(1)" or "[12]"
    name_upper = re.sub(r"\s*[\(\[][0-9]+[\)\]]\s*", " ", name_upper).strip()

    # Handle "SAINT" / "St." / "St " -> "ST" for standardizing
    name_upper = re.sub(r"\b(?:SAINT|ST\.?)(?!\w)", "ST", name_upper)

    # Handle "L.A." / "LA" -> "LOS ANGELES"
    name_upper = re.sub(r"\bL\.?A\.?\b", "LOS ANGELES", name_upper)

    # Handle "N.Y." / "NY" -> "NEW YORK"
    name_upper = re.sub(r"\bN\.?Y\.?\b", "NEW YORK", name_upper)

    # Handle "N.C." / "NC" -> "NORTH CAROLINA" (common in NCAAB)
    # Be careful not to match inside words
    name_upper = re.sub(r"\bN\.C\.", "NORTH CAROLINA", name_upper)

    # 5. League Specific Logic
    is_nfl = False
    is_nhl = False
    if league:
        lg = str(league).strip().upper()
        if lg == "NFL":
            is_nfl = True
        elif lg == "NHL":
            is_nhl = True

    # Proactive NHL Location-Only mapping validation logging
    if is_nhl:
        # Check if the name matches one of the known risky location-only names
        nhl_location_check = {"COLORADO", "FLORIDA", "CAROLINA", "TAMPA BAY", "NEW JERSEY", "SAN JOSE", "VEGAS", "DALLAS", "WASHINGTON"}

        # We only log mapping success for these explicit shorthand names, not every normal NHL team
        # NOTE: name is the original string provided (e.g., 'Colorado')
        # name_upper is the stripped/upper version
        # Check if the exact original input (ignoring case/whitespace) is just the location
        # We only want to log when the input itself is the risky shorthand, not when the clean full name is passed.
        # Ensure we check the original string, stripped, to prevent logging 'Florida Panthers'
        if str(name).strip().upper() in nhl_location_check:
            mapped_val = MANUAL_TEAM_OVERRIDES.get(name_upper)
            if mapped_val and mapped_val != name_upper:
                # Explicitly format as requested by the user
                logger.info(f"NHL TEAM MAPPING: '{str(name).strip()}' -> '{mapped_val}'")
                # Do NOT return immediately here. We want to let it pass through standard logic
                # if needed, or we can just return it. The prompt asked to log it where it is applied.
                # Returning here is fine because it's an exact manual override match.
                return mapped_val
            elif not mapped_val:
                pass # Just let it fall through

    # 1. Base Normalization Override Check (Check overrides BEFORE aggressive stripping)
    if name_upper in MANUAL_TEAM_OVERRIDES:
        return MANUAL_TEAM_OVERRIDES[name_upper]

    # 2. Run Mascot Stripping (Uppercase, AlphaNumeric, Single Space)
    # This removes dots, ampersands, hyphens
    norm = normalize_team(name_upper)

    # 3. Post-processing for remaining variants

    # Handle "LA RAMS" explicitly (if it survived as LA RAMS or LOS ANGELES RAMS)
    if "LA RAMS" in norm:
        norm = norm.replace("LA RAMS", "LOS ANGELES RAMS")

    # Handle "MD" -> "MARYLAND" (e.g. "MD EASTERN SHORE")
    norm = re.sub(r"\bMD\b", "MARYLAND", norm)

    # Clean up multiple spaces again just in case
    norm = re.sub(r"\s+", " ", norm).strip()

    # 4. Check Overrides Again (for the stripped version)
    if norm in MANUAL_TEAM_OVERRIDES:
        return MANUAL_TEAM_OVERRIDES[norm]

    # 6. NFL Aliases (City -> Full Name)
    if is_nfl and norm in NFL_TEAM_ALIASES:
        resolved = NFL_TEAM_ALIASES[norm]
        if _NFL_ALIAS_LOG_COUNT < _NFL_ALIAS_LOG_LIMIT:
            logger.info(f"NFL Alias Applied: '{name}' -> '{resolved}'")
            _NFL_ALIAS_LOG_COUNT += 1
        return resolved

    return norm

def log_stats_match_summary(stats_log: Dict[str, int], league: str):
    """
    Log a summary of stats matching for a league.
    Tracks match rate and fallback percentage for data quality monitoring.
    """
    if league == "default": return

    total = sum(stats_log.values())
    if total == 0: return

    direct = stats_log.get("direct", 0)
    override = stats_log.get("canonical_alias", 0) + stats_log.get("city_alias", 0)
    fuzzy = stats_log.get("fuzzy", 0)
    miss = stats_log.get("miss", 0)

    matched = direct + override + fuzzy
    match_rate = (matched / total) * 100
    fallback_rate = (miss / total) * 100

    logger.info(f"STATS MATCH REPORT [{league}]: {match_rate:.1f}% Matched, {fallback_rate:.1f}% Fallback ({direct} direct, {override} override, {fuzzy} fuzzy, {miss} miss)")

    # Store metrics for export reporting (optional: store in session state)
    if not hasattr(log_stats_match_summary, '_metrics'):
        log_stats_match_summary._metrics = {}
    log_stats_match_summary._metrics[league] = {
        "match_rate": match_rate,
        "fallback_rate": fallback_rate,
        "direct": direct,
        "override": override,
        "fuzzy": fuzzy,
        "miss": miss,
        "total": total
    }


def get_stats_match_metrics() -> Dict[str, Dict[str, Any]]:
    """
    Retrieve stats match metrics for all leagues.
    Returns dict of league -> metrics.
    """
    if hasattr(log_stats_match_summary, '_metrics'):
        return log_stats_match_summary._metrics
    return {}

def debug_team_mapping_health():
    """
    Run manual validation of team mappings for known tricky cases.
    """
    cases = {
        "NFL": ["Carolina", "Chicago", "LA", "NY Giants", "WSH"],
        "NCAAB": ["NC State", "Miami (OH)", "UConn", "Ole Miss", "St. Mary's"],
        "NHL": ["St Louis", "Montreal", "NY Rangers"],
        "NBA": ["LA Lakers", "Philly"]
    }
    results = {}
    for lg, teams in cases.items():
        results[lg] = {}
        for t in teams:
            norm = robust_normalize_team(t, lg)
            results[lg][t] = norm
    return results

def normalize_team_name_simple(name):
    """Simple normalization for fuzzy matching Option B"""
    if not name: return ""
    name = name.lower().replace('.', '').replace("'", '').replace('-', ' ')
    name = ' '.join(name.split())
    return name

def fuzzy_match_team_robust(target: str, choices: List[str], threshold: float = 80.0, league: Optional[str] = None) -> Optional[str]:
    """
    Uses rapidfuzz to find the best match for 'target' in 'choices'.
    Returns the matched string from 'choices' if score > threshold, else None.
    Updated to use fuzz.ratio on normalized strings per user request (Option B).
    """
    if not target or not choices:
        return None

    if rapidfuzz:
        # Option B Implementation: Normalize and use fuzz.ratio
        normalized_target = normalize_team_name_simple(target)
        best_match = None
        best_score = 0

        for choice in choices:
            normalized_choice = normalize_team_name_simple(choice)
            score = fuzz.ratio(normalized_target, normalized_choice)
            if score > best_score:
                best_score = score
                best_match = choice

        if best_score >= threshold:
            return best_match

        # Fallback to token_sort_ratio if direct ratio fails (handles word reordering)
        result = process.extractOne(target, choices, scorer=fuzz.token_sort_ratio)
        if result:
            match, score, _ = result
            if score >= threshold:
                return match
    else:
        # Fallback to TeamNameMatcher (difflib) if rapidfuzz missing
        return TeamNameMatcher.match_team(target, choices, threshold=threshold/100.0, league=league)

    return None

# -------------------------------------------------------------------------
# New Open-Source Stats Fetching Helpers
# -------------------------------------------------------------------------

def fetch_nba_stats(season_year: int) -> List[Dict[str, Any]]:
    """
    Fetch NBA stats using nba_api for the given season year (e.g. 2024 for 2024-25).
    """
    slate_day = datetime.utcnow().strftime("%Y-%m-%d")
    if (
        season_year in _NBA_STATS_RUNTIME_CACHE
        and _NBA_STATS_RUNTIME_CACHE[season_year]
        and _NBA_STATS_RUNTIME_CACHE_DAY.get(season_year) == slate_day
    ):
        _NBA_FETCH_DIAGNOSTICS.update({
            "status": "ok",
            "retries_used": 0,
            "source": "runtime_cache",
            "last_error": "",
            "slate_day": slate_day,
        })
        return _NBA_STATS_RUNTIME_CACHE[season_year]

    if leaguedashteamstats is None:
        max_attempts = 0
        last_exc = "nba_api unavailable"
    else:
        max_attempts = 3
        last_exc = None

    # nba_api expects season format "YYYY-YY", e.g. "2024-25"
    season_str = f"{season_year}-{str(season_year + 1)[-2:]}"
    base_timeout = 6

    for attempt in range(max_attempts):
        try:
            logger.info(f"Fetching NBA stats for season: {season_str} (attempt {attempt + 1}/{max_attempts})")
            dashboard = leaguedashteamstats.LeagueDashTeamStats(
                season=season_str,
                measure_type_detailed_defense='Base',
                per_mode_detailed='PerGame',
                timeout=base_timeout,
            )
            df = dashboard.get_data_frames()[0]

            # Fetch last-5 win% via separate call
            last5_by_team: dict = {}
            try:
                dashboard_l5 = leaguedashteamstats.LeagueDashTeamStats(
                    season=season_str,
                    measure_type_detailed_defense='Base',
                    per_mode_detailed='PerGame',
                    last_n_games=5,
                    timeout=base_timeout,
                )
                df_l5 = dashboard_l5.get_data_frames()[0]
                for _, r in df_l5.iterrows():
                    last5_by_team[str(r['TEAM_NAME'])] = float(r['W_PCT'])
            except Exception as e_l5:
                logger.warning(f"NBA last-5 fetch failed: {e_l5}")

            # Fetch current streak via LeagueStandings
            streak_by_team: dict = {}
            try:
                if nba_leaguestandings is not None:
                    standings = nba_leaguestandings.LeagueStandings(season=season_str, timeout=base_timeout)
                    sdf = standings.get_data_frames()[0]
                    for _, r in sdf.iterrows():
                        raw = str(r.get('strCurrentStreak', '') or '')
                        # Format: "W 3" or "L 2"
                        parts = raw.strip().split()
                        if len(parts) == 2:
                            try:
                                val = int(parts[1])
                                streak_by_team[str(r['TeamName'])] = float(val) if parts[0].upper() == 'W' else float(-val)
                            except ValueError:
                                pass
            except Exception as e_st:
                logger.warning(f"NBA standings/streak fetch failed: {e_st}")

            stats = []
            for _, row in df.iterrows():
                team_name = str(row['TEAM_NAME'])
                gp = float(row['GP'])
                pts = float(row['PTS'])
                plus_minus = float(row['PLUS_MINUS'])
                w_pct = float(row['W_PCT'])
                tov = float(row['TOV']) if 'TOV' in row else 0.0
                ast = float(row['AST']) if 'AST' in row else 0.0
                reb = float(row['REB']) if 'REB' in row else 0.0
                if abs(plus_minus) > 30 and gp > 0:
                    plus_minus /= gp
                oppg = (pts - plus_minus)
                last5 = last5_by_team.get(team_name, w_pct)
                streak = streak_by_team.get(team_name, 0.0)

                stats.append({
                    "team_norm": normalize_team_for_stats(team_name, league="NBA"),
                    "league_key": "NBA",
                    "win_pct": w_pct,
                    "home_win_pct": w_pct,
                    "away_win_pct": w_pct,
                    "points_per_game": pts,
                    "points_allowed_per_game": oppg,
                    "assists_per_game": ast,
                    "rebounds_per_game": reb,
                    "turnovers": tov,
                    "streak": streak,
                    "last5_win_pct": last5,
                })

            _NBA_STATS_RUNTIME_CACHE[season_year] = stats
            _NBA_STATS_RUNTIME_CACHE_DAY[season_year] = slate_day
            _NBA_STATS_SUCCESS_ARCHIVE[season_year] = {
                "slate_day": slate_day,
                "stats": list(stats),
            }
            _save_nba_disk_cache(season_year, slate_day, stats)
            _NBA_FETCH_DIAGNOSTICS.update({
                "status": "ok",
                "retries_used": attempt,
                "source": "live",
                "last_error": "",
                "slate_day": slate_day,
            })
            logger.info(f"Successfully fetched NBA stats for {len(stats)} teams.")
            return stats
        except Exception as e:
            last_exc = e
            if attempt < max_attempts - 1:
                time.sleep(0.75 * (attempt + 1))
            continue

    # After live failure, recover in strict source order: runtime success archive -> disk -> failed.
    # Include previous season as a same-day recovery candidate to handle season-key drift around boundaries.
    recovery_seasons = [season_year]
    if (season_year - 1) not in recovery_seasons:
        recovery_seasons.append(season_year - 1)

    for recovery_season in recovery_seasons:
        archived = _NBA_STATS_SUCCESS_ARCHIVE.get(recovery_season, {})
        archived_stats = archived.get("stats", []) if isinstance(archived, dict) else []
        if archived.get("slate_day") != slate_day or not archived_stats:
            continue
        _NBA_STATS_RUNTIME_CACHE[season_year] = list(archived_stats)
        _NBA_STATS_RUNTIME_CACHE_DAY[season_year] = slate_day
        _NBA_FETCH_DIAGNOSTICS.update({
            "status": "ok",
            "retries_used": max_attempts - 1,
            "source": "runtime_cache",
            "last_error": str(last_exc) if last_exc else "",
            "slate_day": slate_day,
            "recovered_from_season": int(recovery_season),
        })
        logger.warning("NBA live fetch failed; using same-slate runtime-cache NBA stats payload (recovery_season=%s).", recovery_season)
        return _NBA_STATS_RUNTIME_CACHE[season_year]

    for recovery_season in recovery_seasons:
        disk_cached_stats = _load_nba_disk_cache(recovery_season, slate_day)
        if not disk_cached_stats:
            continue
        _NBA_STATS_RUNTIME_CACHE[season_year] = list(disk_cached_stats)
        _NBA_STATS_RUNTIME_CACHE_DAY[season_year] = slate_day
        _NBA_STATS_SUCCESS_ARCHIVE[season_year] = {"slate_day": slate_day, "stats": list(disk_cached_stats)}
        _NBA_FETCH_DIAGNOSTICS.update({
            "status": "ok",
            "retries_used": max_attempts - 1,
            "source": "disk_cache",
            "last_error": str(last_exc) if last_exc else "",
            "slate_day": slate_day,
            "recovered_from_season": int(recovery_season),
        })
        logger.warning("NBA live fetch failed; using same-slate disk-cached NBA stats payload (recovery_season=%s).", recovery_season)
        return _NBA_STATS_RUNTIME_CACHE[season_year]

    _NBA_FETCH_DIAGNOSTICS.update({
        "status": "failed",
        "retries_used": max_attempts - 1,
        "source": "failed",
        "last_error": str(last_exc) if last_exc else "unknown error",
        "slate_day": slate_day,
    })
    logger.error(f"Failed to fetch NBA stats via nba_api after {max_attempts} attempts: {last_exc}", exc_info=True)
    return []

@st.cache_data(ttl=21600)
def fetch_nfl_stats(season_year: int) -> List[Dict[str, Any]]:
    """
    Fetch NFL stats using nfl_data_py for the given season year.
    Uses 'import_schedules' to aggregate team stats from game results.
    """
    if nfl is None:
        return []

    try:
        logger.info(f"Fetching NFL stats for season: {season_year}")
        # Use schedule data which has scores
        df = nfl.import_schedules([season_year])

        team_stats = {}

        def update_team(team, scored, allowed, won, turnovers=0):
            if team not in team_stats:
                team_stats[team] = {'games': 0, 'wins': 0, 'points_for': 0, 'points_against': 0, 'turnovers': 0}
            team_stats[team]['games'] += 1
            team_stats[team]['points_for'] += scored
            team_stats[team]['points_against'] += allowed
            if won:
                team_stats[team]['wins'] += 1
            team_stats[team]['turnovers'] += turnovers

        # nfl_data_py schedules df has 'home_turnovers' and 'away_turnovers' if recent enough
        # checking columns availability
        has_turnovers = 'home_turnovers' in df.columns and 'away_turnovers' in df.columns

        for _, row in df.iterrows():
            if pd.isna(row['result']): # Game not played yet
                continue

            home = row['home_team']
            away = row['away_team']
            home_score = row['home_score']
            away_score = row['away_score']

            home_to = row['home_turnovers'] if has_turnovers else 0
            away_to = row['away_turnovers'] if has_turnovers else 0

            # Handle potential NaNs
            if pd.isna(home_score) or pd.isna(away_score):
                continue

            home_to = home_to if pd.notnull(home_to) else 0
            away_to = away_to if pd.notnull(away_to) else 0

            update_team(home, home_score, away_score, home_score > away_score, home_to)
            update_team(away, away_score, home_score, away_score > home_score, away_to)

        stats = []
        for team_code, data in team_stats.items():
            games = data['games']
            if games == 0: continue

            w_pct = data['wins'] / games
            ppg = data['points_for'] / games
            oppg = data['points_against'] / games
            avg_tov = data['turnovers'] / games

            stats.append({
                "team_norm": normalize_team_for_stats(str(team_code), league="NFL"),
                "league_key": "NFL",
                "win_pct": w_pct,
                "home_win_pct": w_pct,
                "away_win_pct": w_pct,
                "points_per_game": ppg,
                "points_allowed_per_game": oppg,
                "turnovers": avg_tov,
                "streak": 0.0,
                "last5_win_pct": w_pct
            })

        logger.info(f"Successfully fetched NFL stats for {len(stats)} teams.")
        return stats
    except Exception as e:
        logger.error(f"Failed to fetch NFL stats via nfl_data_py: {e}", exc_info=True)
        return []

@st.cache_data(ttl=21600)
def fetch_ncaaf_stats(season_year: int) -> List[Dict[str, Any]]:
    """
    Fetch NCAAF stats using direct requests to api.collegefootballdata.com.

    Returns empty list if API is unavailable, unauthorized, or disabled for this session.
    """
    # Check if CFBD was disabled earlier in this session (e.g., due to 401)
    if st is not None and hasattr(st, "session_state"):
        if st.session_state.get("cfbd_disabled_reason"):
            # Already disabled - skip silently to avoid repeated warnings
            return []

    def _normalize_cfbd_token(raw: Any) -> str:
        if raw is None:
            return ""
        s = str(raw).strip()
        if not s:
            return ""
        # Remove Bearer prefix (case-insensitive) if present
        if s.lower().startswith("bearer"):
            parts = s.split(None, 1)
            s = parts[1].strip() if len(parts) == 2 else ""
        # CRITICAL: remove ALL whitespace/newlines from multiline secrets
        s = "".join(s.split())
        return s

    # Explicitly check st.secrets if _get_secret fails or for robustness
    raw_key = _get_secret("CFBD_API_KEY") or _get_secret("CFBDAPIKEY")
    if not raw_key and st is not None:
        try:
            if hasattr(st, "secrets"):
                if "CFBD_API_KEY" in st.secrets:
                    raw_key = st.secrets["CFBD_API_KEY"]
                elif "CFBDAPIKEY" in st.secrets:
                    raw_key = st.secrets["CFBDAPIKEY"]
        except Exception:
            pass

    token = _normalize_cfbd_token(raw_key)
    if not token:
        logger.warning("CFBD_API_KEY not found. Skipping live NCAAF API stats. (Will use ESPN fallback)")
        return fetch_from_espn_ncaaf(season_year)

    logger.info(f"CFBD auth prepared via raw requests (token_length={len(token)})")

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }

    def _is_unauthorized(status_code: int) -> bool:
        return status_code == 401

    def _fetch_from_api(endpoint: str, params: Dict[str, Any]) -> List[Any]:
        url = f"https://api.collegefootballdata.com{endpoint}"
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, headers=headers, params=params, timeout=10)
                if _is_unauthorized(resp.status_code):
                    if st is not None and hasattr(st, "session_state"):
                        st.session_state["cfbd_disabled_reason"] = "UNAUTHORIZED_401"
                    logger.warning(
                        f"⚠️ CFBD unauthorized (401) when fetching from {endpoint}. "
                        "Disabling CFBD for this session. Check CFBD_API_KEY value in Streamlit secrets."
                    )
                    return []
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.RequestException as e:
                status = getattr(e.response, 'status_code', None)
                if status == 401:
                    if st is not None and hasattr(st, "session_state"):
                        st.session_state["cfbd_disabled_reason"] = "UNAUTHORIZED_401"
                    logger.warning("⚠️ CFBD unauthorized (401). Disabling.")
                    return []
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                logger.warning(f"CFBD API error for {endpoint}: {e}")
        return []

    try:
        # 1) requested year
        season_stats = _fetch_from_api("/stats/season", {"year": season_year})

        # 2) fallback to prior year if empty
        if not season_stats:
            logger.warning(f"No NCAAF stats found for {season_year}. Trying {season_year - 1}...")
            season_stats = _fetch_from_api("/stats/season", {"year": season_year - 1})
            if season_stats:
                season_year = season_year - 1

        season_games = _fetch_from_api("/games", {"year": season_year})

        # Build win pct map
        team_records: Dict[str, Dict[str, int]] = {}
        for g in season_games or []:
            try:
                home = g.get("home_team")
                away = g.get("away_team")
                home_pts = g.get("home_points")
                away_pts = g.get("away_points")

                if not home or not away:
                    continue

                team_records.setdefault(home, {"wins": 0, "losses": 0})
                team_records.setdefault(away, {"wins": 0, "losses": 0})

                if home_pts is None or away_pts is None:
                    continue

                if home_pts > away_pts:
                    team_records[home]["wins"] += 1
                    team_records[away]["losses"] += 1
                elif away_pts > home_pts:
                    team_records[away]["wins"] += 1
                    team_records[home]["losses"] += 1
            except Exception:
                continue

        stats: List[Dict[str, Any]] = []
        for offense in season_stats or []:
            try:
                team = offense.get("team") or offense.get("school")
                if not team:
                    continue

                wins = team_records.get(team, {}).get("wins", 0)
                losses = team_records.get(team, {}).get("losses", 0)
                games_played = wins + losses
                win_pct = (wins / games_played) if games_played > 0 else 0.0

                # Data might be in total points or points_per_game depending on the endpoint structure.
                # Assuming /stats/season returns dicts with 'statName' and 'statValue'
                # or similar depending on the CFBD API version.
                # Note: /stats/season actually returns a list of dictionaries with stats.
                # If the schema from cfbd-python was different, we map it.
                # The CFBD /stats/season endpoint returns:
                # [ { "team": "Air Force", "statName": "scoringOffense", "statValue": 35.5 }, ... ]
                # But looking at old code, it expected `offense.points`, `offense.points_per_game`.
                # Let's handle both.

                ppg = offense.get("points")
                if ppg is None:
                    ppg = offense.get("points_per_game", 0.0) or 0.0

                oppg = offense.get("points_allowed")
                if oppg is None:
                    oppg = offense.get("points_allowed_per_game", 0.0) or 0.0

                ypg = offense.get("yards_per_game")
                if ypg is None:
                    ypg = offense.get("yards", 0.0) or 0.0

                avg_tov = offense.get("turnovers")
                if avg_tov is None:
                    avg_tov = offense.get("turnovers_per_game", 0.0) or 0.0

                def _to_float(x: Any) -> float:
                    try:
                        return float(x)
                    except Exception:
                        return 0.0

                stats.append(
                    {
                        "team_norm": normalize_team_for_stats(team, league="NCAAF"),
                        "wins": wins,
                        "losses": losses,
                        "win_pct": float(win_pct),
                        "points_per_game": _to_float(ppg),
                        "points_allowed_per_game": _to_float(oppg),
                        "yards_per_game": _to_float(ypg),
                        "turnovers": _to_float(avg_tov),
                        "streak": 0.0,
                        "last5_win_pct": float(win_pct),
                    }
                )
            except Exception:
                continue

        # If the API returns key-value stat pairs rather than a single object per team
        # (which CFBD's /stats/season often does), we need to aggregate them if the stats list is empty
        # or mispopulated above.
        if stats and "statName" in season_stats[0]:
            # It's the key-value structure
            agg_stats = {}
            for s in season_stats:
                t = s.get("team")
                if not t: continue
                if t not in agg_stats:
                    agg_stats[t] = {"team": t}
                name = s.get("statName")
                val = s.get("statValue")
                if name == "scoringOffense": agg_stats[t]["points_per_game"] = val
                if name == "scoringDefense": agg_stats[t]["points_allowed_per_game"] = val
                if name == "totalOffense": agg_stats[t]["yards_per_game"] = val
                if name == "turnoversLost": agg_stats[t]["turnovers"] = val

            # Rebuild stats list
            stats = []
            for t, data in agg_stats.items():
                wins = team_records.get(t, {}).get("wins", 0)
                losses = team_records.get(t, {}).get("losses", 0)
                games_played = wins + losses
                win_pct = (wins / games_played) if games_played > 0 else 0.0

                stats.append({
                    "team_norm": normalize_team_for_stats(t, league="NCAAF"),
                    "wins": wins,
                    "losses": losses,
                    "win_pct": float(win_pct),
                    "points_per_game": float(data.get("points_per_game", 0.0)),
                    "points_allowed_per_game": float(data.get("points_allowed_per_game", 0.0)),
                    "yards_per_game": float(data.get("yards_per_game", 0.0)),
                    "turnovers": float(data.get("turnovers", 0.0)),
                    "streak": 0.0,
                    "last5_win_pct": float(win_pct),
                })

        logger.info(f"Successfully fetched NCAAF stats for {len(stats)} teams.")

    except Exception as e:
        logger.error(f"Failed to fetch NCAAF stats via API: {e}", exc_info=True)
        stats = []

    # Merge CFBD stats with ESPN Fallback (Power-6 / Scraper miss fix)
    logger.info("Merging NCAAF stats from ESPN fallback...")
    espn_stats = fetch_from_espn_ncaaf(season_year)

    # Priority: CFBD (Scraper) > ESPN
    merged_stats = {s["team_norm"]: s for s in stats}
    for espn_s in espn_stats:
        team_norm = espn_s["team_norm"]
        if team_norm not in merged_stats:
            merged_stats[team_norm] = espn_s
        else:
            # Check if scraper version has 0 games (or wins+losses = 0)
            existing_s = merged_stats[team_norm]
            games_played = existing_s.get("wins", 0) + existing_s.get("losses", 0)
            if games_played == 0:
                merged_stats[team_norm] = espn_s

    final_stats = list(merged_stats.values())
    logger.info(f"Final merged NCAAF stats count: {len(final_stats)}")
    return final_stats

@st.cache_data(ttl=21600)
def fetch_nhl_stats(season_year: int) -> List[Dict[str, Any]]:
    """
    Fetch NHL stats using the official NHL Edge API directly.
    """
    try:
        logger.info(f"Fetching NHL stats from api-web.nhle.com for season: {season_year}")
        url = "https://api-web.nhle.com/v1/standings/now"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        stats = []
        for entry in data.get('standings', []):
            team_name = entry.get('teamName', {}).get('default', '')
            if not team_name:
                # Some API updates use teamCommonName or placeName
                team_name = entry.get('teamCommonName', {}).get('default', '')
                if not team_name:
                    continue

            # Edge API provides full placeName and commonName (e.g. "New York" and "Rangers")
            # Let's combine them if placeName exists to ensure full "New York Rangers"
            place_name = entry.get('placeName', {}).get('default', '')
            common_name = entry.get('teamCommonName', {}).get('default', '')

            # Use the combined name if possible to be safe
            full_name = f"{place_name} {common_name}".strip()
            if not full_name:
                full_name = team_name
            
            # Extract stats
            games = entry.get('gamesPlayed', 0)
            if games == 0: continue
            
            points_for = entry.get('goalFor', 0)
            points_against = entry.get('goalAgainst', 0)
            wins = entry.get('wins', 0)
            
            win_pct = wins / games

            # Metrics
            ppg = points_for / games
            oppg = points_against / games
            
            # Streak
            streak_code = entry.get('streakCode', '') # e.g. 'W2' or 'L1' or 'W'
            # Streak count is separate in the Edge API sometimes, check both
            streak_count = entry.get('streakCount', 1)
            if streak_code and not any(char.isdigit() for char in streak_code):
                streak_code = f"{streak_code}{streak_count}"

            streak = _parse_streak(streak_code)
            
            l10_wins = entry.get('l10Wins', 0)
            l10_games = entry.get('l10GamesPlayed', 10)
            l10_win_pct = (l10_wins / l10_games) if l10_games > 0 else win_pct
            
            team_norm = normalize_team_for_stats(full_name, league="NHL")
            stats.append({
                "team_norm": team_norm,
                "stats_team_key": team_norm.strip().lower(),
                "team_name_source": full_name,
                "league_key": "NHL",
                "win_pct": win_pct,
                "home_win_pct": win_pct,
                "away_win_pct": win_pct,
                "points_per_game": ppg,
                "points_allowed_per_game": oppg,
                "turnovers": 0.0,
                "streak": streak,
                "last5_win_pct": l10_win_pct
            })
            
        logger.info(f"Successfully fetched NHL stats for {len(stats)} teams.")
        return stats
    except Exception as e:
        logger.error(f"Failed to fetch NHL stats: {e}", exc_info=True)
        return []

def find_entries(node):
    entries = []
    if isinstance(node, dict):
        if "entries" in node:
            entries.extend(node["entries"])
        for key in ["children", "standings", "groups"]:
            if key in node:
                entries.extend(find_entries(node[key]))
    elif isinstance(node, list):
        for child in node:
            entries.extend(find_entries(child))
    return entries

def fetch_from_espn_ncaaf(season_year: int) -> List[Dict[str, Any]]:
    """
    Fetch NCAAF stats from ESPN hidden API (Fallback).
    """
    try:
        logger.info(f"Fetching NCAAF stats from ESPN for season: {season_year}")
        url = "https://site.api.espn.com/apis/v2/sports/football/college-football/standings"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        stats = []

        all_entries = find_entries(data)

        for entry in all_entries:
            team_info = entry.get("team", {})
            team_name = team_info.get("displayName")
            if not team_name:
                continue
            team_norm = normalize_team_for_stats(team_name, "NCAAF")

            # Resilient stat extraction: Search keywords in both name and abbreviation
            wins, losses, gp, p_for, p_against = 0.0, 0.0, 0.0, 0.0, 0.0
            p_avg, d_avg = 0.0, 0.0
            for s in entry.get("stats", []):
                nm, ab = str(s.get("name", "")).lower(), str(s.get("abbreviation", "")).lower()
                val = float(s.get("value", 0))
                if "wins" in nm or ab == "w": wins = val
                elif "losses" in nm or ab == "l": losses = val
                elif "gamesplayed" in nm or ab == "gp": gp = val
                elif any(x in nm for x in ["pointsagainst", "pointsallowed", "runsagainst"]) or ab in ["ra", "pa"]: p_against = val
                elif any(x in nm for x in ["pointsfor", "points", "runs"]) or ab in ["r", "pf"]: p_for = val
                elif nm in ["avg", "ppg", "apf"] or ab == "apf": p_avg = val
                elif nm in ["apa", "opp ppg"] or ab == "apa": d_avg = val
            games = gp if gp > 0 else (wins + losses)
            ppg = p_avg if p_avg > 0 else (p_for / games if games > 0 else 0.0)
            oppg = d_avg if d_avg > 0 else (p_against / games if games > 0 else 0.0)
            win_pct = (wins / games) if games > 0 else 0.5

            stats.append({
                "team_norm": team_norm,
                "league_key": "NCAAF",
                "win_pct": win_pct,
                "home_win_pct": win_pct, # Approx
                "away_win_pct": win_pct, # Approx
                "points_per_game": ppg,
                "points_allowed_per_game": oppg,
                "turnovers": 0.0, # Not available in standings summary
                "streak": 0.0,
                "last5_win_pct": win_pct,
                "source": "ESPN_FALLBACK"
            })

        logger.info(f"Successfully fetched NCAAF stats from ESPN for {len(stats)} teams.")
        return stats

    except Exception as e:
        logger.error(f"ESPN NCAAF fallback failed: {e}")
        return []

def fetch_from_espn_ncaab(season_year: int) -> List[Dict[str, Any]]:
    """
    Fetch NCAAB stats from ESPN hidden API (Fallback).
    """
    try:
        logger.info(f"Fetching NCAAB stats from ESPN for season: {season_year}")
        url = "https://site.api.espn.com/apis/v2/sports/basketball/mens-college-basketball/standings"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        stats = []

        all_entries = find_entries(data)

        for entry in all_entries:
            team_info = entry.get("team", {})
            team_name = team_info.get("displayName")
            if not team_name:
                continue
            team_norm = normalize_team_for_stats(team_name, "NCAAB")

            # Resilient stat extraction: Search keywords in both name and abbreviation
            wins, losses, gp, p_for, p_against = 0.0, 0.0, 0.0, 0.0, 0.0
            p_avg, d_avg = 0.0, 0.0
            for s in entry.get("stats", []):
                nm, ab = str(s.get("name", "")).lower(), str(s.get("abbreviation", "")).lower()
                val = float(s.get("value", 0))
                if "wins" in nm or ab == "w": wins = val
                elif "losses" in nm or ab == "l": losses = val
                elif "gamesplayed" in nm or ab == "gp": gp = val
                elif any(x in nm for x in ["pointsagainst", "pointsallowed", "runsagainst"]) or ab in ["ra", "pa"]: p_against = val
                elif any(x in nm for x in ["pointsfor", "points", "runs"]) or ab in ["r", "pf"]: p_for = val
                elif nm in ["avg", "ppg", "apf"] or ab == "apf": p_avg = val
                elif nm in ["apa", "opp ppg"] or ab == "apa": d_avg = val
            games = gp if gp > 0 else (wins + losses)
            ppg = p_avg if p_avg > 0 else (p_for / games if games > 0 else 0.0)
            oppg = d_avg if d_avg > 0 else (p_against / games if games > 0 else 0.0)
            win_pct = (wins / games) if games > 0 else 0.5

            stats.append({
                "team_norm": team_norm,
                "league_key": "NCAAB",
                "win_pct": win_pct,
                "home_win_pct": win_pct, # Approx
                "away_win_pct": win_pct, # Approx
                "points_per_game": ppg,
                "points_allowed_per_game": oppg,
                "turnovers": 0.0, # Not in basic standings
                "streak": 0.0,
                "last5_win_pct": win_pct, # Approx
                "source": "ESPN_FALLBACK"
            })

        logger.info(f"Successfully fetched NCAAB stats from ESPN for {len(stats)} teams.")
        return stats

    except Exception as e:
        logger.error(f"ESPN NCAAB fallback failed: {e}")
        return []


@st.cache_data(ttl=21600)
def _fetch_mlb_standings_supplement(season_year: int) -> dict:
    """
    Fetch MLB streak and last-10 win% from MLB Stats API.
    Returns dict keyed by normalized team name -> {streak, last10_win_pct}.
    """
    supplement: dict = {}
    try:
        url = (
            f"https://statsapi.mlb.com/api/v1/standings"
            f"?leagueId=103,104&season={season_year}&hydrate=streak,records,team"
        )
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        for division in data.get("records", []):
            for tr in division.get("teamRecords", []):
                name = tr.get("team", {}).get("name", "")
                if not name:
                    continue
                team_norm = normalize_team_for_stats(name, "MLB")
                # Streak
                streak_code = tr.get("streak", {}).get("streakCode", "")
                streak = 0.0
                if streak_code:
                    try:
                        val = int(streak_code[1:])
                        streak = float(val) if streak_code[0].upper() == "W" else float(-val)
                    except (ValueError, IndexError):
                        pass
                # Last-10 win%
                last10_win_pct = None
                for split in tr.get("records", {}).get("splitRecords", []):
                    if split.get("type") == "lastTen":
                        w10 = split.get("wins", 0)
                        l10 = split.get("losses", 0)
                        total10 = w10 + l10
                        last10_win_pct = w10 / total10 if total10 > 0 else None
                        break
                supplement[team_norm] = {
                    "streak": streak,
                    "last10_win_pct": last10_win_pct,
                }
        logger.info(f"MLB Stats API supplement fetched for {len(supplement)} teams.")
    except Exception as e:
        logger.warning(f"MLB Stats API supplement fetch failed: {e}")
    return supplement


@st.cache_data(ttl=21600)
def fetch_from_espn_mlb(season_year: int) -> List[Dict[str, Any]]:
    """Fetch MLB stats from ESPN API, supplemented with streak + last-10 from MLB Stats API."""
    # Fetch streak and last-10 supplement first (best-effort)
    supplement = _fetch_mlb_standings_supplement(season_year)

    try:
        logger.info(f"Fetching MLB stats from ESPN for season: {season_year}")
        url = "https://site.api.espn.com/apis/v2/sports/baseball/mlb/standings"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        stats = []
        all_entries = find_entries(data)

        for entry in all_entries:
            team_info = entry.get("team", {})
            team_name = team_info.get("displayName")
            if not team_name: continue
            team_norm = normalize_team_for_stats(team_name, "MLB")

            # Resilient stat extraction: Search keywords in both name and abbreviation
            wins, losses, gp, p_for, p_against = 0.0, 0.0, 0.0, 0.0, 0.0
            p_avg, d_avg = 0.0, 0.0
            for s in entry.get("stats", []):
                nm, ab = str(s.get("name", "")).lower(), str(s.get("abbreviation", "")).lower()
                val = float(s.get("value", 0))
                if "wins" in nm or ab == "w": wins = val
                elif "losses" in nm or ab == "l": losses = val
                elif "gamesplayed" in nm or ab == "gp": gp = val
                elif any(x in nm for x in ["pointsagainst", "pointsallowed", "runsagainst"]) or ab in ["ra", "pa"]: p_against = val
                elif any(x in nm for x in ["pointsfor", "points", "runs"]) or ab in ["r", "pf"]: p_for = val
                elif nm in ["avg", "ppg", "apf"] or ab == "apf": p_avg = val
                elif nm in ["apa", "opp ppg"] or ab == "apa": d_avg = val
            games = gp if gp > 0 else (wins + losses)
            ppg = p_avg if p_avg > 0 else (p_for / games if games > 0 else 0.0)
            oppg = d_avg if d_avg > 0 else (p_against / games if games > 0 else 0.0)
            win_pct = (wins / games) if games > 0 else 0.5

            supp = supplement.get(team_norm, {})
            streak = supp.get("streak", 0.0)
            last5_win_pct = supp.get("last10_win_pct") or win_pct

            stats.append({
                "team_norm": team_norm,
                "stats_team_key": team_norm.strip().lower(),
                "team_name_source": team_name,
                "league_key": "MLB",
                "win_pct": win_pct,
                "home_win_pct": win_pct,
                "away_win_pct": win_pct,
                "points_per_game": ppg,
                "points_allowed_per_game": oppg,
                "turnovers": 0.0,
                "streak": streak,
                "last5_win_pct": last5_win_pct,
                "source": "ESPN"
            })

        logger.info(f"Successfully fetched MLB stats for {len(stats)} teams.")
        return stats
    except Exception as e:
        logger.error(f"ESPN MLB fetch failed: {e}")
        return []

@st.cache_data(ttl=21600)
def fetch_ncaab_stats(season_year: int) -> List[Dict[str, Any]]:
    """
    Fetch NCAAB stats exclusively using the ESPN Hidden API.
    (CBBpy has been removed due to broken API and performance timeouts).
    """
    logger.info(f"Fetching primary NCAAB stats from ESPN for season: {season_year}")
    espn_stats = fetch_from_espn_ncaab(season_year)

    # We strip 'ESPN_FALLBACK' to ensure it's treated as REAL data
    for s in espn_stats:
        if s.get("source") == "ESPN_FALLBACK":
            s["source"] = "REAL_ESPN"

    logger.info(f"Final NCAAB stats count (ESPN Primary): {len(espn_stats)}")
    return espn_stats

# -------------------------------------------------------------------------

def fetch_team_stats(api_clients: Dict[str, Any], season_year: Optional[int] = None) -> pd.DataFrame:
    """
    Refactored function to fetch stats for all configured leagues using
    specific open-source libraries where requested.

    Replaces old fetch_and_process_standings.
    """
    all_stats = []

    # Determine season year if not passed
    if not season_year:
        now = datetime.now()
        season_year = now.year
        # Simple heuristic: if before August, assume we want previous season starts (e.g. Jan 2025 -> 2024 season)
        if now.month < 8:
            season_year -= 1

    # Get list of leagues we care about from keys
    # leagues = list(api_clients.keys()) # unused, intentionally unconditional fetching

    # Dispatch Logic
    # Unconditionally fetch all leagues to ensure caches are populated
    all_stats.extend(fetch_nba_stats(season_year))
    all_stats.extend(fetch_nfl_stats(season_year))
    all_stats.extend(fetch_ncaaf_stats(season_year))
    all_stats.extend(fetch_nhl_stats(season_year))
    all_stats.extend(fetch_ncaab_stats(season_year))
    all_stats.extend(fetch_from_espn_mlb(season_year))

    # API-Sports fallback logic REMOVED as per instruction to "replace" logic.
    # If the user wants to keep API-Sports as a fallback for other leagues not listed,
    # we can add it back, but the prompt said "replace broken API-Sports logic... with specific free libraries".
    # We will assume these 5 are the core focus. If api_clients has others (MLB?), they get nothing for now.

    return pd.DataFrame(all_stats)

def enrich_with_model_features(df: pd.DataFrame, api_clients: Dict[str, Any], season_year: Optional[int] = None) -> pd.DataFrame:
    """
    Enrich the master dataframe with features required for Vertex AI.
    Uses pd.concat for performance to avoid fragmentation.
    Ensures ALL columns in VERTEX_FEATURE_COLUMNS are present in output.
    """
    global _FALLBACK_LOG_COUNT

    if df is None or df.empty:
        return df
        
    def safe_numeric_fill(val, fill_val=0.0):
        """Safely converts scalars or Series to numeric and fills NaNs."""
        try:
            if isinstance(val, (pd.Series, pd.Index)):
                return pd.to_numeric(val, errors='coerce').fillna(fill_val).infer_objects(copy=False)
            if isinstance(val, (list, tuple, np.ndarray)):
                return pd.to_numeric(pd.Series(val), errors='coerce').fillna(fill_val).infer_objects(copy=False)
            parsed = pd.to_numeric(val, errors='coerce')
            return fill_val if pd.isna(parsed) else parsed
        except Exception:
            return fill_val

    # ------------------------------------------------------------
    # INIT: feature container (MUST exist even if stats_df is empty)
    # ------------------------------------------------------------
    features_data: Dict[str, Any] = {}

    # ------------------------------------------------------------
    # 0) Determine league column (prefer sport_title, then league)
    # ------------------------------------------------------------
    league_col = None
    if "sport_title" in df.columns:
        league_col = "sport_title"
    elif "league" in df.columns:
        league_col = "league"
    elif "League" in df.columns:
        df["league"] = df["League"]
        league_col = "league"

    # 1) Identify home/away columns early (used in logging + normalization)
    home_col = "Home" if "Home" in df.columns else "home_team"
    away_col = "Away" if "Away" in df.columns else "away_team"

    if home_col not in df.columns or away_col not in df.columns:
        logger.error(f"Missing home/away columns in dataframe. Columns: {list(df.columns)}")
        return df

    # ------------------------------------------------------------
    # 2) Row-by-row league_key inference (single correct definition)
    # ------------------------------------------------------------
    # Explicit assignment to local variable
    if league_col:
        league_keys = df[league_col].apply(_model_league_key)
    else:
        league_keys = pd.Series(["default"] * len(df), index=df.index)

    # Make sure we always have a stable League column for downstream use
    # OPTIMIZATION: Add to features_data instead of direct assignment to avoid fragmentation
    # features_data is initialized at start of function
    features_data["League"] = league_keys

    # Explicitly set the OHE league columns
    for lg in ["NBA", "WNBA", "NFL", "NHL", "NCAAB", "NCAAF", "MLB"]:
        col = f"feature_league_{lg}"
        if col in VERTEX_FEATURE_COLUMNS:
            features_data[col] = (league_keys == lg).astype(float)

    # Defaults as Series aligned to df.index
    default_win_pct = pd.Series([LEAGUE_AVERAGES.get(lg, LEAGUE_AVERAGES["default"])["win_pct"] for lg in league_keys], index=df.index)
    default_last5   = pd.Series([LEAGUE_AVERAGES.get(lg, LEAGUE_AVERAGES["default"])["last5_win_pct"] for lg in league_keys], index=df.index)
    default_ppg     = pd.Series([LEAGUE_AVERAGES.get(lg, LEAGUE_AVERAGES["default"])["ppg"] for lg in league_keys], index=df.index)
    default_oppg    = pd.Series([LEAGUE_AVERAGES.get(lg, LEAGUE_AVERAGES["default"])["oppg"] for lg in league_keys], index=df.index)

    # ------------------------------------------------------------
    # 3) Fetch stats AFTER league_keys exists (ok if empty)
    # ------------------------------------------------------------
    # Tier 1: Attempt Live Stats Fetching
    stats_df = fetch_team_stats(api_clients, season_year=season_year)

    # Tier 2: If live stats are missing or incomplete, attempt season averages fallback
    used_historical_stats = False
    if stats_df is None or stats_df.empty:
        if season_year is not None:
            logger.warning(f"Live stats mostly empty for {season_year}. Attempting fallback to previous season averages.")
            fallback_stats = fetch_team_stats(api_clients, season_year=season_year - 1)
            if fallback_stats is not None and not fallback_stats.empty:
                stats_df = fallback_stats
                used_historical_stats = True

    # Tier 3: LEAGUE_AVERAGES are applied dynamically via default Series later

    # ------------------------------------------------------------
    # 4) Normalize team names league-aware
    # ------------------------------------------------------------
    # Explicit definition of locals for lambda safety
    _hc = home_col
    _ac = away_col
    _lk = league_keys

    home_norm = []
    away_norm = []
    
    for idx, row in df.iterrows():
        lk = _lk.at[idx]
        h_name = str(row[_hc])
        a_name = str(row[_ac])

        # Disambiguate NHL teams first using match context
        if lk == "NHL":
            h_name = resolve_nhl_ambiguity(h_name, a_name)
            a_name = resolve_nhl_ambiguity(a_name, h_name)

        h_norm_str = normalize_team_for_stats(h_name, lk)
        a_norm_str = normalize_team_for_stats(a_name, lk)

        home_norm.append(h_norm_str)
        away_norm.append(a_norm_str)

    home_norm = pd.Series(home_norm, index=df.index)
    away_norm = pd.Series(away_norm, index=df.index)
    
    # ------------------------------------------------------------
    # 5) Fuzzy Matching and Feature Population
    # ------------------------------------------------------------
    home_matched_names = pd.Series([None] * len(df), index=df.index)
    away_matched_names = pd.Series([None] * len(df), index=df.index)
    home_resolution_stage = pd.Series(["resolved"] * len(df), index=df.index, dtype="object")
    away_resolution_stage = pd.Series(["resolved"] * len(df), index=df.index, dtype="object")
    global_stats_lookup = {}
    resolution_stage_counts = Counter()
    unresolved_after_canonicalization = 0
    unresolved_after_alias = 0
    unresolved_after_fuzzy = 0
    stats_match_method_counts = Counter()
    stats_binding_failures_by_league: Counter[str] = Counter()

    if not stats_df.empty:
        stats_df = canonicalize_stats_team_index(stats_df)
        # Group stats_df by league_key
        stats_by_league = {}
        for lg in stats_df['league_key'].unique():
            subset = stats_df[stats_df['league_key'] == lg].copy()
            subset = subset[subset["stats_team_key"] != ""]
            stats_by_league[lg] = subset.drop_duplicates(subset=["stats_team_key"]).set_index("stats_team_key")

        # Build global lookup dict: (league, team_norm) -> value
        for lg, s_df in stats_by_league.items():
            d = s_df.to_dict(orient='index')
            for t_norm, cols in d.items():
                global_stats_lookup[(lg, t_norm)] = cols

        unique_leagues_in_games = league_keys.unique()

        # LOGGING REQUIREMENT: Track NCAAB match stats
        ncaab_match_stats = {"total": 0, "matched": 0, "fallback": 0, "unmatched_teams": []}

        for lg_key in unique_leagues_in_games:
            if lg_key not in stats_by_league:
                continue

            lg_mask = league_keys == lg_key
            if not lg_mask.any(): continue

            stats_subset = stats_by_league[lg_key]
            # Build strict league-local stats indexes (no cross-league alias bleed).
            stats_index_norm_map = {str(raw).strip().lower(): str(raw).strip().lower() for raw in stats_subset.index}
            alias_maps = _build_stats_index_maps(stats_subset, lg_key)

            if lg_key == "NFL":
                logger.debug(f"NFL stats index keys (sample): {list(stats_index_norm_map.keys())[:5]}")

            # Track match statistics
            stats_log = {"direct": 0, "canonical_alias": 0, "city_alias": 0, "fuzzy": 0, "miss": 0}

            # Process Home Teams
            current_home_teams = home_norm[lg_mask].unique()
            home_map_local = {}
            home_stage_map_local = {}
            for t_norm in current_home_teams:
                if not t_norm: continue

                matched_name, reason, failure_stage = resolve_stats_team_match(
                    t_norm,
                    lg_key,
                    stats_index_norm_map,
                    alias_maps["canonical"],
                    alias_maps["city_only"],
                )
                home_map_local[t_norm] = matched_name
                home_stage_map_local[t_norm] = failure_stage
                if matched_name is None:
                    stats_log["miss"] += 1
                    stats_binding_failures_by_league[lg_key] += 1
                    resolution_stage_counts[failure_stage] += 1
                    if failure_stage != "before_canonicalization":
                        unresolved_after_canonicalization += 1
                    if failure_stage in {"after_alias", "after_fuzzy"}:
                        unresolved_after_alias += 1
                    if failure_stage == "after_fuzzy":
                        unresolved_after_fuzzy += 1
                elif reason in stats_log:
                    stats_log[reason] += 1
                    stats_match_method_counts[reason] += 1
                else:
                    stats_log["direct"] += 1
                    stats_match_method_counts["direct"] += 1

            home_matched_names[lg_mask] = home_norm[lg_mask].map(home_map_local)
            home_resolution_stage[lg_mask] = home_norm[lg_mask].map(home_stage_map_local).fillna("resolved")

            # Process Away Teams
            current_away_teams = away_norm[lg_mask].unique()
            away_map_local = {}
            away_stage_map_local = {}
            for t_norm in current_away_teams:
                if not t_norm: continue

                matched_name, reason, failure_stage = resolve_stats_team_match(
                    t_norm,
                    lg_key,
                    stats_index_norm_map,
                    alias_maps["canonical"],
                    alias_maps["city_only"],
                )
                away_map_local[t_norm] = matched_name
                away_stage_map_local[t_norm] = failure_stage
                if matched_name is None:
                    stats_log["miss"] += 1
                    stats_binding_failures_by_league[lg_key] += 1
                    resolution_stage_counts[failure_stage] += 1
                    if failure_stage != "before_canonicalization":
                        unresolved_after_canonicalization += 1
                    if failure_stage in {"after_alias", "after_fuzzy"}:
                        unresolved_after_alias += 1
                    if failure_stage == "after_fuzzy":
                        unresolved_after_fuzzy += 1
                elif reason in stats_log:
                    stats_log[reason] += 1
                    stats_match_method_counts[reason] += 1
                else:
                    stats_log["direct"] += 1
                    stats_match_method_counts["direct"] += 1

            away_matched_names[lg_mask] = away_norm[lg_mask].map(away_map_local)
            away_resolution_stage[lg_mask] = away_norm[lg_mask].map(away_stage_map_local).fillna("resolved")

            # Updated Logging
            log_stats_match_summary(stats_log, lg_key)

            if stats_log["miss"] > 0 and lg_key != "default":
                # Find which teams missed
                missing_home = [t for t, m in home_map_local.items() if m is None]
                missing_away = [t for t, m in away_map_local.items() if m is None]
                missing = list(set(missing_home + missing_away))
                if missing:
                    logger.warning(f"STATS MISSING TEAMS [{lg_key}] (Top 20): {missing[:20]}")
                    # Issue #2: Explicitly log all missing teams for ticket creation
                    logger.info(f"Missing stats teams: {missing}")
                    if lg_key == "NCAAB":
                        logger.info("Action: Create ticket 'Investigate NCAAB stats coverage for low-tier conferences'")
                        ncaab_match_stats["unmatched_teams"].extend(missing)

            # Explicit match rate log as requested
            total_matches = (
                stats_log.get('direct', 0)
                + stats_log.get('canonical_alias', 0)
                + stats_log.get('city_alias', 0)
                + stats_log.get('fuzzy', 0)
            )
            total_attempts = total_matches + stats_log.get('miss', 0)
            if total_attempts > 0:
                match_pct = (total_matches / total_attempts) * 100
                logger.info(f"--- {lg_key} STATS MATCHING SUMMARY ---")
                logger.info(f"- Total Lookups: {total_attempts}")
                logger.info(f"- Live Stats Success: {total_matches} ({match_pct:.1f}%)")
                logger.info(f"- Fallback Count: {stats_log.get('miss', 0)}")
                if stats_log.get('miss', 0) > 0:
                    missing_home = [t for t, m in home_map_local.items() if m is None]
                    missing_away = [t for t, m in away_map_local.items() if m is None]
                    missing = list(set(missing_home + missing_away))
                    logger.info(f"- Top Failed Lookups: {missing[:10]}")
            stats_match_method_counts["unresolved"] += int(stats_log.get("miss", 0))

            if lg_key == "NCAAB":
                ncaab_match_stats["total"] = total_attempts // 2 # Approximate games from teams
                ncaab_match_stats["matched"] = total_matches // 2
                ncaab_match_stats["fallback"] = stats_log.get('miss', 0) // 2

        # Log NCAAB Stats Matching Summary (Issue #2 requirement)
        if ncaab_match_stats["total"] > 0:
            match_pct = (ncaab_match_stats["matched"] / ncaab_match_stats["total"]) * 100 if ncaab_match_stats["total"] else 0
            fallback_pct = (ncaab_match_stats["fallback"] / ncaab_match_stats["total"]) * 100 if ncaab_match_stats["total"] else 0
            logger.info("NCAAB STATS MATCHING SUMMARY:")
            logger.info(f"- Total NCAAB games (approx): {ncaab_match_stats['total']}")
            logger.info(f"- Successfully matched: {ncaab_match_stats['matched']} ({match_pct:.1f}%)")
            logger.info(f"- Fallback used: {ncaab_match_stats['fallback']} ({fallback_pct:.1f}%)")
            if ncaab_match_stats["unmatched_teams"]:
                logger.info("- Top unmatched teams:")
                for i, t in enumerate(ncaab_match_stats["unmatched_teams"][:10]):
                    logger.info(f"  {i+1}. \"{t}\"")

    else:
        if not FREE_TIER_MODE:
            logger.warning("No stats fetched. Filling with defaults.")
        home_resolution_stage[:] = "stats_index_lookup"
        away_resolution_stage[:] = "stats_index_lookup"

    # Helper function moved to top level logic within enrich_with_model_features
    # Now explicitly uses arguments instead of closures where possible
    def _map_stat_impl(matched_name_series, col_name, default_series, l_keys, g_lookup, df_idx):
        values = []
        for idx, name in matched_name_series.items():
            lg = l_keys.at[idx]
            if pd.notna(name) and (lg, name) in g_lookup:
                val = g_lookup[(lg, name)].get(col_name)
                values.append(val if val is not None else np.nan)
            else:
                values.append(np.nan)
        return pd.Series(values, index=df_idx).fillna(default_series).infer_objects(copy=False)

    # Populate features_data
    home_fallback = home_matched_names.isna()
    away_fallback = away_matched_names.isna()
    combined_fallback = home_fallback | away_fallback

    # Seed feature_stats_fallback from team-level resolver misses.
    # This can be expanded later for source-level failures (e.g., NBA fetch failure).
    features_data["feature_stats_fallback"] = combined_fallback

    # Task 2: Standardize stats_quality values (REAL/HISTORICAL/FALLBACK/MISSING)
    # Optimized using np.select for vectorization
    conditions = [
        (home_fallback & away_fallback),  # Both missing
        (home_fallback | away_fallback)   # At least one missing (but not both, covered above)
    ]
    choices = ["MISSING", "FALLBACK"]

    # If used_historical_stats is globally true, any matched row technically used historical stats
    default_quality = "HISTORICAL" if used_historical_stats else "REAL"

    # Generate stats_quality array
    quality_arr = np.select(conditions, choices, default=default_quality)

    # Assign as Series with correct index
    quality_series = pd.Series(quality_arr, index=df.index)

    # Task: Identify ESPN source to distinguish from REAL (CFBD/Native)
    default_source = pd.Series([""] * len(df), index=df.index)

    # Re-use _map_stat_impl to pull 'source' field
    home_source = _map_stat_impl(home_matched_names, "source", default_source, league_keys, global_stats_lookup, df.index)
    away_source = _map_stat_impl(away_matched_names, "source", default_source, league_keys, global_stats_lookup, df.index)

    # If source is ESPN_FALLBACK, mark as ESPN (only if currently REAL)
    is_espn = (home_source == "ESPN_FALLBACK") | (away_source == "ESPN_FALLBACK")

    # Update only if currently REAL or HISTORICAL (don't overwrite MISSING/FALLBACK)
    mask_update_espn = is_espn & (quality_series.isin(["REAL", "HISTORICAL"]))
    quality_series[mask_update_espn] = "ESPN"

    features_data["stats_quality"] = quality_series

    # is_live_data strictly evaluates to True only if stats_quality is REAL or ESPN
    features_data["is_live_data"] = quality_series.isin(["REAL", "ESPN"])

    nba_fetch_diag = get_nba_fetch_diagnostics()
    stats_source = pd.Series(["live"] * len(df), index=df.index, dtype="object")
    stats_resolution_status = pd.Series(["resolved"] * len(df), index=df.index, dtype="object")
    stats_fallback_reason = pd.Series([""] * len(df), index=df.index, dtype="object")
    stats_resolution_stage_failure = pd.Series([""] * len(df), index=df.index, dtype="object")

    unresolved_mask = combined_fallback.copy()
    stage_failure_values = []
    for idx in df.index:
        failed_stages = []
        if bool(home_fallback.loc[idx]):
            failed_stages.append(str(home_resolution_stage.loc[idx]))
        if bool(away_fallback.loc[idx]):
            failed_stages.append(str(away_resolution_stage.loc[idx]))
        unique_stages = [s for s in dict.fromkeys(failed_stages) if s and s != "resolved"]
        stage_failure_values.append("|".join(unique_stages))
    stats_resolution_stage_failure = pd.Series(stage_failure_values, index=df.index, dtype="object")

    stats_resolution_status.loc[unresolved_mask] = "unresolved"
    stats_source.loc[unresolved_mask] = "fallback"
    stats_fallback_reason.loc[unresolved_mask] = "team_mapping_unresolved"

    nba_mask = league_keys == "NBA"
    if nba_mask.any():
        nba_resolved_rows = (nba_mask & ~combined_fallback).sum()
        nba_stats_available = False
        if not stats_df.empty and "league_key" in stats_df.columns:
            nba_stats_available = stats_df["league_key"].astype(str).str.upper().eq("NBA").any()
        if str(nba_fetch_diag.get("source", "")).lower() in {"runtime_cache", "disk_cache"}:
            stats_source.loc[nba_mask & ~unresolved_mask] = "cached"
        elif nba_fetch_diag.get("status") == "failed" and nba_resolved_rows == 0 and not nba_stats_available:
            stats_source.loc[nba_mask] = "failed"
            stats_resolution_status.loc[nba_mask] = "unresolved"
            stats_fallback_reason.loc[nba_mask] = "nba_stats_fetch_failed"
            stats_resolution_stage_failure.loc[nba_mask] = "nba_stats_fetch_failed"
            unresolved_mask = unresolved_mask | nba_mask

    features_data["feature_stats_fallback"] = unresolved_mask
    features_data["stats_source"] = stats_source
    features_data["stats_resolution_status"] = stats_resolution_status
    features_data["stats_fallback_reason"] = stats_fallback_reason
    features_data["stats_resolution_stage_failure"] = stats_resolution_stage_failure
    features_data["stats_fetch_retries_used"] = nba_fetch_diag.get("retries_used", 0)
    nba_fetch_source_raw = str(nba_fetch_diag.get("source", "none")).lower()
    nba_row_sources = set(stats_source.loc[nba_mask].astype(str).str.lower().tolist()) if nba_mask.any() else set()
    nba_in_slate = bool(nba_mask.any())
    if not nba_in_slate:
        nba_fetch_status = "not_applicable"
        nba_fetch_source = "not_applicable"
    elif "live" in nba_row_sources:
        nba_fetch_status = "live"
        nba_fetch_source = "live"
    elif "cached" in nba_row_sources:
        nba_fetch_status = "cached"
        if nba_fetch_source_raw in {"runtime_cache", "disk_cache"}:
            nba_fetch_source = nba_fetch_source_raw
        else:
            nba_fetch_source = "cached"
    elif "failed" in nba_row_sources:
        nba_fetch_status = "failed"
        nba_fetch_source = "failed"
    elif nba_fetch_source_raw == "live":
        nba_fetch_status = "live"
        nba_fetch_source = "live"
    elif nba_fetch_source_raw in {"runtime_cache", "disk_cache", "cached"}:
        nba_fetch_status = "cached"
        nba_fetch_source = nba_fetch_source_raw if nba_fetch_source_raw in {"runtime_cache", "disk_cache"} else "cached"
    elif nba_fetch_source_raw == "failed":
        nba_fetch_status = "failed"
        nba_fetch_source = "failed"
    else:
        nba_fetch_status = "not_started"
        nba_fetch_source = "none"
    nba_status_by_row = pd.Series(
        ["not_applicable"] * len(df), index=df.index, dtype="object"
    )
    nba_source_by_row = pd.Series(
        ["not_applicable"] * len(df), index=df.index, dtype="object"
    )
    nba_status_by_row.loc[nba_mask] = nba_fetch_status
    nba_source_by_row.loc[nba_mask] = nba_fetch_source
    features_data["nba_stats_fetch_status"] = nba_status_by_row
    features_data["nba_stats_fetch_source"] = nba_source_by_row
    features_data["nba_stats_fetch_retries_used"] = int(nba_fetch_diag.get("retries_used", 0))
    features_data["ml_feature_eligible"] = stats_resolution_status == "resolved"

    # Mapping calls
    # Note: passing league_keys and global_stats_lookup explicitly
    features_data["feature_home_win_pct"] = _map_stat_impl(home_matched_names, "win_pct", default_win_pct, league_keys, global_stats_lookup, df.index)
    features_data["feature_home_home_win_pct"] = _map_stat_impl(home_matched_names, "home_win_pct", default_win_pct, league_keys, global_stats_lookup, df.index)
    features_data["feature_home_last5_win_pct"] = _map_stat_impl(home_matched_names, "last5_win_pct", default_last5, league_keys, global_stats_lookup, df.index)
    features_data["feature_home_ppg"] = _map_stat_impl(home_matched_names, "points_per_game", default_ppg, league_keys, global_stats_lookup, df.index)
    features_data["feature_home_oppg"] = _map_stat_impl(home_matched_names, "points_allowed_per_game", default_oppg, league_keys, global_stats_lookup, df.index)
    features_data["feature_home_streak"] = _map_stat_impl(home_matched_names, "streak", pd.Series(0.0, index=df.index), league_keys, global_stats_lookup, df.index)
    features_data["feature_home_turnovers"] = _map_stat_impl(home_matched_names, "turnovers", pd.Series(0.0, index=df.index), league_keys, global_stats_lookup, df.index)

    features_data["feature_away_win_pct"] = _map_stat_impl(away_matched_names, "win_pct", default_win_pct, league_keys, global_stats_lookup, df.index)
    features_data["feature_away_away_win_pct"] = _map_stat_impl(away_matched_names, "away_win_pct", default_win_pct, league_keys, global_stats_lookup, df.index)
    features_data["feature_away_last5_win_pct"] = _map_stat_impl(away_matched_names, "last5_win_pct", default_last5, league_keys, global_stats_lookup, df.index)
    features_data["feature_away_ppg"] = _map_stat_impl(away_matched_names, "points_per_game", default_ppg, league_keys, global_stats_lookup, df.index)
    features_data["feature_away_oppg"] = _map_stat_impl(away_matched_names, "points_allowed_per_game", default_oppg, league_keys, global_stats_lookup, df.index)
    features_data["feature_away_streak"] = _map_stat_impl(away_matched_names, "streak", pd.Series(0.0, index=df.index), league_keys, global_stats_lookup, df.index)
    features_data["feature_away_turnovers"] = _map_stat_impl(away_matched_names, "turnovers", pd.Series(0.0, index=df.index), league_keys, global_stats_lookup, df.index)

    # SCALING: NHL stats are ~3.0, model expects ~110.0. Scale by 35x if league is NHL.
    is_nhl = league_keys == "NHL"
    if is_nhl.any():
        nhl_scale_factor = 35.0
        features_data['feature_home_ppg'] = features_data['feature_home_ppg'].mask(is_nhl, features_data['feature_home_ppg'] * nhl_scale_factor)
        features_data['feature_home_oppg'] = features_data['feature_home_oppg'].mask(is_nhl, features_data['feature_home_oppg'] * nhl_scale_factor)
        features_data['feature_away_ppg'] = features_data['feature_away_ppg'].mask(is_nhl, features_data['feature_away_ppg'] * nhl_scale_factor)
        features_data['feature_away_oppg'] = features_data['feature_away_oppg'].mask(is_nhl, features_data['feature_away_oppg'] * nhl_scale_factor)


    # SCALING: MLB stats are ~4.5, model expects ~110.0. Scale by 25x if league is MLB.
    is_mlb = league_keys == "MLB"
    if is_mlb.any():
        mlb_scale_factor = 25.0
        features_data['feature_home_ppg'] = features_data['feature_home_ppg'].mask(is_mlb, features_data['feature_home_ppg'] * mlb_scale_factor)
        features_data['feature_home_oppg'] = features_data['feature_home_oppg'].mask(is_mlb, features_data['feature_home_oppg'] * mlb_scale_factor)
        features_data['feature_away_ppg'] = features_data['feature_away_ppg'].mask(is_mlb, features_data['feature_away_ppg'] * mlb_scale_factor)
        features_data['feature_away_oppg'] = features_data['feature_away_oppg'].mask(is_mlb, features_data['feature_away_oppg'] * mlb_scale_factor)

    # SCALING: NCAAB (~72) and NCAAF (~28) to NBA (~110) magnitude
    is_ncaab = league_keys == "NCAAB"
    if is_ncaab.any():
        for col in ['feature_home_ppg', 'feature_home_oppg', 'feature_away_ppg', 'feature_away_oppg']:
            features_data[col] = features_data[col].mask(is_ncaab, features_data[col] * 1.55)

    is_ncaaf = league_keys == "NCAAF"
    if is_ncaaf.any():
        for col in ['feature_home_ppg', 'feature_home_oppg', 'feature_away_ppg', 'feature_away_oppg']:
            features_data[col] = features_data[col].mask(is_ncaaf, features_data[col] * 4.0)

    # 4. Fill Defaults if stats_df was empty or incomplete
    if 'feature_home_win_pct' not in features_data:
        features_data['feature_stats_fallback'] = True
        features_data['feature_home_win_pct'] = default_win_pct
        features_data['feature_home_home_win_pct'] = default_win_pct
        features_data['feature_home_last5_win_pct'] = default_last5
        features_data['feature_home_ppg'] = default_ppg
        features_data['feature_home_oppg'] = default_oppg
        features_data['feature_home_streak'] = 0.0
        features_data['feature_home_turnovers'] = 0.0
        
        features_data['feature_away_win_pct'] = default_win_pct
        features_data['feature_away_away_win_pct'] = default_win_pct
        features_data['feature_away_last5_win_pct'] = default_last5
        features_data['feature_away_ppg'] = default_ppg
        features_data['feature_away_oppg'] = default_oppg
        features_data['feature_away_streak'] = 0.0
        features_data['feature_away_turnovers'] = 0.0
        
        if not FREE_TIER_MODE:
            logger.warning("Used fallback league averages for ALL games (stats fetch failed)!")

    # 5. Compute Differentials
    # Logic: Only calculate diff if BOTH teams have non-zero data
    def safe_diff(h, a):
        """
        Calculate difference between home and away values.
        Returns 0.0 only if both are zero/missing or if conversion fails.
        Legitimate zero values (e.g., 0% win rate) are valid for calculation.
        """
        try:
            h_val = float(h)
            a_val = float(a)
        except Exception:
            return 0.0

        # Check for NaN (pandas NA values)
        if pd.isna(h_val) or pd.isna(a_val):
            return 0.0

        # Only return 0.0 if BOTH are exactly zero (indicates missing data for both)
        if h_val == 0.0 and a_val == 0.0:
            return 0.0

        # Otherwise calculate legitimate differential (includes cases where one is 0.0)
        return h_val - a_val

    # Helper to ensure we have iterables
    def to_iterable(val, length):
        if isinstance(val, (list, pd.Series, np.ndarray)):
            return val
        return [val] * length

    df_len = len(df)

    # Use list comprehension for explicit row-by-row processing
    features_data['feature_diff_win_pct'] = [safe_diff(h, a) for h, a in zip(to_iterable(features_data['feature_home_win_pct'], df_len), to_iterable(features_data['feature_away_win_pct'], df_len))]
    features_data['feature_diff_ppg'] = [safe_diff(h, a) for h, a in zip(to_iterable(features_data['feature_home_ppg'], df_len), to_iterable(features_data['feature_away_ppg'], df_len))]
    features_data['feature_diff_oppg'] = [safe_diff(h, a) for h, a in zip(to_iterable(features_data['feature_home_oppg'], df_len), to_iterable(features_data['feature_away_oppg'], df_len))]
    features_data['feature_diff_last5'] = [safe_diff(h, a) for h, a in zip(to_iterable(features_data['feature_home_last5_win_pct'], df_len), to_iterable(features_data['feature_away_last5_win_pct'], df_len))]

    # Streak
    s_home = safe_numeric_fill(features_data['feature_home_streak'], 0.0)
    s_away = safe_numeric_fill(features_data['feature_away_streak'], 0.0)

    # If scalars, manual subtract
    if not isinstance(s_home, (pd.Series, np.ndarray, list)) and not isinstance(s_away, (pd.Series, np.ndarray, list)):
        features_data['feature_diff_streak'] = s_home - s_away
    else:
        # If mixed scalar/series, pandas handles it usually, but let's be safe
        s_home_s = s_home if isinstance(s_home, (pd.Series)) else pd.Series([s_home]*df_len, index=df.index)
        s_away_s = s_away if isinstance(s_away, (pd.Series)) else pd.Series([s_away]*df_len, index=df.index)
        features_data['feature_diff_streak'] = s_home_s - s_away_s

    # Ensure unresolved stats rows cannot masquerade as fully enriched ML rows.
    unresolved_feature_cols = [
        "feature_home_win_pct",
        "feature_home_home_win_pct",
        "feature_home_last5_win_pct",
        "feature_home_ppg",
        "feature_home_oppg",
        "feature_home_streak",
        "feature_home_turnovers",
        "feature_away_win_pct",
        "feature_away_away_win_pct",
        "feature_away_last5_win_pct",
        "feature_away_ppg",
        "feature_away_oppg",
        "feature_away_streak",
        "feature_away_turnovers",
        "feature_diff_win_pct",
        "feature_diff_ppg",
        "feature_diff_oppg",
        "feature_diff_last5",
        "feature_diff_streak",
    ]
    for col in unresolved_feature_cols:
        if col in features_data:
            series = pd.Series(features_data[col], index=df.index)
            series.loc[unresolved_mask] = np.nan
            features_data[col] = series
    
    # 6. Map Remaining Features (Existing) using safe_numeric_fill
    
    # Aggregate fallback diagnostics to reduce log spam
    fallback_counts_by_league = league_keys[unresolved_mask].value_counts().to_dict()
    unresolved_counts_by_league = league_keys[stats_resolution_status == "unresolved"].value_counts().to_dict()
    source_counts = stats_source.value_counts(dropna=False).to_dict()
    source_counts = {
        "live": int(source_counts.get("live", 0)),
        "cached": int(source_counts.get("cached", 0)),
        "fallback": int(source_counts.get("fallback", 0)),
        "failed": int(source_counts.get("failed", 0)),
    }
    ml_excluded_by_league = league_keys[~features_data["ml_feature_eligible"]].value_counts().to_dict()
    features_data["stats_unresolved_count_by_league"] = int(sum(unresolved_counts_by_league.values()))
    features_data["stats_unresolved_count_by_league_detail"] = str(
        {str(k): int(v) for k, v in unresolved_counts_by_league.items()}
    )
    features_data["stats_ml_excluded_rows"] = int((~features_data["ml_feature_eligible"]).sum())
    features_data["stats_source_counts"] = str(source_counts)
    features_data["fallback_summary_by_league"] = league_keys.map(
        lambda league: str(
            {str(league): int(fallback_counts_by_league.get(league, 0))}
        )
        if int(fallback_counts_by_league.get(league, 0)) > 0
        else "{}"
    )
    features_data["stats_binding_failures_by_league"] = str(
        {str(k): int(v) for k, v in stats_binding_failures_by_league.items()}
    )
    features_data["stats_match_counts_by_method"] = str(
        {
            "direct": int(stats_match_method_counts.get("direct", 0)),
            "canonical_alias": int(stats_match_method_counts.get("canonical_alias", 0)),
            "city_alias": int(stats_match_method_counts.get("city_alias", 0)),
            "fuzzy": int(stats_match_method_counts.get("fuzzy", 0)),
            "unresolved": int(stats_match_method_counts.get("unresolved", 0)),
        }
    )
    features_data["unresolved_after_canonicalization"] = int(unresolved_after_canonicalization)
    features_data["unresolved_after_alias"] = int(unresolved_after_alias)
    features_data["unresolved_after_fuzzy"] = int(unresolved_after_fuzzy)
    features_data["stats_resolution_stage_failure_counts"] = str(dict(resolution_stage_counts))

    league_row_counts = league_keys.value_counts().to_dict()
    fallback_heavy_by_league = {
        league: (
            float(count) / float(max(int(league_row_counts.get(league, 0)), 1))
            >= 0.25
            or int(unresolved_counts_by_league.get(league, 0)) >= 3
        )
        for league, count in fallback_counts_by_league.items()
    }
    fallback_heavy = league_keys.map(
        lambda league: bool(fallback_heavy_by_league.get(league, False))
    )
    nba_failed = bool(nba_in_slate) and (
        str(nba_fetch_diag.get("source", "")).lower() == "failed"
    )

    def _league_health_warning(league: str) -> str:
        league_fallback_heavy = bool(
            fallback_heavy_by_league.get(league, False)
        )
        if league == "NBA" and nba_failed and league_fallback_heavy:
            return (
                "Run health warning: NBA stats fetch failed and fallback usage "
                "is elevated; card confidence may be reduced."
            )
        if league_fallback_heavy:
            return (
                f"Run health warning: {league} fallback usage is elevated; "
                "verify unresolved team bindings before trusting the card."
            )
        if league == "NBA" and nba_failed:
            return "Run health warning: NBA stats fetch failed for this run."
        return ""

    features_data["fallback_heavy_slate_flag"] = fallback_heavy.astype(bool)
    features_data["run_health_warning"] = league_keys.map(
        _league_health_warning
    )

    logger.info(f"STATS SOURCE COUNTS: {source_counts}")
    logger.info(f"STATS UNRESOLVED COUNT BY LEAGUE: {unresolved_counts_by_league}")
    logger.info(f"STATS BINDING FAILURES BY LEAGUE: {dict(stats_binding_failures_by_league)}")
    logger.info(f"STATS MATCH COUNTS BY METHOD: {dict(stats_match_method_counts)}")
    logger.info(
        "STATS RESOLUTION FAILURE COUNTS: "
        f"stage={dict(resolution_stage_counts)} "
        f"after_canonicalization={unresolved_after_canonicalization} "
        f"after_alias={unresolved_after_alias} "
        f"after_fuzzy={unresolved_after_fuzzy}"
    )
    logger.info(f"ROWS EXCLUDED FROM ML DUE TO UNRESOLVED STATS: {ml_excluded_by_league}")

    if fallback_counts_by_league:
        logger.warning(f"STATS FALLBACK SUMMARY BY LEAGUE: {fallback_counts_by_league}")

    if unresolved_mask.any():
        unresolved_pairs = Counter(
            [
                f"{league_keys.at[idx]} :: {df.loc[idx, home_col]} vs {df.loc[idx, away_col]}"
                for idx in df.index[unresolved_mask]
            ]
        )
        unresolved_teams = Counter()
        for idx in df.index[unresolved_mask]:
            if bool(home_fallback.loc[idx]):
                unresolved_teams[f"{league_keys.at[idx]}::{str(df.loc[idx, home_col])}"] += 1
            if bool(away_fallback.loc[idx]):
                unresolved_teams[f"{league_keys.at[idx]}::{str(df.loc[idx, away_col])}"] += 1
        logger.warning(f"UNRESOLVED STATS MATCHUPS: {dict(unresolved_pairs)}")
        logger.warning(f"UNRESOLVED STATS TEAM COUNTS: {dict(unresolved_teams)}")

    # Identify implied probability column
    imp_col = next((c for c in df.columns if str(c).lower() in ['implied_prob', 'implied_home_prob', 'implied_prob_home']), None)

    # Step 1: Extract real market-derived probabilities first (Highest Priority)
    prob_series = pd.Series([np.nan]*len(df), index=df.index)

    def _is_placeholder(val):
        if pd.isna(val):
            return True
        return abs(float(val) - 0.5) <= 1e-6

    # Track how we populated implied_home_prob
    implied_sources = {"existing_valid": 0, "computed_devig": 0, "side_specific": 0, "defaulted": 0}

    # 1. Existing de-vig/market prob
    if 'implied_home_prob' in df.columns:
        implied_probs = pd.to_numeric(df['implied_home_prob'], errors='coerce')
        valid_implied = implied_probs[(implied_probs.notna()) & (~implied_probs.apply(_is_placeholder))]
        prob_series = prob_series.fillna(valid_implied).infer_objects(copy=False)
        implied_sources["existing_valid"] += int(valid_implied.notna().sum())

    if 'market_probability' in df.columns and prob_series.isna().any():
        mkt_probs = pd.to_numeric(df['market_probability'], errors='coerce')
        # only keep if not 0.5 exactly (often a default placeholder)
        valid_mkt = mkt_probs[(mkt_probs.notna()) & (~mkt_probs.apply(_is_placeholder))]
        mask = prob_series.isna() & valid_mkt.notna()
        if mask.any():
             prob_series = prob_series.fillna(valid_mkt).infer_objects(copy=False)
             implied_sources["existing_valid"] += int(mask.sum())

    # 2. Compute from raw home/away odds if we have both sides
    # We prefer matching rows in the dataframe by matchup_id and market_type to find the opposing side odds.
    # First, try to compute from columns on the same row if they exist.
    h_odds_col = next((c for c in df.columns if str(c).lower() in ['novig_home_price', 'novig_h2h_home_price', 'home_odds', 'odds_home', 'home_price']), None)
    a_odds_col = next((c for c in df.columns if str(c).lower() in ['novig_away_price', 'novig_h2h_away_price', 'away_odds', 'odds_away', 'away_price']), None)

    if h_odds_col and a_odds_col and prob_series.isna().any():
        for idx in prob_series[prob_series.isna()].index:
            try:
                h_val = float(df.loc[idx, h_odds_col])
                a_val = float(df.loc[idx, a_odds_col])

                # Assume american odds if absolute value >= 100
                if abs(h_val) >= 100 and abs(a_val) >= 100:
                    h_prob = 100 / (h_val + 100) if h_val > 0 else abs(h_val) / (abs(h_val) + 100)
                    a_prob = 100 / (a_val + 100) if a_val > 0 else abs(a_val) / (abs(a_val) + 100)
                    # de-vig
                    if h_prob > 0 and a_prob > 0:
                        prob_series.at[idx] = h_prob / (h_prob + a_prob)
                        implied_sources["computed_devig"] += 1
                elif 1 < h_val < 100 and 1 < a_val < 100: # decimal odds
                    h_prob = 1.0 / h_val
                    a_prob = 1.0 / a_val
                    if h_prob > 0 and a_prob > 0:
                        prob_series.at[idx] = h_prob / (h_prob + a_prob)
                        implied_sources["computed_devig"] += 1
            except Exception:
                pass

    # 2b. Compute from paired rows (e.g. if the dataframe has separate rows for Home and Away lines for the same game)
    if 'matchup_id' in df.columns and 'market_type' in df.columns and 'odds_american' in df.columns and prob_series.isna().any():
        # Build a lookup of american odds by matchup_id and market_type
        # e.g., {'matchup1': {'h2h_home': -110, 'h2h_away': -110, ...}}
        odds_lookup = {}
        for idx, row in df.iterrows():
            m_id = row.get('matchup_id')
            m_type = str(row.get('market_type', '')).lower()
            odds = row.get('odds_american')
            if pd.notna(m_id) and pd.notna(m_type) and pd.notna(odds):
                if m_id not in odds_lookup:
                    odds_lookup[m_id] = {}
                odds_lookup[m_id][m_type] = odds

        for idx in prob_series[prob_series.isna()].index:
            m_id = df.loc[idx, 'matchup_id']
            m_type = str(df.loc[idx, 'market_type']).lower()
            if m_id in odds_lookup:
                # Determine opposing side market type
                opp_m_type = None
                if 'home' in m_type:
                    opp_m_type = m_type.replace('home', 'away')
                elif 'away' in m_type:
                    opp_m_type = m_type.replace('away', 'home')
                elif 'over' in m_type:
                    opp_m_type = m_type.replace('over', 'under')
                elif 'under' in m_type:
                    opp_m_type = m_type.replace('under', 'over')

                if opp_m_type and opp_m_type in odds_lookup[m_id] and m_type in odds_lookup[m_id]:
                    h_val = odds_lookup[m_id][m_type if 'home' in m_type or 'over' in m_type else opp_m_type]
                    a_val = odds_lookup[m_id][opp_m_type if 'home' in m_type or 'over' in m_type else m_type]

                    try:
                        h_val = float(h_val)
                        a_val = float(a_val)
                        if abs(h_val) >= 100 and abs(a_val) >= 100:
                            h_prob = 100 / (h_val + 100) if h_val > 0 else abs(h_val) / (abs(h_val) + 100)
                            a_prob = 100 / (a_val + 100) if a_val > 0 else abs(a_val) / (abs(a_val) + 100)
                            if h_prob > 0 and a_prob > 0:
                                prob_series.at[idx] = h_prob / (h_prob + a_prob)
                                implied_sources["computed_devig"] += 1
                    except Exception:
                        pass

    # 3. Use side-specific implied probability oriented to home
    if prob_series.isna().any():
        for idx in prob_series[prob_series.isna()].index:
            market_type = str(df.loc[idx, 'market_type']).lower() if 'market_type' in df.columns else ''

            # If we know it's a home-oriented row, use its implied prob directly
            is_home_side = market_type in ('spread_home', 'h2h_home', 'moneyline_home', 'home')
            is_away_side = market_type in ('spread_away', 'h2h_away', 'moneyline_away', 'away')

            val = None
            if imp_col:
                val = pd.to_numeric(df.loc[idx, imp_col], errors='coerce')

            if pd.isna(val) and 'decimal_odds' in df.columns:
                dec = pd.to_numeric(df.loc[idx, 'decimal_odds'], errors='coerce')
                if pd.notna(dec) and dec > 1.0:
                    val = 1.0 / dec

            if pd.isna(val) and 'odds_american' in df.columns:
                am = df.loc[idx, 'odds_american']
                val = ml_to_prob(am)

            if pd.notna(val) and 0.0 < val < 1.0 and not _is_placeholder(val):
                if is_home_side:
                    prob_series.at[idx] = val
                    implied_sources["side_specific"] += 1
                elif is_away_side and market_type in ('spread_away', 'h2h_away', 'moneyline_away'):
                    # Only use 1.0 - away_implied if the orientation is truly unambiguous (i.e. explicitly labeled as away)
                    prob_series.at[idx] = 1.0 - val
                    implied_sources["side_specific"] += 1
                else:
                    # If orientation is ambiguous, do not guess, skip
                    pass

    # CRITICAL INSTRUCTION: Do NOT backfill `implied_home_prob` from `ml_probability`.

    # Check market_probability for excessive 0.5 use
    if 'market_probability' not in df.columns:
        features_data['market_probability'] = prob_series.fillna(0.5).infer_objects(copy=False)
    else:
        mkt_series = pd.to_numeric(df['market_probability'], errors='coerce')
        valid_mkt_series = mkt_series.where(~mkt_series.apply(_is_placeholder), np.nan)
        features_data['market_probability'] = valid_mkt_series.fillna(prob_series).fillna(0.5).infer_objects(copy=False)

    # Step 4: Final fallback to 0.5 only if still NaN
    implied_default_count = int(prob_series.isna().sum())
    implied_sources["defaulted"] = implied_default_count
    features_data['implied_home_prob'] = prob_series.fillna(0.5).infer_objects(copy=False)

    features_data['sentiment_diff'] = safe_numeric_fill(df.get('sentiment_diff'), 0.0)

    # Kalshi Probability Fallback Logic
    k_col = next((c for c in df.columns if str(c).lower() in ['kalshi_prob', 'kalshi_probability']), None)
    k_series = pd.Series([np.nan]*len(df), index=df.index)

    # 1. Real matched Kalshi prob
    if k_col:
        k_raw = pd.to_numeric(df[k_col], errors='coerce')
        k_series = k_series.fillna(k_raw.where(~k_raw.apply(_is_placeholder), np.nan)).infer_objects(copy=False)

    k_real_count = int(k_series.notna().sum())

    # 2. Logged proxy (market_probability / implied_home_prob)
    kalshi_proxy_counts = {"market_probability": 0, "implied_home_prob": 0}
    if k_series.isna().any():
        # Try market_probability first
        if 'market_probability' in features_data:
            mkt_probs_kalshi = features_data['market_probability']
            fallback_mask = k_series.isna() & mkt_probs_kalshi.notna() & (~mkt_probs_kalshi.apply(_is_placeholder))
            if fallback_mask.any():
                proxy_count = int(fallback_mask.sum())
                kalshi_proxy_counts["market_probability"] += proxy_count
                k_series = k_series.fillna(mkt_probs_kalshi.where(fallback_mask, np.nan)).infer_objects(copy=False)

        # Try implied_home_prob next
        if k_series.isna().any() and 'implied_home_prob' in features_data:
            impl_probs_kalshi = features_data['implied_home_prob']
            fallback_mask = k_series.isna() & impl_probs_kalshi.notna() & (~impl_probs_kalshi.apply(_is_placeholder))
            if fallback_mask.any():
                 proxy_count = int(fallback_mask.sum())
                 kalshi_proxy_counts["implied_home_prob"] += proxy_count
                 k_series = k_series.fillna(impl_probs_kalshi.where(fallback_mask, np.nan)).infer_objects(copy=False)

    k_default_count = int(k_series.isna().sum())

    # 3. Default 0.5
    features_data['kalshi_prob'] = k_series.fillna(0.5).infer_objects(copy=False)

    logger.info(f"--- CONTEXTUAL PROBABILITY AUDIT ---")
    logger.info(f"implied_home_prob:")
    logger.info(f"  - Existing valid: {implied_sources['existing_valid']}")
    logger.info(f"  - Computed de-vig: {implied_sources['computed_devig']}")
    logger.info(f"  - Side-specific orientation: {implied_sources['side_specific']}")
    logger.info(f"  - Defaulted to 0.5: {implied_sources['defaulted']}")
    logger.info(f"kalshi_prob:")
    logger.info(f"  - Real match: {k_real_count}")
    logger.info(f"  - Proxy (market_probability): {kalshi_proxy_counts['market_probability']}")
    logger.info(f"  - Proxy (implied_home_prob): {kalshi_proxy_counts['implied_home_prob']}")
    logger.info(f"  - Defaulted to 0.5: {k_default_count}")
    logger.info(f"------------------------------------")
    features_data['injuries_home_count'] = safe_numeric_fill(df.get('injuries_home_count'), 0)
    features_data['injuries_away_count'] = safe_numeric_fill(df.get('injuries_away_count'), 0)
    
    # Weather flag
    if 'weather_summary' in df.columns:
        features_data['weather_flag'] = df['weather_summary'].astype(str).str.lower().apply(
            lambda w: 1.0 if any(x in w for x in ['rain', 'snow', 'wind']) else 0.0
        )
    else:
        features_data['weather_flag'] = 0.0
        
    # Time features
    if 'Commence (UTC)' in df.columns:
        dt_series = pd.to_datetime(df['Commence (UTC)'], errors='coerce')
        features_data['feature_commence_hour'] = dt_series.dt.hour.fillna(19.0).infer_objects(copy=False)
        features_data['feature_commence_day_of_week'] = dt_series.dt.dayofweek.fillna(6.0).infer_objects(copy=False)
    else:
        features_data['feature_commence_hour'] = 19.0
        features_data['feature_commence_day_of_week'] = 6.0
        
    # Rest Days
    features_data['feature_home_rest_days'] = 3.0
    features_data['feature_away_rest_days'] = 3.0
    
    # --- SANITY CHECK LAYER ---
    features_df = pd.DataFrame(features_data, index=df.index)

    # Win PCT validation
    win_pct_cols = ['feature_home_win_pct', 'feature_away_win_pct', 'feature_home_last5_win_pct', 'feature_away_last5_win_pct']
    for col in win_pct_cols:
        if col in features_df.columns:
            # Check bounds [0, 1]
            invalid_mask = (features_df[col] < 0.0) | (features_df[col] > 1.0)
            if invalid_mask.any():
                logger.warning(f"Validation Warning: Found {invalid_mask.sum()} rows with invalid {col} (outside 0-1). Marking as fallback.")
                if 'feature_stats_fallback' in features_df.columns:
                    features_df.loc[invalid_mask, 'feature_stats_fallback'] = True
                # Clamp
                features_df.loc[invalid_mask, col] = features_df.loc[invalid_mask, col].clip(0.0, 1.0)

    # PPG validation (League dependent)
    ppg_cols = ['feature_home_ppg', 'feature_away_ppg', 'feature_home_oppg', 'feature_away_oppg']
    for col in ppg_cols:
        if col in features_df.columns:
            if 'feature_stats_fallback' in features_df.columns:
                zeros_mask = (features_df[col].abs() < 0.001) & (~features_df['feature_stats_fallback'])
                if zeros_mask.any():
                    features_df.loc[zeros_mask, 'feature_stats_fallback'] = True

    # --- SCHEMA ENFORCEMENT ---
    # PERFORMANCE FIX: Build missing columns as dict first, then add in one operation
    # This prevents DataFrame fragmentation from repeated column additions
    missing_cols = {}
    for col in VERTEX_FEATURE_COLUMNS:
        if col not in features_df.columns:
            # Determine default: 0.5 for probs, 0.0 for others
            default_val = 0.5 if "prob" in col else 0.0
            missing_cols[col] = default_val

    # Add all missing columns at once to avoid fragmentation
    if missing_cols:
        missing_df = pd.DataFrame(missing_cols, index=features_df.index)
        features_df = pd.concat([features_df, missing_df], axis=1)

    # Force float type for all Vertex columns (fixes [0.6, 0, ...] issue)
    for col in VERTEX_FEATURE_COLUMNS:
        if col in features_df.columns:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce').astype(float)

    # Consolidate memory layout to prevent fragmentation warnings
    features_df = features_df.copy()

    # Drop any columns from the original df that we just computed/updated in features_df
    # to prevent pd.concat(axis=1) from creating duplicated column names
    cols_to_drop = [c for c in features_df.columns if c in df.columns]
    if cols_to_drop:
        logger.info(f"Dropping {len(cols_to_drop)} columns from original dataframe to prevent duplication: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    result = pd.concat([df, features_df], axis=1)
    return result

def run_roi_pipeline_validation(df: pd.DataFrame):
    """Checks if the data bridge is actually functioning before export."""
    critical_checks = {
        "Vertex Prediction": "model_spread_prob",
        "Kalshi Probability": "kalshi_prob",
        "Team Stats (PPG)": "feature_home_ppg",
        "Team Stats (Win %)": "feature_home_win_pct"
    }
    
    validation_results = {}
    
    for label, col in critical_checks.items():
        if col not in df.columns:
            validation_results[label] = "❌ COLUMN MISSING"
            continue

        # Handle duplicate columns by selecting the first occurrence
        series_data = df[col]
        if isinstance(series_data, pd.DataFrame):
            series_data = series_data.iloc[:, 0]

        populated_count = series_data.notnull().sum()

        if populated_count == 0:
            validation_results[label] = "⚠️ COLUMN EMPTY (Data not reaching DF)"
        elif col == "feature_home_win_pct":
            # Check if all populated values are exactly 0.50 (default fallback)
            # We use a small tolerance for float comparison, though exactly 0.5 is expected for default
            is_default = ((series_data - 0.5).abs() < 1e-6)
            default_count = is_default.sum()

            if default_count == populated_count:
                validation_results[label] = "⚠️ ALL DEFAULTS (0.50 detected, stats fetch failed)"
            elif default_count > 0:
                validation_results[label] = f"⚠️ MIXED ({populated_count} rows, {default_count} defaults)"
            else:
                validation_results[label] = f"✅ OK ({populated_count} rows populated)"
        else:
            validation_results[label] = f"✅ OK ({populated_count} rows populated)"
            
    logger.info("--- ROI PIPELINE VALIDATION ---")
    for label, status in validation_results.items():
        logger.info(f"{label}: {status}")
        
    return validation_results

def ml_to_prob(ml):
    """
    Robust conversion of American odds to implied probability.
    Handles extreme values by returning NaN.
    """
    try:
        # Check for invalid values like None, NaN, or string 'None'
        if ml is None: return np.nan
        if pd.isna(ml): return np.nan
        if str(ml).strip().lower() == "none": return np.nan

        m = float(ml)
        if m != m or m == 0: return np.nan

        # Check for extreme/unrealistic odds (e.g. +/- 100000)
        if abs(m) > 50000:
            logger.warning(f"Extreme odds detected: {m}. Treating as missing.")
            return np.nan

        if m > 0: return 100/(m+100)
        return abs(m)/(abs(m)+100)
    except:
        return np.nan

def safefloat(val: Any) -> float:
    """Safely convert to float, defaulting to 0.0 on error/None."""
    if val is None:
        return 0.0
    try:
        f = float(val)
        if f != f: return 0.0 # NaN
        if math.isinf(f): return 0.0 # Handle inf/-inf
        return f
    except (ValueError, TypeError):
        return 0.0

def build_model_feature_row_from_record(record: Mapping[str, Any]) -> Dict[str, float]:
    """
    Build one Vertex feature row using the same columns and defaults
    as the batch enrich_with_model_features path.
    PROB features -> default 0.5, others -> default 0.0.
    """
    row: Dict[str, float] = {}

    for col in VERTEX_FEATURE_COLUMNS:
        val = record.get(col)

        # Fallback: try removing 'feature_' prefix if exact key missing
        if val is None and col.startswith("feature_"):
            val = record.get(col.replace("feature_", ""))

        # PROB features must default to 0.5 (Neutral), STATS/COUNTS to 0.0
        default_val = 0.5 if "prob" in col else 0.0

        if val is not None:
            row[col] = safefloat(val)
        else:
            row[col] = default_val

    return row

def calculate_confidence(row):
    # FORCE LOGIC: High Confidence = Probability >= 60%
    # We explicitly IGNORE Moneyline/EV checks here.
    prob = row.get('final_probability', 0)

    if prob >= 0.60:
        return "HIGH"
    elif prob >= 0.55:
        return "MEDIUM"
    else:
        return "LOW"
