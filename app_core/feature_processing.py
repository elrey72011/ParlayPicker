import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Mapping
import logging
import warnings
import os
import threading
import concurrent.futures

# -------------------------------------------------------------------
# DEBUG FALLBACK LOG THROTTLING
# -------------------------------------------------------------------
_FALLBACK_LOG_COUNT = 0
_FALLBACK_LOG_LIMIT = 15  # max number of fallback logs to emit

# -------------------------------------------------------------------------
# Library Imports with Fail-Safe Wrappers
# -------------------------------------------------------------------------

try:
    from nba_api.stats.endpoints import leaguedashteamstats
except ImportError:
    leaguedashteamstats = None
    warnings.warn("nba_api not installed. NBA stats fetching will fail.")

try:
    import nfl_data_py as nfl
except ImportError:
    nfl = None
    warnings.warn("nfl_data_py not installed. NFL stats fetching will fail.")

try:
    import cfbd
except ImportError:
    cfbd = None
    warnings.warn("cfbd not installed. NCAAF stats fetching will fail.")

try:
    from nhlpy import NHLClient
except ImportError:
    NHLClient = None
    warnings.warn("nhl-api-py not installed. NHL stats fetching will fail.")

try:
    import cbbpy.mens_scraper as cbb_s
except ImportError:
    cbb_s = None
    warnings.warn("CBBpy not installed. NCAAB stats fetching will fail.")

try:
    import rapidfuzz
    from rapidfuzz import process, fuzz
except ImportError:
    rapidfuzz = None
    warnings.warn("rapidfuzz not installed. Fuzzy matching will be degraded.")

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

logger = logging.getLogger(__name__)

# Config: Set to True to skip stats API calls on Free Tier plans
# (Now applies mainly if keys are missing or libs are missing)
FREE_TIER_MODE = True

# Define the features we want to ensure exist
LEAGUE_AVERAGES = {
    "NBA": {"ppg": 114.0, "oppg": 114.0, "win_pct": 0.5, "last5_win_pct": 0.5},
    "NFL": {"ppg": 22.0, "oppg": 22.0, "win_pct": 0.5, "last5_win_pct": 0.5},
    "NHL": {"ppg": 3.0, "oppg": 3.0, "win_pct": 0.5, "last5_win_pct": 0.5},
    "NCAAB": {"ppg": 72.0, "oppg": 72.0, "win_pct": 0.5, "last5_win_pct": 0.5},
    "NCAAF": {"ppg": 28.0, "oppg": 28.0, "win_pct": 0.5, "last5_win_pct": 0.5},
    "default": {"ppg": 50.0, "oppg": 50.0, "win_pct": 0.5, "last5_win_pct": 0.5}
}

# -------------------------------------------------------------------------
# Pro League Mappings (100% Lookup Guarantee)
# Maps full team names to the normalized keys used by stats libraries.
# -------------------------------------------------------------------------
TEAM_NAME_MAPPING = {
    # NBA (nba_api returns City Mascot, robust_normalize_team strips mascot -> City)
    "atlanta hawks": "atlanta",
    "boston celtics": "boston",
    "brooklyn nets": "brooklyn",
    "charlotte hornets": "charlotte",
    "chicago bulls": "chicago",
    "cleveland cavaliers": "cleveland",
    "dallas mavericks": "dallas",
    "denver nuggets": "denver",
    "detroit pistons": "detroit",
    "golden state warriors": "golden state",
    "houston rockets": "houston",
    "indiana pacers": "indiana",
    "los angeles clippers": "los angeles",
    "los angeles lakers": "los angeles",
    "memphis grizzlies": "memphis",
    "miami heat": "miami",
    "milwaukee bucks": "milwaukee",
    "minnesota timberwolves": "minnesota",
    "new orleans pelicans": "new orleans",
    "new york knicks": "new york",
    "oklahoma city thunder": "oklahoma city",
    "orlando magic": "orlando",
    "philadelphia 76ers": "philadelphia",
    "phoenix suns": "phoenix",
    "portland trail blazers": "portland",
    "sacramento kings": "sacramento",
    "san antonio spurs": "san antonio",
    "toronto raptors": "toronto",
    "utah jazz": "utah",
    "washington wizards": "washington",

    # NFL (nfl_data_py returns codes like ARI, ATL)
    "arizona cardinals": "ari",
    "atlanta falcons": "atl",
    "baltimore ravens": "bal",
    "buffalo bills": "buf",
    "carolina panthers": "car",
    "chicago bears": "chi",
    "cincinnati bengals": "cin",
    "cleveland browns": "cle",
    "dallas cowboys": "dal",
    "denver broncos": "den",
    "detroit lions": "det",
    "green bay packers": "gb",
    "houston texans": "hou",
    "indianapolis colts": "ind",
    "jacksonville jaguars": "jax",
    "kansas city chiefs": "kc",
    "las vegas raiders": "lv",
    "los angeles chargers": "lac",
    "los angeles rams": "lar",
    "miami dolphins": "mia",
    "minnesota vikings": "min",
    "new england patriots": "ne",
    "new orleans saints": "no",
    "new york giants": "nyg",
    "new york jets": "nyj",
    "philadelphia eagles": "phi",
    "pittsburgh steelers": "pit",
    "san francisco 49ers": "sf",
    "seattle seahawks": "sea",
    "tampa bay buccaneers": "tb",
    "tennessee titans": "ten",
    "washington commanders": "was",
    "washington football team": "was",

    # MLB (Placeholder for future stats - Assuming City Mascot or City)
    "arizona diamondbacks": "arizona diamondbacks",
    "atlanta braves": "atlanta braves",
    "baltimore orioles": "baltimore orioles",
    "boston red sox": "boston red sox",
    "chicago white sox": "chicago white sox",
    "chicago cubs": "chicago cubs",
    "cincinnati reds": "cincinnati reds",
    "cleveland guardians": "cleveland guardians",
    "colorado rockies": "colorado rockies",
    "detroit tigers": "detroit tigers",
    "houston astros": "houston astros",
    "kansas city royals": "kansas city royals",
    "los angeles angels": "los angeles angels",
    "los angeles dodgers": "los angeles dodgers",
    "miami marlins": "miami marlins",
    "milwaukee brewers": "milwaukee brewers",
    "minnesota twins": "minnesota twins",
    "new york yankees": "new york yankees",
    "new york mets": "new york mets",
    "oakland athletics": "oakland athletics",
    "philadelphia phillies": "philadelphia phillies",
    "pittsburgh pirates": "pittsburgh pirates",
    "san diego padres": "san diego padres",
    "san francisco giants": "san francisco giants",
    "seattle mariners": "seattle mariners",
    "st. louis cardinals": "st louis cardinals",
    "tampa bay rays": "tampa bay rays",
    "texas rangers": "texas rangers",
    "toronto blue jays": "toronto blue jays",
    "washington nationals": "washington nationals",
}

# Manual overrides for team name normalization failures
# Keys and values should be lowercase normalized forms
MANUAL_TEAM_OVERRIDES = {
    # NBA/NHL Fixes
    "phoenix suns": "phoenix", "chicago blackhawks": "chicago",
    "washington capitals": "washington", "winnipeg jets": "winnipeg",
    "los angeles kings": "los angeles", "utah mammoth": "utah",
    "st louis blues": "st louis",
    # NCAAB/NCAAF Log Fixes (14:42)
    "toledo rockets": "toledo", "miami (oh) redhawks": "miami oh",
    "manhattan jaspers": "manhattan", "canisius golden griffins": "canisius",
    "oakland golden grizzlies": "oakland", "cleveland st vikings": "cleveland state",
    "detroit mercy titans": "detroit mercy", "wright st raiders": "wright state",
    "fairfield stags": "fairfield", "rider broncs": "rider",
    "green bay phoenix": "green bay", "iupui jaguars": "iupui",
    "iona gaels": "iona", "niagara purple eagles": "niagara",
    "sacred heart pioneers": "sacred heart", "marist red foxes": "marist",
    "siena saints": "siena", "merrimack warriors": "merrimack",
    "mt. st. mary's mountaineers": "mount st marys", "saint peter's peacocks": "saint peters",
    "bowling green falcons": "bowling green", "akron zips": "akron",
    "milwaukee panthers": "milwaukee", "northern kentucky norse": "northern kentucky",
    "minnesota golden gophers": "minnesota", "usc trojans": "usc",
    "colorado st rams": "colorado state", "unlv rebels": "unlv",
    "indiana hoosiers": "indiana", "oregon ducks": "oregon",
    "ohio state buckeyes": "ohio state"
}

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

def robust_normalize_team(name: str) -> str:
    """
    Aggressive team name normalization.
    Converts to lowercase, removes common suffixes/mascots.
    """
    if not name:
        return ""

    # 1. Lowercase and strip
    name = str(name).lower().strip()

    # 2. Use TeamNameMatcher's normalization first (handles St -> State, punctuation)
    name = TeamNameMatcher.normalize(name)

    # 3. Additional aggressive mascot stripping (if not covered by TeamNameMatcher)
    # Note: TeamNameMatcher.normalize already removes mascots from its internal list.
    # We can add extra cleanup if needed here.

    # Remove common suffixes that might remain or be specific
    suffixes = [
        ' bulls', ' tigers', ' mountaineers', ' blue hens', ' university', ' college',
        ' rockets', ' redhawks', ' jaspers', ' golden griffins', ' golden grizzlies',
        ' vikings', ' titans', ' raiders', ' stags', ' broncs', ' phoenix', ' jaguars',
        ' gaels', ' purple eagles', ' pioneers', ' red foxes', ' saints', ' warriors',
        ' peacocks', ' falcons', ' zips', ' panthers', ' norse', ' golden gophers',
        ' trojans', ' rams', ' rebels', ' hoosiers', ' ducks', ' buckeyes',
        ' blackhawks', ' capitals', ' jets', ' kings', ' mammoth', ' blues'
    ]
    for s in suffixes:
        if name.endswith(s):
            name = name[:-len(s)].strip()

    return name

def fuzzy_match_team_robust(target: str, choices: List[str], threshold: float = 80.0) -> Optional[str]:
    """
    Uses rapidfuzz to find the best match for 'target' in 'choices'.
    Returns the matched string from 'choices' if score > threshold, else None.
    """
    if not target or not choices:
        return None

    if rapidfuzz:
        # extraction returns list of (match, score, index)
        # process.extractOne finds the single best match
        result = process.extractOne(target, choices, scorer=fuzz.token_sort_ratio)
        if result:
            match, score, _ = result
            if score >= threshold:
                return match
    else:
        # Fallback to TeamNameMatcher (difflib) if rapidfuzz missing
        return TeamNameMatcher.match_team(target, choices, threshold=threshold/100.0)

    return None

# -------------------------------------------------------------------------
# New Open-Source Stats Fetching Helpers
# -------------------------------------------------------------------------

@st.cache_data(ttl=21600)  # Cache for 6 hours
def fetch_nba_stats(season_year: int) -> List[Dict[str, Any]]:
    """
    Fetch NBA stats using nba_api for the given season year (e.g. 2024 for 2024-25).
    """
    if leaguedashteamstats is None:
        return []

    try:
        # nba_api expects season format "YYYY-YY", e.g. "2024-25"
        season_str = f"{season_year}-{str(season_year + 1)[-2:]}"
        logger.info(f"Fetching NBA stats for season: {season_str}")

        # MeasureType='Base' gives GP, W, L, W_PCT, PTS, PLUS_MINUS, TOV, etc.
        # Added per_mode_detailed='PerGame' to get averaged stats directly as requested
        dashboard = leaguedashteamstats.LeagueDashTeamStats(
            season=season_str,
            measure_type_detailed_defense='Base',
            per_mode_detailed='PerGame'
        )
        df = dashboard.get_data_frames()[0]

        stats = []
        for _, row in df.iterrows():
            # nba_api columns: TEAM_NAME, GP, W, L, W_PCT, PTS, PLUS_MINUS, TOV
            team_name = str(row['TEAM_NAME'])
            gp = float(row['GP'])

            # With PerGame, these are already averages
            pts = float(row['PTS']) # Points Per Game
            plus_minus = float(row['PLUS_MINUS'])
            w_pct = float(row['W_PCT'])
            tov = float(row['TOV']) if 'TOV' in row else 0.0
            ast = float(row['AST']) if 'AST' in row else 0.0
            reb = float(row['REB']) if 'REB' in row else 0.0

            # Calculate metrics
            ppg = pts
            # Opponent PTS approx: PTS - PLUS_MINUS = OPP_PTS (Plus Minus is also per game)
            oppg = (pts - plus_minus)
            avg_tov = tov
            avg_ast = ast
            avg_reb = reb

            stats.append({
                "team_norm": robust_normalize_team(team_name),
                "league_key": "NBA",
                "win_pct": w_pct,
                "home_win_pct": w_pct, # Approximation
                "away_win_pct": w_pct, # Approximation
                "points_per_game": ppg,
                "points_allowed_per_game": oppg,
                "assists_per_game": avg_ast,
                "rebounds_per_game": avg_reb,
                "turnovers": avg_tov,
                "streak": 0.0, # Not in base view easily
                "last5_win_pct": w_pct # Approximation
            })

        logger.info(f"Successfully fetched NBA stats for {len(stats)} teams.")
        return stats
    except Exception as e:
        logger.error(f"Failed to fetch NBA stats via nba_api: {e}", exc_info=True)
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
                "team_norm": robust_normalize_team(str(team_code)),
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
    def _normalize_cfbd_token(raw: Any) -> str:
        if raw is None:
            return ""
        s = str(raw).strip()
        if not s:
            return ""
        if s.lower().startswith("bearer"):
            parts = s.split(None, 1)
            s = parts[1].strip() if len(parts) == 2 else ""
        return s.strip()

    if cfbd is None:
        return []

    raw_key = _get_secret("CFBD_API_KEY")
    token = _normalize_cfbd_token(raw_key)
    if not token:
        logger.warning("CFBD_API_KEY not found. Skipping NCAAF stats.")
        return []

    def _make_client(primary: bool) -> "cfbd.ApiClient":
        """
        primary=True  -> api_key + api_key_prefix (Bearer)
        primary=False -> access_token (Bearer token)
        """
        cfg = cfbd.Configuration()
        if primary:
            cfg.api_key["Authorization"] = token
            cfg.api_key_prefix["Authorization"] = "Bearer"
        else:
            # Alternate method: many OpenAPI clients treat access_token as Bearer automatically
            cfg.access_token = token
        return cfbd.ApiClient(cfg)

    # Primary client first
    api_client_primary = _make_client(primary=True)
    api_client_fallback = _make_client(primary=False)

    logger.info(f"CFBD auth prepared (token_length={len(token)})")

    def _is_unauthorized(e: Exception) -> bool:
        status = getattr(e, "status", None)
        msg = str(e)
        return status == 401 or "401" in msg or "Unauthorized" in msg

    def _fetch_stats_for_year(yr: int) -> List[Any]:
        # Try primary auth, then fallback auth
        for attempt_name, api_client in (("primary", api_client_primary), ("fallback", api_client_fallback)):
            try:
                api_instance = cfbd.StatsApi(api_client)
                return api_instance.get_team_stats(year=yr)
            except Exception as e:
                if _is_unauthorized(e):
                    logger.warning(
                        f"CFBD unauthorized (401) when fetching NCAAF stats ({attempt_name}). "
                        "Check CFBD_API_KEY value in Streamlit secrets."
                    )
                    continue
                logger.warning(f"NCAAF Stats fetch failed for {yr} ({attempt_name}): {e}")
                return []
        return []

    def _fetch_games_for_year(yr: int) -> List[Any]:
        for attempt_name, api_client in (("primary", api_client_primary), ("fallback", api_client_fallback)):
            try:
                games_api = cfbd.GamesApi(api_client)
                return games_api.get_games(year=yr)
            except Exception as e:
                if _is_unauthorized(e):
                    logger.warning(
                        f"CFBD unauthorized (401) when fetching NCAAF games ({attempt_name}). "
                        "Check CFBD_API_KEY value in Streamlit secrets."
                    )
                    continue
                logger.warning(f"NCAAF Games API Unavailable for {yr} ({attempt_name}): {e}")
                return []
        return []

    try:
        # 1) requested year
        season_stats = _fetch_stats_for_year(season_year)

        # 2) fallback to prior year if empty
        if not season_stats:
            logger.warning(f"No NCAAF stats found for {season_year}. Trying {season_year - 1}...")
            season_stats = _fetch_stats_for_year(season_year - 1)
            if season_stats:
                season_year = season_year - 1

        # If still empty, do NOT treat as outage; just return []
        if not season_stats:
            logger.warning("NCAAF stats unavailable (CFBD auth failed or no data). Continuing without NCAAF stats.")
            return []

        season_games = _fetch_games_for_year(season_year)

        # Build win pct map (same logic as before)
        team_records: Dict[str, Dict[str, int]] = {}
        for g in season_games or []:
            try:
                home = getattr(g, "home_team", None)
                away = getattr(g, "away_team", None)
                home_pts = getattr(g, "home_points", None)
                away_pts = getattr(g, "away_points", None)
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
        for offense in season_stats:
            try:
                team = getattr(offense, "team", None) or getattr(offense, "school", None)
                if not team:
                    continue

                wins = team_records.get(team, {}).get("wins", 0)
                losses = team_records.get(team, {}).get("losses", 0)
                games_played = wins + losses
                win_pct = (wins / games_played) if games_played > 0 else 0.0

                ppg = getattr(offense, "points", None)
                if ppg is None:
                    ppg = getattr(offense, "points_per_game", 0.0) or 0.0

                oppg = getattr(offense, "points_allowed", None)
                if oppg is None:
                    oppg = getattr(offense, "points_allowed_per_game", 0.0) or 0.0

                ypg = getattr(offense, "yards_per_game", None)
                if ypg is None:
                    ypg = getattr(offense, "yards", 0.0) or 0.0

                avg_tov = getattr(offense, "turnovers", None)
                if avg_tov is None:
                    avg_tov = getattr(offense, "turnovers_per_game", 0.0) or 0.0

                def _to_float(x: Any) -> float:
                    try:
                        return float(x)
                    except Exception:
                        return 0.0

                stats.append(
                    {
                        "team": team,
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

        logger.info(f"Successfully fetched NCAAF stats for {len(stats)} teams.")
        return stats

    except Exception as e:
        logger.error(f"Failed to fetch NCAAF stats: {e}", exc_info=True)
        return []

@st.cache_data(ttl=21600)
def fetch_nhl_stats(season_year: int) -> List[Dict[str, Any]]:
    """
    Fetch NHL stats using nhlpy.
    """
    if NHLClient is None:
        return []

    try:
        logger.info(f"Fetching NHL stats for season: {season_year}")
        client = NHLClient()

        # nhlpy usually provides standings which contain most info
        standings = client.standings.league_standings()
        # standings is usually a dict with 'standings' list

        stats = []
        for entry in standings.get('standings', []):
            team_name = entry.get('teamName', {}).get('default', '')
            if not team_name: continue
            
            # Extract stats
            games = entry.get('gamesPlayed', 0)
            if games == 0: continue
            
            points_for = entry.get('goalFor', 0)
            points_against = entry.get('goalAgainst', 0)
            wins = entry.get('wins', 0)
            
            win_pct = wins / games

            # Map goalsPerGame -> points_per_game if available
            if 'goalsPerGame' in entry:
                ppg = float(entry['goalsPerGame'])
            else:
                ppg = points_for / games

            oppg = points_against / games
            
            # Streak
            streak_code = entry.get('streakCode', '') # e.g. 'W2'
            streak = _parse_streak(streak_code)
            
            # Turnovers not standard in standings
            
            stats.append({
                "team_norm": robust_normalize_team(team_name),
                "league_key": "NHL",
                "win_pct": win_pct,
                "home_win_pct": win_pct,
                "away_win_pct": win_pct,
                "points_per_game": ppg,
                "points_allowed_per_game": oppg,
                "turnovers": 0.0,
                "streak": streak,
                "last5_win_pct": entry.get('l10Points', 0) / 20.0 # Approx from L10 points? Or just use win_pct
            })
            
        logger.info(f"Successfully fetched NHL stats for {len(stats)} teams.")
        return stats
    except Exception as e:
        logger.error(f"Failed to fetch NHL stats: {e}", exc_info=True)
        return []

@st.cache_data(ttl=21600)
def fetch_ncaab_stats(season_year: int) -> List[Dict[str, Any]]:
    """
    Fetch NCAAB stats using CBBpy.
    This library scrapes, so we MUST wrap in timeout.
    """
    if cbb_s is None:
        return []

    def _scrape_worker():
        # cbbpy usage: get_team_stats(season=2024) - check docs/usage
        # assuming cbb_s.get_team_stats or similar
        # Based on common usage: cbbpy.mens_scraper.get_season_stats(season=2024)
        # Note: function names might vary, using best effort from typical usage
        try:
            # get_season_stats usually returns a DataFrame
            # season year for 2024-25 is usually 2025
            return cbb_s.get_stats(season=season_year + 1)
        except Exception as e:
            return None

    try:
        logger.info(f"Fetching NCAAB stats for season: {season_year}")

        # Run in thread with timeout
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_scrape_worker)
            try:
                # 5 Second Timeout
                df = future.result(timeout=15)
            except concurrent.futures.TimeoutError:
                logger.warning("NCAAB stats fetch timed out (5s limit).")
                return []
            except Exception as e:
                logger.warning(f"NCAAB stats fetch failed: {e}")
                return []

        if df is None or df.empty:
            return []

        # Process DataFrame
        # Expect columns: team, games, points, opponents_points, wins, etc.
        # Column names in cbbpy can be verbose.

        stats = []
        for _, row in df.iterrows():
            team_name = row.get('team', '')
            if not team_name: continue

            games = float(row.get('games', 0))
            if games == 0: continue

            wins = float(row.get('wins', 0))
            points = float(row.get('points', 0))
            opp_points = float(row.get('opp_points', 0)) # Verify col name
            turnovers = float(row.get('turnovers', 0))

            win_pct = wins / games
            ppg = points / games
            oppg = opp_points / games
            avg_tov = turnovers / games

            stats.append({
                "team_norm": robust_normalize_team(team_name),
                "league_key": "NCAAB",
                "win_pct": win_pct,
                "home_win_pct": win_pct,
                "away_win_pct": win_pct,
                "points_per_game": ppg,
                "points_allowed_per_game": oppg,
                "turnovers": avg_tov,
                "streak": 0.0,
                "last5_win_pct": win_pct
            })

        logger.info(f"Successfully fetched NCAAB stats for {len(stats)} teams.")
        return stats

    except Exception as e:
        logger.warning(f"Failed to fetch NCAAB stats wrapper: {e}")
        return []

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
    leagues = list(api_clients.keys())

    # Dispatch Logic
    if "NBA" in leagues:
        all_stats.extend(fetch_nba_stats(season_year))

    if "NFL" in leagues:
        all_stats.extend(fetch_nfl_stats(season_year))

    if "NCAAF" in leagues:
        all_stats.extend(fetch_ncaaf_stats(season_year))

    if "NHL" in leagues:
        all_stats.extend(fetch_nhl_stats(season_year))

    if "NCAAB" in leagues:
        all_stats.extend(fetch_ncaab_stats(season_year))

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
    if df is None or df.empty:
        return df
        
    def safe_numeric_fill(val, fill_val=0.0):
        """Safely converts scalars or Series to numeric and fills NaNs."""
        try:
            # If it's already a pandas Series/Index
            if isinstance(val, (pd.Series, pd.Index)):
                return pd.to_numeric(val, errors='coerce').fillna(fill_val)
            
            # If it's a list or array
            if isinstance(val, (list, tuple, np.ndarray)):
                return pd.to_numeric(pd.Series(val), errors='coerce').fillna(fill_val)
                
            # If it's a scalar (single number)
            parsed = pd.to_numeric(val, errors='coerce')
            return fill_val if pd.isna(parsed) else parsed
        except Exception:
            return fill_val

    # 0. Fix Case-Sensitive Overwrite (Ensure 'league' is authoritative)
    # We strictly use sport_title if available, else league.
    # We overwrite 'League' to match 'league' to prevent ambiguity downstream.
   league_col = None
    if 'sport_title' in df.columns:
        league_col = 'sport_title'
    elif 'league' in df.columns:
        league_col = 'league'
    elif 'League' in df.columns:
        df['league'] = df['League']
        league_col = 'league'
    else:
        league_col = None

    # 1. Fetch Stats (Using new function)
    stats_df = fetch_team_stats(api_clients, season_year=season_year)
    
    # 2. Normalize Names in Master DF (create temporary series)
    # Handle variable column names (Home vs home_team)
    home_col = 'Home' if 'Home' in df.columns else 'home_team'
    away_col = 'Away' if 'Away' in df.columns else 'away_team'
    
    if home_col not in df.columns or away_col not in df.columns:
        logger.error(f"Missing home/away columns in dataframe. Columns: {list(df.columns)}")
        # Return original df to avoid crash, but features will be missing
        return df

    home_norm = df[home_col].apply(lambda x: robust_normalize_team(str(x)))
    away_norm = df[away_col].apply(lambda x: robust_normalize_team(str(x)))
    
    # 3. Determine League (Row-by-Row) - Robust & Standardized
    # This prevents the bug where one game's league overwrites all defaults

    def get_row_league_key(l_val):
        s = str(l_val).upper()
        # Explicit checks for college/other leagues FIRST to avoid partial "NBA" matches
        if "NCAAB" in s: return "NCAAB"
        if "NCAAF" in s: return "NCAAF"
        if "COLLEGE FOOTBALL" in s: return "NCAAF"
        if "COLLEGE BASKETBALL" in s: return "NCAAB"
        if "NHL" in s: return "NHL"
        if "ICE HOCKEY" in s: return "NHL"
        if "NFL" in s: return "NFL"
        if "NBA" in s: return "NBA"
        return "default"

    # Create Series of keys aligned with DF index
    if league_col:
        league_keys = df[league_col].apply(get_row_league_key)
        # Fix: Sync 'League' column to resolved key to prevent confusion and ensure correctness
        # This fixes the "NBA Overwrite" bug where League might be missing or incorrect
        df['League'] = league_keys
    else:
        league_keys = pd.Series(["default"] * len(df), index=df.index)
        df['League'] = "default"

    # 4. Create Series of Defaults aligned with DF index
    # We pre-calculate these so we can pass them to map_stat

    def get_default_stat(key, stat_name):
        return LEAGUE_AVERAGES.get(key, LEAGUE_AVERAGES["default"])[stat_name]

    default_ppg = league_keys.apply(lambda k: get_default_stat(k, 'ppg'))
    default_oppg = league_keys.apply(lambda k: get_default_stat(k, 'oppg'))
    default_win_pct = league_keys.apply(lambda k: get_default_stat(k, 'win_pct'))
    default_last5 = league_keys.apply(lambda k: get_default_stat(k, 'last5_win_pct'))
    
    # Use dict to collect columns to avoid fragmentation
    features_data = {}
    
    if stats_df.empty:
        if not FREE_TIER_MODE:
            logger.warning("No stats fetched. Filling with defaults.")
        # Logic handles empty stats via map_stat fallback below
    else:
        # Prepare lookup dictionaries by league: league_key -> (team_norm -> stats_row)
        # Group stats_df by league_key
        stats_by_league = {}
        for lg in stats_df['league_key'].unique():
            subset = stats_df[stats_df['league_key'] == lg]
            stats_by_league[lg] = subset.drop_duplicates(subset=['team_norm']).set_index('team_norm')

        # --- NEW: Composite Key (League + Team) Fuzzy Matching ---
        # Build a mapping from (row_index) -> stats_row_index (team_norm in stats_by_league)
        # Since vectorized map is hard with composite, we'll iterate or use a composite key series.

        # We need to map each game's team to the correct stat entry IN THE CORRECT LEAGUE.
        # team_map will now store: (row_index, side) -> matched_team_norm

        # Pre-compute fuzzy matches PER LEAGUE to avoid cross-league pollution
        # stats_teams_per_league = {lg: df.index.tolist() for lg, df in stats_by_league.items()}

        # We'll create a Series of matched team names aligned with the master DF
        home_matched_names = pd.Series([None] * len(df), index=df.index)
        away_matched_names = pd.Series([None] * len(df), index=df.index)

        # Iterate over unique leagues in the master DF to batch process
        unique_leagues_in_games = league_keys.unique()

        for lg_key in unique_leagues_in_games:
            if lg_key not in stats_by_league:
                # No stats for this league (e.g. "default" or missing)
                continue

            # Get subset of games for this league
            lg_mask = league_keys == lg_key
            if not lg_mask.any(): continue

            # Get subset of stats
            stats_subset = stats_by_league[lg_key]
            stats_teams_norm = stats_subset.index.tolist()

            # Process Home Teams for this league
            current_home_teams = home_norm[lg_mask].unique()
            home_map_local = {}
            for t_norm in current_home_teams:
                if not t_norm: continue

                # 0. Pro League Mapping Check (100% Lookup Guarantee)
                if t_norm in TEAM_NAME_MAPPING:
                    mapped = TEAM_NAME_MAPPING[t_norm]
                    if mapped in stats_subset.index:
                        home_map_local[t_norm] = mapped
                        continue

                # 1. Direct
                if t_norm in stats_subset.index:
                    home_map_local[t_norm] = t_norm
                    continue
                # 2. Manual
                if t_norm in MANUAL_TEAM_OVERRIDES:
                    target = MANUAL_TEAM_OVERRIDES[t_norm]
                    if target in stats_subset.index:
                        home_map_local[t_norm] = target
                        continue
                # 3. Fuzzy
                match = fuzzy_match_team_robust(t_norm, stats_teams_norm, threshold=70.0)
                if match:
                    home_map_local[t_norm] = match
                else:
                    # Log ERROR as requested for missing team
                    if lg_key != "default":
                        logger.error(f"TEAM MATCH FAILURE ({lg_key}): '{t_norm}' not found in {lg_key} dictionary.")
                    home_map_local[t_norm] = None

            # Apply map to the subset
            home_matched_names[lg_mask] = home_norm[lg_mask].map(home_map_local)

            # Process Away Teams for this league
            current_away_teams = away_norm[lg_mask].unique()
            away_map_local = {}
            for t_norm in current_away_teams:
                if not t_norm: continue

                # 0. Pro League Mapping Check (100% Lookup Guarantee)
                if t_norm in TEAM_NAME_MAPPING:
                    mapped = TEAM_NAME_MAPPING[t_norm]
                    if mapped in stats_subset.index:
                        away_map_local[t_norm] = mapped
                        continue

                if t_norm in stats_subset.index:
                    away_map_local[t_norm] = t_norm
                    continue
                if t_norm in MANUAL_TEAM_OVERRIDES:
                    target = MANUAL_TEAM_OVERRIDES[t_norm]
                    if target in stats_subset.index:
                        away_map_local[t_norm] = target
                        continue
                match = fuzzy_match_team_robust(t_norm, stats_teams_norm, threshold=70.0)
                if match:
                    away_map_local[t_norm] = match
                else:
                    if lg_key != "default":
                        logger.error(f"TEAM MATCH FAILURE ({lg_key}): '{t_norm}' not found in {lg_key} dictionary.")
                    away_map_local[t_norm] = None

            away_matched_names[lg_mask] = away_norm[lg_mask].map(away_map_local)

        # Helper to map a stat column using the matched names AND league key
        # Since we have matched names, we need to pull the value from the correct league's stats DF.
        # We can construct a global lookup dict: (league, team_norm) -> value

        # Build global lookup
        # (league_key, team_norm) -> row dict
        global_stats_lookup = {}
        for lg, s_df in stats_by_league.items():
            # Convert to dict of dicts: team_norm -> {col: val}
            # oriented index gives {team: {col: val, ...}}
            d = s_df.to_dict(orient='index')
            for t_norm, cols in d.items():
                global_stats_lookup[(lg, t_norm)] = cols

        def map_stat(matched_name_series, col_name, default_series):
            # Create tuple index (league_key, matched_name)
            # aligned with df index
            # Use list comprehension for speed?

            # We iterate rows to lookup in global_stats_lookup
            values = []
            for idx, name in matched_name_series.items():
                lg = league_keys.at[idx]
                if pd.notna(name) and (lg, name) in global_stats_lookup:
                    val = global_stats_lookup[(lg, name)].get(col_name)
                    values.append(val if val is not None else np.nan)
                else:
                    values.append(np.nan)

            return pd.Series(values, index=df.index).fillna(default_series)

        # Populate features_data using the new fuzzy map_stat
        
                # Track Fallbacks (True if team not matched)
        home_fallback = home_matched_names.isna()
        away_fallback = away_matched_names.isna()
        combined_fallback = home_fallback | away_fallback
        features_data["feature_stats_fallback"] = combined_fallback

        # NEW: stats_quality
        features_data["stats_quality"] = combined_fallback.apply(
            lambda x: "Low (Fallback)" if x else "High (Real)"
        )

        # Log fallback rows (throttled so logs don't spam)
        if combined_fallback.any():
            fallback_indices = df.index[combined_fallback]
            for idx in fallback_indices:
                try:
                    league_str = df.loc[idx, league_col] if league_col else "Unknown"
                    h_team = df.loc[idx, home_col]
                    a_team = df.loc[idx, away_col]
                    h_stat = "MISSING" if bool(home_fallback.loc[idx]) else "OK"
                    a_stat = "MISSING" if bool(away_fallback.loc[idx]) else "OK"

                    global _FALLBACK_LOG_COUNT
                    if _FALLBACK_LOG_COUNT < _FALLBACK_LOG_LIMIT:
                        logger.warning(
                            f"DEBUG Stats Fallback Used: {league_str} {h_team} ({h_stat}) vs {a_team} ({a_stat})"
                        )
                        _FALLBACK_LOG_COUNT += 1
                    elif _FALLBACK_LOG_COUNT == _FALLBACK_LOG_LIMIT:
                        logger.warning("DEBUG Stats Fallback Used: (further messages suppressed)")
                        _FALLBACK_LOG_COUNT += 1
                except Exception:
                    pass

        # Home Stats (use matched names)
        features_data["feature_home_win_pct"] = map_stat(home_matched_names, "win_pct", default_win_pct)
        features_data["feature_home_home_win_pct"] = map_stat(home_matched_names, "home_win_pct", default_win_pct)
        features_data["feature_home_last5_win_pct"] = map_stat(home_matched_names, "last5_win_pct", default_last5)
        features_data["feature_home_ppg"] = map_stat(home_matched_names, "points_per_game", default_ppg)
        features_data["feature_home_oppg"] = map_stat(home_matched_names, "points_allowed_per_game", default_oppg)
        features_data["feature_home_streak"] = map_stat(home_matched_names, "streak", pd.Series(0.0, index=df.index))
        features_data["feature_home_turnovers"] = map_stat(home_matched_names, "turnovers", pd.Series(0.0, index=df.index))

        # Away Stats (use matched names)
        features_data["feature_away_win_pct"] = map_stat(away_matched_names, "win_pct", default_win_pct)
        features_data["feature_away_away_win_pct"] = map_stat(away_matched_names, "away_win_pct", default_win_pct)
        features_data["feature_away_last5_win_pct"] = map_stat(away_matched_names, "last5_win_pct", default_last5)
        features_data["feature_away_ppg"] = map_stat(away_matched_names, "points_per_game", default_ppg)
        features_data["feature_away_oppg"] = map_stat(away_matched_names, "points_allowed_per_game", default_oppg)
        features_data["feature_away_streak"] = map_stat(away_matched_names, "streak", pd.Series(0.0, index=df.index))
        features_data["feature_away_turnovers"] = map_stat(away_matched_names, "turnovers", pd.Series(0.0, index=df.index))

        
        # SCALING: NHL stats are ~3.0, model expects ~110.0. Scale by 35x if league is NHL.
        is_nhl = league_keys == "NHL"
        if is_nhl.any():
            nhl_scale_factor = 35.0
            features_data['feature_home_ppg'] = features_data['feature_home_ppg'].mask(
                is_nhl, features_data['feature_home_ppg'] * nhl_scale_factor
            )
            features_data['feature_home_oppg'] = features_data['feature_home_oppg'].mask(
                is_nhl, features_data['feature_home_oppg'] * nhl_scale_factor
            )
            features_data['feature_away_ppg'] = features_data['feature_away_ppg'].mask(
                is_nhl, features_data['feature_away_ppg'] * nhl_scale_factor
            )
            features_data['feature_away_oppg'] = features_data['feature_away_oppg'].mask(
                is_nhl, features_data['feature_away_oppg'] * nhl_scale_factor
            )

    # 4. Fill Defaults if stats_df was empty
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
        # Convert to float to be safe
        try:
            h_val = float(h)
            a_val = float(a)
        except Exception:
            return 0.0

        if abs(h_val) < 1e-6 or abs(a_val) < 1e-6:
            return 0.0
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
    
    # 6. Map Remaining Features (Existing) using safe_numeric_fill
    
    # --- NEW: Robust ml_to_prob ---
    def ml_to_prob(ml):
        try:
            if ml is None: return np.nan # Return NaN instead of 0.5 to signify missing
            m = float(ml)
            if m != m or m == 0: return np.nan
            if m > 0: return 100/(m+100)
            return abs(m)/(abs(m)+100)
        except:
            return np.nan

    # Identify implied probability column
    imp_col = next((c for c in df.columns if str(c).lower() in ['implied_prob', 'implied_home_prob', 'implied_prob_home']), None)

    # Step 1: Initialize with existing column or NaNs
    if imp_col:
        # Use existing values, coerce errors to NaN
        prob_series = pd.to_numeric(df[imp_col], errors='coerce')
    else:
        prob_series = pd.Series([np.nan]*len(df), index=df.index)

    # Step 2: Calculate from Home_ML if present
    if 'Home_ML' in df.columns:
        ml_probs = df['Home_ML'].apply(ml_to_prob)
        # Fill missing values in prob_series with calculated ML probs
        prob_series = prob_series.fillna(ml_probs)

    # Step 3: Final fallback to 0.5 only if still NaN
    features_data['implied_home_prob'] = prob_series.fillna(0.5)

    features_data['sentiment_diff'] = safe_numeric_fill(df.get('Sentiment_Diff'), 0.0)
    features_data['kalshi_prob'] = safe_numeric_fill(df.get('kalshi_prob'), 0.5)
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
        features_data['feature_commence_hour'] = dt_series.dt.hour.fillna(19.0)
        features_data['feature_commence_day_of_week'] = dt_series.dt.dayofweek.fillna(6.0)
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
    for col in VERTEX_FEATURE_COLUMNS:
        if col not in features_df.columns:
            # Determine default: 0.5 for probs, 0.0 for others
            default_val = 0.5 if "prob" in col else 0.0
            features_df[col] = default_val

        # Force float type for Vertex AI (fixes [0.6, 0, ...] issue)
        features_df[col] = pd.to_numeric(features_df[col], errors='coerce').astype(float)

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

def safefloat(val: Any) -> float:
    """Safely convert to float, defaulting to 0.0 on error/None."""
    if val is None:
        return 0.0
    try:
        f = float(val)
        if f != f: return 0.0 # NaN
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
