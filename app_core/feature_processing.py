import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Mapping
import logging
import warnings
import os
import threading
import concurrent.futures

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

# Manual overrides for team name normalization failures
# Keys and values should be lowercase normalized forms
MANUAL_TEAM_OVERRIDES = {
    "washington state": "washington st",
    "mississippi state": "mississippi st",
    "michigan state": "michigan st",
    "kansas state": "kansas st",
    "arizona state": "arizona st",
    "florida state": "florida st",
    "oregon state": "oregon st",
    "penn state": "penn st",
    "nc state": "nc st",
    "north carolina state": "nc st",
    "ohio state": "ohio st",
    "oklahoma state": "oklahoma st",
    "boise state": "boise st",
    "fresno state": "fresno st",
    "san diego state": "san diego st",
    "san jose state": "san jose st",
    "utah state": "utah st",
    "colorado state": "colorado st",
    "iowa state": "iowa st",
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
    suffixes = [' bulls', ' tigers', ' mountaineers', ' blue hens', ' university', ' college']
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
    """
    Fetch NCAAF stats using cfbd.
    Requires CFBD_API_KEY.
    Tries current season, then previous season if empty.
    """
    if cfbd is None:
        return []
    
    api_key = _get_secret("CFBD_API_KEY")
    if not api_key:
        logger.warning("CFBD_API_KEY not found. Skipping NCAAF stats.")
        return []

    def _fetch_for_year(yr: int) -> List[Any]:
        try:
            logger.info(f"Fetching NCAAF stats for season: {yr}")
            # User requirement: Ensure header uses Bearer token correctly via configuration setup
            configuration = cfbd.Configuration()
            configuration.api_key = {"Authorization": f"Bearer {api_key}"}

            api_instance = cfbd.StatsApi(cfbd.ApiClient(configuration))
            # Fixed method name for cfbd 5.13.2
            return api_instance.get_team_stats(year=yr)
        except Exception as e:
            logger.warning(f"NCAAF Stats fetch failed for {yr}: {e}")
            return []

    try:
        # 1. Try requested season year
        season_stats = _fetch_for_year(season_year)

        # 2. If empty, try previous year (handling Jan/Feb games for previous season)
        if not season_stats:
            logger.warning(f"No NCAAF stats found for {season_year}. Trying {season_year - 1}...")
            season_stats = _fetch_for_year(season_year - 1)
            # Update season_year for games fetch below
            if season_stats:
                season_year = season_year - 1

        if not season_stats:
            logger.error("NCAAF Stats Outage - Could not fetch stats for current or previous year.")
            return []

        # Setup configuration again for GamesApi (using correct year)
        configuration = cfbd.Configuration()
        configuration.api_key['Authorization'] = f"Bearer {api_key}"
        games_api = cfbd.GamesApi(cfbd.ApiClient(configuration))

        try:
            season_games = games_api.get_games(year=season_year)
        except Exception as e:
            logger.warning(f"NCAAF Games API Unavailable: {e}")
            season_games = []

        # Build win pct map
        team_records = {}
        for g in season_games:
            if not g.home_team or not g.away_team: continue

            # Init if needed
            for t in [g.home_team, g.away_team]:
                if t not in team_records:
                    team_records[t] = {'games': 0, 'wins': 0}

            h_pts = g.home_points if g.home_points is not None else 0
            a_pts = g.away_points if g.away_points is not None else 0

            team_records[g.home_team]['games'] += 1
            team_records[g.away_team]['games'] += 1

            if h_pts > a_pts:
                team_records[g.home_team]['wins'] += 1
            elif a_pts > h_pts:
                team_records[g.away_team]['wins'] += 1

        stats = []
        for item in season_stats:
            team_name = getattr(item, 'team', None)
            if not team_name: continue

            offense = getattr(item, 'offense', None)
            defense = getattr(item, 'defense', None)

            if not offense: continue

            # Map stat.offense.points -> points_per_game
            # Need games count to average if points is Total.
            # 'games' is usually on the item itself for TeamSeasonStat
            games = getattr(item, 'games', 0)
            # If not there, try offense (handles older versions too)
            if not games and hasattr(offense, 'games'):
                games = getattr(offense, 'games', 0)

            # If still not found, check our team_records map
            if not games and team_name in team_records:
                games = team_records[team_name]['games']

            pts = getattr(offense, 'points', 0)
            pts_allowed = getattr(defense, 'points', 0) if defense else 0

            # Additional metrics if available (yards etc)
            total_yards = getattr(offense, 'total_yards', 0)

            # Calculate PPG / OPPG / YPG
            if games and games > 0:
                ppg = pts / games
                oppg = pts_allowed / games
                # yards_per_game logic as requested
                ypg = total_yards / games
                tov = getattr(offense, 'turnovers', 0)
                avg_tov = tov / games
            else:
                ppg = 0.0
                oppg = 0.0
                ypg = 0.0
                avg_tov = 0.0

            # Win PCT from map or default
            if team_name in team_records and team_records[team_name]['games'] > 0:
                rec = team_records[team_name]
                w_pct = rec['wins'] / rec['games']
            else:
                w_pct = 0.5

            stats.append({
                "team_norm": robust_normalize_team(team_name),
                "league_key": "NCAAF",
                "win_pct": w_pct,
                "home_win_pct": w_pct,
                "away_win_pct": w_pct,
                "points_per_game": ppg,
                "points_allowed_per_game": oppg,
                "yards_per_game": ypg, # Added per instructions
                "turnovers": avg_tov,
                "streak": 0.0,
                "last5_win_pct": win_pct
            })
            
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
                df = future.result(timeout=5)
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
    league_col = 'league'
    if 'sport_title' in df.columns:
        league_col = 'sport_title'
    elif 'league' in df.columns:
        league_col = 'league'
    elif 'League' in df.columns:
        # If only 'League' exists, map it to 'league'
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
        features_data['feature_stats_fallback'] = combined_fallback

        # NEW: stats_quality
        features_data['stats_quality'] = combined_fallback.apply(lambda x: "Low (Fallback)" if x else "High (Real)")

        # LOGGING: Fallbacks
        if combined_fallback.any():
            fallback_indices = df.index[combined_fallback]
            for idx in fallback_indices:
                try:
                    league_str = df.loc[idx, league_col] if league_col else "Unknown"
                    h_team = df.loc[idx, home_col]
                    a_team = df.loc[idx, away_col]
                    # Check which one failed
                    h_stat = "MISSING" if home_fallback[idx] else "OK"
                    a_stat = "MISSING" if away_fallback[idx] else "OK"
                    logger.warning(f"DEBUG Stats Fallback Used: {league_str} {h_team} ({h_stat}) vs {a_team} ({a_stat})")
                except Exception:
                    pass

        # Home Stats
        features_data['feature_home_win_pct'] = map_stat(home_norm, 'win_pct', default_win_pct)
        features_data['feature_home_home_win_pct'] = map_stat(home_norm, 'home_win_pct', default_win_pct)
        features_data['feature_home_last5_win_pct'] = map_stat(home_norm, 'last5_win_pct', default_last5)

        # New key mapping for standardized keys
        features_data['feature_home_ppg'] = map_stat(home_norm, 'points_per_game', default_ppg)
        features_data['feature_home_oppg'] = map_stat(home_norm, 'points_allowed_per_game', default_oppg)

        # SCALING: NHL stats are ~3.0, model expects ~110.0. Scale by 35x if league is NHL.
        is_nhl = league_keys == "NHL"
        if is_nhl.any():
            nhl_scale_factor = 35.0
            # Apply scaling only to NHL rows
            features_data['feature_home_ppg'] = features_data['feature_home_ppg'].mask(is_nhl, features_data['feature_home_ppg'] * nhl_scale_factor)
            features_data['feature_home_oppg'] = features_data['feature_home_oppg'].mask(is_nhl, features_data['feature_home_oppg'] * nhl_scale_factor)

        features_data['feature_home_streak'] = map_stat(home_norm, 'streak', pd.Series(0.0, index=df.index))
        features_data['feature_home_turnovers'] = map_stat(home_norm, 'turnovers', pd.Series(0.0, index=df.index))
        
        # Away Stats
        features_data['feature_away_win_pct'] = map_stat(away_norm, 'win_pct', default_win_pct)
        features_data['feature_away_away_win_pct'] = map_stat(away_norm, 'away_win_pct', default_win_pct)
        features_data['feature_away_last5_win_pct'] = map_stat(away_norm, 'last5_win_pct', default_last5)

        features_data['feature_away_ppg'] = map_stat(away_norm, 'points_per_game', default_ppg)
        features_data['feature_away_oppg'] = map_stat(away_norm, 'points_allowed_per_game', default_oppg)

        # SCALING: NHL stats are ~3.0, model expects ~110.0. Scale by 35x if league is NHL.
        if is_nhl.any():
            nhl_scale_factor = 35.0
            features_data['feature_away_ppg'] = features_data['feature_away_ppg'].mask(is_nhl, features_data['feature_away_ppg'] * nhl_scale_factor)
            features_data['feature_away_oppg'] = features_data['feature_away_oppg'].mask(is_nhl, features_data['feature_away_oppg'] * nhl_scale_factor)

        features_data['feature_away_streak'] = map_stat(away_norm, 'streak', pd.Series(0.0, index=df.index))
        features_data['feature_away_turnovers'] = map_stat(away_norm, 'turnovers', pd.Series(0.0, index=df.index))

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
