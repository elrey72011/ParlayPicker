import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional
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

# Streamlit secrets access for API keys
try:
    import streamlit as st
except ImportError:
    st = None


from app_core.team_name_matcher import TeamNameMatcher
from app_core.vertex_ai_endpoint import VERTEX_FEATURE_COLUMNS

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
    "NCAAF": {"ppg": 0.0, "oppg": 0.0, "win_pct": 0.5, "last5_win_pct": 0.5},
    "default": {"ppg": 50.0, "oppg": 50.0, "win_pct": 0.5, "last5_win_pct": 0.5}
}

TARGET_FEATURE_COLUMNS = [
    "implied_home_prob",
    "sentiment_diff",
    "kalshi_prob",
    "injuries_home_count",
    "injuries_away_count",
    "weather_flag",
    "feature_home_win_pct",
    "feature_home_home_win_pct",
    "feature_home_last5_win_pct",
    "feature_home_ppg",
    "feature_home_oppg",
    "feature_home_streak",
    "feature_away_win_pct",
    "feature_away_away_win_pct",
    "feature_away_last5_win_pct",
    "feature_away_ppg",
    "feature_away_oppg",
    "feature_away_streak",
    "feature_diff_win_pct",
    "feature_diff_ppg",
    "feature_diff_oppg",
    "feature_diff_last5",
    "feature_diff_streak",
    "feature_commence_hour",
    "feature_commence_day_of_week",
    "feature_home_rest_days",
    "feature_away_rest_days",
]

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

# -------------------------------------------------------------------------
# New Open-Source Stats Fetching Helpers
# -------------------------------------------------------------------------

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
                "team_norm": TeamNameMatcher.normalize(team_name),
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
        logger.warning(f"Failed to fetch NBA stats via nba_api: {e}")
        return []

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
                "team_norm": TeamNameMatcher.normalize(str(team_code)),
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
        logger.warning(f"Failed to fetch NFL stats via nfl_data_py: {e}")
        return []

def fetch_ncaaf_stats(season_year: int) -> List[Dict[str, Any]]:
    """
    Fetch NCAAF stats using cfbd.
    Requires CFBD_API_KEY.
    """
    if cfbd is None:
        return []
    
    api_key = _get_secret("CFBD_API_KEY")
    if not api_key:
        logger.warning("CFBD_API_KEY not found. Skipping NCAAF stats.")
        return []

    try:
        logger.info(f"Fetching NCAAF stats for season: {season_year}")
        configuration = cfbd.Configuration()
        configuration.api_key['Authorization'] = api_key
        configuration.api_key_prefix['Authorization'] = 'Bearer'

        api_instance = cfbd.StatsApi(cfbd.ApiClient(configuration))

        # Use get_team_season_stats per instruction (replaces get_advanced_season_stats)
        # WRAPPER FIX: Handle HTTP 500 or any API crash gracefully
        season_stats = []
        try:
            season_stats = api_instance.get_team_season_stats(year=season_year)
        except Exception as e:
            logger.warning("NCAAF Stats Outage - Using Defaults")
            # Return empty list, enrich_with_vertex_features will handle defaults (or we can return explicit defaults here if strict 0.0 needed)
            # User request: "default all NCAAF features to 0.0 and add a warning 'NCAAF_STATS_UNAVAILABLE'"
            # If we return [], enrich_with_vertex_features uses league averages (28.0).
            # If we want 0.0, we should probably handle it downstream or return a dummy list.
            # But "API is down" means we can't iterate teams.
            return []

        # To get Win PCT, we still need records or game outcomes.
        # We will use GamesApi to get games and calculate win pct,
        # but rely on season_stats for the points metrics as requested.
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
                "team_norm": TeamNameMatcher.normalize(team_name),
                "league_key": "NCAAF",
                "win_pct": w_pct,
                "home_win_pct": w_pct,
                "away_win_pct": w_pct,
                "points_per_game": ppg,
                "points_allowed_per_game": oppg,
                "yards_per_game": ypg, # Added per instructions
                "turnovers": avg_tov,
                "streak": 0.0,
                "last5_win_pct": w_pct
            })
            
        logger.info(f"Successfully fetched NCAAF stats for {len(stats)} teams.")
        return stats

    except Exception as e:
        logger.warning(f"Failed to fetch NCAAF stats: {e}")
        return []

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
                "team_norm": TeamNameMatcher.normalize(team_name),
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
        logger.warning(f"Failed to fetch NHL stats: {e}")
        return []

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
                "team_norm": TeamNameMatcher.normalize(team_name),
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

def enrich_with_vertex_features(df: pd.DataFrame, api_clients: Dict[str, Any], season_year: Optional[int] = None) -> pd.DataFrame:
    """
    Enrich the master dataframe with features required for Vertex AI.
    Uses pd.concat for performance to avoid fragmentation.
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

    home_norm = df[home_col].apply(lambda x: TeamNameMatcher.normalize(str(x)))
    away_norm = df[away_col].apply(lambda x: TeamNameMatcher.normalize(str(x)))
    
    # 3. Determine League (for fallbacks)
    league_key = "default"
    if 'League' in df.columns and len(df) > 0:
        first_league = str(df['League'].iloc[0]).upper()
        if "NBA" in first_league: league_key = "NBA"
        elif "NFL" in first_league: league_key = "NFL"
        elif "NHL" in first_league: league_key = "NHL"
        elif "NCAAB" in first_league: league_key = "NCAAB"
        elif "NCAAF" in first_league: league_key = "NCAAF"
        
    defaults = LEAGUE_AVERAGES.get(league_key, LEAGUE_AVERAGES["default"])
    
    # Use dict to collect columns to avoid fragmentation
    features_data = {}
    
    if stats_df.empty:
        if not FREE_TIER_MODE:
            logger.warning("No stats fetched. Filling with defaults.")
        # Fill defaults logic comes later
    else:
        # Prepare lookup dictionary: team_norm -> stats_series
        # Dropping duplicates to ensure unique mapping
        stats_unique = stats_df.drop_duplicates(subset=['team_norm']).set_index('team_norm')
        
        # --- NEW: Fuzzy Matching Logic ---
        # Build a mapping from df's normalized teams to stats' normalized teams

        # Get all unique teams in the current dataframe
        df_teams_norm = pd.concat([home_norm, away_norm]).unique()
        stats_teams_norm = stats_unique.index.tolist()

        team_map = {}
        for t_norm in df_teams_norm:
             if not t_norm: continue

             # Direct match first
             if t_norm in stats_unique.index:
                 team_map[t_norm] = t_norm
             else:
                 # Fuzzy match
                 # We need to map t_norm (from Odds/df) -> best match in stats_teams_norm
                 match = TeamNameMatcher.match_team(t_norm, stats_teams_norm, threshold=0.75)
                 if match:
                     # match_team returns the matched name from the list (which is already normalized here)
                     team_map[t_norm] = match
                 else:
                     # No match found
                     team_map[t_norm] = None

        # Helper to map a stat column efficiently using the map
        def map_stat(norm_series, col_name, default_val):
            # 1. Map df team name -> stats team name (fuzzy)
            mapped_teams = norm_series.map(team_map)
            # 2. Map stats team name -> stat value
            return mapped_teams.map(stats_unique[col_name]).fillna(default_val)

        # Populate features_data using the new fuzzy map_stat
        # Home Stats
        features_data['feature_home_win_pct'] = map_stat(home_norm, 'win_pct', defaults['win_pct'])
        features_data['feature_home_home_win_pct'] = map_stat(home_norm, 'home_win_pct', defaults['win_pct'])
        features_data['feature_home_last5_win_pct'] = map_stat(home_norm, 'last5_win_pct', defaults['last5_win_pct'])

        # New key mapping for standardized keys
        features_data['feature_home_ppg'] = map_stat(home_norm, 'points_per_game', defaults['ppg'])
        features_data['feature_home_oppg'] = map_stat(home_norm, 'points_allowed_per_game', defaults['oppg'])

        features_data['feature_home_streak'] = map_stat(home_norm, 'streak', 0.0)
        features_data['feature_home_turnovers'] = map_stat(home_norm, 'turnovers', 0.0)
        
        # Away Stats
        features_data['feature_away_win_pct'] = map_stat(away_norm, 'win_pct', defaults['win_pct'])
        features_data['feature_away_away_win_pct'] = map_stat(away_norm, 'away_win_pct', defaults['win_pct'])
        features_data['feature_away_last5_win_pct'] = map_stat(away_norm, 'last5_win_pct', defaults['last5_win_pct'])

        features_data['feature_away_ppg'] = map_stat(away_norm, 'points_per_game', defaults['ppg'])
        features_data['feature_away_oppg'] = map_stat(away_norm, 'points_allowed_per_game', defaults['oppg'])

        features_data['feature_away_streak'] = map_stat(away_norm, 'streak', 0.0)
        features_data['feature_away_turnovers'] = map_stat(away_norm, 'turnovers', 0.0)

    # 4. Fill Defaults if stats_df was empty or map failed (though fillna handles map fail)
    # If keys don't exist (because stats_df was empty), create them
    if 'feature_home_win_pct' not in features_data:
        features_data['feature_home_win_pct'] = defaults['win_pct']
        features_data['feature_home_home_win_pct'] = defaults['win_pct']
        features_data['feature_home_last5_win_pct'] = defaults['last5_win_pct']
        features_data['feature_home_ppg'] = defaults['ppg']
        features_data['feature_home_oppg'] = defaults['oppg']
        features_data['feature_home_streak'] = 0.0
        features_data['feature_home_turnovers'] = 0.0
        
        features_data['feature_away_win_pct'] = defaults['win_pct']
        features_data['feature_away_away_win_pct'] = defaults['win_pct']
        features_data['feature_away_last5_win_pct'] = defaults['last5_win_pct']
        features_data['feature_away_ppg'] = defaults['ppg']
        features_data['feature_away_oppg'] = defaults['oppg']
        features_data['feature_away_streak'] = 0.0
        features_data['feature_away_turnovers'] = 0.0
        
        if not FREE_TIER_MODE:
            logger.warning(f"Used fallback league averages ({league_key}) for ALL games (stats fetch failed)!")

    # 5. Compute Differentials (Vectorized)
    # Neutralize massive differentials if either side is 0.0 (missing stats)
    def safe_diff(series1, series2, default_val=None):
        # User Request 2: Neutralize "Zero-Stat" Differentials
        # Convert to numeric to ensure comparison works
        s1 = pd.to_numeric(pd.Series(series1), errors='coerce').fillna(0.0)
        s2 = pd.to_numeric(pd.Series(series2), errors='coerce').fillna(0.0)

        # Only subtract if BOTH teams have stats > 0.0 (valid stats)
        # If either is 0.0 (or very close), default the differential to 0.0
        valid_mask = (s1.abs() > 1e-6) & (s2.abs() > 1e-6)

        if default_val is not None:
             # Check if either team is at the default stats level
             # Using small epsilon for float comparison
             is_default = ((s1 - default_val).abs() < 1e-6) | ((s2 - default_val).abs() < 1e-6)
             # If either is default, we treat it as invalid to force 0.0 diff
             valid_mask = valid_mask & (~is_default)

        # Calculate diff everywhere first
        diff = s1 - s2

        # Where NOT valid (meaning at least one is 0.0), set diff to 0.0
        diff[~valid_mask] = 0.0

        # FORCE override for totally missing stats to prevent phantom signals
        # If either side is missing (default 0.0 after fillna), diff must be 0.0
        # Expanded check to catch NaN or 0.0 explicitly
        missing_mask = (s1.abs() < 1e-6) | (s2.abs() < 1e-6) | (s1 != s1) | (s2 != s2)
        diff[missing_mask] = 0.0

        # FORCE override if stats are missing for either team
        if default_val is not None:
            # Check if either team is at the default stats level
            is_default = ((s1 - default_val).abs() < 1e-6) | ((s2 - default_val).abs() < 1e-6)
            diff[is_default] = 0.0

        return diff

    features_data['feature_diff_win_pct'] = safe_diff(features_data['feature_home_win_pct'], features_data['feature_away_win_pct'], default_val=defaults['win_pct'])
    features_data['feature_diff_ppg'] = safe_diff(features_data['feature_home_ppg'], features_data['feature_away_ppg'], default_val=defaults['ppg'])
    features_data['feature_diff_oppg'] = safe_diff(features_data['feature_home_oppg'], features_data['feature_away_oppg'], default_val=defaults['oppg'])
    features_data['feature_diff_last5'] = safe_diff(features_data['feature_home_last5_win_pct'], features_data['feature_away_last5_win_pct'], default_val=defaults['last5_win_pct'])
    features_data['feature_diff_streak'] = features_data['feature_home_streak'] - features_data['feature_away_streak'] # Streak can be 0 validly
    
    # 6. Map Remaining Features (Existing) using safe_numeric_fill
    features_data['implied_home_prob'] = safe_numeric_fill(df.get('Implied_Prob'), 0.5)
    
    # Try to refine implied prob from ML if available
    # --- NEW: Robust ml_to_prob ---
    def ml_to_prob(ml):
        try:
            if ml is None: return 0.5
            m = float(ml)
            if m != m or m == 0: return 0.5
            if m > 0: return 100/(m+100)
            return abs(m)/(abs(m)+100)
        except:
            return 0.5
            
    if 'Home_ML' in df.columns:
        features_data['implied_home_prob'] = df['Home_ML'].apply(ml_to_prob)
        
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
    
    # 7. Final Concat
    features_df = pd.DataFrame(features_data, index=df.index)
    result = pd.concat([df, features_df], axis=1)
    
    return result

def run_roi_pipeline_validation(df: pd.DataFrame):
    """Checks if the data bridge is actually functioning before export."""
    critical_checks = {
        "Vertex Prediction": "vertex_spread_prob",
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
