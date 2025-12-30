import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging
import warnings

# Import new open-source libraries
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

from app_core.team_name_matcher import TeamNameMatcher
from app_core.vertex_ai_endpoint import VERTEX_FEATURE_COLUMNS

logger = logging.getLogger(__name__)

# Config: Set to True to skip stats API calls on Free Tier plans
# (This still applies to API-Sports calls, but NOT to open-source libs)
FREE_TIER_MODE = True

# Define the 27 features we want to ensure exist
# We will update VERTEX_FEATURE_COLUMNS in vertex_ai_endpoint.py to match these.
LEAGUE_AVERAGES = {
    "NBA": {"ppg": 114.0, "oppg": 114.0, "win_pct": 0.5, "last5_win_pct": 0.5},
    "NFL": {"ppg": 22.0, "oppg": 22.0, "win_pct": 0.5, "last5_win_pct": 0.5},
    "NHL": {"ppg": 3.0, "oppg": 3.0, "win_pct": 0.5, "last5_win_pct": 0.5},
    "NCAAB": {"ppg": 72.0, "oppg": 72.0, "win_pct": 0.5, "last5_win_pct": 0.5},
    "NCAAF": {"ppg": 28.0, "oppg": 28.0, "win_pct": 0.5, "last5_win_pct": 0.5},
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

        # MeasureType='Base' gives GP, W, L, W_PCT, PTS, PLUS_MINUS, etc.
        dashboard = leaguedashteamstats.LeagueDashTeamStats(
            season=season_str,
            measure_type_detailed_defense='Base'
        )
        df = dashboard.get_data_frames()[0]

        stats = []
        for _, row in df.iterrows():
            # nba_api columns: TEAM_NAME, GP, W, L, W_PCT, PTS, PLUS_MINUS
            team_name = str(row['TEAM_NAME'])
            gp = float(row['GP'])
            pts = float(row['PTS'])
            plus_minus = float(row['PLUS_MINUS'])
            w_pct = float(row['W_PCT'])

            # Calculate metrics
            ppg = pts / gp if gp > 0 else 0.0
            # Opponent PTS approx: PTS - PLUS_MINUS = OPP_PTS
            oppg = (pts - plus_minus) / gp if gp > 0 else 0.0

            # For streaks/splits we might need more calls, but for speed we use overall as proxy
            stats.append({
                "team_norm": TeamNameMatcher.normalize(team_name),
                "league_key": "NBA",
                "win_pct": w_pct,
                "home_win_pct": w_pct, # Approximation
                "away_win_pct": w_pct, # Approximation
                "ppg": ppg,
                "oppg": oppg,
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

        def update_team(team, scored, allowed, won):
            if team not in team_stats:
                team_stats[team] = {'games': 0, 'wins': 0, 'points_for': 0, 'points_against': 0}
            team_stats[team]['games'] += 1
            team_stats[team]['points_for'] += scored
            team_stats[team]['points_against'] += allowed
            if won:
                team_stats[team]['wins'] += 1

        for _, row in df.iterrows():
            if pd.isna(row['result']): # Game not played yet
                continue

            home = row['home_team']
            away = row['away_team']
            home_score = row['home_score']
            away_score = row['away_score']

            # Handle potential NaNs in scores
            if pd.isna(home_score) or pd.isna(away_score):
                continue

            update_team(home, home_score, away_score, home_score > away_score)
            update_team(away, away_score, home_score, away_score > home_score)

        stats = []
        for team_code, data in team_stats.items():
            games = data['games']
            if games == 0: continue

            w_pct = data['wins'] / games
            ppg = data['points_for'] / games
            oppg = data['points_against'] / games

            # nfl_data_py uses abbreviations (e.g. KC, SF).
            # TeamNameMatcher.normalize might need help if it expects full names.
            # But we will pass it anyway.
            # For robustness, we might want a small map or trust fuzzy match later.
            # Usually 'KC' -> 'Kansas City' mapping exists in some layers,
            # but TeamNameMatcher is primarily for full names.
            # However, downstream matching might be fuzzy enough.

            stats.append({
                "team_norm": TeamNameMatcher.normalize(str(team_code)),
                "league_key": "NFL",
                "win_pct": w_pct,
                "home_win_pct": w_pct,
                "away_win_pct": w_pct,
                "ppg": ppg,
                "oppg": oppg,
                "streak": 0.0,
                "last5_win_pct": w_pct
            })

        logger.info(f"Successfully fetched NFL stats for {len(stats)} teams.")
        return stats
    except Exception as e:
        logger.warning(f"Failed to fetch NFL stats via nfl_data_py: {e}")
        return []

# -------------------------------------------------------------------------

def fetch_and_process_standings(api_clients: Dict[str, Any], season_year: Optional[int] = None) -> pd.DataFrame:
    """
    Fetch standings for all configured leagues and return a normalized stats DataFrame.
    Returns DataFrame with columns: ['team_norm', 'league_key', ...stats...]

    Updated to use open-source libraries for NBA and NFL.
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

    # Track which leagues we've handled to avoid double fetching
    handled_leagues = set()

    # 1. Open Source Fetching (NBA/NFL) - Bypass Free Tier Mode checks
    if "NBA" in leagues:
        nba_data = fetch_nba_stats(season_year)
        if nba_data:
            all_stats.extend(nba_data)
            handled_leagues.add("NBA")

    if "NFL" in leagues:
        nfl_data = fetch_nfl_stats(season_year)
        if nfl_data:
            all_stats.extend(nfl_data)
            handled_leagues.add("NFL")

    # 2. Existing API-Sports Fetching for others
    for league_key, client in api_clients.items():
        if league_key in handled_leagues:
            continue

        if not client or not client.is_configured():
            continue
            
        # Free Tier Mode: Skip API calls to prevent 403 errors
        if FREE_TIER_MODE:
            continue

        # Use get_team_stats instead of get_standings to avoid 403 errors
        try:
            standings = client.get_team_stats()
        except Exception as e:
            logger.warning(f"Error fetching stats for {league_key}: {e}")
            standings = []

        if not standings:
            status_code = getattr(client, 'last_status_code', 'N/A')
            if not FREE_TIER_MODE:
                logger.warning(f"Failed to fetch team stats for {league_key}: {getattr(client, 'last_error', 'Unknown error')} (Status: {status_code})")
            continue
            
        for team_entry in standings:
            # Extract basic info
            team_info = team_entry.get("team") or {}
            stats_all = team_entry.get("all") or {}
            stats_home = team_entry.get("home") or {}
            stats_away = team_entry.get("away") or {}
            
            raw_name = team_info.get("name")
            if not raw_name:
                continue
                
            norm_name = TeamNameMatcher.normalize(raw_name)
            
            # Helper to safely get float
            def get_val(d, k, sub_k=None):
                try:
                    if sub_k:
                        return float((d.get(k) or {}).get(sub_k) or 0.0)
                    return float(d.get(k) or 0.0)
                except:
                    return 0.0

            # Games played
            played = get_val(stats_all, "played")
            played_home = get_val(stats_home, "played")
            played_away = get_val(stats_away, "played")
            
            # Win Pcts
            win_count = get_val(stats_all, "win")
            win_home = get_val(stats_home, "win")
            win_away = get_val(stats_away, "win")
            
            win_pct = win_count / played if played > 0 else 0.5
            home_win_pct = win_home / played_home if played_home > 0 else 0.5
            away_win_pct = win_away / played_away if played_away > 0 else 0.5
            
            # PPG / OPPG
            goals = stats_all.get("goals") or stats_all.get("points") or {}
            points_for = get_val(goals, "for")
            points_against = get_val(goals, "against")
            
            ppg = points_for / played if played > 0 else 0.0
            oppg = points_against / played if played > 0 else 0.0
            
            # Streak/Form
            streak = _parse_streak(team_entry.get("streak") or "")
            last5_win_pct = _parse_form(team_entry.get("form") or "")
            
            all_stats.append({
                "team_norm": norm_name,
                "league_key": league_key,
                "win_pct": win_pct,
                "home_win_pct": home_win_pct,
                "away_win_pct": away_win_pct,
                "ppg": ppg,
                "oppg": oppg,
                "streak": streak,
                "last5_win_pct": last5_win_pct,
            })
            
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

    # 1. Fetch Stats
    stats_df = fetch_and_process_standings(api_clients, season_year=season_year)
    
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
        
        # Helper to map a stat column efficiently
        def map_stat(norm_series, col_name, default_val):
            # Using map against the series from the indexed dataframe
            return norm_series.map(stats_unique[col_name]).fillna(default_val)

        # Populate features_data
        # Home Stats
        features_data['feature_home_win_pct'] = map_stat(home_norm, 'win_pct', defaults['win_pct'])
        features_data['feature_home_home_win_pct'] = map_stat(home_norm, 'home_win_pct', defaults['win_pct'])
        features_data['feature_home_last5_win_pct'] = map_stat(home_norm, 'last5_win_pct', defaults['last5_win_pct'])
        features_data['feature_home_ppg'] = map_stat(home_norm, 'ppg', defaults['ppg'])
        features_data['feature_home_oppg'] = map_stat(home_norm, 'oppg', defaults['oppg'])
        features_data['feature_home_streak'] = map_stat(home_norm, 'streak', 0.0)
        
        # Away Stats
        features_data['feature_away_win_pct'] = map_stat(away_norm, 'win_pct', defaults['win_pct'])
        features_data['feature_away_away_win_pct'] = map_stat(away_norm, 'away_win_pct', defaults['win_pct'])
        features_data['feature_away_last5_win_pct'] = map_stat(away_norm, 'last5_win_pct', defaults['last5_win_pct'])
        features_data['feature_away_ppg'] = map_stat(away_norm, 'ppg', defaults['ppg'])
        features_data['feature_away_oppg'] = map_stat(away_norm, 'oppg', defaults['oppg'])
        features_data['feature_away_streak'] = map_stat(away_norm, 'streak', 0.0)

    # 4. Fill Defaults if stats_df was empty or map failed (though fillna handles map fail)
    # If keys don't exist (because stats_df was empty), create them
    if 'feature_home_win_pct' not in features_data:
        features_data['feature_home_win_pct'] = defaults['win_pct']
        features_data['feature_home_home_win_pct'] = defaults['win_pct']
        features_data['feature_home_last5_win_pct'] = defaults['last5_win_pct']
        features_data['feature_home_ppg'] = defaults['ppg']
        features_data['feature_home_oppg'] = defaults['oppg']
        features_data['feature_home_streak'] = 0.0
        
        features_data['feature_away_win_pct'] = defaults['win_pct']
        features_data['feature_away_away_win_pct'] = defaults['win_pct']
        features_data['feature_away_last5_win_pct'] = defaults['last5_win_pct']
        features_data['feature_away_ppg'] = defaults['ppg']
        features_data['feature_away_oppg'] = defaults['oppg']
        features_data['feature_away_streak'] = 0.0
        
        if not FREE_TIER_MODE:
            logger.warning(f"Used fallback league averages ({league_key}) for ALL games (stats fetch failed)!")

    # 5. Compute Differentials (Vectorized)
    # We can use numpy subtraction on the Series/arrays/scalars in features_data
    # Note: features_data values might be Series or Scalars. 
    # If Series, arithmetic works. If scalar, arithmetic works.
    
    features_data['feature_diff_win_pct'] = features_data['feature_home_win_pct'] - features_data['feature_away_win_pct']
    features_data['feature_diff_ppg'] = features_data['feature_home_ppg'] - features_data['feature_away_ppg']
    features_data['feature_diff_oppg'] = features_data['feature_home_oppg'] - features_data['feature_away_oppg']
    features_data['feature_diff_last5'] = features_data['feature_home_last5_win_pct'] - features_data['feature_away_last5_win_pct']
    features_data['feature_diff_streak'] = features_data['feature_home_streak'] - features_data['feature_away_streak']
    
    # 6. Map Remaining Features (Existing) using safe_numeric_fill
    features_data['implied_home_prob'] = safe_numeric_fill(df.get('Implied_Prob'), 0.5)
    
    # Try to refine implied prob from ML if available
    def ml_to_prob(ml):
        try:
            m = float(ml)
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
