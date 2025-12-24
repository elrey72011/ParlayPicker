import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

from app_core.team_name_matcher import TeamNameMatcher
from app_core.vertex_ai_endpoint import VERTEX_FEATURE_COLUMNS

logger = logging.getLogger(__name__)

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

def fetch_and_process_standings(api_clients: Dict[str, Any]) -> pd.DataFrame:
    """
    Fetch standings for all configured leagues and return a normalized stats DataFrame.
    Returns DataFrame with columns: ['team_norm', 'league_key', ...stats...]
    """
    all_stats = []
    
    for league_key, client in api_clients.items():
        if not client or not client.is_configured():
            continue
            
        standings = client.get_standings()
        if not standings:
            logger.warning(f"Failed to fetch standings for {league_key}: {getattr(client, 'last_error', 'Unknown error')}")
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
            
            # Success logging as requested
            logger.info(f"✅ Stats loaded for {norm_name}")
            
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
                "home_win_pct": home_win_pct, # Win % when playing at home
                "away_win_pct": away_win_pct, # Win % when playing away
                "ppg": ppg,
                "oppg": oppg,
                "streak": streak,
                "last5_win_pct": last5_win_pct,
            })
            
    return pd.DataFrame(all_stats)

def enrich_with_vertex_features(df: pd.DataFrame, api_clients: Dict[str, Any]) -> pd.DataFrame:
    """
    Enrich the master dataframe with features required for Vertex AI.
    Uses pd.concat for performance to avoid fragmentation.
    """
    if df is None or df.empty:
        return df
        
    # 1. Fetch Stats
    stats_df = fetch_and_process_standings(api_clients)
    
    # Create a features dataframe aligned with df index
    features_df = pd.DataFrame(index=df.index)
    
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
    
    if stats_df.empty:
        logger.warning("No stats fetched. Filling with defaults.")
        # We will populate defaults later in the 'missing' logic
    else:
        # Prepare lookup dictionary: team_norm -> stats_series
        # Dropping duplicates to ensure unique mapping
        stats_unique = stats_df.drop_duplicates(subset=['team_norm']).set_index('team_norm')
        
        # Helper to map a stat column efficiently
        def map_stat(norm_series, col_name, default_val):
            # Using map against the series from the indexed dataframe
            return norm_series.map(stats_unique[col_name]).fillna(default_val)

        # Populate features_df
        # Home Stats
        features_df['feature_home_win_pct'] = map_stat(home_norm, 'win_pct', defaults['win_pct'])
        features_df['feature_home_home_win_pct'] = map_stat(home_norm, 'home_win_pct', defaults['win_pct'])
        features_df['feature_home_last5_win_pct'] = map_stat(home_norm, 'last5_win_pct', defaults['last5_win_pct'])
        features_df['feature_home_ppg'] = map_stat(home_norm, 'ppg', defaults['ppg'])
        features_df['feature_home_oppg'] = map_stat(home_norm, 'oppg', defaults['oppg'])
        features_df['feature_home_streak'] = map_stat(home_norm, 'streak', 0.0)
        
        # Away Stats
        features_df['feature_away_win_pct'] = map_stat(away_norm, 'win_pct', defaults['win_pct'])
        features_df['feature_away_away_win_pct'] = map_stat(away_norm, 'away_win_pct', defaults['win_pct'])
        features_df['feature_away_last5_win_pct'] = map_stat(away_norm, 'last5_win_pct', defaults['last5_win_pct'])
        features_df['feature_away_ppg'] = map_stat(away_norm, 'ppg', defaults['ppg'])
        features_df['feature_away_oppg'] = map_stat(away_norm, 'oppg', defaults['oppg'])
        features_df['feature_away_streak'] = map_stat(away_norm, 'streak', 0.0)

    # 4. Fill Defaults if stats_df was empty or map failed (though fillna handles map fail)
    # If features_df columns don't exist (because stats_df was empty), create them
    if 'feature_home_win_pct' not in features_df.columns:
        features_df['feature_home_win_pct'] = defaults['win_pct']
        features_df['feature_home_home_win_pct'] = defaults['win_pct']
        features_df['feature_home_last5_win_pct'] = defaults['last5_win_pct']
        features_df['feature_home_ppg'] = defaults['ppg']
        features_df['feature_home_oppg'] = defaults['oppg']
        features_df['feature_home_streak'] = 0.0
        
        features_df['feature_away_win_pct'] = defaults['win_pct']
        features_df['feature_away_away_win_pct'] = defaults['win_pct']
        features_df['feature_away_last5_win_pct'] = defaults['last5_win_pct']
        features_df['feature_away_ppg'] = defaults['ppg']
        features_df['feature_away_oppg'] = defaults['oppg']
        features_df['feature_away_streak'] = 0.0
        
        logger.critical(f"Used fallback league averages ({league_key}) for ALL games (stats fetch failed)!")

    # 5. Compute Differentials (Vectorized)
    features_df['feature_diff_win_pct'] = features_df['feature_home_win_pct'] - features_df['feature_away_win_pct']
    features_df['feature_diff_ppg'] = features_df['feature_home_ppg'] - features_df['feature_away_ppg']
    features_df['feature_diff_oppg'] = features_df['feature_home_oppg'] - features_df['feature_away_oppg']
    features_df['feature_diff_last5'] = features_df['feature_home_last5_win_pct'] - features_df['feature_away_last5_win_pct']
    features_df['feature_diff_streak'] = features_df['feature_home_streak'] - features_df['feature_away_streak']
    
    # 6. Map Remaining Features (Existing)
    # Safer logic for Implied_Prob which might be messy/scalar
    raw_implied = df.get('Implied_Prob', 0.5)
    
    # Ensure we get a Series for vectorization, even if input was scalar/list
    if isinstance(raw_implied, pd.Series):
        features_df['implied_home_prob'] = pd.to_numeric(raw_implied, errors='coerce').fillna(0.5)
    elif isinstance(raw_implied, (list, np.ndarray)):
        features_df['implied_home_prob'] = pd.to_numeric(pd.Series(raw_implied), errors='coerce').fillna(0.5).values
    else:
        # Scalar fallback - assign to all rows
        try:
            val = float(raw_implied)
        except:
            val = 0.5
        features_df['implied_home_prob'] = val

    # Try to refine implied prob from ML if available
    def ml_to_prob(ml):
        try:
            m = float(ml)
            if m > 0: return 100/(m+100)
            return abs(m)/(abs(m)+100)
        except:
            return 0.5
            
    if 'Home_ML' in df.columns:
        # Use vectorized apply? Or map.
        features_df['implied_home_prob'] = df['Home_ML'].apply(ml_to_prob)
        
    features_df['sentiment_diff'] = pd.to_numeric(df.get('Sentiment_Diff'), errors='coerce').fillna(0.0)
    features_df['kalshi_prob'] = pd.to_numeric(df.get('kalshi_prob'), errors='coerce').fillna(0.5)
    features_df['injuries_home_count'] = pd.to_numeric(df.get('injuries_home_count'), errors='coerce').fillna(0)
    features_df['injuries_away_count'] = pd.to_numeric(df.get('injuries_away_count'), errors='coerce').fillna(0)
    
    # Weather flag
    if 'weather_summary' in df.columns:
        features_df['weather_flag'] = df['weather_summary'].astype(str).str.lower().apply(
            lambda w: 1.0 if any(x in w for x in ['rain', 'snow', 'wind']) else 0.0
        )
    else:
        features_df['weather_flag'] = 0.0
        
    # Time features
    if 'Commence (UTC)' in df.columns:
        dt_series = pd.to_datetime(df['Commence (UTC)'], errors='coerce')
        features_df['feature_commence_hour'] = dt_series.dt.hour.fillna(19.0)
        features_df['feature_commence_day_of_week'] = dt_series.dt.dayofweek.fillna(6.0)
    else:
        features_df['feature_commence_hour'] = 19.0
        features_df['feature_commence_day_of_week'] = 6.0
        
    # Rest Days
    features_df['feature_home_rest_days'] = 3.0
    features_df['feature_away_rest_days'] = 3.0
    
    # 7. Final Concat
    result = pd.concat([df, features_df], axis=1)
    
    return result

def run_roi_pipeline_validation(df: pd.DataFrame):
    """Checks if the data bridge is actually functioning before export."""
    critical_checks = {
        "Vertex Prediction": "vertex_spread_prob",
        "Kalshi Probability": "kalshi_prob_spread",
        "Team Stats (PPG)": "feature_home_ppg",
        "Team Stats (Win %)": "feature_home_win_pct"
    }
    
    validation_results = {}
    
    for label, col in critical_checks.items():
        if col not in df.columns:
            validation_results[label] = "❌ COLUMN MISSING"
        elif df[col].notnull().sum() == 0:
            validation_results[label] = "⚠️ COLUMN EMPTY (Data not reaching DF)"
        else:
            validation_results[label] = f"✅ OK ({df[col].notnull().sum()} rows populated)"
            
    logger.info("--- ROI PIPELINE VALIDATION ---")
    for label, status in validation_results.items():
        logger.info(f"{label}: {status}")
        
    return validation_results
