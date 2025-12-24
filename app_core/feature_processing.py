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
TARGET_FEATURE_COLUMNS = [
    "implied_home_prob",
    "sentiment_diff",
    "kalshi_prob",
    "injuries_home_count",
    "injuries_away_count",
    "weather_flag",
    "home_win_pct",
    "home_home_win_pct",
    "home_last5_win_pct",
    "home_ppg",
    "home_oppg",
    "home_streak",
    "away_win_pct",
    "away_away_win_pct",
    "away_last5_win_pct",
    "away_ppg",
    "away_oppg",
    "away_streak",
    "diff_win_pct",
    "diff_ppg",
    "diff_oppg",
    "diff_last5",
    "diff_streak",
    "commence_hour",
    "commence_day_of_week",
    "home_rest_days",
    "away_rest_days",
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
            win_pct = get_val(stats_all, "win") / played if played > 0 else 0.5
            # API-Sports sometimes gives 'win' as count, sometimes as pct? 
            # Usually 'win' is count. 'goals' object has 'for'/'against'.
            
            # Actually, standard structure: { "all": { "played": 82, "win": 50, ... } }
            win_count = get_val(stats_all, "win")
            win_home = get_val(stats_home, "win")
            win_away = get_val(stats_away, "win")
            
            win_pct = win_count / played if played > 0 else 0.5
            home_win_pct = win_home / played_home if played_home > 0 else 0.5
            away_win_pct = win_away / played_away if played_away > 0 else 0.5
            
            # PPG / OPPG
            # For NBA/NFL, use "points" or "goals" depending on sport config.
            # Client usually sets SCORING_METRIC_LABEL.
            # But get_standings payload usually has "goals" or "points".
            # Let's check both or generic.
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
    Uses pd.merge for performance.
    """
    if df is None or df.empty:
        return df
        
    # 1. Fetch Stats
    stats_df = fetch_and_process_standings(api_clients)
    
    if stats_df.empty:
        logger.warning("No stats fetched. Filling with defaults.")
        # We still need to populate columns to avoid Vertex errors
        for col in TARGET_FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0
        return df
    
    # 2. Normalize Names in Master DF
    df['home_norm'] = df['Home'].apply(lambda x: TeamNameMatcher.normalize(str(x)))
    df['away_norm'] = df['Away'].apply(lambda x: TeamNameMatcher.normalize(str(x)))
    
    # 3. Merge Stats
    # We allow cross-league merging if needed, but ideally filter by league. 
    # Since team names are fairly unique within sport context, merging by normalized name should be safe enough 
    # provided we don't have collisions. To be safe, we can merge on [league, team].
    # But master_df league names (e.g. "NBA") might match api_client keys.
    # Let's try merging on team_norm only first, dropping duplicates in stats_df if any.
    
    stats_unique = stats_df.drop_duplicates(subset=['team_norm'])
    
    # Merge Home
    df = df.merge(
        stats_unique, 
        left_on='home_norm', 
        right_on='team_norm', 
        how='left',
        suffixes=('', '_home')
    )
    # Rename merged columns to feature names
    home_cols = {
        'win_pct': 'home_win_pct',
        'home_win_pct': 'home_home_win_pct', # Win % at home
        'away_win_pct': 'home_away_win_pct', # Not used directly?
        'ppg': 'home_ppg',
        'oppg': 'home_oppg',
        'streak': 'home_streak',
        'last5_win_pct': 'home_last5_win_pct'
    }
    df.rename(columns=home_cols, inplace=True)
    
    # Merge Away
    df = df.merge(
        stats_unique, 
        left_on='away_norm', 
        right_on='team_norm', 
        how='left',
        suffixes=('', '_away')
    )
    # Rename merged columns (pandas adds suffix to colliding, but we renamed home ones already)
    # Wait, 'win_pct' became 'home_win_pct'. The new merge brings 'win_pct' again (from stats_unique).
    # So we can just rename the new columns.
    away_cols = {
        'win_pct': 'away_win_pct',
        'home_win_pct': 'away_home_win_pct',
        'away_win_pct': 'away_away_win_pct', # Win % at away
        'ppg': 'away_ppg',
        'oppg': 'away_oppg',
        'streak': 'away_streak',
        'last5_win_pct': 'away_last5_win_pct'
    }
    df.rename(columns=away_cols, inplace=True)
    
    # 4. Fill Missing with League Averages
    # We calculate averages from stats_df
    avg_stats = stats_df.select_dtypes(include=[np.number]).mean().to_dict()
    
    fill_map = {
        'home_win_pct': avg_stats.get('win_pct', 0.5),
        'home_home_win_pct': avg_stats.get('home_win_pct', 0.5),
        'home_last5_win_pct': avg_stats.get('last5_win_pct', 0.5),
        'home_ppg': avg_stats.get('ppg', 0.0),
        'home_oppg': avg_stats.get('oppg', 0.0),
        'home_streak': 0.0,
        'away_win_pct': avg_stats.get('win_pct', 0.5),
        'away_away_win_pct': avg_stats.get('away_win_pct', 0.5),
        'away_last5_win_pct': avg_stats.get('last5_win_pct', 0.5),
        'away_ppg': avg_stats.get('ppg', 0.0),
        'away_oppg': avg_stats.get('oppg', 0.0),
        'away_streak': 0.0,
    }
    
    for col, val in fill_map.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
        else:
            df[col] = val

    # 5. Compute Differentials
    df['diff_win_pct'] = df['home_win_pct'] - df['away_win_pct']
    df['diff_ppg'] = df['home_ppg'] - df['away_ppg']
    df['diff_oppg'] = df['home_oppg'] - df['away_oppg']
    df['diff_last5'] = df['home_last5_win_pct'] - df['away_last5_win_pct']
    df['diff_streak'] = df['home_streak'] - df['away_streak']
    
    # 6. Map Remaining Features (Existing)
    # Ensure they map to the exact names in TARGET_FEATURE_COLUMNS
    # "implied_home_prob" is usually "Implied_Prob" (or calculated if missing)
    # We need to ensure we populate them.
    
    df['implied_home_prob'] = pd.to_numeric(df.get('Implied_Prob'), errors='coerce').fillna(0.5)
    # If Implied_Prob is for Away team, this might be wrong if we just take 'Implied_Prob'.
    # Master DF usually has 'Implied_Prob' for the *selected pick*. 
    # We want 'Implied Home Probability'.
    # We can approximate from 'Home_ML' if available.
    
    def ml_to_prob(ml):
        try:
            m = float(ml)
            if m > 0: return 100/(m+100)
            return abs(m)/(abs(m)+100)
        except:
            return 0.5
            
    if 'Home_ML' in df.columns:
        df['implied_home_prob'] = df['Home_ML'].apply(ml_to_prob)
        
    df['sentiment_diff'] = pd.to_numeric(df.get('Sentiment_Diff'), errors='coerce').fillna(0.0)
    df['kalshi_prob'] = pd.to_numeric(df.get('kalshi_prob'), errors='coerce').fillna(0.5)
    df['injuries_home_count'] = pd.to_numeric(df.get('injuries_home_count'), errors='coerce').fillna(0)
    df['injuries_away_count'] = pd.to_numeric(df.get('injuries_away_count'), errors='coerce').fillna(0)
    
    # Weather flag
    def parse_weather(w):
        if not w or not isinstance(w, str): return 0.0
        w = w.lower()
        if 'rain' in w or 'snow' in w or 'wind' in w: return 1.0
        return 0.0
    
    if 'weather_summary' in df.columns:
        df['weather_flag'] = df['weather_summary'].apply(parse_weather)
    else:
        df['weather_flag'] = 0.0
        
    # Time features
    if 'Commence (UTC)' in df.columns:
        dt_series = pd.to_datetime(df['Commence (UTC)'], errors='coerce')
        df['commence_hour'] = dt_series.dt.hour.fillna(19.0) # Default 7 PM
        df['commence_day_of_week'] = dt_series.dt.dayofweek.fillna(6.0) # Default Sunday
    else:
        df['commence_hour'] = 19.0
        df['commence_day_of_week'] = 6.0
        
    # Rest Days (Placeholder 3.0 unless we implement calendar logic)
    df['home_rest_days'] = 3.0
    df['away_rest_days'] = 3.0
    
    # Cleanup auxiliary columns
    # We don't drop them, just ensure TARGET_FEATURE_COLUMNS exist
    for col in TARGET_FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
            
    return df
