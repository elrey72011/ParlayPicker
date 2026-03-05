import pandas as pd
import numpy as np
from datetime import datetime, timezone
import logging
from typing import Optional, Dict, Any, Tuple
from app_core.team_name_matcher import TeamNameMatcher

logger = logging.getLogger(__name__)

def attach_results(master_df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attaches actual game results to the master DataFrame from a results DataFrame.

    Args:
        master_df: DataFrame containing the daily export (one row per game).
        results_df: DataFrame containing final scores (requires 'league', 'home_team', 'away_team', 'home_score', 'away_score', 'date'/'commence').

    Returns:
        DataFrame with new outcome columns attached:
        - actual_home_score, actual_away_score
        - spread_result (WIN, LOSS, PUSH, N/A)
        - total_result (WIN, LOSS, PUSH, N/A)
    """
    if master_df is None or master_df.empty:
        return master_df

    if results_df is None or results_df.empty:
        logger.warning("Empty results_df provided to attach_results.")
        # Return with NaNs/None for outcome columns
        return _add_empty_outcome_columns(master_df.copy())

    df = master_df.copy()

    # Normalize team names in results for matching
    res_df = results_df.copy()

    # Handle different possible column names in results_df
    home_col = next((c for c in ['home_team', 'Home', 'home'] if c in res_df.columns), None)
    away_col = next((c for c in ['away_team', 'Away', 'away'] if c in res_df.columns), None)
    h_score_col = next((c for c in ['home_score', 'HomeScore'] if c in res_df.columns), None)
    a_score_col = next((c for c in ['away_score', 'AwayScore'] if c in res_df.columns), None)
    league_col = next((c for c in ['league', 'League'] if c in res_df.columns), None)
    date_col = next((c for c in ['date', 'commence', 'Commence (UTC)', 'Date'] if c in res_df.columns), None)

    if not all([home_col, away_col, h_score_col, a_score_col, league_col]):
        logger.error(f"results_df missing required columns. Found: {res_df.columns.tolist()}")
        return _add_empty_outcome_columns(df)

    # Pre-compute normalized names
    res_df['_norm_home'] = res_df[home_col].apply(lambda x: TeamNameMatcher.normalize(str(x)) if pd.notnull(x) else "")
    res_df['_norm_away'] = res_df[away_col].apply(lambda x: TeamNameMatcher.normalize(str(x)) if pd.notnull(x) else "")
    res_df['_norm_league'] = res_df[league_col].apply(lambda x: str(x).upper().strip() if pd.notnull(x) else "")

    # Process date if available (for exact day matching)
    if date_col:
        res_df['_date_str'] = pd.to_datetime(res_df[date_col], errors='coerce').dt.strftime('%Y-%m-%d')
    else:
        res_df['_date_str'] = ""

    # Initialize outcome columns
    df['actual_home_score'] = pd.NA
    df['actual_away_score'] = pd.NA
    df['spread_result'] = 'N/A'
    df['total_result'] = 'N/A'
    df['ml_result'] = 'N/A'

    # Match and attach
    for idx, row in df.iterrows():
        league = str(row.get('league', '')).upper().strip()
        home = TeamNameMatcher.normalize(str(row.get('Home', '')))
        away = TeamNameMatcher.normalize(str(row.get('Away', '')))

        # Get date string from master_df
        master_date_str = ""
        if 'Commence (UTC)' in row and pd.notnull(row['Commence (UTC)']):
            try:
                dt = pd.to_datetime(row['Commence (UTC)'])
                master_date_str = dt.strftime('%Y-%m-%d')
            except:
                pass

        # Helper to check if string contains another string or if they are equal
        def match_name(target, candidates_series):
             return candidates_series.apply(lambda x: target in x or x in target or target == x)

        # Find match
        match_mask = (res_df['_norm_league'] == league) & \
                     match_name(home, res_df['_norm_home']) & \
                     match_name(away, res_df['_norm_away'])

        if master_date_str and date_col:
            # Try strict date match first
            strict_mask = match_mask & (res_df['_date_str'] == master_date_str)
            if strict_mask.any():
                match_mask = strict_mask

        matches = res_df[match_mask]

        if len(matches) > 0:
            # Use first match
            match = matches.iloc[0]

            h_score = _safefloat(match[h_score_col])
            a_score = _safefloat(match[a_score_col])

            if h_score is not None and a_score is not None:
                df.at[idx, 'actual_home_score'] = h_score
                df.at[idx, 'actual_away_score'] = a_score

                # Calculate Spread Result
                spread_line = _safefloat(row.get('spread_pick_line'))
                if spread_line is None:
                    # Fallback to parsing from 'Spread & Pick' or 'Pick'
                    pick_str = str(row.get('Spread & Pick') or row.get('Pick', ''))
                    import re
                    # Extract last number which might have + or -
                    m = re.search(r'([+-]?\d+\.?\d*)\s*(?:\(.*\))?$', pick_str)
                    if m:
                         spread_line = _safefloat(m.group(1))

                if spread_line is not None:
                     # Determine which team we picked
                     pick_team = str(row.get('spread_pick_team') or row.get('Pick', '')).split(' ')[0] # naive fallback
                     pick_side = str(row.get('spread_pick_side') or '')

                     if not pick_side:
                         # Infer from team name
                         norm_pick = TeamNameMatcher.normalize(pick_team)
                         if home in norm_pick or norm_pick in home:
                             pick_side = 'home'
                         elif away in norm_pick or norm_pick in away:
                             pick_side = 'away'

                     if pick_side == 'home':
                         # Home spread
                         margin = h_score - a_score
                         if margin + spread_line > 0:
                             df.at[idx, 'spread_result'] = 'WIN'
                         elif margin + spread_line < 0:
                             df.at[idx, 'spread_result'] = 'LOSS'
                         else:
                             df.at[idx, 'spread_result'] = 'PUSH'
                     elif pick_side == 'away':
                         # Away spread
                         margin = a_score - h_score
                         if margin + spread_line > 0:
                             df.at[idx, 'spread_result'] = 'WIN'
                         elif margin + spread_line < 0:
                             df.at[idx, 'spread_result'] = 'LOSS'
                         else:
                             df.at[idx, 'spread_result'] = 'PUSH'

                # Calculate Total Result
                total_line = _safefloat(row.get('total_pick_line'))
                if total_line is None:
                    pick_str = str(row.get('Total & Pick') or row.get('Pick', ''))
                    import re
                    m = re.search(r'(Over|Under)\s*(\d+\.?\d*)', pick_str, re.IGNORECASE)
                    if m:
                        total_line = _safefloat(m.group(2))

                total_side = str(row.get('total_pick_side') or '')
                if not total_side:
                    pick_str = str(row.get('Total & Pick') or row.get('Pick', '')).lower()
                    if 'over' in pick_str:
                        total_side = 'over'
                    elif 'under' in pick_str:
                        total_side = 'under'

                if total_line is not None and total_side:
                    actual_total = h_score + a_score
                    if total_side == 'over':
                        if actual_total > total_line:
                            df.at[idx, 'total_result'] = 'WIN'
                        elif actual_total < total_line:
                            df.at[idx, 'total_result'] = 'LOSS'
                        else:
                            df.at[idx, 'total_result'] = 'PUSH'
                    elif total_side == 'under':
                        if actual_total < total_line:
                            df.at[idx, 'total_result'] = 'WIN'
                        elif actual_total > total_line:
                            df.at[idx, 'total_result'] = 'LOSS'
                        else:
                            df.at[idx, 'total_result'] = 'PUSH'

                # Calculate ML result if best_pick_type is ML (optional fallback)
                ml_pick_team = str(row.get('Pick', '')).replace(' ML', '')
                norm_ml = TeamNameMatcher.normalize(ml_pick_team)

                if h_score > a_score:
                    # Home won
                    if home in norm_ml or norm_ml in home:
                        df.at[idx, 'ml_result'] = 'WIN'
                    else:
                        df.at[idx, 'ml_result'] = 'LOSS'
                elif a_score > h_score:
                    # Away won
                    if away in norm_ml or norm_ml in away:
                        df.at[idx, 'ml_result'] = 'WIN'
                    else:
                        df.at[idx, 'ml_result'] = 'LOSS'
                else:
                    df.at[idx, 'ml_result'] = 'PUSH'

    return df

def _add_empty_outcome_columns(df: pd.DataFrame) -> pd.DataFrame:
    df['actual_home_score'] = pd.NA
    df['actual_away_score'] = pd.NA
    df['spread_result'] = 'N/A'
    df['total_result'] = 'N/A'
    df['ml_result'] = 'N/A'
    return df

def _safefloat(val: Any) -> Optional[float]:
    if pd.isna(val) or val is None or val == "":
        return None
    try:
        return float(val)
    except:
        return None
