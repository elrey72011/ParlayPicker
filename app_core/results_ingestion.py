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

    # Allow manual results (matchup_id, outcome) to bypass the column check
    is_manual_check = 'matchup_id' in res_df.columns and 'outcome' in res_df.columns
    if not is_manual_check and not all([home_col, away_col, h_score_col, a_score_col, league_col]):
        logger.error(f"results_df missing required columns. Found: {res_df.columns.tolist()}")
        return _add_empty_outcome_columns(df)

    from app_core.feature_processing import robust_normalize_team
    from app_core.prediction_engine import clean_team_name

    # Pre-compute normalized names
    res_df['_norm_home'] = res_df[home_col].apply(lambda x: robust_normalize_team(str(x)).lower() if pd.notnull(x) else "")
    res_df['_norm_away'] = res_df[away_col].apply(lambda x: robust_normalize_team(str(x)).lower() if pd.notnull(x) else "")
    res_df['_clean_home'] = res_df[home_col].apply(lambda x: clean_team_name(str(x)) if pd.notnull(x) else "")
    res_df['_clean_away'] = res_df[away_col].apply(lambda x: clean_team_name(str(x)) if pd.notnull(x) else "")
    res_df['_norm_league'] = res_df[league_col].apply(lambda x: str(x).upper().strip() if pd.notnull(x) else "")

    # Process date if available (for exact day matching)
    if date_col:
        res_df['_date_str'] = pd.to_datetime(res_df[date_col], errors='coerce').dt.strftime('%Y-%m-%d')
    else:
        res_df['_date_str'] = ""

# Initialize outcome columns
    df['actual_home_score'] = np.nan
    df['actual_away_score'] = np.nan
    df['spread_result'] = 'N/A'
    df['total_result'] = 'N/A'
    df['ml_result'] = 'N/A'
    df['Pick_Outcome'] = 'N/A'

    # Check if this is a "Manual Results" dataframe (has matchup_id and outcome, but no scores)
    is_manual = 'matchup_id' in res_df.columns and 'outcome' in res_df.columns and h_score_col is None and a_score_col is None

    if is_manual:
        logger.info("Processing manual results based on matchup_id")
        # Create a mapping dictionary for quick lookup
        manual_map = dict(zip(res_df['matchup_id'], res_df['outcome']))

        for idx, row in df.iterrows():
            m_id = str(row.get('matchup_id', ''))
            if m_id in manual_map:
                outcome = str(manual_map[m_id]).upper().strip()
                if outcome in ['WIN', 'LOSS', 'PUSH']:
                    df.loc[idx, 'Pick_Outcome'] = outcome

                    # Also populate specific results so determining outcome doesn't fail later
                    best_pick_str = str(row.get('best_pick', '')).lower()
                    if 'over' in best_pick_str or 'under' in best_pick_str:
                        df.loc[idx, 'total_result'] = outcome
                    elif '+' in best_pick_str or '-' in best_pick_str:
                        df.loc[idx, 'spread_result'] = outcome
                    else:
                        df.loc[idx, 'ml_result'] = outcome

        return df

    # Match and attach
    for idx, row in df.iterrows():
        league = str(row.get('league', '')).upper().strip()
        # Fallback names in case master_df has 'home_team' instead of 'Home'
        home_name = str(row.get('home_team', row.get('Home', '')))
        away_name = str(row.get('away_team', row.get('Away', '')))

        home = robust_normalize_team(home_name).lower()
        away = robust_normalize_team(away_name).lower()
        home_clean = clean_team_name(home_name)
        away_clean = clean_team_name(away_name)

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
            # We want token-based match or substring match that works both ways
            # target = 'los angeles lakers' (master)
            # candidate = 'lakers' (results)
            # 'lakers' in 'los angeles lakers' -> True
            # if candidate is empty, False
            def is_match(c):
                if pd.isna(c) or not c: return False
                c = str(c)
                # exact or substring
                # also lower it again just in case
                c = c.lower()
                target_lower = target.lower()
                if c == target_lower or c in target_lower or target_lower in c: return True
                return False

            return candidates_series.apply(is_match).fillna(False).astype(bool)

        mask1 = (res_df['_norm_league'] == league).fillna(False).astype(bool)

        # 1. Try robust normalize match
        mask2 = match_name(home, res_df['_norm_home'])
        mask3 = match_name(away, res_df['_norm_away'])
        match_mask = mask1 & mask2 & mask3

        # 2. Fallback to strict clean match (strips ALL spaces)
        if not match_mask.any():
            mask2_clean = match_name(home_clean, res_df['_clean_home'])
            mask3_clean = match_name(away_clean, res_df['_clean_away'])
            match_mask = mask1 & mask2_clean & mask3_clean

        # 3. Last resort fallback: check raw names with original logic
        if not match_mask.any():
            raw_home = home_name.lower()
            raw_away = away_name.lower()
            mask2_raw = match_name(raw_home, res_df[home_col].fillna("").str.lower())
            mask3_raw = match_name(raw_away, res_df[away_col].fillna("").str.lower())
            match_mask = mask1 & mask2_raw & mask3_raw

        if master_date_str and date_col:
            # Try strict date match first
            strict_mask = match_mask & (res_df['_date_str'] == master_date_str).fillna(False).astype(bool)
            if strict_mask.any():
                match_mask = strict_mask

        matches = res_df[match_mask.fillna(False).astype(bool)]

        if len(matches) > 0:
            # Use first match
            match = matches.iloc[0]

            h_score = _safefloat(match[h_score_col])
            a_score = _safefloat(match[a_score_col])

            if h_score is not None and a_score is not None:
                df.loc[idx, 'actual_home_score'] = h_score
                df.loc[idx, 'actual_away_score'] = a_score

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
                             df.loc[idx, 'spread_result'] = 'WIN'
                         elif margin + spread_line < 0:
                             df.loc[idx, 'spread_result'] = 'LOSS'
                         else:
                             df.loc[idx, 'spread_result'] = 'PUSH'
                     elif pick_side == 'away':
                         # Away spread
                         margin = a_score - h_score
                         if margin + spread_line > 0:
                             df.loc[idx, 'spread_result'] = 'WIN'
                         elif margin + spread_line < 0:
                             df.loc[idx, 'spread_result'] = 'LOSS'
                         else:
                             df.loc[idx, 'spread_result'] = 'PUSH'

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
                            df.loc[idx, 'total_result'] = 'WIN'
                        elif actual_total < total_line:
                            df.loc[idx, 'total_result'] = 'LOSS'
                        else:
                            df.loc[idx, 'total_result'] = 'PUSH'
                    elif total_side == 'under':
                        if actual_total < total_line:
                            df.loc[idx, 'total_result'] = 'WIN'
                        elif actual_total > total_line:
                            df.loc[idx, 'total_result'] = 'LOSS'
                        else:
                            df.loc[idx, 'total_result'] = 'PUSH'

                # Calculate ML result if best_pick_type is ML (optional fallback)
                ml_pick_team = str(row.get('best_pick', row.get('Pick', ''))).replace(' ML', '').lower()
                norm_ml_raw = robust_normalize_team(ml_pick_team).lower()
                norm_ml_clean = clean_team_name(ml_pick_team)

                if h_score > a_score:
                    # Home won
                    if home in norm_ml_raw or norm_ml_raw in home or home_clean in norm_ml_clean or norm_ml_clean in home_clean:
                        df.loc[idx, 'ml_result'] = 'WIN'
                    else:
                        df.loc[idx, 'ml_result'] = 'LOSS'
                elif a_score > h_score:
                    # Away won
                    if away in norm_ml_raw or norm_ml_raw in away or away_clean in norm_ml_clean or norm_ml_clean in away_clean:
                        df.loc[idx, 'ml_result'] = 'WIN'
                    else:
                        df.loc[idx, 'ml_result'] = 'LOSS'
                else:
                    df.loc[idx, 'ml_result'] = 'PUSH'

                # Assign explicitly based on best_pick string
                best_pick_str = str(row.get('best_pick', '')).lower()
                if best_pick_str:
                    if 'over' in best_pick_str or 'under' in best_pick_str:
                        df.loc[idx, 'Pick_Outcome'] = df.loc[idx, 'total_result']
                    elif '+' in best_pick_str or '-' in best_pick_str:
                        df.loc[idx, 'Pick_Outcome'] = df.loc[idx, 'spread_result']
                    else:
                        df.loc[idx, 'Pick_Outcome'] = df.loc[idx, 'ml_result']
                else:
                    df.loc[idx, 'Pick_Outcome'] = 'N/A'

    return df

def _add_empty_outcome_columns(df: pd.DataFrame) -> pd.DataFrame:
    df['actual_home_score'] = np.nan
    df['actual_away_score'] = np.nan
    df['spread_result'] = 'N/A'
    df['total_result'] = 'N/A'
    df['ml_result'] = 'N/A'
    df['Pick_Outcome'] = 'N/A'
    return df

def _safefloat(val: Any) -> Optional[float]:
    if pd.isna(val) or val is None or val == "":
        return None
    try:
        return float(val)
    except:
        return None
