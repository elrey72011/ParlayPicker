# feature_engine.py
import pandas as pd
import numpy as np

def prepare_features_for_inference(history_df, todays_df):
    """
    Combines history + today to calculate rolling stats (Last 5, etc.) correctly.
    This ensures today's game row has the context of the previous games.
    """
    print("🔄 Calculating rolling features...")
    
    # 1. Combine History and Today
    combined = pd.concat([history_df, todays_df], axis=0, ignore_index=True)
    combined['commence_time'] = pd.to_datetime(combined['commence_time'])
    
    if "game_id" not in combined.columns:
        combined["game_id"] = np.arange(len(combined))

    # 2. Build Long Format (One row per team)
    base_cols = ["game_id", "sport", "commence_time", "home_team", "away_team", "home_score", "away_score"]
    
    # Home perspective
    home = combined[base_cols].copy()
    home = home.rename(columns={
        "home_team": "team", "away_team": "opponent", 
        "home_score": "points_for", "away_score": "points_against"
    })
    home["is_home"] = 1
    
    # Away perspective
    away = combined[base_cols].copy()
    away = away.rename(columns={
        "away_team": "team", "home_team": "opponent", 
        "away_score": "points_for", "home_score": "points_against"
    })
    away["is_home"] = 0
    
    team_games = pd.concat([home, away], ignore_index=True)
    team_games = team_games.sort_values(["sport", "team", "commence_time"])
    
    # 3. Calculate Rolling Stats
    grp = team_games.groupby(["sport", "team"])
    window = 5
    
    # Calculate 'Won' (Only for historical games)
    team_games["won"] = (team_games["points_for"] > team_games["points_against"]).astype(float)
    team_games.loc[team_games["points_for"].isna(), "won"] = np.nan # NaN for today's games
    
    team_games["point_diff"] = team_games["points_for"] - team_games["points_against"]
    
    # -- Rolling Calculations --
    team_games["form_last5"] = grp["won"].shift(1).rolling(window, min_periods=1).mean()
    team_games["pf_last5"] = grp["points_for"].shift(1).rolling(window, min_periods=1).mean()
    team_games["pd_last5"] = grp["point_diff"].shift(1).rolling(window, min_periods=1).mean()
    
    # Pre-game Win %
    team_games["games_played"] = grp.cumcount()
    team_games["wins_cum"] = grp["won"].shift(1).cumsum().fillna(0)
    team_games["win_pct_before"] = np.where(
        team_games["games_played"] > 0,
        team_games["wins_cum"] / team_games["games_played"],
        0.5
    )

    # SOS (Strength of Schedule)
    opp_stats = team_games[["sport", "team", "commence_time", "win_pct_before"]].rename(
        columns={"team": "opponent", "win_pct_before": "opp_win_pct"}
    )
    team_games = team_games.merge(opp_stats, on=["sport", "opponent", "commence_time"], how="left")
    
    grp_v2 = team_games.groupby(["sport", "team"])
    team_games["sos_last5"] = grp_v2["opp_win_pct"].shift(1).rolling(window, min_periods=1).mean()
    
    # 4. Merge Back to Home/Away Format
    cols_to_keep = ["game_id", "win_pct_before", "pf_last5", "form_last5", "pd_last5", "sos_last5"]
    
    home_stats = team_games[team_games["is_home"] == 1][cols_to_keep].rename(columns={
        "win_pct_before": "home_win_pct", "pf_last5": "home_ppg",
        "form_last5": "home_form_last5", "pd_last5": "home_pd_last5", "sos_last5": "home_sos_last5"
    })
    
    away_stats = team_games[team_games["is_home"] == 0][cols_to_keep].rename(columns={
        "win_pct_before": "away_win_pct", "pf_last5": "away_ppg",
        "form_last5": "away_form_last5", "pd_last5": "away_pd_last5", "sos_last5": "away_sos_last5"
    })
    
    # Drop columns if they already exist to avoid conflicts
    drop_cols = ["home_win_pct", "away_win_pct", "home_ppg", "away_ppg"]
    combined = combined.drop(columns=[c for c in drop_cols if c in combined.columns], errors='ignore')
    
    combined = combined.merge(home_stats, on="game_id", how="left")
    combined = combined.merge(away_stats, on="game_id", how="left")
    
    # 5. Differentials & Defaults
    combined = combined.fillna({
        "home_win_pct": 0.5, "away_win_pct": 0.5,
        "home_ppg": 0.5, "away_ppg": 0.5,
        "home_form_last5": 0.5, "away_form_last5": 0.5,
        "home_pd_last5": 0, "away_pd_last5": 0,
        "home_sos_last5": 0.5, "away_sos_last5": 0.5,
        "home_oppg": 0.5, "away_oppg": 0.5,
        "home_streak": 0, "away_streak": 0,
        "public_betting_pct": 50, "sharp_money_indicator": 0
    })

    combined["win_pct_diff"] = combined["home_win_pct"] - combined["away_win_pct"]
    combined["ppg_diff"] = combined["home_ppg"] - combined["away_ppg"]
    combined["form_diff"] = combined["home_form_last5"] - combined["away_form_last5"]
    combined["pd_last5_diff"] = combined["home_pd_last5"] - combined["away_pd_last5"]
    combined["sos_diff"] = combined["home_sos_last5"] - combined["away_sos_last5"]
    
    if "home_oppg" not in combined.columns:
        combined["home_oppg"] = combined["away_ppg"]
        combined["away_oppg"] = combined["home_ppg"]
        
    combined["oppg_diff"] = combined["home_oppg"] - combined["away_oppg"]
    combined["streak_diff"] = combined["home_streak"] - combined["away_streak"]
    
    combined["public_betting_centered"] = (combined["public_betting_pct"] - 50.0) / 50.0
    combined["sharp_vs_public"] = combined["sharp_money_indicator"] - combined["public_betting_pct"]
    
    # Return only the rows that correspond to the "today" games
    return combined[combined['game_id'].isin(todays_df['game_id'])].copy()

# Ensured avg_points is ppg and def_rating is oppg
