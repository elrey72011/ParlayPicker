import sys
import pandas as pd
from core.streamlit_pipeline import build_best_picks_df

data = {
    "league": ["NBA"] * 5,
    "home_team": ["LAL", "BOS", "CHI", "MIA", "DEN"],
    "away_team": ["GSW", "PHI", "DET", "ORL", "PHX"],
    "game_date": ["2026-03-01"] * 5,
    "market_type": ["spread_home"] * 5,
    "expected_value": [0.05, 0.10, 0.02, 0.12, 0.08],
    "edge": [0.03, 0.05, 0.015, 0.06, 0.04],
    "spread_line": [-5.5] * 5,
    "total_line": [pd.NA] * 5,
    "is_live_data": [True] * 5,
    "stats_quality": ["good"] * 5,
    "odds_american": [-110] * 5,
    "odds_source": ["odds_api"] * 5,
    "best_pick": ["LAL -5.5", "BOS -5.5", "CHI -5.5", "MIA -5.5", "DEN -5.5"],
    "market_probability": [0.5] * 5,
    "ml_probability": [0.55, 0.45, 0.52, 0.6, 0.505],
    "used_stale_features": [False] * 5,
    "calibrated_probability": [0.55, 0.45, 0.52, 0.6, 0.505],
    "matchup_id": ["LAL_GSW", "BOS_PHI", "CHI_DET", "MIA_ORL", "DEN_PHX"]
}

df = pd.DataFrame(data)

best_picks = build_best_picks_df(df)
print(best_picks[["home_team", "Pick_Status", "Triple_Filter_Rank", "expected_value", "edge"]])
