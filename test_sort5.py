import sys
import logging
import pandas as pd
from core.streamlit_pipeline import build_best_picks_df

# Configure logging to print to console
logging.basicConfig(level=logging.INFO, format="%(message)s")

# create dummy data to test the pick status branches
data = {
    "league": ["NBA"] * 5,
    "home_team": ["LAL", "BOS", "CHI", "MIA", "DEN"],
    "away_team": ["GSW", "PHI", "DET", "ORL", "PHX"],
    "game_date": ["2026-03-01"] * 5,
    "market_type": ["spread_home"] * 5,
    "expected_value": [0.05, -0.01, 0.02, 0.10, 0.001],
    "edge": [0.03, -0.05, 0.015, 0.08, 0.005],
    "spread_line": [-5.5, -4.5, -5.5, 4.5, -5.5],
    "total_line": [pd.NA, 220.5, pd.NA, pd.NA, pd.NA],
    "is_live_data": [True, True, False, True, True], # Chicago Bulls gets Fallback due to False
    "stats_quality": ["good", "good", "fallback", "good", "good"],
    "odds_american": [-110, -110, -110, -110, -110],
    "odds_source": ["odds_api", "odds_api", "fallback_novig", "odds_api", "odds_api"],
    "best_pick": ["LAL -5.5", "BOS -4.5", "CHI -5.5", "MIA +4.5", "DEN -5.5"],
    "market_probability": [0.5] * 5,
    "ml_probability": [0.55, 0.45, 0.52, 0.6, 0.505],
    "used_stale_features": [False] * 5,
    "calibrated_probability": [0.55, 0.45, 0.52, 0.6, 0.505],
    "matchup_id": ["LAL_GSW", "BOS_PHI", "CHI_DET", "MIA_ORL", "DEN_PHX"]
}

df = pd.DataFrame(data)

best_picks = build_best_picks_df(df)
