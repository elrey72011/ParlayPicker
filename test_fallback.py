import pandas as pd
from core.streamlit_pipeline import build_best_picks_df
import logging
import sys

# Configure logging for the test output
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# Simulate data to test fallback enforcement
df = pd.DataFrame({
    "market_type": ["spread_home"] * 4,
    "home_team": ["T1", "T2", "T3", "T4"],
    "away_team": ["A1", "A2", "A3", "A4"],
    "game_date": ["2026-03-09"] * 4,
    "expected_value": [0.1, -0.1, 0.1, -0.1],
    "edge": [0.06, -0.05, 0.06, -0.05],
    "ml_probability": [0.6] * 4,
    "theover_probability": [0.5] * 4,
    "calibrated_probability": [0.55] * 4,
    # T1: odds_api, +EV -> Actionable
    # T2: odds_api, -EV -> No Play
    # T3: fallback_novig, +EV -> Fallback / Low Confidence
    # T4: fallback_novig, -EV -> No Play
    "odds_source": ["odds_api", "odds_api", "fallback_novig", "fallback_novig"],
    "odds_american": [-110] * 4,
    "spread_line": [3.5] * 4,
    "is_live_data": [True] * 4,
    "used_stale_features": [False] * 4,
    "stats_quality": ["REAL"] * 4,
    "kalshi_probability": [pd.NA] * 4,
    "best_pick": ["T1", "T2", "T3", "T4"]
})

best_picks = build_best_picks_df(df)
print(best_picks[["Pick_Status", "odds_source", "expected_value", "edge", "home_team"]].to_string())
