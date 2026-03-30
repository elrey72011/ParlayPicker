import pandas as pd
from core.streamlit_pipeline import build_best_picks_df

# Simulate data to test sorting
df = pd.DataFrame({
    "market_type": ["spread_home"] * 6,
    "home_team": [f"Team_{i}" for i in range(6)],
    "away_team": [f"Away_{i}" for i in range(6)],
    "game_date": ["2026-03-09"] * 6,
    "expected_value": [0.05, 0.05, 0.1, 0.1, 0.2, 0.2],
    "edge": [0.03, 0.04, 0.06, 0.07, 0.1, 0.12],
    "ml_probability": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
    "theover_probability": [0.5] * 6,
    "calibrated_probability": [0.55, 0.65, 0.75, 0.85, 0.92, 0.96],
    "odds_source": ["odds_api"] * 3 + ["fallback_novig"] * 3,
    "odds_american": [-110] * 6,
    "spread_line": [3.5] * 6,
    "is_live_data": [True] * 6,
    "used_stale_features": [False] * 6,
    "stats_quality": ["REAL"] * 6,
    "kalshi_probability": [pd.NA] * 6,
    "Triple_Filter_Rank": [6, 5, 4, 3, 2, 1], # Inverse order
})

best_picks = build_best_picks_df(df)
print(best_picks[["Pick_Status", "Triple_Filter_Rank", "expected_value", "edge", "home_team"]].to_string())
