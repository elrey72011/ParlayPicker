import pandas as pd
import numpy as np
from core.streamlit_pipeline import build_best_picks_df

base_row = {
    "market_type": "spread",
    "expected_value": 0.05,
    "edge": 0.05,
    "best_pick": "Test Pick -3.5",
    "league": "NBA",
    "home_team": "Team A",
    "away_team": "Team B",
    "game_date": "2024-01-01T00:00:00Z",
    "model_probability": 0.60,
    "ml_probability": 0.60,
    "kalshi_probability": 0.60,
    "calibrated_probability": 0.60,
    "is_live_data": True,
    "odds_source": "fanduel",
    "spread_line": -3.5,
    "total_line": 220.5,
    "candidate_source": "ml",
    "orientation_source": "home",
    "upload_match_reason": "none",
}

def _build_df(rows):
    return pd.DataFrame([ {**base_row, **row} for row in rows ])

df = _build_df([
    {"league": "NBA", "market_type": "total_over", "expected_value": 0.05, "edge": 0.05, "calibrated_probability": 0.57, "kalshi_probability": 0.57, "best_pick": "Over 222.5", "home_team": "Team C", "away_team": "Team D"},
    {"league": "NBA", "market_type": "total_over", "expected_value": 0.05, "edge": 0.05, "calibrated_probability": 0.565, "kalshi_probability": 0.565, "best_pick": "Over 220.5", "home_team": "Team E", "away_team": "Team F"},
])
best = build_best_picks_df(df)
print(best[["home_team", "Pick_Status", "Status_Reason", "consensus_agreement"]].to_dict('records'))
