import pandas as pd
from core.streamlit_pipeline import build_best_picks_df

df = pd.DataFrame([
    {"matchup_id": "1", "home_team": "Team A", "away_team": "Team B", "market_type": "spread_home", "expected_value": 0.10, "edge": 0.05, "is_live_data": True, "odds_source": "novig", "spread_line": -5, "ml_probability": 0.6, "theover_probability": 0.5, "calibrated_probability": 0.6},
    {"matchup_id": "2", "home_team": "Team C", "away_team": "Team D", "market_type": "spread_home", "expected_value": 0.20, "edge": 0.05, "is_live_data": True, "odds_source": "novig", "spread_line": -5, "ml_probability": 0.7, "theover_probability": 0.5, "calibrated_probability": 0.7},
    {"matchup_id": "3", "home_team": "Team E", "away_team": "Team F", "market_type": "spread_home", "expected_value": 0.05, "edge": 0.01, "is_live_data": True, "odds_source": "novig", "spread_line": -5, "ml_probability": 0.55, "theover_probability": 0.5, "calibrated_probability": 0.55},
    {"matchup_id": "4", "home_team": "Team G", "away_team": "Team H", "market_type": "spread_home", "expected_value": 0.15, "edge": 0.05, "is_live_data": False, "odds_source": "novig", "spread_line": -5, "ml_probability": 0.65, "theover_probability": 0.5, "calibrated_probability": 0.65}, # Fallback
    {"matchup_id": "5", "home_team": "Team I", "away_team": "Team J", "market_type": "spread_home", "expected_value": 0.25, "edge": 0.05, "is_live_data": True, "odds_source": "novig", "spread_line": -5, "ml_probability": 0.75, "theover_probability": 0.5, "calibrated_probability": 0.75},
])
res = build_best_picks_df(df)
print(res[["home_team", "Pick_Status", "Triple_Filter_Rank", "expected_value", "edge"]])
