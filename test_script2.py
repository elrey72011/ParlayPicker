import pandas as pd
from core.streamlit_pipeline import build_best_picks_df

df = pd.DataFrame([
    {"matchup_id": "1", "home_team": "Team A", "away_team": "Team B", "market_type": "spread_home", "expected_value": 0.10, "edge": 0.05, "is_live_data": True, "odds_source": "novig", "spread_line": -5},
    {"matchup_id": "2", "home_team": "Team C", "away_team": "Team D", "market_type": "spread_home", "expected_value": 0.20, "edge": 0.05, "is_live_data": True, "odds_source": "novig", "spread_line": -5},
    {"matchup_id": "3", "home_team": "Team E", "away_team": "Team F", "market_type": "spread_home", "expected_value": 0.05, "edge": 0.01, "is_live_data": True, "odds_source": "novig", "spread_line": -5},
    {"matchup_id": "4", "home_team": "Team G", "away_team": "Team H", "market_type": "spread_home", "expected_value": 0.15, "edge": 0.05, "is_live_data": False, "odds_source": "novig", "spread_line": -5}, # Fallback
])
res = build_best_picks_df(df)
print(res[["home_team", "Pick_Status", "Triple_Filter_Rank", "expected_value", "edge"]])
