import pandas as pd
from core.streamlit_pipeline import build_best_picks_df

df = pd.DataFrame([
    {"matchup_id": "1", "market_type": "spread_home", "expected_value": 0.10, "edge": 0.05, "is_live_data": True, "odds_source": "novig"},
    {"matchup_id": "2", "market_type": "spread_home", "expected_value": 0.20, "edge": 0.05, "is_live_data": True, "odds_source": "novig"},
    {"matchup_id": "3", "market_type": "spread_home", "expected_value": 0.05, "edge": 0.01, "is_live_data": True, "odds_source": "novig"},
    {"matchup_id": "4", "market_type": "spread_home", "expected_value": 0.15, "edge": 0.05, "is_live_data": False, "odds_source": "novig"}, # Fallback
])
res = build_best_picks_df(df)
print(res[["Pick_Status", "Triple_Filter_Rank", "expected_value", "edge"]])
