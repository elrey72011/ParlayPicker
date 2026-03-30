import sys
import pandas as pd
from core.streamlit_pipeline import run_analysis_pipeline, build_best_picks_df

# create dummy data to test the pick status branches
data = {
    "league": ["NBA", "NBA", "NBA", "NBA", "NBA"],
    "home_team": ["Los Angeles Lakers", "Boston Celtics", "Chicago Bulls", "Miami Heat", "Denver Nuggets"],
    "away_team": ["Golden State Warriors", "Philadelphia 76ers", "Detroit Pistons", "Orlando Magic", "Phoenix Suns"],
    "game_date": ["2026-03-01", "2026-03-01", "2026-03-01", "2026-03-01", "2026-03-01"],
    "market_type": ["spread_home", "total_over", "h2h_home", "spread_away", "total_under"],
    "expected_value": [0.05, -0.01, 0.02, 0.10, 0.001],
    "edge": [0.03, -0.05, 0.015, 0.08, 0.005],
    "spread_line": [-5.5, pd.NA, pd.NA, 4.5, pd.NA],
    "total_line": [pd.NA, 220.5, pd.NA, pd.NA, pd.NA],
    "is_live_data": [True, True, False, True, True], # Chicago Bulls gets Fallback due to False
    "stats_quality": ["good", "good", "fallback", "good", "good"],
    "odds_american": [-110, -110, -110, -110, -110],
    "odds_source": ["odds_api", "odds_api", "fallback_novig", "odds_api", "odds_api"],
    "best_pick": ["Los Angeles Lakers -5.5", "Over 220.5", "Chicago Bulls", "Orlando Magic +4.5", "Under (No Line)"], # Denver gets Missing Line
    "market_probability": [0.5, 0.5, 0.5, 0.5, 0.5],
    "ml_probability": [0.55, 0.45, 0.52, 0.6, 0.505],
    "used_stale_features": [False, False, False, False, False],
    "calibrated_probability": [0.55, 0.45, 0.52, 0.6, 0.505],
    "matchup_id": ["LAL_GSW", "BOS_PHI", "CHI_DET", "MIA_ORL", "DEN_PHX"]
}

df = pd.DataFrame(data)

best_picks = build_best_picks_df(df)
print(best_picks[["home_team", "Pick_Status", "Triple_Filter_Rank", "expected_value", "edge", "best_pick", "odds_source"]])

status_counts = best_picks["Pick_Status"].value_counts().to_dict()
print(f"\nStatus Counts: {status_counts}")

for status in ["Actionable", "Below Threshold", "Fallback / Low Confidence", "No Play", "Missing Line"]:
    if status in status_counts:
        print(f"Validated branch: {status}")
    else:
        print(f"Branch not exercised: {status}")
