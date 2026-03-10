import pandas as pd
from core.streamlit_pipeline import build_best_picks_df, BEST_PICK_COLUMNS
import numpy as np

# Create a sample analysis dataframe
data = {
    "league": ["NCAAB", "NCAAB", "NBA"],
    "home_team": ["Eastern Washington", "Eastern Washington", "Lakers"],
    "away_team": ["Portland State", "Portland State", "Celtics"],
    "game_date": [np.nan, "2023-11-20T00:00:00Z", np.nan],
    "market_type": ["spread_home", "total_over", "spread_away"],
    "expected_value": [0.10, 0.05, 0.08],
    "edge": [0.05, 0.04, 0.06],
    "model_probability": [0.6, 0.55, 0.58],
    "theover_probability": [np.nan, np.nan, np.nan],
    "ml_probability": [np.nan, np.nan, np.nan],
    "spread_line": [-2.5, np.nan, 3.5],
    "total_line": [np.nan, 145.5, np.nan]
}
analysis_df = pd.DataFrame(data)

analysis_df["game_date"] = pd.to_datetime(analysis_df["game_date"], errors="coerce", utc=True)

best_picks = build_best_picks_df(analysis_df)

print(f"Total best picks: {len(best_picks)}")
for idx, row in best_picks.iterrows():
    print(f"Matchup: {row.get('home_team')} vs {row.get('away_team')}, Pick: {row.get('best_pick')}, EV: {row.get('expected_value')}")

assert len(best_picks) == 2, f"Expected 2 best picks, got {len(best_picks)}"
print("Success!")
