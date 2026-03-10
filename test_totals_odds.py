import pandas as pd
from core.streamlit_pipeline import build_theover_bet_rows

totals_df = pd.DataFrame(
    {
        "league": ["NBA"],
        "home_team": ["Boston Celtics"],
        "away_team": ["Miami Heat"],
        "selection": ["Over"],
        "points": [220.5],
        "probability": [58],
        "american_odds": [-105],
    }
)

out = build_theover_bet_rows(None, totals_df, ["NBA"])
print(out[["market_type", "total_line", "odds_american", "theover_probability", "best_pick"]])
