import pandas as pd
from core.streamlit_pipeline import _first_existing_numeric, _normalize_upload, _build_total_rows

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

normalized = _normalize_upload(totals_df)
print(normalized.columns)
print(normalized.head())
total_odds = _first_existing_numeric(normalized, ["odds_american", "american_odds", "odds"], default=-110.0)
print(total_odds)
rows = _build_total_rows(normalized)
print(rows[0]["odds_american"])
