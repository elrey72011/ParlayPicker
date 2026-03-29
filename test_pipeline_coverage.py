import os
import sys
import logging
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')

def main():
    import streamlit as st
    st.secrets = {}
    st.secrets["ODDS_API_KEY"] = "dummy"

    from core.streamlit_pipeline import run_analysis_pipeline

    # We will pass dummy data to simulate OddsAPI response
    from app_core.odds_api import TheOddsAPIClient
    original_get_odds = TheOddsAPIClient.get_odds

    def mock_get_odds(self, sport_key, date=None):
        return [
            {
                "id": f"game_{i}",
                "sport_key": "basketball_ncaab",
                "home_team": f"Home_{i}",
                "away_team": f"Away_{i}",
                "commence_time": "2026-03-30T00:00:00Z",
                "matchup_id": f"H_{i}|A_{i}",
                "bookmakers": [
                    {
                        "key": "novig",
                        "markets": [
                            {
                                "key": "spreads",
                                "outcomes": [
                                    {"name": f"Home_{i}", "point": -5.5, "price": -110},
                                    {"name": f"Away_{i}", "point": 5.5, "price": -110},
                                ]
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "point": 140.5, "price": -110},
                                    {"name": "Under", "point": 140.5, "price": -110},
                                ]
                            }
                        ]
                    }
                ]
            }
            for i in range(100)
        ]

    TheOddsAPIClient.get_odds = mock_get_odds

    print("Running pipeline...")
    analysis_df, best_picks_df, diagnostics = run_analysis_pipeline(
        sports=["NBA", "NHL", "NCAAB", "MLB"],
        max_rows=1000,
        use_ml=True,
    )

    print(f"\nanalysis_df len: {len(analysis_df)}")
    print(f"best_picks_df len: {len(best_picks_df)}")
    if not analysis_df.empty:
        if 'ml_probability' in analysis_df.columns:
            ml_count = analysis_df['ml_probability'].notna().sum()
            ml_unique = analysis_df['ml_probability'].nunique()
            print(f"ml_probability valid rows: {ml_count}")
            print(f"ml_probability unique values: {ml_unique}")

if __name__ == "__main__":
    main()
