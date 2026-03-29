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
    from app_core.odds_api import TheOddsAPIClient

    # Simulating what actually causes the rows to drop.
    # The actual output says "Generated 32 total predictions" and "uniqueness is still too compressed".
    # I want to load data/master_all_sports.csv or just some real team names.

    def mock_get_odds(self, sport_key, date=None):
        return [
            {
                "id": f"game_nba_{i}",
                "sport_key": "basketball_nba",
                "home_team": ["Los Angeles Lakers", "Chicago Bulls", "Boston Celtics", "Miami Heat"][i%4],
                "away_team": ["Golden State Warriors", "Milwaukee Bucks", "New York Knicks", "Orlando Magic"][i%4],
                "commence_time": "2026-03-30T00:00:00Z",
                "matchup_id": f"NBA_{i}",
                "bookmakers": [
                    {
                        "key": "novig",
                        "markets": [
                            {
                                "key": "spreads",
                                "outcomes": [
                                    {"name": ["Los Angeles Lakers", "Chicago Bulls", "Boston Celtics", "Miami Heat"][i%4], "point": -5.5, "price": -110},
                                    {"name": ["Golden State Warriors", "Milwaukee Bucks", "New York Knicks", "Orlando Magic"][i%4], "point": 5.5, "price": -110},
                                ]
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "point": 220.5, "price": -110},
                                    {"name": "Under", "point": 220.5, "price": -110},
                                ]
                            }
                        ]
                    }
                ]
            }
            for i in range(16)
        ] + [
             {
                "id": f"game_nhl_{i}",
                "sport_key": "icehockey_nhl",
                "home_team": ["New York Rangers", "Boston Bruins"][i%2],
                "away_team": ["Colorado Avalanche", "Florida Panthers"][i%2],
                "commence_time": "2026-03-30T00:00:00Z",
                "matchup_id": f"NHL_{i}",
                "bookmakers": [
                    {
                        "key": "novig",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": ["New York Rangers", "Boston Bruins"][i%2], "price": -150},
                                    {"name": ["Colorado Avalanche", "Florida Panthers"][i%2], "price": +130},
                                ]
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "point": 6.5, "price": -110},
                                    {"name": "Under", "point": 6.5, "price": -110},
                                ]
                            }
                        ]
                    }
                ]
            }
            for i in range(16)
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
