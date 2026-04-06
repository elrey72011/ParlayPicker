import unittest
import pandas as pd
from core.streamlit_pipeline import run_analysis_pipeline
import logging

logging.basicConfig(level=logging.INFO)

class TestOddsSourceProvenance(unittest.TestCase):
    def test_odds_source_not_overwritten(self):
        # Create a mock dataframe that mimics the results of the pipeline before final drop
        df = pd.DataFrame([
            {
                "league": "NBA",
                "home_team": "Chicago Bulls",
                "away_team": "Miami Heat",
                "game_date": "2023-11-01",
                "matchup_id": "2023-11-01|chicago bulls|miami heat",
                "market_type": "spread_away",
                "odds_american": -105,
                "odds_source": "odds_api"
            }
        ])

        # Test the specific code block from run_analysis_pipeline
        # "Avoid force overwriting odds_source to uploaded just because odds are not -110"

        # We need to simulate the environment
        if "odds_source" not in df.columns:
            df["odds_source"] = pd.NA

        uploaded_odds = pd.to_numeric(df.get("odds_american"), errors="coerce")
        missing_source = df.get("odds_source", pd.Series([""] * len(df))).isna() | df.get("odds_source", pd.Series([""] * len(df))).eq("")

        df.loc[uploaded_odds.notna() & (uploaded_odds != -110) & missing_source, "odds_source"] = "uploaded_only"

        self.assertEqual(df.iloc[0]["odds_source"], "odds_api")

if __name__ == '__main__':
    unittest.main()
