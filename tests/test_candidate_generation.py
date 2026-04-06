import unittest
import pandas as pd
from core.streamlit_pipeline import _expand_live_odds_to_bet_rows, build_best_picks_df, BEST_PICK_COLUMNS
import logging

logging.basicConfig(level=logging.INFO)

class TestCandidateGeneration(unittest.TestCase):
    def setUp(self):
        self.live_odds_df = pd.DataFrame([
            {
                "league": "NBA",
                "home_team": "Chicago Bulls",
                "away_team": "Miami Heat",
                "game_date": "2023-11-01",
                "matchup_id": "2023-11-01|chicago bulls|miami heat",
                "commence_time_raw": "2023-11-01T20:00:00Z",
                "novig_home_price": -110,
                "novig_home_point": -5.5,
                "novig_away_price": -110,
                "novig_away_point": 5.5,
                "novig_over_price": -110,
                "novig_over_point": 220.5,
                "novig_under_price": -110,
                "novig_under_point": 220.5,
            }
        ])

    def test_no_upload_retains_full_candidate_set(self):
        result, _ = _expand_live_odds_to_bet_rows(self.live_odds_df, None)
        self.assertEqual(len(result), 4)
        markets = set(result["market_type"])
        self.assertEqual(markets, {"spread_home", "spread_away", "total_over", "total_under"})
        self.assertTrue(all(result["candidate_source"] == "live_unfiltered"))

    def test_upload_match_retains_only_uploaded_markets(self):
        theover_rows = pd.DataFrame([
            {
                "matchup_id": "2023-11-01|chicago bulls|miami heat",
                "market_type": "spread_away",
                "league": "NBA",
                "home_team": "Chicago Bulls",
                "away_team": "Miami Heat"
            },
            {
                "matchup_id": "2023-11-01|chicago bulls|miami heat",
                "market_type": "total_under",
                "league": "NBA",
                "home_team": "Chicago Bulls",
                "away_team": "Miami Heat"
            }
        ])
        result, _ = _expand_live_odds_to_bet_rows(self.live_odds_df, theover_rows)
        self.assertEqual(len(result), 2)
        markets = set(result["market_type"])
        self.assertEqual(markets, {"spread_away", "total_under"})
        self.assertTrue(all(result["candidate_source"] == "upload_matched"))

    def test_failed_join_retains_full_candidate_set(self):
        theover_rows = pd.DataFrame([
            {
                "matchup_id": "2023-11-01|different team|another team", # mismatched
                "market_type": "spread_away",
                "league": "NBA",
                "home_team": "Different Team",
                "away_team": "Another Team"
            }
        ])
        result, _ = _expand_live_odds_to_bet_rows(self.live_odds_df, theover_rows)
        self.assertEqual(len(result), 4)
        markets = set(result["market_type"])
        self.assertEqual(markets, {"spread_home", "spread_away", "total_over", "total_under"})
        self.assertTrue(all(result["candidate_source"] == "live_unfiltered"))

    def test_canonical_join_retains_uploaded_markets(self):
        theover_rows = pd.DataFrame([
            {
                "matchup_id": "different-id", # mismatched exact ID
                "market_type": "spread_away",
                "league": "NBA",
                "home_team": "miami heat", # Reversed!
                "away_team": "chicago bulls",
                "game_date": "2023-11-01"
            }
        ])
        result, _ = _expand_live_odds_to_bet_rows(self.live_odds_df, theover_rows)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["market_type"], "spread_away")
        self.assertEqual(result.iloc[0]["candidate_source"], "upload_matched")
        self.assertEqual(result.iloc[0]["orientation_source"], "canonical_match")

    def test_fuzzy_join_retains_uploaded_markets(self):
        theover_rows = pd.DataFrame([
            {
                "matchup_id": "different-id",
                "market_type": "total_over",
                "league": "NBA",
                "home_team": "Chicgo Bull", # Typo to avoid canonical match
                "away_team": "Maimi Heat", # Typo to avoid canonical match
                "game_date": "2023-11-01"
            }
        ])
        result, _ = _expand_live_odds_to_bet_rows(self.live_odds_df, theover_rows)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["market_type"], "total_over")
        self.assertEqual(result.iloc[0]["candidate_source"], "upload_matched")
        self.assertEqual(result.iloc[0]["orientation_source"], "fuzzy_match")

    def test_preserved_columns_present(self):
        theover_rows = pd.DataFrame([
            {
                "matchup_id": "2023-11-01|chicago bulls|miami heat",
                "market_type": "spread_home",
                "league": "NBA",
                "home_team": "Chicago Bulls",
                "away_team": "Miami Heat",
                "game_date": "2023-11-01"
            }
        ])
        result, _ = _expand_live_odds_to_bet_rows(self.live_odds_df, theover_rows)
        self.assertIn("market_type", result.columns)
        self.assertIn("candidate_source", result.columns)
        self.assertIn("orientation_source", result.columns)
        self.assertIn("upload_match_reason", result.columns)

if __name__ == '__main__':
    unittest.main()
