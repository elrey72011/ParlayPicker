import unittest
from unittest.mock import patch

import pandas as pd
from core.streamlit_pipeline import _expand_live_odds_to_bet_rows, build_best_picks_df, BEST_PICK_COLUMNS
import app_core.weights_config as wc
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

    def test_default_config_excludes_moneylines_even_when_priced(self):
        live = self.live_odds_df.copy()
        live["novig_h2h_home_price"] = -135
        live["novig_h2h_away_price"] = 120

        with patch.object(wc, "ENABLE_MONEYLINE_BEST_AVAILABLE", False), patch.object(
            wc, "ENABLE_MONEYLINE_PARLAY_LEGS", False
        ):
            result, diag = _expand_live_odds_to_bet_rows(live, None)

        self.assertEqual(len(result), 4)
        self.assertFalse(result["market_type"].str.startswith("moneyline").any())
        self.assertEqual(diag["generated"]["moneyline_home"], 0)
        self.assertEqual(diag["generated"]["moneyline_away"], 0)

    def test_real_priced_moneylines_join_the_best_available_candidate_pool_when_enabled(self):
        live = self.live_odds_df.copy()
        live["novig_h2h_home_price"] = -135
        live["novig_h2h_away_price"] = 120

        with patch.object(wc, "ENABLE_MONEYLINE_BEST_AVAILABLE", True), patch.object(
            wc, "ENABLE_MONEYLINE_PARLAY_LEGS", False
        ):
            result, diag = _expand_live_odds_to_bet_rows(live, None)

        self.assertEqual(len(result), 6)
        self.assertEqual(
            set(result["market_type"]),
            {
                "spread_home", "spread_away", "total_over", "total_under",
                "moneyline_home", "moneyline_away",
            },
        )
        moneylines = result[result["market_type"].str.startswith("moneyline")]
        self.assertEqual(set(moneylines["odds_american"]), {-135.0, 120.0})
        self.assertEqual(diag["missing_live_moneyline_price"], 0)

    def test_missing_moneyline_prices_are_not_fabricated_at_minus_110(self):
        with patch.object(wc, "ENABLE_MONEYLINE_BEST_AVAILABLE", True), patch.object(
            wc, "ENABLE_MONEYLINE_PARLAY_LEGS", False
        ):
            result, diag = _expand_live_odds_to_bet_rows(self.live_odds_df, None)

        self.assertFalse(result["market_type"].str.startswith("moneyline").any())
        self.assertEqual(diag["missing_live_moneyline_price"], 2)

    def test_upload_match_enriches_uploaded_markets_without_erasing_live_markets(self):
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
        result, diag = _expand_live_odds_to_bet_rows(self.live_odds_df, theover_rows)
        self.assertEqual(len(result), 4)
        markets = set(result["market_type"])
        self.assertEqual(markets, {"spread_home", "spread_away", "total_over", "total_under"})
        source = result.set_index("market_type")["candidate_source"].to_dict()
        self.assertEqual(source["spread_away"], "upload_matched")
        self.assertEqual(source["total_under"], "upload_matched")
        self.assertEqual(source["spread_home"], "live_market_only")
        self.assertEqual(source["total_over"], "live_market_only")
        self.assertEqual(diag["rows_dropped_by_join"], 0)
        self.assertEqual(diag["rows_retained_without_upload_market"], 2)

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
        self.assertEqual(len(result), 4)
        selected = result[result["market_type"].eq("spread_away")].iloc[0]
        self.assertEqual(selected["candidate_source"], "upload_matched")
        self.assertEqual(selected["orientation_source"], "canonical_match")
        self.assertTrue(
            result.loc[~result["market_type"].eq("spread_away"), "candidate_source"]
            .eq("live_market_only")
            .all()
        )

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
        self.assertEqual(len(result), 4)
        selected = result[result["market_type"].eq("total_over")].iloc[0]
        self.assertEqual(selected["candidate_source"], "upload_matched")
        self.assertEqual(selected["orientation_source"], "fuzzy_match")
        self.assertTrue(
            result.loc[~result["market_type"].eq("total_over"), "candidate_source"]
            .eq("live_market_only")
            .all()
        )

    def test_mlb_novig_outlier_falls_back_to_real_priced_standard_consensus(self):
        live = pd.DataFrame([
            {
                "league": "MLB",
                "home_team": "Chicago White Sox",
                "away_team": "New York Yankees",
                "game_date": "2026-07-30",
                "matchup_id": "2026-07-30|chicago white sox|new york yankees",
                "commence_time_raw": "2026-07-30T18:11:00Z",
                "novig_home_point": -5.5,
                "novig_away_point": 5.5,
                "novig_home_price": pd.NA,
                "novig_away_price": pd.NA,
                "novig_h2h_home_price": -108,
                "novig_h2h_away_price": 106,
                "fanduel_home_point": 1.5,
                "fanduel_away_point": -1.5,
                "fanduel_home_price": -108,
                "fanduel_away_price": -112,
                "fanduel_h2h_home_price": -116,
                "fanduel_h2h_away_price": -102,
                "draftkings_home_point": 1.5,
                "draftkings_away_point": -1.5,
                "draftkings_home_price": -110,
                "draftkings_away_price": -110,
                "draftkings_h2h_home_price": -110,
                "draftkings_h2h_away_price": -110,
                "betmgm_home_point": 1.5,
                "betmgm_away_point": -1.5,
                "betmgm_home_price": -115,
                "betmgm_away_price": -105,
                "betmgm_h2h_home_price": -115,
                "betmgm_h2h_away_price": -105,
            }
        ])

        result, _ = _expand_live_odds_to_bet_rows(live, None)
        spreads = result[result["market_type"].str.startswith("spread")].set_index(
            "market_type"
        )

        self.assertEqual(float(spreads.loc["spread_home", "spread_line"]), 1.5)
        self.assertEqual(float(spreads.loc["spread_away", "spread_line"]), -1.5)
        self.assertEqual(float(spreads.loc["spread_home", "odds_american"]), -108.0)
        self.assertEqual(float(spreads.loc["spread_away", "odds_american"]), -112.0)
        self.assertTrue(spreads["odds_source"].eq("odds_api").all())
        self.assertFalse(spreads["odds_source"].str.contains("fallback", case=False).any())

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
