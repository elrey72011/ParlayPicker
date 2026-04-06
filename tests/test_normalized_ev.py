import unittest
import pandas as pd
from core.streamlit_pipeline import build_best_picks_df

class TestNormalizedEVPicking(unittest.TestCase):
    def test_structural_bias_fixed(self):
        # We simulate a pool where totals are structurally inflated (higher mean EV).
        # We want to show that normalized Z-score EV fixes this so that an
        # excellent side pick can beat an average total pick, even if the
        # total pick has higher absolute EV.

        # We simulate 3 games.
        df = pd.DataFrame([
            {"matchup_id": "G1", "market_type": "spread_home", "expected_value": 0.04, "tier_score": 2, "league": "NBA", "home_team": "A", "away_team": "B"},
            {"matchup_id": "G1", "market_type": "total_over", "expected_value": 0.10, "tier_score": 2, "league": "NBA", "home_team": "A", "away_team": "B"},

            {"matchup_id": "G2", "market_type": "spread_home", "expected_value": 0.03, "tier_score": 2, "league": "NBA", "home_team": "C", "away_team": "D"},
            {"matchup_id": "G2", "market_type": "total_over", "expected_value": 0.14, "tier_score": 2, "league": "NBA", "home_team": "C", "away_team": "D"},

            {"matchup_id": "G3", "market_type": "spread_home", "expected_value": 0.01, "tier_score": 2, "league": "NBA", "home_team": "E", "away_team": "F"},
            {"matchup_id": "G4", "market_type": "spread_home", "expected_value": 0.00, "tier_score": 2, "league": "NBA", "home_team": "G", "away_team": "H"},

            {"matchup_id": "G3", "market_type": "total_over", "expected_value": 0.06, "tier_score": 2, "league": "NBA", "home_team": "E", "away_team": "F"},
            {"matchup_id": "G4", "market_type": "total_over", "expected_value": 0.10, "tier_score": 2, "league": "NBA", "home_team": "G", "away_team": "H"},
        ])

        # Mock what the df needs
        df["game_date"] = "2023-11-01"

        best = build_best_picks_df(df)
        # Note: build_best_picks_df returns a df where matchup_id is an index or column.
        # But wait, it recreates matchup_id based on game_date, home_team, away_team.

        g1_pick = best[(best["home_team"] == "A") & (best["away_team"] == "B")].iloc[0]
        self.assertEqual(g1_pick["market_type"], "spread_home")

        g2_pick = best[(best["home_team"] == "C") & (best["away_team"] == "D")].iloc[0]
        self.assertEqual(g2_pick["market_type"], "total_over")

if __name__ == "__main__":
    unittest.main()
