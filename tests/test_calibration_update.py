import unittest
import pandas as pd
from core.streamlit_pipeline import build_best_picks_df

class TestCalibrationUpdate(unittest.TestCase):
    def setUp(self):
        # We need to simulate the structure expected by build_best_picks_df
        # Specifically, VALID_MARKETS includes side, spread, total_over, total_under, etc.
        self.base_row = {
            "market_type": "spread",
            "expected_value": 0.05,
            "edge": 0.05,
            "best_pick": "Test Pick -3.5",
            "league": "NBA",
            "home_team": "Team A",
            "away_team": "Team B",
            "game_date": "2024-01-01T00:00:00Z",
            "model_probability": 0.60,
            "ml_probability": 0.60,
            "kalshi_probability": None,
            "calibrated_probability": 0.60,
            "is_live_data": True,
            "odds_source": "fanduel",
            "spread_line": -3.5,
            "total_line": 220.5,
            "candidate_source": "ml",
            "orientation_source": "home",
            "upload_match_reason": "none",
        }

    def _build_df(self, rows):
        return pd.DataFrame([ {**self.base_row, **row} for row in rows ])

    def test_generic_totals_below_prob_threshold(self):
        # Generic totals below TOTAL_MIN_WIN_PROB (0.56) should fail
        df = self._build_df([
            # Weak over (EV/edge met, but win_prob too low)
            {"league": "NBA", "market_type": "total_over", "expected_value": 0.05, "edge": 0.05, "calibrated_probability": 0.55, "best_pick": "Over 220.5", "home_team": "Team A", "away_team": "Team B"},
            # Strong over (EV/edge met, win_prob met)
            {"league": "NBA", "market_type": "total_over", "expected_value": 0.05, "edge": 0.05, "calibrated_probability": 0.57, "best_pick": "Over 222.5", "home_team": "Team C", "away_team": "Team D"},
            # Weak under (EV/edge met, but win_prob too low)
            {"league": "NBA", "market_type": "total_under", "expected_value": 0.03, "edge": 0.03, "calibrated_probability": 0.55, "best_pick": "Under 220.5", "home_team": "Team E", "away_team": "Team F"},
        ])

        best = build_best_picks_df(df)
        self.assertEqual(len(best), 3)

        # Team A vs Team B (Weak over) -> Below Threshold
        weak_over = best[best["home_team"] == "Team A"].iloc[0]
        self.assertEqual(weak_over["Pick_Status"], "Below Threshold")

        # Team C vs Team D (Strong over) -> Actionable
        strong_over = best[best["home_team"] == "Team C"].iloc[0]
        self.assertEqual(strong_over["Pick_Status"], "Actionable")

        # Team E vs Team F (Weak under) -> Below Threshold
        weak_under = best[best["home_team"] == "Team E"].iloc[0]
        self.assertEqual(weak_under["Pick_Status"], "Below Threshold")


    def test_nhl_totals_require_stricter_threshold(self):
        # NHL totals require NHL_TOTAL_MIN_WIN_PROB (0.57)
        df = self._build_df([
            # NHL total over at 0.56 (meets generic, fails NHL)
            {"league": "NHL", "market_type": "total_over", "expected_value": 0.05, "edge": 0.05, "calibrated_probability": 0.565, "best_pick": "Over 5.5", "home_team": "Team A", "away_team": "Team B"},
            # NHL total over at 0.58 (meets NHL)
            {"league": "NHL", "market_type": "total_over", "expected_value": 0.05, "edge": 0.05, "calibrated_probability": 0.58, "best_pick": "Over 5.5", "home_team": "Team C", "away_team": "Team D"},
            # NBA total over at 0.565 (meets generic, would fail NHL if applied)
            {"league": "NBA", "market_type": "total_over", "expected_value": 0.05, "edge": 0.05, "calibrated_probability": 0.565, "best_pick": "Over 220.5", "home_team": "Team E", "away_team": "Team F"},
        ])

        best = build_best_picks_df(df)

        # Team A (NHL 0.565) -> Below Threshold
        nhl_weak = best[best["home_team"] == "Team A"].iloc[0]
        self.assertEqual(nhl_weak["Pick_Status"], "Below Threshold")

        # Team C (NHL 0.58) -> Actionable
        nhl_strong = best[best["home_team"] == "Team C"].iloc[0]
        self.assertEqual(nhl_strong["Pick_Status"], "Actionable")

        # Team E (NBA 0.565) -> Actionable
        nba_strong = best[best["home_team"] == "Team E"].iloc[0]
        self.assertEqual(nba_strong["Pick_Status"], "Actionable")

    def test_spread_divergence_override(self):
        df = self._build_df([
            # Strong spread with divergence > 20% (Override applies)
            # ML=0.40, Kalshi=0.70 (diff=0.30)
            # win_prob=0.56, EV=0.04, Edge=0.05 (all >= override thresholds)
            {"league": "NBA", "market_type": "spread_home", "expected_value": 0.04, "edge": 0.05, "calibrated_probability": 0.61, "ml_probability": 0.40, "kalshi_probability": 0.70, "best_pick": "Team A -3.5", "home_team": "Team A", "away_team": "Team B"},

            # Weak spread with divergence > 20% (Downgrade applies)
            # EV=0.02, Edge=0.03 (meets baseline, but not override thresholds)
            {"league": "NBA", "market_type": "spread_home", "expected_value": 0.02, "edge": 0.03, "calibrated_probability": 0.53, "ml_probability": 0.40, "kalshi_probability": 0.70, "best_pick": "Team C -3.5", "home_team": "Team C", "away_team": "Team D"},

            # Total with divergence > 20% (Override does not apply to totals)
            {"league": "NBA", "market_type": "total_over", "expected_value": 0.06, "edge": 0.06, "calibrated_probability": 0.60, "ml_probability": 0.40, "kalshi_probability": 0.70, "best_pick": "Over 220.5", "home_team": "Team E", "away_team": "Team F"},
        ])

        best = build_best_picks_df(df)

        # Team A (Strong Spread) -> Actionable
        strong_spread = best[best["home_team"] == "Team A"].iloc[0]
        self.assertEqual(strong_spread["Pick_Status"], "Actionable")
        self.assertIn("override applied", strong_spread["Status_Reason"])

        # Team C (Weak Spread) -> High Variance/Speculative
        weak_spread = best[best["home_team"] == "Team C"].iloc[0]
        self.assertEqual(weak_spread["Pick_Status"], "High Variance/Speculative")
        self.assertIn("diverge by > 20%", weak_spread["Status_Reason"])

        # Team E (Total) -> High Variance/Speculative
        strong_total = best[best["home_team"] == "Team E"].iloc[0]
        self.assertEqual(strong_total["Pick_Status"], "High Variance/Speculative")


    def test_exports_preserve_required_columns(self):
        df = self._build_df([
            {"league": "NBA", "market_type": "spread", "home_team": "Team A", "away_team": "Team B"},
        ])
        best = build_best_picks_df(df)

        required_cols = [
            "Pick_Status", "market_type", "candidate_source", "orientation_source",
            "upload_match_reason", "odds_source", "expected_value", "edge", "calibrated_probability"
        ]

        for col in required_cols:
            self.assertIn(col, best.columns)

if __name__ == '__main__':
    unittest.main()
