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
            "kalshi_probability": 0.60,
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
        out = []
        for i, row in enumerate(rows):
            merged = {**self.base_row, **row}
            merged["matchup_id"] = f"MOCK_{i}"
            out.append(merged)
        return pd.DataFrame(out)

    def test_generic_totals_below_prob_threshold(self):
        # Generic totals below TOTAL_MIN_WIN_PROB (0.56) should fail
        # With new NBA rule, NBA is 0.58
        df = self._build_df([
            # Weak over (EV/edge met, but win_prob too low)
            {"league": "NFL", "market_type": "total_over", "expected_value": 0.05, "edge": 0.05, "calibrated_probability": 0.55, "best_pick": "Over 45.5", "home_team": "Team A", "away_team": "Team B"},
            # Strong over (EV/edge met, win_prob met) -- Note: For Agrees, we make gap >= 0.03
            {"league": "NFL", "market_type": "total_over", "expected_value": 0.05, "edge": 0.05, "calibrated_probability": 0.57, "kalshi_probability": 0.53, "best_pick": "Over 45.5", "home_team": "Team C", "away_team": "Team D"},
            # Weak under (EV/edge met, but win_prob too low)
            {"league": "NFL", "market_type": "total_under", "expected_value": 0.03, "edge": 0.03, "calibrated_probability": 0.55, "best_pick": "Under 45.5", "home_team": "Team E", "away_team": "Team F"},
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
        # NHL totals require NHL_TOTAL_MIN_WIN_PROB_STRICT (0.58) and penalty
        # For total_over: req_edge logic: max(BASELINE_MIN_EDGE + NHL_TOTAL_EXTRA_EDGE_PENALTY, TOTAL_OVER_MIN_EDGE) = max(0.02+0.01, 0.04) = 0.04. + NHL_TOTAL_OVER_ACTIONABLE_PENALTY (0.02) = 0.06.
        # For total_over: req_ev logic: max(0.01, 0.03) = 0.03. + NHL_TOTAL_OVER_ACTIONABLE_PENALTY (0.02) = 0.05.
        df = self._build_df([
            # NHL total over at 0.57 (fails strict NHL 0.58)
            {"league": "NHL", "market_type": "total_over", "expected_value": 0.05, "edge": 0.06, "calibrated_probability": 0.57, "best_pick": "Over 5.5", "home_team": "Team A", "away_team": "Team B"},
            # NHL total over at 0.58 but fails new penalty edge requirement (0.05 is < 0.06)
            {"league": "NHL", "market_type": "total_over", "expected_value": 0.05, "edge": 0.05, "calibrated_probability": 0.58, "kalshi_probability": 0.54, "best_pick": "Over 5.5", "home_team": "Team C", "away_team": "Team D"},
            # NHL total over at 0.58 and meets penalty edge requirement (0.06 >= 0.06) and EV req (0.05 >= 0.05)
            {"league": "NHL", "market_type": "total_over", "expected_value": 0.05, "edge": 0.06, "calibrated_probability": 0.58, "kalshi_probability": 0.54, "best_pick": "Over 5.5", "home_team": "Team I", "away_team": "Team J"},
            # NBA total over at 0.57 (fails NBA 0.58)
            {"league": "NBA", "market_type": "total_over", "expected_value": 0.05, "edge": 0.06, "calibrated_probability": 0.57, "kalshi_probability": 0.53, "best_pick": "Over 220.5", "home_team": "Team E", "away_team": "Team F"},
            # NBA total over at 0.58 (meets NBA 0.58), requires EV=0.03+0.02=0.05, Edge=0.04+0.02=0.06
            {"league": "NBA", "market_type": "total_over", "expected_value": 0.05, "edge": 0.06, "calibrated_probability": 0.58, "kalshi_probability": 0.54, "best_pick": "Over 220.5", "home_team": "Team G", "away_team": "Team H"},
        # MLB total under at 0.57 (meets TOTAL_UNDER_MIN_WIN_PROB 0.57), requires EV=0.02+0.02=0.04, Edge=0.03+0.02=0.05
        {"league": "MLB", "market_type": "total_under", "expected_value": 0.04, "edge": 0.05, "calibrated_probability": 0.57, "kalshi_probability": 0.53, "best_pick": "Under 8.5", "home_team": "Team K", "away_team": "Team L"},
        # MLB total under at 0.57 (meets TOTAL_UNDER_MIN_WIN_PROB 0.57), fails EV=0.02+0.02=0.04 (has 0.03)
        {"league": "MLB", "market_type": "total_under", "expected_value": 0.03, "edge": 0.05, "calibrated_probability": 0.57, "kalshi_probability": 0.53, "best_pick": "Under 8.5", "home_team": "Team M", "away_team": "Team N"},
        ])

        best = build_best_picks_df(df)

        # Team A (NHL 0.57) -> Below Threshold
        nhl_weak = best[best["home_team"] == "Team A"].iloc[0]
        self.assertEqual(nhl_weak["Pick_Status"], "Below Threshold")

        # Team C (NHL 0.58, but fails edge 0.06) -> Below Threshold
        nhl_edge_fail = best[best["home_team"] == "Team C"].iloc[0]
        self.assertEqual(nhl_edge_fail["Pick_Status"], "Below Threshold")

        # Team I (NHL 0.58, meets edge 0.06) -> Actionable
        nhl_strong = best[best["home_team"] == "Team I"].iloc[0]
        self.assertEqual(nhl_strong["Pick_Status"], "Actionable")

        # Team E (NBA 0.57) -> Below Threshold
        nba_weak = best[best["home_team"] == "Team E"].iloc[0]
        self.assertEqual(nba_weak["Pick_Status"], "Below Threshold")

        # Team G (NBA 0.58) -> Actionable
        nba_strong = best[best["home_team"] == "Team G"].iloc[0]
        self.assertEqual(nba_strong["Pick_Status"], "Actionable")

        # Team K (MLB Under 0.57, meets EV 0.04, Edge 0.05) -> Actionable
        mlb_strong_under = best[best["home_team"] == "Team K"].iloc[0]
        self.assertEqual(mlb_strong_under["Pick_Status"], "Actionable")

        # Team M (MLB Under 0.57, fails EV 0.04) -> Below Threshold
        mlb_weak_under = best[best["home_team"] == "Team M"].iloc[0]
        self.assertEqual(mlb_weak_under["Pick_Status"], "Below Threshold")

    def test_mlb_spreads_get_lighter_gate(self):
        # SIDE_MIN_WIN_PROB is 0.52 for normal spreads, MLB gets 0.50. Set Kalshi gap to trigger 'Agrees' (bypass overlays)
        df = self._build_df([
            # MLB Spread at 0.51 (meets lighter gate)
            {"league": "MLB", "market_type": "spread_home", "expected_value": 0.05, "edge": 0.05, "calibrated_probability": 0.51, "kalshi_probability": 0.45, "best_pick": "Team A -1.5", "home_team": "Team A", "away_team": "Team B"},
            # NBA Spread at 0.51 (fails normal 0.52 floor)
            {"league": "NBA", "market_type": "spread_home", "expected_value": 0.05, "edge": 0.05, "calibrated_probability": 0.51, "kalshi_probability": 0.45, "best_pick": "Team C -3.5", "home_team": "Team C", "away_team": "Team D"},
        ])

        best = build_best_picks_df(df)

        # Team A (MLB Spread 0.51) -> Actionable
        mlb_spread = best[best["home_team"] == "Team A"].iloc[0]
        self.assertEqual(mlb_spread["Pick_Status"], "Actionable")

        # Team C (NBA Spread 0.51) -> Below Threshold
        nba_spread = best[best["home_team"] == "Team C"].iloc[0]
        self.assertEqual(nba_spread["Pick_Status"], "Below Threshold")
        self.assertIn("Fails side minimum Win Probability", nba_spread["Status_Reason"])

    def test_total_under_gets_stricter_gate(self):
        # TOTAL_UNDER_MIN_WIN_PROB is 0.57. Also we need kalshi gap >= 0.03 to be "Agrees" to bypass neutral overlay
        df = self._build_df([
            # generic over at 0.56 (meets TOTAL_MIN_WIN_PROB 0.56), set Kalshi=0.50 so ML=0.56 implies gap=0.06 >= 0.03 (Agrees)
            {"league": "NFL", "market_type": "total_over", "expected_value": 0.05, "edge": 0.05, "calibrated_probability": 0.56, "kalshi_probability": 0.50, "best_pick": "Over 45.5", "home_team": "Team A", "away_team": "Team B"},
            # generic under at 0.56 (fails TOTAL_UNDER_MIN_WIN_PROB 0.57)
            {"league": "NFL", "market_type": "total_under", "expected_value": 0.05, "edge": 0.05, "calibrated_probability": 0.56, "kalshi_probability": 0.50, "best_pick": "Under 45.5", "home_team": "Team C", "away_team": "Team D"},
            # generic under at 0.57 (meets TOTAL_UNDER_MIN_WIN_PROB 0.57)
            {"league": "NFL", "market_type": "total_under", "expected_value": 0.05, "edge": 0.05, "calibrated_probability": 0.57, "kalshi_probability": 0.50, "best_pick": "Under 45.5", "home_team": "Team E", "away_team": "Team F"},
        ])

        best = build_best_picks_df(df)

        # Team A (Total Over 0.56) -> Actionable
        over_pick = best[best["home_team"] == "Team A"].iloc[0]
        self.assertEqual(over_pick["Pick_Status"], "Actionable")

        # Team C (Total Under 0.56) -> Below Threshold
        under_fail = best[best["home_team"] == "Team C"].iloc[0]
        self.assertEqual(under_fail["Pick_Status"], "Below Threshold")

        # Team E (Total Under 0.57) -> Actionable
        under_pass = best[best["home_team"] == "Team E"].iloc[0]
        self.assertEqual(under_pass["Pick_Status"], "Actionable")

    def test_ev_dampener_on_fallback_heavy_slates(self):
        # A total that barely meets EV threshold (e.g. 0.03 for total_over)
        df = self._build_df([
            # Over with exactly 0.03 EV and strong win prob
            {"league": "NFL", "market_type": "total_over", "expected_value": 0.03, "edge": 0.04, "calibrated_probability": 0.60, "kalshi_probability": 0.55, "best_pick": "Over 45.5", "home_team": "Team A", "away_team": "Team B"},
        ])

        # 1. Normal slate -> Actionable
        diags = {"is_fallback_heavy": False}
        best_normal = build_best_picks_df(df.copy(), diagnostics_out=diags)
        self.assertEqual(best_normal.iloc[0]["Pick_Status"], "Actionable")
        self.assertEqual(diags.get("ev_dampener_impact_count"), 0)

        # 2. Fallback-heavy slate -> Dampened EV is 0.03 * 0.85 = 0.0255 < 0.03, so Below Threshold
        diags_heavy = {"is_fallback_heavy": True}
        best_heavy = build_best_picks_df(df.copy(), diagnostics_out=diags_heavy)
        self.assertEqual(best_heavy.iloc[0]["Pick_Status"], "Below Threshold")
        self.assertEqual(diags_heavy.get("ev_dampener_impact_count"), 1)
        self.assertEqual(best_heavy.iloc[0]["expected_value"], 0.03) # Must not overwrite the raw column

    def test_spread_divergence_override(self):
        df = self._build_df([
            # Strong spread with divergence > 20% (Override applies)
            # ML=0.40, Kalshi=0.70 (diff=0.30)
            # win_prob=0.60, EV=0.04, Edge=0.05 (all >= override thresholds and >= Disagree overlay threshold since Kalshi=0.70 implies gap=-0.1 < -0.03 => Disagrees => needs prob>=0.60)
            {"league": "NBA", "market_type": "spread_home", "expected_value": 0.04, "edge": 0.05, "calibrated_probability": 0.60, "ml_probability": 0.40, "kalshi_probability": 0.70, "best_pick": "Team A -3.5", "home_team": "Team A", "away_team": "Team B"},

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
