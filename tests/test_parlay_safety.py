import unittest
import pandas as pd

from core.parlay_safety import (
    conservative_joint_probability,
    has_unique_games,
    parlay_expected_value,
    production_candidate_mask,
    row_is_untrusted,
    shares_game,
)


class TestParlaySafety(unittest.TestCase):
    def test_fallback_rows_are_not_candidates(self):
        frame = pd.DataFrame([{
            "Pick_Status": "Actionable",
            "odds_american": -110,
            "effective_win_probability": 0.62,
            "edge": 0.04,
            "is_fallback": True,
        }])
        self.assertTrue(row_is_untrusted(frame.iloc[0]))
        self.assertFalse(bool(production_candidate_mask(frame).iloc[0]))

    def test_missing_production_evidence_is_not_synthesized(self):
        frame = pd.DataFrame([{
            "Pick_Status": "Actionable",
            "odds_american": -110,
            "effective_win_probability": 0.62,
        }])
        self.assertFalse(bool(production_candidate_mask(frame).iloc[0]))

    def test_joint_probability_uses_haircut(self):
        self.assertAlmostEqual(conservative_joint_probability([0.60, 0.60]), 0.3492)
        self.assertAlmostEqual(parlay_expected_value(0.3492, 3.0), 0.0476)

    def test_distinct_game_ids_override_shared_market_terms(self):
        left = {"matchup_id": "MLB|A|B|2026-07-27", "best_pick": "Cubs moneyline"}
        right = {"matchup_id": "MLB|C|D|2026-07-27", "best_pick": "Pirates moneyline"}
        self.assertFalse(shares_game(left, right))

    def test_same_game_is_conservatively_rejected(self):
        left = {"matchup_id": "MLB|A|B|2026-07-09"}
        right = {"matchup_id": "MLB|A|B|2026-07-09"}
        self.assertTrue(shares_game(left, right))

    def test_matching_teams_reject_provider_specific_event_ids(self):
        left = {
            "matchup_id": "odds-api-123",
            "away_team": "Chicago Cubs",
            "home_team": "Los Angeles Dodgers",
        }
        right = {
            "game_id": "novig-987",
            "away_team": "Los Angeles Dodgers",
            "home_team": "Chicago Cubs",
        }
        self.assertTrue(shares_game(left, right))

    def test_unique_game_validator_is_pairwise(self):
        legs = pd.DataFrame([
            {"matchup_id": "g1"},
            {"matchup_id": "g2"},
            {"matchup_id": "g1"},
        ])
        self.assertFalse(has_unique_games(legs))
        self.assertTrue(has_unique_games(legs.iloc[:2]))


if __name__ == "__main__":
    unittest.main()

