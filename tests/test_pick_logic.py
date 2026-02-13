import unittest
import sys
import os
import pandas as pd
import logging

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from streamlit_app import select_best_spread_pick, select_best_total_pick

# Setup logger
logging.basicConfig(level=logging.INFO)

class TestPickLogic(unittest.TestCase):

    def test_select_best_spread_pick_home_favored(self):
        # Case 1: Home favored (-8.5), Home covers (60% vs 40%)
        result = select_best_spread_pick(
            home_team="Team A",
            away_team="Team B",
            spread_line=-8.5,
            prob_home_covers=0.60,
            prob_away_covers=0.40
        )
        self.assertEqual(result['pick_team'], "Team A")
        self.assertEqual(result['pick_label'], "Team A -8.5")
        self.assertEqual(result['pick_prob'], 0.60)
        self.assertEqual(result['pick_side'], "home")
        self.assertTrue(result['pick_prob'] >= 0.50)

    def test_select_best_spread_pick_home_favored_away_covers(self):
        # Case 2: Home favored (-8.5), Away covers (60% vs 40%)
        result = select_best_spread_pick(
            home_team="Team A",
            away_team="Team B",
            spread_line=-8.5,
            prob_home_covers=0.40,
            prob_away_covers=0.60
        )
        self.assertEqual(result['pick_team'], "Team B")
        self.assertEqual(result['pick_label'], "Team B +8.5") # Flip sign
        self.assertEqual(result['pick_prob'], 0.60)
        self.assertEqual(result['pick_side'], "away")
        self.assertTrue(result['pick_prob'] >= 0.50)

    def test_select_best_spread_pick_away_favored(self):
        # Case 3: Away favored (+5.5 from home perspective), Home covers (70%)
        # spread_line = 5.5 (Home +5.5)
        result = select_best_spread_pick(
            home_team="Team A",
            away_team="Team B",
            spread_line=5.5,
            prob_home_covers=0.70,
            prob_away_covers=0.30
        )
        self.assertEqual(result['pick_team'], "Team A")
        self.assertEqual(result['pick_label'], "Team A +5.5")
        self.assertEqual(result['pick_prob'], 0.70)
        self.assertEqual(result['pick_side'], "home")

    def test_select_best_total_pick_over(self):
        # Case 4: Over is better
        result = select_best_total_pick(
            total_line=145.5,
            prob_over=0.55,
            prob_under=0.45
        )
        self.assertEqual(result['pick_label'], "Over 145.5")
        self.assertEqual(result['pick_prob'], 0.55)
        self.assertEqual(result['pick_side'], "Over")

    def test_select_best_total_pick_under(self):
        # Case 5: Under is better
        result = select_best_total_pick(
            total_line=145.5,
            prob_over=0.45,
            prob_under=0.55
        )
        self.assertEqual(result['pick_label'], "Under 145.5")
        self.assertEqual(result['pick_prob'], 0.55)
        self.assertEqual(result['pick_side'], "Under")

    def test_map_kalshi_prob_for_pick_matchup_string(self):
        from streamlit_app import map_kalshi_prob_for_pick

        # Scenario 1: Pick is Left Side (Yes)
        # Yes Side = "Lakers vs Celtics" (Yes means Lakers win)
        # Pick = "Lakers"
        # Prob = 0.60
        res = map_kalshi_prob_for_pick(0.60, "Lakers vs Celtics", "home", pick_team="Lakers", home_team="Lakers", away_team="Celtics")
        self.assertEqual(res, 0.60)

        # Scenario 2: Pick is Right Side (No)
        # Yes Side = "Lakers vs Celtics" (Yes means Lakers win)
        # Pick = "Celtics"
        # Prob = 0.60 (Lakers win prob)
        # Expected: 1 - 0.60 = 0.40
        res = map_kalshi_prob_for_pick(0.60, "Lakers vs Celtics", "away", pick_team="Celtics", home_team="Lakers", away_team="Celtics")
        self.assertEqual(res, 0.40)

        # Scenario 3: Non-matchup string
        # Yes Side = "Lakers"
        # Pick = "Lakers"
        res = map_kalshi_prob_for_pick(0.60, "Lakers", "home", pick_team="Lakers", home_team="Lakers", away_team="Celtics")
        self.assertEqual(res, 0.60)

        # Scenario 4: Matchup string with VS
        res = map_kalshi_prob_for_pick(0.70, "Kansas VS Duke", "home", pick_team="Kansas", home_team="Kansas", away_team="Duke")
        self.assertAlmostEqual(res, 0.70)

        res = map_kalshi_prob_for_pick(0.70, "Kansas VS Duke", "away", pick_team="Duke", home_team="Kansas", away_team="Duke")
        self.assertAlmostEqual(res, 0.30)

if __name__ == '__main__':
    unittest.main()
