import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from streamlit_app import select_best_spread_pick

class TestStreamlitSpreadLogic(unittest.TestCase):

    def test_spread_sign_home_favorite(self):
        # Home -8.5, Away +8.5
        # If model picks away (underdog), should return +8.5
        result = select_best_spread_pick(
            home_team="CSUB", away_team="CSUN",
            spread_line=-8.5,  # Home perspective
            prob_home_covers=0.35, prob_away_covers=0.65
        )
        self.assertEqual(result['pick_team'], "CSUN")
        self.assertEqual(result['pick_line'], 8.5)
        self.assertEqual(result['pick_label'], "CSUN +8.5")

        self.assertEqual(result['alt_team'], "CSUB")
        self.assertEqual(result['alt_line'], -8.5)
        self.assertEqual(result['alt_label'], "CSUB -8.5")

        # If model picks home (favorite), should return -8.5
        result_home = select_best_spread_pick(
            home_team="CSUB", away_team="CSUN",
            spread_line=-8.5,  # Home perspective
            prob_home_covers=0.65, prob_away_covers=0.35
        )
        self.assertEqual(result_home['pick_team'], "CSUB")
        self.assertEqual(result_home['pick_line'], -8.5)
        self.assertEqual(result_home['pick_label'], "CSUB -8.5")


    def test_spread_sign_away_favorite(self):
        # Home +1.5, Away -1.5
        # If model picks home (underdog), should return +1.5
        result = select_best_spread_pick(
            home_team="SEMO", away_team="Linden",
            spread_line=1.5,  # Home perspective (home is underdog)
            prob_home_covers=0.62, prob_away_covers=0.38
        )
        self.assertEqual(result['pick_team'], "SEMO")
        self.assertEqual(result['pick_line'], 1.5)
        self.assertEqual(result['pick_label'], "SEMO +1.5")

        self.assertEqual(result['alt_team'], "Linden")
        self.assertEqual(result['alt_line'], -1.5)
        self.assertEqual(result['alt_label'], "Linden -1.5")

        # If model picks away (favorite), should return -1.5
        result_away = select_best_spread_pick(
            home_team="SEMO", away_team="Linden",
            spread_line=1.5,  # Home perspective
            prob_home_covers=0.38, prob_away_covers=0.62
        )
        self.assertEqual(result_away['pick_team'], "Linden")
        self.assertEqual(result_away['pick_line'], -1.5)
        self.assertEqual(result_away['pick_label'], "Linden -1.5")

if __name__ == '__main__':
    unittest.main()
