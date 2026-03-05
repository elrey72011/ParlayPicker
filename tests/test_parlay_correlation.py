import unittest
import pandas as pd
from parlay_optimizer import ParlayOptimizer
import sys
import types

class TestParlayCorrelation(unittest.TestCase):
    def setUp(self):
        self.optimizer = ParlayOptimizer(model_dir='./models', min_edge=0.01)

    def test_calculate_parlay_probability(self):
        # Mock config with specific values
        import config
        config.SAME_LEAGUE_CORR_FACTOR = 0.95

        # Cross-league (should not be discounted)
        legs_diff_league = [
            {'prob': 0.6, 'league': 'NBA'},
            {'prob': 0.6, 'league': 'NFL'}
        ]
        prob_diff = self.optimizer.calculate_parlay_probability(legs_diff_league)
        self.assertAlmostEqual(prob_diff, 0.36)

        # Same league (should be discounted)
        legs_same_league = [
            {'prob': 0.6, 'league': 'NBA'},
            {'prob': 0.6, 'league': 'NBA'}
        ]
        prob_same = self.optimizer.calculate_parlay_probability(legs_same_league)
        self.assertAlmostEqual(prob_same, 0.36 * 0.95)

    def test_same_game_disallowed(self):
        # Master df for shotgun parlays
        data = {
            "Pick": ["Pick 1", "Pick 2"],
            "AI_Prob": [0.6, 0.6],
            "AI_Edge": [0.05, 0.05],
            "Game": ["Game_A", "Game_A"],
            "league": ["NBA", "NBA"],
            "Market": ["SPREAD", "TOTAL"],
            "odds": [-110, -110]
        }
        df = pd.DataFrame(data)

        import config
        config.SAME_GAME_DISALLOWED = True

        # Test generation with shotgun parlays method which creates 2-leg combinations
        res = self.optimizer.generate_shotgun_parlays(df)
        self.assertIsNone(res['best_overall']) # Should have failed to form a parlay because it's the same game

        # Now change config and ensure it does generate
        config.SAME_GAME_DISALLOWED = False
        res2 = self.optimizer.generate_shotgun_parlays(df)
        self.assertIsNotNone(res2['best_overall'])

if __name__ == '__main__':
    unittest.main()
