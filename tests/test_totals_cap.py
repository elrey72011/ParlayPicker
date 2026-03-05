import unittest
import pandas as pd
from streamlit_app import calculate_best_pick_metrics

class TestTotalsCap(unittest.TestCase):
    def test_totals_cap_applied(self):
        # We'll just verify the logic we put in `calculate_best_pick_metrics` works.
        # Note: testing full streamlit_app.py functions outside of environment might require mocking,
        # but since calculate_best_pick_metrics is a pure pandas manipulation, it should run.

        # Actually since calculate_best_pick_metrics has a lot of external dependencies, let's write a simple
        # test logic mimicking what happens inside it.
        from config import MAX_TOTAL_PROB

        data = {
            "Market": ["Total", "Spread", "Total"],
            "Pick": ["Over 145.5", "Lakers -5.5", "Under 200.5"],
            "Total & Pick": ["Over 145.5", None, "Under 200.5"],
            "Spread & Pick": [None, "Lakers -5.5", None],
            "total_prob_adj": [0.75, None, 0.65],
            "spread_prob_adj": [None, 0.75, None],
            "total_edge": [0.05, None, 0.05],
            "spread_edge": [None, 0.05, None],
            "total_implied_prob": [0.5, None, 0.5],
            "spread_implied_prob": [None, 0.5, None],
            "Home": ["A", "A", "A"],
            "Away": ["B", "B", "B"],
            "total_odds_valid": [True, True, True],
            "spread_odds_valid": [True, True, True],
        }
        df = pd.DataFrame(data)

        res = calculate_best_pick_metrics(df)

        # Row 0: Total over max prob. Should be capped.
        self.assertAlmostEqual(res.iloc[0]['final_prob'], MAX_TOTAL_PROB)

        # Row 1: Spread over max prob. Should NOT be capped.
        self.assertAlmostEqual(res.iloc[1]['final_prob'], 0.75)

        # Row 2: Total under max prob. Should NOT be capped.
        self.assertAlmostEqual(res.iloc[2]['final_prob'], 0.65)

if __name__ == '__main__':
    unittest.main()
