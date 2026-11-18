import unittest
import pandas as pd
from app_core.diagnostics import compute_daily_diagnostics

class TestDiagnostics(unittest.TestCase):
    def test_compute_daily_diagnostics(self):
        data = {
            'league': ['NBA', 'NBA', 'NBA', 'NCAAB', 'NHL'],
            'Market': ['SPREAD', 'SPREAD', 'TOTAL', 'SPREAD', 'TOTAL'],
            'final_probability': [0.52, 0.65, 0.72, 0.58, 0.61],
            'edge': [0.02, 0.05, 0.06, 0.04, 0.03],
            'spread_result': ['WIN', 'LOSS', 'N/A', 'WIN', 'N/A'],
            'total_result': ['N/A', 'N/A', 'WIN', 'N/A', 'LOSS']
        }
        df = pd.DataFrame(data)

        diag = compute_daily_diagnostics(df)

        self.assertIn('overall', diag)
        self.assertEqual(diag['overall']['total_picks'], 5)
        self.assertEqual(diag['overall']['wins'], 3)
        self.assertEqual(diag['overall']['win_rate'], 0.6)

        self.assertIn('by_group', diag)
        self.assertIn('by_bucket', diag)

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        diag = compute_daily_diagnostics(df)
        self.assertIn('error', diag)

if __name__ == '__main__':
    unittest.main()
