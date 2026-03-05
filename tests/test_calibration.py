import unittest
import pandas as pd
from app_core.calibration import generate_calibration_dataset

class TestCalibration(unittest.TestCase):
    def test_generate_calibration_dataset(self):
        data = {
            'league': ['NBA', 'NBA', 'NBA', 'NBA', 'NCAAB'],
            'best_pick_type': ['SPREAD', 'SPREAD', 'SPREAD', 'TOTAL', 'SPREAD'],
            'final_probability': [0.52, 0.54, 0.57, 0.62, 0.72],
            'edge': [0.03, 0.04, 0.05, 0.06, 0.10],
            'spread_result': ['WIN', 'LOSS', 'PUSH', 'N/A', 'WIN'],
            'total_result': ['N/A', 'N/A', 'N/A', 'WIN', 'N/A']
        }
        df = pd.DataFrame(data)

        cal_df = generate_calibration_dataset(df)

        # We expect:
        # NBA SPREAD bucket [0.5, 0.55): 2 picks (1 WIN, 1 LOSS), win_rate 0.5
        # NBA SPREAD bucket [0.55, 0.6): 0 valid picks (PUSH is excluded)
        # NBA TOTAL bucket [0.6, 0.65): 1 pick (1 WIN), win_rate 1.0
        # NCAAB SPREAD bucket [0.7, 0.75): 1 pick (1 WIN), win_rate 1.0

        self.assertEqual(len(cal_df), 3)

        nba_spread_50 = cal_df[(cal_df['league'] == 'NBA') & (cal_df['market_type'] == 'SPREAD') & (cal_df['bucket'] == '[0.5, 0.55)')].iloc[0]
        self.assertEqual(nba_spread_50['num_picks'], 2)
        self.assertEqual(nba_spread_50['wins'], 1)
        self.assertEqual(nba_spread_50['empirical_win_rate'], 0.5)
        self.assertAlmostEqual(nba_spread_50['avg_predicted_prob'], 0.53)

        nba_total_60 = cal_df[(cal_df['league'] == 'NBA') & (cal_df['market_type'] == 'TOTAL') & (cal_df['bucket'] == '[0.6, 0.65)')].iloc[0]
        self.assertEqual(nba_total_60['num_picks'], 1)
        self.assertEqual(nba_total_60['empirical_win_rate'], 1.0)

        ncaab_spread_70 = cal_df[(cal_df['league'] == 'NCAAB') & (cal_df['market_type'] == 'SPREAD') & (cal_df['bucket'] == '[0.7, 0.75)')].iloc[0]
        self.assertEqual(ncaab_spread_70['num_picks'], 1)
        self.assertEqual(ncaab_spread_70['empirical_win_rate'], 1.0)

if __name__ == '__main__':
    unittest.main()
