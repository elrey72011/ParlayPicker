import unittest
import pandas as pd
from app_core.results_ingestion import attach_results

class TestResultsIngestion(unittest.TestCase):
    def test_attach_results(self):
        master_data = {
            'league': ['NBA', 'NCAAB', 'NHL'],
            'Home': ['Los Angeles Lakers', "St. John's", 'Boston Bruins'],
            'Away': ['Boston Celtics', 'Villanova', 'Toronto Maple Leafs'],
            'Commence (UTC)': ['2026-03-04', '2026-03-04', '2026-03-04'],
            'Pick': ['Los Angeles Lakers -3.5', 'Over 145.5', 'Toronto Maple Leafs ML'],
            'spread_pick_line': [-3.5, None, None],
            'spread_pick_side': ['home', None, None],
            'total_pick_line': [None, 145.5, None],
            'total_pick_side': [None, 'over', None]
        }
        master_df = pd.DataFrame(master_data)

        results_data = {
            'league': ['NBA', 'NCAAB', 'NHL', 'NBA'],
            'home_team': ['Lakers', 'ST JOHNS', 'Bruins', 'Heat'], # Slight variations
            'away_team': ['Celtics', 'VILLANOVA', 'Maple Leafs', 'Magic'],
            'date': ['2026-03-04', '2026-03-04', '2026-03-04', '2026-03-04'],
            'home_score': [110, 75, 2, 100],
            'away_score': [100, 70, 3, 90]
        }
        results_df = pd.DataFrame(results_data)

        enriched_df = attach_results(master_df, results_df)

        self.assertEqual(len(enriched_df), 3)
        self.assertEqual(enriched_df.iloc[0]['actual_home_score'], 110.0)
        self.assertEqual(enriched_df.iloc[0]['actual_away_score'], 100.0)

        # Lakers -3.5: 110 - 100 = 10 -> 10 - 3.5 = +6.5 -> WIN
        self.assertEqual(enriched_df.iloc[0]['spread_result'], 'WIN')

        # Over 145.5: 75 + 70 = 145 -> LOSS
        self.assertEqual(enriched_df.iloc[1]['total_result'], 'LOSS')

        # Maple Leafs ML: 2 vs 3 -> WIN
        self.assertEqual(enriched_df.iloc[2]['ml_result'], 'WIN')

    def test_missing_match(self):
         master_df = pd.DataFrame({'league': ['NBA'], 'Home': ['Bulls'], 'Away': ['Knicks']})
         results_df = pd.DataFrame({'league': ['NBA'], 'home_team': ['Heat'], 'away_team': ['Magic'], 'home_score': [1], 'away_score': [2]})
         enriched_df = attach_results(master_df, results_df)
         self.assertTrue(pd.isna(enriched_df.iloc[0]['actual_home_score']))
         self.assertEqual(enriched_df.iloc[0]['spread_result'], 'N/A')

    def test_neutral_site_result_with_reversed_home_away_orientation(self):
        master_df = pd.DataFrame(
            {
                'league': ['NCAAF', 'NCAAF'],
                'home_team': ['Alabama State', 'Howard'],
                'away_team': ['Southern', 'Alabama Am'],
                'Commence (UTC)': ['2026-08-29', '2026-08-29'],
                'best_pick': ['Southern +19.5', 'Howard -1.5'],
                'spread_pick_line': [19.5, -1.5],
                'spread_pick_side': ['away', 'home'],
            }
        )
        results_df = pd.DataFrame(
            {
                'league': ['NCAAF', 'NCAAF'],
                'home_team': ['Southern Jaguars', 'Alabama A&M Bulldogs'],
                'away_team': ['Alabama State Hornets', 'Howard Bison'],
                'date': ['2026-08-29', '2026-08-29'],
                'home_score': [17, 24],
                'away_score': [30, 31],
            }
        )

        enriched_df = attach_results(master_df, results_df)

        self.assertEqual(enriched_df['actual_home_score'].tolist(), [30.0, 31.0])
        self.assertEqual(enriched_df['actual_away_score'].tolist(), [17.0, 24.0])
        self.assertEqual(enriched_df['spread_result'].tolist(), ['WIN', 'WIN'])
        self.assertEqual(enriched_df['Pick_Outcome'].tolist(), ['WIN', 'WIN'])

if __name__ == '__main__':
    unittest.main()
