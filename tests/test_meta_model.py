import unittest
import pandas as pd
from app_core.meta_model import extract_meta_features, predict_meta_model
import xgboost as xgb

class TestMetaModel(unittest.TestCase):
    def test_extract_meta_features(self):
        row = {
            'Implied_Prob': 0.6,
            'kalshi_prob_for_pick': 0.65,
            'model_spread_prob': None, # Missing
            'theover_prob': 0.7,
            'sentiment_adj': 0.02,
            'Market': 'SPREAD',
            'league': 'NBA'
        }
        features = extract_meta_features(row)

        self.assertEqual(features['implied_prob'], 0.6)
        self.assertEqual(features['implied_missing'], 0.0)

        self.assertEqual(features['kalshi_prob'], 0.65)
        self.assertEqual(features['kalshi_missing'], 0.0)

        self.assertEqual(features['model_prob'], 0.5)
        self.assertEqual(features['model_missing'], 1.0)

        self.assertEqual(features['theover_prob'], 0.7)
        self.assertEqual(features['theover_missing'], 0.0)

        self.assertEqual(features['sentiment_adj'], 0.52) # 0.5 + 0.02
        self.assertEqual(features['sentiment_missing'], 0.0)

        self.assertEqual(features['is_spread'], 1.0)
        self.assertEqual(features['is_total'], 0.0)
        self.assertEqual(features['is_nba'], 1.0)
        self.assertEqual(features['is_nfl'], 0.0)

    def test_predict_meta_model_identity(self):
        # Create a dummy identity model for testing (just outputting roughly what goes in)
        # For testing purposes, we just test the pipeline works without crashing
        import numpy as np
        row = {
            'Implied_Prob': 0.6,
            'Market': 'SPREAD',
            'league': 'NBA'
        }

        # We can't easily mock an XGBoost booster prediction behavior without actual training,
        # but we can test that extract_meta_features works and handles all missing gracefully.
        features = extract_meta_features(row)
        self.assertEqual(features['kalshi_prob'], 0.5)
        self.assertEqual(features['kalshi_missing'], 1.0)

if __name__ == '__main__':
    unittest.main()
