import os
import sys
import pandas as pd
import logging
import numpy as np
from app_core.prediction_engine import PredictionEngine

logging.basicConfig(level=logging.DEBUG, format="%(message)s")

def test_inference():
    print("Testing ML inference pipeline directly...")

    # 1. Load the historical test data as input
    master_file = "data/master_all_sports.csv"
    if not os.path.exists(master_file):
        print("Master file missing, can't simulate features.")
        return

    df = pd.read_csv(master_file).head(150).copy()

    # Map to standard prediction schema roughly
    df = df.rename(columns={"sport": "league"})
    df['market_type'] = 'spread_home'
    df['odds_american'] = np.random.uniform(-150, 150, len(df))
    df['market_probability'] = np.random.uniform(0.4, 0.6, len(df))
    df['kalshi_probability'] = np.random.uniform(0.4, 0.6, len(df))

    from app_core.feature_processing import enrich_with_model_features
    # Enrich to get the features
    enriched = enrich_with_model_features(df, {})

    print(f"Enriched {len(enriched)} rows with model features.")

    # Count unique vectors
    from app_core.prediction_engine import VERTEX_FEATURE_COLUMNS
    feature_cols = [c for c in VERTEX_FEATURE_COLUMNS if c in enriched.columns]
    unique_vectors = enriched[feature_cols].apply(tuple, axis=1).nunique()
    print(f"Unique input feature vectors: {unique_vectors} out of {len(enriched)}")

    # Check default filled rows
    num_mostly_empty = (enriched[feature_cols].isna() | (enriched[feature_cols] == 0.0) | (enriched[feature_cols] == 0.5)).sum(axis=1) > (len(feature_cols) * 0.8)
    print(f"Rows heavily default-filled (>80% zeros/defaults/NaNs): {num_mostly_empty.sum()}")

    # Just call predict_batch normally to trigger our logging inside the engine
    engine = PredictionEngine()
    preds = engine.predict_batch(enriched)

    print(f"Got {len(preds)} predictions.")

if __name__ == "__main__":
    test_inference()
