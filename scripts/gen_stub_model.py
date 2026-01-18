import xgboost as xgb
import pandas as pd
import numpy as np
import json
import os
import sys

# Define the columns that the model expects (must match prediction_engine.py)
VERTEX_FEATURE_COLUMNS = [
    "implied_home_prob",
    "sentiment_diff",
    "kalshi_prob",
    "injuries_home_count",
    "injuries_away_count",
    "weather_flag",
    "feature_home_win_pct",
    "feature_home_ppg",
    "feature_home_oppg",
    "feature_home_streak",
    "feature_away_win_pct",
    "feature_away_ppg",
    "feature_away_oppg",
    "feature_away_streak",
    "feature_diff_win_pct",
    "feature_diff_ppg",
    "feature_diff_oppg",
    "feature_diff_last5",
    "feature_diff_streak",
    "feature_home_rest_days",
    "feature_away_rest_days",
]

def create_stub_model():
    print("Creating stub model...")
    # Create dummy training data
    # 100 samples
    X = pd.DataFrame(np.random.rand(100, len(VERTEX_FEATURE_COLUMNS)), columns=VERTEX_FEATURE_COLUMNS)
    # Binary target
    y = np.random.randint(0, 2, 100)

    # Train a simple model
    model = xgb.XGBClassifier(n_estimators=10, max_depth=3, use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)

    # Save the model
    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "model.json")

    # XGBoost saves to JSON
    model.save_model(output_path)
    print(f"Model saved to {output_path}")

    # Verify it can be loaded
    print("Verifying load...")
    booster = xgb.Booster()
    booster.load_model(output_path)
    print("Load successful.")

if __name__ == "__main__":
    create_stub_model()
