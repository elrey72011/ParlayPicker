import sys
import os
import pandas as pd
import logging

# Configure logging to stdout
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("app_core.prediction_engine")

try:
    from app_core.prediction_engine import PredictionEngine, get_prediction_prob
except ImportError:
    # Add current directory to sys.path to find app_core
    sys.path.append(os.getcwd())
    from app_core.prediction_engine import PredictionEngine, get_prediction_prob

def test_engine():
    print("--- Testing Prediction Engine ---")

    # 1. Initialize Engine
    print("Initializing PredictionEngine...")
    engine = PredictionEngine()

    # 2. Check if fallback is active
    print(f"Fallback active: {engine.use_fallback}")

    # 3. Test Single Prediction
    print("\nTesting Single Prediction...")
    mock_features = {
        "implied_home_prob": 0.6,
        "sentiment_diff": 0.1,
        "kalshi_prob": 0.55,
        "feature_home_win_pct": 0.7,
        "feature_away_win_pct": 0.3,
        "feature_home_ppg": 115.0,
        "feature_away_ppg": 105.0,
        "feature_home_oppg": 105.0,
        "feature_away_oppg": 115.0,
    }

    pred = engine.get_prediction(mock_features)
    print(f"Single Prediction Result: {pred}")

    if pred['prob'] == 0.623034656047821:
        print("FAIL: Exact placeholder value returned!")
    else:
        print("PASS: Prediction is not placeholder.")

    # 4. Test Batch Prediction
    print("\nTesting Batch Prediction...")
    df = pd.DataFrame([mock_features, mock_features])
    batch_preds = engine.predict_batch(df)
    print(f"Batch Predictions: {batch_preds}")

    if any(abs(p - 0.623034656047821) < 0.000001 for p in batch_preds):
        print("FAIL: Placeholder value found in batch!")
    else:
        print("PASS: No placeholders in batch.")

    # 5. Test with missing file simulation (if possible, but we rely on current state)

if __name__ == "__main__":
    test_engine()
