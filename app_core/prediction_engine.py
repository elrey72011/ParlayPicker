import pandas as pd
import xgboost as xgb
import os
import logging
from pathlib import Path
from typing import List, Optional

# Jules: Initializing local logging for the new engine
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# MODEL INPUT SCHEMA
# -------------------------------------------------------------------

# Columns the XGBoost model expects. Order matters for DMatrix.
VERTEX_FEATURE_COLUMNS: List[str] = [
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

class PredictionEngine:
    def __init__(self, model_path=None):
        self.model = xgb.Booster()

        # Robust path resolution relative to this file
        if model_path is None:
            # app_core/prediction_engine.py -> parents[1] = root -> models/model.json
            root_dir = Path(__file__).resolve().parents[1]
            model_path = str(root_dir / "models" / "model.json")

        if os.path.exists(model_path):
            self.model.load_model(model_path)
            self.use_fallback = False
            logger.info(f"Jules: Loaded local model from {model_path}")
        else:
            self.use_fallback = True
            logger.warning(f"Jules: Model file missing at {model_path}. Using statistical fallback.")

    def get_prediction(self, features):
        """
        Jules: Replacing Vertex AI request with local XGBoost inference.
        Zero latency, zero cost.
        """
        if self.use_fallback:
            # Basic statistical fallback (e.g., win rate average)
            return {"prob": 0.52, "note": "Statistical Fallback (No Model Found)"}

        # Ensure input is 2D (batch of 1)
        dmatrix = xgb.DMatrix(pd.DataFrame([features]))
        prob = self.model.predict(dmatrix)[0]
        return {"prob": float(prob), "note": "Local XGBoost Inference"}

    def predict_batch(self, df: pd.DataFrame) -> List[float]:
        """
        Jules: Batch prediction optimization for handling full slates.
        """
        if df is None or df.empty:
            return []

        if self.use_fallback:
            # Basic statistical fallback
            return [0.52] * len(df)

        try:
            # Ensure input has the correct columns
            missing_cols = [col for col in VERTEX_FEATURE_COLUMNS if col not in df.columns]
            if missing_cols:
                 # Add missing columns with default 0.0
                 for c in missing_cols:
                     df[c] = 0.0

            # Select only the required columns in the correct order
            inference_data = df[VERTEX_FEATURE_COLUMNS].copy()

            # Ensure proper casting and fillna to prevent errors
            inference_data = inference_data.apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(float)
            dmatrix = xgb.DMatrix(inference_data)
            probs = self.model.predict(dmatrix)
            return [float(p) for p in probs]
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}", exc_info=True)
            return [0.52] * len(df)
