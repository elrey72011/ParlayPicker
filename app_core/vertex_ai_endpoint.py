"""
app_core.vertex_ai_endpoint

Standalone local inference module replacing the Google Vertex AI wrapper.
Uses a local XGBoost model.

Exports:
    - VERTEX_FEATURE_COLUMNS: list[str]
    - predict_win_probabilities(df, feature_columns=None) -> list[float]
    - is_vertex_prediction_configured() -> bool
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np

# Streamlit integration (optional, for session state error logging)
try:
    import streamlit as st  # type: ignore
except ImportError:
    st = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# CONFIG & PATHS
# -------------------------------------------------------------------

# Resolve the root directory (parents[1] relative to app_core/vertex_ai_endpoint.py)
ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = str(ROOT_DIR / "models" / "model.json")

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

# -------------------------------------------------------------------
# CONFIG CHECK
# -------------------------------------------------------------------

def is_vertex_prediction_configured() -> bool:
    """
    Always returns True for local inference mode.
    This satisfies checks in the main app that verify if prediction is enabled.
    """
    return True

# -------------------------------------------------------------------
# PREDICTION ENGINE (LOCAL ONLY)
# -------------------------------------------------------------------

def predict_win_probabilities(df: pd.DataFrame, feature_cols: Optional[List[str]] = None, model_path: Optional[str] = None) -> List[float]:
    """
    Predict Home Win Probability using local XGBoost model.
    Falls back to blended statistical probability if model is missing.
    """
    if df is None or df.empty:
        return []

    # 1. Clean Data
    # Deduplicate columns to avoid XGBoost errors
    df = df.loc[:, ~df.columns.duplicated()].copy()

    if feature_cols is None:
        feature_cols = VERTEX_FEATURE_COLUMNS

    if model_path is None:
        model_path = MODEL_PATH

    # Check model existence and XGBoost availability
    use_model = False
    if xgb is not None:
        if Path(model_path).exists():
            use_model = True
        else:
            logger.warning(f"Local Model Inference: Model file missing at {model_path}. Using fallback.")
    else:
        logger.warning("Local Model Inference: XGBoost not installed. Using fallback.")

    # 2. Schema Enforcement: Ensure all features exist (needed for both model and fallback logic)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        # Fill missing with 0.0
        zeros = pd.DataFrame(0.0, index=df.index, columns=missing)
        df = pd.concat([df, zeros], axis=1)

    # 3. Local XGBoost Prediction
    if use_model:
        try:
            # Strict casting to float to match model expectation and prevent data type errors
            inference_data = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(float)
            dmatrix = xgb.DMatrix(inference_data)

            booster = xgb.Booster()
            booster.load_model(model_path)

            preds = booster.predict(dmatrix)

            logger.info(f"Local Model Inference: Generated {len(preds)} predictions")

            # Handle output shape (could be list or numpy array)
            if isinstance(preds, (list, np.ndarray)):
                return [float(p) for p in preds]

        except Exception as e:
            logger.error(f"Local Model Inference Failed: {e}. Switching to fallback.", exc_info=True)
            if st:
                try:
                    st.session_state["vertex_last_error"] = f"Local Inference Error: {e}"
                except:
                    pass
            # Fall through to fallback logic below

    # 4. Statistical Fallback
    # Formula: (implied_home_prob * 0.5) + (kalshi_prob * 0.5)
    # Both features are guaranteed to exist in df due to Step 2 schema enforcement

    logger.info("Using Statistical Fallback for predictions.")

    def _fallback_calc(row):
        try:
            imp = float(row.get("implied_home_prob", 0.5))
            kal = float(row.get("kalshi_prob", 0.5))
            return (imp * 0.5) + (kal * 0.5)
        except:
            return 0.5

    return df.apply(_fallback_calc, axis=1).tolist()
