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
    """
    if df is None or df.empty:
        return []

    if xgb is None:
        logger.critical("XGBoost not installed. Cannot perform local inference.")
        return [0.5] * len(df)

    # 1. Clean Data
    # Deduplicate columns to avoid XGBoost errors
    df = df.loc[:, ~df.columns.duplicated()].copy()

    if feature_cols is None:
        feature_cols = VERTEX_FEATURE_COLUMNS

    if model_path is None:
        model_path = MODEL_PATH

    # Check model existence
    if not Path(model_path).exists():
        logger.critical(f"Local Model Inference CRITICAL: Model file missing at {model_path}")
        return [0.5] * len(df)

    # 2. Schema Enforcement: Ensure all features exist
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        # Fill missing with 0.0
        zeros = pd.DataFrame(0.0, index=df.index, columns=missing)
        df = pd.concat([df, zeros], axis=1)

    # 3. Local XGBoost Prediction
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

        # Should not be reached if predict works
        return [0.5] * len(df)

    except Exception as e:
        logger.critical(f"Local Model Inference CRITICAL: Prediction failed: {e}", exc_info=True)
        if st:
            try:
                st.session_state["vertex_last_error"] = f"Local Inference Error: {e}"
            except:
                pass
        return [0.5] * len(df)
