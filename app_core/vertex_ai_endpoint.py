"""
app_core.vertex_ai_endpoint

Standalone local inference module replacing the Google Vertex AI wrapper.
Uses a local XGBoost model or a statistical fallback.

Exports:
    - VERTEX_FEATURE_COLUMNS: list[str]
    - predict_win_probabilities(df, feature_columns=None) -> list[float]
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
# PREDICTION ENGINE (LOCAL ONLY)
# -------------------------------------------------------------------

def predict_win_probabilities(df: pd.DataFrame, feature_cols: Optional[List[str]] = None, model_path: Optional[str] = None) -> List[float]:
    """
    Predict Home Win Probability using local XGBoost model.
    Falls back to a statistical blend if the model cannot be loaded.
    """
    if df is None or df.empty:
        return []

    try:
        import xgboost as xgb
    except ImportError:
        logger.warning("XGBoost not installed. Using statistical fallback.")
        xgb = None

    # 1. Clean Data
    # Deduplicate columns to avoid XGBoost errors
    df = df.loc[:, ~df.columns.duplicated()].copy()

    if feature_cols is None:
        feature_cols = VERTEX_FEATURE_COLUMNS

    if model_path is None:
        model_path = MODEL_PATH

    # 2. Schema Enforcement: Ensure all features exist
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        # Fill missing with 0.0 (or appropriate defaults handled in feature_processing, but safe here)
        zeros = pd.DataFrame(0.0, index=df.index, columns=missing)
        df = pd.concat([df, zeros], axis=1)

    # 3. Try Local XGBoost Prediction
    if xgb is not None and Path(model_path).exists():
        try:
            # Cast to float to match model expectation
            inference_data = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
            dmatrix = xgb.DMatrix(inference_data)

            booster = xgb.Booster()
            booster.load_model(model_path)

            preds = booster.predict(dmatrix)

            # Handle output shape (could be list or numpy array)
            if isinstance(preds, (list, np.ndarray)):
                return [float(p) for p in preds]

            return [0.5] * len(df) # Should not happen if predict works

        except Exception as e:
            logger.error(f"Local XGBoost inference failed: {e}", exc_info=True)
            if st:
                try:
                    st.session_state["vertex_last_error"] = f"Local Inference Error: {e}"
                except:
                    pass
            # Proceed to fallback below
    else:
        status_msg = "Model file missing" if not Path(model_path).exists() else "XGBoost library missing"
        logger.warning(f"Local prediction skipped ({status_msg}). Using statistical fallback.")

    # 4. Statistical Fallback (The "Free AI")
    # Logic: Blend implied probability (market) and Kalshi (prediction market)
    # Defaulting to 0.5 if data is missing.

    def _fallback_calc(row):
        try:
            # Implied Prob
            imp = row.get("implied_home_prob")
            # If imp is NaN/None, try "Implied_Prob" (raw column) or default 0.5
            if imp is None or pd.isna(imp):
                imp = row.get("Implied_Prob")

            # Safe float conversion
            try:
                imp_val = float(imp)
            except (ValueError, TypeError):
                imp_val = 0.5

            # Kalshi Prob (strong signal)
            kal = row.get("kalshi_prob")
            try:
                kal_val = float(kal)
            except (ValueError, TypeError):
                kal_val = 0.5

            # Blend 50/50
            return (imp_val * 0.5) + (kal_val * 0.5)
        except Exception:
            return 0.5

    return df.apply(_fallback_calc, axis=1).tolist()
