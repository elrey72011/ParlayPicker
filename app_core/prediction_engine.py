import pandas as pd
import xgboost as xgb
import os
import logging
from pathlib import Path
from typing import List, Optional, Any, Dict, Mapping, Tuple
from app_core.team_name_matcher import TeamNameMatcher

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

def safefloat(val: Any) -> float:
    """Safely convert to float, defaulting to 0.0 on error/None."""
    if val is None:
        return 0.0
    try:
        f = float(val)
        if f != f: return 0.0 # NaN
        return f
    except (ValueError, TypeError):
        return 0.0

def build_vertex_feature_row_from_record(record: Mapping[str, Any]) -> Dict[str, float]:
    """
    Build one Vertex feature row using the same columns and defaults
    as the batch enrich_with_vertex_features path.
    PROB features -> default 0.5, others -> default 0.0.
    """
    row: Dict[str, float] = {}
    for col in VERTEX_FEATURE_COLUMNS:
        val = record.get(col)
        # Fallback: try removing 'feature_' prefix if exact key missing
        if val is None and col.startswith("feature_"):
             val = record.get(col.replace("feature_", ""))

        # PROB features must default to 0.5 (Neutral), STATS/COUNTS to 0.0
        default_val = 0.5 if "prob" in col else 0.0

        if val is not None:
             row[col] = safefloat(val)
        else:
             row[col] = default_val

    return row

def match_team_name(target: str, candidates: List[str], threshold: float = 80.0) -> Optional[str]:
    """
    Wrapper for TeamNameMatcher to support rapidfuzz/fuzzy matching.
    """
    return TeamNameMatcher.match_team(target, candidates, threshold=threshold/100.0)

class PredictionEngine:
    def __init__(self, model_path=None):
        self.model = xgb.Booster()

        # Robust path resolution relative to this file
        if model_path is None:
            # app_core/prediction_engine.py -> parents[1] = root -> models/model.json
            root_dir = Path(__file__).resolve().parents[1]
            model_path = str(root_dir / "models" / "model.json")

        if os.path.exists(model_path):
            try:
                self.model.load_model(model_path)
                self.use_fallback = False
                logger.info(f"Jules: Loaded local model from {model_path}")
            except Exception as e:
                self.use_fallback = True
                logger.error(f"Jules: Failed to load model from {model_path}: {e}. Using statistical fallback.")
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

# Global singleton or helper for single-row prediction
def get_prediction_prob(game_row: Dict[str, Any], sentiment_diff: float = 0.0) -> Tuple[Optional[float], Optional[str]]:
    """
    Wrapper for single-row prediction to match legacy interface.
    """
    try:
        engine = PredictionEngine()
        # Ensure features are extracted/formatted correctly
        features = build_vertex_feature_row_from_record(game_row)

        # Inject sentiment_diff if needed
        if 'sentiment_diff' not in features and sentiment_diff is not None:
            features['sentiment_diff'] = float(sentiment_diff)

        result = engine.get_prediction(features)
        return result['prob'], result['note']
    except Exception as e:
        logger.error(f"get_prediction_prob failed: {e}")
        return 0.52, "Error/Fallback"
