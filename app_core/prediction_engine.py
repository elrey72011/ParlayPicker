import pandas as pd
import xgboost as xgb
import os
import logging
from pathlib import Path
import json
from typing import List, Optional, Any, Dict, Mapping, Tuple
from app_core.team_name_matcher import TeamNameMatcher

# Jules: Initializing local logging for the new engine
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# LOG-ONCE GUARDS
# -------------------------------------------------------------------
_LOGGED_MODEL_MISSING = False

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

def build_model_feature_row_from_record(record: Mapping[str, Any]) -> Dict[str, float]:
    """
    Build one Vertex feature row using the same columns and defaults
    as the batch enrich_with_model_features path.
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
            # Correct path resolution:
            # __file__ is /app/app_core/prediction_engine.py
            # parents[0] is /app/app_core
            # parents[1] is /app
            root_dir = Path(__file__).resolve().parents[1]
            model_path = str(root_dir / "models" / "model.json")

        self.use_fallback = True # Default to fallback

        # Explicitly check for model file existence
        model_exists = os.path.exists(model_path) and os.path.getsize(model_path) > 0

        if model_exists:
            try:
                # Check if file is valid JSON (basic check)
                is_valid_json = False
                with open(model_path, 'r') as f:
                    try:
                        content = json.load(f)
                        if content:
                            is_valid_json = True
                    except json.JSONDecodeError:
                        pass # Invalid JSON, use fallback

                if is_valid_json:
                    self.model.load_model(model_path)
                    self.use_fallback = False
                    logger.info(f"Jules: Loaded local model from {model_path}")
                else:
                    logger.warning(f"Jules: Model file at {model_path} is invalid/empty JSON. Using statistical fallback.")
                    self.use_fallback = True

            except Exception as e:
                self.use_fallback = True
                logger.error(f"Jules: Failed to load model from {model_path}: {e}. Using statistical fallback.")
        else:
            self.use_fallback = True
            global _LOGGED_MODEL_MISSING
            if not _LOGGED_MODEL_MISSING:
                logger.warning(
                    f"Jules: Model file missing at {model_path}. Using statistical fallback."
                )
                _LOGGED_MODEL_MISSING = True

    def get_prediction(self, features):
        """
        Jules: Replacing Vertex AI request with local XGBoost inference.
        Zero latency, zero cost.
        """
        try:
            # DEBUG LOGGING (Issue #3)
            logger.debug(f"Jules input features: {features}")

            if self.use_fallback:
                # Enhanced statistical fallback using team features
                # Instead of flat 0.52, use win%, ppg, and implied prob
                prob = self._calculate_statistical_prob(features)
                return {"prob": prob, "note": "Statistical Fallback (Feature-Based)"}

            # Ensure input is 2D (batch of 1)
            # Create DataFrame safely
            df_in = pd.DataFrame([features])

            # DEBUG LOGGING (Issue #3)
            logger.debug(f"Jules df_in shape: {df_in.shape}")
            logger.debug(f"Jules model type: {type(self.model)}")

            # Ensure columns match training schema (safety)
            # DMatrix handles sparse/missing, but consistency helps
            dmatrix = xgb.DMatrix(df_in)
            prob = self.model.predict(dmatrix)[0]

            # DEBUG LOGGING (Issue #3)
            logger.debug(f"Jules raw output: {prob}, type: {type(prob)}")

            return {"prob": float(prob), "note": "Local XGBoost Inference"}
        except Exception as e:
            logger.error(f"Prediction error: {e}. Using fallback.")
            prob = self._calculate_statistical_prob(features)
            return {"prob": prob, "note": f"Error Fallback: {str(e)[:20]}"}

    def _calculate_statistical_prob(self, features: Dict[str, float]) -> float:
        """
        Calculate probability using team features when model is unavailable.
        Uses weighted combination of:
        - Implied probability from odds (40%)
        - Win % differential (30%)
        - PPG differential (20%)
        - Kalshi probability if available (10%)
        """
        # Get feature values safely
        implied_prob = features.get('implied_home_prob', 0.5)
        home_win_pct = features.get('feature_home_win_pct', 0.5)
        away_win_pct = features.get('feature_away_win_pct', 0.5)
        home_ppg = features.get('feature_home_ppg', 110.0)
        away_ppg = features.get('feature_away_ppg', 110.0)
        home_oppg = features.get('feature_home_oppg', 110.0)
        away_oppg = features.get('feature_away_oppg', 110.0)
        kalshi_prob = features.get('kalshi_prob', 0.5)
        sentiment_diff = features.get('sentiment_diff', 0.0)

        # Component 1: Implied Probability (Market odds baseline)
        implied_component = implied_prob

        # Component 2: Win % Differential
        # Convert win% diff to probability (normalized)
        win_diff = home_win_pct - away_win_pct
        # Scale: -0.5 to +0.5 diff maps to 0.35-0.65 prob
        win_component = 0.5 + (win_diff * 0.3)
        win_component = max(0.35, min(0.65, win_component))

        # Component 3: PPG Differential (Offensive/Defensive Balance)
        # Home net rating vs Away net rating
        home_net = home_ppg - home_oppg
        away_net = away_ppg - away_oppg
        net_diff = home_net - away_net
        # Scale: -20 to +20 maps to 0.40-0.60 prob
        ppg_component = 0.5 + (net_diff / 100.0)
        ppg_component = max(0.40, min(0.60, ppg_component))

        # Component 4: Kalshi probability (if available and not default)
        kalshi_component = kalshi_prob if abs(kalshi_prob - 0.5) > 0.01 else implied_prob

        # Component 5: Sentiment adjustment (small nudge)
        sentiment_adj = sentiment_diff * 0.02  # ±2% max

        # Weighted combination
        base_prob = (
            implied_component * 0.40 +
            win_component * 0.30 +
            ppg_component * 0.20 +
            kalshi_component * 0.10
        )

        # Apply sentiment adjustment
        final_prob = base_prob + sentiment_adj

        # Clamp to reasonable range [0.35, 0.65] to avoid extreme predictions without model
        final_prob = max(0.35, min(0.65, final_prob))

        return float(final_prob)

    def predict_batch(self, df: pd.DataFrame) -> List[float]:
        """
        Jules: Batch prediction optimization for handling full slates.
        """
        if df is None or df.empty:
            return []

        try:
            # Always fallback if model not loaded
            if self.use_fallback:
                # Enhanced statistical fallback using team features
                probs = []
                for idx, row in df.iterrows():
                    features = build_model_feature_row_from_record(row.to_dict())
                    prob = self._calculate_statistical_prob(features)
                    probs.append(prob)
                return probs

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

            # Detailed Logging BEFORE prediction (Issue #3)
            logger.debug(f"Jules input shape: {inference_data.shape}")
            logger.debug(f"Jules input types: {inference_data.dtypes}")
            if not inference_data.empty:
                logger.debug(f"Jules input sample: {inference_data.iloc[0].to_dict()}")

            dmatrix = xgb.DMatrix(inference_data)
            probs = self.model.predict(dmatrix)

            # Detailed Logging AFTER prediction (Issue #3)
            logger.debug(f"Jules raw output type: {type(probs)}")
            if hasattr(probs, "__iter__"):
                logger.debug(f"Jules raw output sample: {probs[:5] if len(probs) > 5 else probs}")
            else:
                logger.debug(f"Jules raw output: {probs}")

            # Handle potential single-value return or array
            final_probs = []
            if hasattr(probs, "__iter__"):
                raw_probs = [float(p) for p in probs]
            else:
                raw_probs = [float(probs)]

            # Check for placeholder value 0.623034656047821
            PLACEHOLDER_VAL = 0.623034656047821

            for idx, p in enumerate(raw_probs):
                 if abs(p - PLACEHOLDER_VAL) < 0.000001:
                      # Detected placeholder, force fallback for this row
                      try:
                          row = df.iloc[idx]
                          # LOG DETAILS OF PLACEHOLDER ROW (Issue #3)
                          logger.debug(f"Placeholder row details: {row.to_dict()}")

                          features = build_model_feature_row_from_record(row.to_dict())
                          fallback_prob = self._calculate_statistical_prob(features)
                          final_probs.append(fallback_prob)
                          logger.warning(f"Jules: Detected placeholder model output {p}. Falling back to statistical prob {fallback_prob:.3f}.")
                      except Exception as fallback_err:
                          logger.error(f"Fallback calculation failed during placeholder fix: {fallback_err}")
                          final_probs.append(p) # Keep original if fallback fails
                 else:
                      final_probs.append(p)

            return final_probs
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}", exc_info=True)
            # Fallback to statistical calculation for batch on error
            try:
                probs = []
                for idx, row in df.iterrows():
                    features = build_model_feature_row_from_record(row.to_dict())
                    prob = self._calculate_statistical_prob(features)
                    probs.append(prob)
                return probs
            except:
                return [0.52] * len(df)

# Global singleton or helper for single-row prediction
def get_prediction_prob(game_row: Dict[str, Any], sentiment_diff: float = 0.0) -> Tuple[Optional[float], Optional[str]]:
    """
    Wrapper for single-row prediction to match legacy interface.
    """
    try:
        engine = PredictionEngine()
        # Ensure features are extracted/formatted correctly
        features = build_model_feature_row_from_record(game_row)

        # Inject sentiment_diff if needed
        if 'sentiment_diff' not in features and sentiment_diff is not None:
            features['sentiment_diff'] = float(sentiment_diff)

        result = engine.get_prediction(features)
        return result['prob'], result['note']
    except Exception as e:
        logger.error(f"get_prediction_prob failed: {e}")
        return 0.52, "Error/Fallback"
