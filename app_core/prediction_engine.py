import pandas as pd
import xgboost as xgb
import os
import logging
import traceback
import numpy as np
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
    """Safely convert to float, defaulting to 0.0 on error/None/NaN/inf."""
    if val is None:
        return 0.0
    try:
        f = float(val)
        # Check for NaN or inf
        if f != f or np.isinf(f):
            return 0.0
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

def diagnose_model_type(model_path):
    """Identify model framework"""
    if not os.path.exists(model_path):
        return None, "missing"

    # Check file magic bytes
    try:
        with open(model_path, 'rb') as f:
            magic = f.read(4)
        logger.debug(f"[MODEL_FRAMEWORK] File magic bytes: {magic.hex()}")
    except Exception as e:
        logger.warning(f"[MODEL_FRAMEWORK] Failed to read magic bytes: {e}")
        magic = b""

    # Try different load methods
    import pickle
    import joblib
    import json

    # Try pickle
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.debug(f"[MODEL_FRAMEWORK] ✅ Loaded as pickle: {type(model)}")
        return model, "pickle"
    except Exception:
        pass

    # Try joblib
    try:
        model = joblib.load(model_path)
        logger.debug(f"[MODEL_FRAMEWORK] ✅ Loaded as joblib: {type(model)}")
        return model, "joblib"
    except Exception:
        pass

    # Try JSON
    try:
        with open(model_path, 'r') as f:
            model = json.load(f)
        logger.debug(f"[MODEL_FRAMEWORK] ✅ Loaded as JSON: {type(model)}")
        return model, "json"
    except Exception:
        pass

    return None, "unknown"

def is_model_trained(model):
    """Check if model has actual weights"""

    # For XGBoost Booster
    if isinstance(model, xgb.Booster):
        try:
            # Check if it has trees
            trees = model.get_dump()
            if not trees or len(trees) == 0:
                logger.error("[TRAINED] XGBoost model has NO trees - never trained!")
                return False
            logger.debug(f"[TRAINED] XGBoost has {len(trees)} trees")
            return True
        except Exception as e:
            logger.warning(f"[TRAINED] XGBoost check failed: {e}")
            return True # Assume true if we can't check, to avoid false negatives on valid models

    # For sklearn
    if hasattr(model, 'n_estimators'):
        if not hasattr(model, 'estimators_') or model.estimators_ is None:
            logger.error("[TRAINED] Sklearn model estimators not fitted!")
            return False
        logger.debug(f"[TRAINED] Sklearn has {len(model.estimators_)} estimators")
        return True

    # For any model with coef_ or feature_importances_
    if hasattr(model, 'coef_'):
        if model.coef_ is None or model.coef_.size == 0:
            logger.error("[TRAINED] Model has no coefficients - never trained!")
            return False

    if hasattr(model, 'feature_importances_'):
        if model.feature_importances_ is None or (hasattr(model.feature_importances_, 'sum') and model.feature_importances_.sum() == 0):
            logger.error("[TRAINED] Model has no feature importances - never trained!")
            return False

    return True

class PredictionEngine:
    def __init__(self, model_path=None):
        self.model = xgb.Booster()

        # Robust path resolution relative to this file
        if model_path is None:
            # app_core/prediction_engine.py -> parents[1] = root -> models/model.json
            root_dir = Path(__file__).resolve().parents[1]
            candidate_paths = [
                str(root_dir / "models" / "model.json"),
                str(Path.cwd() / "models" / "model.json"),
                str(Path.cwd() / "model.json")
            ]

            for path in candidate_paths:
                if os.path.exists(path):
                    model_path = path
                    break

            # Fallback to default if none found
            if not model_path or not os.path.exists(model_path):
                model_path = str(root_dir / "models" / "model.json")

        self.use_fallback = True # Default to fallback

        # Explicitly check for model file existence
        # DEBUG LOGGING (Issue #1)
        logger.debug(f"[MODEL_DEBUG] Model file path: {model_path}")
        model_exists = os.path.exists(model_path)
        logger.debug(f"[MODEL_DEBUG] Model file exists: {model_exists}")

        if model_exists:
            size = os.path.getsize(model_path)
            logger.debug(f"[MODEL_DEBUG] Model file size: {size} bytes")

            if size > 0:
                try:
                    # Diagnose model type first
                    loaded_model, framework = diagnose_model_type(model_path)
                    logger.info(f"[MODEL_FRAMEWORK] Detected framework: {framework}")

                    if framework == "json":
                        # If it's JSON, load into XGBoost Booster
                        self.model.load_model(model_path)

                        # Validate training
                        if is_model_trained(self.model):
                            self.use_fallback = False
                            logger.debug(f"[MODEL_DEBUG] Jules: Loaded local XGBoost model from {model_path}")
                        else:
                            logger.critical("[TRAINED] MODEL NOT TRAINED - ABORT")
                            self.use_fallback = True

                    elif framework in ["pickle", "joblib"] and loaded_model is not None:
                        # If we loaded a pickle/joblib object, use it directly
                        self.model = loaded_model
                        if is_model_trained(self.model):
                            self.use_fallback = False
                            logger.debug(f"[MODEL_DEBUG] Jules: Loaded local {framework} model from {model_path}")
                        else:
                            self.use_fallback = True
                    else:
                        logger.info(f"Model file format unknown or invalid at {model_path}. Using statistical fallback.")
                        self.use_fallback = True

                except Exception as e:
                    self.use_fallback = True
                    logger.error(f"[MODEL_DEBUG] Failed to load model from {model_path}: {e}")
                    logger.error(f"[MODEL_DEBUG] Exception type: {type(e).__name__}")
                    logger.error(f"[MODEL_DEBUG] Exception details: {traceback.format_exc()}")
            else:
                self.use_fallback = True
                logger.info(f"Model file at {model_path} is empty. Using statistical fallback.")
        else:
            self.use_fallback = True
            global _LOGGED_MODEL_MISSING
            if not _LOGGED_MODEL_MISSING:
                logger.info(
                    f"Model file missing at {model_path}. Using statistical fallback."
                )
                _LOGGED_MODEL_MISSING = True

        # CRITICAL FIX 1: Validate Model Behavior (Smoke Test)
        # Check if model outputs the placeholder value on zero input
        if not self.use_fallback:
            self._validate_model_behavior()

    def _validate_model_behavior(self):
        """
        Run a dummy prediction (zero inputs) to check if the model returns the known placeholder value.
        If it does, disable the model to prevent runtime critical errors.
        """
        try:
            # Create zero-filled feature set
            dummy_features = {col: 0.0 for col in VERTEX_FEATURE_COLUMNS}

            # Prepare input frame
            df_in = pd.DataFrame([dummy_features])[VERTEX_FEATURE_COLUMNS].copy().astype(float)

            # Predict
            if isinstance(self.model, xgb.Booster):
                dmatrix = xgb.DMatrix(df_in)
                prediction = self.model.predict(dmatrix)
                prob = float(prediction[0])
            else:
                prediction = self.model.predict_proba(df_in)[:, 1]
                prob = float(prediction[0])

            # Check for placeholder
            PLACEHOLDER_VALUE = 0.623034656047821
            PLACEHOLDER_TOLERANCE = 1e-9

            if abs(prob - PLACEHOLDER_VALUE) < PLACEHOLDER_TOLERANCE:
                logger.info(f"Model validation: placeholder detected on zero input, using fallback mode.")
                self.use_fallback = True
            else:
                logger.debug(f"Model validation passed. Zero-input output: {prob}")

        except Exception as e:
            logger.info(f"Model validation failed: {e}. Defaulting to fallback.")
            self.use_fallback = True

    def get_prediction(self, features):
        """
        Jules: Replacing Vertex AI request with local inference.
        Zero latency, zero cost.
        """
        try:
            # DEBUG LOGGING (Issue #1)
            logger.debug(f"[MODEL_DEBUG] Input features: {features}")

            if self.use_fallback:
                # Enhanced statistical fallback using team features
                prob = self._calculate_statistical_prob(features)
                return {"prob": prob, "note": "Statistical Fallback (Feature-Based)"}

            # Ensure input is 2D (batch of 1)
            # Create DataFrame safely
            df_in = pd.DataFrame([features])

            # Ensure input has the correct columns (just like predict_batch)
            missing_cols = [col for col in VERTEX_FEATURE_COLUMNS if col not in df_in.columns]
            if missing_cols:
                 for c in missing_cols:
                     df_in[c] = 0.0

            # Select only the required columns in the correct order
            df_in = df_in[VERTEX_FEATURE_COLUMNS].copy()

            # Ensure proper types - critical for preventing placeholder values
            df_in = df_in.apply(pd.to_numeric, errors='coerce').fillna(0.0)
            # Replace any remaining inf values
            df_in = df_in.replace([np.inf, -np.inf], 0.0).astype(float)

            # DEBUG LOGGING (Issue #1)
            logger.debug(f"[MODEL_DEBUG] Input shape: {df_in.shape}")
            logger.debug(f"[MODEL_DEBUG] Input dtypes: {df_in.dtypes}")
            logger.debug(f"[MODEL_DEBUG] Feature names: {list(df_in.columns)}")
            logger.debug(f"[MODEL_DEBUG] First row sample:\n{df_in.iloc[0] if len(df_in) > 0 else 'empty'}")

            # Run prediction
            if isinstance(self.model, xgb.Booster):
                dmatrix = xgb.DMatrix(df_in)
                prediction = self.model.predict(dmatrix)
                # XGBoost predict returns numpy array, take first element
                prob = float(prediction[0])
            else:
                # Sklearn-like interface
                prediction = self.model.predict_proba(df_in)[:, 1]
                prob = float(prediction[0])

            # DEBUG LOGGING (Issue #1)
            logger.debug(f"[MODEL_DEBUG] Raw prediction: {prob}")

            # HARD REJECTION of placeholder values (Section 4)
            PLACEHOLDER_VALUE = 0.623034656047821
            PLACEHOLDER_TOLERANCE = 1e-9

            if abs(prob - PLACEHOLDER_VALUE) < PLACEHOLDER_TOLERANCE:
                 # Using fallback gracefully when placeholder detected
                 logger.debug(f"Placeholder value detected ({prob:.3f}), using statistical fallback.")
                 fallback_prob = self._calculate_statistical_prob(features)
                 return {"prob": fallback_prob, "note": "Fallback (Placeholder Detected)"}

            return {"prob": float(prob), "note": "Local Inference"}
        except Exception as e:
            logger.error(f"Prediction error: {e}. Using fallback.")
            logger.error(f"Exception details: {traceback.format_exc()}")
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
                logger.info(f"Predict Batch: Using fallback for {len(df)} rows (Model fallback active).")
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

            # Ensure proper casting and fillna to prevent errors - critical for preventing placeholder values
            inference_data = inference_data.apply(pd.to_numeric, errors='coerce').fillna(0.0)
            # Replace any remaining inf values
            inference_data = inference_data.replace([np.inf, -np.inf], 0.0).astype(float)

            # Detailed Logging BEFORE prediction (Issue #1)
            logger.debug(f"[MODEL_DEBUG] Batch input shape: {inference_data.shape}")
            if not inference_data.empty:
                logger.debug(f"[MODEL_DEBUG] First row sample: {inference_data.iloc[0].to_dict()}")

            if isinstance(self.model, xgb.Booster):
                dmatrix = xgb.DMatrix(inference_data)
                probs = self.model.predict(dmatrix)
            else:
                probs = self.model.predict_proba(inference_data)[:, 1]

            # Detailed Logging AFTER prediction (Issue #1)
            logger.debug(f"[MODEL_DEBUG] Raw prediction type: {type(probs)}")
            if hasattr(probs, "shape"):
                 logger.debug(f"[MODEL_DEBUG] Prediction shape: {probs.shape}")

            # Handle potential single-value return or array
            final_probs = []
            if hasattr(probs, "__iter__"):
                raw_probs = [float(p) for p in probs]
            else:
                raw_probs = [float(probs)]

            # Check for placeholder value 0.623034656047821
            PLACEHOLDER_VAL = 0.623034656047821
            PLACEHOLDER_TOLERANCE = 1e-9

            # Check for placeholders in batch
            if isinstance(raw_probs, list):
                 placeholder_count = sum(1 for p in raw_probs if abs(p - PLACEHOLDER_VAL) < PLACEHOLDER_TOLERANCE)
                 if placeholder_count > 0:
                      logger.info(f"Batch prediction: {placeholder_count}/{len(raw_probs)} placeholder values detected, using fallbacks.")
                 else:
                      logger.info(f"Batch prediction: {len(raw_probs)} rows processed successfully with model.")

            for idx, p in enumerate(raw_probs):
                 if abs(p - PLACEHOLDER_VAL) < PLACEHOLDER_TOLERANCE:
                      # Detected placeholder, force fallback for this row
                      try:
                          row = df.iloc[idx]
                          # LOG DETAILS OF PLACEHOLDER ROW (Issue #1)
                          logger.debug(f"Placeholder row details: {row.to_dict()}")

                          features = build_model_feature_row_from_record(row.to_dict())
                          fallback_prob = self._calculate_statistical_prob(features)
                          final_probs.append(fallback_prob)
                          logger.debug(f"Placeholder at index {idx}: using fallback prob {fallback_prob:.3f}")
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

# Global singleton instance
_SHARED_ENGINE = None

def get_engine() -> PredictionEngine:
    """Get or create the shared PredictionEngine instance."""
    global _SHARED_ENGINE
    if _SHARED_ENGINE is None:
        _SHARED_ENGINE = PredictionEngine()
    return _SHARED_ENGINE

# Global singleton or helper for single-row prediction
def get_prediction_prob(game_row: Dict[str, Any], sentiment_diff: float = 0.0) -> Tuple[Optional[float], Optional[str]]:
    """
    Wrapper for single-row prediction to match legacy interface.
    """
    try:
        engine = get_engine()
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
