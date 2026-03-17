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



def _american_to_prob_safe(odds_val: Any) -> float | None:
    """Convert American odds to implied probability, returning None when unavailable."""
    try:
        odds = float(odds_val)
    except (TypeError, ValueError):
        return None
    if odds == 0:
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def _build_fallback_features_from_row(row_dict: Dict[str, Any]) -> Dict[str, float]:
    """Build fallback features with market-derived priors to avoid flat 0.5 outputs."""
    features = build_model_feature_row_from_record(row_dict)

    implied_candidates = [
        row_dict.get("implied_home_prob"),
        row_dict.get("market_probability"),
        row_dict.get("theover_probability"),
    ]
    implied_prob = None
    for candidate in implied_candidates:
        try:
            if candidate is not None:
                val = float(candidate)
                if val > 1.0:
                    val = val / 100.0
                if 0.0 < val < 1.0:
                    implied_prob = val
                    break
        except (TypeError, ValueError):
            continue

    if implied_prob is None:
        implied_prob = _american_to_prob_safe(row_dict.get("odds_american"))

    if implied_prob is not None:
        features["implied_home_prob"] = float(implied_prob)

    kalshi_candidates = [row_dict.get("kalshi_prob"), row_dict.get("kalshi_probability")]
    for candidate in kalshi_candidates:
        try:
            if candidate is not None:
                val = float(candidate)
                if 0.0 < val < 1.0:
                    features["kalshi_prob"] = val
                    break
        except (TypeError, ValueError):
            continue

    return features


def _clean_team_for_matchup(value: Any) -> str:
    team = str(value or "").lower()
    team = "".join(ch for ch in team if ch.isalnum())
    if team in {"sacremento", "sacrementokings", "sacramentokings"}:
        return "sacramento"
    return team


def _build_matchup_id(home: Any, away: Any) -> str:
    home_clean = _clean_team_for_matchup(home)
    away_clean = _clean_team_for_matchup(away)
    if not home_clean and not away_clean:
        return ""
    a, b = sorted([home_clean, away_clean])
    return f"{a}|{b}"


def _to_et_game_date(series: pd.Series) -> pd.Series:
    raw = series.astype("string")
    out = pd.Series([pd.NA] * len(series), index=series.index, dtype="string")

    # Preserve canonical date-only values as-is (already ET day keys).
    date_only_mask = raw.str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False)
    out.loc[date_only_mask] = raw.loc[date_only_mask]

    # Convert timestamp-like values from UTC -> ET day key.
    ts_mask = ~date_only_mask
    if ts_mask.any():
        dt_utc = pd.to_datetime(raw.loc[ts_mask], errors="coerce", utc=True)
        dt_et = dt_utc.dt.tz_convert("America/New_York")
        out.loc[ts_mask] = dt_et.dt.floor("D").dt.strftime("%Y-%m-%d")

    return out


def _series_or_default(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series([default] * len(df), index=df.index)
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
        self.use_fallback = True

        root_dir = Path(__file__).resolve().parents[1]
        execution_root = Path(os.path.abspath(os.getcwd()))
        candidate_paths = []
        if model_path:
            candidate_paths.append(Path(os.path.abspath(str(model_path))))
        candidate_paths.extend([
            Path(os.path.abspath(str(execution_root / "models" / "model.json"))),
            Path(os.path.abspath(str(execution_root / "model.json"))),
            Path(os.path.abspath(str(root_dir / "models" / "model.json"))),
            Path(os.path.abspath(str(root_dir / "models" / "xgboost_model.json"))),
        ])

        resolved_model_path: Path | None = None
        for candidate in candidate_paths:
            try:
                candidate_abs = candidate.expanduser().resolve()
            except Exception:
                candidate_abs = Path(candidate)
            if candidate_abs.exists() and candidate_abs.is_file():
                resolved_model_path = candidate_abs
                break

        if resolved_model_path is None:
            global _LOGGED_MODEL_MISSING
            if not _LOGGED_MODEL_MISSING:
                logger.error("[MODEL_LOAD] No model artifact found. Checked paths: %s", [str(c) for c in candidate_paths])
                _LOGGED_MODEL_MISSING = True
            return

        resolved_model_path = resolved_model_path.resolve()
        logger.info("[MODEL_LOAD] Attempting model load from %s", resolved_model_path)

        try:
            self.model.load_model(str(resolved_model_path))
            if is_model_trained(self.model):
                self.use_fallback = False
                logger.info("[MODEL_LOAD] XGBoost model loaded successfully from %s", resolved_model_path)
            else:
                logger.error("[MODEL_LOAD] Model loaded from %s but failed trained-model validation", resolved_model_path)
        except Exception as xgb_error:
            logger.error("[MODEL_LOAD] XGBoost.load_model failed for %s: %s", resolved_model_path, xgb_error)
            logger.error("[MODEL_LOAD] Traceback: %s", traceback.format_exc())
            try:
                loaded_model, framework = diagnose_model_type(str(resolved_model_path))
                if framework in ["pickle", "joblib"] and loaded_model is not None and is_model_trained(loaded_model):
                    self.model = loaded_model
                    self.use_fallback = False
                    logger.info("[MODEL_LOAD] Loaded %s model from %s", framework, resolved_model_path)
                else:
                    logger.error("[MODEL_LOAD] Fallback loader failed for %s (framework=%s)", resolved_model_path, framework)
            except Exception as fallback_error:
                logger.error("[MODEL_LOAD] Secondary model loading failed for %s: %s", resolved_model_path, fallback_error)

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
                logger.info("[MODEL_ROUTING] Triggered Statistical Fallback")
                fallback_prob = self._calculate_statistical_prob(features)
                logger.debug(
                    "Model fallback triggered. Returning statistical fallback probability %.4f",
                    fallback_prob,
                )
                return {"prob": float(fallback_prob), "note": "Statistical Fallback"}

            logger.info("[MODEL_ROUTING] Triggered XGBoost Inference")

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
                 logger.debug(f"Placeholder value detected ({prob:.3f}), skipping prediction.")
                 return {"prob": None, "note": "Fallback (Placeholder Detected)"}

            return {"prob": float(prob), "note": "Local Inference"}
        except Exception as e:
            logger.error(f"Prediction error: {e}. Using fallback.")
            logger.error(f"Exception details: {traceback.format_exc()}")
            return {"prob": None, "note": f"Error Fallback: {str(e)[:20]}"}

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

        # Prevent inference on dates > 60 days from historical matrix max date
        try:
            if "game_date" in df.columns:
                from config import DATA_DIR
                master_file = DATA_DIR / 'master_all_sports.csv'
                if master_file.exists():
                    master_df = pd.read_csv(master_file, usecols=["commence_time"])
                    max_hist_date = pd.to_datetime(master_df["commence_time"]).max()

                    df_dates = pd.to_datetime(df["game_date"], errors="coerce")
                    if (df_dates > max_hist_date + pd.Timedelta(days=60)).any():
                        logger.warning("Predict Batch: Predicting on dates beyond historical data limits. Features may be stale.")
                        # self.use_fallback = True  # Bypassed per user request
        except Exception as e:
            logger.error(f"Failed to validate historical date limits: {e}")

        try:
            working_df = df.copy()
            # Preserve label/identity columns needed by stale-feature fallback joins.
            if "league" not in working_df.columns:
                working_df["league"] = ""
            working_df["league"] = working_df["league"].astype(str).str.upper()
            if "game_date" not in working_df.columns:
                working_df["game_date"] = ""
            home_series = _series_or_default(working_df, "home_team", "")
            away_series = _series_or_default(working_df, "away_team", "")
            working_df["matchup_id"] = [
                _build_matchup_id(h, a)
                for h, a in zip(home_series, away_series)
            ]
            game_date_src = _series_or_default(working_df, "game_date", "")
            if game_date_src.astype("string").str.len().eq(0).all() and "commence_time" in working_df.columns:
                game_date_src = _series_or_default(working_df, "commence_time", "")
            working_df["game_date"] = _to_et_game_date(game_date_src)
            working_df["canonical_match_key"] = (
                _series_or_default(working_df, "league", "").astype("string").str.upper()
                + "|"
                + working_df["matchup_id"].astype("string")
                + "|"
                + working_df["game_date"].astype("string")
            )

            # Formula-based fallback if model is unavailable.
            if self.use_fallback:
                logger.info("[MODEL_ROUTING] Triggered Statistical Fallback in Predict Batch")
                logger.info(
                    f"Predict Batch: Model unavailable, generating statistical fallback probabilities for {len(df)} rows."
                )
                fallback_probs: List[float] = []
                for _, row in working_df.iterrows():
                    features = _build_fallback_features_from_row(row.to_dict())
                    fallback_probs.append(float(self._calculate_statistical_prob(features)))
                return fallback_probs

            # Select required columns while preserving missing columns as NaN.
            # This allows pre-inference validation to detect schedule/feature join failures.
            raw_inference_data = working_df.reindex(columns=VERTEX_FEATURE_COLUMNS).copy()
            raw_numeric = raw_inference_data.apply(pd.to_numeric, errors='coerce')

            # Merge Hardening: use sanitized team names for "Stale Feature Fallback"
            row_nan_ratio = raw_numeric.isna().sum(axis=1) / max(len(VERTEX_FEATURE_COLUMNS), 1)

            # Keep track of which rows use stale features
            used_stale_features = pd.Series(False, index=df.index)

            if row_nan_ratio.mean() > 0.5:
                logger.warning("Feature matrix mostly empty. Attempting Stale Feature Fallback (up to 7 days).")
                try:
                    from config import DATA_DIR
                    master_file = DATA_DIR / 'master_all_sports.csv'
                    if master_file.exists():
                        # Load historical features
                        hist_df = pd.read_csv(master_file)
                        if "commence_time" in hist_df.columns:
                            hist_df["commence_time"] = pd.to_datetime(hist_df["commence_time"], errors="coerce", utc=True)
                            now_utc = pd.Timestamp.utcnow()
                            hist_df = hist_df[hist_df["commence_time"] >= (now_utc - pd.Timedelta(days=7))]

                            # Clean team names in hist_df
                            if "home_team" in hist_df.columns and "away_team" in hist_df.columns:
                                hist_df["matchup_id"] = [
                                    _build_matchup_id(h, a)
                                    for h, a in zip(hist_df["home_team"], hist_df["away_team"])
                                ]
                                if "league" not in hist_df.columns:
                                    hist_df["league"] = ""
                                hist_df["league_norm"] = hist_df["league"].astype(str).str.upper()
                                hist_df["game_date"] = _to_et_game_date(hist_df["commence_time"])
                                hist_df["canonical_match_key"] = (
                                    hist_df["league_norm"].astype("string")
                                    + "|"
                                    + hist_df["matchup_id"].astype("string")
                                    + "|"
                                    + hist_df["game_date"].astype("string")
                                )

                                # Process each predominantly empty row
                                for idx in df.index:
                                    if row_nan_ratio[idx] > 0.5:
                                        row_matchup = str(working_df.at[idx, "matchup_id"]) if "matchup_id" in working_df.columns else ""
                                        row_league = str(working_df.at[idx, "league"]).upper() if "league" in working_df.columns else ""
                                        row_game_date = str(working_df.at[idx, "game_date"]) if "game_date" in working_df.columns else ""
                                        row_match_key = str(working_df.at[idx, "canonical_match_key"]) if "canonical_match_key" in working_df.columns else ""

                                        if row_matchup:
                                            if row_match_key:
                                                match = hist_df[hist_df["canonical_match_key"] == row_match_key]
                                            else:
                                                match = hist_df[hist_df["matchup_id"] == row_matchup]
                                                if row_league and "league" in hist_df.columns:
                                                    match = match[match["league_norm"] == row_league]
                                                if row_game_date and row_game_date.lower() not in {"", "<na>", "nan", "nat", "none"}:
                                                    match = match[match["game_date"] == row_game_date]
                                            # Find most recent match for this exact matchup (direction matters for home/away features)
                                            match = match.sort_values("commence_time", ascending=False)

                                            if not match.empty:
                                                latest = match.iloc[0]
                                                used_stale_features.at[idx] = True
                                                for col in VERTEX_FEATURE_COLUMNS:
                                                    if col in latest and pd.notna(latest[col]):
                                                        raw_numeric.at[idx, col] = float(latest[col])
                except Exception as e:
                    logger.error(f"Stale Feature Fallback failed: {e}")

                # Re-check NaN ratio after fallback
                row_nan_ratio_after = raw_numeric.isna().sum(axis=1) / max(len(VERTEX_FEATURE_COLUMNS), 1)
                if row_nan_ratio_after.mean() > 0.5:
                    raise ValueError(
                        "Feature matrix is empty due to schedule merge failure. "
                        "Aborting ML predictions to prevent baseline default (0.1906)."
                    )

            # Ensure proper casting and fillna to prevent errors - critical for preventing placeholder values
            inference_data = raw_numeric.fillna(0.0)

            # Store flag so calling code can retrieve it if needed
            self.last_batch_used_stale_features = used_stale_features.tolist()
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
                      logger.debug(f"Placeholder at index {idx}: omitting probability.")
                      final_probs.append(None)
                 else:
                      final_probs.append(p)

            return final_probs
        except ValueError as e:
            logger.error(f"Batch prediction failed: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}", exc_info=True)
            return [None] * len(df)

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
