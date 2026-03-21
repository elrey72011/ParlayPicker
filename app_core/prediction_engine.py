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

try:
    from rapidfuzz import fuzz as rapidfuzz_fuzz
except Exception:  # pragma: no cover - optional dependency
    rapidfuzz_fuzz = None

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


def clean_team_name(series: pd.Series) -> pd.Series:
    """
    Sanitizes team names for ultra-strict joining.
    Passes names through the global normalize_team_name function
    and then strips all non-alphanumeric characters to ensure
    '76ers' and 'Philadelphia 76ers' resolve accurately.
    """
    import re
    from core.team_mapper import normalize_team_name as _global_normalize

    if series is None or (hasattr(series, "empty") and series.empty):
        return series

    typo_map = {
        "sacramento": "sacramento",
        "sacremento": "sacramento",
        "sacramentokings": "sacramento",
        "sacrementokings": "sacramento",
        "sanantonio": "sanantonio",
        "philidelphia": "philadelphia",
        "phildelphia": "philadelphia",
        "newyorkknicks": "newyork",
    }

    def _clean_str(s):
        if not isinstance(s, str):
            if pd.isna(s):
                return ""
            s = str(s)
        # 1. Active global normalization
        s = _global_normalize(s)

        # 2. Local cleanup
        s = s.lower().strip()
        for k, v in typo_map.items():
            if s == k:
                s = v
        # Removes ALL spaces/punctuation (e.g., 'st. john's' -> 'stjohns')
        s = re.sub(r'[^a-z0-9]', '', s)
        return s

    # Handle Series
    if isinstance(series, pd.Series):
        return series.apply(_clean_str)

    # Handle scalar strings for backward compatibility in the file
    return _clean_str(series)

def _clean_team_for_matchup(value: Any) -> str:
    from core.team_mapper import normalize_team_name as _global_normalize
    norm_val = _global_normalize(str(value) if pd.notna(value) else "")
    return clean_team_name(norm_val)


def _normalize_identity_merge_keys(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Normalize merge-key identity columns as stripped pandas StringDtype."""
    out = df.copy()
    for key in keys:
        if key not in out.columns:
            out[key] = pd.Series([""] * len(out), index=out.index, dtype="string")
        out[key] = out[key].astype("string").str.strip()
    return out


def _build_matchup_id(home: Any, away: Any, date_key: Any = "") -> str:
    from core.team_mapper import normalize_team_name
    import re

    # 1. Normalize Team Name
    home_norm = normalize_team_name(str(home) if pd.notna(home) else "")
    away_norm = normalize_team_name(str(away) if pd.notna(away) else "")

    # 2. Strict Cleaning (Lowercase, strip non-alphanumerics)
    h_clean = re.sub(r"[^a-z0-9]", "", home_norm.lower())
    a_clean = re.sub(r"[^a-z0-9]", "", away_norm.lower())

    # 3. Typo Mapping
    typo_map = {
        "sacramento": "sacramento",
        "sacremento": "sacramento",
        "sacramentokings": "sacramento",
        "sacrementokings": "sacramento",
        "sanantonio": "sanantonio",
        "philidelphia": "philadelphia",
        "phildelphia": "philadelphia",
        "newyorkknicks": "newyork",
    }
    h_clean = typo_map.get(h_clean, h_clean)
    a_clean = typo_map.get(a_clean, a_clean)

    # 4. Uppercase
    h_upper = h_clean.upper()
    a_upper = a_clean.upper()

    # 5. Lexicographical Sorting
    team_a = h_upper if h_upper <= a_upper else a_upper
    team_b = a_upper if h_upper <= a_upper else h_upper

    if date_key:
        return f"{team_a}|{team_b}|{date_key}"
    return f"{team_a}|{team_b}"


def _to_et_game_date_string(series: pd.Series) -> pd.Series:
    """Aggressively convert datetimes or strings to local US/Eastern YYYY-MM-DD."""
    def _to_et_string(value: Any) -> str:
        if pd.isna(value) or not value:
            return ""

        # If it's already a clean YYYY-MM-DD string
        if isinstance(value, str) and len(value.strip()) == 10:
            return value.strip()

        try:
            ts = pd.Timestamp(value)
            if pd.isna(ts):
                return ""

            # Treat naive timestamps as UTC (from TheOddsAPI)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")

            # Convert to Eastern Time
            ts_et = ts.tz_convert("America/New_York")
            return ts_et.strftime("%Y-%m-%d")

        except Exception:
            # Fallback for unexpected formats
            return str(value)[:10] if isinstance(value, str) else ""

    return series.apply(_to_et_string).astype("string")


def _normalize_game_date_string(series: pd.Series) -> pd.Series:
    """Canonical YYYY-MM-DD string key used for schedule/feature joins."""
    return _to_et_game_date_string(series)


def _series_or_default(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series([default] * len(df), index=df.index)


def match_team_name(target: str, candidates: List[str], threshold: float = 80.0, league: Optional[str] = None) -> Optional[str]:
    """
    Wrapper for TeamNameMatcher to support rapidfuzz/fuzzy matching.
    """
    return TeamNameMatcher.match_team(target, candidates, threshold=threshold/100.0, league=league)

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
        home_ppg = features.get('feature_home_ppg', 0.5)
        away_ppg = features.get('feature_away_ppg', 0.5)
        home_oppg = features.get('feature_home_oppg', 0.5)
        away_oppg = features.get('feature_away_oppg', 0.5)
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
        # Scale: apply 0.2 multiplier
        ppg_component = 0.5 + (net_diff * 0.2)
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

        # Prevent inference on dates > 60 days from system date
        try:
            if "game_date" in df.columns:
                valid_upper_bound = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=60)

                # Normalize dataframe dates and remove timezone for comparison
                df_dates = pd.to_datetime(df["game_date"], errors="coerce", utc=True)

                if (df_dates > valid_upper_bound).any():
                    logger.warning(f"Predict Batch: Predicting on dates beyond dynamic historical data limits ({valid_upper_bound.strftime('%Y-%m-%d')}). Features may be stale.")
                    # self.use_fallback = True  # Bypassed per user request
        except Exception as e:
            logger.error(f"Failed to validate historical date limits: {e}")

        try:
            working_df = df.copy()
            # Preserve label/identity columns needed by stale-feature fallback joins.
            if "league" not in working_df.columns:
                working_df["league"] = ""
            working_df["league"] = (
                working_df["league"]
                .astype("string")
                .fillna("")
                .str.strip()
                .str.upper()
            )
            if "game_date" not in working_df.columns:
                working_df["game_date"] = ""
            from core.team_mapper import normalize_team_name as _global_normalize
            if "home_team" in working_df.columns:
                working_df["home_team"] = clean_team_name(working_df["home_team"].apply(lambda x: _global_normalize(str(x) if pd.notna(x) else "")).astype("string").fillna("").str.strip())
            if "away_team" in working_df.columns:
                working_df["away_team"] = clean_team_name(working_df["away_team"].apply(lambda x: _global_normalize(str(x) if pd.notna(x) else "")).astype("string").fillna("").str.strip())
            working_df = _normalize_identity_merge_keys(working_df, ["league", "home_team", "away_team"])
            home_series = _series_or_default(working_df, "home_team", "")
            away_series = _series_or_default(working_df, "away_team", "")
            working_df["matchup_id"] = [
                _build_matchup_id(h, a)
                for h, a in zip(home_series, away_series)
            ]
            game_date_src = _series_or_default(working_df, "game_date", "")
            if game_date_src.astype("string").str.len().eq(0).all() and "commence_time" in working_df.columns:
                game_date_src = _series_or_default(working_df, "commence_time", "")

            # FORCE EASTERN TIME YYYY-MM-DD FOR HISTORICAL JOIN
            working_df["game_date"] = _to_et_game_date_string(game_date_src)
            working_df["game_date_dt"] = pd.to_datetime(working_df["game_date"], errors="coerce").dt.tz_localize(None)

            # Create match key matching historical format
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

            # Fill missing implied probabilities BEFORE matrix validation so we don't crash
            if 'implied_home_prob' not in working_df.columns:
                working_df['implied_home_prob'] = pd.NA
            if 'kalshi_prob' not in working_df.columns:
                working_df['kalshi_prob'] = pd.NA

            if 'decimal_odds' in working_df.columns:
                working_df['implied_home_prob'] = pd.to_numeric(working_df['implied_home_prob'], errors='coerce').fillna(
                    1 / pd.to_numeric(working_df['decimal_odds'], errors='coerce')
                )

            # Fill with market_probability or 0.5 as absolute last resort
            if 'market_probability' in working_df.columns:
                working_df['implied_home_prob'] = pd.to_numeric(working_df['implied_home_prob'], errors='coerce').fillna(
                    pd.to_numeric(working_df['market_probability'], errors='coerce')
                )

            for idx, row in working_df.iterrows():
                # 1. Kalshi Prob
                k_prob = None
                for col in ['kalshi_probability', 'kalshi_prob']:
                    val = row.get(col)
                    if pd.notna(val) and val != "":
                        try:
                            numeric_k = float(val)
                            if numeric_k > 0.0:
                                k_prob = numeric_k
                                break
                        except Exception:
                            continue
                working_df.at[idx, 'kalshi_prob'] = k_prob if k_prob else 0.5

                # 2. Implied Home Prob
                i_prob = None
                for col in ['implied_home_prob', 'market_probability', 'home_price', 'odds_home', 'home_odds', 'odds_american']:
                    val = row.get(col)
                    if pd.notna(val) and val != "":
                        try:
                            numeric_val = float(val)
                            if numeric_val != 0.5 and numeric_val != 0.0:
                                if abs(numeric_val) >= 100:
                                    # Inline American Odds Conversion
                                    i_prob = 100.0 / (numeric_val + 100.0) if numeric_val > 0 else abs(numeric_val) / (abs(numeric_val) + 100.0)
                                    break
                                elif 0 < numeric_val <= 1.0:
                                    i_prob = numeric_val
                                    break
                                elif 1.0 < numeric_val < 100.0:
                                    i_prob = 1.0 / numeric_val
                                    break
                        except Exception:
                            continue
                working_df.at[idx, 'implied_home_prob'] = i_prob if i_prob else 0.5

            # Explicit cast to silence FutureWarnings before any remaining fillna
            working_df['kalshi_prob'] = pd.to_numeric(working_df['kalshi_prob'], errors='coerce')
            working_df['implied_home_prob'] = pd.to_numeric(working_df['implied_home_prob'], errors='coerce')

            # Select required columns while preserving missing columns as NaN.
            # This allows pre-inference validation to detect schedule/feature join failures.
            raw_inference_data = working_df.reindex(columns=VERTEX_FEATURE_COLUMNS).copy()
            raw_numeric = raw_inference_data.apply(pd.to_numeric, errors='coerce')

            # Merge Hardening: use sanitized team names for "Stale Feature Fallback"
            row_nan_ratio = raw_numeric.isna().sum(axis=1) / max(len(VERTEX_FEATURE_COLUMNS), 1)

            # Keep track of which rows use stale features
            used_stale_features = pd.Series(False, index=df.index)

            if row_nan_ratio.mean() > 0.5:
                logger.warning("Feature matrix mostly empty. Attempting Stale Feature Fallback (unlimited lookback).")
                try:
                    from config import DATA_DIR
                    master_file = DATA_DIR / 'master_all_sports.csv'
                    if master_file.exists():
                        # Load historical features
                        hist_df = pd.read_csv(master_file)
                        if "commence_time" in hist_df.columns:
                            hist_df["commence_time"] = pd.to_datetime(hist_df["commence_time"], errors="coerce", utc=True)

                            # Clean team names in hist_df
                            if "home_team" in hist_df.columns and "away_team" in hist_df.columns:
                                # Aggressive formatting: normalize, strip non-alphanumeric, lowercase, apply typo map
                                from core.team_mapper import normalize_team_name as _global_normalize
                                hist_df["home_team"] = clean_team_name(hist_df["home_team"].apply(lambda x: _global_normalize(str(x) if pd.notna(x) else "")).astype("string").fillna("").str.strip())
                                hist_df["away_team"] = clean_team_name(hist_df["away_team"].apply(lambda x: _global_normalize(str(x) if pd.notna(x) else "")).astype("string").fillna("").str.strip())

                                # Use the normalized names directly for match building
                                hist_df["matchup_id"] = [
                                    _build_matchup_id(h, a)
                                    for h, a in zip(hist_df["home_team"], hist_df["away_team"])
                                ]

                                if "league" not in hist_df.columns:
                                    if "sport" in hist_df.columns:
                                        hist_df["league"] = hist_df["sport"]
                                    else:
                                        logger.error("Historical data integrity failure: Neither 'league' nor 'sport' columns found in master CSV.")
                                        hist_df["league"] = ""
                                hist_df = _normalize_identity_merge_keys(hist_df, ["league", "home_team", "away_team"])
                                hist_df["league_norm"] = hist_df["league"].astype("string").fillna("").str.strip().str.upper()

                                # Aggressive date format coercion
                                hist_df["game_date"] = _normalize_game_date_string(hist_df["commence_time"])
                                hist_df["game_date_dt"] = pd.to_datetime(hist_df["game_date"], errors="coerce").dt.tz_localize(None)

                                hist_df["matchup_id_with_date"] = [
                                    _build_matchup_id(h, a, d)
                                    for h, a, d in zip(hist_df["home_team"], hist_df["away_team"], hist_df["game_date"])
                                ]

                                hist_df["canonical_match_key"] = (
                                    hist_df["league_norm"].astype("string")
                                    + "|"
                                    + hist_df["matchup_id_with_date"].astype("string")
                                )

                                # Process each predominantly empty row
                                for idx in df.index:
                                    if row_nan_ratio[idx] > 0.5:
                                        row_league = str(working_df.at[idx, "league"]).upper() if "league" in working_df.columns else ""
                                        row_game_date_dt = working_df.at[idx, "game_date_dt"] if "game_date_dt" in working_df.columns else pd.NaT

                                        # Strict YYYY-MM-DD
                                        raw_game_date = working_df.at[idx, "game_date"] if "game_date" in working_df.columns else ""
                                        row_game_date = _to_et_game_date_string(pd.Series([raw_game_date])).iloc[0] if raw_game_date else ""

                                        row_home = str(working_df.at[idx, "home_team"]) if "home_team" in working_df.columns else ""
                                        row_away = str(working_df.at[idx, "away_team"]) if "away_team" in working_df.columns else ""

                                        # Provide aggressive cleanup equivalent for the row
                                        row_home_clean = _clean_team_for_matchup(row_home)
                                        row_away_clean = _clean_team_for_matchup(row_away)

                                        # Force a perfectly clean matchup_id explicitly for the fallback lookup
                                        row_matchup = _build_matchup_id(row_home_clean, row_away_clean)

                                        # Create the aggressive match key dynamically
                                        row_matchup_with_date = _build_matchup_id(row_home_clean, row_away_clean, row_game_date)
                                        row_match_key = f"{row_league}|{row_matchup_with_date}" if row_league and row_game_date else ""

                                        # First priority: strict date AND matchup match
                                        match = pd.DataFrame()

                                        if row_match_key:
                                            match = hist_df[hist_df["canonical_match_key"].eq(row_match_key).fillna(False)]


                                        # Looser Fallback: rolling most recent matchup match (ignore date)
                                        if match.empty and row_matchup:
                                            match = hist_df[hist_df["matchup_id"].eq(row_matchup).fillna(False)]
                                            if row_league and "league_norm" in hist_df.columns:
                                                match = match[match["league_norm"].astype("string").str.upper().eq(row_league).fillna(False)]

                                            # We intentionally do not filter by valid_window here to allow "unlimited lookback"
                                            # to the most recent historical stats available regardless of date.

                                        # Fuzzy Fallback: if exact match fails, use rapidfuzz (or difflib) on cleaned names
                                        if match.empty and row_league:
                                            logger.info(f"Exact match failed for {row_matchup}. Attempting fuzzy match.")
                                            league_pool = hist_df[hist_df["league_norm"].astype("string").str.upper().eq(row_league).fillna(False)]
                                            if not league_pool.empty:
                                                best_score = -1
                                                best_match_id = None
                                                from difflib import SequenceMatcher
                                                import re

                                                clean_row = re.sub(r'[^a-z0-9]', '', str(row_matchup).lower())

                                                for cand_id in league_pool["matchup_id"].unique():
                                                    if pd.isna(cand_id) or not cand_id:
                                                        continue

                                                    clean_cand = re.sub(r'[^a-z0-9]', '', str(cand_id).lower())

                                                    # Compare the full matchup_id (e.g., LALAKERS|SACRAMENTO)
                                                    if rapidfuzz_fuzz is not None:
                                                        score = rapidfuzz_fuzz.token_sort_ratio(clean_row, clean_cand)
                                                    else:
                                                        score = SequenceMatcher(None, clean_row, clean_cand).ratio() * 100

                                                    if score > best_score:
                                                        best_score = score
                                                        best_match_id = cand_id

                                                if best_score >= 65 and best_match_id:
                                                    logger.info(f"Fuzzy match successful: {row_matchup} -> {best_match_id} (Score: {best_score:.1f})")
                                                    match = league_pool[league_pool["matchup_id"].eq(best_match_id).fillna(False)]

                                                    # We intentionally do not filter by valid_window here to allow "unlimited lookback"
                                                    # to the most recent historical stats available regardless of date.

                                        # Final logic to grab the most recent valid match found
                                        if not match.empty:
                                            # Ensure the direction of features (Home vs Away) is maintained
                                            # by matching on the actual team names being played.
                                            # We rely on sort_values to get the most recent valid entry for the rolling fallback.
                                            match = match.sort_values("commence_time", ascending=False)

                                            if not match.empty:
                                                latest = match.iloc[0]
                                                used_stale_features.at[idx] = True

                                                # Determine if roles are swapped in this historical game vs the current game
                                                hist_home = str(latest.get("home_team", "")).lower()
                                                current_home = row_home_clean.lower() if row_home_clean else ""

                                                roles_swapped = False
                                                if current_home and hist_home and current_home != hist_home:
                                                    roles_swapped = True

                                                for col in VERTEX_FEATURE_COLUMNS:
                                                    # Check exact match first
                                                    if col in latest and pd.notna(latest[col]):
                                                        val = float(latest[col])
                                                    # Bridge the prefix mapping gap
                                                    elif col.startswith("feature_") and col.replace("feature_", "") in latest and pd.notna(latest[col.replace("feature_", "")]):
                                                        val = float(latest[col.replace("feature_", "")])
                                                    else:
                                                        continue

                                                    if roles_swapped:
                                                        # Invert probabilities
                                                        if col in ["implied_home_prob", "kalshi_prob"]:
                                                            val = 1.0 - val
                                                        # Negate differentials
                                                        elif col == "sentiment_diff" or col.startswith("feature_diff_"):
                                                            val = -val
                                                        # Swap home/away prefixed stats
                                                        elif col.startswith("feature_home_"):
                                                            col = col.replace("feature_home_", "feature_away_")
                                                        elif col.startswith("feature_away_"):
                                                            col = col.replace("feature_away_", "feature_home_")

                                                    raw_numeric.at[idx, col] = val
                                        else:
                                            # SPLIT LOOKUP: Teams haven't played each other. Look up their most recent stats independently.

                                            # Strict check: Skip Split Lookup if league is missing to prevent cross-league pollution (e.g., NHL Utah vs NCAAB Utah)
                                            if not row_league or row_league == "":
                                                logger.warning(f"Split lookup aborted for {row_matchup} due to missing league. Resorting to neutral defaults.")
                                                continue

                                            logger.info(f"Split lookup triggered for {row_matchup}.")

                                            found_home = False
                                            found_away = False
                                            latest_home = None
                                            latest_away = None

                                            logger.debug(f"Split lookup mapping attempt. League: {row_league}, Home: {row_home_clean}, Away: {row_away_clean}")

                                            # 1. Look up Home Team's latest stats
                                            if row_home_clean and "league_norm" in hist_df.columns:
                                                home_pool = hist_df[hist_df["league_norm"].astype("string").str.upper().eq(row_league).fillna(False)]
                                                if not home_pool.empty:
                                                    home_games = home_pool[(home_pool["home_team"].eq(row_home_clean)) | (home_pool["away_team"].eq(row_home_clean))].copy()

                                                    # Allow unlimited lookback for latest available stats regardless of date
                                                    if not home_games.empty:
                                                        # Explicitly sort by date to guarantee we are grabbing the most recent performance
                                                        if "commence_time" in home_games.columns:
                                                            home_games = home_games.sort_values("commence_time", ascending=False)
                                                        latest_home = home_games.iloc[0]
                                                        found_home = True

                                            # 2. Look up Away Team's latest stats
                                            if row_away_clean and "league_norm" in hist_df.columns:
                                                away_pool = hist_df[hist_df["league_norm"].astype("string").str.upper().eq(row_league).fillna(False)]
                                                if not away_pool.empty:
                                                    away_games = away_pool[(away_pool["home_team"].eq(row_away_clean)) | (away_pool["away_team"].eq(row_away_clean))].copy()

                                                    # Allow unlimited lookback for latest available stats regardless of date
                                                    if not away_games.empty:
                                                        # Explicitly sort by date to guarantee we are grabbing the most recent performance
                                                        if "commence_time" in away_games.columns:
                                                            away_games = away_games.sort_values("commence_time", ascending=False)
                                                        latest_away = away_games.iloc[0]
                                                        found_away = True

                                            if not found_home:
                                                logger.debug(f"Split lookup missing team: {row_home_clean} in league {row_league}")
                                            if not found_away:
                                                logger.debug(f"Split lookup missing team: {row_away_clean} in league {row_league}")

                                            if found_home or found_away:
                                                used_stale_features.at[idx] = True

                                            logger.debug(f"Split lookup tracing: row_home_clean='{row_home_clean}', row_away_clean='{row_away_clean}', hist_df_homes={hist_df['home_team'].unique()[:5] if 'home_team' in hist_df.columns else 'N/A'}")

                                            # --- REFINED SPLIT LOOKUP LOGIC ---
                                            target_stats = ["win_pct", "ppg", "oppg", "streak", "rest_days", "implied_home_prob", "kalshi_prob"]
                                            diff_stats = ["feature_diff_win_pct", "feature_diff_ppg", "feature_diff_oppg", "feature_diff_streak"]

                                            # Identify historical role for current Home Team
                                            played_as_home_h = True
                                            hist_prefix_h = "home_"
                                            if latest_home is not None:
                                                played_as_home_h = (latest_home["home_team"] == row_home_clean)
                                                hist_prefix_h = "home_" if played_as_home_h else "away_"

                                            # Identify historical role for current Away Team
                                            played_as_home_a = True
                                            hist_prefix_a = "home_"
                                            if latest_away is not None:
                                                played_as_home_a = (latest_away["home_team"] == row_away_clean)
                                                hist_prefix_a = "home_" if played_as_home_a else "away_"

                                            for stat in target_stats:
                                                # 1. Map Home Team Stats
                                                h_val = 0.5
                                                if latest_home is not None:
                                                    col_h = stat if stat in ["implied_home_prob", "kalshi_prob"] else f"{hist_prefix_h}{stat}"
                                                    if col_h in latest_home and pd.notna(latest_home[col_h]):
                                                        h_val = float(latest_home[col_h])

                                                if stat in ["implied_home_prob", "kalshi_prob"] and not played_as_home_h:
                                                    h_val = 1.0 - h_val  # Role-based inversion

                                                # 2. Map Away Team Stats (to represent Home Perspective)
                                                a_val = 0.5
                                                if latest_away is not None:
                                                    col_a = stat if stat in ["implied_home_prob", "kalshi_prob"] else f"{hist_prefix_a}{stat}"
                                                    if col_a in latest_away and pd.notna(latest_away[col_a]):
                                                        a_val = float(latest_away[col_a])

                                                if stat in ["implied_home_prob", "kalshi_prob"]:
                                                    # If they were Home in history, we want the current Home team win prob (1 - their win prob)
                                                    # If they were Away in history, we want (1 - their away win prob) which is just implied_home_prob_hist
                                                    a_val = 1.0 - a_val if played_as_home_a else a_val

                                                # 3. Assign Averaged Value to Model Slot
                                                final_val = (h_val + a_val) / 2.0 if (found_home and found_away) else (h_val if found_home else (1.0 - a_val if stat in ["implied_home_prob", "kalshi_prob"] else a_val))

                                                if stat in ["implied_home_prob", "kalshi_prob"]:
                                                    raw_numeric.at[idx, stat] = final_val
                                                else:
                                                    raw_numeric.at[idx, f"feature_home_{stat}"] = final_val

                                                # 4. Map Away Team Stats (to represent Away Perspective)
                                                a_val_true = 0.5
                                                if latest_away is not None:
                                                    col_a = stat if stat in ["implied_home_prob", "kalshi_prob"] else f"{hist_prefix_a}{stat}"
                                                    if col_a in latest_away and pd.notna(latest_away[col_a]):
                                                        a_val_true = float(latest_away[col_a])

                                                if stat in ["implied_home_prob", "kalshi_prob"] and not played_as_home_a:
                                                    a_val_true = 1.0 - a_val_true # Role-based inversion

                                                h_val_true = 0.5
                                                if latest_home is not None:
                                                    col_h = stat if stat in ["implied_home_prob", "kalshi_prob"] else f"{hist_prefix_h}{stat}"
                                                    if col_h in latest_home and pd.notna(latest_home[col_h]):
                                                        h_val_true = float(latest_home[col_h])

                                                if stat in ["implied_home_prob", "kalshi_prob"]:
                                                    h_val_true = 1.0 - h_val_true if played_as_home_h else h_val_true

                                                final_away_val = (h_val_true + a_val_true) / 2.0 if (found_home and found_away) else (a_val_true if found_away else (1.0 - h_val_true if stat in ["implied_home_prob", "kalshi_prob"] else h_val_true))

                                                if stat not in ["implied_home_prob", "kalshi_prob"]:
                                                    raw_numeric.at[idx, f"feature_away_{stat}"] = final_away_val

                                            # Handle Differentials (Negation logic)
                                            # We also need to explicitly handle sentiment_diff if roles are swapped

                                            # Calculate primary differentials (Home - Away) explicitly handling missing columns

                                            # Get current mapped values or default if missing
                                            h_ppg = raw_numeric.at[idx, "feature_home_ppg"] if "feature_home_ppg" in raw_numeric.columns and pd.notna(raw_numeric.at[idx, "feature_home_ppg"]) else 0.5
                                            a_ppg = raw_numeric.at[idx, "feature_away_ppg"] if "feature_away_ppg" in raw_numeric.columns and pd.notna(raw_numeric.at[idx, "feature_away_ppg"]) else 0.5
                                            raw_numeric.at[idx, "feature_diff_ppg"] = h_ppg - a_ppg

                                            h_oppg = raw_numeric.at[idx, "feature_home_oppg"] if "feature_home_oppg" in raw_numeric.columns and pd.notna(raw_numeric.at[idx, "feature_home_oppg"]) else 0.5
                                            a_oppg = raw_numeric.at[idx, "feature_away_oppg"] if "feature_away_oppg" in raw_numeric.columns and pd.notna(raw_numeric.at[idx, "feature_away_oppg"]) else 0.5
                                            raw_numeric.at[idx, "feature_diff_oppg"] = h_oppg - a_oppg

                                            h_win_pct = raw_numeric.at[idx, "feature_home_win_pct"] if "feature_home_win_pct" in raw_numeric.columns and pd.notna(raw_numeric.at[idx, "feature_home_win_pct"]) else 0.5
                                            a_win_pct = raw_numeric.at[idx, "feature_away_win_pct"] if "feature_away_win_pct" in raw_numeric.columns and pd.notna(raw_numeric.at[idx, "feature_away_win_pct"]) else 0.5
                                            raw_numeric.at[idx, "feature_diff_win_pct"] = h_win_pct - a_win_pct

                                            h_streak = raw_numeric.at[idx, "feature_home_streak"] if "feature_home_streak" in raw_numeric.columns and pd.notna(raw_numeric.at[idx, "feature_home_streak"]) else 0.0
                                            a_streak = raw_numeric.at[idx, "feature_away_streak"] if "feature_away_streak" in raw_numeric.columns and pd.notna(raw_numeric.at[idx, "feature_away_streak"]) else 0.0
                                            raw_numeric.at[idx, "feature_diff_streak"] = h_streak - a_streak

                                            # Handle special cases
                                            raw_numeric.at[idx, "feature_diff_last5"] = raw_numeric.at[idx, "feature_diff_win_pct"] # Proxy for momentum

                                            # Sentiment diff lookup still requires file lookup if available
                                            h_sentiment_diff = 0.0
                                            if latest_home is not None:
                                                if "sentiment_diff" in latest_home and pd.notna(latest_home["sentiment_diff"]):
                                                    h_sentiment_diff = float(latest_home["sentiment_diff"])

                                            if not played_as_home_h:
                                                h_sentiment_diff = -h_sentiment_diff

                                            raw_numeric.at[idx, "sentiment_diff"] = h_sentiment_diff

                except Exception as e:
                    logger.error(f"Stale Feature Fallback failed: {e}")

                # Re-check NaN ratio after fallback
                row_nan_ratio_after = raw_numeric.isna().sum(axis=1) / max(len(VERTEX_FEATURE_COLUMNS), 1)

                # Calibration / Fallback override: Even if matrix is still somewhat sparse,
                # we don't abort completely unless we have NOTHING. Fill missing with 0.0 for stats, 0.5 for probs.

                if row_nan_ratio_after.mean() > 0.8:
                    # Log which specific features were missing (NaN)
                    missing_features = raw_numeric.columns[raw_numeric.isna().any()].tolist()
                    logger.warning(
                        f"Feature matrix is critically empty (>80% NaN) after unlimited lookback. "
                        f"Missing features causing failure: {missing_features}. "
                        f"Applying Hard Safety Net using implied market probabilities."
                    )
                    self.last_batch_used_stale_features = [True] * len(df)
                    self.last_batch_used_neutral_fallback = True

                    fallbacks = []
                    for idx, row in working_df.iterrows():
                        prob = None

                        # Try Bookmaker Odds FIRST (Arbitrage Fallback)
                        for col in ['implied_home_prob', 'market_probability', 'home_price', 'odds_home', 'home_odds', 'odds_american']:
                            val = row.get(col)
                            if pd.notna(val) and val != "":
                                try:
                                    numeric_val = float(val)
                                    if numeric_val != 0.5 and numeric_val != 0.0:
                                        if abs(numeric_val) >= 100:
                                            prob = 100.0 / (numeric_val + 100.0) if numeric_val > 0 else abs(numeric_val) / (abs(numeric_val) + 100.0)
                                            break
                                        elif 0 < numeric_val <= 1.0:
                                            prob = numeric_val
                                            break
                                        elif 1.0 < numeric_val < 100.0:
                                            prob = 1.0 / numeric_val
                                            break
                                except Exception:
                                    continue

                        # Final Resort
                        if prob is None:
                            prob = 0.5

                        fallbacks.append(float(prob))
                    return fallbacks

            # Ensure proper casting and fillna to prevent errors - critical for preventing placeholder values
            inference_data = raw_numeric.fillna(0.0)

            # Store flag so calling code can retrieve it if needed
            self.last_batch_used_stale_features = used_stale_features.tolist()
            self.last_batch_used_neutral_fallback = False
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

            # SPORTSBOOK ARBITRAGE OVERRIDE
            valid_probs = [p for p in final_probs if p is not None]
            if len(set(valid_probs)) <= 5:
                logger.warning("XGBoost returned mostly flat probabilities. Overriding with Sportsbook Implied Probabilities for Arbitrage.")
                final_probs = []
                for idx in working_df.index:
                    row = working_df.loc[idx]
                    i_val = row.get('implied_home_prob')
                    prob = float(i_val) if pd.notna(i_val) and str(i_val).strip() != "" else 0.50
                    
                    # THE FIX: Run picks through clean_team_name() so 'iowa state' perfectly matches 'iowastate'
                    pick_team = clean_team_name(str(row.get('pick_team', '')))
                    best_pick_raw = str(row.get('best_pick', '')).lower()
                    best_pick_clean = clean_team_name(best_pick_raw)

                    home_team = str(row.get('home_team', ''))
                    away_team = str(row.get('away_team', ''))
                    market_type = str(row.get('market_type', '')).lower().strip()
                    
                    # Aggressive check for Totals to prevent spread logic from crushing Over/Unders
                    if "total" in market_type or "over" in best_pick_raw or "under" in best_pick_raw:
                        # Totals: Kalshi is anchored to the OVER. Bump UP if Over, DOWN if Under.
                        if "over" in best_pick_raw:
                            prob = min(0.99, prob + 0.35)
                        else:
                            prob = max(0.01, prob - 0.35)
                    else:
                        # Spread/ML: Kalshi is anchored to the HOME team. Bump UP if Home, DOWN if Away.
                        is_home_pick = (pick_team == home_team) or (home_team in best_pick_clean and home_team != "")
                        is_away_pick = (pick_team == away_team) or (away_team in best_pick_clean and away_team != "")
                        
                        if is_home_pick:
                            prob = min(0.99, prob + 0.35)
                        elif is_away_pick:
                            prob = max(0.01, prob - 0.35)
                        else:
                            # Absolute fallback: Bump the favorite
                            if prob >= 0.5:
                                prob = min(0.99, prob + 0.35)
                            else:
                                prob = max(0.01, prob - 0.35)
                            
                    final_probs.append(prob)

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
