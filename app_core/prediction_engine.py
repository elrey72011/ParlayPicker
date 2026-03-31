import pandas as pd
import xgboost as xgb
import os
import logging
import traceback
import numpy as np
from pathlib import Path
import json
import hashlib
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

from config import VERTEX_CONFIG
# Columns the XGBoost model expects. Order matters for DMatrix.
VERTEX_FEATURE_COLUMNS: List[str] = VERTEX_CONFIG['feature_cols']

def _collapse_duplicate_columns(df: pd.DataFrame, critical_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Detects and collapses duplicate columns in a DataFrame row-wise, preferring non-null values.
    If multiple non-null values exist, the first one is taken. If ambiguity remains, the last column is kept.
    Logs diagnostics about the duplicates found.
    """
    if df.empty or not df.columns.duplicated().any():
        return df

    critical_cols = critical_cols or []
    dup_mask = df.columns.duplicated(keep=False)
    dup_cols = df.columns[dup_mask].unique()

    logger.warning(f"[_collapse_duplicate_columns] Found {len(dup_cols)} duplicate column names: {list(dup_cols)}")

    for col in dup_cols:
        count = (df.columns == col).sum()
        is_crit = "YES" if col in critical_cols else "NO"
        logger.warning(f"  - Column '{col}' appears {count} times (Critical: {is_crit})")

    # Create a new DataFrame to hold the collapsed columns
    collapsed_df = pd.DataFrame(index=df.index)

    # Iterate through unique columns
    for col in df.columns.unique():
        # Get all columns matching this name
        col_data = df.loc[:, df.columns == col]

        if isinstance(col_data, pd.DataFrame):
            if col_data.shape[1] > 1:
                # We have duplicate columns, collapse them
                # bfill along columns (axis=1) fills NaNs with the next valid value in the row
                # Then take the first column (iloc[:, 0]) which now has the first available non-null value
                collapsed = col_data.bfill(axis=1).iloc[:, 0]
                collapsed_df[col] = collapsed
            else:
                collapsed_df[col] = col_data.iloc[:, 0]
        else:
            # Single column as Series
            collapsed_df[col] = col_data

    # Hard assertion to ensure no duplicates remain
    assert not collapsed_df.columns.duplicated().any(), f"Duplicate columns still exist after collapse: {collapsed_df.columns[collapsed_df.columns.duplicated()].unique()}"

    logger.info(f"[_collapse_duplicate_columns] Successfully collapsed DataFrame from shape {df.shape} to {collapsed_df.shape}")
    return collapsed_df

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


def clean_team_name(series: Any) -> Any:
    """
    Sanitizes team names for ultra-strict joining.
    Passes names through the global normalize_team_name function,
    TeamNameMatcher.normalize, and then strips all non-alphanumeric
    characters to ensure '76ers' and 'Philadelphia 76ers' resolve accurately.
    """
    import re
    from core.team_mapper import normalize_team_name as _global_normalize
    from app_core.team_name_matcher import TeamNameMatcher

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

        # 2. TeamNameMatcher normalization
        s = TeamNameMatcher.normalize(s)

        # 3. Local cleanup
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
    # Delegate entirely to clean_team_name which already applies the full normalize stack
    return clean_team_name(value)


def _normalize_identity_merge_keys(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Normalize merge-key identity columns as stripped pandas StringDtype."""
    out = df.copy()
    for key in keys:
        if key not in out.columns:
            out[key] = pd.Series([""] * len(out), index=out.index, dtype="string")
        out[key] = out[key].astype("string").str.strip()
    return out


def _build_matchup_id(home: Any, away: Any, date_key: Any = "") -> str:
    # Use the robust clean_team_name function which correctly handles all mapping and stripping
    h_clean = clean_team_name(home)
    a_clean = clean_team_name(away)

    h_upper = h_clean.upper()
    a_upper = a_clean.upper()

    # Lexicographical Sorting
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
            Path(os.path.abspath(str(execution_root / "models" / "xgboost_model_v2.json"))),
            Path(os.path.abspath(str(execution_root / "models" / "model.json"))),
            Path(os.path.abspath(str(execution_root / "model.json"))),
            Path(os.path.abspath(str(root_dir / "models" / "xgboost_model_v2.json"))),
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
            PLACEHOLDER_VALUES = [0.623034656047821, 0.10671072453260422, 0.48637846, 0.31053704, 0.5622388124465942]
            PLACEHOLDER_TOLERANCE = 1e-7

            if any(abs(prob - val) < PLACEHOLDER_TOLERANCE for val in PLACEHOLDER_VALUES):
                logger.info(f"Model validation: placeholder detected ({prob:.4f}) on zero input, using fallback mode.")
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
            PLACEHOLDER_VALUES = [0.623034656047821, 0.10671072453260422, 0.48637846, 0.31053704, 0.5622388124465942]
            PLACEHOLDER_TOLERANCE = 1e-5

            if any(abs(prob - val) < PLACEHOLDER_TOLERANCE for val in PLACEHOLDER_VALUES):
                 # Using fallback gracefully when placeholder detected
                 logger.debug(f"Placeholder detected ({prob:.4f}), triggering Statistical Fallback.")
                 return {"prob": None, "note": "Fallback (Placeholder Detected)"}

            return {"prob": float(prob), "note": "Local Inference"}
        except Exception as e:
            logger.error(f"Prediction error: {e}. Using fallback.")
            logger.error(f"Exception details: {traceback.format_exc()}")
            return {"prob": None, "note": f"Error Fallback: {str(e)[:20]}"}

    def _calculate_statistical_prob(self, features: Dict[str, float], market_type: str = '') -> float:
        """
        Calculate probability using team features when model is unavailable.
        Uses weighted combination of:
        - Implied probability from odds (40%)
        - Win % differential (30%)
        - PPG differential (20%)
        - Kalshi probability if available (10%)
        """
        # League Average Healing (Example Baselines)
        LEAGUE_STATS = {
            "NBA": {"win_pct": 0.50, "ppg": 114.0},
            "NCAAB": {"win_pct": 0.50, "ppg": 72.0},
            "NHL": {"win_pct": 0.50, "ppg": 3.1},
            "MLB": {"win_pct": 0.50, "ppg": 4.5},
            "NFL": {"win_pct": 0.50, "ppg": 46.0}
        }

        league = features.get('league', '')
        if league == 'CBB':
            league = 'NCAAB'
        league_defaults = LEAGUE_STATS.get(league, {"win_pct": 0.50, "ppg": 0.0})
        default_win_pct = league_defaults["win_pct"]
        default_ppg = league_defaults["ppg"]

        # Get feature values safely, using league baselines for missing data
        implied_prob = features.get('implied_home_prob', 0.5)

        # Use league defaults if values are missing, 0.0, or 0.5 (which is the fallback from missing feature mappings)
        h_win_raw = features.get('feature_home_win_pct', default_win_pct)
        home_win_pct = h_win_raw if h_win_raw not in (0.0, 0.5) else default_win_pct

        a_win_raw = features.get('feature_away_win_pct', default_win_pct)
        away_win_pct = a_win_raw if a_win_raw not in (0.0, 0.5) else default_win_pct

        h_ppg_raw = features.get('feature_home_ppg', default_ppg)
        home_ppg = h_ppg_raw if h_ppg_raw not in (0.0, 0.5) else default_ppg

        a_ppg_raw = features.get('feature_away_ppg', default_ppg)
        away_ppg = a_ppg_raw if a_ppg_raw not in (0.0, 0.5) else default_ppg

        h_oppg_raw = features.get('feature_home_oppg', default_ppg)
        home_oppg = h_oppg_raw if h_oppg_raw not in (0.0, 0.5) else default_ppg

        a_oppg_raw = features.get('feature_away_oppg', default_ppg)
        away_oppg = a_oppg_raw if a_oppg_raw not in (0.0, 0.5) else default_ppg

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

        # Gaussian Noise Injection (Deterministic using team stats)
        # Add a deterministic but unique 'nudge' based on team stats to prevent identical probabilities
        home_team = features.get('home_team', 'home')
        away_team = features.get('away_team', 'away')
        date_string = features.get('game_date', 'date')

        nudge_seed = int(hashlib.md5(f"{home_team}{away_team}{date_string}{market_type}".encode()).hexdigest(), 16)
        nudge = ((nudge_seed % 200) - 100) / 5000.0  # ±2% nudge

        # Apply sentiment adjustment and deterministic noise
        final_prob = base_prob + sentiment_adj + nudge

        # Clamp to reasonable range [0.35, 0.65] to avoid extreme predictions without model
        final_prob = max(0.35, min(0.65, final_prob))

        return float(final_prob)

    def predict_batch(self, df: pd.DataFrame) -> List[float]:
        """
        Jules: Batch prediction optimization for handling full slates.
        """
        if df is None or df.empty:
            return []

        # DIAGNOSTICS & DEDUPLICATION: check for duplicate columns at the start of predict_batch
        logger.info(f"[predict_batch] Input DataFrame shape: {df.shape}")
        has_dupes = df.columns.duplicated().any()
        logger.info(f"[predict_batch] Duplicate columns exist in input: {has_dupes}")
        if has_dupes:
            dup_cols = df.columns[df.columns.duplicated()].unique()
            logger.warning(f"[predict_batch] Duplicate columns found in input: {list(dup_cols)}")
            critical_cols = ['implied_home_prob', 'kalshi_prob', 'market_probability', 'decimal_odds']
            critical_dupes = [col for col in dup_cols if col in critical_cols]
            if critical_dupes:
                logger.warning(f"[predict_batch] CRITICAL duplicate columns found in input: {critical_dupes}")

        from app_core.dataframe_utils import collapse_duplicate_columns
        df = collapse_duplicate_columns(df, critical_cols=['implied_home_prob', 'kalshi_prob', 'market_probability', 'decimal_odds', 'ml_probability'])
        logger.info(f"[predict_batch] Shape after collapse: {df.shape}")

        if df.columns.duplicated().any():
            remaining_dupes = df.columns[df.columns.duplicated()].unique()
            raise RuntimeError(f"Duplicate columns STILL exist in input df after collapse in predict_batch: {list(remaining_dupes)}")

        critical_cols_present = [c for c in ['implied_home_prob', 'kalshi_prob', 'market_probability', 'decimal_odds', 'ml_probability'] if c in df.columns]
        logger.info(f"[predict_batch] Critical columns present: {critical_cols_present}")

        LEAGUE_MAP = {"NCAAM": "NCAAB", "NCAAMB": "NCAAB", "NCAA MENS BASKETBALL": "NCAAB"}

        # Apply standard league mapping to original data immediately
        if "league" in df.columns:
            df["league"] = df["league"].astype("string").fillna("").str.strip().str.upper().replace(LEAGUE_MAP)

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
                .replace(LEAGUE_MAP)
            )
            if "game_date" not in working_df.columns:
                working_df["game_date"] = ""
            from core.team_mapper import normalize_team_name as _global_normalize
            from app_core.team_name_matcher import TeamNameMatcher
            if "home_team" in working_df.columns:
                working_df["home_team"] = clean_team_name(working_df["home_team"].apply(lambda x: TeamNameMatcher.normalize(_global_normalize(str(x) if pd.notna(x) else ""))).astype("string").fillna("").str.strip())
            if "away_team" in working_df.columns:
                working_df["away_team"] = clean_team_name(working_df["away_team"].apply(lambda x: TeamNameMatcher.normalize(_global_normalize(str(x) if pd.notna(x) else ""))).astype("string").fillna("").str.strip())
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
                    # Ensure league and current row identity are injected for statistical fallback nudge
                    features['league'] = row.get('league', '')
                    features['home_team'] = row.get('home_team', '')
                    features['away_team'] = row.get('away_team', '')
                    features['game_date'] = row.get('game_date', '')
                    market_type = str(row.get('market_type', '')).lower().strip()
                    fallback_probs.append(float(self._calculate_statistical_prob(features, market_type)))
                return fallback_probs

            # Fill missing implied probabilities BEFORE matrix validation so we don't crash
            if 'implied_home_prob' not in working_df.columns:
                working_df['implied_home_prob'] = pd.NA
            if 'kalshi_prob' not in working_df.columns:
                working_df['kalshi_prob'] = pd.NA

            # Log Python type of target columns before numeric coercion
            logger.info(f"Type of working_df['implied_home_prob']: {type(working_df.get('implied_home_prob'))}")
            logger.info(f"Type of working_df['kalshi_prob']: {type(working_df.get('kalshi_prob'))}")
            logger.info(f"Type of working_df['market_probability']: {type(working_df.get('market_probability'))}")

            if isinstance(working_df.get('implied_home_prob'), pd.DataFrame):
                logger.error("working_df['implied_home_prob'] is a DataFrame, indicating duplicate columns survived!")
            if isinstance(working_df.get('kalshi_prob'), pd.DataFrame):
                logger.error("working_df['kalshi_prob'] is a DataFrame, indicating duplicate columns survived!")
            if isinstance(working_df.get('market_probability'), pd.DataFrame):
                logger.error("working_df['market_probability'] is a DataFrame, indicating duplicate columns survived!")

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
                def _get_scalar(row_obj, col_name):
                    v = row_obj.get(col_name)
                    if isinstance(v, pd.DataFrame):
                        logger.error(f"[predict_batch row loop] '{col_name}' is a DataFrame!")
                        v = v.iloc[:, 0]
                    if isinstance(v, pd.Series):
                        logger.error(f"[predict_batch row loop] '{col_name}' is a Series!")
                        # Safely collapse to first valid scalar
                        v = v.dropna()
                        v = v.iloc[0] if not v.empty else pd.NA
                    return v

                # 1. Kalshi Prob
                k_prob = None
                for col in ['kalshi_probability', 'kalshi_prob']:
                    val = _get_scalar(row, col)

                    if pd.notna(val) and val != "":
                        try:
                            numeric_k = float(val)
                            if numeric_k > 0.0:
                                k_prob = numeric_k
                                break
                        except Exception:
                            continue

                # Kalshi Proxy Fallback: use market_probability or implied_home_prob if kalshi_prob is missing
                if not k_prob:
                    for col in ['market_probability', 'implied_home_prob']:
                        val = _get_scalar(row, col)
                        if pd.notna(val) and val != "":
                            try:
                                numeric_val = float(val)
                                if 0.0 < numeric_val < 1.0 and abs(numeric_val - 0.5) > 1e-6:
                                    k_prob = numeric_val
                                    break
                            except Exception:
                                continue

                # Note: kalshi_prob inversion happens properly downstream based on orientation.
                working_df.at[idx, 'kalshi_prob'] = k_prob if k_prob else 0.5

                # 2. Implied Home Prob
                i_prob = None
                for col in ['implied_home_prob', 'market_probability', 'home_price', 'odds_home', 'home_odds', 'odds_american']:
                    val = _get_scalar(row, col)

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

                market_type = str(row.get('market_type', '')).lower()
                is_away_side = market_type in ('spread_away', 'h2h_away', 'moneyline_away', 'away')
                is_under = market_type == 'total_under'

                # We need the model to evaluate the "Home/Over" perspective consistently.
                # If we are building a row for the away/under outcome, invert the probability
                # since the XGBoost model expects orientation to be Home/Over.
                if i_prob and i_prob != 0.5:
                     if is_away_side or is_under:
                          i_prob = 1.0 - i_prob

                working_df.at[idx, 'implied_home_prob'] = i_prob if i_prob else 0.5

            # Explicit cast to silence FutureWarnings before any remaining fillna
            working_df['kalshi_prob'] = pd.to_numeric(working_df['kalshi_prob'], errors='coerce')
            working_df['implied_home_prob'] = pd.to_numeric(working_df['implied_home_prob'], errors='coerce')

            # Log exact type to catch silent dataframe anomalies before coercion
            logger.info(f"Type of working_df['implied_home_prob']: {type(working_df.get('implied_home_prob'))}")
            logger.info(f"Type of working_df['kalshi_prob']: {type(working_df.get('kalshi_prob'))}")
            logger.info(f"Type of working_df['market_probability']: {type(working_df.get('market_probability'))}")

            # Select required columns while preserving missing columns as NaN.
            # This allows pre-inference validation to detect schedule/feature join failures.
            raw_inference_data = working_df.reindex(columns=VERTEX_FEATURE_COLUMNS).copy()
            raw_numeric = raw_inference_data.apply(pd.to_numeric, errors='coerce')

            # --- DIAGNOSTIC: True Coverage Logging Before Fills/Fallbacks ---
            # Identifies explicitly non-null features that came through the live pipeline
            if not raw_numeric.empty:
                true_coverage_stats = []
                for col in VERTEX_FEATURE_COLUMNS:
                    if col in raw_numeric.columns:
                        valid_count = raw_numeric[col].notna().sum()
                        pct = (valid_count / len(raw_numeric)) * 100
                        true_coverage_stats.append(f"{col}: {pct:.0f}%")
                    else:
                        true_coverage_stats.append(f"{col}: 0%")
                logger.info(f"TRUE Feature coverage before ANY fallback/fill (batch size {len(raw_numeric)}):\n  " + "\n  ".join(true_coverage_stats))

            # Merge Hardening: use sanitized team names for "Stale Feature Fallback"
            row_nan_ratio = raw_numeric.isna().sum(axis=1) / max(len(VERTEX_FEATURE_COLUMNS), 1)

            # Metrics tracking for the summary block
            metrics = {
                "total_rows": len(raw_numeric),
                "strict_join_rescued": 0,
                "fuzzy_join_rescued": 0,
                "split_lookup_rescued": 0,
            }

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

                        # Force column name parity for the merge
                        if "sport" in hist_df.columns and "league" not in hist_df.columns:
                            hist_df = hist_df.rename(columns={"sport": "league"})

                        # Apply standard league mapping to historical data
                        if "league" in hist_df.columns:
                            hist_df["league"] = (
                                hist_df["league"]
                                .astype("string")
                                .fillna("")
                                .str.strip()
                                .str.upper()
                                .replace(LEAGUE_MAP)
                            )

                        if "commence_time" in hist_df.columns:
                            hist_df["commence_time"] = pd.to_datetime(hist_df["commence_time"], errors="coerce", utc=True)

                            # Clean team names in hist_df
                            if "home_team" in hist_df.columns and "away_team" in hist_df.columns:
                                # Aggressive formatting: normalize, strip non-alphanumeric, lowercase, apply typo map
                                from core.team_mapper import normalize_team_name as _global_normalize
                                from app_core.team_name_matcher import TeamNameMatcher
                                hist_df["home_team"] = clean_team_name(hist_df["home_team"].apply(lambda x: TeamNameMatcher.normalize(_global_normalize(str(x) if pd.notna(x) else ""))).astype("string").fillna("").str.strip())
                                hist_df["away_team"] = clean_team_name(hist_df["away_team"].apply(lambda x: TeamNameMatcher.normalize(_global_normalize(str(x) if pd.notna(x) else ""))).astype("string").fillna("").str.strip())

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
                                hist_df["league_norm"] = hist_df["league"].astype("string").fillna("").str.strip().str.upper().replace(LEAGUE_MAP)

                                # Aggressive date format coercion
                                hist_df["game_date"] = _normalize_game_date_string(hist_df["commence_time"])
                                hist_df["game_date_dt"] = pd.to_datetime(hist_df["game_date"], errors="coerce").dt.tz_localize(None)

                                # Consistent match key: league|matchup_id|game_date
                                hist_df["canonical_match_key"] = (
                                    hist_df["league_norm"].astype("string")
                                    + "|"
                                    + hist_df["matchup_id"].astype("string")
                                    + "|"
                                    + hist_df["game_date"].astype("string")
                                )

                                # Process each predominantly empty row
                                for idx in df.index:
                                    row_league = df.at[idx, 'league']
                                    # If we don't know the league, we CANNOT join stats accurately
                                    if not row_league or pd.isna(row_league):
                                        continue

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

                                        # Create the aggressive match key dynamically consistently
                                        row_match_key = f"{row_league}|{row_matchup}|{row_game_date}" if row_league and row_matchup and row_game_date else ""

                                        # First priority: strict date AND matchup match
                                        match = pd.DataFrame()

                                        if row_match_key:
                                            match = hist_df[hist_df["canonical_match_key"].eq(row_match_key).fillna(False)]
                                            if not match.empty:
                                                metrics["strict_join_rescued"] += 1


                                        # Looser Fallback: rolling most recent matchup match (ignore date)
                                        if match.empty and row_matchup:
                                            match = hist_df[hist_df["matchup_id"].eq(row_matchup).fillna(False)]
                                            if row_league and "league_norm" in hist_df.columns:
                                                match = match[match["league_norm"].astype("string").str.upper().eq(row_league).fillna(False)]

                                            if not match.empty:
                                                # Treat recent lookup as a strict match
                                                metrics["strict_join_rescued"] += 1
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
                                                    metrics["fuzzy_join_rescued"] += 1

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

                                                # Create explicit mapping layer from historical to engine schema
                                                # Documenting synthetic / unmapped features:
                                                #   - injuries_home_count: Proxy mapped from injuries_impact
                                                #   - injuries_away_count: Unmapped in CSV (defaults to 0.0)
                                                #   - feature_home_oppg: Often unmapped in CSV (defaults to 0.0 if not found)
                                                #   - feature_home_streak: Proxy mapped from home_form_last5
                                                #   - feature_away_oppg: Often unmapped in CSV (defaults to 0.0 if not found)
                                                #   - feature_away_streak: Proxy mapped from away_form_last5
                                                #   - feature_home_rest_days: Proxy mapped from rest_advantage
                                                #   - feature_away_rest_days: Unmapped in CSV (defaults to 0.0)
                                                hist_mapping = {
                                                    "implied_home_prob": latest.get("implied_home_prob", latest.get("home_win_pct")),
                                                    "kalshi_prob": latest.get("kalshi_prob", latest.get("theover_probability")),
                                                    "sentiment_diff": latest.get("sentiment_diff", latest.get("sharp_vs_public", 0.0)),
                                                    "injuries_home_count": latest.get("injuries_home_count", latest.get("injuries_impact", 0.0)), # Proxy
                                                    "injuries_away_count": latest.get("injuries_away_count", 0.0), # Unmapped in CSV
                                                    "weather_flag": latest.get("weather_flag", latest.get("weather_factor", 0.0)),
                                                    "feature_home_win_pct": latest.get("feature_home_win_pct", latest.get("home_win_pct")),
                                                    "feature_home_ppg": latest.get("feature_home_ppg", latest.get("home_ppg")),
                                                    "feature_home_oppg": latest.get("feature_home_oppg", latest.get("home_oppg")), # Often unmapped in CSV
                                                    "feature_home_streak": latest.get("feature_home_streak", latest.get("home_form_last5")), # Proxy
                                                    "feature_away_win_pct": latest.get("feature_away_win_pct", latest.get("away_win_pct")),
                                                    "feature_away_ppg": latest.get("feature_away_ppg", latest.get("away_ppg")),
                                                    "feature_away_oppg": latest.get("feature_away_oppg", latest.get("away_oppg")), # Often unmapped
                                                    "feature_away_streak": latest.get("feature_away_streak", latest.get("away_form_last5")), # Proxy
                                                    "feature_diff_win_pct": latest.get("feature_diff_win_pct", latest.get("win_pct_diff")),
                                                    "feature_diff_ppg": latest.get("feature_diff_ppg", latest.get("ppg_diff")),
                                                    "feature_diff_oppg": latest.get("feature_diff_oppg", latest.get("oppg_diff")),
                                                    "feature_diff_last5": latest.get("feature_diff_last5", latest.get("form_diff")),
                                                    "feature_diff_streak": latest.get("feature_diff_streak", latest.get("streak_diff")),
                                                    "feature_home_rest_days": latest.get("feature_home_rest_days", latest.get("rest_advantage")), # Proxy
                                                    "feature_away_rest_days": latest.get("feature_away_rest_days", 0.0) # Unmapped in CSV
                                                }

                                                # Warning: feature_home_oppg, feature_away_oppg, injuries_away_count, feature_away_rest_days
                                                # may not exist in master_all_sports.csv and rely heavily on defaults when missing
                                                # They will be tracked as synthetic coverage.

                                                for col in VERTEX_FEATURE_COLUMNS:
                                                    val = hist_mapping.get(col)
                                                    if pd.isna(val) or val is None:
                                                        continue

                                                    val = float(val)

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
                                                metrics["split_lookup_rescued"] += 1

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

                        def _get_scalar_fallback(row_obj, col_name):
                            v = row_obj.get(col_name)
                            if isinstance(v, pd.DataFrame):
                                logger.error(f"[fallback row loop] '{col_name}' is a DataFrame!")
                                v = v.iloc[:, 0]
                            if isinstance(v, pd.Series):
                                logger.error(f"[fallback row loop] '{col_name}' is a Series!")
                                v = v.dropna()
                                v = v.iloc[0] if not v.empty else pd.NA
                            return v

                        # Try Bookmaker Odds FIRST (Arbitrage Fallback)
                        for col in ['implied_home_prob', 'market_probability', 'home_price', 'odds_home', 'home_odds', 'odds_american']:
                            val = _get_scalar_fallback(row, col)
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

            # --- DIAGNOSTIC: Model-Ready Coverage Logging AFTER Fills/Fallbacks but BEFORE zero-fill ---
            # Identifies coverage after applying historical lookups and synthetic defaults.
            if not raw_numeric.empty:
                model_ready_coverage_stats = []
                for col in VERTEX_FEATURE_COLUMNS:
                    if col in raw_numeric.columns:
                        valid_count = raw_numeric[col].notna().sum()
                        pct = (valid_count / len(raw_numeric)) * 100
                        model_ready_coverage_stats.append(f"{col}: {pct:.0f}%")
                    else:
                        model_ready_coverage_stats.append(f"{col}: 0%")
                logger.info(f"MODEL-READY Feature coverage (batch size {len(raw_numeric)}):\n  " + "\n  ".join(model_ready_coverage_stats))

                # Log matching statistics
                if len(used_stale_features) > 0:
                    stale_count = used_stale_features.sum()
                    logger.info(f"Fallback Summary: {stale_count} out of {len(raw_numeric)} rows used historical fallback.")

            self._last_metrics = metrics

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

            # Diagnose flat predictions
            total_rows_entering_model = len(inference_data)

            # Use only the strict 21 VERTEX_FEATURE_COLUMNS for uniqueness check
            model_matrix_cols = [c for c in VERTEX_FEATURE_COLUMNS if c in inference_data.columns]
            strict_model_input = inference_data[model_matrix_cols]
            unique_feature_vectors_count = strict_model_input.drop_duplicates().shape[0]
            exact_duplicate_vectors_count = total_rows_entering_model - unique_feature_vectors_count
            rows_using_live_stats = sum(~used_stale_features)
            rows_using_stale_stats = sum(used_stale_features)

            # Contextual inputs checks (must pull from working_df for non-model columns like market_probability)
            prob_cols = ['market_probability', 'implied_home_prob', 'kalshi_prob']

            prob_defaults = {}
            for col in prob_cols:
                count = 0
                if col in inference_data.columns:
                    count = (abs(pd.to_numeric(inference_data[col], errors='coerce').fillna(0.5) - 0.5) <= 1e-6).sum()
                elif col in working_df.columns:
                    count = (abs(pd.to_numeric(working_df[col], errors='coerce').fillna(0.5) - 0.5) <= 1e-6).sum()
                prob_defaults[col] = count

            logger.info(f"--- MODEL INFERENCE DIAGNOSTICS ---")
            logger.info(f"Total rows entering model: {total_rows_entering_model}")
            logger.info(f"Unique feature-vector count: {unique_feature_vectors_count}")
            logger.info(f"Exact duplicate vectors count: {exact_duplicate_vectors_count}")
            logger.info(f"Rows using true live stats: {rows_using_live_stats}")
            logger.info(f"Rows using stale/historical/default-filled features: {rows_using_stale_stats}")
            for col, count in prob_defaults.items():
                logger.info(f"Rows where {col} == 0.5: {count}")
            logger.info(f"-----------------------------------")

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

            # ML Variance Diagnostic 1: Raw model output uniqueness
            raw_valid_probs = [p for p in raw_probs if p is not None]
            self._last_metrics["raw_unique_count"] = len(set(raw_valid_probs))

            # --- START DIAGNOSTIC LOGGING FOR RAW DUPLICATES ---
            # Trigger protective override checks to determine if we should log leaf paths
            raw_unique_ratio = len(set(raw_valid_probs)) / len(raw_probs) if len(raw_probs) > 0 else 0
            is_unhealthy_flatness = (len(raw_probs) > 0 and raw_unique_ratio < 0.60)

            try:
                from collections import Counter
                raw_prob_counts = Counter(raw_valid_probs)
                duplicates = [(p, c) for p, c in raw_prob_counts.items() if c > 1]

                # Update unhealthy flag based on absolute thresholds too (same as downstream override logic)
                total_count = len(inference_data)
                if not is_unhealthy_flatness and total_count > 0:
                    max_repeats = max(raw_prob_counts.values()) if raw_prob_counts else 0
                    if total_count <= 30 and max_repeats >= 2: is_unhealthy_flatness = True
                    elif total_count <= 60 and max_repeats >= 3: is_unhealthy_flatness = True
                    elif total_count > 60 and (max_repeats >= 4 or (max_repeats/total_count) >= 0.10): is_unhealthy_flatness = True

                if duplicates and is_unhealthy_flatness:
                    logger.info(f"ML RAW COMPRESSION AUDIT: Found {len(duplicates)} repeated raw prob values. Top 3: {sorted(duplicates, key=lambda x: x[1], reverse=True)[:3]}")

                    # Log a sample for the most frequent duplicated probability
                    top_dup_prob, top_dup_count = sorted(duplicates, key=lambda x: x[1], reverse=True)[0]
                    logger.info(f"ML RAW COMPRESSION AUDIT: Sampling rows for raw probability = {top_dup_prob} (Count: {top_dup_count})")

                    sampled_count = 0

                    # Store hashes to determine if inputs were identical
                    sampled_feature_hashes = set()

                    # Gather rows sharing this probability
                    dup_indices = []
                    for idx_batch, p in enumerate(raw_probs):
                        if p is not None and abs(p - top_dup_prob) < 1e-7:
                            dup_indices.append(idx_batch)

                    # Get leaves ONLY for the duplicated rows
                    leaves = None
                    if isinstance(self.model, xgb.Booster) and len(dup_indices) > 0:
                        try:
                            # subset dmatrix
                            subset_data = inference_data.iloc[dup_indices]
                            subset_dmatrix = xgb.DMatrix(subset_data)
                            leaves = self.model.predict(subset_dmatrix, pred_leaf=True)
                            logger.info(f"ML RAW COMPRESSION AUDIT: Successfully extracted leaf indices for duplicated rows. Shape: {leaves.shape}")
                        except Exception as e:
                            logger.warning(f"ML RAW COMPRESSION AUDIT: Could not extract leaves for subset: {e}")

                    for local_i, idx_batch in enumerate(dup_indices):
                        if sampled_count >= 3:
                            break

                        original_idx = inference_data.index[idx_batch]
                        row = working_df.loc[original_idx]

                        # Only hash the strict model columns
                        feat_dict_strict = inference_data.iloc[idx_batch][model_matrix_cols].to_dict()
                        feat_dict_full = inference_data.iloc[idx_batch].to_dict()

                        # Create a list of the exact values in the order of VERTEX_FEATURE_COLUMNS
                        ordered_values = [round(float(feat_dict_strict.get(c, 0.0)), 5) for c in VERTEX_FEATURE_COLUMNS]

                        # Create a stable hash of the STRICT feature vector
                        feat_hash_str = json.dumps(ordered_values)
                        feat_hash = hashlib.md5(feat_hash_str.encode()).hexdigest()

                        is_duplicate = feat_hash in sampled_feature_hashes
                        sampled_feature_hashes.add(feat_hash)

                        logger.info(f"  --- Duplicate Row {sampled_count+1} ---")
                        logger.info(f"  League: {row.get('league')}, Matchup: {row.get('matchup_id')}, Market: {row.get('market_type')}")
                        logger.info(f"  Teams: {row.get('home_team')} vs {row.get('away_team')}")

                        m_prob = round(float(row.get('market_probability', 0.5)), 4) if row.get('market_probability') is not None else 0.5
                        i_prob = round(float(feat_dict_strict.get('implied_home_prob', 0.5)), 4)
                        k_prob = round(float(feat_dict_strict.get('kalshi_prob', 0.5)), 4)

                        logger.info(f"  Implied Home Prob: {i_prob}, Kalshi Prob: {k_prob}, Market Prob: {m_prob}")
                        logger.info(f"  Diff Win Pct: {round(float(feat_dict_strict.get('feature_diff_win_pct', 0.0)), 4)}, Diff PPG: {round(float(feat_dict_strict.get('feature_diff_ppg', 0.0)), 4)}, Diff Streak: {round(float(feat_dict_strict.get('feature_diff_streak', 0.0)), 4)}")
                        logger.info(f"  Stale Features Used: {used_stale_features.iloc[idx_batch]}")
                        logger.info(f"  Feature Vector Signature (Hash): {feat_hash}")
                        if is_duplicate:
                            logger.info(f"  Input identical to previously sampled row: YES (Exact duplicate inputs)")
                        else:
                            # Check if they are near-identical (e.g. contextual features are default filled)
                            near_identical = False
                            if abs(feat_dict_strict.get('implied_home_prob', 0) - 0.5) <= 1e-6 and abs(feat_dict_strict.get('kalshi_prob', 0) - 0.5) <= 1e-6:
                                 near_identical = True
                            if near_identical:
                                 logger.info(f"  Input identical to previously sampled row: NO (Different inputs mapping to same output, but Contextual Features are DEFAULT/NEUTRAL)")
                            else:
                                 logger.info(f"  Input identical to previously sampled row: NO (Genuinely different inputs mapping to same coarse XGBoost bucket/leaf path)")

                        if leaves is not None and local_i < len(leaves):
                            # Log a hash and the first few tree leaves to show if they differ
                            row_leaves = leaves[local_i]
                            leaf_hash = hashlib.md5(row_leaves.tobytes()).hexdigest()
                            logger.info(f"  XGBoost Leaf Path Hash: {leaf_hash} (First 5 leaves: {row_leaves[:5].tolist()})")

                        sampled_count += 1

                    # Overall conclusion for this group
                    if sampled_count > 1:
                        if len(sampled_feature_hashes) == 1:
                            logger.info(f"ML RAW COMPRESSION AUDIT CONCLUSION: Flatness caused by EXACTLY IDENTICAL INPUTS (feature collapse).")
                        elif len(sampled_feature_hashes) < sampled_count:
                            logger.info(f"ML RAW COMPRESSION AUDIT CONCLUSION: Flatness caused by MIXED identical and near-identical default-filled inputs.")
                        else:
                            logger.info(f"ML RAW COMPRESSION AUDIT CONCLUSION: Flatness caused by MODEL COARSENESS (Different inputs mapping to the same leaf path/bucket).")
            except Exception as e:
                logger.error(f"Error during ML RAW COMPRESSION AUDIT logging: {e}")
            # --- END DIAGNOSTIC LOGGING ---

            # Jules: MANDATORY Blacklist and Statistical Healing recovery
            BLACKLIST = [0.623034656047821, 0.10671072453260422, 0.48637846, 0.31053704, 0.5622388124465942, 0.562238]
            healed_count = 0
            for idx_batch, p in enumerate(raw_probs):
                # If model returns a known bias value, discard and use performance stats instead
                if p is None or any(abs(p - b) < 1e-5 for b in BLACKLIST):
                    # Pull prepared performance features from the matrix
                    row_features = inference_data.iloc[idx_batch].to_dict()

                    # Ensure league and identity is injected for statistical fallback by getting it from the original working_df
                    original_idx = inference_data.index[idx_batch]
                    row_features['league'] = working_df.loc[original_idx, 'league'] if 'league' in working_df.columns else ''
                    row_features['home_team'] = working_df.loc[original_idx, 'home_team'] if 'home_team' in working_df.columns else ''
                    row_features['away_team'] = working_df.loc[original_idx, 'away_team'] if 'away_team' in working_df.columns else ''
                    row_features['game_date'] = working_df.loc[original_idx, 'game_date'] if 'game_date' in working_df.columns else ''

                    market_type = str(row_features.get('market_type', '')).lower().strip()
                    final_probs.append(self._calculate_statistical_prob(row_features, market_type))
                    healed_count += 1
                else:
                    final_probs.append(p)

            self._last_metrics["healed_count"] = healed_count

            # ML Variance Diagnostic 2: Post-healing output uniqueness
            valid_probs = [p for p in final_probs if p is not None]
            unique_count = len(set(valid_probs))
            total_count = len(df)
            self._last_metrics["post_healing_unique_count"] = unique_count

            # Trigger protective override based on RAW unique count (before blacklist healing)
            # using refined tiered logic as requested
            raw_unique_count = self._last_metrics.get("raw_unique_count", 0)
            is_flat = False

            # Rule 1: Ratio-based rule for ALL slate sizes (must be >= 60% unique)
            raw_unique_ratio = raw_unique_count / total_count if total_count > 0 else 0
            if total_count > 0 and raw_unique_ratio < 0.60:
                logger.warning(f"ML RAW COMPRESSION AUDIT: Raw unique ratio ({raw_unique_ratio:.1%}) is below 60%. Triggering intervention.")
                is_flat = True

            # Rule 2: Absolute repeat thresholds based on slate size
            if not is_flat and total_count > 0:
                try:
                    from collections import Counter
                    raw_counts = Counter(raw_valid_probs)
                    max_repeats = max(raw_counts.values()) if raw_counts else 0

                    if total_count <= 30:
                        if max_repeats >= 2:
                            logger.warning(f"ML RAW COMPRESSION AUDIT: Slate <= 30 rows and a raw probability repeated {max_repeats} times (threshold: 2+). Triggering intervention.")
                            is_flat = True
                    elif total_count <= 60:
                        if max_repeats >= 3:
                            logger.warning(f"ML RAW COMPRESSION AUDIT: Slate 31-60 rows and a raw probability repeated {max_repeats} times (threshold: 3+). Triggering intervention.")
                            is_flat = True
                    else:
                        max_ratio = max_repeats / total_count if total_count > 0 else 0
                        if max_repeats >= 4:
                            logger.warning(f"ML RAW COMPRESSION AUDIT: Slate > 60 rows and a raw probability repeated {max_repeats} times (threshold: 4+). Triggering intervention.")
                            is_flat = True
                        elif max_ratio >= 0.10:
                            logger.warning(f"ML RAW COMPRESSION AUDIT: Slate > 60 rows and a raw probability covers {max_ratio:.1%} of rows (threshold: 10%+). Triggering intervention.")
                            is_flat = True
                except Exception as e:
                    logger.error(f"Error during ML RAW COMPRESSION AUDIT tier check: {e}")

            # Log output variance for monitoring
            if total_count > 0:
                unique_ratio = unique_count / total_count
                logger.info(f"Output variance check (post-healing): {unique_count} unique probabilities out of {total_count} rows ({unique_ratio:.1%}). Raw unique: {raw_unique_count}")

                if total_count >= 20 and unique_ratio < 0.45:
                    logger.warning(f"Low post-healing output variance detected: {unique_count} unique probabilities across {total_count} games.")

            if is_flat:
                logger.warning(f"ML UNIQUENESS AUDIT: XGBoost returned critically flat raw probabilities ({raw_unique_count}/{total_count}). Overriding with Hybrid Fallback Score.")
                logger.info(f"PIPELINE TRACE: Flat probabilities detected. Overriding all {total_count} predictions with Hybrid Fallback.")
                final_probs = []

                chosen_market_probs = []
                scores_before_eps = []
                scores_after_eps = []

                for idx_batch, idx in enumerate(working_df.index):
                    row = working_df.loc[idx]
                    league_str = str(row.get('league', '')).upper()
                    market_type = str(row.get('market_type', '')).lower().strip()
                    
                    is_home_side = market_type in ('spread_home', 'h2h_home')
                    is_away_side = market_type in ('spread_away', 'h2h_away')
                    is_over = market_type == 'total_over'
                    is_under = market_type == 'total_under'

                    # Component 1: Market Probability (45%)
                    chosen_price = None
                    opp_price = None

                    if is_home_side:
                        chosen_price = row.get('novig_home_price') or row.get('novig_h2h_home_price')
                        opp_price = row.get('novig_away_price') or row.get('novig_h2h_away_price')
                    elif is_away_side:
                        chosen_price = row.get('novig_away_price') or row.get('novig_h2h_away_price')
                        opp_price = row.get('novig_home_price') or row.get('novig_h2h_home_price')
                    elif is_over:
                        chosen_price = row.get('novig_over_price')
                        opp_price = row.get('novig_under_price')
                    elif is_under:
                        chosen_price = row.get('novig_under_price')
                        opp_price = row.get('novig_over_price')

                    def get_american(val):
                        try:
                            if pd.notna(val) and str(val).strip() != "":
                                return float(val)
                        except Exception:
                            pass
                        return None

                    def am_to_prob(odds):
                        if odds is None or odds == 0: return None
                        if odds > 0: return 100.0 / (odds + 100.0)
                        return abs(odds) / (abs(odds) + 100.0)

                    c_price = get_american(chosen_price)
                    o_price = get_american(opp_price)

                    market_prob = None
                    if c_price is not None and o_price is not None:
                        cp = am_to_prob(c_price)
                        op = am_to_prob(o_price)
                        if cp and op:
                            market_prob = cp / (cp + op)
                    elif c_price is not None:
                        market_prob = am_to_prob(c_price)
                    else:
                        am = get_american(row.get('odds_american'))
                        if am is not None:
                            market_prob = am_to_prob(am)

                    # Component 2: Kalshi Probability (20%)
                    k_prob = None
                    try:
                        kalshi = row.get('kalshi_prob')
                        if pd.notna(kalshi):
                            val = float(kalshi)
                            if val > 0 and abs(val - 0.5) > 0.01:
                                if is_away_side or is_under:
                                    k_prob = 1.0 - val
                                else:
                                    k_prob = val
                    except Exception:
                        pass

                    # Component 3: Magnitude Signal (10%)
                    mag_comp = None
                    try:
                        if is_home_side or is_away_side:
                            line = row.get('spread_line') if pd.notna(row.get('spread_line')) else row.get('spread')
                            if pd.notna(line):
                                norm = min(abs(float(line)) / 15.0, 1.0)
                                mag_comp = 0.50 + 0.10 * norm
                        elif is_over or is_under:
                            # Prefer total movement if available, else use league median
                            total_movement = row.get('total_movement')
                            if pd.notna(total_movement):
                                norm = min(abs(float(total_movement)) / 10.0, 1.0)
                                mag_comp = 0.50 + 0.10 * norm
                            else:
                                line = row.get('total_line')
                                if pd.notna(line):
                                    league_median = {'NBA': 228.0, 'NCAAB': 142.0, 'NFL': 45.0, 'NHL': 6.0}.get(league_str, 150.0)
                                    band = {'NBA': 20.0, 'NCAAB': 20.0, 'NFL': 10.0, 'NHL': 1.5}.get(league_str, 20.0)
                                    norm = min(abs(float(line) - league_median) / band, 1.0)
                                    mag_comp = 0.50 + 0.10 * norm
                    except Exception:
                        pass

                    # Component 4: Feature Differential (10%)
                    feat_comp = None
                    try:
                        if is_home_side or is_away_side:
                            diff_win_pct = inference_data.iloc[idx_batch].get('feature_diff_win_pct', 0.0)
                            diff_ppg = inference_data.iloc[idx_batch].get('feature_diff_ppg', 0.0)
                            diff_streak = inference_data.iloc[idx_batch].get('feature_diff_streak', 0.0)

                            blend = 0.5 * float(diff_win_pct) + 0.3 * (float(diff_ppg) / 20.0) + 0.2 * (float(diff_streak) / 10.0)
                            blend = max(-1.0, min(1.0, blend))
                            if is_away_side:
                                blend = -blend
                            feat_comp = 0.50 + 0.12 * blend
                        elif is_over or is_under:
                            h_ppg = inference_data.iloc[idx_batch].get('feature_home_ppg', 0.0)
                            a_ppg = inference_data.iloc[idx_batch].get('feature_away_ppg', 0.0)
                            proxy = (float(h_ppg) + float(a_ppg)) / 2.0
                            median_ppg = {'NBA': 114.0, 'NCAAB': 72.0, 'NFL': 22.0, 'NHL': 3.0}.get(league_str, 100.0)
                            norm = (proxy - median_ppg) / (median_ppg * 0.2)
                            norm = max(-1.0, min(1.0, norm))
                            if is_under:
                                norm = -norm
                            feat_comp = 0.50 + 0.12 * norm
                    except Exception:
                        pass

                    # Component 5: Rest Differential (5%)
                    rest_comp = None
                    try:
                        if is_home_side or is_away_side:
                            h_rest = inference_data.iloc[idx_batch].get('feature_home_rest_days', 3.0)
                            a_rest = inference_data.iloc[idx_batch].get('feature_away_rest_days', 3.0)
                            rest_diff = float(h_rest) - float(a_rest)
                            rest_diff = max(-4.0, min(4.0, rest_diff))
                            if is_away_side:
                                rest_diff = -rest_diff
                            rest_comp = 0.50 + (rest_diff / 4.0) * 0.05
                        elif is_over or is_under:
                            rest_comp = 0.50
                    except Exception:
                        pass

                    # Component 6: Sentiment / Movement (5%)
                    sent_comp = None
                    try:
                        sent_val = None
                        move_val = None

                        if is_home_side or is_away_side:
                            sent_diff = inference_data.iloc[idx_batch].get('sentiment_diff', 0.0)
                            sent_diff = float(sent_diff)
                            if is_away_side:
                                sent_diff = -sent_diff
                            sent_val = 0.50 + sent_diff * 0.05

                            line_move = row.get('line_movement')
                            if pd.notna(line_move):
                                move = float(line_move)
                                # negative movement means line moved in favor of home
                                if is_away_side:
                                    move = -move
                                # cap movement impact
                                move_val = 0.50 - (move / 10.0) * 0.05
                                move_val = max(0.40, min(0.60, move_val))

                        elif is_over or is_under:
                            total_move = row.get('total_movement')
                            if pd.notna(total_move):
                                move = float(total_move)
                                # positive movement means total went up (favoring over)
                                if is_under:
                                    move = -move
                                move_val = 0.50 + (move / 5.0) * 0.05
                                move_val = max(0.40, min(0.60, move_val))

                        if sent_val is not None and move_val is not None:
                            sent_comp = (sent_val + move_val) / 2.0
                        elif sent_val is not None:
                            sent_comp = sent_val
                        elif move_val is not None:
                            sent_comp = move_val
                        else:
                            sent_comp = 0.50
                    except Exception:
                        pass

                    # Component 7: Residual Fallback (5%)
                    residual_prob = None
                    try:
                        row_features = inference_data.iloc[idx_batch].to_dict()
                        row_features['league'] = league_str
                        row_features['home_team'] = row.get('home_team', '')
                        row_features['away_team'] = row.get('away_team', '')
                        row_features['game_date'] = row.get('game_date', '')
                        
                        stat_prob = self._calculate_statistical_prob(row_features, market_type)
                        if is_away_side or is_under:
                            residual_prob = 1.0 - stat_prob
                        else:
                            residual_prob = stat_prob
                    except Exception:
                        pass

                    # Blend components with renormalizing weights
                    components = {}
                    weights = {}

                    if market_prob is not None:
                        components['market'] = market_prob
                        weights['market'] = 0.45
                    if k_prob is not None:
                        components['kalshi'] = k_prob
                        weights['kalshi'] = 0.20
                    if mag_comp is not None:
                        components['magnitude'] = mag_comp
                        weights['magnitude'] = 0.10
                    if feat_comp is not None:
                        components['feature'] = feat_comp
                        weights['feature'] = 0.10
                    if rest_comp is not None:
                        components['rest'] = rest_comp
                        weights['rest'] = 0.05
                    if sent_comp is not None:
                        components['sentiment'] = sent_comp
                        weights['sentiment'] = 0.05
                    if residual_prob is not None:
                        components['residual'] = residual_prob
                        weights['residual'] = 0.05

                    total_weight = sum(weights.values())
                    if total_weight > 0:
                        hybrid_score = sum(components[k] * weights[k] for k in components) / total_weight
                    else:
                        hybrid_score = 0.5

                    score_before_eps = hybrid_score

                    # Deterministic Tie-break Epsilon
                    matchup_id = row.get('matchup_id', '')
                    game_date = row.get('game_date', '')
                    seed_str = f"{matchup_id}|{game_date}|{market_type}"
                    md5_hash = hashlib.md5(seed_str.encode()).hexdigest()
                    # map hash to a tiny positive float between 0 and 9e-7
                    epsilon = (int(md5_hash[:8], 16) / 0xFFFFFFFF) * 9e-7

                    hybrid_score += epsilon
                    hybrid_score = max(0.01, min(0.99, hybrid_score))

                    chosen_market_probs.append(market_prob if market_prob is not None else 0.5)
                    scores_before_eps.append(score_before_eps)
                    scores_after_eps.append(hybrid_score)
                    final_probs.append(hybrid_score)

                # Variance Diagnostics Logging
                logger.info(f"ML UNIQUENESS AUDIT: Unique prob count before fallback (raw_unique_count): {self._last_metrics.get('raw_unique_count')}")
                logger.info(f"ML UNIQUENESS AUDIT: Unique prob count after fallback: {len(set(final_probs))}")

                from collections import Counter
                unique_before = len(set(scores_before_eps))
                unique_after = len(set(scores_after_eps))

                self._last_metrics["hybrid_override_triggered"] = True
                self._last_metrics["hybrid_unique_before_eps"] = unique_before

                market_prob_counts = Counter(chosen_market_probs)
                before_eps_counts = Counter(scores_before_eps)

                identical_market_rows = sum(count for count in market_prob_counts.values() if count > 1)
                identical_before_rows = sum(count for count in before_eps_counts.values() if count > 1)

                logger.info(f"Hybrid Fallback Variance Report:")
                logger.info(f"  - Unique chosen market probabilities: {len(set(chosen_market_probs))}/{total_count} (Rows sharing exact values: {identical_market_rows})")
                logger.info(f"  - Unique raw scores BEFORE epsilon: {unique_before}/{total_count} (Rows sharing exact values: {identical_before_rows})")
                logger.info(f"  - Unique final scores AFTER epsilon: {unique_after}/{total_count}")

                if identical_before_rows > 0:
                    top_dupes = before_eps_counts.most_common(3)
                    logger.info(f"  - Sample duplicate groups (before epsilon): {top_dupes}")

                    # Detailed sampling of repeated groups
                    for dup_score, dup_count in top_dupes:
                        if dup_count > 1:
                            logger.info(f"    - Inspecting group collapsing to raw score {dup_score:.5f} ({dup_count} rows):")
                            for idx_batch, score in enumerate(scores_before_eps):
                                if abs(score - dup_score) < 1e-7:
                                    row = working_df.iloc[idx_batch]
                                    feat = inference_data.iloc[idx_batch]
                                    kalshi_val = feat.get('kalshi_prob', 0.5)
                                    mkt_prob = chosen_market_probs[idx_batch]
                                    stale_flag = used_stale_features.iloc[idx_batch]

                                    # Determine what was synthetic
                                    synth_flags = []
                                    if abs(feat.get('feature_home_oppg', 0.0)) < 1e-5: synth_flags.append('home_oppg')
                                    if abs(feat.get('feature_away_rest_days', 0.0)) < 1e-5: synth_flags.append('away_rest')

                                    stats_qual = row.get('stats_quality', 'UNKNOWN')
                                    sanitized = row.get('sanitized_value', False)
                                    logger.info(f"ML UNIQUENESS AUDIT:      [{row.get('league')}] {row.get('matchup_id')} | Pick: {row.get('market_type')} | Mkt: {mkt_prob:.3f} | Kalshi: {kalshi_val:.3f} | Stale: {stale_flag} | Synth: {synth_flags} | StatsQual: {stats_qual} | Sanitized: {sanitized}")
            else:
                self._last_metrics["hybrid_override_triggered"] = False
                logger.info(f"ML UNIQUENESS AUDIT: Variance looks good ({unique_count}/{total_count}). Hybrid fallback NOT triggered.")

            final_unique = len(set(final_probs))
            self._last_metrics["final_unique_count"] = final_unique
            logger.info(f"PIPELINE TRACE: Final unique probabilities exiting predict_batch: {final_unique} out of {len(final_probs)}")

            # --- HEALTH SCORE / SUMMARY BLOCK ---
            logger.info("=" * 60)
            logger.info("PIPELINE HEALTH SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Total Games/Rows Evaluated: {total_count}")
            logger.info(f"1. Raw Model Unique Outputs: {self._last_metrics.get('raw_unique_count', 0)}")
            logger.info(f"2. Post-Healing Unique Outputs: {self._last_metrics.get('post_healing_unique_count', 0)} (Rows healed: {self._last_metrics.get('healed_count', 0)})")

            if self._last_metrics.get("hybrid_override_triggered", False):
                logger.info(f"3. Post-Hybrid Unique Outputs (Pre-Epsilon): {self._last_metrics.get('hybrid_unique_before_eps', 0)}")
            else:
                logger.info(f"3. Post-Hybrid Unique Outputs (Pre-Epsilon): N/A (Override Not Triggered)")

            logger.info(f"4. Final Unique Outputs: {final_unique}")

            if not raw_numeric.empty:
                true_cov_mean = (raw_numeric.notna().sum(axis=1) / max(len(VERTEX_FEATURE_COLUMNS), 1)).mean() * 100
                synth_cov_mean = 100 - true_cov_mean
                logger.info(f"True Feature Coverage (Avg/Row): {true_cov_mean:.1f}%")
                logger.info(f"Synthetic/Default-Filled Coverage (Avg/Row): {synth_cov_mean:.1f}%")
            else:
                logger.info("Feature Coverage: N/A (Empty Matrix)")

            strict_pct = (self._last_metrics["strict_join_rescued"] / max(total_count, 1)) * 100
            fuzzy_pct = (self._last_metrics["fuzzy_join_rescued"] / max(total_count, 1)) * 100
            split_pct = (self._last_metrics["split_lookup_rescued"] / max(total_count, 1)) * 100

            # Calculate rows that used true live stats (neither fallback nor historical reconstruction)
            total_historical_rescued = self._last_metrics['strict_join_rescued'] + self._last_metrics['fuzzy_join_rescued'] + self._last_metrics['split_lookup_rescued']
            live_stats_count = total_count - total_historical_rescued
            live_stats_pct = (live_stats_count / max(total_count, 1)) * 100

            logger.info(f"Data Source Breakdown:")
            logger.info(f" - Rows using TRUE Live Stats (APIs): {live_stats_pct:.1f}% ({live_stats_count})")
            logger.info(f" - Rows using Strict Historical Reconstruction: {strict_pct:.1f}% ({self._last_metrics['strict_join_rescued']})")
            logger.info(f" - Rows using Fuzzy Historical Reconstruction: {fuzzy_pct:.1f}% ({self._last_metrics['fuzzy_join_rescued']})")
            logger.info(f" - Rows using Split Historical Lookup: {split_pct:.1f}% ({self._last_metrics['split_lookup_rescued']})")

            hybrid_pct = 100.0 if is_flat else 0.0
            logger.info(f"Rows using Hybrid Override (Fallback): {hybrid_pct:.1f}%")

            sanitized_pct = (self._last_metrics.get('healed_count', 0) / max(total_count, 1)) * 100
            logger.info(f"Rows patched by Sanitization: {sanitized_pct:.1f}%")
            logger.info("=" * 60)

            # Summarizing for the user explicitly:
            logger.info(f"FALLBACK AUDIT: Live Stats = {live_stats_count}, Historical/Stale = {total_historical_rescued}, Hybrid = {total_count if is_flat else 0}")
            logger.info("=" * 60)

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
