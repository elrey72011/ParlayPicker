"""
app_core.vertex_ai_endpoint

Thin wrapper around a deployed Google Vertex AI Endpoint used for
numeric win-probability predictions in ParlayPicker.

Exports:
    - VERTEX_FEATURE_COLUMNS: list[str]
    - is_vertex_prediction_configured() -> bool
    - predict_win_probabilities(df, feature_columns=None) -> list[float]
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

try:
    import streamlit as st  # type: ignore
except Exception:  # Streamlit not always available (e.g., offline tests)
    st = None  # type: ignore

try:
    from google.cloud import aiplatform
    _GCP_AVAILABLE = True
except Exception:
    aiplatform = None  # type: ignore
    _GCP_AVAILABLE = False

from google.oauth2 import service_account  # type: ignore

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# USER REQUEST 1: Robust File Pathing (Global Scope)
# -------------------------------------------------------------------
from pathlib import Path
# Resolve the root directory regardless of where the script is called
ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = str(ROOT_DIR / "models" / "model.json")

# Global flag to track local model availability
USE_LOCAL_MODEL = True

def _get_gcp_credentials():
    """
    Prefer Streamlit secrets: [gcp_service_account] (dict)
    Fallback: GOOGLE_APPLICATION_CREDENTIALS (path)
    """
    # 1) Streamlit secrets dict
    if st and getattr(st, "secrets", None):
        sa = st.secrets.get("gcp_service_account")
        if sa:
            return service_account.Credentials.from_service_account_info(dict(sa))

    # 2) Env var path
    key_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if key_path:
        return service_account.Credentials.from_service_account_file(key_path)

    return None

# -------------------------------------------------------------------
# MODEL INPUT SCHEMA
# -------------------------------------------------------------------

#: Columns your Vertex tabular model expects.
#: Make sure this matches the model's training schema.
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
# DEFAULT CONFIG (you can override in secrets or env)
# -------------------------------------------------------------------

DEFAULT_PROJECT_ID = "elite-hangar-479017-m8"
DEFAULT_LOCATION = "us-central1"
# USER REQUEST 3: Match Endpoint IDs
DEFAULT_ENDPOINT_ID = "3242045274427752448"  # numeric endpoint id from your logs

# Cached endpoint client & name so we don't recreate it constantly
_vertex_endpoint_client = None
_vertex_endpoint_name: Optional[str] = None


# -------------------------------------------------------------------
# INTERNAL CONFIG HELPERS
# -------------------------------------------------------------------

def _get_vertex_config() -> tuple[str, str, str]:
    """
    Resolve (project_id, location, endpoint_id) from:
      1. Streamlit session_state
      2. Streamlit secrets
      3. Environment variables
      4. Internal defaults
    """
    # Project
    project_id = (
        (st and st.session_state.get("gcp_project_id"))
        or (st and getattr(st, "secrets", {}).get("gcp_project_id", None))
        or os.environ.get("GCP_PROJECT_ID")
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or DEFAULT_PROJECT_ID
    )

    # Location
    location = (
        (st and st.session_state.get("gcp_region"))
        or (st and getattr(st, "secrets", {}).get("gcp_region", None))
        or os.environ.get("GCP_REGION")
        or DEFAULT_LOCATION
    )

    # Endpoint (either raw id or full resource name)
    endpoint_id = (
        (st and getattr(st, "secrets", {}).get("vertex_endpoint_id", None))
        or os.environ.get("VERTEX_ENDPOINT_ID")
    )

    if not endpoint_id:
        # Fallback to hardcoded default if secret is missing (User Requirement)
        logger.info(f"Vertex: No endpoint ID in secrets/env. Using fallback: {DEFAULT_ENDPOINT_ID}")
        endpoint_id = DEFAULT_ENDPOINT_ID

    # Normalize just in case
    project_id = str(project_id).strip()
    location = str(location).strip()
    endpoint_id = str(endpoint_id).strip()

    return project_id, location, endpoint_id


def _build_endpoint_name(project_id: str, location: str, endpoint_id: str) -> str:
    """
    Build full endpoint resource name.
    """
    if "projects/" in endpoint_id and "/endpoints/" in endpoint_id:
        return endpoint_id

    return f"projects/{project_id}/locations/{location}/endpoints/{endpoint_id}"


def get_vertex_endpoint():
    """
    Get a configured Vertex Endpoint client.
    """
    global _vertex_endpoint_client, _vertex_endpoint_name

    if not _GCP_AVAILABLE:
        raise RuntimeError(
            "google-cloud-aiplatform is not installed. "
            "Install with: pip install google-cloud-aiplatform"
        )

    project_id, location, endpoint_id = _get_vertex_config()
    endpoint_name = _build_endpoint_name(project_id, location, endpoint_id)

    # Return cached client if same endpoint
    if _vertex_endpoint_client is not None and _vertex_endpoint_name == endpoint_name:
        return _vertex_endpoint_client

    # Explicitly init AI Platform with proper project & region
    creds = _get_gcp_credentials()
    aiplatform.init(project=project_id, location=location, credentials=creds)

    logger.info(f"[VertexEndpoint] Using endpoint: {endpoint_name}")
    endpoint = aiplatform.Endpoint(endpoint_name)

    _vertex_endpoint_client = endpoint
    _vertex_endpoint_name = endpoint_name
    return endpoint

# -------------------------------------------------------------------
# PUBLIC HELPERS
# -------------------------------------------------------------------

def is_vertex_prediction_configured() -> bool:
    """
    Return True if Vertex appears to be configured well enough
    to attempt predictions.
    """
    if not _GCP_AVAILABLE:
        logger.warning("Vertex prediction not configured: google-cloud-aiplatform missing")
        return False

    try:
        # User Request 2: Use _get_vertex_config() for robust fallback
        project_id, location, endpoint_id = _get_vertex_config()
    except Exception as e:
        logger.warning(f"Vertex config resolution failed: {e}")
        return False

    if not project_id or project_id.lower() == "none":
        logger.warning("Vertex prediction not configured: project_id is missing/None")
        return False

    if not endpoint_id:
        logger.warning("Vertex prediction not configured: endpoint_id is missing")
        return False

    if not location:
        logger.warning("Vertex prediction not configured: location is missing")
        return False

    return True


def predict_win_probabilities(df, feature_cols=None, model_path=None):
    """
    Calls Vertex AI (or local XGBoost) to predict Home Win Probability.
    Catches exceptions and stores them in st.session_state['vertex_last_error'] for debugging.
    """
    global USE_LOCAL_MODEL

    try:
        import xgboost as xgb
        # Ensure duplicates are stripped here too just in case
        df = df.loc[:, ~df.columns.duplicated()].copy()
        
        if feature_cols is None:
            feature_cols = VERTEX_FEATURE_COLUMNS

        # Default model path if not provided
        if model_path is None:
            # User Request 1: Use Global MODEL_PATH
            model_path = MODEL_PATH

        # 2. VALIDATE: Ensure all features exist
        # This handles the User Request to ensure 0.0 initialization
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            logger.warning(f"Vertex Warning: Missing cols {missing}. Filling 0.")
            # Use concat to avoid DataFrame fragmentation from loop insertion
            zeros = pd.DataFrame(0.0, index=df.index, columns=missing)
            df = pd.concat([df, zeros], axis=1)

        # User Request 1: Fix Exception Flow
        # Try local model first
        if USE_LOCAL_MODEL:
            try:
                dmatrix = xgb.DMatrix(df[feature_cols])
                booster = xgb.Booster()
                booster.load_model(model_path)
                return booster.predict(dmatrix)
            except Exception as load_err:
                # Disable local model and proceed (do not raise)
                USE_LOCAL_MODEL = False
                logger.warning(f"Local XGBoost model load failed: {load_err}. Disabling USE_LOCAL_MODEL and proceeding to fallback.")

        # Fallback: Vertex AI
        if is_vertex_prediction_configured():
            endpoint = get_vertex_endpoint()
            # Vertex expects list of lists/dicts
            # Ensure we strictly use the defined feature columns
            instances = df[feature_cols].astype(float).values.tolist()

            # DEBUG PRINT as requested for troubleshooting
            if st:
                st.write(f"DEBUG: Feature Vector: {instances}")

            response = endpoint.predict(instances=instances)

            # USER REQUEST 2: Fix Parser & Debug
            print(f"DEBUG: Vertex Raw Prediction: {response.predictions}")

            predictions = response.predictions

            if not predictions:
                return None

            # Inside the fallback extraction
            results = []
            for pred in response.predictions:
                if isinstance(pred, dict):
                    # Try 'scores' index 1 (Home Win) or 'value'
                    val = pred.get("scores", [0.5, 0.5])[1] if "scores" in pred else pred.get("value", 0.5)
                    results.append(val)
                else:
                    results.append(pred)
            return results
        else:
            # If Vertex not configured, return None to signal fallback to defaults
            logger.warning("Vertex not configured and local model failed. Returning None for fallback.")
            return None

    except Exception as e:
        logger.error(f"Vertex Crash Prevented: {e}")
        # Expose error to UI if available
        if st is not None:
            try:
                st.session_state["vertex_last_error"] = str(e)
            except Exception:
                pass

        # Return 50% probabilities so the app FINISHES instead of restarting
        return [0.5] * len(df)
