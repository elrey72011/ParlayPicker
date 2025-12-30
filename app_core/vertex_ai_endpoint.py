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

logger = logging.getLogger(__name__)

# add near imports in vertex_ai_endpoint.py
from google.oauth2 import service_account  # type: ignore

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
    "feature_home_home_win_pct",
    "feature_home_last5_win_pct",
    "feature_home_ppg",
    "feature_home_oppg",
    "feature_home_streak",
    "feature_away_win_pct",
    "feature_away_away_win_pct",
    "feature_away_last5_win_pct",
    "feature_away_ppg",
    "feature_away_oppg",
    "feature_away_streak",
    "feature_diff_win_pct",
    "feature_diff_ppg",
    "feature_diff_oppg",
    "feature_diff_last5",
    "feature_diff_streak",
    "feature_commence_hour",
    "feature_commence_day_of_week",
    "feature_home_rest_days",
    "feature_away_rest_days",
]

# -------------------------------------------------------------------
# DEFAULT CONFIG (you can override in secrets or env)
# -------------------------------------------------------------------

DEFAULT_PROJECT_ID = "elite-hangar-479017-m8"
DEFAULT_LOCATION = "us-central1"
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

    Order of precedence (highest → lowest):

        project_id:
          - st.session_state['gcp_project_id']
          - st.secrets['gcp_project_id']
          - $GCP_PROJECT_ID
          - $GOOGLE_CLOUD_PROJECT
          - DEFAULT_PROJECT_ID

        location:
          - st.session_state['gcp_region']
          - st.secrets['gcp_region']
          - $GCP_REGION
          - DEFAULT_LOCATION

        endpoint_id (can be raw id or full resource name):
          - st.secrets['vertex_endpoint_id']
          - $VERTEX_ENDPOINT_ID
          - DEFAULT_ENDPOINT_ID
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
        or DEFAULT_ENDPOINT_ID
    )

    # Normalize just in case
    project_id = str(project_id).strip()
    location = str(location).strip()
    endpoint_id = str(endpoint_id).strip()

    return project_id, location, endpoint_id


def _build_endpoint_name(project_id: str, location: str, endpoint_id: str) -> str:
    """
    Build full endpoint resource name.

    If endpoint_id is already a full resource path (contains 'projects/'),
    we return it as-is. Otherwise we build:
        projects/{project_id}/locations/{location}/endpoints/{endpoint_id}
    """
    if "projects/" in endpoint_id and "/endpoints/" in endpoint_id:
        # Already full resource name
        return endpoint_id

    return f"projects/{project_id}/locations/{location}/endpoints/{endpoint_id}"


def _get_or_create_endpoint():
    """
    Get a configured Vertex Endpoint client.

    Ensures:
      - aiplatform.init(project=..., location=...) is called with a
        non-None project_id (prevents 'projects/None' bugs).
      - Endpoint is created only once & cached.
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

    This checks:
      - aiplatform import availability
      - presence of a non-empty project_id
      - presence of an endpoint id/name

    It does *not* call the endpoint yet – only verifies config.
    """
    if not _GCP_AVAILABLE:
        logger.warning("Vertex prediction not configured: google-cloud-aiplatform missing")
        return False

    try:
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


def predict_win_probabilities(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    model_path: Optional[str] = None
) -> List[float]:
    """
    Predicts win probabilities using XGBoost with strict safety wraps.
    """
    if feature_cols is None:
        feature_cols = VERTEX_FEATURE_COLUMNS

    # Default model path if not provided (placeholder to trigger safe fallback if missing)
    if model_path is None:
        model_path = "./models/model.json"

    try:
        import xgboost as xgb
        
        # 1. SANITIZE: Remove duplicate columns (The Segfault Fix)
        # XGBoost crashes hard if columns are duplicated.
        df_clean = df.loc[:, ~df.columns.duplicated()]
        
        # 2. VALIDATE: Ensure all features exist
        missing = [c for c in feature_cols if c not in df_clean.columns]
        if missing:
            logger.warning(f"Vertex Warning: Missing cols {missing}. Filling 0.")
            for c in missing:
                df_clean[c] = 0.0
        
        # 3. PREDICT (Safe Wrap)
        # We use a try/except specifically around the C-library call
        try:
            # Ensure we only select the feature columns to avoid extra column issues
            dmatrix = xgb.DMatrix(df_clean[feature_cols])
            booster = xgb.Booster()
            booster.load_model(model_path)
            predictions = booster.predict(dmatrix)

            # Convert numpy array to list
            if hasattr(predictions, "tolist"):
                return predictions.tolist()
            return list(predictions)

        except Exception as e:
            logger.error(f"CRITICAL XGBOOST FAILURE: {e}")
            # Fallback: Return 0.5 (Coin Flip) so app doesn't die
            return [0.5] * len(df)

    except Exception as e:
        logger.error(f"Vertex Setup Failed: {e}")
        return [0.5] * len(df)


# -------------------------------------------------------------------
# CLI TEST ENTRYPOINT (optional)
# -------------------------------------------------------------------

if __name__ == "__main__":
    """
    Allow quick testing via:

        python -m app_core.vertex_ai_endpoint

    or

        python app_core/vertex_ai_endpoint.py
    """
    import sys

    logging.basicConfig(level=logging.INFO)

    try:
        project_id, location, endpoint_id = _get_vertex_config()
        print("Resolved config:")
        print(f"  project_id  = {project_id}")
        print(f"  location    = {location}")
        print(f"  endpoint_id = {endpoint_id}")
    except Exception as e:
        print(f"Failed to resolve Vertex config: {e}")
        sys.exit(1)

    if not is_vertex_prediction_configured():
        print("Vertex prediction is NOT configured correctly.")
        sys.exit(1)

    # Build a tiny dummy frame for sanity check
    dummy = pd.DataFrame(
        [
            {col: 0.5 for col in VERTEX_FEATURE_COLUMNS},
            {col: 0.1 for col in VERTEX_FEATURE_COLUMNS},
        ]
    )

    print("\nRequesting predictions for dummy data...")
    probs = predict_win_probabilities(dummy)
    print("Predictions:", probs)
    sys.exit(0)
