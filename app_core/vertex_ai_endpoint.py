"""Vertex AI prediction helper for ParlayPicker.

This module centralizes configuration for the deployed Vertex endpoint and
exposes a small helper to score DataFrames of model features.
"""
from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

try:
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover - Streamlit not available in tests
    st = None

try:  # Lazy import to avoid hard dependency during cold starts
    from google.cloud import aiplatform
except Exception:  # pragma: no cover - allow module import without SDK
    aiplatform = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_vertex_settings() -> dict:
    """Load Vertex settings from env vars or Streamlit secrets.

    Environment variables take precedence, followed by ``st.secrets["vertex"]``.
    """
    vertex_secret = {}
    try:
        if st is not None:
            vertex_secret = st.secrets.get("vertex", {})  # type: ignore[arg-type]
    except Exception:
        vertex_secret = {}

    project_id = os.environ.get("VERTEX_PROJECT_ID") or vertex_secret.get("project_id")
    location = os.environ.get("VERTEX_LOCATION") or vertex_secret.get("location") or "us-central1"
    endpoint_id = os.environ.get("VERTEX_ENDPOINT_ID") or vertex_secret.get("endpoint_id")

    # Optional SA credentials for local/dev usage
    credentials_info = None
    try:
        if st is not None and "gcp_service_account" in st.secrets:
            credentials_info = st.secrets.get("gcp_service_account")
        elif st is not None and "gcp_service_account" in st.session_state:
            credentials_info = st.session_state.get("gcp_service_account")
    except Exception:
        credentials_info = None

    return {
        "project_id": project_id,
        "location": location,
        "endpoint_id": endpoint_id,
        "credentials_info": credentials_info,
    }


def is_vertex_prediction_configured() -> bool:
    settings = _load_vertex_settings()
    if not settings.get("endpoint_id"):
        logger.warning("Vertex prediction disabled: missing endpoint id")
        return False
    if aiplatform is None:
        logger.warning("Vertex prediction disabled: google-cloud-aiplatform not installed")
        return False
    return True


# ---------------------------------------------------------------------------
# Endpoint caching + prediction
# ---------------------------------------------------------------------------

_vertex_endpoint = None


def _get_vertex_endpoint():
    global _vertex_endpoint
    if _vertex_endpoint is not None:
        return _vertex_endpoint

    settings = _load_vertex_settings()
    endpoint_id = settings.get("endpoint_id")
    project_id = settings.get("project_id")
    location = settings.get("location")

    if not endpoint_id:
        logger.error("VERTEX_ENDPOINT_ID is not configured; skipping Vertex predictions")
        return None

    try:
        credentials = None
        if settings.get("credentials_info"):
            cred_info = settings["credentials_info"]
            if isinstance(cred_info, str):
                cred_info = json.loads(cred_info)
            from google.oauth2 import service_account  # type: ignore

            credentials = service_account.Credentials.from_service_account_info(
                cred_info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )

        aiplatform.init(project=project_id, location=location, credentials=credentials)
        endpoint_path = endpoint_id
        if not endpoint_id.startswith("projects/"):
            endpoint_path = f"projects/{project_id}/locations/{location}/endpoints/{endpoint_id}"
        _vertex_endpoint = aiplatform.Endpoint(endpoint_path)
        logger.info("Vertex endpoint initialized: %s", endpoint_path)
        return _vertex_endpoint
    except Exception as exc:  # pragma: no cover - requires cloud access
        logger.error("Failed to initialize Vertex endpoint: %s", exc)
        return None


VERTEX_FEATURE_COLUMNS: List[str] = [
    "home_win_pct",
    "away_win_pct",
    "home_avg_points",
    "away_avg_points",
    "home_form_last5",
    "away_form_last5",
    "home_pd_last5",
    "away_pd_last5",
    "home_sos_last5",
    "away_sos_last5",
    "win_pct_diff",
    "avg_points_diff",
    "form_diff",
    "pd_last5_diff",
    "sos_diff",
    "def_rating_diff",
    "streak_diff",
    "spread_normalized",
    "public_betting_centered",
    "sharp_vs_public",
    "rest_advantage",
    "back_to_back",
    "primetime_game",
    "division_game",
    "injuries_impact",
    "weather_factor",
    "line_movement",
    "total_movement",
    "model_consensus",
    "theover_probability",
]


def predict_win_probabilities(df: pd.DataFrame, feature_columns: Iterable[str]) -> Optional[np.ndarray]:
    """Call the deployed Vertex endpoint and return win probabilities.

    Returns ``None`` on configuration/API errors so callers can gracefully
    fall back without crashing the Streamlit app.
    """
    if not is_vertex_prediction_configured():
        return None

    endpoint = _get_vertex_endpoint()
    if endpoint is None:
        return None

    try:
        # Ensure column order and numeric dtype
        df_payload = df.copy()
        for col in feature_columns:
            if col not in df_payload.columns:
                df_payload[col] = 0.0
        df_payload = df_payload[list(feature_columns)].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        instances = df_payload.values.tolist()
        prediction = endpoint.predict(instances=instances, timeout=5)
        raw_preds = prediction.predictions or []
        probs: List[float] = []
        for entry in raw_preds:
            if isinstance(entry, (list, tuple)):
                # Most AutoML/XGB endpoints return [class0, class1]
                if len(entry) == 1:
                    probs.append(float(entry[0]))
                else:
                    probs.append(float(entry[-1]))
            else:
                probs.append(float(entry))
        return np.array(probs, dtype=float)
    except Exception as exc:  # pragma: no cover - network boundary
        logger.warning("Vertex prediction failed: %s", exc)
        return None


def score_with_vertex(df: pd.DataFrame, feature_cols: Iterable[str]) -> tuple[pd.Series, str]:
    """Score a DataFrame with Vertex and return probabilities + status.

    The status string is "vertex" on success or "fallback:<reason>" on
    failure. Returned Series always aligns with ``df.index``.
    """

    try:
        if not is_vertex_prediction_configured():
            return pd.Series(np.nan, index=df.index), "fallback:not_configured"

        endpoint = _get_vertex_endpoint()
        if endpoint is None:
            return pd.Series(np.nan, index=df.index), "fallback:endpoint"

        df_payload = df.copy()
        for col in feature_cols:
            if col not in df_payload.columns:
                df_payload[col] = 0.0
        df_payload = df_payload[list(feature_cols)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        instances = df_payload.values.tolist()
        prediction = endpoint.predict(instances=instances, timeout=5)
        raw_preds = prediction.predictions or []
        probs: List[float] = []
        for entry in raw_preds:
            if isinstance(entry, (list, tuple)):
                if len(entry) == 1:
                    probs.append(float(entry[0]))
                else:
                    probs.append(float(entry[-1]))
            else:
                probs.append(float(entry))
        series = pd.Series(np.array(probs, dtype=float), index=df.index)
        return series, "vertex"
    except Exception as exc:  # pragma: no cover - network boundary
        print(f"[Vertex] Error: {exc}")
        logger.warning("Vertex prediction failed in score_with_vertex: %s", exc)
        return pd.Series(np.nan, index=df.index), "fallback:error"


def quick_vertex_sanity_check():
    """Tiny sanity check to run in a REPL without Streamlit."""
    sample = pd.DataFrame(
        [
            {
                "home_win_pct": 0.55,
                "away_win_pct": 0.45,
                "home_avg_points": 112,
                "away_avg_points": 108,
                "home_form_last5": 3,
                "away_form_last5": 2,
                "home_pd_last5": 4,
                "away_pd_last5": -1,
                "home_sos_last5": 1.2,
                "away_sos_last5": 0.8,
                "win_pct_diff": 0.1,
                "avg_points_diff": 4,
                "form_diff": 1,
                "pd_last5_diff": 5,
                "sos_diff": 0.4,
                "def_rating_diff": 1.0,
                "streak_diff": 1,
                "spread_normalized": -1.5,
                "public_betting_centered": 0.0,
                "sharp_vs_public": 0.1,
                "rest_advantage": 1,
                "back_to_back": 0,
                "primetime_game": 0,
                "division_game": 1,
                "injuries_impact": 0,
                "weather_factor": 0,
                "line_movement": 0.5,
                "total_movement": -0.25,
                "model_consensus": 0.52,
                "theover_probability": 0.51,
            }
        ]
    )
    probs = predict_win_probabilities(sample, VERTEX_FEATURE_COLUMNS)
    print("Sample Vertex probabilities:", probs)


__all__ = [
    "VERTEX_FEATURE_COLUMNS",
    "predict_win_probabilities",
    "is_vertex_prediction_configured",
    "quick_vertex_sanity_check",
]
