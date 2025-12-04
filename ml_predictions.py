"""
ML Predictions Module for ParlayDesk
Uses Google Gemini via Vertex AI for sports betting predictions
"""

import json
import logging
import re
from typing import Optional

import streamlit as st
import vertexai
from google.oauth2 import service_account
from vertexai.generative_models import GenerativeModel

logger = logging.getLogger(__name__)

def is_vertex_ai_enabled() -> bool:
    """
    Returns True when we have enough config to *attempt* a Vertex / Gemini call.
    We treat either Streamlit secrets or an uploaded service-account JSON as valid.
    """
    try:
        # Check for service account in secrets.toml (preferred)
        if 'gcp_service_account' in st.secrets:
            project_id = st.secrets.get('gcp_project_id')
            if project_id:
                logger.info("✅ Vertex AI configured via secrets.toml")
                return True
        
        # Fallback: Check session state (uploaded file)
        if 'gcp_service_account' in st.session_state:
            logger.info("✅ Vertex AI configured via uploaded file")
            return True
            
        logger.warning("⚠️ Vertex AI not configured - no credentials found")
        return False
        
    except Exception as e:
        logger.error(f"Error checking Vertex AI config: {e}")
        return False


# Cache the model initialization
#@st.cache_resource
def get_gemini_model(
    project_id: Optional[str] = None,
    location: Optional[str] = None,
    model_name: str = "gemini-2.0-flash-001",
    _cache_version: int = 2  # ← ADD THIS to force new cache
) -> Optional[GenerativeModel]:
    """
    Initialize a Gemini model using either:
      * Sidebar-uploaded service account JSON (st.session_state["gcp_service_account"]), or
      * Streamlit secrets (st.secrets["gcp_service_account"]).

    Returns a GenerativeModel or None on failure.
    """
    try:
        # 1. Resolve project/location from UI or secrets
        project_id = (
            project_id
            or st.session_state.get("gcp_project_id")
            or st.secrets.get("gcp_project_id")
        )
        location = (
            location
            or st.session_state.get("gcp_region")
            or st.secrets.get("gcp_region")
            or "us-central1"
        )

        if not project_id:
            st.warning("⚠️ Vertex AI project ID is not configured.")
            logger.warning("Vertex AI disabled: missing project_id.")
            return None

        # 2. Build credentials
        credentials = None

        # Preferred: JSON uploaded in the sidebar, stored in session_state
        if "gcp_service_account" in st.session_state:
            sa_info = st.session_state["gcp_service_account"]
            if isinstance(sa_info, str):
                sa_info = json.loads(sa_info)
            credentials = service_account.Credentials.from_service_account_info(
                sa_info,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        # Fallback: JSON in st.secrets
        elif "gcp_service_account" in st.secrets:
            sa_info = st.secrets["gcp_service_account"]
            if isinstance(sa_info, str):
                sa_info = json.loads(sa_info)
            credentials = service_account.Credentials.from_service_account_info(
                sa_info,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )

        # 3. Init Vertex AI SDK
        vertexai.init(project=project_id, location=location, credentials=credentials)
        model = GenerativeModel(model_name)

        logger.info(f"Initialized Gemini model '{model_name}' for project={project_id}, location={location}")
        return model

    except Exception as e:
        st.warning(f"⚠️ Failed to initialize Vertex / Gemini: {e}")
        logger.error("Vertex/Gemini init failed", exc_info=True)
        return None


def get_vertex_ai_prediction(
    features: dict,
    game_context: str,
    project_id: Optional[str] = None,
    location: Optional[str] = None,
) -> Optional[float]:
    """
    Call Gemini/Vertex AI to get a win probability.
    Returns None on ANY failure — we never default to 0.5.
    """
    try:
        # 1. Check if ML is enabled
        if not is_vertex_ai_enabled():
            st.session_state["last_ml_source"] = "disabled"
            return None

        # 2. Build / load Gemini model
        model = get_gemini_model(project_id=project_id, location=location)
        if model is None:
            st.session_state["last_ml_source"] = "disabled"
            return None

        # 3. Build prompt
        prompt = (
            "You are predicting sports outcomes.\n"
            f"Context: {game_context}\n"
            f"Features: {json.dumps(features)}\n"
            "Return ONLY a float between 0 and 1 representing "
            "the model's estimated probability that the event is TRUE.\n"
        )

        # 4. Call Gemini
        response = model.generate_content(prompt)
        raw = response.text if hasattr(response, "text") else str(response)

        # 5. Parse a float
        try:
            p = float(raw.strip())
        except Exception:
            print("Vertex AI: Unable to parse response:", raw)
            return None

        # 6. Validate range
        if not (0.0 < p < 1.0):
            print("Vertex AI: Invalid prob (not 0–1):", p)
            return None

        return p

    except Exception as e:
        print("Vertex AI error:", e)
        return None

def show_vertex_ai_prediction_section(home_team: str, away_team: str, league: str, 
                                       home_ml: float, away_ml: float, 
                                       home_spread: float, implied_home_prob: float):
    """
    Display Vertex AI prediction section in Streamlit
    
    This is a UI component that shows the AI prediction with details
    """
    if not is_vertex_ai_enabled():
        st.warning("⚠️ Vertex AI not configured. Add credentials to secrets.toml to enable AI predictions.")
        return
    
    context = f"{away_team} @ {home_team} ({league})"
    
    features = {
        'home_team': home_team,
        'away_team': away_team,
        'league': league,
        'home_ml_odds': home_ml,
        'away_ml_odds': away_ml,
        'implied_home_prob': implied_home_prob,
        'home_spread': home_spread,
    }
    
    with st.spinner(f"🤖 Analyzing {context} with AI..."):
        prediction = get_vertex_ai_prediction(features, context)
    
    if prediction is not None:
        st.success(f"🎯 AI Prediction: {home_team} has {prediction:.1%} chance to win")
        
        # Show comparison with market
        edge = prediction - implied_home_prob
        if abs(edge) > 0.03:  # 3% edge threshold
            if edge > 0:
                st.info(f"📈 AI favors {home_team} (+{edge:.1%} edge vs market)")
            else:
                st.info(f"📉 AI favors {away_team} ({edge:.1%} edge vs market)")
        else:
            st.info("🤝 AI agrees with market pricing")
    else:
        st.error("❌ AI prediction failed")


def get_credential_source() -> str:
    """
    Determine where credentials are loaded from
    Returns: 'secrets.toml', 'uploaded_file', or 'none'
    """
    if 'gcp_service_account' in st.secrets:
        return 'secrets.toml'
    elif 'gcp_service_account' in st.session_state:
        return 'uploaded_file'
    else:
        return 'none'


def show_credential_status():
    """Display credential status in sidebar"""
    source = get_credential_source()
    
    if source == 'secrets.toml':
        st.success("✅ Credentials loaded from secrets.toml")
        st.caption("No file upload needed - using secrets configuration")
    elif source == 'uploaded_file':
        st.success("✅ Credentials loaded from uploaded file")
    else:
        st.warning("⚠️ No GCP credentials found")
        st.caption("Add [gcp_service_account] section to secrets.toml or upload JSON file")


# Backward compatibility
def process_moneyline_with_ml_predictions(*args, **kwargs):
    """Legacy function for compatibility"""
    logger.warning("process_moneyline_with_ml_predictions called but not implemented - using Vertex AI instead")
    # Return a clear "no prediction" payload instead of fake 50/50 defaults
    return {
        'home_prob': None,
        'away_prob': None,
        'confidence': None,
        'edge': None,
        'model_used': 'vertex_ai',
        'prediction_successful': False
    }
