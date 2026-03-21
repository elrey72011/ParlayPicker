"""
ML Predictions Module for ParlayDesk
Uses Google Gemini via Vertex AI for sports betting predictions
"""

import json
import logging
import re
from typing import Optional

import pandas as pd

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


from google.cloud import aiplatform

# Replace this with the ID from your screenshot/verify.py output
# Endpoint: parlaypicker-xgb-v3-master-endpoint
ENDPOINT_ID = "6435317312558989312" 
PROJECT_ID = "elite-hangar-479017-m8"
LOCATION = "us-central1"

def get_vertex_ai_prediction(features_dict, game_context=None):
    """
    Get prediction from your custom trained XGBoost model.
    """
    try:
        # 1. Define the exact feature order your model expects 
        # (Must match FEATURE_NAMES from train_vertex_model.py)
        expected_features = [
            "home_win_pct", "away_win_pct", "home_ppg", "away_ppg",
            "home_oppg", "away_oppg", "spread_normalized", "home_last_5",
            "away_last_5", "home_home_record", "away_away_record", "head_to_head",
            "rest_advantage", "injuries_impact", "weather_factor", "public_betting_pct",
            "sharp_money_indicator", "line_movement", "total_movement", "model_consensus",
            "theover_probability", "implied_home_prob", "home_streak", "away_streak",
            "division_game", "back_to_back", "primetime_game"
        ]
        
        # 2. Convert dictionary to sorted list (XGBoost expects an array, not a dict)
        instance_list = []
        for feature in expected_features:
            # Default to 0.5 or 0 if feature is missing
            val = features_dict.get(feature, 0.0)
            try:
                instance_list.append(float(val))
            except:
                instance_list.append(0.0)

        # 3. Initialize Endpoint
        endpoint = aiplatform.Endpoint(
            endpoint_name=f"projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}"
        )

        # 4. Make Prediction
        prediction = endpoint.predict(instances=[instance_list])
        
        # Vertex usually returns [probability_class_0, probability_class_1]
        # Assuming class 1 is "Home Win"
        probs = prediction.predictions[0] 
        
        # Handle different return formats (scalar vs list)
        if isinstance(probs, list):
            home_win_prob = probs[1] # Probability of class 1
        else:
            home_win_prob = probs    # Scalar probability
            
        return float(home_win_prob)

    except Exception as e:
        print(f"❌ Custom Model Prediction Failed: {e}")
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


from app_core.sharp_engine.probability_engine import ensemble_probability
from app_core.sharp_engine.ev_engine import expected_value, bet_signal
from app_core.sharp_engine.bankroll_manager import kelly_fraction


def enrich_predictions_dataframe(df: pd.DataFrame, min_edge: float = 0.03) -> pd.DataFrame:
    """Add sharp-engine betting columns to prediction output without breaking existing schema."""
    if df is None or df.empty:
        return df

    out = df.copy()

    if "ml_probability" in out.columns and "ml_prob" not in out.columns:
        out["ml_prob"] = pd.to_numeric(out["ml_probability"], errors="coerce")
    elif "ml_prob" not in out.columns:
        out["ml_prob"] = pd.to_numeric(out.get("final_probability"), errors="coerce")

    if "market_probability" in out.columns and "market_prob" not in out.columns:
        out["market_prob"] = pd.to_numeric(out["market_probability"], errors="coerce")
    elif "market_prob" not in out.columns:
        out["market_prob"] = pd.to_numeric(out.get("Implied_Prob"), errors="coerce")

    for c in ["kalshi_prob", "theover_prob", "sentiment_prob", "ml_prob", "market_prob"]:
        if c not in out.columns:
            out[c] = 0.0

    out["ensemble_probability"] = out.apply(lambda r: ensemble_probability(r.to_dict()), axis=1)

    odds_col = "Odds" if "Odds" in out.columns else "price"
    out["expected_value"] = out.apply(
        lambda r: expected_value(float(r["ensemble_probability"]), float(r[odds_col]))
        if pd.notna(r.get(odds_col)) else None,
        axis=1,
    )
    out["kelly_fraction"] = out.apply(
        lambda r: kelly_fraction(float(r["ensemble_probability"]), float(r[odds_col]))
        if pd.notna(r.get(odds_col)) else 0.0,
        axis=1,
    )
    out["bet_signal"] = out.apply(lambda r: bet_signal(r, min_edge=min_edge) if pd.notna(r.get("expected_value")) else False, axis=1)

    return out


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
