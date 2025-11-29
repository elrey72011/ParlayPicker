"""
ML Predictions Module for ParlayDesk
Uses Google Gemini via Vertex AI for sports betting predictions
"""

import streamlit as st
import logging
import json
from typing import Dict, Any, Optional
import vertexai
from vertexai.generative_models import GenerativeModel

logger = logging.getLogger(__name__)

# Cache the model initialization
@st.cache_resource
def get_gemini_model(project_id: str, location: str = "us-central1"):
    """Initialize and cache the Gemini model"""
    try:
        # Set up authentication from secrets.toml or uploaded file
        import os
        import tempfile
        
        service_account_info = None
        
        # Try to get from secrets.toml first (preferred)
        try:
            if 'gcp_service_account' in st.secrets:
                # Secrets.toml format - convert to dict
                gcp_secrets = st.secrets['gcp_service_account']
                service_account_info = {
                    'type': gcp_secrets.get('type', 'service_account'),
                    'project_id': gcp_secrets.get('project_id'),
                    'private_key_id': gcp_secrets.get('private_key_id'),
                    'private_key': gcp_secrets.get('private_key'),
                    'client_email': gcp_secrets.get('client_email'),
                    'client_id': gcp_secrets.get('client_id'),
                    'auth_uri': gcp_secrets.get('auth_uri'),
                    'token_uri': gcp_secrets.get('token_uri'),
                    'auth_provider_x509_cert_url': gcp_secrets.get('auth_provider_x509_cert_url'),
                    'client_x509_cert_url': gcp_secrets.get('client_x509_cert_url'),
                    'universe_domain': gcp_secrets.get('universe_domain', 'googleapis.com'),
                }
                logger.info("✓ Loaded service account from secrets.toml")
        except Exception as e:
            logger.warning(f"Could not load from secrets.toml: {e}")
        
        # Fallback to session state (uploaded file)
        if not service_account_info and 'gcp_service_account' in st.session_state:
            service_account_info = st.session_state['gcp_service_account']
            logger.info("✓ Loaded service account from uploaded file")
        
        if service_account_info:
            # Write credentials to temporary file for Vertex AI
            import json
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(service_account_info, f)
                cred_path = f.name
            
            # Set environment variable for Google Cloud auth
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = cred_path
            logger.info(f"✓ Set GCP credentials file: {cred_path}")
        else:
            logger.warning("⚠️ No GCP service account found in secrets or session state")
        
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=location)
        model = GenerativeModel("gemini-1.5-flash-002")
        logger.info(f"✓ Gemini model initialized for project {project_id}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to initialize Gemini model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def is_vertex_ai_enabled() -> bool:
    """Check if Vertex AI is properly configured"""
    try:
        # Check for GCP project ID in session state or secrets
        project_id = st.session_state.get('gcp_project_id')
        if not project_id:
            project_id = st.secrets.get('gcp_project_id')
        
        if not project_id:
            return False
            
        # Check for service account credentials
        if 'gcp_service_account' not in st.secrets:
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error checking Vertex AI config: {e}")
        return False


def get_vertex_ai_prediction(features: Dict[str, Any], context: str) -> Optional[float]:
    """
    Get game prediction from Gemini via Vertex AI
    
    Args:
        features: Dictionary containing game features
            - home_team: str
            - away_team: str
            - league: str
            - home_ml_odds: float
            - away_ml_odds: float
            - implied_home_prob: float
            - home_spread: float
        context: String describing the matchup (e.g., "Team A @ Team B (NFL)")
    
    Returns:
        float: Predicted HOME team win probability (0.0 to 1.0)
        None: If prediction fails
    """
    try:
        # Get project configuration
        project_id = st.session_state.get('gcp_project_id', 'elite-hangar-479017-m8')
        region = st.session_state.get('gcp_region', 'us-central1')
        
        # Get or initialize the model
        model = get_gemini_model(project_id, region)
        if not model:
            logger.error("Gemini model not available")
            return None
        
        # Extract features
        home_team = features.get('home_team', '')
        away_team = features.get('away_team', '')
        league = features.get('league', '')
        home_ml = features.get('home_ml_odds', 0)
        away_ml = features.get('away_ml_odds', 0)
        implied_prob = features.get('implied_home_prob', 0.5)
        spread = features.get('home_spread', 0)
        
        # Create prompt for Gemini
        prompt = f"""You are a sports betting expert analyzing a {league} game.

Game: {away_team} @ {home_team}

Market Data:
- Home ML Odds: {home_ml:+d}
- Away ML Odds: {away_ml:+d}
- Spread: {spread:+.1f}
- Market Implied Home Win Probability: {implied_prob:.1%}

Analyze this matchup and predict the HOME team's win probability.

Consider:
1. Recent team performance and trends
2. Head-to-head history
3. Home/away splits
4. Key player availability
5. Situational factors (rest, travel, motivation)
6. Statistical matchups and team strengths

Return ONLY a JSON object with this exact format:
{{
    "home_win_probability": 0.XX,
    "confidence": 0.XX,
    "key_factors": ["factor1", "factor2", "factor3"]
}}

The home_win_probability should be between 0.0 and 1.0 (e.g., 0.55 for 55% chance).
The confidence should be between 0.0 and 1.0 indicating prediction confidence.
Provide 2-4 key factors influencing your prediction.

JSON response:"""

        # Call Gemini
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Parse JSON response
        # Remove markdown code blocks if present
        if response_text.startswith('```'):
            response_text = response_text.split('```')[1]
            if response_text.startswith('json'):
                response_text = response_text[4:]
        response_text = response_text.strip()
        
        result = json.loads(response_text)
        
        # Extract probability
        home_prob = float(result.get('home_win_probability', 0.5))
        
        # Validate probability
        if not 0.0 <= home_prob <= 1.0:
            logger.warning(f"Invalid probability {home_prob} for {context}, defaulting to 0.5")
            home_prob = 0.5
        
        logger.info(f"✓ Gemini prediction for {context}: {home_prob:.1%} (confidence: {result.get('confidence', 0):.1%})")
        
        return home_prob
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Gemini response for {context}: {e}")
        logger.error(f"Response text: {response_text}")
        return None
        
    except Exception as e:
        logger.error(f"Vertex AI prediction failed for {context}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def show_vertex_ai_prediction_section(home_team: str, away_team: str, league: str, 
                                       home_ml: float, away_ml: float, 
                                       home_spread: float, implied_home_prob: float):
    """
    Display Vertex AI prediction section in Streamlit
    
    This is a UI component that shows the AI prediction with details
    """
    if not is_vertex_ai_enabled():
        st.warning("⚠️ Vertex AI not configured. Configure in sidebar to enable AI predictions.")
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


# Backward compatibility
def process_moneyline_with_ml_predictions(*args, **kwargs):
    """Legacy function for compatibility"""
    logger.warning("process_moneyline_with_ml_predictions called but not implemented - using Vertex AI instead")
    return {
        'home_prob': 0.5,
        'away_prob': 0.5,
        'confidence': 0.5,
        'edge': 0.0,
        'model_used': 'vertex_ai',
        'prediction_successful': False
    }
