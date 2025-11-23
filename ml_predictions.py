"""
Vertex AI ML Predictions for Parlay Desk
Adds Google Cloud Vertex AI predictions to your existing ML features
"""
import streamlit as st
from google.cloud import aiplatform
from google.oauth2 import service_account
import logging

logger = logging.getLogger(__name__)

def get_vertex_ai_prediction(features):
    """
    Get prediction from Vertex AI endpoint
    
    Args:
        features: List of feature values [feature1, feature2, ...]
        
    Returns:
        float: Prediction probability (0.0 to 1.0) or None if failed
    """
    try:
        # Check if Vertex AI is enabled
        if not st.secrets.get("vertex_ai", {}).get("enabled", False):
            return None
        
        # Get credentials from secrets
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
        
        # Initialize Vertex AI
        aiplatform.init(
            project=st.secrets["vertex_ai"]["project_id"],
            location=st.secrets["vertex_ai"]["location"],
            credentials=credentials
        )
        
        # Get endpoint
        endpoint_name = (
            f"projects/{st.secrets['vertex_ai']['project_id']}"
            f"/locations/{st.secrets['vertex_ai']['location']}"
            f"/endpoints/{st.secrets['vertex_ai']['endpoint_id']}"
        )
        endpoint = aiplatform.Endpoint(endpoint_name=endpoint_name)
        
        # Make prediction
        prediction = endpoint.predict(instances=[features])
        
        return float(prediction.predictions[0])
        
    except Exception as e:
        logger.error(f"Vertex AI prediction error: {e}")
        return None


def is_vertex_ai_enabled():
    """Check if Vertex AI predictions are enabled"""
    try:
        return st.secrets.get("vertex_ai", {}).get("enabled", False)
    except:
        return False


def show_vertex_ai_prediction_section(home_team, away_team):
    """
    Display Vertex AI prediction section in Streamlit
    
    Args:
        home_team: Home team name
        away_team: Away team name
    """
    if not is_vertex_ai_enabled():
        return
    
    st.markdown("---")
    st.subheader("🤖 Google Cloud Vertex AI Prediction")
    st.caption("Advanced ML prediction powered by Google Cloud")
    
    # Example: Create simple features from team names
    # TODO: Replace with actual features from your app
    features = [1, 1]  # Placeholder - use real features
    
    if st.button("Get Vertex AI Prediction", key="vertex_ai_predict"):
        with st.spinner("Getting prediction from Vertex AI..."):
            prediction = get_vertex_ai_prediction(features)
            
            if prediction is not None:
                st.success("✅ Prediction received!")
                
                # Display prediction
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Win Probability",
                        f"{prediction * 100:.1f}%",
                        help="Vertex AI predicted win probability"
                    )
                
                with col2:
                    confidence = abs(prediction - 0.5) * 2  # Convert to 0-1 scale
                    st.metric(
                        "Confidence",
                        f"{confidence * 100:.1f}%",
                        help="How confident the model is"
                    )
                
                # Show recommendation
                if prediction > 0.5:
                    st.info(f"📈 Model favors: **{home_team}**")
                else:
                    st.info(f"📈 Model favors: **{away_team}**")
                
            else:
                st.error("❌ Vertex AI prediction failed. Check your configuration.")
