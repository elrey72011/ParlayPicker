"""
Vertex AI ML Predictions for Parlay Desk
Adds Google Cloud Vertex AI predictions to your existing ML features
"""
import streamlit as st
from google.cloud import aiplatform
from google.oauth2 import service_account
import logging
import time

logger = logging.getLogger(__name__)


@st.cache_resource
def get_endpoint_connection():
    """
    Cache the endpoint connection (huge speedup!)
    This runs once and reuses the connection
    """
    try:
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
        
        aiplatform.init(
            project=st.secrets["vertex_ai"]["project_id"],
            location=st.secrets["vertex_ai"]["location"],
            credentials=credentials
        )
        
        endpoint_name = (
            f"projects/{st.secrets['vertex_ai']['project_id']}"
            f"/locations/{st.secrets['vertex_ai']['location']}"
            f"/endpoints/{st.secrets['vertex_ai']['endpoint_id']}"
        )
        
        return aiplatform.Endpoint(endpoint_name=endpoint_name)
    except Exception as e:
        logger.error(f"Failed to connect to Vertex AI endpoint: {e}")
        return None


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_vertex_ai_prediction_cached(features_tuple):
    """
    Cached predictions - instant for repeated requests
    
    Args:
        features_tuple: Tuple of features (must be hashable for caching)
        
    Returns:
        float: Prediction probability or None if failed
    """
    endpoint = get_endpoint_connection()
    
    if endpoint is None:
        return None
    
    try:
        features = list(features_tuple)
        prediction = endpoint.predict(instances=[features])
        return float(prediction.predictions[0])
    except Exception as e:
        logger.error(f"Vertex AI prediction error: {e}")
        return None


def get_vertex_ai_prediction(features):
    """
    Get prediction from Vertex AI endpoint (with caching)
    
    Args:
        features: List of feature values
        
    Returns:
        float: Prediction probability (0.0 to 1.0) or None if failed
    """
    return get_vertex_ai_prediction_cached(tuple(features))


def is_vertex_ai_enabled():
    """Check if Vertex AI predictions are enabled"""
    try:
        return st.secrets.get("vertex_ai", {}).get("enabled", False)
    except:
        return False


def show_vertex_ai_prediction_section(home_team, away_team, home_stats=None, away_stats=None):
    """
    Display Vertex AI prediction section in Streamlit
    
    Args:
        home_team: Home team name
        away_team: Away team name
        home_stats: Optional dict with home team stats (win_pct, avg_points, etc.)
        away_stats: Optional dict with away team stats
    """
    if not is_vertex_ai_enabled():
        return
    
    st.markdown("---")
    st.subheader("🤖 Google Cloud Vertex AI Prediction")
    st.caption("Advanced ML prediction powered by Google Cloud")
    
    # BUILD FEATURES
    # TODO: Replace with your actual feature extraction logic
    if home_stats and away_stats:
        features = [
            home_stats.get('win_pct', 0.5),     # Home team win %
            away_stats.get('win_pct', 0.5),     # Away team win %
            # Add more features as needed for your model
        ]
        st.info("✅ Using real team statistics")
    else:
        # Fallback to dummy features
        features = [1, 1]
        st.warning("⚠️ Using dummy features - pass real stats for accurate predictions")
    
    if st.button("Get Vertex AI Prediction", key=f"vertex_ai_predict_{home_team}_{away_team}"):
        start_time = time.time()
        
        with st.spinner("Getting prediction from Vertex AI..."):
            prediction = get_vertex_ai_prediction(features)
        
        elapsed_time = time.time() - start_time
        
        if prediction is not None:
            st.success(f"✅ Prediction received in {elapsed_time:.2f}s")
            
            # Show if cached
            if elapsed_time < 0.5:
                st.info("⚡ Cached result (instant)")
            
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
                st.info(f"📈 Model favors: **{home_team}** ({prediction * 100:.1f}% win probability)")
            else:
                st.info(f"📈 Model favors: **{away_team}** ({(1-prediction) * 100:.1f}% win probability)")
            
            # Show features used (for debugging)
            with st.expander("🔍 Debug: Features Used"):
                st.json({
                    "features": features,
                    "prediction_raw": float(prediction),
                    "elapsed_time": f"{elapsed_time:.2f}s"
                })
                
        else:
            st.error("❌ Vertex AI prediction failed. Check your configuration and logs.")


def show_vertex_ai_debug_info():
    """Show debug information about Vertex AI configuration"""
    st.markdown("---")
    st.subheader("🔍 Vertex AI Debug Info")
    
    try:
        enabled = is_vertex_ai_enabled()
        st.write(f"**Enabled:** {enabled}")
        
        if enabled:
            st.write(f"**Project ID:** {st.secrets['vertex_ai']['project_id']}")
            st.write(f"**Endpoint ID:** {st.secrets['vertex_ai']['endpoint_id']}")
            st.write(f"**Region:** {st.secrets['vertex_ai']['location']}")
            st.success("✅ Secrets configured correctly")
        else:
            st.warning("⚠️ Vertex AI is disabled in secrets")
            
    except Exception as e:
        st.error(f"❌ Configuration error: {e}")
