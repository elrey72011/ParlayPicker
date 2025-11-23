"""
Vertex AI ML Predictions for Parlay Desk
"""
import streamlit as st
from google.cloud import aiplatform
from google.oauth2 import service_account
import json

def get_vertex_ai_prediction(features):
    """
    Get prediction from Vertex AI endpoint
    
    Args:
        features: List of feature values for prediction
        
    Returns:
        Prediction result (probability or class)
    """
    try:
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
        endpoint = aiplatform.Endpoint(
            endpoint_name=f"projects/{st.secrets['vertex_ai']['project_id']}/locations/{st.secrets['vertex_ai']['location']}/endpoints/{st.secrets['vertex_ai']['endpoint_id']}"
        )
        
        # Make prediction
        prediction = endpoint.predict(instances=[features])
        
        return prediction.predictions[0]
        
    except Exception as e:
        st.error(f"ML Prediction Error: {e}")
        return None

def is_ml_enabled():
    """Check if ML predictions are enabled"""
    try:
        return st.secrets["vertex_ai"]["enabled"]
    except:
        return False
