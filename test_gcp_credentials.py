"""
Test script to verify GCP credentials are loading correctly
Add this temporarily to your streamlit_app.py to debug
"""

import streamlit as st

st.title("GCP Credentials Test")

# Test 1: Check if secrets exist
st.header("1. Secrets Check")
if 'gcp_service_account' in st.secrets:
    st.success("✓ gcp_service_account found in secrets")
    
    # Show what keys are available
    try:
        keys = list(st.secrets['gcp_service_account'].keys())
        st.write("Available keys:", keys)
        
        # Test reading each field
        st.subheader("Field Values:")
        for key in ['type', 'project_id', 'client_email', 'private_key_id']:
            try:
                value = st.secrets['gcp_service_account'][key]
                if key == 'private_key':
                    st.write(f"- {key}: [REDACTED - length {len(str(value))}]")
                else:
                    st.write(f"- {key}: {value}")
            except Exception as e:
                st.error(f"- {key}: ERROR - {e}")
                
    except Exception as e:
        st.error(f"Error reading secrets: {e}")
else:
    st.error("✗ gcp_service_account NOT found in secrets")
    st.write("Available secrets:", list(st.secrets.keys()))

# Test 2: Try to create credentials
st.header("2. Credentials Creation Test")
try:
    from google.oauth2 import service_account
    
    if 'gcp_service_account' in st.secrets:
        creds_dict = {
            'type': str(st.secrets['gcp_service_account']['type']),
            'project_id': str(st.secrets['gcp_service_account']['project_id']),
            'private_key_id': str(st.secrets['gcp_service_account']['private_key_id']),
            'private_key': str(st.secrets['gcp_service_account']['private_key']),
            'client_email': str(st.secrets['gcp_service_account']['client_email']),
            'client_id': str(st.secrets['gcp_service_account']['client_id']),
            'auth_uri': str(st.secrets['gcp_service_account']['auth_uri']),
            'token_uri': str(st.secrets['gcp_service_account']['token_uri']),
            'auth_provider_x509_cert_url': str(st.secrets['gcp_service_account']['auth_provider_x509_cert_url']),
            'client_x509_cert_url': str(st.secrets['gcp_service_account']['client_x509_cert_url']),
        }
        
        credentials = service_account.Credentials.from_service_account_info(creds_dict)
        st.success("✓ Credentials object created successfully!")
        st.write(f"Project ID: {credentials.project_id}")
        st.write(f"Service Account: {credentials.service_account_email}")
        
except Exception as e:
    st.error(f"✗ Failed to create credentials: {e}")
    import traceback
    st.code(traceback.format_exc())

# Test 3: Try to initialize Vertex AI
st.header("3. Vertex AI Initialization Test")
try:
    import vertexai
    from google.oauth2 import service_account
    
    if 'gcp_service_account' in st.secrets:
        creds_dict = {
            'type': str(st.secrets['gcp_service_account']['type']),
            'project_id': str(st.secrets['gcp_service_account']['project_id']),
            'private_key_id': str(st.secrets['gcp_service_account']['private_key_id']),
            'private_key': str(st.secrets['gcp_service_account']['private_key']),
            'client_email': str(st.secrets['gcp_service_account']['client_email']),
            'client_id': str(st.secrets['gcp_service_account']['client_id']),
            'auth_uri': str(st.secrets['gcp_service_account']['auth_uri']),
            'token_uri': str(st.secrets['gcp_service_account']['token_uri']),
            'auth_provider_x509_cert_url': str(st.secrets['gcp_service_account']['auth_provider_x509_cert_url']),
            'client_x509_cert_url': str(st.secrets['gcp_service_account']['client_x509_cert_url']),
        }
        
        credentials = service_account.Credentials.from_service_account_info(creds_dict)
        vertexai.init(project="elite-hangar-479017-m8", location="us-central1", credentials=credentials)
        
        st.success("✓ Vertex AI initialized successfully!")
        
        # Try to create a model
        from vertexai.generative_models import GenerativeModel
        model = GenerativeModel("gemini-1.5-flash-002")
        st.success("✓ Gemini model created successfully!")
        
except Exception as e:
    st.error(f"✗ Failed to initialize Vertex AI: {e}")
    import traceback
    st.code(traceback.format_exc())

st.info("If all 3 tests pass, the issue is elsewhere. If any fail, check the error messages above.")
