import streamlit as st

st.title("Secrets Debug Test")

st.write("### Testing st.secrets access:")

try:
    st.write("**All available secrets:**")
    st.write(list(st.secrets.keys()))
    
    st.write("### Trying to access GCP_PROJECT_ID:")
    if "GCP_PROJECT_ID" in st.secrets:
        st.success(f"✅ Found GCP_PROJECT_ID: {st.secrets['GCP_PROJECT_ID']}")
    else:
        st.error("❌ GCP_PROJECT_ID not in secrets")
    
    st.write("### Trying lowercase gcp_project_id:")
    if "gcp_project_id" in st.secrets:
        st.success(f"✅ Found gcp_project_id: {st.secrets['gcp_project_id']}")
    else:
        st.error("❌ gcp_project_id not in secrets")
    
    st.write("### Trying to access ODDS_API_KEY:")
    if "ODDS_API_KEY" in st.secrets:
        st.success(f"✅ Found ODDS_API_KEY: {st.secrets['ODDS_API_KEY'][:10]}...")
    else:
        st.error("❌ ODDS_API_KEY not in secrets")
        
    st.write("### Trying gcp_service_account:")
    if "gcp_service_account" in st.secrets:
        st.success("✅ Found gcp_service_account section")
        st.write(list(st.secrets.gcp_service_account.keys()))
    else:
        st.error("❌ gcp_service_account not in secrets")
        
except Exception as e:
    st.error(f"❌ FATAL ERROR accessing secrets: {e}")
    import traceback
    st.code(traceback.format_exc())

st.write("### Session State Check:")
st.write(f"gcp_project_id in session: {st.session_state.get('gcp_project_id', 'NOT SET')}")
st.write(f"vertex_endpoint_id in session: {st.session_state.get('vertex_endpoint_id', 'NOT SET')}")
