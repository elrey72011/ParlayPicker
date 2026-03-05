import streamlit as st


DEFAULT_SPORTS = ["NBA", "NFL", "NHL", "MLB", "NCAAB"]


def render_sidebar():
    st.sidebar.header("ParlayPicker Controls")

    sports = st.sidebar.multiselect("Select Sports", DEFAULT_SPORTS)

    max_rows = st.sidebar.slider("Max Games", min_value=10, max_value=500, value=150)

    st.sidebar.subheader("Analysis Engines")

    use_ml = st.sidebar.checkbox("Enable ML Predictions", True)
    use_vertex = st.sidebar.checkbox("Enable Vertex AI")
    use_gemini = st.sidebar.checkbox("Enable Gemini Analysis")

    st.sidebar.subheader("Diagnostics")
    show_debug = st.sidebar.checkbox("Display Debug Information", value=False)
    show_kalshi_diagnostics = st.sidebar.checkbox("Show Kalshi Diagnostics", value=False)

    st.sidebar.subheader("Data Uploads")

    theover_spreads = st.sidebar.file_uploader("Upload TheOver Spreads CSV", type=["csv"])
    theover_totals = st.sidebar.file_uploader("Upload TheOver Totals CSV", type=["csv"])

    run_analysis = st.sidebar.button("Run Master Analysis", type="primary")

    return {
        "sports": sports,
        "max_rows": max_rows,
        "use_ml": use_ml,
        "use_vertex": use_vertex,
        "use_gemini": use_gemini,
        "show_debug": show_debug,
        "show_kalshi_diagnostics": show_kalshi_diagnostics,
        "theover_spreads": theover_spreads,
        "theover_totals": theover_totals,
        "run_analysis": run_analysis,
    }
