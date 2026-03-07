from typing import MutableMapping

import streamlit as st


FALLBACK_SPORTS = ["NBA", "NHL", "NCAAB", "NFL", "MLB"]




def _request_run_analysis(state: MutableMapping[str, object]) -> None:
    state["run_analysis_requested"] = True

def _resolve_sports_options(dynamic_sports: list[str] | None = None) -> list[str]:
    if dynamic_sports:
        cleaned = [str(s).strip().upper() for s in dynamic_sports if str(s).strip()]
        if cleaned:
            deduped = sorted(set(cleaned))
            if deduped:
                return deduped
    return FALLBACK_SPORTS.copy()


def render_sidebar(dynamic_sports: list[str] | None = None):
    st.sidebar.header("ParlayPicker Controls")

    sports_options = _resolve_sports_options(dynamic_sports)
    if not sports_options:
        sports_options = FALLBACK_SPORTS.copy()

    # Initialise once — do NOT pass default= to a keyed widget; Streamlit owns
    # the value after first render and the mismatch causes an infinite rerun.
    if "selected_sports" not in st.session_state:
        st.session_state["selected_sports"] = sports_options.copy()

    sports = st.sidebar.multiselect(
        "Select Sports",
        sports_options,
        key="selected_sports",
    )
    if not sports:
        sports = sports_options.copy()

    bankroll = st.sidebar.number_input("Bankroll", min_value=100.0, value=1000.0, step=50.0, key="bankroll")

    st.sidebar.subheader("Analysis Engines")

    use_ml = st.sidebar.checkbox("Enable ML Predictions", True, key="use_ml")
    use_gemini = st.sidebar.checkbox("Enable Gemini Analysis", key="use_gemini")

    st.sidebar.subheader("Diagnostics")
    show_debug = st.sidebar.checkbox("Display Debug Information", value=False, key="show_debug")
    show_kalshi_diagnostics = st.sidebar.checkbox("Show Kalshi Diagnostics", value=False, key="show_kalshi_diagnostics")

    st.sidebar.subheader("Data Uploads")

    theover_spreads = st.sidebar.file_uploader("Upload TheOver Spreads CSV", type=["csv"], key="theover_spreads")
    theover_totals = st.sidebar.file_uploader("Upload TheOver Totals CSV", type=["csv"], key="theover_totals")

    st.sidebar.button(
        "Run Master Analysis",
        type="primary",
        on_click=_request_run_analysis,
        args=(st.session_state,),
    )

    run_clicked = bool(st.session_state.pop("run_analysis_requested", False))

    return {
        "sports": sports,
        "bankroll": bankroll,
        "use_ml": use_ml,
        "use_gemini": use_gemini,
        "show_debug": show_debug,
        "show_kalshi_diagnostics": show_kalshi_diagnostics,
        "theover_spreads": theover_spreads,
        "theover_totals": theover_totals,
        "run_analysis": run_clicked,
    }
