from typing import MutableMapping
from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st


FALLBACK_SPORTS = ["NBA", "WNBA", "NHL", "NCAAB", "NFL", "NCAAF", "MLB"]
ALL_SPORTS_LABEL = "All Sports"


def _request_run_analysis(state: MutableMapping[str, object]) -> None:
    state["run_analysis_counter"] = int(state.get("run_analysis_counter", 0)) + 1

def _resolve_sports_options(dynamic_sports: list[str] | None = None) -> list[str]:
    if dynamic_sports:
        cleaned = [str(s).strip().upper() for s in dynamic_sports if str(s).strip()]
        if cleaned:
            deduped = sorted(set(cleaned))
            if deduped:
                return deduped
    return FALLBACK_SPORTS.copy()


def _sports_for_view(selected_view: str, sports_options: list[str]) -> list[str]:
    """Resolve the single sidebar view to the leagues sent to the pipeline."""
    options = list(dict.fromkeys(
        str(sport).strip().upper()
        for sport in sports_options
        if str(sport).strip()
    ))
    normalized = str(selected_view or "").strip().upper()
    if normalized == ALL_SPORTS_LABEL.upper():
        return options
    option_lookup = {option.upper(): option for option in options}
    selected = option_lookup.get(normalized)
    return [selected] if selected else options


def _initial_sport_view(previous_sports, sports_options: list[str]) -> str:
    """Migrate the old multiselect state without changing existing all-sport runs."""
    previous = (
        list(previous_sports)
        if isinstance(previous_sports, (list, tuple, set))
        else [previous_sports] if previous_sports else []
    )
    resolved = [
        str(sport).strip().upper()
        for sport in previous
        if str(sport).strip()
    ]
    option_lookup = {option.upper(): option for option in sports_options}
    if len(resolved) == 1 and resolved[0] in option_lookup:
        return option_lookup[resolved[0]]
    return ALL_SPORTS_LABEL


def render_sidebar(dynamic_sports: list[str] | None = None):
    st.sidebar.header("ParlayPicker Controls")

    sports_options = _resolve_sports_options(dynamic_sports)
    if not sports_options:
        sports_options = FALLBACK_SPORTS.copy()

    # The old multiselect initialized every sport as selected. Clicking "NFL"
    # therefore did not mean NFL-only and could remove NFL instead. Use an
    # explicit single-sport view while retaining an unambiguous all-sports mode.
    view_options = [ALL_SPORTS_LABEL, *sports_options]
    if "selected_sport_view" not in st.session_state:
        st.session_state["selected_sport_view"] = _initial_sport_view(
            st.session_state.get("selected_sports"), sports_options
        )
    elif st.session_state["selected_sport_view"] not in view_options:
        st.session_state["selected_sport_view"] = ALL_SPORTS_LABEL

    selected_view = st.sidebar.selectbox(
        "Select Sport",
        view_options,
        key="selected_sport_view",
    )
    sports = _sports_for_view(selected_view, sports_options)
    # Preserve the legacy state key used by Data Diagnostics and maintenance tools.
    st.session_state["selected_sports"] = sports

    bankroll = st.sidebar.number_input("Bankroll", min_value=100.0, value=1000.0, step=50.0, key="bankroll")

    st.sidebar.button(
        "Refresh picks",
        type="primary",
        on_click=_request_run_analysis,
        args=(st.session_state,),
    )

    advanced = st.sidebar.expander("Settings & research", expanded=False)
    advanced.subheader("Analysis Engines")

    use_ml = advanced.checkbox("Enable ML Predictions", True, key="use_ml")
    from app.ui.ncaaf_inventory import schedule_controls
    schedule_start, schedule_end = schedule_controls(advanced, sports)
    coverage_date = advanced.date_input('Coverage Eastern date',
        value=datetime.now(ZoneInfo('America/New_York')).date(), key='coverage_eastern_date',
        help='Reconcile retained schedules for this date. This control does not fetch a schedule or run analysis.')
    use_gemini = advanced.checkbox(
        "Require Gemini Review for Bets",
        value=True,
        key="use_gemini",
        help=(
            "When enabled, funded game picks require a matching "
            "MEDIUM/HIGH Gemini review. MEDIUM uses 75% of the normal stake; "
            "disagreement, weak/missing analysis, or API failure holds the bet at $0."
        ),
    )

    advanced.caption("Gemini structured reviews: up to 20 requests per UTC day by default on this deployment. Exact unchanged batches are cached for 10 minutes. Configure PARLAYPICKER_GEMINI_DAILY_REQUESTS in secrets; 0 disables new review requests. Local accounting resets if deployment storage is replaced.")
    if st.session_state.get("gemini_review_limit_status"):
        advanced.warning(st.session_state["gemini_review_limit_status"])
    advanced.subheader("Diagnostics")
    show_debug = advanced.checkbox("Display Debug Information", value=False, key="show_debug")
    show_kalshi_diagnostics = advanced.checkbox("Show Kalshi Diagnostics", value=False, key="show_kalshi_diagnostics")

    advanced.subheader("Data Uploads")

    # The workbook (.xlsx) also carries the raw page's sharp-money and public-split signals.
    theover_spreads = advanced.file_uploader("Upload TheOver Spreads (CSV or workbook)", type=["csv", "xlsx"], key="theover_spreads")
    theover_totals = advanced.file_uploader("Upload TheOver Totals (CSV or workbook)", type=["csv", "xlsx"], key="theover_totals")

    run_counter = int(st.session_state.get("run_analysis_counter", 0))

    return {
        "sports": sports,
        "bankroll": bankroll,
        "use_ml": use_ml,
        "use_gemini": use_gemini,
        "show_debug": show_debug,
        "show_kalshi_diagnostics": show_kalshi_diagnostics,
        "theover_spreads": theover_spreads,
        "theover_totals": theover_totals,
        "schedule_start": schedule_start,
        "schedule_end": schedule_end,
        "coverage_date": coverage_date.isoformat(),
        "run_analysis_counter": run_counter,
    }
