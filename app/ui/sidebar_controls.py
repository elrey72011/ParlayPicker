from typing import MutableMapping
from datetime import date, timedelta

import pandas as pd
import streamlit as st


FALLBACK_SPORTS = ["NBA", "NHL", "NCAAB", "NFL", "MLB"]




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
    prop_results_log = st.sidebar.file_uploader(
        "Upload Cumulative Graded Prop Ledger (optional)",
        type=["csv"],
        key="prop_results_log",
        help=(
            "Settled WIN/LOSS rows calibrate each prop market and direction. "
            "Upload the latest cumulative ledger at the start of each new app "
            "session; without it, props remain research-only."
        ),
    )

    previous_prop_exports = st.sidebar.file_uploader(
        "Upload Yesterday's Player-Prop Export(s)",
        type=["csv"],
        accept_multiple_files=True,
        key="previous_prop_exports",
        help=(
            "Upload the all-props export. For older slates, upload both the funded "
            "and research exports so every prediction can be graded."
        ),
    )
    previous_prop_date = st.sidebar.date_input(
        "Prop Slate Date",
        value=date.today() - timedelta(days=1),
        max_value=date.today(),
        key="previous_prop_date",
    )
    if st.sidebar.button("Grade Uploaded Player Props", key="grade_previous_props"):
        if not previous_prop_exports:
            st.sidebar.error("Upload at least one previous player-prop export first.")
        else:
            try:
                from app_core.prop_grading import (
                    grade_prop_export,
                    grading_summary,
                    merge_prop_ledgers,
                )

                cards = []
                for uploaded in previous_prop_exports:
                    uploaded.seek(0)
                    cards.append(pd.read_csv(uploaded))
                previous_card = pd.concat(cards, ignore_index=True, sort=False)
                previous_card = previous_card.drop_duplicates(
                    subset=[
                        column
                        for column in ("market_type", "best_pick", "pick", "line")
                        if column in previous_card.columns
                    ],
                    keep="last",
                )
                inferred_date = None
                if "game_date" in previous_card.columns:
                    dates = (
                        previous_card["game_date"]
                        .dropna().astype(str).str[:10].unique()
                    )
                    if len(dates) == 1:
                        inferred_date = dates[0]
                grade_date = inferred_date or previous_prop_date.isoformat()

                generated_prior = st.session_state.get("generated_prop_results_log")
                uploaded_prior = None
                if prop_results_log is not None:
                    prop_results_log.seek(0)
                    uploaded_prior = pd.read_csv(prop_results_log)
                    prop_results_log.seek(0)
                prior_ledger = merge_prop_ledgers(uploaded_prior, generated_prior)
                with st.spinner(f"Grading MLB player props for {grade_date}..."):
                    graded = grade_prop_export(previous_card, grade_date)
                    ledger = merge_prop_ledgers(prior_ledger, graded)
                st.session_state["generated_prop_results_log"] = ledger
                summary = grading_summary(graded)
                st.sidebar.success(
                    f"Graded {summary['graded']} props: "
                    f"{summary['wins']}-{summary['losses']} "
                    f"({summary['unresolved']} unresolved)."
                )
            except Exception as exc:
                st.sidebar.error(f"Player-prop grading failed: {exc}")

    generated_ledger = st.session_state.get("generated_prop_results_log")
    active_ledger = generated_ledger
    if (
        not isinstance(active_ledger, pd.DataFrame)
        or active_ledger.empty
    ) and prop_results_log is not None:
        try:
            prop_results_log.seek(0)
            active_ledger = pd.read_csv(prop_results_log)
            prop_results_log.seek(0)
        except (OSError, ValueError, TypeError):
            active_ledger = None

    if isinstance(active_ledger, pd.DataFrame) and not active_ledger.empty:
        from app_core.prop_grading import ledger_coverage_summary

        coverage = ledger_coverage_summary(active_ledger)
        date_range = (
            coverage["start_date"]
            if coverage["start_date"] == coverage["end_date"]
            else f"{coverage['start_date']} to {coverage['end_date']}"
        )
        st.sidebar.caption(
            f"Prop calibration history: {coverage['settled']} settled rows "
            f"across {coverage['date_count']} slate date(s) ({date_range})."
        )
        if coverage["settled"] > 0 and coverage["date_count"] <= 1:
            st.sidebar.warning(
                "Only one slate date is loaded. This is not yet a cumulative "
                "calibration history; download the updated ledger and upload it "
                "again at the start of the next app session."
            )

    if isinstance(generated_ledger, pd.DataFrame) and not generated_ledger.empty:
        st.sidebar.download_button(
            "Download Updated Graded Prop Ledger",
            generated_ledger.to_csv(index=False, encoding="utf-8-sig"),
            "prop_results_log.csv",
            mime="text/csv",
            key="download_prop_results_log",
        )

    st.sidebar.button(
        "Run Master Analysis",
        type="primary",
        on_click=_request_run_analysis,
        args=(st.session_state,),
    )

    run_counter = int(st.session_state.get("run_analysis_counter", 0))

    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠️ Data Maintenance")
    if st.sidebar.button("🔄 Sync Historical Rosters"):
        with st.spinner("Syncing rosters from The Odds API..."):
            try:
                from collect_historical_data import run_backfill
                run_backfill(sports=sports, days=2)
                st.sidebar.success("✅ Database Synced!")
            except Exception as e:
                st.sidebar.error(f"Sync failed: {e}")

    return {
        "sports": sports,
        "bankroll": bankroll,
        "use_ml": use_ml,
        "use_gemini": use_gemini,
        "show_debug": show_debug,
        "show_kalshi_diagnostics": show_kalshi_diagnostics,
        "theover_spreads": theover_spreads,
        "theover_totals": theover_totals,
        "prop_results_log": prop_results_log,
        "run_analysis_counter": run_counter,
    }
