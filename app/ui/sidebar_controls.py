from typing import MutableMapping
from datetime import date, timedelta

import pandas as pd
import streamlit as st


FALLBACK_SPORTS = ["NBA", "WNBA", "NHL", "NCAAB", "NFL", "NCAAF", "MLB"]
ALL_SPORTS_LABEL = "All Sports"


def _read_uploaded_prop_ledgers(uploaded_files) -> pd.DataFrame:
    """Read one or more uploaded ledgers and deduplicate cumulative history."""
    from app_core.prop_grading import merge_prop_ledgers

    uploads = (
        list(uploaded_files)
        if isinstance(uploaded_files, (list, tuple))
        else [uploaded_files] if uploaded_files is not None else []
    )
    merged = pd.DataFrame()
    for uploaded in uploads:
        try:
            uploaded.seek(0)
            frame = pd.read_csv(uploaded)
            uploaded.seek(0)
        except (AttributeError, OSError, TypeError, ValueError):
            continue
        merged = merge_prop_ledgers(merged, frame)
    return merged


def _read_bundled_prop_ledger() -> pd.DataFrame:
    """Load the deploy baseline with any locally recovered newer history."""
    from app_core.prop_calibration import load_prop_results_log

    ledger = load_prop_results_log()
    return ledger if isinstance(ledger, pd.DataFrame) else pd.DataFrame()


def _assemble_active_prop_ledger(
    uploaded_files=None,
    generated_ledger: pd.DataFrame | None = None,
    bundled_ledger: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Layer uploads and newly graded rows over the bundled calibration history."""
    from app_core.prop_grading import assemble_prop_ledgers

    bundled = (
        _read_bundled_prop_ledger()
        if bundled_ledger is None
        else bundled_ledger.copy()
    )
    uploaded = _read_uploaded_prop_ledgers(uploaded_files)
    return assemble_prop_ledgers(bundled, uploaded, generated_ledger)


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

    st.sidebar.subheader("Analysis Engines")

    use_ml = st.sidebar.checkbox("Enable ML Predictions", True, key="use_ml")
    use_gemini = st.sidebar.checkbox(
        "Require Gemini Review for Bets",
        value=True,
        key="use_gemini",
        help=(
            "When enabled, funded game picks and player props require a matching "
            "MEDIUM/HIGH Gemini review. MEDIUM uses 75% of the normal stake; "
            "disagreement, weak/missing analysis, or API failure holds the bet at $0."
        ),
    )

    st.sidebar.subheader("Diagnostics")
    show_debug = st.sidebar.checkbox("Display Debug Information", value=False, key="show_debug")
    show_kalshi_diagnostics = st.sidebar.checkbox("Show Kalshi Diagnostics", value=False, key="show_kalshi_diagnostics")

    st.sidebar.subheader("Data Uploads")

    theover_spreads = st.sidebar.file_uploader("Upload TheOver Spreads CSV", type=["csv"], key="theover_spreads")
    theover_totals = st.sidebar.file_uploader("Upload TheOver Totals CSV", type=["csv"], key="theover_totals")
    prop_results_log = st.sidebar.file_uploader(
        "Upload Latest Downloaded Graded Prop Ledger(s)",
        type=["csv"],
        accept_multiple_files=True,
        key="prop_results_log",
        help=(
            "The app starts with the cumulative ledger bundled in the repo. Upload "
            "your latest downloaded ledger whenever it is newer than that baseline, "
            "especially after an app deployment or restart. All sources are merged "
            "and deduplicated automatically."
        ),
    )
    bundled_ledger = _read_bundled_prop_ledger()

    previous_prop_exports = st.sidebar.file_uploader(
        "Upload Yesterday's Combined Player-Prop Export(s)",
        type=["csv"],
        accept_multiple_files=True,
        key="previous_prop_exports",
        help=(
            "Prefer player_props_all_export.csv, which includes every selected "
            "league and all DO NOT BET research rows. League-specific exports can "
            "silently omit another league. For older slates, upload every funded "
            "and research export so every prediction can be graded."
        ),
    )
    previous_prop_date = st.sidebar.date_input(
        "Prop Slate Date",
        value=date.today() - timedelta(days=1),
        max_value=date.today(),
        key="previous_prop_date",
    )
    generated_prior = st.session_state.get("generated_prop_results_log")
    pregrade_ledger = _assemble_active_prop_ledger(
        prop_results_log,
        generated_prior,
        bundled_ledger,
    )
    from app_core.prop_grading import ledger_history_gap_summary

    selected_gap = ledger_history_gap_summary(
        pregrade_ledger, previous_prop_date.isoformat()
    )
    if selected_gap["gap_detected"]:
        st.sidebar.warning(
            "The loaded cumulative prop ledger ends on "
            f"{selected_gap['latest_date']}, but you selected "
            f"{selected_gap['target_date']}. Grading remains available because the "
            "merge is additive, but upload any missing downloaded ledgers when "
            "available so calibration coverage stays complete."
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
                upload_names = []
                for uploaded in previous_prop_exports:
                    upload_names.append(str(getattr(uploaded, "name", "")))
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
                _league_counts = previous_card.get(
                    "league", pd.Series("MLB", index=previous_card.index)
                ).fillna("MLB").astype(str).str.upper().value_counts().to_dict()
                st.sidebar.caption(
                    "Uploaded grading rows by league: "
                    + ", ".join(
                        f"{league} {count}" for league, count in sorted(_league_counts.items())
                    )
                )
                _normalized_upload_names = [name.lower() for name in upload_names]
                _combined_export_present = any(
                    "player_props_all_export" in name
                    and "mlb_player_props_all_export" not in name
                    and "nfl_player_props_all_export" not in name
                    for name in _normalized_upload_names
                )
                if not _combined_export_present:
                    st.sidebar.warning(
                        "No combined player_props_all_export.csv was detected. "
                        "Verify that you uploaded every league-specific prop export; "
                        "otherwise the downloaded ledger will be incomplete."
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

                prior_ledger = _assemble_active_prop_ledger(
                    prop_results_log,
                    generated_prior,
                    bundled_ledger,
                )
                actual_gap = ledger_history_gap_summary(prior_ledger, grade_date)
                if actual_gap["gap_detected"]:
                    st.sidebar.warning(
                        "This slate will be appended across a calendar gap. Existing "
                        "rows will be preserved; upload missing ledgers later to backfill "
                        "the uncovered dates."
                    )
                with st.spinner(f"Grading player props for {grade_date}..."):
                    graded = grade_prop_export(previous_card, grade_date)
                    ledger = merge_prop_ledgers(prior_ledger, graded)
                st.session_state["generated_prop_results_log"] = ledger
                st.session_state["active_prop_results_log"] = ledger
                summary = grading_summary(graded)
                st.sidebar.success(
                    f"Graded {summary['graded']} props: "
                    f"{summary['wins']}-{summary['losses']} "
                    f"({summary['voids']} void/DNP, "
                    f"{summary['unresolved']} unresolved)."
                )
            except Exception as exc:
                st.sidebar.error(f"Player-prop grading failed: {exc}")

    generated_ledger = st.session_state.get("generated_prop_results_log")
    active_ledger = _assemble_active_prop_ledger(
        prop_results_log,
        generated_ledger,
        bundled_ledger,
    )
    st.session_state["active_prop_results_log"] = active_ledger

    uploaded_history_present = bool(prop_results_log)
    generated_history_present = (
        isinstance(generated_ledger, pd.DataFrame) and not generated_ledger.empty
    )
    recovery_saved = True
    if (
        isinstance(active_ledger, pd.DataFrame)
        and not active_ledger.empty
        and (uploaded_history_present or generated_history_present)
    ):
        from app_core.prop_calibration import persist_prop_results_log

        recovery_saved = persist_prop_results_log(active_ledger)

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
        if uploaded_history_present or generated_history_present:
            if recovery_saved:
                st.sidebar.caption(
                    "Newest prop history is saved for automatic restart recovery."
                )
            else:
                st.sidebar.warning(
                    "The newest prop history could not be saved for restart "
                    "recovery. Download the updated ledger before leaving this session."
                )
        if coverage["settled"] <= 0:
            st.sidebar.error(
                "The uploaded prop ledger has no settled WIN/LOSS rows. "
                "Player props will remain research-only."
            )
        elif coverage["date_count"] <= 1:
            st.sidebar.warning(
                "Only one slate date is loaded. This is not yet a cumulative "
                "calibration history; keep the downloaded ledger as a portable backup."
            )
    else:
        st.sidebar.error(
            "No cumulative graded prop ledger is loaded. Player props will be "
            "research-only and cannot receive production stakes."
        )

    if isinstance(active_ledger, pd.DataFrame) and not active_ledger.empty:
        st.sidebar.download_button(
            "Download Updated Graded Prop Ledger",
            active_ledger.to_csv(index=False, encoding="utf-8-sig"),
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
        "prop_results_log": active_ledger,
        "run_analysis_counter": run_counter,
    }
