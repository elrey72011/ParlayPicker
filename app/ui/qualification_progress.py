"""Qualification progress panel (roadmap C3): load a market-readiness report and show each market's gates."""
import json
import os

import pandas as pd
import streamlit as st

from core.qualification_progress import latest_report, rows, summary

REPO = "elrey72011/ParlayPicker"


def _token():
    try:
        value = st.secrets.get("PARLAYPICKER_GITHUB_READ_TOKEN")
    except Exception:  # no secrets file
        value = None
    return value or os.environ.get("PARLAYPICKER_GITHUB_READ_TOKEN", "")

COLUMNS = ("sport", "market", "stage", "training_games", "training_progress", "calibration_decided",
           "calibration_progress", "validation_settled", "validation_progress", "games_per_day",
           "window_days", "price_screen_picks", "price_screen_beat_close", "next_blocker")


def render_qualification_progress():
    with st.expander("Qualification progress", expanded=False):
        st.caption("Upload the latest market-readiness.json from the 'Market readiness (read-only)' workflow "
                   "(it runs daily). Descriptive only: progress never authorizes a wager.")
        token = _token()
        if token and st.button("Load latest daily report", key="qualification_progress_latest"):
            try:
                import requests
                report, created = latest_report(REPO, token, requests.get)
                st.session_state["qualification_progress"] = rows(report)
                st.session_state["qualification_progress_at"] = created
            except Exception as exc:
                st.error(f"The latest report could not be loaded ({type(exc).__name__}). Upload it instead.")
        elif not token:
            st.caption("To load the daily report automatically, add a read-only GitHub token as the "
                       "Streamlit secret PARLAYPICKER_GITHUB_READ_TOKEN.")
        upload = st.file_uploader("market-readiness.json", type=["json"], key="qualification_progress_upload")
        if upload is not None:
            try:
                report = json.loads(upload.getvalue().decode("utf-8"))
                st.session_state["qualification_progress"] = rows(report)
                st.session_state["qualification_progress_at"] = None
            except (ValueError, UnicodeDecodeError, AttributeError, TypeError):
                st.session_state.pop("qualification_progress", None)
                st.error("That file is not a market-readiness report.")
        _render_research_picks(token)
        progress = st.session_state.get("qualification_progress")
        if progress is None:
            return
        st.write(summary(progress, st.session_state.get("qualification_progress_at")))
        percent = {name: st.column_config.ProgressColumn(name.replace("_", " "), min_value=0.0, max_value=1.0,
                                                          format="percent")
                   for name in ("training_progress", "calibration_progress", "validation_progress")}
        st.dataframe(pd.DataFrame(progress).reindex(columns=COLUMNS), hide_index=True, column_config=percent)
        st.caption("Gates: 120 training games for the pinned model, 40 settled calibration predictions, "
                   "then 200 settled games in the frozen validation window. Price screen: picks at 2%+ EV "
                   "and the share that beat the certified close.")


def _render_research_picks(token):
    """Latest research picks board (NBA, NCAAB, NHL) from the 'Research picks' workflow."""
    if not token:
        return
    if st.button("Load latest research picks", key="research_picks_latest"):
        try:
            import requests
            report, created = latest_report(REPO, token, requests.get, artifact="research-picks",
                                            filename="research-picks.json")
            st.session_state["research_picks"] = (report, created)
        except Exception as exc:
            st.error(f"The research picks could not be loaded ({type(exc).__name__}).")
    loaded = st.session_state.get("research_picks")
    if not loaded:
        return
    report, created = loaded
    st.caption(f"Research picks ({created}). Best expected value at the best available price from the "
               "in-season score models. Research only: every pick is NO_BET with zero stake.")
    picks = [p for board in report.get("boards", []) for p in board.get("picks", [])]
    if picks:
        st.dataframe(pd.DataFrame(picks).reindex(columns=[
            "sport", "market_family", "scheduled_start", "away_team", "home_team", "selection", "line",
            "american_odds", "sportsbook", "win_probability", "ev", "consensus_probability"]), hide_index=True)
    blockers = {f"{b['sport']} {b['market_family']}": b["blocker"] for b in report.get("boards", []) if b.get("blocker")}
    if blockers:
        st.caption("Not ready: " + "; ".join(f"{k}: {v}" for k, v in blockers.items()))
    records = report.get("records") or {}
    if records:
        st.dataframe(pd.DataFrame([dict(market=k, **{x: v for x, v in r.items() if x != "positive_ev"})
                                   for k, r in records.items()]), hide_index=True)
