"""Manual prospective research controls; no background calls or approved stakes."""
import os
import json
import streamlit as st
from app_core import ncaaf_prospective as prospective
from app_core import ncaaf_prospective_store as store
from app.ui.ncaaf_data_access import _token


def _odds_key():
    try:
        return st.secrets.get("ODDS_API_KEY") or os.environ.get("ODDS_API_KEY")
    except (FileNotFoundError, KeyError):
        return os.environ.get("ODDS_API_KEY")


def render_ncaaf_prospective():
    with st.expander("NCAAF Prospective Evaluation"):
        st.caption("Paper evaluation only. Freeze once, refresh current inputs, capture before kickoff, then grade after games finish. These actions are manual.")
        if st.button("Restore / back up prospective evidence", key="ncaaf_prospective_sync"):
            try:
                with st.spinner("Restoring and verifying prospective records…"):
                    result = store.sync()
                st.success(f"Verified {result['records_verified']} records in Drive.")
            except Exception:
                st.error("Remote verification failed. Local records remain available; download them and check Drive configuration.")
        result = st.session_state.get("ncaaf_research_result")
        if st.button("Freeze research models for prospective capture", key="ncaaf_prospective_freeze", disabled=result is None):
            try:
                prospective.freeze(result)
                st.success("Research models frozen locally. Back up prospective evidence to Drive.")
            except ValueError:
                st.error("Research artifact validation failed. Run the fixed research evaluation first.")
        all_records = store.records()
        models = [r for r in all_records if r["kind"] == "model"]
        active = models[-1] if models else None
        if active:
            st.caption("Active frozen cohort: " + active["id"][:12])
        else:
            st.info("Restore a frozen model from Drive, or load the historical checkpoint, run the research evaluation, and freeze its models here.")
        if st.button("Refresh current NCAAF inputs", key="ncaaf_prospective_refresh"):
            with st.spinner("Fetching up to six CFBD requests…"):
                state, status = prospective.refresh(None, _token())
            st.session_state["ncaaf_current_inputs"] = state
            st.session_state["ncaaf_current_status"] = status
        state = st.session_state.get("ncaaf_current_inputs")
        if state and prospective.pending(state):
            if st.button("Continue current NCAAF inputs", key="ncaaf_prospective_continue"):
                with st.spinner("Fetching up to six CFBD requests…"):
                    state, status = prospective.refresh(state, _token())
                st.session_state["ncaaf_current_inputs"] = state
                st.session_state["ncaaf_current_status"] = status
        status = st.session_state.get("ncaaf_current_status")
        if status:
            st.caption("Current inputs: " + status)
        st.caption("Capture uses one odds request for three markets in the US region. Input age limit: 24 hours; quote age limit: 15 minutes. Three prior games per team are required, so early-season captures may be empty.")
        if st.button("Capture prospective NCAAF predictions", key="ncaaf_prospective_capture", disabled=not active or not state or bool(prospective.pending(state))):
            try:
                with st.spinner("Fetching and freezing pregame quotes…"):
                    _, info = prospective.capture(state, active, _odds_key())
                st.success(f"Saved {info['saved_games']} games; skipped {info['skipped_games']}. Back up prospective evidence to Drive.")
            except ValueError as exc:
                st.error(str(exc))
        if st.button("Grade pending NCAAF predictions", key="ncaaf_prospective_grade"):
            with st.spinner("Checking up to six games…"):
                info = prospective.grade(_token())
            st.info(f"Graded {info['graded']} games; {info['pending_before_request']} were pending. Repeat if needed.")
            if info["error"]:
                st.warning(info["error"])
        report = prospective.report()
        st.caption(f"Captured games across cohorts: {report['captured_games_by_cohort']} · Graded model selections: {report['graded_selections']}")
        if report["summary"]:
            st.dataframe([{k:v for k,v in row.items() if k != "calibration"} for row in report["summary"]], hide_index=True)
        st.download_button("Download prospective evaluation", json.dumps(report, indent=2), "ncaaf-prospective-report.json", mime="application/json", key="ncaaf_prospective_report")
        st.download_button("Download prospective records", json.dumps(store.records(), indent=2), "ncaaf-prospective-records.json", mime="application/json", key="ncaaf_prospective_records")
