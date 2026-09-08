"""Manual paired MLB prospective research controls."""
import json
import streamlit as st
from app_core import mlb_prospective as p
from app_core import mlb_prospective_store as store


def render_mlb_prospective():
    with st.expander("MLB Prospective Evaluation"):
        st.caption("Research only. Compare frozen score forecasts using provider-listed probable pitchers. No wagering approval.")
        if st.button("Restore / back up MLB evidence", key="mlb_pro_sync"):
            try:
                result=store.sync()
                st.success(f"Verified {result['records_verified']} MLB records in Drive.")
            except Exception:
                st.error("MLB remote verification failed. Download local records and check storage configuration.")
        history=st.file_uploader("MLB history checkpoint JSON",type=["json"],key="mlb_pro_history")
        pitchers=st.file_uploader("MLB pitcher checkpoint JSON",type=["json"],key="mlb_pro_pitchers")
        if st.button("Freeze MLB comparison models",disabled=history is None or pitchers is None,key="mlb_pro_freeze"):
            try:
                key=p.freeze(history.getvalue(),pitchers.getvalue())
                st.success("Frozen cohort: "+key[:12]+". Back up MLB evidence to Drive.")
            except Exception:
                st.error("Freeze failed. Use the matching completed history and pitcher checkpoints.")
        models=[r for r in store.records() if r["kind"]=="model"]
        if models:st.caption("Active MLB cohort: "+models[-1]["id"][:12])
        if st.button("Refresh upcoming MLB games",key="mlb_pro_refresh"):
            try:st.session_state["mlb_pro_games"]=p.upcoming()
            except Exception:st.error("MLB schedule request failed. Retry refresh.")
        games=st.session_state.get("mlb_pro_games",[])
        if games:
            game=st.selectbox("Game to capture",games,format_func=lambda g:g["teams"]["away"]["team"]["name"]+" at "+g["teams"]["home"]["team"]["name"]+" — "+g["gameDate"],key="mlb_pro_game")
            if st.button("Capture paired MLB predictions",disabled=not models,key="mlb_pro_capture"):
                try:
                    p.capture(game["gamePk"])
                    st.success("Pregame evidence saved. Back up MLB evidence to Drive.")
                except ValueError as exc:st.warning(str(exc))
                except Exception:st.error("MLB capture failed; no forecast approved. Refresh and retry.")
        if st.button("Grade pending MLB predictions",key="mlb_pro_grade"):
            try:st.success(f"Saved {p.grade()} completed game results.")
            except Exception:st.error("MLB grading failed. Previously saved evidence remains available.")
        report=p.report()
        st.caption(f"Captured games across cohorts: {report['captured_games_by_cohort']} · Graded: {report['graded_games_by_cohort']}")
        st.download_button("Download MLB prospective report",json.dumps(report,indent=2),"mlb-prospective-report.json",key="mlb_pro_report")
        st.download_button("Download MLB prospective records",json.dumps(store.records(),indent=2),"mlb-prospective-records.json",key="mlb_pro_records")
