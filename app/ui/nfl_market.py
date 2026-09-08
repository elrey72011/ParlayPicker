"""Read/restore scheduled NFL market evidence without manual paid requests."""
import json
import streamlit as st
from app_core import nfl_market as nfl, nfl_market_store as store


def render_nfl_market():
    with st.expander("NFL Market Tracking"):
        st.caption("Research only: pregame market quotes and final-score comparisons. No independent NFL model or wager approval.")
        if st.button("Restore / back up NFL evidence", key="nfl_market_sync"):
            try:
                status = store.sync()
                st.success(f"Verified {status['records_verified']} NFL records in Drive.")
            except Exception:
                st.error("NFL storage verification failed. Check Drive configuration; local evidence remains available.")
        report = nfl.report()
        st.caption(f"Captured games: {report['captured_games']} · Graded games: {report['graded_games']}")
        if report["unresolved_past_score_window"]:
            st.warning("Some captured games remain unresolved beyond the provider's three-day score window. Review the report.")
        if not report["captured_games"]:
            st.info("Enable NFL in the GitHub research scheduler, then restore evidence here after an eligible pregame run.")
        st.download_button("Download NFL market report", json.dumps(report, indent=2), "nfl-market-report.json", mime="application/json", key="nfl_market_report")
        st.download_button("Download NFL market records", json.dumps(store.records(), indent=2), "nfl-market-records.json", mime="application/json", key="nfl_market_records")
