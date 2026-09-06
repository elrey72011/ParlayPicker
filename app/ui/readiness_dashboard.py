"""Current and saved run diagnostics, without changing decisions."""
import json

import pandas as pd
import streamlit as st

from core.run_readiness import build_readiness, game_table, render_readiness


def render_readiness_dashboard(audit=None, final=None, diagnostics=None):
    with st.expander("Run Readiness Report", expanded=False):
        st.caption("Evidence readiness and wager approval are separate. This report does not change picks or thresholds.")
        source = st.selectbox("Readiness source", ["Current run", "Saved snapshot"], key="readiness_source")
        if source == "Saved snapshot":
            if st.button("Load saved snapshots", key="readiness_load"):
                from app_core.prediction_evidence import load_snapshots
                try:
                    st.session_state["readiness_snapshots"] = load_snapshots()
                except Exception:
                    st.error("Saved evidence could not be loaded. Check Prediction Evidence Status.")
                    st.session_state.pop("readiness_snapshots", None)
            saved = st.session_state.get("readiness_snapshots", [])
            if not saved:
                st.info("Load saved snapshots to inspect an earlier run.")
                return
            by_id = {sid: (a, f) for sid, a, f in saved}
            sid = st.selectbox("Snapshot", list(reversed(by_id)), key="readiness_snapshot")
            audit, final = by_id[sid]
            diagnostics = None  # Current run warnings must not describe an older run.
        if audit is None or audit.empty:
            st.info("No candidate evidence is available for this run. Run Master Analysis or select a saved snapshot.")
            return
        report = build_readiness(audit, final, diagnostics=diagnostics)
        counts = report["counts"]
        st.write(f"Games: {counts['games']} · Evidence ready for grading: {counts['ready_for_grading']} · Approved wagers: {counts['approved_wagers']}")
        st.caption("Quote age warning: 15 minutes at capture, for diagnostics only. Feature freshness is unavailable without a source timestamp.")
        table = game_table(report)
        visible = ["league", "matchup", "selected_pick", "readiness", "wager_decision", "displayed_probability",
                   "production_probability", "independent_model_probability", "verified_quote_candidates", "candidate_count",
                   "evidence_blockers", "data_warnings", "wager_reasons"]
        st.dataframe(table[visible], hide_index=True)
        for warning in report["run_warnings"]:
            st.warning(warning)
        st.caption("Quote verified means the provider quote matched. Line eligible separately reflects final line rejection. Push-capable lines require verified probability semantics for validation.")
        candidates = pd.DataFrame(report["candidates"])
        candidates["issues"] = candidates["issues"].map(lambda values: "; ".join(values))
        st.dataframe(candidates[["matchup_id", "pick", "selected", "quote_verified", "line_eligible", "quoted_line", "settlement_rule", "probability_semantics", "quote_age_minutes_at_capture", "odds_source", "issues"]], hide_index=True)
        st.download_button("Download Readiness Report", render_readiness(report), file_name="run-readiness.md", mime="text/markdown")
        st.download_button("Download Readiness Metrics", json.dumps(report, indent=2, allow_nan=False), file_name="run-readiness.json", mime="application/json")
        st.download_button("Download Game Readiness CSV", table.to_csv(index=False), file_name="game-readiness.csv", mime="text/csv")
        st.download_button("Download Candidate Readiness CSV", candidates.to_csv(index=False), file_name="candidate-readiness.csv", mime="text/csv")
