"""Explicitly exploratory threshold report over saved, verified development slates."""
import json

import pandas as pd
import streamlit as st


def render_threshold_dashboard():
    with st.expander("Development Threshold Comparison", expanded=False):
        st.caption("Compare accuracy, selection volume and returns on development data. This does not change live picks or demonstrate a future 75% win rate.")
        from app_core.prediction_evidence import materialize
        from core.threshold_validation import compare_thresholds, render_threshold_report

        if st.button("Load saved evidence for comparison", key="load_threshold_evidence"):
            try:
                st.session_state["threshold_evidence"] = materialize()
                st.session_state.pop("threshold_comparison_report", None)
            except Exception as exc:
                st.error(f"Evidence could not be loaded: {exc}")
        evidence = st.session_state.get("threshold_evidence")
        if evidence is None:
            return
        audit, final = evidence
        if audit.empty:
            st.info("No snapshots are saved yet. Run Master Analysis first.")
            return
        starts = pd.to_datetime(audit["game_start_utc"], errors="coerce", utc=True).dropna()
        if starts.empty:
            st.info("Saved snapshots have no usable game start times.")
            return
        first = starts.min().tz_convert("America/New_York").date()
        last = min(starts.max().tz_convert("America/New_York").date(), pd.Timestamp.now(tz="America/New_York").date())
        training_end = st.date_input("Last training/development cutoff before the comparison window", value=first - pd.Timedelta(days=1), key="threshold_training_end")
        development_end = st.date_input("Last slate allowed in this development comparison", value=last, key="threshold_development_end")
        scope_label = st.selectbox("Selection scope", ["Approved wagers", "All coverage picks"], key="threshold_scope")
        if st.button("Generate development comparison", key="generate_threshold_comparison"):
            try:
                st.session_state["threshold_comparison_report"] = compare_thresholds(
                    audit, train_through=str(training_end), development_through=str(development_end),
                    selections=final, scope="qualified_wagers" if scope_label == "Approved wagers" else "all_selected")
            except ValueError as exc:
                st.error(str(exc))
                st.session_state.pop("threshold_comparison_report", None)
        report = st.session_state.get("threshold_comparison_report")
        if not report:
            return
        st.caption(f"Generated report: {report['scope']}, through {report['development_through']} · {report['status']}")
        rows = [{"League": r["league"], "Market": r["market"], "Threshold": r["threshold"],
                 "Picks": r["selector"]["games"], "Slates": r["selector"]["slates"],
                 "Coverage": r["selector"]["coverage"], "Hit rate": r["selector"]["hit_rate"],
                 "Flat ROI": r["selector"]["flat_roi"], "Market ROI": r["market_only_same_games"]["flat_roi"]}
                for r in report["rows"]]
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True)
        else:
            st.info("No complete verified candidate pools qualify. Download the report for exclusion reasons.")
        st.download_button("Download Threshold Comparison", render_threshold_report(report),
                           file_name="threshold-comparison.md", mime="text/markdown")
        st.download_button("Download Threshold Metrics", json.dumps(report, indent=2, allow_nan=False),
                           file_name="threshold-comparison.json", mime="application/json")
