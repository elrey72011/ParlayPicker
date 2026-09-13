"""On-demand accuracy diagnostics; no network calls or live setting changes."""
from datetime import date
import json
import pandas as pd
import streamlit as st
from app_core.pick_accuracy import build_accuracy_report, render_accuracy_markdown


def render_pick_accuracy(ledger):
    if not st.checkbox("Show pick accuracy comparison", key="show_pick_accuracy"):
        return
    start = st.date_input("Evaluate games from", value=date(2026, 9, 11), key="accuracy_evaluation_start",
                          help="Only later slates with declared training completed before this date qualify.")
    st.caption("Compares saved ranking, probability-first selection, and sportsbook baseline on the same games. Uses saved results only; it does not refresh picks, call APIs, or change weights.")
    try:
        report = build_accuracy_report(ledger, evaluation_start=start.isoformat())
    except ValueError as exc:
        st.info("The ledger cannot support this report yet: "+str(exc))
        return
    inv = report["validation"]["inventory"]
    st.write(f"Verified comparison: {inv['eligible_events']} games")
    if not inv['eligible_events']:
        st.info("Not enough verified evidence. Review exclusions below; save pregame candidate snapshots and grade them after games finish. Changing a timestamp on an old export cannot establish pregame evidence.")
    else:
        st.dataframe(pd.DataFrame([{"Ranking": name, "Games": r['games'], "Wins": r['wins'], "Losses": r['losses'], "Pushes": r['pushes'], "Win rate": r['hit_rate'], "Simulated ROI": r['flat_roi'], "Brier": r['brier'], "Log loss": r['log_loss']} for name, r in report['rankings'].items()]), hide_index=True, width="stretch")
        st.markdown("**Sources on identical original selected tickets**")
        st.dataframe(pd.DataFrame([{"League": r['league'], "Market": r['market'], "Source": r['source'], "Available games": r['available_games'], "Missing or unverified": r['missing_or_unverified_games'], "Decisions": r['source_metrics']['n'], "Source Brier": r['source_metrics']['brier'], "Baseline Brier": r['sportsbook_on_same_tickets']['brier'], "Source log loss": r['source_metrics']['log_loss'], "Baseline log loss": r['sportsbook_on_same_tickets']['log_loss']} for r in report['sources']]), hide_index=True, width="stretch")
        st.caption("Lower Brier and log loss are better. Each source can cover different games; its baseline uses the same tickets. Gemini remains a qualitative review comparison below.")
    with st.expander("Accuracy report exclusions and limits"):
        st.json(report['validation']['exclusions'])
        st.write("Historical comparison only. Source-alone accuracy does not prove that adding a source improves the blend. Original data declarations are checked but not independently attested. Live ranking and weights remain unchanged.")
    st.download_button("Download pick accuracy report", render_accuracy_markdown(report), "pick-accuracy.md", "text/markdown", key="download_pick_accuracy_md")
    st.download_button("Download detailed accuracy data", json.dumps(report, indent=2, allow_nan=False), "pick-accuracy.json", "application/json", key="download_pick_accuracy_json")
