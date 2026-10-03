"""Schedule coverage presentation independent of the selected research card."""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st


def schedule_controls(panel, sports):
    if "NCAAF" not in sports:
        return None, None
    today = datetime.now(ZoneInfo("America/New_York")).date()
    panel.caption("NCAAF schedule dates use the Eastern calendar. Schedule inclusion does not authorize a wager.")
    first = panel.date_input("NCAAF schedule from", value=today, key="ncaaf_schedule_from")
    last = panel.date_input("NCAAF schedule through", value=first, min_value=first,
                            max_value=first + timedelta(days=30), key="ncaaf_schedule_through")
    return first.isoformat(), last.isoformat()


def render_inventory(diagnostics):
    report = (diagnostics or {}).get("ncaaf_coverage")
    if not isinstance(report, dict):
        return
    st.subheader("NCAAF scheduled games — FBS and FCS")
    counts = report["counts"]
    st.caption(" · ".join(f"{key.title()}: {counts[key]}" for key in ("scheduled", "matched", "quoted", "timestamped", "ranked", "qualified")))
    if report["inventory_status"] != "COMPLETE":
        st.warning("Partial schedule inventory. Full coverage is unverified: " + ", ".join(report["inventory_reasons"]))
    else:
        st.caption("Schedule inventory reconciled to both division event indexes. Quote coverage and qualification are separate.")
    frame = pd.DataFrame(report["rows"])
    if not frame.empty:
        st.dataframe(frame, hide_index=True)
    st.download_button("Export NCAAF schedule coverage", frame.to_csv(index=False),
                       file_name="ncaaf_schedule_coverage.csv", mime="text/csv", key="ncaaf_schedule_coverage_export")
    if report["unresolved"]:
        st.caption(f"{len(report['unresolved'])} provider/candidate/selection record(s) have unresolved or conflicting schedule identity.")
