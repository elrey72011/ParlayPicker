"""Qualification progress panel (roadmap C3): load a market-readiness report and show each market's gates."""
import json

import pandas as pd
import streamlit as st

from core.qualification_progress import rows, summary

COLUMNS = ("sport", "market", "stage", "training_games", "training_progress", "calibration_decided",
           "calibration_progress", "validation_settled", "validation_progress", "games_per_day",
           "window_days", "price_screen_picks", "price_screen_beat_close", "next_blocker")


def render_qualification_progress():
    with st.expander("Qualification progress", expanded=False):
        st.caption("Upload the latest market-readiness.json from the 'Market readiness (read-only)' workflow "
                   "(it runs daily). Descriptive only: progress never authorizes a wager.")
        upload = st.file_uploader("market-readiness.json", type=["json"], key="qualification_progress_upload")
        if upload is not None:
            try:
                report = json.loads(upload.getvalue().decode("utf-8"))
                st.session_state["qualification_progress"] = rows(report)
            except (ValueError, UnicodeDecodeError, AttributeError, TypeError):
                st.session_state.pop("qualification_progress", None)
                st.error("That file is not a market-readiness report.")
        progress = st.session_state.get("qualification_progress")
        if progress is None:
            return
        st.write(summary(progress))
        percent = {name: st.column_config.ProgressColumn(name.replace("_", " "), min_value=0.0, max_value=1.0,
                                                          format="percent")
                   for name in ("training_progress", "calibration_progress", "validation_progress")}
        st.dataframe(pd.DataFrame(progress).reindex(columns=COLUMNS), hide_index=True, column_config=percent)
        st.caption("Gates: 120 training games for the pinned model, 40 settled calibration predictions, "
                   "then 200 settled games in the frozen validation window. Price screen: picks at 2%+ EV "
                   "and the share that beat the certified close.")
