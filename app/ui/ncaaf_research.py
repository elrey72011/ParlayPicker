"""Research-only evaluation controls, with no implicit retraining on rerun."""
import hashlib
import io
import json
import zipfile
import streamlit as st
from app_core.ncaaf_history import checkpoint_bytes, pending_requests
from app_core.ncaaf_research import run_research, markdown_report


def result_zip(result):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("report", "artifact", "predictions"):
            archive.writestr(name + ".json", json.dumps(result[name], indent=2, allow_nan=False))
        archive.writestr("report.md", markdown_report(result))
    return buffer.getvalue()


def render_ncaaf_research(state):
    st.markdown("**NCAAF research model**")
    st.caption("Fixed experiment: train 2023, calibrate 2024, evaluate 2025. Repeating this evaluation does not create a new holdout. No live picks are changed.")
    fingerprint = hashlib.sha256(checkpoint_bytes(state)).hexdigest()
    if st.button("Run fixed NCAAF research evaluation", key="ncaaf_research_run", disabled=bool(pending_requests(state))):
        try:
            with st.spinner("Fitting and evaluating the research models…"):
                result = run_research(state)
            st.session_state["ncaaf_research_result"] = result
        except ValueError as exc:
            st.error(str(exc))
    result = st.session_state.get("ncaaf_research_result")
    if result and result["report"]["checkpoint_hash"] == fingerprint:
        st.markdown(markdown_report(result))
        st.download_button("Download NCAAF research results", result_zip(result),
                           "ncaaf-research.zip", mime="application/zip", key="ncaaf_research_download")
