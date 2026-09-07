"""On-demand historical CFBD access diagnostics."""
import json
import os
import streamlit as st
from app_core.ncaaf_data_access import probe_cfbd_access


def _token():
    for name in ("CFBD_API_KEY", "CFBDAPIKEY", "cfbd_api_key"):
        try:
            value = st.secrets.get(name)
        except (FileNotFoundError, KeyError):
            value = None
        value = value or os.environ.get(name)
        if value:
            return value
    return None


def render_ncaaf_data_access():
    with st.expander("NCAAF Data Access"):
        st.caption("Check historical CFBD access using up to 6 requests. This samples data availability; it does not train or approve a model.")
        if st.button("Check NCAAF Data Access", key="check_ncaaf_data_access"):
            with st.spinner("Checking three historical seasons…"):
                st.session_state["ncaaf_data_access_report"] = probe_cfbd_access(_token())
        report = st.session_state.get("ncaaf_data_access_report")
        if report:
            if report["status"] == "samples_accessible":
                st.success("Historical samples accessible. Download the report for the training-data review.")
            elif report["status"] == "missing_or_invalid_key":
                st.warning("Configure CFBD_API_KEY or cfbd_api_key in Streamlit secrets, then try again.")
            else:
                st.warning("Access check incomplete. Download the report to identify the affected season or request.")
            st.caption(report["scope"])
            if report["seasons"]:
                st.dataframe(report["seasons"], hide_index=True)
            st.download_button("Download NCAAF Data Access Report",
                               json.dumps(report, indent=2), "ncaaf-data-access.json",
                               mime="application/json", key="download_ncaaf_data_access")
