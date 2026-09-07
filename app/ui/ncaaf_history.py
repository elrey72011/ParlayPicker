"""Explicit, bounded collection controls; no requests on ordinary reruns."""
import json
import streamlit as st
from app_core.ncaaf_history import (new_collection, collect_batch, pending_requests,
    checkpoint_bytes, load_checkpoint, build_dataset, archive_bytes, backup_checkpoint)
from app.ui.ncaaf_data_access import _token


@st.cache_data(show_spinner=False, max_entries=2)
def _outputs(state):
    return build_dataset(state)[0], archive_bytes(state)


def render_ncaaf_history():
    with st.expander("NCAAF Historical Collection"):
        st.caption("Collect the three prior seasons in batches of up to 6 CFBD requests. Continue until requests are complete, then review coverage. Save a checkpoint before leaving this session.")
        uploaded = st.file_uploader("Resume from NCAAF checkpoint JSON", type=["json"], key="ncaaf_history_upload")
        if st.button("Load NCAAF checkpoint", disabled=uploaded is None, key="ncaaf_history_load"):
            try:
                st.session_state["ncaaf_history"] = load_checkpoint(uploaded.getvalue())
                st.session_state.pop("ncaaf_history_status", None)
            except ValueError:
                st.error("Invalid NCAAF collection checkpoint. Use checkpoint.json from the collection download or the history JSON from Drive.")
        state = st.session_state.get("ncaaf_history")
        if st.button("Collect next NCAAF batch", key="ncaaf_history_collect",
                     disabled=state is not None and not pending_requests(state)):
            with st.spinner("Collecting up to six historical requestsâ€¦"):
                state, status = collect_batch(state if state is not None else new_collection(), _token())
                st.session_state["ncaaf_history"] = state
                st.session_state["ncaaf_history_status"] = status
        state = st.session_state.get("ncaaf_history")
        if state is None:
            return
        status = st.session_state.get("ncaaf_history_status")
        if status and status not in ("batch_saved", "requests_complete"):
            st.warning(f"Collection stopped: {status}. Completed requests are retained. Correct access or wait for quota reset before continuing.")
        audit, archive = _outputs(state)
        remaining = audit["requests_remaining"]
        st.caption(f"Requests saved: {audit['requests_finished']} Â· Pending: {remaining}. Weekly requests are discovered as season schedules arrive.")
        if not remaining:
            st.success("Planned requests complete. Review the audit for missing records before training.")
        st.dataframe(audit["seasons"], hide_index=True)
        st.caption("Features use only same-season games more than 7 days earlier. Original publication timestamps are unverified; this is research data and does not enable wagers.")
        st.download_button("Download NCAAF collection ZIP", archive, "ncaaf-history.zip", mime="application/zip", key="ncaaf_history_zip")
        st.download_button("Download NCAAF coverage audit", json.dumps(audit, indent=2), "ncaaf-history-audit.json", mime="application/json", key="ncaaf_history_audit")
        st.download_button("Download NCAAF checkpoint", checkpoint_bytes(state), "ncaaf-history-checkpoint.json", mime="application/json", key="ncaaf_history_checkpoint")
        if st.button("Back up NCAAF checkpoint to Drive", key="ncaaf_history_backup"):
            try:
                with st.spinner("Saving and verifying the historical checkpointâ€¦"):
                    backup_checkpoint(state)
                st.success("NCAAF checkpoint saved to Drive and read-back verified. Download that history JSON from Drive to resume after a restart.")
            except Exception:
                st.error("NCAAF backup could not be verified. Download the checkpoint now and check the existing Drive configuration.")

        from app.ui.ncaaf_research import render_ncaaf_research

        render_ncaaf_research(state)
