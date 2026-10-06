"""Owner-only local source intake; no acceptance or collection action."""
import sqlite3
import streamlit as st
from app_core import source_evidence_intake as intake


def render():
    # Called only after publish_panel's constant-time owner token check.
    with st.expander("Private exact-offer source evidence",expanded=False):
        st.caption("Retain an existing packet locally for future runs. Uploading does not accept its source review, register a listing or authorize collection or wagers. Configure private evidence storage outside the repository.")
        upload=st.file_uploader("Existing exact-offer packet",type=["json"],key="private_source_packet")
        if st.button("Retain private source packet",key="retain_private_source_packet",disabled=upload is None):
            try:
                receipt=intake.intake(upload.getvalue())
            except (OSError,sqlite3.Error,ValueError,TypeError,KeyError) as exc:
                st.error("Source packet was not retained: "+str(exc))
            else:
                staged=st.session_state.setdefault("private_source_intakes",{})
                staged[receipt["reference"]]=receipt
                st.write(receipt)
        staged=st.session_state.get("private_source_intakes",{})
        refs=st.multiselect("Source packets for the next analysis",list(staged),default=[r for r in st.session_state.get("source_evidence_refs",[]) if r in staged],key="selected_private_source_intakes")
        st.session_state["source_evidence_refs"]=list(refs)
        st.caption("Selection affects the next explicitly requested analysis only. Saved runs, original inference and UI-refresh clocks are never rewritten.")
        for reference in refs:
            try: raw=intake.download(reference)
            except (OSError,sqlite3.Error,ValueError,TypeError,KeyError) as exc:
                st.warning("Private source packet download unavailable: "+str(exc))
            else:
                st.download_button("Download original private source packet",raw,"private-source-evidence-"+reference.split(":")[-1]+".json","application/json",key="source_packet_"+reference,on_click="ignore")
