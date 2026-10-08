"""Owner-only explicit existing-packet selection. No analysis or storage action."""
import streamlit as st
from app_core import ncaaf_pipeline_evidence as evidence


def render():
    # publish_panel invokes this only after the constant-time owner token gate.
    with st.expander("Private NCAAF research inputs", expanded=False):
        st.caption("Select an existing native v1 or compatible recovered-model prospective packet for a future explicitly requested analysis. Staging inspects bounded JSON and hashes only; it does not verify features or run analysis. The compatible caller separately verifies original dependency bytes, event mapping, derived features and accepted public-output rights before computation. Stale inputs and unaccepted reviews remain unavailable. No fitting or collection occurs here.")
        upload = st.file_uploader("Existing private NCAAF target packet", type=["json"], key="ncaaf_native_target_packet")
        if st.button("Stage existing NCAAF packet", disabled=upload is None, key="ncaaf_native_stage"):
            try:
                packet = evidence.load(upload.getvalue(), owner_upload=True)
            except (ValueError, TypeError, KeyError):
                st.error("NCAAF packet could not be staged: invalid original JSON, hash, size or evidence label.")
            else:
                staged = st.session_state.setdefault("ncaaf_native_packets", {})
                if packet["sha256"] in staged or len(staged) < evidence.MAX_PACKETS:
                    staged[packet["sha256"]] = packet
                    st.success("Existing packet staged privately. Source review and display remain separately gated.")
                else:
                    st.error("At most four exact packets may be staged.")
        staged = st.session_state.get("ncaaf_native_packets", {})
        chosen = st.multiselect("NCAAF packets for the next analysis", list(staged),
            default=[k for k in st.session_state.get("ncaaf_native_selected", ()) if k in staged], key="ncaaf_native_selected")
        packets = [staged[k] for k in chosen]
        try:
            with evidence.selected(packets):
                pass
        except (ValueError, TypeError, KeyError):
            packets = []
            st.error("The selected packets exceed the bounded size or count limit.")
        st.session_state["ncaaf_research_packets"] = packets
        st.caption("Half-point spreads and totals have separate targets. Integer push mass remains unvalidated. Historical inference clocks are never replaced by capture-finish clocks. Research supplies no wager approval.")
