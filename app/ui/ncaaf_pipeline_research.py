"""Owner-only explicit existing-packet selection. No analysis or storage action."""
import streamlit as st
from app_core import ncaaf_pipeline_evidence as evidence
from app_core import ncaaf_owner_research as owner


def render(games=None, candidates=None):
    # publish_panel invokes this only after the constant-time owner token gate.
    if games is None and candidates is None:
        games, candidates = st.session_state.get('ncaaf_private_captured_views', (None, None))
    with st.expander("Private NCAAF research inputs", expanded=False):
        st.caption("Select an existing native v1 or compatible recovered-model prospective packet for a future explicitly requested analysis. Staging inspects bounded JSON and hashes only; it does not verify features or run analysis. The caller separately verifies original dependency bytes, event mapping, derived features and the selected route's applicable output rights before computation. Missing reviews and stale inputs remain unavailable. No fitting or collection occurs here.")
        private = st.checkbox("Use OWNER_REVIEWED / PRIVATE_RESEARCH mode", value=False, key="ncaaf_private_research")
        st.caption("Private mode requires an already completed Robert Velarde review of exact evidence. Uploading or selecting never performs that review or creates independent acceptance. Its model probability is shown only here; no public-output permission is inferred.")
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
        if not private and any(p["payload"]["version"] == owner.VERSION for p in packets):
            packets = []
            st.warning("Owner-reviewed packets require explicit private mode; no inference is selected.")
        try:
            with evidence.selected(packets, private_research=private):
                pass
        except (ValueError, TypeError, KeyError):
            packets = []
            st.error("The selected packets exceed the bounded size or count limit.")
        st.session_state["ncaaf_research_packets"] = packets
        st.caption("Half-point spreads and totals have separate targets. Integer push mass remains unvalidated. Historical inference clocks are never replaced by capture-finish clocks. Research supplies no wager approval.")

        if private:
            import json
            if games is not None and candidates is not None:
                # Use the same captured/export identities supplied to preview.
                # The view performs static read-back; it never runs inference.
                from app_core.per_game_boards import per_game_board
                private_ids = {row.get("candidate_id") for row in candidates.to_dict("records")
                               if owner.is_private_source(row) and row.get("candidate_id")}
                for family in ("sides", "totals"):
                    board = per_game_board(games, candidates, family=family, novig_only=True, private_research=True, college_fallback=True)
                    for exported in board.to_dict("records"):
                        if exported.get("candidate_id") not in private_ids: continue
                        result = json.loads(exported["research_display"])
                        st.write(exported.get("matchup"), exported.get("pick"))
                        st.caption(result["basis"])
                        if result["availability_reason"] == "AVAILABLE":
                            st.metric("OWNER_REVIEWED / PRIVATE_RESEARCH probability", f"{result['probability']:.2%}")
                        else:
                            st.warning(result["availability_reason"])
                        st.caption("Uncalibrated research only. PASS - zero stake. No scientific qualification or wagering authority.")
