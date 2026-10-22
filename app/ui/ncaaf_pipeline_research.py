"""Owner-only explicit existing-packet selection. No analysis or storage action."""
import streamlit as st
from app_core import ncaaf_pipeline_evidence as evidence
from app_core import ncaaf_owner_research as owner
from app_core import ncaaf_owner_terms_research as owner_terms
from app_core import ncaaf_owner_staged_research as owner_staged


def render_refresh_controls():
    """Private selection before Refresh; independent of customer publishing."""
    from app_core import ncaaf_private_refresh as private
    with st.expander('Private NCAAF model prediction', expanded=False):
        enabled=st.checkbox('Use reviewed private NCAAF inputs for Refresh',
                            key='ncaaf_private_refresh')
        st.caption('Conditional research only. Select a completed target/history/feature and mathematical-market review, plus separately configured inference authorization. Staging runs no model and fetches nothing. Missing evidence remains UNAVAILABLE.')
        upload=st.file_uploader('Reviewed private Refresh selection',type=['json'],key='ncaaf_private_refresh_upload')
        if st.button('Stage reviewed Refresh selection',disabled=upload is None,key='ncaaf_private_refresh_stage'):
            try:
                selection=private.load(upload.getvalue())
            except (ValueError,TypeError,KeyError):
                st.session_state.pop('ncaaf_private_refresh_selection',None)
                st.error('UNAVAILABLE: the private selection is incomplete, altered or incompatible.')
            else:
                st.session_state['ncaaf_private_refresh_selection']=selection
                st.success('Staged. Only a later Refresh action can use separately authorized inference.')
        if enabled and not st.session_state.get('ncaaf_private_refresh_selection'):
            st.info('UNAVAILABLE: no reviewed private Refresh selection is staged.')


def render_refresh_result():
    rows=st.session_state.get('ncaaf_private_refresh_ranked',[])
    if not rows:return
    st.subheader('Private NCAAF model estimate')
    st.caption('Ranked by unconditional model cover probability. Price and estimated EV are separate; a higher cover estimate does not establish greater value or profitability. Calibration is unvalidated. Conditional market assumptions are not verified offer admission. PASS — zero stake.')
    event=rows[0]['event']
    st.caption(f"{event['away_team']} at {event['home_team']} | kickoff {event['start_utc']} | input label: {rows[0]['evidence_label']}")
    visible=[dict(Team=r['team'],Side=r['side'],Line=r['line'],Price=r['price'],
        **{'Unconditional cover':r['cover'],'Push':r['push'],'Loss':r['loss'],
           'Cover given no push':r['conditional_on_no_push']['value'],
           'Estimated EV / unit':r['estimated_ev_per_unit'],'Comparison':r['selection']}) for r in rows]
    st.dataframe(visible,hide_index=True)
    st.caption('Original inputs, native model identity, computations and static replay are retained in private custody. Receipt or replay does not refresh the original observation clock.')


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
        if not private and any(p["payload"]["version"] in {owner.VERSION,owner_terms.VERSION,owner_staged.VERSION} for p in packets):
            packets = []
            st.warning("Owner-reviewed packets require explicit private mode; no inference is selected.")
        try:
            with evidence.selected(packets, private_research=private):
                pass
        except (ValueError, TypeError, KeyError):
            packets = []
            st.error("The selected packets exceed the bounded size or count limit.")
        st.session_state["ncaaf_research_packets"] = packets
        st.caption("Historical half-point targets stay separate. The explicitly versioned private spread route retains unconditional cover/push/loss; comparison requires one reviewed observed market and two attributable results. Uploading does not authorize inference. Historical clocks are never replaced by capture clocks. Research supplies no wager approval.")

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
                            spread=exported.get('ncaaf_spread_private_display')
                            if spread:
                                shown=json.loads(spread)
                                st.metric('Push probability',f"{shown['outcome_probabilities']['push']:.2%}")
                                st.metric('Loss probability',f"{shown['outcome_probabilities']['loss']:.2%}")
                                conditional=shown['conditional_no_push']
                                if conditional['state']=='VALUE':
                                    st.metric('Cover conditional on no push',f"{conditional['value']:.2%}")
                                else:st.caption('Cover conditional on no push: UNKNOWN (zero denominator).')
                                comparison=shown.get('comparison')
                                if comparison:
                                    st.metric('Home unconditional cover',f"{comparison['home_cover']:.2%}")
                                    st.metric('Away unconditional cover',f"{comparison['away_cover']:.2%}")
                                    st.caption('Higher model cover estimate: '+comparison['higher_cover']+'. This is not EV or a validated recommendation.')
                        else:
                            st.warning(result["availability_reason"])
                        st.caption("Uncalibrated research only. PASS - zero stake. No scientific qualification or wagering authority.")
