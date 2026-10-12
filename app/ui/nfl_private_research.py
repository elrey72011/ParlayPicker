"""Token-gated staging/browser and explicit retained-input scenario action."""
import streamlit as st
from app_core import nfl_private_routes as evidence


def render(candidates=None):
    with st.expander('Private NFL owner-reviewed research', expanded=False):
        enabled = st.checkbox('Use explicitly selected OWNER_REVIEWED NFL inputs', key='nfl_private_enabled')
        st.caption('PRIVATE_RESEARCH only. Staging checks original bytes; it creates no review or acceptance. A separately requested analysis must verify the exact event, offer, listing rules, permissions, availability and owner findings before computation. No collection, source registration, calibration or wagering authority.')
        upload = st.file_uploader('Existing owner-reviewed NFL packet', type=['json'], key='nfl_private_upload')
        if st.button('Stage existing NFL private packet', disabled=not enabled or upload is None, key='nfl_private_stage'):
            try:
                packet = evidence.load(upload.getvalue(), owner_upload=True)
                staged = st.session_state.setdefault('nfl_private_staged', {})
                evidence.require(packet['sha256'] in staged or len(staged) < evidence.MAX_PACKETS, 'NFL_PRIVATE_PACKET_SCHEMA')
                staged[packet['sha256']] = packet
            except (ValueError, TypeError, KeyError):
                st.error('NFL_PRIVATE_PACKET_SCHEMA: staging rejected; original facts were not changed.')
        staged = st.session_state.get('nfl_private_staged', {})
        keys = st.multiselect('NFL private inputs for the next requested analysis', list(staged), key='nfl_private_selected', disabled=not enabled)
        packets = [staged[k] for k in keys] if enabled else []
        try:
            with evidence.selected(packets):
                pass
        except (ValueError, TypeError, KeyError):
            packets = []
            st.error('NFL_PRIVATE_PACKET_SCHEMA: selection rejected.')
        st.session_state['nfl_private_packets'] = packets
        if candidates is not None:
            for _, row in candidates.iterrows():
                if not evidence.private(row):
                    continue
                assessed = evidence.diagnose(row)
                if assessed['status'] == 'COMPLETE':
                    st.write('OWNER_REVIEWED / PRIVATE_RESEARCH — ' + str(row.get('best_pick', row.get('market_type', ''))))
                    st.metric('Raw NFL research probability', f"{assessed['probability']:.2%}")
                    st.caption('Uncalibrated; EV/edge unavailable. PASS — zero stake. Private evidence is excluded from public packages.')
                    if evidence.saved_route(row).VERSION in {'nfl-owner-observed-private-inputs-v2','nfl-owner-scoring-private-inputs-v3'}:
                        st.caption('Prospective first-observation evidence; publisher first-publication time UNKNOWN. This supplies no historical availability claim.')
                else:
                    st.write('Private NFL estimate unavailable: ' + assessed['reason'])
    render_scenario()


def render_scenario():
    """Same owner-token panel, separate explicit action; no general live analysis."""
    from app_core import nfl_scenario_research as scenario, research_replay
    with st.expander('Private NFL sporting scenario', expanded=False):
        enabled = st.checkbox('Enable user-specified NFL scenario', value=False, key='nfl_scenario_enabled')
        st.caption('Full-game final points including overtime, conditional on a completed sporting result. No observed offer, price, EV, edge or blend. Integer handicaps unavailable. Staging creates no owner review. Execution is a separate explicit action using retained inputs only.')
        upload = st.file_uploader('Existing owner-reviewed sporting-scenario packet', type=['json'], key='nfl_scenario_upload')
        if st.button('Stage scenario packet', disabled=not enabled or upload is None, key='nfl_scenario_stage'):
            try:
                packet = scenario.load(upload.getvalue(), owner_upload=True)
                st.session_state['nfl_scenario_staged'] = packet
                st.session_state['nfl_scenario_confirmed'] = False
                st.session_state.pop('nfl_scenario_receipt', None)
            except (ValueError, TypeError, KeyError) as exc:
                st.error(scenario.reason(exc))
        packet = st.session_state.get('nfl_scenario_staged') if enabled else None
        if packet is not None:
            assumptions = packet['payload']['scenario']
            st.write('User-specified assumption: ' + str(assumptions.get('selected_team')) + ' ' + str(assumptions.get('signed_handicap')))
        confirmed = st.checkbox('Use this exact staged team and signed half-point assumption', value=False, key='nfl_scenario_confirmed', disabled=packet is None)
        if st.button('Run and privately capture reviewed scenario', disabled=not enabled or packet is None or not confirmed, key='nfl_scenario_run'):
            try:
                st.session_state['nfl_scenario_receipt'] = scenario.run(packet)
            except (ValueError, TypeError, KeyError, OSError) as exc:
                st.session_state.pop('nfl_scenario_receipt', None)
                st.error(scenario.reason(exc))
        receipt = st.session_state.get('nfl_scenario_receipt') if enabled else None
        if receipt is not None:
            try:
                result = research_replay.read_scenario(receipt)
                p = scenario.read_result(result)
                st.write('OWNER_REVIEWED / PRIVATE_RESEARCH / USER-SPECIFIED SCENARIO')
                st.metric('Uncalibrated NFL model estimate', f"{p['probability']:.2%}")
                st.caption('Conditional on a completed full game including overtime. PASS — zero stake. Publisher first availability UNKNOWN; original observation and actual inference clocks are preserved. No operator verification, market blend, calibration or qualification.')
                archive, _ = research_replay.download_scenario(receipt)
                st.download_button('Download private scenario evidence', data=archive, file_name=receipt['result_sha256'] + '.private-scenario.zip', mime='application/zip', key='nfl_scenario_download')
            except (ValueError, TypeError, KeyError, OSError) as exc:
                st.error(scenario.reason(exc))
