"""Token-gated static staging and private browser; no acquisition or inference."""
import streamlit as st
from app_core import nfl_owner_research as evidence


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
                else:
                    st.write('Private NFL estimate unavailable: ' + assessed['reason'])
