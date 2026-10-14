"""Explicit retained-input action inside the existing owner-token panel."""
import streamlit as st
from app_core import wnba_scenario_research as scenario, wnba_scenario_replay as replay


def render():
    with st.expander('Private WNBA sporting scenario', expanded=False):
        enabled = st.checkbox('Enable user-specified WNBA scenario', value=False, key='wnba_scenario_enabled')
        st.caption('Retained scoring only. A signed half-point sporting assumption, full game including overtime conditional on completion. Staging and confirmation create no review. No observed offer, price, EV, edge or market blend.')
        upload = st.file_uploader('Existing owner-reviewed WNBA scenario packet', type=['json'], key='wnba_scenario_upload')
        if st.button('Stage WNBA scenario', disabled=not enabled or upload is None, key='wnba_scenario_stage'):
            try:
                st.session_state['wnba_scenario_staged'] = scenario.load(upload.getvalue(), owner_upload=True)
                st.session_state['wnba_scenario_confirmed'] = False
                st.session_state.pop('wnba_scenario_receipt', None)
            except (ValueError, TypeError, KeyError) as exc:
                st.error(scenario.reason(exc))
        packet = st.session_state.get('wnba_scenario_staged') if enabled else None
        if packet is not None:
            st.write(packet['payload']['scenario'])
        confirmed = st.checkbox('Use this exact staged WNBA assumption', value=False, key='wnba_scenario_confirmed', disabled=packet is None)
        if st.button('Run and privately capture reviewed WNBA scenario', disabled=not enabled or packet is None or not confirmed, key='wnba_scenario_run'):
            try:
                st.session_state['wnba_scenario_receipt'] = scenario.run(packet)
            except (ValueError, TypeError, KeyError, OSError) as exc:
                st.session_state.pop('wnba_scenario_receipt', None)
                st.error(scenario.reason(exc))
        receipt = st.session_state.get('wnba_scenario_receipt') if enabled else None
        if receipt is not None:
            try:
                view = replay.display(replay.read(receipt))
                st.write(' / '.join(view['labels']))
                st.caption('Evidence: ' + view['evidence_label'])
                st.metric(view['description'], f"{view['probability']:.6%}")
                st.caption('PASS — zero stake. Full-game sporting scenario; publisher availability UNKNOWN. Private evidence is excluded from public packages and qualification. No operator settlement verification.')
                archive, _ = replay.download(receipt)
                st.download_button('Download private WNBA scenario evidence', data=archive, file_name=receipt['result_sha256']+'.private-wnba-scenario.zip', mime='application/zip', key='wnba_scenario_download')
            except (ValueError, TypeError, KeyError, OSError) as exc:
                st.error(scenario.reason(exc))
