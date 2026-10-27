"""Explicit owner-only entry point for the private NFL storage diagnostic."""
from datetime import datetime, timezone
import streamlit as st


def render(setting, state):
    st.caption('One read-only storage check. It does not acquire data or calculate a forecast. '
               'Run only with current owner authorization; do not retry an interrupted attempt.')
    expected_host = st.text_input('Verified production direct hostname', type='password',
                                  key='nfl_workflow_diag_expected_host',
                                  help='Enter the privately verified host only; it is never displayed in results.')
    spent = bool(state.get('nfl_workflow_diag_spent', False))
    if st.button('Run one read-only NFL storage diagnostic', disabled=spent or not expected_host.strip(), key='nfl_workflow_diag_run'):
        # Record the attempt before importing or opening anything. A rerun cannot repeat it.
        invoked_at = datetime.now(timezone.utc).isoformat()
        state['nfl_workflow_diag_spent'] = True
        try:
            from app_core.nfl_private_workflow import authenticate
            from app_core.nfl_storage_diagnostic import diagnose
            access = authenticate(state.get('publication_token', ''), str(setting('PARLAYPICKER_PUBLISH_TOKEN')))
            state['nfl_workflow_diag_result'] = diagnose(setting, access, expected_host)
        except Exception:
            state['nfl_workflow_diag_result'] = {'status': 'STOPPED', 'attempted_connections': 0,
                'failure': {'phase': 'OWNER_OR_IMPORT', 'exception_class': None,
                            'sqlstate': None, 'category': 'DIAGNOSTIC_UNAVAILABLE'}}
        state['nfl_workflow_diag_result']['invoked_at'] = invoked_at
    result = state.get('nfl_workflow_diag_result')
    if isinstance(result, dict):
        st.json(result)
