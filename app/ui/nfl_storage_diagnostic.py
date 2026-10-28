"""Explicit owner-only entry point for the private NFL storage diagnostic."""
import streamlit as st
from app_core import nfl_diagnostic_receipts as receipts


def _invoke(setting, state):
    # A callback runs before render recomputes the button's disabled state.
    # Authentication and the independent guard also reject stale/repeated events.
    try:
        from app_core.nfl_private_workflow import authenticate
        access = authenticate(state.get('publication_token', ''), str(setting('PARLAYPICKER_PUBLISH_TOKEN')))
    except Exception:
        return
    started = receipts.claim(state)
    if started is None:
        return
    try:
        receipts.emit(started)  # retained before imports, connection work or display
        from app_core.nfl_storage_diagnostic import diagnose
        state['nfl_workflow_diag_result'] = {**started, 'stage': 'DIAGNOSTIC_STARTED'}
        receipts.emit(state['nfl_workflow_diag_result'])
        result = receipts.sanitize(diagnose(setting, access, state.get('nfl_workflow_diag_expected_host', '')))
    except Exception as exc:
        # No returned instrumentation means the connect count is UNKNOWN, not zero.
        result = receipts.unknown_failure('DIAGNOSTIC_UNAVAILABLE', exc)
    except BaseException as exc:
        # Streamlit interruption/stop and interpreter interruption retain a spent
        # receipt. Re-raise control flow; no rerender may resume the operation.
        receipts.finish(state, started, receipts.unknown_failure('INTERRUPTED_UNKNOWN', exc), 'INTERRUPTED_UNKNOWN')
        raise
    receipts.finish(state, started, result, 'COMPLETED' if result['status'] == 'PASS' else 'FAILED')


def render(setting, state):
    st.caption('Build preflight makes no database or provider connection and does not spend or reset the storage-check allowance. Compare the observed revision and loaded-code findings with the cleared release before a storage check.')
    if st.button('Read running NFL build information (no connection)', key='nfl_workflow_build_run'):
        try:
            from app_core.nfl_private_workflow import authenticate
            from app_core.nfl_build_preflight import inspect_build
            access = authenticate(state.get('publication_token', ''), str(setting('PARLAYPICKER_PUBLISH_TOKEN')))
            state['nfl_workflow_build_result'] = inspect_build(access)
        except Exception:
            state['nfl_workflow_build_result'] = {'status': 'UNVERIFIED',
                'failure': {'phase': 'OWNER_OR_IMPORT', 'category': 'BUILD_EVIDENCE_UNAVAILABLE'}}
    build = state.get('nfl_workflow_build_result')
    if isinstance(build, dict):
        st.caption('Last explicit build observation; rerendering does not recheck it. Read again after a deployment or restart.')
        st.json(build)
    st.caption('One read-only storage check. It does not acquire data or calculate a forecast. '
               'Run only with current owner authorization; do not retry an interrupted attempt.')
    expected_host = st.text_input('Verified production direct hostname', type='password',
                                  key='nfl_workflow_diag_expected_host',
                                  help='Enter the privately verified host only; it is never displayed in results.')
    spent = bool(state.get(receipts.GUARD) or state.get('nfl_workflow_diag_spent', False))
    st.button('Run one read-only NFL storage diagnostic', disabled=spent or not expected_host.strip(),
              key='nfl_workflow_diag_run', on_click=_invoke, args=(setting, state))
    result = state.get('nfl_workflow_diag_result')
    if isinstance(result, dict):
        if result.get('outcome') == 'STARTED':
            st.warning('Invocation started, but no terminal receipt is retained. Outcome and connection '
                       'counts are UNKNOWN. Do not run again. Interruption may prevent finalization.')
        st.caption('Retained invocation evidence. Viewing or reopening it does not rerun the check. '
                   'A STARTED receipt without a terminal receipt has an unknown outcome; never retry it.')
        st.json(result)
    elif spent:
        st.caption('This session has already spent its storage-check action. Private receipt display '
                   'was cleared with authorization; no new check will run. Retain the server-log receipt.')
