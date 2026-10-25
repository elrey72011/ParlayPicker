"""Saved-result controls within the existing owner-token protected NFL panel."""
from datetime import datetime, timezone

import streamlit as st

from app_core import nfl_saved_scenarios as saved, nfl_scenario_research as scenario


def label(row):
    return (f"{row['selected_team']} {row['signed_handicap']:+g} — "
            f"{row['away']} at {row['home']} — calculated {row['inference_time']} — "
            + row['receipt']['result_sha256'][:12])


def render():
    st.subheader('View saved NFL scenarios')
    st.caption('Static replay only. Loading saved results never runs a calculation, '
               'Refresh, provider request or public publication. Postgame viewing preserves the original calculation clock.')
    current_store = str(saved.store_path())
    st.caption('Configured private store: ' + current_store)
    if st.session_state.get('nfl_saved_store') != current_store:
        st.session_state['nfl_saved_loaded'] = []
        st.session_state['nfl_saved_selected'] = []
        st.session_state['nfl_saved_store'] = current_store
    try:
        index = saved.catalog()
    except (ValueError, TypeError, KeyError, OSError) as exc:
        st.session_state['nfl_saved_loaded'] = []
        st.info('Saved NFL scenarios unavailable: ' + scenario.reason(exc))
        return
    if index['rejected']:
        st.warning(f"{len(index['rejected'])} saved record(s) could not be authenticated and are unavailable. Originals were not changed.")
    if index['truncated']:
        st.info(f'Only the first {saved.MAX_LIST} saved records are listed.')
    records = {x['receipt']['export_id']: x for x in index['records']}
    if not records:
        st.session_state['nfl_saved_loaded'] = []
        st.info('No authenticated saved NFL scenarios in this configured store. No database was created or moved.')
        return
    # Remove stale selections before creating the widget if a record disappears.
    previous = st.session_state.get('nfl_saved_selected', [])
    st.session_state['nfl_saved_selected'] = [x for x in previous if x in records]
    selected = st.multiselect('Saved NFL scenarios', list(records), key='nfl_saved_selected',
                              format_func=lambda key: label(records[key]),
                              max_selections=saved.MAX_SELECTED)
    loaded = st.session_state.get('nfl_saved_loaded', [])
    if [x['receipt']['export_id'] for x in loaded] != selected:
        st.session_state['nfl_saved_loaded'] = []
    if st.button('Load saved results', key='nfl_saved_load', disabled=not selected):
        st.session_state['nfl_saved_loaded'] = [dict(receipt=records[key]['receipt'],
            viewed_at=datetime.now(timezone.utc).isoformat()) for key in selected]
    for item in st.session_state.get('nfl_saved_loaded', []):
        try:
            p = saved.load(item['receipt'])
            s = p['input']['payload']['scenario']
            st.write(f"Saved scenario: {s['selected_team']} {s['signed_handicap']:+g}")
            st.write('OWNER_REVIEWED / PRIVATE_RESEARCH / USER-SPECIFIED SCENARIO')
            st.metric('Uncalibrated saved native cover probability', f"{p['probability']:.2%}")
            native = p['native_output']
            st.write(f"Native margin projection: {native['ml_projection']:+g} points "
                     '(selected-team final points minus opponent final points, before handicap).')
            st.write(f"Native residual scale: {native['ml_residual_scale']:g} points.")
            st.write('Original inference time: ' + p['inference_time'])
            st.caption('Loaded for viewing at: ' + item['viewed_at'] + ' (UTC). This is not a new inference time.')
            st.caption('Conditional on a completed full game including overtime. Uncalibrated; PASS — zero stake. '
                       'Not a current sportsbook offer, verified settlement recommendation or EV estimate.')
            st.caption('Saved result SHA-256: ' + item['receipt']['result_sha256'])
            archive, _ = saved.private_download(item['receipt'])
            st.download_button('Download saved private scenario evidence', data=archive,
                file_name=item['receipt']['result_sha256'] + '.private-scenario.zip',
                mime='application/zip', key='nfl_saved_download_' + item['receipt']['result_sha256'])
        except (ValueError, TypeError, KeyError, OSError) as exc:
            st.warning('Saved NFL scenario rejected: ' + scenario.reason(exc))
