"""Saved-result controls within the existing owner-token protected NFL panel."""
from datetime import datetime, timezone

import streamlit as st

from app_core import nfl_saved_scenarios as saved, nfl_scenario_research as scenario
from app.ui.nfl_saved_viewer_state import clear_payloads


def label(row):
    return (f"{row['selected_team']} {row['signed_handicap']:+g} — "
            f"{row['away']} at {row['home']} — calculated {row['inference_time']} — "
            + row['receipt']['result_sha256'][:12])


def render(*, synthetic_policy=None):
    st.subheader('View saved NFL scenarios')
    st.caption('Static replay only. Loading saved results never runs a calculation, '
               'Refresh, provider request or public publication. Postgame viewing preserves the original calculation clock.')
    source = st.radio('Saved NFL result source',
        ['Existing private database', 'Upload saved scenario ZIPs'], key='nfl_saved_source')
    st.caption('Choose the source before enabling the viewer. ZIP uploads stay in this authenticated session and never access the database.')
    if not st.checkbox('Show saved NFL scenarios', value=False, key='nfl_saved_show'):
        clear_payloads(st.session_state)
        return
    if st.session_state.get('nfl_saved_current_source') != source:
        clear_payloads(st.session_state)
        st.session_state['nfl_saved_current_source'] = source
    if source == 'Upload saved scenario ZIPs':
        render_uploaded(synthetic_policy=synthetic_policy)
        return
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


def render_uploaded(*, synthetic_policy=None):
    from app_core import nfl_scenario_zip as replay
    st.caption('Upload existing private-scenario exports, not inference-input JSON. Only supported, externally trusted completed results can be replayed. No calculation, storage, Refresh or publishing occurs.')
    if st.button('Clear uploaded saved results', key='nfl_saved_zip_clear'):
        clear_payloads(st.session_state)
        st.rerun()
    generation = st.session_state.get('nfl_saved_zip_generation', 0)
    files = st.file_uploader('Saved NFL scenario ZIP exports (multiple files)', type=['zip'],
        accept_multiple_files=True, max_upload_size=4, key='nfl_saved_zip_upload_' + str(generation))
    files = files or []
    if len(files) > replay.MAX_SELECTED or sum(f.size for f in files) > replay.MAX_BATCH:
        st.session_state['nfl_saved_zip_loaded'] = None
        st.error('NFL_SCENARIO_ZIP_SELECTION_LIMIT (selection)')
        return
    # Compressed-byte identities invalidate changed files without ZIP parsing or
    # authenticating any result before the explicit Load action.
    identities = [(str(f.file_id), f.name, f.size, replay.sha(f.getbuffer())) for f in files]
    if st.session_state.get('nfl_saved_zip_files') != identities:
        st.session_state['nfl_saved_zip_loaded'] = None
        for key in list(st.session_state):
            if str(key).startswith('nfl_saved_zip_selected_'):
                del st.session_state[key]
        st.session_state['nfl_saved_zip_files'] = identities
    by_id = {str(i): f for i, f in enumerate(files)}
    selected = st.multiselect('Uploaded ZIPs to view', list(by_id), default=[],
        format_func=lambda key: by_id[key].name,
        key='nfl_saved_zip_selected_' + str(generation) + '_' + replay.digest(identities)[:16],
        max_selections=replay.MAX_SELECTED)
    loaded = st.session_state.get('nfl_saved_zip_loaded')
    if loaded and loaded['selected'] != selected:
        st.session_state['nfl_saved_zip_loaded'] = None
    if st.button('Load uploaded saved results', disabled=not selected, key='nfl_saved_zip_load') and selected:
        st.session_state['nfl_saved_zip_loaded'] = None
        try:
            replay.read_archives([by_id[key].getvalue() for key in selected], synthetic_policy=synthetic_policy)
            st.session_state['nfl_saved_zip_loaded'] = dict(selected=list(selected),
                viewed_at=datetime.now(timezone.utc).isoformat())
        except replay.ReplayError as exc:
            st.error(exc.code + ' (' + exc.field + ')')
    loaded = st.session_state.get('nfl_saved_zip_loaded')
    if not loaded:
        return
    try:
        # Fresh authentication on every display; no cached acceptance verdict.
        items = replay.read_archives([by_id[key].getvalue() for key in loaded['selected']], synthetic_policy=synthetic_policy)
    except replay.ReplayError as exc:
        st.session_state['nfl_saved_zip_loaded'] = None
        st.error(exc.code + ' (' + exc.field + ')')
        return
    for item in items:
        p = item['payload']; q = p['input']['payload']; e, s = q['event'], q['scenario']
        if item['evidence_label'] == 'SYNTHETIC':
            st.warning('SYNTHETIC test fixture — not an authentic model execution or owner finding.')
        st.write('Event: ' + e['away'] + ' at ' + e['home'] + ' — ' + e['start'])
        side = 'HOME' if s['selected_team'] == e['home'] else 'AWAY'
        st.write(f"Saved scenario: {s['selected_team']} {side} {s['signed_handicap']:+g}")
        st.write('OWNER_REVIEWED / PRIVATE_RESEARCH / USER-SPECIFIED SCENARIO')
        st.metric('Uncalibrated saved native cover probability', f"{p['probability']:.4%}")
        st.write(f"Native margin projection: {p['native_output']['ml_projection']:+g} points (selected-team final points minus opponent final points, before handicap).")
        st.write(f"Native residual scale: {p['native_output']['ml_residual_scale']:g} points.")
        st.write('Original inference time: ' + p['inference_time'])
        st.caption('Loaded for viewing at: ' + loaded['viewed_at'] + ' (UTC). This is not a new inference time.')
        st.caption('Saved result SHA-256: ' + item['result_sha256'])
        st.caption('Saved input SHA-256: ' + item['input_sha256'])
        st.caption('Recorded execution profile: ' + item['profile_id'] + '. Static replay validates the recorded producer, not the viewing runtime as an inference runtime.')
        st.caption('Conditional on a completed full game including overtime. Uncalibrated private research; PASS — zero stake. Not a current quote, EV estimate, sportsbook recommendation or execution authority.')
