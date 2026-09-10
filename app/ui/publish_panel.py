"""Private, explicit local publication of the current analysis."""
import hashlib
import hmac
import json
import os
from pathlib import Path
import pandas as pd
import streamlit as st
from app_core.per_game_boards import per_game_board
from app_core.public_board import build_package
from scripts.publish_board import ROOT, render, publish_package


def setting(name, default=''):
    value = os.environ.get(name)
    if value is not None:
        return value
    try:
        return st.secrets.get(name, default)
    except (FileNotFoundError, KeyError):
        return default


def source_fingerprint(games, candidates, props, dfs, options):
    digest = hashlib.sha256(json.dumps(options, sort_keys=True).encode())
    for frame in (games, candidates, props, dfs):
        digest.update((frame.to_json(orient='split', date_format='iso') if isinstance(frame,pd.DataFrame) else '').encode())
    return digest.hexdigest()


def render_publish_panel(games, candidates, props=None, dfs=None):
    st.subheader('Preview & Publish')
    st.caption('Private publishing workspace. Preview first, then choose local output or the separate configured public website controls below. Local output on Streamlit Cloud stays on the server; download the HTML to keep a copy.')
    token = str(setting('PARLAYPICKER_PUBLISH_TOKEN'))
    if len(token) < 16:
        st.info('Publishing is locked. Configure PARLAYPICKER_PUBLISH_TOKEN with at least 16 characters in Streamlit secrets or the local environment. Never put it in the repository.')
        return
    supplied = st.text_input('Publishing token', type='password', key='publication_token')
    if not hmac.compare_digest(supplied.encode(), token.encode()):
        st.info('Enter the publishing token to preview or publish.')
        return
    from app.ui.public_results import render_history
    public_results = render_history(setting)
    if games is None or games.empty:
        st.info('Run Master Analysis to prepare game picks first.')
        return
    props = props if isinstance(props,pd.DataFrame) else pd.DataFrame()
    dfs = dfs or {}
    include_props = st.checkbox('Include current player props', value=False, disabled=props.empty)
    choices = ['None', *sorted(k for k,v in dfs.items() if isinstance(v,pd.DataFrame) and not v.empty)]
    chosen = st.selectbox('DraftKings slate to include', choices)
    slate = start = ''
    if chosen != 'None':
        slate = st.text_input('DFS slate name', help='Use the exact contest slate label.')
        start = st.text_input('DFS lock time (with timezone)', placeholder='2026-09-13T13:00:00-04:00')
    st.caption('DFS lineups must be generated in Full Pick Board during this run. Only one Classic slate is included per publication. Empty sections remain visible as empty tabs.')
    selected_props = props if include_props else pd.DataFrame()
    selected_dfs = dfs.get(chosen)
    options = {'results':public_results, 'props':include_props, 'dfs':chosen, 'slate':slate, 'start':start}
    fingerprint = source_fingerprint(games,candidates,selected_props,selected_dfs,options)
    saved = st.session_state.get('publication_preview')
    if saved and saved['fingerprint'] != fingerprint:
        st.session_state.pop('publication_preview', None)
        saved = None
        st.info('Inputs changed. Build and review a new preview before publishing.')
    if st.button('Build preview', key='publication_build'):
        try:
            boards = [per_game_board(games,candidates,family) for family in ('overall','sides','totals')]
            package = build_package(*boards, props=selected_props,
                                    dfs=selected_dfs, dfs_sport=chosen if chosen!='None' else None,
                                    dfs_slate=slate, dfs_start=start)
            package['schema_version'] = 4
            package['results'] = public_results or []
            html = render(package)
            saved = {'fingerprint':fingerprint, 'package':package, 'html':html}
            st.session_state['publication_preview'] = saved
        except (ValueError, TypeError, KeyError) as exc:
            st.session_state.pop('publication_preview',None)
            saved = None
            st.error('Preview could not be built: '+str(exc))
    if not saved:
        return
    package = saved['package']
    st.write(f"{len(package['games']['overall'])} games · {len(package['props'])} props · {len(package['dfs'])} DFS lineups")
    import streamlit.components.v1 as components
    components.html(saved['html'], height=650, scrolling=True)
    st.download_button('Download preview HTML', saved['html'], 'parlaypicker-preview.html','text/html')
    st.download_button('Download public data', json.dumps(package,indent=2), 'public-board.json','application/json')
    destination = Path(str(setting('PARLAYPICKER_PUBLICATION_DIR', str(ROOT/'outputs/public-board-site'))))
    st.caption('Local output: '+str(destination))
    if st.button('Publish reviewed board locally', key='publication_publish'):
        try:
            publish_package(package,destination)
            st.success('Published locally. This local action does not update the public website. Download the HTML or use the separate public publish controls below.')
        except (OSError,ValueError) as exc:
            st.error('Local publication failed: '+str(exc))

    from app.ui.remote_publish import render_remote_publish
    if public_results is None:
        st.info('Restore public history above to enable public publication. Preview and local downloads remain available.')
    else:
        render_remote_publish(package, fingerprint, setting)
