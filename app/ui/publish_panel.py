"""Private, explicit local publication of the current analysis."""
import hashlib
import hmac
import json
import os
from datetime import datetime, time
from zoneinfo import ZoneInfo
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
    digest.update((ROOT / "publishing/board.html").read_bytes())
    for frame in (games, candidates, props, dfs):
        digest.update((frame.to_json(orient='split', date_format='iso') if isinstance(frame,pd.DataFrame) else '').encode())
    return digest.hexdigest()


def dfs_lock_timestamp(day, clock):
    return datetime.combine(day, clock, tzinfo=ZoneInfo('America/New_York')).isoformat()


def render_dfs_lock_picker():
    date_column, time_column = st.columns(2)
    with date_column:
        day = st.date_input('DFS lock date', value=datetime.now(ZoneInfo('America/New_York')).date(),
                            key='dfs_lock_date', format='MM/DD/YYYY')
    with time_column:
        clock = st.time_input('DFS lock time', value=time(13, 0), key='dfs_lock_clock', step=60)
    start = dfs_lock_timestamp(day, clock)
    display = datetime.fromisoformat(start)
    st.caption('Locks '+display.strftime('%A, %B %d, %Y at %I:%M %p')+' Eastern time. Daylight saving time is handled automatically.')
    return start


def render_publish_panel(games, candidates, props=None, dfs=None):
    st.subheader('Publish board')
    st.caption('History and previews load automatically. Lock selected picks to save and publish them, or review the board below and publish it. Player props and DraftKings options are included as selected.')
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
    publish_results_requested = st.session_state.pop('publish_results_requested', False)
    if st.session_state.pop('lock_saved_notice', False):
        st.success('Picks locked in Drive. Earlier locks keep their original selections and odds.')
        notice=st.session_state.pop('lock_publish_notice', None)
        if notice:
            st.info(notice)
    if games is None or games.empty:
        if publish_results_requested:
            from copy import deepcopy
            from app.ui.sftp_publish import publish_action
            record=st.session_state.get('public_results_'+str(setting('PARLAYPICKER_NETLIFY_SITE_ID')).strip(),{})
            pubs=record.get('publications',[])
            if pubs:
                package=deepcopy(max(pubs,key=lambda p:p['confirmed_at'])['package'])
                package['schema_version']=5
                package.setdefault('parlays',[])
                package['results']=public_results or []
                st.info(publish_action(package,setting))
            else:
                st.info('Results saved. Refresh picks to create your first public board.')
        st.info('Use Refresh picks to prepare game picks.')
        return
    props = props if isinstance(props,pd.DataFrame) else pd.DataFrame()
    def describe_dates(frame):
        from app_core.public_board import timestamp
        values = []
        for _, row in frame.iterrows():
            raw = row.get('prediction_generated_at')
            if pd.isna(raw) or not str(raw or '').strip():
                raw = row.get('export_run_id')
            try:
                value = timestamp(raw)
                if value:
                    values.append(value)
            except (ValueError, TypeError):
                pass
        return min(values) if values else None
    game_date, prop_date = describe_dates(games), describe_dates(props)
    st.caption('Game analysis (UTC): ' + str(game_date or 'Unknown') +
               ' · Player props (UTC): ' + str(prop_date or 'Not run'))
    if prop_date and pd.Timestamp.now(tz='UTC') - pd.Timestamp(prop_date) > pd.Timedelta(minutes=15):
        st.warning('Saved player props are older than 15 minutes. Run Player Props to refresh them before including them as current selections.')
    dfs = dfs or {}
    include_props = st.checkbox('Include saved player props', value=bool(prop_date and pd.Timestamp.now(tz='UTC')-pd.Timestamp(prop_date)<=pd.Timedelta(minutes=15)), disabled=props.empty)
    choices = ['None', *sorted(k for k,v in dfs.items() if isinstance(v,pd.DataFrame) and not v.empty)]
    chosen = st.selectbox('DraftKings slate to include', choices)
    slate = start = ''
    if chosen != 'None':
        slate = st.text_input('DFS slate name', help='Use the exact contest slate label.')
        start = render_dfs_lock_picker()
    st.caption('DFS lineups must be generated in Full Pick Board during this run. Only one Classic slate is included per publication. Empty sections remain visible as empty tabs.')
    selected_props = props if include_props else pd.DataFrame()
    selected_dfs = dfs.get(chosen)
    options = {'results':public_results, 'props':include_props, 'dfs':chosen, 'slate':slate, 'start':start}
    fingerprint = source_fingerprint(games,candidates,selected_props,selected_dfs,options)
    saved = st.session_state.get('publication_preview')
    if saved and saved['fingerprint'] != fingerprint:
        st.session_state.pop('publication_preview', None)
        saved = None
        st.info('Preview updated to reflect your latest inputs.')
    rebuild = st.button('Refresh preview', key='publication_build')
    if saved is None or rebuild:
        try:
            boards = [per_game_board(games,candidates,family) for family in ('overall','sides','totals')]
            package = build_package(*boards, props=selected_props,
                                    dfs=selected_dfs, dfs_sport=chosen if chosen!='None' else None,
                                    dfs_slate=slate, dfs_start=start)
            package['schema_version'] = 5
            package['results'] = public_results or []
            html = render(package)
            saved = {'fingerprint':fingerprint, 'package':package, 'html':html}
            st.session_state['publication_preview'] = saved
        except (ValueError, TypeError, KeyError) as exc:
            st.session_state.pop('publication_preview',None)
            saved = None
            st.error('Preview could not be built: '+str(exc))
    if not saved:
        if publish_results_requested:
            st.error('Results were saved, but the preview could not be prepared. Correct the inputs and click Publish board.')
        return
    package = saved['package']
    if publish_results_requested:
        from app.ui.sftp_publish import publish_action
        st.info(publish_action(package, setting))
    st.write(f"{len(package['games']['overall'])} games · {len(package['props'])} props · {len(package['dfs'])} DFS lineups")
    import streamlit.components.v1 as components
    components.html(saved['html'], height=650, scrolling=True)
    from app.ui.lock_picks import render_lock_picks
    render_lock_picks(package, setting)
    with st.expander('Downloads and local copy', expanded=False):
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
