"""Private, explicit local publication of the current analysis."""
import hashlib
import hmac
import json
import os
from datetime import datetime, time
from zoneinfo import ZoneInfo
from pathlib import Path
from app_core.quote_freshness import QUOTE_MAX_AGE_MINUTES
import pandas as pd
import streamlit as st
from app_core.per_game_boards import per_game_board
from app_core.public_board import build_package
from scripts.publish_board import ROOT, render, publish_package, production_source_fingerprint


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
    digest.update(production_source_fingerprint().encode())
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


def render_publish_panel(games, candidates, props=None, dfs=None, *, lazy_history=False):
    st.subheader('Publish board')
    st.caption('Load saved history when needed. Previews load automatically. Lock selected picks to save and publish them, or review the board below and publish it. Player props and DraftKings options are included as selected.')
    token = str(setting('PARLAYPICKER_PUBLISH_TOKEN'))
    if len(token) < 16:
        st.info('Publishing is locked. Configure PARLAYPICKER_PUBLISH_TOKEN with at least 16 characters in Streamlit secrets or the local environment. Never put it in the repository.')
        return
    supplied = st.text_input('Publishing token', type='password', key='publication_token')
    if not hmac.compare_digest(supplied.encode(), token.encode()):
        st.info('Enter the publishing token to preview or publish.')
        return
    from app.ui.canonical_evidence_download import render as render_canonical_download
    render_canonical_download(setting)
    from app.ui.remote_canonical_download import render as render_remote_canonical_download
    render_remote_canonical_download(setting)
    from app.ui.source_evidence_panel import render as render_source_evidence
    render_source_evidence()
    from app.ui.ncaaf_pipeline_research import render as render_ncaaf_pipeline_research
    st.session_state['ncaaf_private_captured_views'] = (games, candidates)
    render_ncaaf_pipeline_research()
    from app.ui.activation_panel import render as render_activation
    render_activation(games)
    from app.ui.public_results import render_history
    public_results = render_history(setting, lazy=True) if lazy_history else render_history(setting)
    publish_results_requested = st.session_state.pop('publish_results_requested', False)
    if st.session_state.pop('lock_saved_notice', False):
        st.success('Picks locked in Drive. Earlier locks keep their original selections and odds.')
        notice=st.session_state.pop('lock_publish_notice', None)
        if notice:
            st.info(notice)
    if games is None or (games.empty and not games.attrs.get('slate_coverage')):
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
        st.info('No game analysis is loaded in this session. You can score saved picks with Update results and publish above. Use Refresh picks only when you want a new game board.')
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
    if prop_date and pd.Timestamp.now(tz='UTC') - pd.Timestamp(prop_date) > pd.Timedelta(minutes=QUOTE_MAX_AGE_MINUTES):
        st.warning(f'Saved player props are older than {QUOTE_MAX_AGE_MINUTES} minutes. Run Player Props to refresh them before including them as current selections.')
    dfs = dfs or {}
    include_props = st.checkbox('Include saved player props', value=bool(prop_date and pd.Timestamp.now(tz='UTC')-pd.Timestamp(prop_date)<=pd.Timedelta(minutes=QUOTE_MAX_AGE_MINUTES)), disabled=props.empty)
    choices = ['None', *sorted(k for k,v in dfs.items() if isinstance(v,pd.DataFrame) and not v.empty)]
    chosen = st.selectbox('DraftKings slate to include', choices)
    slate = start = ''
    if chosen != 'None':
        slate = st.text_input('DFS slate name', help='Use the exact contest slate label.')
        start = render_dfs_lock_picker()
    st.caption('DFS lineups must be generated in Full Pick Board during this run. Only one Classic slate is included per publication. Empty sections remain visible as empty tabs.')
    nfl_fallback = st.checkbox('Allow NFL sportsbook fallback for research locks', value=True, key='publication_nfl_fallback',
                               help='Show a best available NFL pick when Novig is unavailable, using an exact fresh DraftKings, FanDuel or BetMGM quote labeled with its source. Novig remains preferred. These selections can be research-locked; this does not approve a wager.')
    research_fallback = st.checkbox('Allow MLB and WNBA sportsbook fallback for research locks', value=True, key='publication_research_fallback',
                                    help='When Novig is unavailable, use an exact fresh DraftKings, FanDuel or BetMGM quote for a research-only lock. This never approves or funds a wager.')
    selected_props = props if include_props else pd.DataFrame()
    selected_dfs = dfs.get(chosen)
    from app.ui.slate_coverage import render_coverage
    render_coverage(games.attrs.get('slate_coverage'), key='publication_slate_coverage')
    options = {'results':public_results, 'props':include_props, 'dfs':chosen, 'slate':slate, 'start':start, 'nfl_fallback':nfl_fallback, 'research_fallback':research_fallback}
    if games.attrs.get('slate_coverage') is not None:
        from app_core.slate_coverage import digest
        options['slate_coverage_hash'] = digest(games.attrs['slate_coverage'])
        # A previous successful placeholder preview must not hide a new rejected
        # original row whose DataFrame cells otherwise resemble that placeholder.
        options['coverage_binding_failures_hash'] = digest(games.attrs.get('coverage_binding_failures', []))
    fingerprint = source_fingerprint(games,candidates,selected_props,selected_dfs,options)
    saved = st.session_state.get('publication_preview')
    if saved and saved['fingerprint'] != fingerprint:
        st.session_state.pop('publication_preview', None)
        saved = None
        st.info('Preview updated to reflect your latest inputs.')
    rebuild = st.button('Refresh preview', key='publication_build')
    if saved is None or rebuild:
        try:
            boards = [per_game_board(games,candidates,family,novig_only=True,college_fallback=True,nfl_fallback=nfl_fallback,research_fallback=research_fallback) for family in ('overall','sides','totals')]
            package = build_package(*boards, props=selected_props,
                                    dfs=selected_dfs, dfs_sport=chosen if chosen!='None' else None,
                                    dfs_slate=slate, dfs_start=start)
            package['schema_version'] = 5
            package['results'] = public_results or []
            html = render(package)
            from app_core.current_wagers_trace import build_private_candidate_trace
            private_trace = build_private_candidate_trace(
                candidates, package, evaluated_at=package['built_at'],
                selection_options={
                    'college_fallback': True, 'nfl_fallback': nfl_fallback,
                    'research_fallback': research_fallback,
                },
            )
            saved = {'fingerprint':fingerprint, 'package':package, 'html':html,
                     'private_candidate_trace':private_trace}
            # Local owner evidence is distinct from publishing and remote sync.
            import sqlite3
            from app_core.research_replay import retain_export
            try:
                saved['research_replay_receipt'] = retain_export(boards, package, games, candidates)
            except (OSError, sqlite3.Error, ValueError, TypeError) as exc:
                saved['research_replay_error'] = str(exc)
                st.warning('Private research replay evidence was not saved: '+str(exc))
            st.session_state['publication_preview'] = saved
        except (ValueError, TypeError, KeyError) as exc:
            st.session_state.pop('publication_preview',None)
            saved = None
            st.error('Preview could not be built: '+str(exc))
            diagnostic = getattr(exc, 'diagnostic', None)
            if isinstance(diagnostic, dict):
                # Existing owner-token gate applies before this pre-build error.
                # Never log raw rows or include private prediction dependencies.
                allowed = ('version', 'code', 'stage', 'board_category', 'canonical_event_id',
                    'inventory_id', 'run_id', 'field', 'expected', 'actual')
                from app_core.coverage_presentation import _safe
                safe = {k: _safe(diagnostic[k]) for k in allowed if k in diagnostic}
                safe = {k: v.replace(token, '[redacted]') if isinstance(v, str) else v for k, v in safe.items()}
                serialized = json.dumps(safe, sort_keys=True, indent=2)
                with st.expander('Private preview conflict diagnostic', expanded=True):
                    st.caption('Original identity descriptors only; no prediction inputs or credentials. No analysis refresh is needed.')
                    st.json(json.loads(serialized))
                    st.download_button('Download private preview conflict JSON', serialized,
                        file_name='private-preview-coverage-conflict.json', mime='application/json',
                        key='publication_coverage_conflict_download')
    if not saved:
        if publish_results_requested:
            st.error('Results were saved, but the preview could not be prepared. Correct the inputs and click Publish board.')
        return
    package = saved['package']
    if publish_results_requested:
        from app.ui.sftp_publish import publish_action
        st.info(publish_action(package, setting))
    st.write(f"{len(package['games']['overall'])} games · {len(package['props'])} props · {len(package['dfs'])} DFS lineups")
    private_trace = saved.get('private_candidate_trace')
    if private_trace:
        with st.expander('Private candidate-to-release diagnostics', expanded=False):
            st.caption('Owner-only trace of supplied candidate evidence. It does not fetch odds, create authority, or enter the public package.')
            st.write({
                'trace_status': private_trace['trace_status'],
                'candidate_count': private_trace['all_market_candidate_count'],
                'primary_blockers': private_trace['primary_blocker_counts'],
                'release_preflight': {
                    key: private_trace['release_preflight'][key]
                    for key in ('evaluated_at','actionable_row_count','actionable_release_allowed','blocker_counts')
                },
            })
            st.download_button(
                'Download private candidate trace',
                json.dumps(private_trace, indent=2, sort_keys=True),
                'current-wagers-candidate-trace.json', 'application/json',
                key='download_current_wagers_candidate_trace',
            )
    replay_receipt = saved.get('research_replay_receipt')
    if replay_receipt:
        with st.expander('Private research replay evidence', expanded=False):
            st.caption('Download the retained source and per-game traces for this preview. Missing original sources remain UNKNOWN.')
            import sqlite3
            from app_core.research_replay import download_bundle, digest, encode
            try:
                bundle, verified = download_bundle(replay_receipt, expected_package_hash=digest(encode(package)))
            except (OSError, sqlite3.Error, ValueError, TypeError, KeyError) as exc:
                st.warning('Private research replay download is unavailable: '+str(exc))
            else:
                st.write({key:verified[key] for key in ('export_id','package_hash','source_boundary','source_links')})
                if verified['source_boundary'] == 'UNKNOWN':
                    st.warning('Original source evidence is unavailable for one or more snapshot links. The bundle preserves UNKNOWN.')
                st.download_button(
                    'Download private research replay bundle', bundle,
                    'private-research-replay-'+verified['export_id']+'.zip', 'application/zip',
                    key='download_private_research_replay', on_click='ignore',
                )
    from app_core.public_parlays import parlay_funnel
    with st.expander('Parlay eligibility funnel', expanded=False):
        if package.get('parlay_policy')=='canonical-v3':
            from app_core.production_parlays import canonical_funnel
            funnel=canonical_funnel(package['games']['overall'])
        else:
            funnel = parlay_funnel(package['games']['overall'])
        st.write(funnel['counts'])
        st.write(funnel['exclusions'])
        st.caption('Leg-qualified pairs still require an actual ticket price and validated joint model before any parlay stake. Counts reflect the current clock.')
    from app.ui.lock_picks import render_lock_picks
    render_lock_picks(package, setting)
    from app.ui.activation_panel import render_ticket_confirmation
    render_ticket_confirmation(package)
    with st.expander('Website preview', expanded=False):
        st.iframe(saved['html'], height=650)
    with st.expander('Downloads and local copy', expanded=False):
        st.download_button('Download preview HTML', saved['html'], 'parlaypicker-preview.html','text/html')
        st.download_button('Download public data', json.dumps(package,indent=2), 'public-board.json','application/json')
        destination = Path(str(setting('PARLAYPICKER_PUBLICATION_DIR', str(ROOT/'outputs/public-board-site'))))
        st.caption('Local output: '+str(destination))
        if st.button('Publish reviewed board locally', key='publication_publish'):
            try:
                publish_package(package,destination,setting=setting)
                st.success('Published locally. This local action does not update the public website. Download the HTML or use the separate public publish controls below.')
            except (OSError,ValueError) as exc:
                st.error('Local publication failed: '+str(exc))

    from app.ui.remote_publish import render_remote_publish
    if public_results is None:
        st.info('Restore public history above to enable public publication. Preview and local downloads remain available.')
    else:
        render_remote_publish(package, fingerprint, setting)
