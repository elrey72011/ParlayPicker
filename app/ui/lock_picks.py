"""Explicit locks on the reviewed overall board; no navigation-triggered writes."""
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import streamlit as st
from app_core.locked_picks import lock_candidates
from app_core.public_history import now, report
from app_core.public_record import current_records


def render_lock_picks(package, setting):
    from app.ui.public_results import history
    key = 'public_results_' + str(setting('PARLAYPICKER_NETLIFY_SITE_ID')).strip()
    saved = st.session_state.get(key)
    if saved is None:
        return
    with st.expander('Lock Overall Best Picks', expanded=True):
        st.caption('Lock selected picks saves the original selections to Drive and publishes the board below, including your selected props and DFS slate. Existing locks cannot be replaced. Prices must be under 15 minutes old and games must not have started. Locking does not place a bet.')
        existing = saved.get('locks', [])
        published_ids=set()
        pubs=saved.get('publications',[])
        if pubs:
            latest=max(pubs,key=lambda p:p['confirmed_at'])
            published_ids={r['id'] for r in latest['package'].get('results',[]) if r['group']=='Locked'}
        for job in st.session_state.get('sftp_jobs',{}).values():
            if job.get('history_saved'):
                published_ids.update(job.get('locked_ids',[]))
        if existing:
            st.dataframe(pd.DataFrame([{'Date':r['date'], 'Game':r['legs'][0]['game'],
                'Locked pick':r['legs'][0]['pick'], 'Odds':r['legs'][0]['odds'],
                'Locked at (UTC)':r['published_at'],
                'Website':'Published' if r['id'] in published_ids else 'Not verified as published'} for r in existing]), hide_index=True)
        try:
            choices = {r['id']:r for r in lock_candidates(package, now()) if r['id'] not in {x['id'] for x in existing}}
        except ValueError:
            st.info('Rebuild a valid preview before locking picks.')
            return
        today=datetime.now(ZoneInfo('America/New_York')).date().isoformat()
        locked_today=sum(r['date']==today for r in existing)
        st.caption(f'Locked today: {locked_today} · Not locked and eligible now: {len(choices)}')
        if not choices:
            st.info('No new eligible picks to lock. Existing locks remain saved; refresh analysis for stale prices.')
            return
        selected = st.multiselect('Picks to lock', list(choices), default=list(choices),
            format_func=lambda key: choices[key]['legs'][0]['game'] + ': ' + choices[key]['legs'][0]['pick'] + ' (' + str(choices[key]['legs'][0]['odds']) + ')',
            key='lock_pick_selection')
        if st.button('Lock selected picks', key='lock_picks_action', disabled=not selected):
            try:
                store = history(setting)
                store.lock_picks(package, selected)
                # Restore authoritative records: a concurrent first lock may differ.
                locks = store.all('locks')
                saved['locks'] = locks
                saved['rows'] = report(saved['publications'], saved['revisions'], saved.get('imports',[]), locks)
                from copy import deepcopy
                from app_core import public_prop_history
                from app.ui.sftp_publish import publish_action
                updated=deepcopy(package)
                updated['results']=current_records(saved['rows']+public_prop_history.report(saved['publications'],saved.get('prop_revisions',[]),saved.get('prop_imports',[])))
                st.session_state['lock_publish_notice']=publish_action(updated,setting)
                st.session_state.pop('publication_preview', None)
                st.session_state['lock_saved_notice'] = True
                st.rerun()
            except (ValueError, RuntimeError):
                st.error('Lock could not complete. Restore history to check saved locks, then rebuild the preview.')
            except Exception:
                st.error('Drive lock save or verification failed. Some selections may have saved; restore history before retrying.')
