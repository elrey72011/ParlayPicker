"""Explicit locks on the reviewed overall board; no navigation-triggered writes."""
import pandas as pd
import streamlit as st
from app_core.locked_picks import lock_candidates
from app_core.public_history import now, report


def render_lock_picks(package, setting):
    from app.ui.public_results import history
    key = 'public_results_' + str(setting('PARLAYPICKER_NETLIFY_SITE_ID')).strip()
    saved = st.session_state.get(key)
    if saved is None:
        return
    with st.expander('Lock Overall Best Picks', expanded=True):
        st.caption('Choose today’s picks to track. Locks preserve the exact line and odds and cannot be replaced by later analysis. Locking does not approve a wager or place a bet. Prices must be under 15 minutes old and games must not have started.')
        existing = saved.get('locks', [])
        if existing:
            st.dataframe(pd.DataFrame([{'Date':r['date'], 'Game':r['legs'][0]['game'],
                'Locked pick':r['legs'][0]['pick'], 'Odds':r['legs'][0]['odds'],
                'Locked at (UTC)':r['published_at']} for r in existing]), hide_index=True)
        try:
            choices = {r['id']:r for r in lock_candidates(package, now()) if r['id'] not in {x['id'] for x in existing}}
        except ValueError:
            st.info('Rebuild a valid preview before locking picks.')
            return
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
                st.session_state.pop('publication_preview', None)
                st.session_state['lock_saved_notice'] = True
                st.rerun()
            except (ValueError, RuntimeError):
                st.error('Lock could not complete. Restore history to check saved locks, then rebuild the preview.')
            except Exception:
                st.error('Drive lock save or verification failed. Some selections may have saved; restore history before retrying.')
