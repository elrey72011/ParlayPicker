"""Explicit locks on the reviewed overall board; no navigation-triggered writes."""
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import streamlit as st
from app_core.locked_picks import lock_candidates, lock_audit
from app_core.quote_freshness import package_age_minutes
from app_core.public_history import now, report, digest, lock_stage
from app_core.public_record import current_records


def render_lock_picks(package, setting):
    from app.ui.public_results import history
    key = 'public_results_' + str(setting('PARLAYPICKER_NETLIFY_SITE_ID')).strip()
    notice=st.session_state.pop('lock_correction_notice',None)
    if notice:st.info(notice)
    saved = st.session_state.get(key)
    if saved is None:
        return
    with st.expander('Lock Overall Best Picks', expanded=True):
        st.caption(f'Lock selected picks saves the original selections to Drive and publishes the board below, including your selected props and DFS slate. Existing locks cannot be replaced. Quote updates or labeled ESPN observations must be at most {package_age_minutes(package)} minutes old and games must not have started. Locking does not place a bet.')
        if any(r.get('quote_time_basis') == 'espn_observed' for r in package['games']['overall']):
            st.caption('ESPN college research picks use the time we observed the snapshot. DraftKings update time is unknown. Refresh preview does not renew the observation time.')
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
                'Sportsbook':r['legs'][0].get('quote_source','Not recorded'),
                **({'Observed at (UTC)':r['legs'][0]['quote_time']} if r['legs'][0].get('quote_time_basis') == 'espn_observed' else {}),
                'Locked at (UTC)':r['published_at'],
                'Website':'Published' if r['id'] in published_ids else 'Not verified as published'} for r in existing]), hide_index=True)
        render_lock_correction(package, setting, saved)
        checked_at = now()
        try:
            audit = lock_audit(package, checked_at, existing)
            render_lock_audit(audit, checked_at)
            choices = {r['id']:r for r in lock_candidates(package, checked_at) if r['id'] not in {x['id'] for x in existing}}
        except ValueError:
            st.info('Rebuild a valid preview before locking picks.')
            return
        today=datetime.fromisoformat(checked_at).astimezone(ZoneInfo('America/New_York')).date().isoformat()
        locked_today=sum(r['date']==today for r in existing)
        st.caption(f'Locked today: {locked_today} · Not locked and eligible now: {len(choices)}')
        if not choices:
            blocked = {r['Lock status'] for r in audit}
            if blocked & {'Stale quote', 'Stale analysis'}:
                st.info('No new eligible picks to lock. Some saved prices or analysis are stale; click Refresh picks, then return here. Refresh preview and Run Player Props do not refresh game quotes. See Why games cannot be locked for each reason.')
            else:
                st.info('No new eligible picks to lock. See Why games cannot be locked for the current breakdown. Existing locks remain saved; started games cannot be locked, and other-date games must wait until their game day.')
            st.button('Lock selected picks', key='lock_picks_action', disabled=True)
            return
        selected = st.multiselect('Picks to lock', list(choices), default=list(choices),
            format_func=lambda key: choices[key]['legs'][0]['game'] + ': ' + choices[key]['legs'][0]['pick'] + ' (' + str(choices[key]['legs'][0]['odds']) + ')',
            key='lock_pick_selection')
        if st.button('Lock selected picks', key='lock_picks_action', disabled=not selected):
            try:
                with st.status('Saving locks...', expanded=True) as saving:
                    def show_progress(label, done, total):
                        saving.update(label=f'{label} ({done}/{total})' if total else label)
                    with lock_stage('open_storage'):
                        store = history(setting)
                    store.lock_picks(package, selected, progress=show_progress)
                    saving.update(label='Checking saved locks...')
                    # Fresh authoritative read retains concurrent locks/removals.
                    with lock_stage('verify_lock_history'):
                        locks = store.all('locks')
                    saved['locks'] = locks
                    saving.update(label='Locks saved and verified', state='complete')
                st.success('Your locks are saved. Preparing and publishing the website now.')
                with st.status('Publishing website...', expanded=True) as publishing:
                    with lock_stage('rebuild_results'):
                        saved['rows'] = report(saved['publications'], saved['revisions'], saved.get('imports',[]), locks)
                        from copy import deepcopy
                        from app_core import public_prop_history
                        from app.ui.sftp_publish import publish_action
                        updated=deepcopy(package)
                        updated['results']=current_records(saved['rows']+public_prop_history.report(saved['publications'],saved.get('prop_revisions',[]),saved.get('prop_imports',[])))
                    with lock_stage('publish_website'):
                        notice = publish_action(updated,setting)
                        st.session_state['lock_publish_notice'] = notice
                    publishing.update(label=notice, state='complete' if notice.startswith(('Published:', 'Records saved.')) else 'error')
                st.session_state.pop('publication_preview', None)
                st.session_state['lock_saved_notice'] = True
                st.rerun()
            except (ValueError, RuntimeError):
                st.error('Lock could not complete. Restore history to check saved locks, then rebuild the preview.')
            except Exception:
                st.error('Drive lock save or verification failed. Some selections may have saved; restore history before retrying.')


def render_lock_correction(package, setting, saved):
    from app.ui.public_results import history
    today = datetime.fromisoformat(now()).astimezone(ZoneInfo('America/New_York')).date().isoformat()
    rows = {digest(r):r for r in saved.get('locks',[]) if r['date']==today}
    if not rows:
        return
    with st.expander("Correct today's locks"):
        st.caption('Keep the checked locks. Unchecked locks are archived as removed by owner and excluded from the locked record. This does not cancel bets. Relocking requires a fresh pregame quote.')
        keep = st.multiselect('Locks to keep',list(rows),default=list(rows),
            format_func=lambda key:rows[key]['legs'][0]['game']+': '+rows[key]['legs'][0]['pick'],key='locks_to_keep')
        removed = set(rows)-set(keep)
        st.write(f'{len(keep)} locks kept; {len(removed)} locks will be removed.')
        reason = st.text_input('Correction reason',value='Locked unintentionally before Novig quote audit',key='lock_correction_reason')
        if st.button('Remove unchecked locks and publish',key='remove_locks_action',disabled=not removed or not reason.strip()):
            try:
                store=history(setting)
                saved['locks']=store.remove_locks(removed,reason)
                saved['rows']=report(saved['publications'],saved['revisions'],saved.get('imports',[]),saved['locks'])
                from copy import deepcopy
                from app_core import public_prop_history
                from app.ui.sftp_publish import publish_action
                updated=deepcopy(package)
                updated['results']=current_records(saved['rows']+public_prop_history.report(saved['publications'],saved.get('prop_revisions',[]),saved.get('prop_imports',[])))
                st.session_state['lock_correction_notice']='Lock correction saved. '+publish_action(updated,setting)
                st.session_state.pop('publication_preview',None)
                st.rerun()
            except Exception:
                st.error('Correction or publication could not complete. Restore history to check saved changes before retrying.')


def render_lock_audit(rows, at):
    from collections import Counter
    counts = Counter(r['Lock status'] for r in rows)
    st.caption('Current board: ' + str(len(rows)) + ' rows · ' + ' · '.join(f'{status}: {count}' for status, count in counts.items()))
    with st.expander('Why games cannot be locked', expanded=False):
        st.caption('Checked ' + datetime.fromisoformat(at).astimezone(ZoneInfo('America/New_York')).strftime('%I:%M:%S %p Eastern') + '. One status per board row; existing locks take priority over current price age. Locks from other boards or dates are not included in this breakdown.')
        frame = pd.DataFrame(rows)
        if rows:
            st.dataframe(frame, hide_index=True)
            st.download_button('Download lock eligibility audit', frame.to_csv(index=False).encode('utf-8'),
                               'lock-eligibility-audit.csv', 'text/csv', key='lock_eligibility_audit_download')
