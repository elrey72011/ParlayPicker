"""Explicit locks on the reviewed overall board; no navigation-triggered writes."""
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import streamlit as st
from app_core.locked_picks import lock_candidates, lock_audit
from app_core.quote_freshness import package_age_minutes
from app_core.public_history import now, report, digest, lock_stage
from app_core.public_record import current_records
from app_core.relock_changes import (latest_removed, compare, review_token, acknowledged, requirements,
    validate_candidates, clear_review_state, log_review, RelockReviewExpired, RelockAlreadyLocked)


def render_lock_picks(package, setting):
    from app.ui.public_results import history
    key = 'public_results_' + str(setting('PARLAYPICKER_NETLIFY_SITE_ID')).strip()
    if st.session_state.pop('relock_reset_requested', False):
        clear_review_state(st.session_state)
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
        try:
            if 'lock_removals' not in saved:
                saved['lock_removals'] = history(setting).all('lock_removals')
            removals = saved['lock_removals']
        except Exception:
            st.error('Removed lock history could not be loaded. Retry before locking.')
            if st.button('Retry removed lock history'):
                st.rerun()
            return
        context = digest([package, existing, removals])
        if st.session_state.get('relock_context') != context:
            clear_review_state(st.session_state)
            st.session_state['relock_context'] = context
        for message in st.session_state.pop('changed_relock_notices', []):
            st.success(message)
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
                'Change from prior lock':change_label(removals, r),
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
        st.caption(f'Locked today: {locked_today} Â· Not locked and eligible now: {len(choices)}')
        if not choices:
            clear_review_state(st.session_state)
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
        quality_rows = total_input_review([choices[identity] for identity in selected])
        if quality_rows:
            st.warning('Review total inputs before locking. Research ranking is not wager approval; these warnings do not change the selected pick or its probability.')
            st.dataframe(pd.DataFrame(quality_rows), hide_index=True)
        tokens, changes, ready = {}, [], bool(selected)
        for identity in selected:
            prior = latest_removed(removals, identity)
            if prior:
                change = compare(prior['lock'], choices[identity], prior)
                token = review_token(prior, choices[identity])
                tokens[identity] = token
                changes.append((choices[identity]['legs'][0]['game'], change))
                ready = render_relock_review(change, identity + '_' + token) and ready
        severity = max((c['severity'] for _, c in changes), key=lambda x: ['NORMAL','WARNING','HIGH','CRITICAL'].index(x), default=None)
        label = {'NORMAL':'Re-lock at current price', 'WARNING':'Confirm changed re-lock',
                 'HIGH':'Confirm market-change re-lock', 'CRITICAL':'Re-lock opposite side'}.get(severity, 'Lock selected picks')
        if st.button(label, key='lock_picks_action', disabled=not ready):
            if any(requirements(c)['dialog'] for _, c in changes):
                st.session_state['relock_pending'] = digest([tokens, selected, package])
            else:
                save_reviewed_locks(package, selected, setting, saved, tokens, success_notices(changes), choices, changes)

        if st.session_state.get('relock_pending') == digest([tokens, selected, package]) and changes and ready:
            confirm_relock(package, selected, setting, saved, tokens, changes, choices)


def change_label(removals, row):
    prior = latest_removed(removals, row['id'])
    return compare(prior['lock'], row)['label'] if prior else '—'


def comparison_table(change):
    st.dataframe(pd.DataFrame({'Field': list(change['previous']),
        'PREVIOUS LOCK': [str(v) if v not in (None, '') else 'Not recorded' for v in change['previous'].values()],
        'CURRENT SELECTION': [str(v) if v not in (None, '') else 'Not recorded' for v in change['current'].values()]}), hide_index=True)


def render_relock_review(change, token):
    severity = change['severity']
    title = {'NORMAL':'Fresh price required for re-lock', 'WARNING':'RE-LOCK SELECTION CHANGED',
             'HIGH':'RE-LOCK MARKET CHANGED', 'CRITICAL':'RE-LOCK REVERSES THE SELECTED TEAM'}[severity]
    (st.info if severity == 'NORMAL' else st.warning)(title)
    comparison_table(change)
    if change['flags']['market_favorite_changed']:
        st.write('Market favorite also changed: ' + change['previous']['Market favorite'] + ' → ' + change['current']['Market favorite'])
    body = {
        'NORMAL':'This game was previously locked and later removed. The selection is unchanged, but the current line, price, or quote time may be different.',
        'WARNING':'This game was previously locked with a different selection. The current reviewed board now contains a different selection for this game.',
        'HIGH':'The current best selection is from a different market than the previous lock. This is not a routine price refresh. The current reviewed board selected a different type of market.',
        'CRITICAL':'The current reviewed board selects the opposing team from your previous lock. This is a material change, not a routine quote update. Review the current selection, line, price, analysis time, and quote time before continuing.'}
    st.write(body[severity])
    if len(change['differences']) > 1:
        st.write('MATERIAL RE-LOCK CHANGE')
        st.write('This re-lock differs from the previous lock in ' + str(len(change['differences'])) + ' recorded ways:')
    for field in change['differences']:
        st.write(field + ': ' + str(change['previous'][field]) + ' → ' + str(change['current'][field]))
    st.caption('Re-locking saves the current reviewed selection and quote. The previous removed lock remains archived. This does not place a sportsbook wager.')
    with st.expander('Why did the current pick change?'):
        st.caption('Recorded differences between the two saved analyses. These comparisons do not establish causality.')
        st.write(dict(change['recorded_analysis']) or 'No additional saved analysis values available.')
    checked, typed = False, ''
    if severity != 'NORMAL':
        text = ('I understand that the selected team reversed.' if severity == 'CRITICAL' else
                'I understand that the market changed from ' + change['previous']['Market family'] + ' to ' + change['current']['Market family'] + '.' if severity == 'HIGH' else
                'I understand that this re-lock changes the selected team or side.')
        checked = st.checkbox(text, key='relock_ack_' + token)
    if severity == 'CRITICAL':
        typed = st.text_input('Type RELOCK to confirm', key='relock_type_' + token)
    return acknowledged(change, checked, typed)


def success_notices(changes):
    notices = []
    for game, change in changes:
        title = 'Re-lock saved at the current price' if change['severity'] == 'NORMAL' else 'Changed re-lock saved'
        notices.append(title + ': ' + game + ' is now locked as ' + change['current']['Pick'] +
            ' (' + str(change['current']['Odds']) + '). Previous removed lock: ' + change['previous']['Pick'] +
            ' (' + str(change['previous']['Odds']) + '). New lock saved: ' + change['current']['Pick'] +
            ' (' + str(change['current']['Odds']) + '). The original lock remains preserved in history.')
    return notices


@st.dialog('Confirm re-lock')
def confirm_relock(package, selected, setting, saved, tokens, changes, reviewed):
    for game, change in changes:
        st.write('You are saving a new lock for: ' + game)
        st.write('Previous removed lock: ' + change['previous']['Pick'] + ' (' + str(change['previous']['Odds']) + ')')
        st.write('New lock: ' + change['current']['Pick'] + ' (' + str(change['current']['Odds']) + ')')
        comparison_table(change)
    st.write('The previous record remains archived. This action saves the current selection and quote and publishes the updated locked board. This does not place a sportsbook wager.')
    if st.button('Cancel', key='relock_cancel', type='primary'):
        for _, change in changes:
            log_review(change['lock_id'], change, 'cancelled')
        st.session_state['relock_reset_requested'] = True
        st.rerun()
    if st.button('Confirm and save new lock', key='relock_confirm'):
        st.session_state.pop('relock_pending', None)
        save_reviewed_locks(package, selected, setting, saved, tokens, success_notices(changes), reviewed, changes)


def save_reviewed_locks(package, selected, setting, saved, tokens, notices, reviewed, changes):
    from app.ui.public_results import history
    try:
        validate_candidates(package, selected, reviewed, now())
        for _, change in changes:
            log_review(change['lock_id'], change, 'confirmed')
        with st.status('Saving locks...', expanded=True) as saving:
            def show_progress(label, done, total):
                saving.update(label=f'{label} ({done}/{total})' if total else label)
            with lock_stage('open_storage'):
                store = history(setting)
            store.lock_picks(package, selected, progress=show_progress, relock_review=tokens)
            saving.update(label='Checking saved locks...')
            # Fresh authoritative read retains concurrent locks/removals.
            with lock_stage('verify_lock_history'):
                locks = store.all('locks')
            saved['locks'] = locks
            saving.update(label='Locks saved and verified', state='complete')
        st.session_state['relock_reset_requested'] = True
        for _, change in changes:
            log_review(change['lock_id'], change, 'saved')
        st.session_state['changed_relock_notices'] = notices
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
    except RelockReviewExpired as exc:
        saved.pop('lock_removals', None)
        st.session_state['relock_reset_requested'] = True
        if str(exc) == 'Candidate changed':
            st.error('Re-lock changed while you were reviewing it. The current selection or quote changed before the lock was saved. Nothing was locked. Refresh the preview and review the updated selection again.')
        else:
            st.error('Re-lock review is no longer current. The game, selection, quote, or lock history changed while this screen was open. Nothing was saved. Refresh the preview and review the current state again.')
        for _, change in changes:
            log_review(change['lock_id'], change, 'review_expired')
    except RelockAlreadyLocked as exc:
        saved.pop('lock_removals', None)
        st.session_state['relock_reset_requested'] = True
        st.error('This game was already locked elsewhere. Restore lock history to review the saved selection. No additional lock was created for that game.')
        if exc.after_write:
            st.warning('Other selections in this batch may have saved. Restore history before retrying.')
        for _, change in changes:
            log_review(change['lock_id'], change, 'concurrent_lock')
    except (ValueError, RuntimeError):
        for _, change in changes:
            log_review(change['lock_id'], change, 'save_failed')
        saved.pop('lock_removals', None)
        st.error('Lock could not complete. Restore history to check saved locks, then rebuild the preview.')
    except Exception:
        for _, change in changes:
            log_review(change['lock_id'], change, 'save_failed')
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
                saved.pop('lock_removals', None)
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
    st.caption('Current board: ' + str(len(rows)) + ' rows Â· ' + ' Â· '.join(f'{status}: {count}' for status, count in counts.items()))
    with st.expander('Why games cannot be locked', expanded=False):
        st.caption('Checked ' + datetime.fromisoformat(at).astimezone(ZoneInfo('America/New_York')).strftime('%I:%M:%S %p Eastern') + '. One status per board row; existing locks take priority over current price age. Locks from other boards or dates are not included in this breakdown.')
        frame = pd.DataFrame(rows)
        if rows:
            st.dataframe(frame, hide_index=True)
            st.download_button('Download lock eligibility audit', frame.to_csv(index=False).encode('utf-8'),
                               'lock-eligibility-audit.csv', 'text/csv', key='lock_eligibility_audit_download')


def total_input_review(records):
    """Display only immutable saved labels; never infer missing historical inputs."""
    reasons = {'missing_theover': 'TheOver input missing',
               'stale_empirical_evidence': 'Empirical evidence stale',
               'missing_target_model': 'Matching total model missing',
               'degraded_feature_subset': 'Reduced model feature set'}
    rows = []
    for record in records:
        for leg in record.get('legs', []):
            if leg.get('sport') != 'MLB' or not str(leg.get('market', '')).startswith('total_'):
                continue
            recorded = leg.get('total_input_version') == 'mlb-total-inputs-v1'
            status = leg.get('total_input_status') if recorded else 'Not recorded'
            if status == 'COMPLETE':
                continue
            codes = str(leg.get('total_input_reason_codes') or '').split('|') if recorded else []
            rows.append({'Game': leg.get('game', ''), 'Pick': leg.get('pick', ''),
                         'Saved total-input status': status,
                         'Recorded input warnings': '; '.join(reasons.get(c, c) for c in codes if c) or 'Not recorded'})
    return rows
