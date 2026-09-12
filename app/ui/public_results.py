"""Owner history loading, consolidated grading, and advanced recovery controls."""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import streamlit as st
from app_core.public_history import History, selections, report, fetch_scores, digest
from app_core import public_record
from app_core.public_record import current_records


def history(setting):
    from app_core.evidence_config import EvidenceConfigurationError
    from app_core.netlify_publishing import identifier
    site = str(setting('PARLAYPICKER_NETLIFY_SITE_ID') or '').strip()
    folder = str(setting('PARLAYPICKER_DRIVE_FOLDER_ID') or '').strip()
    try:
        identifier(site)
    except ValueError:
        raise EvidenceConfigurationError('Restore the original PARLAYPICKER_NETLIFY_SITE_ID in Streamlit Secrets. It remains the history identifier after switching to SFTP; do not replace it with the domain.') from None
    if not folder:
        raise EvidenceConfigurationError('PARLAYPICKER_DRIVE_FOLDER_ID is missing. Restore the original Shared Drive folder ID in Streamlit Secrets.')
    return History(site, folder)


def restore_history(setting):
    site=str(setting("PARLAYPICKER_NETLIFY_SITE_ID")).strip()
    key="public_results_"+site
    stage = 'opening history storage'
    try:
        store=history(setting)
        stage = 'recovering unconfirmed publications'
        # Recover known deployments after a Streamlit restart. Only the
        # site's currently published deployment can be confirmed.
        from app_core.netlify_publishing import deployment_status
        token=str(setting('PARLAYPICKER_NETLIFY_TOKEN')).strip()
        for pending in store.all('deployments'):
            try:
                store.read('confirmed/'+pending['deploy_id']+'.json')
            except Exception:
                status = None
                try:
                    if pending['deploy_id'].startswith('sftp-'):
                        from app_core import sftp_publishing
                        status=sftp_publishing.deployment_status(pending['deploy_id'],sftp_publishing.configuration(setting))
                    elif token:
                        status=deployment_status(pending['deploy_id'],site,token)
                except (ValueError,RuntimeError):
                    st.warning('An unconfirmed hosting publication could not be verified. Continuing to restore confirmed history; the unverified publication will not be counted.')
                if status and status['state']=='ready':
                    # Storage/integrity failures must still stop restore.
                    store.confirm(pending['deploy_id'],pending['package_hash'])
        stage = 'reading saved publications and results'
        pubs=store.publications();revisions=store.all('scores');imports=store.all('imports');locks=store.all('locks')
        st.session_state[key]={'publications':pubs,'revisions':revisions,'imports':imports,'grading_runs':store.all('grading_runs'),'prop_revisions':store.all('prop_stats'),'prop_imports':store.all('prop_imports'),'locks':locks,'rows':report(pubs,revisions,imports,locks)}
        st.success('Public history restored.')
        return True
    except Exception as exc:
        from app_core.evidence_config import safe_error
        detail = safe_error(exc, 'History restore while ' + stage)
        st.error('Public history restore failed while ' + stage + '. ' + detail + ' No saved history was replaced.')
        return False


def render_history(setting):
    site=str(setting('PARLAYPICKER_NETLIFY_SITE_ID')).strip()
    key='public_results_'+site
    attempt='history_restore_attempt_'+site
    refresh_requested=st.session_state.pop('history_refresh_requested',False)
    if refresh_requested or (key not in st.session_state and not st.session_state.get(attempt)):
        st.session_state[attempt]=True
        with st.spinner('Loading saved picks and results...'):
            restore_history(setting)
    saved=st.session_state.get(key)
    st.caption('Public record starts September 11, 2026 (Eastern) for games and player props. Earlier history remains archived.')
    if st.button('Update results and publish',key='public_update_all',disabled=saved is None):
        try:
            if not restore_history(setting):
                raise RuntimeError('History unavailable')
            saved=st.session_state[key]
            update_pending_results(setting,saved)
            st.session_state['publish_results_requested']=True
            st.success('Results saved. Preparing the updated website.')
        except Exception:
            st.error('Results update did not complete. Saved records are retained; no website upload was requested.')
    # Collapsing a Streamlit expander alone does not defer its Python work.
    # Keep detailed imports/grading tables off the normal path to locking.
    saved=st.session_state.get(key)
    if not st.checkbox('Show history, imports and individual grading', key='public_history_tools'):
        if saved is None:
            st.info('History is unavailable. Open history tools to retry restoring it.')
            return None
        from app_core import public_prop_history as prop_history
        return current_records(saved['rows']+prop_history.report(saved['publications'],saved.get('prop_revisions',[]),saved.get('prop_imports',[])))
    with st.expander('History, imports and individual grading', expanded=True):
        st.caption('Saved history loads automatically once per session. Use Update results and publish for outstanding game and MLB prop results. MLB collection uses bounded batches; unresolved entries remain pending or need review. Imports and individual grading are available below.')
        if st.button('Restore public history from Drive',key='public_history_restore'):
            restore_history(setting)
        saved=st.session_state.get(key)
        if saved is None:
            st.info('Restore history before publishing so the public tracker includes its existing record.')
            return None
        runs=saved.get('grading_runs',[])
        if runs:
            latest=max(runs,key=lambda r:r['started_at'])
            st.caption('Automatic grading: '+latest['status']+' · Last successful run: '+str(latest.get('last_success_at') or 'None')+' · Pending category entries: '+str(latest.get('pending',0)))
            if latest.get('errors'):
                st.warning('Automatic grading needs attention: '+', '.join(latest['errors']))
            st.caption('Status reflects the last Drive restore. Restore again to retrieve newer runs. Publishing remains manual.')
        else:
            st.info('No automatic public-grading run restored yet. Configure PARLAYPICKER_NETLIFY_SITE_ID in GitHub Actions variables; manual grading remains available.')
        st.markdown('**Import an older recap**')
        st.caption('Upload all three per-game CSVs from the same run. They remain Imported research, never verified public predictions or approved betting returns. Duplicate imports do not increase counts.')
        uploads=[st.file_uploader(label,type=['csv'],key='recap_import_'+family) for family,label in [('overall','Overall per-game CSV'),('sides','Sides per-game CSV'),('totals','Totals per-game CSV')]]
        if st.button('Import historical recap to Drive',key='public_history_import',disabled=not all(x is not None for x in uploads)):
            try:
                import io
                import pandas as pd
                from app_core.imported_recaps import import_exports
                batch=import_exports(*(pd.read_csv(io.BytesIO(f.getvalue())) for f in uploads))
                history(setting).put('imports/'+batch['id']+'.json',batch)
                imports=saved.setdefault('imports',[])
                if not any(x['id']==batch['id'] for x in imports):imports.append(batch)
                saved['rows']=report(saved['publications'],saved['revisions'],imports,saved.get('locks',[]))
                st.success('Historical recap saved as Imported research. Select its game date below and grade it.')
            except ValueError as exc:
                st.error('Import rejected: '+str(exc))
            except Exception:
                st.error('Import or Drive backup failed. Existing history is unchanged.')
        from app_core import public_prop_history as prop_history
        prop_rows=prop_history.report(saved['publications'],saved.get('prop_revisions',[]),saved.get('prop_imports',[]))
        dates=sorted({r['date'] for r in saved['rows']+prop_rows})
        st.caption('Available game dates: '+(', '.join(dates) if dates else 'None yet'))
        day=st.date_input('Public results date',value=datetime.now(ZoneInfo('America/New_York')).date()-timedelta(days=1),key='public_results_day')
        if st.button('Grade picks for selected date',key='public_history_grade'):
            try:
                from app_core.imported_recaps import imported_selections
                from app_core.locked_picks import locked_selections
                entries=[r for r in selections(saved['publications'])+imported_selections(saved.get('imports',[]))+locked_selections(saved.get('locks',[])) if r['date']==day.isoformat()]
                sports={leg['sport'] for r in entries for leg in r['legs']}
                if not entries:
                    st.info('No eligible records for this date. Check the available dates above or import the historical CSVs.')
                else:
                    revision=fetch_scores(day,sports)
                    history(setting).put('scores/'+digest(revision)+'.json',revision)
                    saved['revisions'].append(revision)
                    saved['rows']=report(saved['publications'],saved['revisions'],saved.get('imports',[]),saved.get('locks',[]))
                    st.success('Final scores saved to Drive. Build a new preview and publish to update the website.')
            except Exception:
                st.error('Grading or backup failed. Existing results remain available; retry explicitly.')
        render_prop_history(setting,saved,day)
        prop_rows=prop_history.report(saved['publications'],saved.get('prop_revisions',[]),saved.get('prop_imports',[]))
        st.caption(f"{len(saved['rows'])} game category entries and {len(prop_rows)} MLB prop entries. Automatic grading status above covers games; MLB props use the explicit grading button. DFS is excluded.")
        review_rows=[r for r in prop_rows if r['outcome']=='NEEDS_REVIEW' and r['date']==day.isoformat()]
        if review_rows:
            with st.expander(f'MLB props needing review for {day}: {len(review_rows)}',expanded=True):
                st.caption('Excluded from wins, losses and win rate. These are not confirmed sportsbook voids. Recheck only when new stats or matching evidence is available.')
                st.dataframe(review_rows,hide_index=True)
        if saved['rows']+prop_rows:
            st.dataframe(saved['rows']+prop_rows,hide_index=True)
        return current_records(saved['rows']+prop_rows)


def render_prop_history(setting,saved,day):
    from app_core import public_prop_history as props
    st.markdown('**MLB player-prop results**')
    archived=sum(leg.get('sport','').upper()=='MLB' for pub in saved['publications'] for leg in pub['package'].get('props',[]))
    tracked=len(props.selections(saved['publications']))
    st.caption(f'{archived} archived MLB prop rows across publications; {tracked} unique eligible published props. Start times may be recovered from a matching game in the same saved analysis.')
    if archived and not tracked:
        st.info('Archived props were found, but none meet the supported-market, matching start-time and fresh pregame publication requirements. Postgame publications cannot be retroactively verified. An original historical export can be imported separately as research.')
    elif not archived:
        st.info('No MLB props were included in the restored publications. For older picks, import the original combined prop export.')
    st.caption('Published props use their original pregame record. Missing stats, absent appearances and ambiguous matches are labeled Needs review, separate from games awaiting results. MLB hits, total bases, strikeouts, walks and outs are supported; other leagues are not graded here.')
    uploaded=st.file_uploader('Original combined player-prop CSV (optional historical import)',type=['csv'],key='public_prop_import_file')
    if st.button('Import historical MLB props to Drive',disabled=uploaded is None,key='public_prop_import'):
        try:
            import io
            import pandas as pd
            batch=props.import_export(pd.read_csv(io.BytesIO(uploaded.getvalue())))
            history(setting).put('prop_imports/'+batch['id']+'.json',batch)
            imports=saved.setdefault('prop_imports',[])
            if not any(b['id']==batch['id'] for b in imports):imports.append(batch)
            count=len(props.selections([], [batch]))
            st.success(f'{count} supported MLB props saved as Imported research. Unsupported rows or rows without original pregame timestamps are excluded.')
        except ValueError:
            st.error('Import requires one original MLB prop export with player, market, pick, matchup, odds, export_run_id and timezone-aware game start times. No dates or grades are inferred.')
        except Exception:
            st.error('Prop import or backup failed. Existing records remain available.')
    st.caption('Manual collection: one MLB schedule request plus at most 10 final-game boxscores per click. No CFBD or Odds API requests. Repeat if more completed games remain; settlement corrections can be requested explicitly.')
    reviews=st.checkbox('Recheck props needing review',key='public_props_review')
    corrections=st.checkbox('Recheck already settled props for score corrections',key='public_props_recheck')
    if st.button('Grade MLB props for selected date',key='public_props_grade'):
        try:
            entries=props.selections(saved['publications'],saved.get('prop_imports',[]))
            rows=props.report(saved['publications'],saved.get('prop_revisions',[]),saved.get('prop_imports',[]))
            pending={r['id'] for r in rows if r['outcome']=='PENDING' or (reviews and r['outcome']=='NEEDS_REVIEW') or (corrections and r['outcome'] in {'WIN','LOSS','PUSH'})}
            selected=[e for e in entries if e['date']==day.isoformat() and e['id'] in pending]
            if not selected:
                st.info('No matching MLB props need grading for this date.')
            else:
                checked={}
                for old in sorted(saved.get('prop_revisions',[]),key=lambda r:r['recorded_at']):
                    for entry_id in old.get('checked',[]):checked[entry_id]=old['recorded_at']
                selected.sort(key=lambda e:(checked.get(e['id'],''),e['id']))
                revision=props.fetch_actuals(day,selected)
                if revision['actuals'] or revision.get('checked'):
                    history(setting).put('prop_stats/'+digest(revision)+'.json',revision)
                    saved.setdefault('prop_revisions',[]).append(revision)
                    st.success(f"Saved {len(revision['actuals'])} prop statistics to Drive. Build and publish a fresh preview.")
                    if revision.get('unresolved'):
                        from collections import Counter
                        reasons=Counter(item['reason'] for item in revision['unresolved'])
                        st.info('Unresolved results: '+ '; '.join(f'{reason}: {count}' for reason,count in reasons.items()))
                else:
                    st.info('No unambiguous final player statistics found. Props remain pending.')
        except Exception:
            st.error('MLB prop grading or backup failed. Existing results remain available; retry explicitly.')


def update_pending_results(setting,saved):
    """Grade outstanding game records by date; persist each successful revision."""
    from app_core.imported_recaps import imported_selections
    from app_core.locked_picks import locked_selections
    entries=selections(saved['publications'])+imported_selections(saved.get('imports',[]))+locked_selections(saved.get('locks',[]))
    pending={r['id'] for r in saved['rows'] if r['outcome']=='PENDING' and r['date']>=public_record.START_DATE}
    today=datetime.now(ZoneInfo('America/New_York')).date().isoformat()
    dates=sorted({r['date'] for r in entries if r['id'] in pending and r['date']<=today})
    store=history(setting)
    for day in dates:
        sports={leg['sport'] for r in entries if r['date']==day and r['id'] in pending for leg in r['legs']}
        revision=fetch_scores(datetime.fromisoformat(day).date(),sports)
        store.put('scores/'+digest(revision)+'.json',revision)
        saved['revisions'].append(revision)
        saved['rows']=report(saved['publications'],saved['revisions'],saved.get('imports',[]),saved.get('locks',[]))

    from app_core import public_prop_history as props
    prop_entries=props.selections(saved['publications'],saved.get('prop_imports',[]))
    prop_rows=props.report(saved['publications'],saved.get('prop_revisions',[]),saved.get('prop_imports',[]))
    prop_pending={r['id'] for r in prop_rows if r['outcome']=='PENDING' and r['date']>=public_record.START_DATE}
    for day in sorted({r['date'] for r in prop_entries if r['id'] in prop_pending and r['date']<=today}):
        selected=[r for r in prop_entries if r['id'] in prop_pending and r['date']==day]
        checked={entry_id:r['recorded_at'] for r in sorted(saved.get('prop_revisions',[]),key=lambda r:r['recorded_at']) for entry_id in r.get('checked',[])}
        selected.sort(key=lambda e:(checked.get(e['id'],''),e['id']))
        revision=props.fetch_actuals(datetime.fromisoformat(day).date(),selected)
        if revision['actuals'] or revision.get('checked'):
            store.put('prop_stats/'+digest(revision)+'.json',revision)
            saved.setdefault('prop_revisions',[]).append(revision)
