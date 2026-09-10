"""Owner-only, explicit Drive restore and one-day public-result grading."""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import streamlit as st
from app_core.public_history import History, selections, report, fetch_scores, digest


def history(setting):
    return History(str(setting('PARLAYPICKER_NETLIFY_SITE_ID')).strip(), str(setting('PARLAYPICKER_DRIVE_FOLDER_ID')).strip())


def render_history(setting):
    site=str(setting('PARLAYPICKER_NETLIFY_SITE_ID')).strip()
    key='public_results_'+site
    with st.expander('Public results history', expanded=False):
        st.caption('Only confirmed, fresh pregame publications count. Restore from Drive, grade a date, then build and publish a new preview to update the public tracker. These controls do not run during navigation.')
        if st.button('Restore public history from Drive',key='public_history_restore'):
            try:
                store=history(setting)
                # Recover known deployments after a Streamlit restart. Only the
                # site's currently published deployment can be confirmed.
                from app_core.netlify_publishing import deployment_status
                token=str(setting('PARLAYPICKER_NETLIFY_TOKEN')).strip()
                for pending in store.all('deployments'):
                    try:
                        store.read('confirmed/'+pending['deploy_id']+'.json')
                    except Exception:
                        if token:
                            status=deployment_status(pending['deploy_id'],site,token)
                            if status['state']=='ready':
                                store.confirm(pending['deploy_id'],pending['package_hash'])
                pubs=store.publications();revisions=store.all('scores');imports=store.all('imports')
                st.session_state[key]={'publications':pubs,'revisions':revisions,'imports':imports,'grading_runs':store.all('grading_runs'),'prop_revisions':store.all('prop_stats'),'prop_imports':store.all('prop_imports'),'rows':report(pubs,revisions,imports)}
                st.success('Public history restored.')
            except Exception:
                st.error('Public history restore failed. Check Shared Drive access; no history was replaced.')
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
                saved['rows']=report(saved['publications'],saved['revisions'],imports)
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
                entries=[r for r in selections(saved['publications'])+imported_selections(saved.get('imports',[])) if r['date']==day.isoformat()]
                sports={leg['sport'] for r in entries for leg in r['legs']}
                if not entries:
                    st.info('No eligible records for this date. Check the available dates above or import the historical CSVs.')
                else:
                    revision=fetch_scores(day,sports)
                    history(setting).put('scores/'+digest(revision)+'.json',revision)
                    saved['revisions'].append(revision)
                    saved['rows']=report(saved['publications'],saved['revisions'],saved.get('imports',[]))
                    st.success('Final scores saved to Drive. Build a new preview and publish to update the website.')
            except Exception:
                st.error('Grading or backup failed. Existing results remain available; retry explicitly.')
        render_prop_history(setting,saved,day)
        prop_rows=prop_history.report(saved['publications'],saved.get('prop_revisions',[]),saved.get('prop_imports',[]))
        st.caption(f"{len(saved['rows'])} game category entries and {len(prop_rows)} MLB prop entries. Automatic grading status above covers games; MLB props use the explicit grading button. DFS is excluded.")
        if saved['rows']+prop_rows:
            st.dataframe(saved['rows']+prop_rows,hide_index=True)
        return saved['rows']+prop_rows


def render_prop_history(setting,saved,day):
    from app_core import public_prop_history as props
    st.markdown('**MLB player-prop results**')
    st.caption('Published props use their original pregame record. Missing stats, DNPs and ambiguous matches remain pending. MLB hits, total bases, strikeouts, walks and outs are supported; other leagues are not graded here.')
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
    corrections=st.checkbox('Recheck already settled props for score corrections',key='public_props_recheck')
    if st.button('Grade MLB props for selected date',key='public_props_grade'):
        try:
            entries=props.selections(saved['publications'],saved.get('prop_imports',[]))
            rows=props.report(saved['publications'],saved.get('prop_revisions',[]),saved.get('prop_imports',[]))
            pending={r['id'] for r in rows if r['outcome']=='PENDING'}
            selected=[e for e in entries if e['date']==day.isoformat() and (corrections or e['id'] in pending)]
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
                else:
                    st.info('No unambiguous final player statistics found. Props remain pending.')
        except Exception:
            st.error('MLB prop grading or backup failed. Existing results remain available; retry explicitly.')
