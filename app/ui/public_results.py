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
                st.session_state[key]={'publications':pubs,'revisions':revisions,'imports':imports,'rows':report(pubs,revisions,imports)}
                st.success('Public history restored.')
            except Exception:
                st.error('Public history restore failed. Check Shared Drive access; no history was replaced.')
        saved=st.session_state.get(key)
        if saved is None:
            st.info('Restore history before publishing so the public tracker includes its existing record.')
            return None
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
        dates=sorted({r['date'] for r in saved['rows']})
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
        st.caption(f"{len(saved['rows'])} tracked category entries. Props and DFS are not included in this tracker.")
        if saved['rows']:
            st.dataframe(saved['rows'],hide_index=True)
        return saved['rows']
