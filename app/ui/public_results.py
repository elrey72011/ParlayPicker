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
                pubs=store.publications();revisions=store.all('scores')
                st.session_state[key]={'publications':pubs,'revisions':revisions,'rows':report(pubs,revisions)}
                st.success('Public history restored.')
            except Exception:
                st.error('Public history restore failed. Check Shared Drive access; no history was replaced.')
        saved=st.session_state.get(key)
        if saved is None:
            st.info('Restore history before publishing so the public tracker includes its existing record.')
            return None
        day=st.date_input('Public results date',value=datetime.now(ZoneInfo('America/New_York')).date()-timedelta(days=1),key='public_results_day')
        if st.button('Grade published picks for selected date',key='public_history_grade'):
            try:
                entries=[r for r in selections(saved['publications']) if r['date']==day.isoformat()]
                sports={leg['sport'] for r in entries for leg in r['legs']}
                if not entries:
                    st.info('No eligible confirmed publications for this date.')
                else:
                    revision=fetch_scores(day,sports)
                    history(setting).put('scores/'+digest(revision)+'.json',revision)
                    saved['revisions'].append(revision)
                    saved['rows']=report(saved['publications'],saved['revisions'])
                    st.success('Final scores saved to Drive. Build a new preview and publish to update the website.')
            except Exception:
                st.error('Grading or backup failed. Existing results remain available; retry explicitly.')
        st.caption(f"{len(saved['rows'])} tracked category entries. Props and DFS are not included in this tracker.")
        if saved['rows']:
            st.dataframe(saved['rows'],hide_index=True)
        return saved['rows']
