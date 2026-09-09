"""Owner-triggered Netlify deployment controls; no automatic uploads."""
import streamlit as st
from app_core import netlify_publishing as remote


def render_remote_publish(package, fingerprint, setting):
    st.subheader('Publish to public website')
    site_id=str(setting('PARLAYPICKER_NETLIFY_SITE_ID')).strip()
    token=str(setting('PARLAYPICKER_NETLIFY_TOKEN')).strip()
    if not site_id or not token:
        st.info('Public hosting is not configured. Add PARLAYPICKER_NETLIFY_SITE_ID and PARLAYPICKER_NETLIFY_TOKEN to Streamlit secrets. Local preview and downloads remain available.')
        return
    # Reverify when either the site or credential changes, without storing the token.
    import hashlib
    config_hash=hashlib.sha256((site_id+'\0'+token).encode()).hexdigest()
    site=st.session_state.get('publication_remote_site')
    if site and site.get('config_hash')!=config_hash:
        site=None
        st.session_state.pop('publication_remote_site',None)
    if st.button('Verify public destination',key='publication_remote_verify'):
        try:
            site={**remote.site_info(site_id,token),'config_hash':config_hash}
            st.session_state['publication_remote_site']=site
        except (ValueError,RuntimeError) as exc:
            site=None
            st.session_state.pop('publication_remote_site',None)
            st.error(str(exc))
    if not site:
        return
    st.write('Public destination: '+site['url'])
    st.caption('Publishing replaces this website with the reviewed board. Anyone with its URL can read all included picks, props, and lineups. This is not a subscription paywall. Netlify usage is subject to your account plan.')
    key=fingerprint+':'+config_hash
    jobs=st.session_state.setdefault('publication_remote_jobs',{})
    job=jobs.get(key)
    if st.button('Publish reviewed board publicly',key='publication_remote_publish',disabled=job is not None) and job is None:
        # Record submission before sending; uncertain responses must not be retried automatically.
        jobs[key]={'state':'uncertain'}
        try:
            from app.ui.public_results import history
            archive_hash=history(setting).archive(package)
            jobs[key]=remote.deploy(package,site_id,token)
            jobs[key]['archive_hash']=archive_hash
            history(setting).submitted(jobs[key]['id'],archive_hash)
        except Exception:
            jobs[key]['message']='Publication or history backup failed; check Drive and Netlify before retrying.'
        st.rerun()
    if not job:
        return
    if job.get('id') and st.button('Check public deployment status',key='publication_remote_status'):
        try:
            updated=remote.deployment_status(job['id'],site_id,token)
            jobs[key]={**job, **updated}
            job=jobs[key]
        except (ValueError,RuntimeError) as exc:
            st.error(str(exc))
    if job['state']=='ready':
        if not job.get('history_saved'):
            try:
                from app.ui.public_results import history
                history(setting).confirm(job['id'],job['archive_hash'])
                job['history_saved']=True
            except Exception:
                st.error('Website is published, but recording its public history failed. Check Drive access and retry status verification; it is not counted yet.')
        st.success('Netlify confirms this deployment is the published website.')
        st.link_button('Open public website',job['url'])
    elif job['state']=='uncertain':
        st.warning(job.get('message','Submission outcome is unknown.')+' Check the Netlify dashboard before trying again.')
    else:
        st.info('Deployment status: '+job['state']+'. Use Check public deployment status to refresh.')
    if job['state'] in {'uncertain','error'}:
        if st.checkbox('I checked Netlify and want to allow another attempt',key='publication_remote_retry_ack'):
            if st.button('Allow another publish attempt',key='publication_remote_retry'):
                jobs.pop(key,None)
                st.rerun()
