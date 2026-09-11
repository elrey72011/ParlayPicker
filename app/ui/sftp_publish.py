"""One-click Namecheap publication with durable receipts and explicit recovery."""
import hashlib
import json
import streamlit as st
from app_core import sftp_publishing as remote


def board_signature(package):
    return hashlib.sha256(json.dumps({k:v for k,v in package.items() if k != 'built_at'},sort_keys=True).encode()).hexdigest()


def publish_once(package, config, store, jobs):
    """Never resubmit an uncertain upload, including after the preview changes."""
    config_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    content,deploy_id=remote.prepare(package)
    key=deploy_id+':'+config_hash
    for existing in jobs.values():
        if existing.get('config_hash')==config_hash and not existing.get('history_saved'):
            return existing
    if key in jobs:
        old=jobs[key]
        try:
            old.update(remote.deployment_status(deploy_id,config))
            if old['state']!='ready':
                old['history_saved']=False
        except Exception:
            old['state']='uncertain'
            old['history_saved']=False
        return old
    job=jobs[key]={'state':'uncertain','config_hash':config_hash,'id':deploy_id,'signature':board_signature(package),
        'locked_ids':[r['id'] for r in package.get('results',[]) if r['group']=='Locked']}
    try:
        remote.site_info(config)
        archive_hash=store.archive(package)
        store.submitted(deploy_id,archive_hash)
        job['archive_hash']=archive_hash
        job.update(remote.deploy(content,deploy_id,config))
        job.update(remote.deployment_status(deploy_id,config))
        if job['state']=='ready':
            store.confirm(deploy_id,archive_hash)
            job['history_saved']=True
    except Exception:
        job['message']='Saved picks are retained. Publication needs attention; check status before retrying.'
    return job


def publish_action(package, setting):
    if str(setting('PARLAYPICKER_PUBLIC_PROVIDER') or 'netlify').strip().lower()!='sftp':
        return 'Records saved. Use the public hosting controls below to publish this board.'
    try:
        from app.ui.public_results import history
        job=publish_once(package,remote.configuration(setting),history(setting),st.session_state.setdefault('sftp_jobs',{}))
        if job.get('history_saved'):
            return 'Published: the HTTPS website matches the board and history is saved.'
        return job.get('message','Publication needs attention. Check status below; no automatic retry will occur.')
    except Exception:
        return 'Publication could not start. Check the hosting configuration below; saved picks are retained.'


def render_sftp_publish(package, fingerprint, setting):
    st.subheader('Website publication')
    try:
        config=remote.configuration(setting)
    except ValueError as exc:
        st.info(str(exc));return
    st.write('Destination: '+config['url'])
    jobs=st.session_state.setdefault('sftp_jobs',{})
    config_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    relevant=[j for j in jobs.values() if j.get('config_hash')==config_hash]
    unresolved=next((j for j in relevant if not j.get('history_saved')),None)
    latest=relevant[-1] if relevant else None
    current=bool(latest and latest.get('history_saved') and latest.get('signature')==board_signature(package))
    st.caption('Published — current board' if current else 'Not published — this preview has changes or has not been verified in this session.')
    if st.button('Publish board',key='sftp_publish',disabled=unresolved is not None or current):
        st.session_state['publish_notice']=publish_action(package,setting)
        st.rerun()
    notice=st.session_state.pop('publish_notice',None)
    if notice:st.info(notice)
    st.link_button('Open public website',config['url'])
    if unresolved:
        st.warning(unresolved.get('message','Publication needs verification.'))
        if st.button('Check public status',key='sftp_status'):
            try:
                from app.ui.public_results import history
                unresolved.update(remote.deployment_status(unresolved['id'],config))
                if unresolved['state']=='ready' and unresolved.get('archive_hash'):
                    history(setting).confirm(unresolved['id'],unresolved['archive_hash'])
                    unresolved['history_saved']=True
                    st.rerun()
                st.info('Website status: '+unresolved['state'])
            except Exception:
                st.error('Could not verify the website. No upload was retried.')
        if st.checkbox('I checked the public page and want to allow another attempt',key='sftp_retry_ack'):
            if st.button('Allow another publish attempt',key='sftp_retry'):
                for key in list(jobs):
                    if jobs[key] is unresolved:del jobs[key]
                st.rerun()
