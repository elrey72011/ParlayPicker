"""Explicit owner controls for Namecheap publication, with durable receipts."""
import hashlib
import json
import streamlit as st
from app_core import sftp_publishing as remote


def render_sftp_publish(package, fingerprint, setting):
    st.subheader('Publish to Namecheap')
    try:
        config = remote.configuration(setting)
    except ValueError as exc:
        st.info(str(exc))
        return
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    verified = st.session_state.get('sftp_verified') == config_hash
    if st.button('Verify public destination', key='sftp_verify'):
        st.session_state.pop('sftp_verified', None)
        verified = False
        try:
            remote.site_info(config)
            st.session_state['sftp_verified'] = config_hash
            verified = True
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
    if not verified:
        return
    st.write('Public destination: ' + config['url'])
    st.caption('Publishing replaces index.html in the dedicated subdomain folder. Anyone with the URL can read the included board.')
    jobs = st.session_state.setdefault('sftp_jobs', {})
    key = fingerprint + ':' + config_hash
    job = jobs.get(key)
    if st.button('Publish reviewed board publicly', key='sftp_publish', disabled=job is not None):
        from app.ui.public_results import history
        job = jobs[key] = {'state': 'uncertain'}
        try:
            content, deploy_id = remote.prepare(package)
            store = history(setting)
            archive_hash = store.archive(package)
            # Save recovery information BEFORE changing the live page.
            store.submitted(deploy_id, archive_hash)
            job.update(id=deploy_id, archive_hash=archive_hash)
            job.update(remote.deploy(content, deploy_id, config))
        except Exception:
            job['message'] = 'Publication or Drive backup failed. Check public status before retrying.'
        st.rerun()
    if not job:
        return
    if job.get('id') and st.button('Check public deployment status', key='sftp_status'):
        try:
            job.update(remote.deployment_status(job['id'], config))
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
    if job['state'] == 'ready':
        try:
            from app.ui.public_results import history
            if not job.get('history_saved'):
                history(setting).confirm(job['id'], job['archive_hash'])
                job['history_saved'] = True
            st.success('The HTTPS website matches this reviewed board and its publication is recorded.')
            st.link_button('Open public website', config['url'])
        except Exception:
            st.error('Website verified, but Drive confirmation failed. Retry status verification to record history.')
    else:
        st.info(job.get('message', 'Publication status: ' + job['state'] + '. Check public deployment status to verify the exact page.'))
        if st.checkbox('I checked the public page and want to allow another attempt', key='sftp_retry_ack'):
            if st.button('Allow another publish attempt', key='sftp_retry'):
                jobs.pop(key, None)
                st.rerun()
