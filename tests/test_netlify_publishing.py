import io
import json
import zipfile
from types import SimpleNamespace
import pytest
import requests
from app_core import netlify_publishing as remote


def package():
    return {'schema_version':1,'built_at':'2026-09-08T20:00:00Z','stale_after_minutes':15,
            'games':{'overall':[],'sides':[],'totals':[]},'props':[],'dfs':[]}


def test_upload_archive_contains_only_public_page_and_headers():
    with zipfile.ZipFile(io.BytesIO(remote.archive(package()))) as archive:
        assert set(archive.namelist())=={'index.html','_headers'}
        assert b'board-data' in archive.read('index.html')
    data=package();data['secret']='do not upload'
    with pytest.raises(ValueError):remote.archive(data)


def test_transport_is_fixed_host_and_does_not_follow_redirects(monkeypatch):
    calls=[]
    def request(method,url,**kwargs):
        calls.append((method,url,kwargs))
        return SimpleNamespace(status_code=200,json=lambda:{'id':'deploy-123','site_id':'site-1234','state':'processing'})
    monkeypatch.setattr(remote.requests,'request',request)
    job=remote.deploy(package(),'site-1234','private-token')
    assert job['state']=='processing'
    assert calls[0][1]=='https://api.netlify.com/api/v1/sites/site-1234/deploys'
    assert calls[0][2]['allow_redirects'] is False
    assert calls[0][2]['headers']['Content-Type']=='application/zip'


def test_network_error_does_not_echo_credentials(monkeypatch):
    def fail(*args,**kwargs):raise requests.RequestException('private-token')
    monkeypatch.setattr(remote.requests,'request',fail)
    with pytest.raises(RuntimeError) as exc:remote.site_info('site-1234','private-token')
    assert 'private-token' not in str(exc.value)


def test_mismatched_site_is_rejected(monkeypatch):
    monkeypatch.setattr(remote,'api_call',lambda *a,**k:{'id':'wrong','ssl_url':'https://example.netlify.app'})
    with pytest.raises(ValueError):remote.site_info('site-1234','token')
    with pytest.raises(ValueError):remote.identifier('../other')


def ui_app():
    from app.ui.remote_publish import render_remote_publish
    data={'schema_version':1,'built_at':'2026-09-08T20:00:00Z','stale_after_minutes':15,
          'games':{'overall':[],'sides':[],'totals':[]},'props':[],'dfs':[]}
    config={'PARLAYPICKER_NETLIFY_SITE_ID':'site-1234','PARLAYPICKER_NETLIFY_TOKEN':'test-token'}
    render_remote_publish(data,'fingerprint',lambda k:config.get(k,''))


def test_ui_only_uploads_after_explicit_publish_and_waits_for_ready(monkeypatch):
    from streamlit.testing.v1 import AppTest
    calls=[]
    monkeypatch.setattr(remote,'site_info',lambda *a:{'id':'site-1234','url':'https://example.netlify.app'})
    monkeypatch.setattr(remote,'deploy',lambda *a:calls.append('upload') or {'id':'deploy-123','site_id':'site-1234','state':'processing'})
    monkeypatch.setattr(remote,'deployment_status',lambda *a:{'id':'deploy-123','site_id':'site-1234','state':'ready','url':'https://example.netlify.app'})
    at=AppTest.from_function(ui_app).run()
    assert not at.exception and not calls
    at.button(key='publication_remote_verify').click().run()
    assert not calls
    at.button(key='publication_remote_publish').click().run()
    assert calls==['upload'] and not at.success
    assert at.button(key='publication_remote_publish').disabled
    at.button(key='publication_remote_status').click().run()
    assert not at.exception and at.success
    assert calls==['upload']


def test_uncertain_submission_blocks_automatic_retry(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(remote,'site_info',lambda *a:{'id':'site-1234','url':'https://example.netlify.app'})
    def fail(*a):raise RuntimeError('Unknown outcome')
    monkeypatch.setattr(remote,'deploy',fail)
    at=AppTest.from_function(ui_app).run()
    at.button(key='publication_remote_verify').click().run()
    at.button(key='publication_remote_publish').click().run()
    assert not at.exception
    assert at.button(key='publication_remote_publish').disabled
    assert at.warning

def test_ready_deploy_is_not_claimed_live_until_published(monkeypatch):
    def call(method,path,token):
        if path.startswith('/deploys/'):
            return {'id':'deploy-123','site_id':'site-1234','state':'ready'}
        return {'id':'site-1234','published_deploy':{'id':'older-123'},'ssl_url':'https://example.netlify.app'}
    monkeypatch.setattr(remote,'api_call',call)
    assert remote.deployment_status('deploy-123','site-1234','token')['state']=='ready_not_published'
