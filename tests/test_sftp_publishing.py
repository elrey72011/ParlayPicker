import base64
from contextlib import contextmanager
import hashlib
from types import SimpleNamespace
import pytest
from app_core import sftp_publishing as remote


def package():
    return {'schema_version':1,'built_at':'2026-09-08T20:00:00Z','stale_after_minutes':15,
            'games':{'overall':[],'sides':[],'totals':[]},'props':[],'dfs':[]}


def settings():
    return {'PARLAYPICKER_PUBLIC_PROVIDER':'sftp', 'PARLAYPICKER_PUBLIC_URL':'https://picks.example.com',
            'PARLAYPICKER_SFTP_HOST':'server.example.com','PARLAYPICKER_SFTP_USER':'owner',
            'PARLAYPICKER_SFTP_PASSWORD':'private-password', 'PARLAYPICKER_SFTP_DIRECTORY':'/home/owner/picks.example.com',
            'PARLAYPICKER_SFTP_HOST_KEY_SHA256':'SHA256:'+'A'*43}


def config():
    return remote.configuration(lambda k:settings().get(k,''))


@pytest.mark.parametrize('name,value', [('PARLAYPICKER_PUBLIC_URL','http://picks.example.com'),
    ('PARLAYPICKER_SFTP_DIRECTORY','/home/owner/public_html'),
    ('PARLAYPICKER_SFTP_DIRECTORY','/home/owner/picks.example.com/../public_html'),
    ('PARLAYPICKER_SFTP_HOST_KEY_SHA256','unknown')])
def test_reject_unsafe_destination(name,value):
    data=settings();data[name]=value
    with pytest.raises(ValueError):remote.configuration(lambda k:data.get(k,''))


def test_host_key_pin_rejects_different_server():
    key=SimpleNamespace(asbytes=lambda:b'actual-server-key')
    expected='SHA256:'+base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip('=')
    remote.check_host_key(key,expected)
    with pytest.raises(ValueError):remote.check_host_key(key,'SHA256:'+'A'*43)


def test_only_validated_public_content_is_prepared():
    content,identity=remote.prepare(package())
    assert b'board-data' in content
    assert identity=='sftp-'+hashlib.sha256(content).hexdigest()
    bad=package();bad['secret']='private'
    with pytest.raises(ValueError):remote.prepare(bad)


@pytest.mark.parametrize('fail',[False,True])
def test_atomic_upload_never_removes_live_page(monkeypatch,fail):
    calls=[]
    class SFTP:
        def putfo(self,file,path,**kwargs):calls.append(('put',path,file.read()))
        def chmod(self,path,mode):calls.append(('chmod',path,mode))
        def posix_rename(self,source,dest):
            calls.append(('rename',source,dest))
            if fail:raise OSError('rename unavailable')
        def remove(self,path):calls.append(('remove',path))
    @contextmanager
    def connection(c):yield SFTP()
    monkeypatch.setattr(remote,'connection',connection)
    content,identity=remote.prepare(package())
    if fail:
        with pytest.raises(OSError):remote.deploy(content,identity,config())
    else:assert remote.deploy(content,identity,config())['state']=='uploaded'
    assert calls[0][2]==content
    assert [x for x in calls if x[0]=='rename'][0][2]=='/home/owner/picks.example.com/index.html'
    assert all(x[1].endswith('.tmp') for x in calls if x[0]=='remove')


def test_public_verification_requires_exact_content(monkeypatch):
    content,identity=remote.prepare(package())
    monkeypatch.setattr(remote,'public_bytes',lambda c:b'old page')
    assert remote.deployment_status(identity,config())['state']=='content_mismatch'
    monkeypatch.setattr(remote,'public_bytes',lambda c:content)
    assert remote.deployment_status(identity,config())['state']=='ready'


def test_https_failure_hides_error_details(monkeypatch):
    def fail(*a,**kw):raise remote.requests.exceptions.SSLError('private-password')
    monkeypatch.setattr(remote.requests,'get',fail)
    with pytest.raises(RuntimeError) as exc:remote.public_bytes(config())
    assert 'private-password' not in str(exc.value)
    assert 'SSL' in str(exc.value)


def ui_app():
    from app.ui.remote_publish import render_remote_publish
    from tests.test_sftp_publishing import package,settings
    render_remote_publish(package(),'fingerprint',lambda k:settings().get(k,''))


def test_ui_archives_before_upload_and_confirms_only_verified_page(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from app.ui import public_results
    calls=[]
    monkeypatch.setattr(public_results,'history',lambda setting:SimpleNamespace(
        archive=lambda p:calls.append('archive') or 'hash',
        submitted=lambda *a:calls.append('submitted'),confirm=lambda *a:calls.append('confirmed')))
    monkeypatch.setattr(remote,'site_info',lambda c:{'url':c['url']})
    monkeypatch.setattr(remote,'deploy',lambda content,identity,c:calls.append('upload') or {'id':identity,'state':'uploaded'})
    monkeypatch.setattr(remote,'deployment_status',lambda identity,c:{'state':'ready'})
    at=AppTest.from_function(ui_app).run()
    assert not at.exception and not calls
    at.button(key='sftp_verify').click().run()
    assert not calls
    at.button(key='sftp_publish').click().run()
    assert not at.exception
    assert calls==['archive','submitted','upload'] and not at.success
    at.button(key='sftp_status').click().run()
    assert calls==['archive','submitted','upload','confirmed']
    assert at.success and not at.exception


def test_transport_pins_key_before_auth_and_closes_client(monkeypatch):
    import sys
    calls=[]
    key=SimpleNamespace(asbytes=lambda:b'server')
    class SFTP:
        def __enter__(self):return self
        def __exit__(self,*args):calls.append('sftp_closed')
        def get_channel(self):return SimpleNamespace(settimeout=lambda value:None)
        def normalize(self,path):return path
        def lstat(self,path):return SimpleNamespace(st_mode=0o40755)
    class Client:
        def set_missing_host_key_policy(self,policy):self.policy=policy
        def connect(self,host,**kwargs):
            assert kwargs['allow_agent'] is False and kwargs['look_for_keys'] is False
            self.policy.missing_host_key(self,host,key)
            calls.append('authenticated')
        def open_sftp(self):return SFTP()
        def close(self):calls.append('closed')
    monkeypatch.setitem(sys.modules,'paramiko',SimpleNamespace(SSHClient=Client,MissingHostKeyPolicy=object))
    with pytest.raises(ValueError):
        with remote.connection(config()):pass
    assert calls==['closed']
    good=config();good['host_key_sha256']='SHA256:'+base64.b64encode(hashlib.sha256(b'server').digest()).decode().rstrip('=')
    with remote.connection(good):pass
    assert calls==['closed','authenticated','sftp_closed','closed']


def test_failed_backup_does_not_upload_or_retry_on_rerun(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from app.ui import public_results
    calls=[]
    def fail(p):raise RuntimeError('Drive unavailable')
    monkeypatch.setattr(public_results,'history',lambda setting:SimpleNamespace(archive=fail))
    monkeypatch.setattr(remote,'site_info',lambda c:{'url':c['url']})
    monkeypatch.setattr(remote,'deploy',lambda *a:calls.append('upload'))
    at=AppTest.from_function(ui_app).run()
    at.button(key='sftp_verify').click().run()
    at.button(key='sftp_publish').click().run()
    at.run()
    assert not calls and not at.exception
    assert at.button(key='sftp_publish').disabled


@pytest.mark.parametrize('stage,expected',[
    ('connecting and authenticating','SSH connection failed'),
    ('opening the SFTP session','SSH login completed'),
    ('checking the document root','document root could not be accessed'),
    ('transferring the reviewed page','atomic replacement failed'),
])
def test_connection_errors_identify_stage_without_server_secrets(stage,expected):
    message=remote.connection_error(OSError('private-password from server'),stage)
    assert expected in message
    assert 'private-password' not in message


def test_authentication_and_network_errors_have_specific_guidance():
    import socket
    class AuthenticationException(Exception):pass
    message=remote.connection_error(AuthenticationException('secret'),'connecting and authenticating')
    assert 'cPanel username and password' in message and 'secret' not in message
    assert 'hostname could not be resolved' in remote.connection_error(socket.gaierror('secret'),'connecting and authenticating')
    assert 'Streamlit Cloud' in remote.connection_error(TimeoutError('secret'),'connecting and authenticating')
