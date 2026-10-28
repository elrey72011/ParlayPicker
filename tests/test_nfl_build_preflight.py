"""All inputs synthetic; no provider or database access is permitted."""
import importlib
import ast
import base64
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import types
import zlib

import pytest
from streamlit.testing.v1 import AppTest

from app_core import nfl_build_preflight as build
from app_core.nfl_private_workflow import authenticate, OwnerAccess

TOKEN = 'SYNTHETIC-owner-token-valid'
ROOT = Path(build.__file__).resolve().parents[1]


@pytest.fixture
def offline(monkeypatch):
    def blocked(*a, **k):
        pytest.fail('Connection or prediction forbidden')
    monkeypatch.setattr(socket, 'socket', blocked)
    for name in ('app_core.nfl_workflow_store.Store.postgres',
                 'app_core.nfl_workflow_store.Store.verify',
                 'app_core.nfl_workflow_acquisition.capture_scores',
                 'app_core.nfl_workflow_acquisition.capture_market',
                 'app_core.nfl_scenario_research.run'):
        monkeypatch.setattr(name, blocked)
    fake = types.ModuleType('psycopg')
    fake.__version__ = '3.2.10'
    fake.pq = types.SimpleNamespace(version=lambda:170002)
    fake.connect = blocked
    monkeypatch.setitem(sys.modules, 'psycopg', fake)


def git_fixture(root, *args):
    if args == ('rev-parse', 'HEAD'): return b'1'*40
    if args[0]=='rev-parse' and args[1].endswith('^{tree}'): return b'2'*40
    if args[0] == 'show': return (root / args[1].split(':',1)[1]).read_bytes()
    if args[0] == 'diff': return b''
    raise AssertionError('Unexpected Git command')


def test_owner_required_before_inspection(monkeypatch):
    monkeypatch.setattr(build, '_git', lambda *a: pytest.fail('Unauthorized inspection'))
    for access in (None, object(), OwnerAccess('Robert Velarde',object())):
        with pytest.raises(ValueError, match='OWNER_REQUIRED'):
            build.inspect_build(access)


@pytest.mark.parametrize('name', build.MODULES)
def test_live_code_matches_independently_compiled_source(name,monkeypatch):
    module = importlib.import_module(name)
    raw = Path(module.__file__).read_bytes()
    assert build._loaded_matches(module,raw,module.__file__) == (True,True)
    if name=='app_core.nfl_build_preflight':
        # Exercise the complete real Git/source reader, not a mocked verdict.
        import psycopg
        from app_core.nfl_workflow_store import Store
        from app_core.nfl_private_workflow import production_service
        def blocked(*a,**k):pytest.fail('Connection, provider or prediction forbidden')
        monkeypatch.setattr(socket.socket,'connect',blocked)
        monkeypatch.setattr(socket,'create_connection',blocked)
        monkeypatch.setattr(psycopg,'connect',blocked)
        for target in ('app_core.nfl_scenario_research.compute','app_core.nfl_scenario_research.run',
                       'app_core.nfl_workflow_acquisition.capture_scores','app_core.nfl_workflow_acquisition.capture_market'):
            monkeypatch.setattr(target,blocked)
        forbidden={Store.postgres.__func__.__code__,Store.verify.__code__,production_service.__code__}
        incoming=sys.gettrace()
        def trace(frame,event,arg):
            if event=='call' and frame.f_code in forbidden:blocked()
            return trace
        sys.settrace(trace)
        try:
            report=build.inspect_build(authenticate(TOKEN,TOKEN))
        finally:
            sys.settrace(incoming)
        assert report['status']=='VERIFIED_RELEVANT_LOADED_CODE'
        expected=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD']).decode().strip()
        assert report['checkout']['commit']==expected and report['checkout_stable'] is True
        assert report['process']['pid']==os.getpid()
        assert report['database_connections']==report['provider_requests']==0


def test_stale_code_and_literal_constants_are_distinct():
    module = types.ModuleType('SYNTHETIC')
    raw = b'VALUE = 1\ndef f():\n    return VALUE\n'
    exec(compile(raw,'SYNTHETIC.py','exec'),module.__dict__)
    assert build._loaded_matches(module,raw,'SYNTHETIC.py') == (True,True)
    assert build._loaded_matches(module,raw.replace(b'return VALUE',b'return 2'),'SYNTHETIC.py')[0] is False
    assert build._loaded_matches(module,raw.replace(b'VALUE = 1',b'VALUE = 3'),'SYNTHETIC.py')[1] is False
    del module.f
    assert build._loaded_matches(module,raw,'SYNTHETIC.py')[0] is False


def test_stale_default_argument_is_rejected():
    module=types.ModuleType('SYNTHETIC')
    old=b'def f(value=1):\n    return value\n'
    exec(compile(old,'SYNTHETIC.py','exec'),module.__dict__)
    assert build._loaded_matches(module,old.replace(b'value=1',b'value=2'),'SYNTHETIC.py')[0] is False


def test_literal_comparison_does_not_execute_custom_equality():
    class SecretObject:
        def __eq__(self,other):pytest.fail('Custom equality executed')
    assert build._safe_equal(SecretObject(),'SYNTHETIC') is False
    assert build._safe_equal([SecretObject()],['SYNTHETIC']) is False


def test_missing_git_is_sanitized(offline,monkeypatch):
    def fail(*a): raise subprocess.CalledProcessError(1,['SECRET-HOST-PASSWORD'],stderr=b'SECRET-HOST-PASSWORD')
    monkeypatch.setattr(build,'_git',fail)
    report=build.inspect_build(authenticate(TOKEN,TOKEN))
    assert report['status']=='UNVERIFIED' and report['failure']['phase']=='GIT'
    assert 'SECRET' not in str(report) and report['database_connections']==0


def test_git_commands_are_local_bounded_and_sanitized(monkeypatch):
    calls=[]
    monkeypatch.setenv('PGPASSWORD','SYNTHETIC-secret')
    monkeypatch.setenv('GIT_TRACE','SYNTHETIC-private-file')
    def run(args,**kw):
        calls.append((args,kw))
        return types.SimpleNamespace(stdout=b'1'*40)
    monkeypatch.setattr(subprocess,'run',run)
    assert build._git(ROOT,'rev-parse','HEAD')==b'1'*40
    assert calls[0][1]['timeout']==5 and calls[0][1]['capture_output']
    assert 'PGPASSWORD' not in calls[0][1]['env'] and 'GIT_TRACE' not in calls[0][1]['env']
    assert calls[0][1]['env']['GIT_NO_LAZY_FETCH']=='1'
    assert 'protocol.allow=never' in calls[0][0]


def test_observed_identity_is_not_an_expected_revision_constant(offline,monkeypatch):
    monkeypatch.setattr(build,'_git',git_fixture)
    result=build.inspect_build(authenticate(TOKEN,TOKEN))
    assert result['checkout']['commit']=='1'*40 and result['checkout']['tree']=='2'*40
    assert result['status']=='IDENTITY_MISMATCH'  # Deliberately patched live functions are detected.
    assert result['database_connections']==result['provider_requests']==0
    assert result['diagnostic_allowance_changed'] is False


def test_loaded_alias_mismatch_is_explicit(offline,monkeypatch):
    monkeypatch.setattr(build,'_git',git_fixture)
    from app_core import nfl_storage_diagnostic as diag
    monkeypatch.setattr(diag,'Store',object())
    result=build.inspect_build(authenticate(TOKEN,TOKEN))
    assert result['loaded_aliases_match'] is False
    assert result['status']=='IDENTITY_MISMATCH'


def test_checkout_move_during_observation_fails_closed(offline,monkeypatch):
    commits=iter((b'1'*40,b'3'*40));calls=[]
    def moving(root,*args):
        calls.append(args)
        return next(commits) if args==('rev-parse','HEAD') else git_fixture(root,*args)
    monkeypatch.setattr(build,'_git',moving)
    result=build.inspect_build(authenticate(TOKEN,TOKEN))
    assert result['checkout_stable'] is False and result['status']=='IDENTITY_MISMATCH'
    assert all(args[1].startswith('1'*40+':') for args in calls if args[0]=='show')
    assert ('rev-parse','1'*40+'^{tree}') in calls


def test_source_changed_after_checkout_is_not_accepted(offline,monkeypatch):
    def changed(root,*args):
        raw=git_fixture(root,*args)
        return raw+b'\n# SYNTHETIC changed source\n' if args[0]=='show' else raw
    monkeypatch.setattr(build,'_git',changed)
    result=build.inspect_build(authenticate(TOKEN,TOKEN))
    assert result['status']=='IDENTITY_MISMATCH'
    assert all(not row['source_matches_git'] for row in result['loaded_modules'])


def test_runtime_versions_cannot_emit_secret_text(offline,monkeypatch):
    monkeypatch.setattr(build,'_git',git_fixture)
    monkeypatch.setattr(sys.modules['psycopg'],'__version__','SYNTHETIC-password-host-never-emit')
    monkeypatch.setattr(build.metadata,'version',lambda name:'SYNTHETIC-private-version')
    result=build.inspect_build(authenticate(TOKEN,TOKEN))
    assert result['runtime']['psycopg']==result['runtime']['streamlit']=='UNKNOWN'
    assert 'SYNTHETIC-password' not in str(result) and 'SYNTHETIC-private-version' not in str(result)


def test_authenticated_qualification_predecessor_preserves_frozen_expectations():
    binding=json.loads((ROOT/'docs/research/ncaaf-filtered-history-bindings-v1.json').read_bytes())
    path='scripts/qualification_repair_scope.py'
    row=binding['nfl_capacity_predecessors'][path]
    old=zlib.decompress(base64.b85decode(row['before_b85']))
    assert len(old)==14106
    assert hashlib.sha256(old).hexdigest()=='86ee38eac2ef169962dae9c5b24a46f937711bf69c0fc54f201f191e3a8d7a8a'
    def approved(raw):
        return next(ast.literal_eval(n.value) for n in ast.parse(raw).body
                    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='APPROVED' for t in n.targets))
    assert approved(old)==approved((ROOT/path).read_bytes())
    from scripts import ncaaf_filtered_history_scope as scope
    ui_path='app/ui/nfl_storage_diagnostic.py'
    ui_parent=scope.capacity_predecessor(binding,ui_path,(ROOT/ui_path).read_bytes())
    assert hashlib.sha256(ui_parent).hexdigest()=='4385c038f1cac48fe6784b61d6dff034977010b634eb089b726d3e506d12f352'
    assert hashlib.sha256(scope.capacity_parent_binding(binding)).hexdigest()=='24d9bc417f9cb323f35395191d394fd6423b72ac924b7040b4bbf45f92a813db'
    current=(ROOT/path).read_bytes()
    assert scope.capacity_predecessor(binding,path,current)==old
    for warm in (False,True):
        if warm: assert scope.capacity_predecessor(binding,path,current)==old
        bad=copy.deepcopy(binding)
        bad['nfl_capacity_predecessors'][path]['before_sha256']='0'*64
        with pytest.raises(ValueError,match='PREDECESSOR_CHANGED'):
            scope.capacity_predecessor(bad,path,current)


@pytest.mark.skipif(sys.platform!='linux',reason='Linux process-start evidence uses procfs')
def test_process_identity_is_observed_and_stable_for_rerenders():
    first=build._process();second=build._process()
    assert first==second and first['pid']>0 and isinstance(first['start_ticks'],int)
    assert first['started_at']!='UNKNOWN'


def test_ui_rerenders_do_not_spend_or_repeat_connection(offline,monkeypatch):
    monkeypatch.setenv('PARLAYPICKER_PUBLISH_TOKEN',TOKEN)
    calls=[]
    def probe(access):
        calls.append(access)
        return {'status':'SYNTHETIC','database_connections':0}
    monkeypatch.setattr(build,'inspect_build',probe)
    code='from app.ui.publish_panel import render_publish_panel\nrender_publish_panel(None,None)'
    at=AppTest.from_string(code).run()
    assert not at.exception and not calls
    at.text_input(key='publication_token').set_value('wrong').run()
    assert not calls and not at.button
    at.text_input(key='publication_token').set_value(TOKEN).run()
    at.checkbox(key='nfl_workflow_diag_enabled').check().run()
    assert not calls
    at.button(key='nfl_workflow_build_run').click().run()
    assert len(calls)==1 and not at.exception
    assert 'nfl_workflow_diag_spent' not in at.session_state.filtered_state
    at.run()
    assert len(calls)==1
    at.session_state['nfl_workflow_diag_spent']=True
    at.button(key='nfl_workflow_build_run').click().run()
    assert len(calls)==2 and at.session_state['nfl_workflow_diag_spent'] is True
    at.text_input(key='publication_token').set_value('wrong').run()
    assert 'nfl_workflow_build_result' not in at.session_state.filtered_state
    fresh=AppTest.from_string(code).run()
    fresh.text_input(key='publication_token').set_value(TOKEN).run()
    fresh.checkbox(key='nfl_workflow_diag_enabled').check().run()
    assert 'nfl_workflow_build_result' not in fresh.session_state.filtered_state
    assert len(calls)==2


def test_ui_build_failure_is_sanitized_without_resetting_spent(offline,monkeypatch):
    monkeypatch.setenv('PARLAYPICKER_PUBLISH_TOKEN',TOKEN)
    def fail(access):raise RuntimeError('SYNTHETIC-secret-password-host')
    monkeypatch.setattr(build,'inspect_build',fail)
    at=AppTest.from_string('from app.ui.publish_panel import render_publish_panel\nrender_publish_panel(None,None)').run()
    at.text_input(key='publication_token').set_value(TOKEN).run()
    at.checkbox(key='nfl_workflow_diag_enabled').check().run()
    at.session_state['nfl_workflow_diag_spent']=True
    at.button(key='nfl_workflow_build_run').click().run()
    result=at.session_state['nfl_workflow_build_result']
    assert result['status']=='UNVERIFIED' and 'SYNTHETIC-secret' not in str(result)
    assert at.session_state['nfl_workflow_diag_spent'] is True and not at.exception
