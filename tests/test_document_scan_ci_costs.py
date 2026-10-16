"""Completed timing provenance and exact isolated historical inputs; offline only."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import check_launch_change_scope as guard, run_ci_tests as runner
from tests import test_ncaaf_socket_ci_costs as historical

ROOT=Path(__file__).resolve().parents[1]
PROFILE='scripts/ci_test_file_costs_v1.json'


def test_fixed_completed_evidence_rule_and_variance():
    raw=(ROOT/PROFILE).read_bytes()
    prior=json.loads(guard._document_scan_ci_previous_source(PROFILE,raw))
    now=json.loads(raw);before={v['path']:v for v in prior['files']};after={v['path']:v for v in now['files']}
    metadata=now['document_scan_ci_successor'];changed=set(metadata['changed_paths'])
    assert len(changed)==45 and now['revision']==prior['revision']==3
    assert changed=={p for p in set(before)|set(after) if before.get(p)!=after.get(p)}
    assert len(changed-set(before))==36
    for path in changed:
        row=after[path];measurements=row['document_scan_ci_completed_measurements']
        assert row['seconds']==round(max(v['seconds'] for v in measurements)*1.15,6)
        assert row['profiled_tests']==max(v['count'] for v in measurements)
        assert row['kind']=='estimated' and row['document_scan_ci_variance_multiplier']==1.15
        assert path not in before or abs(row['seconds']-row['document_scan_ci_previous_scheduling_seconds'])>=60
        for value in measurements:
            source=now['sources'][value['source']]
            assert source['kind']=='completed_junit' and source['execution_complete'] and source['attempt']==1
            assert len(value['test_git_blob'])==40
            assert len(source['junit_sha256'])==len(source['assignment_manifest_sha256'])==len(source['checkout_receipt_log_sha256'])==64
    assert metadata['overhead_seconds']==120 and metadata['overhead_is_estimate']
    for key,value in prior.items():
        if key not in {'files','sources'}:assert now[key]==value
    assert all(now['sources'][key]==value for key,value in prior['sources'].items())


def test_recorded_checkout_is_distinct_from_pr_head():
    data=json.loads((ROOT/PROFILE).read_bytes())
    expected={38764258973:('6e31a802ac3a39ac58ba38082747d9046efd167d','40c4562c858a5f3da005374ac959a408951ad71a'),
              38684830215:('b528be527c49de3056a8834b17334f26feb0cd77','0ec1d6d0e189116d65dcba1de591d6146889b6ec')}
    sources=[v for k,v in data['sources'].items() if k.startswith('document-scan-ci-')]
    assert len(sources)==12
    for v in sources:
        assert (v['head'],v['checkout_commit'])==expected[v['run_id']]
        assert v['head']!=v['checkout_commit']
    cancelled=[v for v in sources if v['job_gate']!='success']
    assert len(cancelled)==1 and cancelled[0]['execution_complete']


def test_original_test_functions_and_assertions_preserved():
    def tests(raw):
        return {v.name:ast.dump(v,include_attributes=False) for v in ast.parse(raw).body
                if isinstance(v,ast.FunctionDef) and v.name.startswith('test_')}
    for path in ('tests/test_ncaaf_socket_ci_costs.py','tests/test_football_document_ci_closure.py'):
        current=(ROOT/path).read_bytes()
        prior=guard._document_scan_ci_previous_source(path,current)
        assert current!=prior and tests(current)==tests(prior)
        assert guard._document_scan_ci_previous_source(path,current+b'\nSYNTHETIC_CHANGED')==current+b'\nSYNTHETIC_CHANGED'


@pytest.mark.parametrize('attack',['append','cost','metadata'])
def test_exact_historical_input_rejects_changed_source(attack):
    original=(ROOT/PROFILE).read_bytes()
    changed=original+b' ' if attack=='append' else json.loads(original)
    if attack=='cost':changed['files'][0]['seconds']+=1
    if attack=='metadata':changed['revision']=4
    if not isinstance(changed,bytes):changed=json.dumps(changed).encode()
    with pytest.raises(AssertionError):historical._verified_socket_ci_profile_bytes(changed)


def test_historical_input_is_exact_without_git_history(tmp_path,monkeypatch):
    raw=(ROOT/PROFILE).read_bytes()
    def forbidden(*a,**kw):raise AssertionError('SYNTHETIC_GIT_HISTORY_FORBIDDEN')
    monkeypatch.setattr(__import__('subprocess'),'check_output',forbidden)
    first=historical._verified_socket_ci_profile_bytes(raw)
    assert hashlib.sha256(first).hexdigest()=='4ed86b6f3577ba1a9874cc3632889afec45650e857326bda5ffcb3ffd421bc49'
    assert historical._verified_socket_ci_profile_bytes(first)==first
    assert not (tmp_path/'.git').exists()


def test_historical_fixture_is_isolated_and_preserves_rejection(tmp_path,monkeypatch):
    request=SimpleNamespace(node=SimpleNamespace(name='test_exact_twelve_advisory_rows_and_frozen_metadata_preserved'))
    original=historical.ROOT;source=(ROOT/PROFILE).read_bytes()
    with monkeypatch.context() as patch:
        historical.historical_socket_ci_profile.__wrapped__(request,tmp_path/'one',patch)
        first=historical.ROOT
        historical.test_exact_twelve_advisory_rows_and_frozen_metadata_preserved()
        p=first/PROFILE;data=json.loads(p.read_bytes());data['files'][0]['seconds']+=1;p.write_text(json.dumps(data))
        with pytest.raises(AssertionError):historical.test_exact_twelve_advisory_rows_and_frozen_metadata_preserved()
    assert historical.ROOT==original
    with monkeypatch.context() as patch:
        historical.historical_socket_ci_profile.__wrapped__(request,tmp_path/'two',patch)
        assert historical.ROOT!=first
        historical.test_exact_twelve_advisory_rows_and_frozen_metadata_preserved()
        historical.test_only_completed_whole_file_xml_supplies_advisory_costs()
    assert (ROOT/PROFILE).read_bytes()==source


def test_assignment_is_deterministic_whole_file_exact_once_and_ordered():
    profile=runner.load_profile(ROOT/PROFILE)
    paths=['tests/test_document_scan_scope.py','tests/test_nfl_document_scan_diagnostics.py','tests/test_slate_audit_scope.py',
           'tests/test_ncaaf_socket_ci_costs.py','tests/test_football_document_ci_closure.py','tests/test_nfl_pdf_ci_scope.py']
    items=[SimpleNamespace(nodeid=f'{p}::SYNTHETIC_{i}',path=ROOT/p) for p in paths for i in range(4)]
    a=runner.assignment(items,6,profile,ROOT)
    assert a==runner.assignment(items,6,profile,ROOT)
    assert [v['nodeid'] for v in a[0]]==[v.nodeid for v in items]
    assert len(a[0])==len({v['nodeid'] for v in a[0]})==24
    for path in paths:assert len({v['shard'] for v in a[0] if v['path']==path})==1


# Exact historical test-module input. Never monkeypatch the live current-main
# module or replace any original assertion/expected value.
import types as _historical_types
import sys as _historical_sys
_historical_path=ROOT/'docs/paid-launch/football-document-scan-ci-historical-cost-test-v1.py.txt'
_historical_raw=_historical_path.read_bytes()
assert hashlib.sha256(_historical_raw).hexdigest()=='7364327e6d181a2c3138c71a1d99143c62fad2569dbebbd42c55f51eb2ac688f'
historical=_historical_types.ModuleType('reviewed_document_scan_historical_socket_costs')
historical.__file__=str(_historical_path)
_historical_sys.modules[historical.__name__]=historical
exec(compile(_historical_raw,str(_historical_path),'exec'),historical.__dict__)
historical.ROOT=ROOT

@pytest.fixture(autouse=True)
def exact_historical_diagnostic_profile(request,tmp_path,monkeypatch):
    if request.node.name not in {'test_fixed_completed_evidence_rule_and_variance','test_recorded_checkout_is_distinct_from_pr_head','test_original_test_functions_and_assertions_preserved'}:return
    original=ROOT;root=tmp_path/'SYNTHETIC-exact-diagnostic-profile'
    files={PROFILE:(original/'docs/paid-launch/football-document-scan-ci-profile-v1.json').read_bytes()}
    if request.node.name=='test_original_test_functions_and_assertions_preserved':
        for path in ('tests/test_ncaaf_socket_ci_costs.py','tests/test_football_document_ci_closure.py'):
            raw=(original/path).read_bytes();files[path]=guard._document_main_previous(path,raw,'old')
    assert hashlib.sha256(files[PROFILE]).hexdigest()=='271805a99755c215685a17fcf3470a783bb8a1c58843e724b1a6ec5799a0ee4d'
    for path,value in files.items():
        target=root/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(value)
    monkeypatch.setattr(__import__(__name__,fromlist=['ROOT']),'ROOT',root)
