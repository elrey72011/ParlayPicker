"""Completed synthetic CI timing provenance; cancelled progress is not timing."""
import hashlib
import json
from pathlib import Path
import importlib.util
import pytest
from types import SimpleNamespace

from scripts import run_ci_tests as runner

ROOT=Path(__file__).resolve().parents[1]


def _exact_predecessor_profile():
    binding=json.loads((ROOT/'docs/paid-launch/ncaaf-socket-ci-binding-v1.json').read_bytes())
    from scripts import check_launch_change_scope as guard
    profile_path='scripts/ci_test_file_costs_v1.json'
    raw=guard._ncaaf_socket_ci_previous_source(profile_path,(ROOT/profile_path).read_bytes().replace(b"\r\n",b"\n"))
    assert hashlib.sha256(raw).hexdigest()==binding['source_reconstructions'][profile_path]['before_sha256']
    return raw, binding


def test_exact_twelve_advisory_rows_and_frozen_metadata_preserved(historical_nfl_pdf_capacity_profile, monkeypatch):
    monkeypatch.setattr(__import__(__name__,fromlist=['ROOT']), 'ROOT', historical_nfl_pdf_capacity_profile)
    raw,binding=_exact_predecessor_profile()
    prior=json.loads(raw);now=json.loads((ROOT/'scripts/ci_test_file_costs_v1.json').read_bytes())
    successor=now['ncaaf_socket_ci_successor'];paths=set(successor['changed_paths'])
    assert len(paths)==12 and prior['revision']==now['revision']==3
    assert successor['predecessor_profile_sha256']==hashlib.sha256(raw).hexdigest()
    before={r['path']:r for r in prior['files']};after={r['path']:r for r in now['files']}
    assert {p for p in set(before)|set(after) if before.get(p)!=after.get(p)}==paths
    for key,value in prior.items():
        if key not in {'files','sources'}:assert now[key]==value
    assert all(now['sources'][key]==value for key,value in prior['sources'].items())


def test_only_completed_whole_file_xml_supplies_advisory_costs(historical_nfl_pdf_capacity_profile, monkeypatch):
    monkeypatch.setattr(__import__(__name__,fromlist=['ROOT']), 'ROOT', historical_nfl_pdf_capacity_profile)
    data=json.loads((ROOT/'scripts/ci_test_file_costs_v1.json').read_bytes())
    rows={r['path']:r for r in data['files']}
    for path in data['ncaaf_socket_ci_successor']['changed_paths']:
        row=rows[path];observations=row['successor_completed_measurements']
        assert row['kind']=='estimated' and row['successor_variance_multiplier']==1.15
        assert row['seconds']==round(max(v['seconds'] for v in observations)*1.15,6)
        assert row['profiled_tests']==max(v['count'] for v in observations)
        for value in observations:
            source=data['sources'][value['source']]
            assert source['kind']=='completed_junit' and source['attempt']==1
            assert len(source['junit_sha256'])==len(source['assignment_manifest_sha256'])==64
            assert len(value['test_git_blob'])==40
    assert data['ncaaf_socket_ci_successor']['completed_measurements_only'] is True


def test_six_whole_file_assignment_is_deterministic_and_preserves_order():
    profile=runner.load_profile(ROOT/'scripts/ci_test_file_costs_v1.json')
    items=[SimpleNamespace(nodeid=f'tests/test_SYNTHETIC_{i}.py::test_{j}',path=ROOT/f'tests/test_SYNTHETIC_{i}.py') for i in range(12) for j in range(3)]
    first=runner.assignment(items,6,profile,ROOT);second=runner.assignment(items,6,profile,ROOT)
    assert first==second and len(first[0])==len({v['nodeid'] for v in first[0]})==36
    assert [v['nodeid'] for v in first[0]]==[i.nodeid for i in items]
    assert set(v['shard'] for v in first[0])==set(range(1,7))
    for file in {v['path'] for v in first[0]}:
        assert len({v['shard'] for v in first[0] if v['path']==file})==1


def test_predecessor_costs_need_no_git_history(tmp_path,monkeypatch):
    snapshot=tmp_path/'SYNTHETIC-no-history'
    for name in ('scripts/ci_test_file_costs_v1.json','docs/paid-launch/ncaaf-socket-ci-binding-v1.json'):
        dest=snapshot/name;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((ROOT/name).read_bytes())
    assert not (snapshot/'.git').exists()
    monkeypatch.setattr(__import__(__name__,fromlist=['ROOT']),'ROOT',snapshot)
    import subprocess
    def forbidden(*args,**kwargs):raise AssertionError('SYNTHETIC_HISTORY_ACCESS_FORBIDDEN')
    monkeypatch.setattr(subprocess,'check_output',forbidden)
    raw,binding=_exact_predecessor_profile()
    assert hashlib.sha256(raw).hexdigest()==binding['source_reconstructions']['scripts/ci_test_file_costs_v1.json']['before_sha256']


@pytest.mark.parametrize('attack',['changed_body','changed_binding'])
def test_exact_profile_reconstruction_rejects_changed_inputs(tmp_path,monkeypatch,attack):
    snapshot=tmp_path/'SYNTHETIC-changed-source'
    for name in ('scripts/ci_test_file_costs_v1.json','docs/paid-launch/ncaaf-socket-ci-binding-v1.json'):
        dest=snapshot/name;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((ROOT/name).read_bytes())
    monkeypatch.setattr(__import__(__name__,fromlist=['ROOT']),'ROOT',snapshot)
    if attack=='changed_body':
        p=snapshot/'scripts/ci_test_file_costs_v1.json';p.write_bytes(p.read_bytes()+b' ')
    else:
        p=snapshot/'docs/paid-launch/ncaaf-socket-ci-binding-v1.json';b=json.loads(p.read_bytes())
        b['source_reconstructions']['scripts/ci_test_file_costs_v1.json']['before_sha256']='0'*64
        p.write_text(json.dumps(b))
    with pytest.raises(AssertionError):_exact_predecessor_profile()


def test_closure_profile_fixture_is_exact_isolated_and_rejects_substitution(tmp_path,monkeypatch):
    name='SYNTHETIC_closure_fixture'
    spec=importlib.util.spec_from_file_location(name,ROOT/'tests/test_football_document_ci_closure.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    first=module.historical_ci_closure_profile.__wrapped__(tmp_path/'first')
    second=module.historical_ci_closure_profile.__wrapped__(tmp_path/'second')
    original=(ROOT/module.PROFILE).read_bytes()
    module.test_new_provenance_records_observed_checkout_without_rewriting_history(first)
    p=first/module.PROFILE;data=json.loads(p.read_bytes());data['files'][0]['seconds']+=1;p.write_text(json.dumps(data))
    with pytest.raises(AssertionError):module.test_new_provenance_records_observed_checkout_without_rewriting_history(first)
    module.test_new_provenance_records_observed_checkout_without_rewriting_history(second)
    assert (ROOT/module.PROFILE).read_bytes()==original
    snapshot=tmp_path/'SYNTHETIC-substituted-source'
    for rel in (module.PROFILE,'docs/paid-launch/ncaaf-socket-ci-binding-v1.json'):
        dest=snapshot/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/rel).read_bytes())
    p=snapshot/module.PROFILE;p.write_bytes(p.read_bytes()+b' ')
    monkeypatch.setattr(module,'ROOT',snapshot)
    with pytest.raises(AssertionError):module.historical_ci_closure_profile.__wrapped__(tmp_path/'rejected')


from tests.nfl_pdf_capacity_fixtures import historical_nfl_pdf_capacity_profile
