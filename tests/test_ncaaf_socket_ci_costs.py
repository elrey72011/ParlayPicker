"""Completed synthetic CI timing provenance; cancelled progress is not timing."""
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from scripts import run_ci_tests as runner

ROOT=Path(__file__).resolve().parents[1]


def test_exact_twelve_advisory_rows_and_frozen_metadata_preserved():
    binding=json.loads((ROOT/'docs/paid-launch/ncaaf-socket-ci-binding-v1.json').read_bytes())
    raw=subprocess.check_output(['git','show',binding['base']+':scripts/ci_test_file_costs_v1.json'],cwd=ROOT)
    prior=json.loads(raw);now=json.loads((ROOT/'scripts/ci_test_file_costs_v1.json').read_bytes())
    successor=now['ncaaf_socket_ci_successor'];paths=set(successor['changed_paths'])
    assert len(paths)==12 and prior['revision']==now['revision']==3
    assert successor['predecessor_profile_sha256']==hashlib.sha256(raw).hexdigest()
    before={r['path']:r for r in prior['files']};after={r['path']:r for r in now['files']}
    assert {p for p in set(before)|set(after) if before.get(p)!=after.get(p)}==paths
    for key,value in prior.items():
        if key not in {'files','sources'}:assert now[key]==value
    assert all(now['sources'][key]==value for key,value in prior['sources'].items())


def test_only_completed_whole_file_xml_supplies_advisory_costs():
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
