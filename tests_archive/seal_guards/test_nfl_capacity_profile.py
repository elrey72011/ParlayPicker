"""No-network capacity and exact-generation regression checks."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts import run_ci_tests as runner
from scripts import ncaaf_filtered_history_scope as history
from scripts import nfl_capacity_profile_scope as scope
from tests.qualification_ops.reviewed_profile_inputs import reviewed_workflow_bytes

ROOT=Path(__file__).resolve().parents[1]


def test_profile_preserves_every_prior_cost_and_slower_measurement():
    binding=history.authenticate(ROOT)
    raw=history.physical(ROOT/scope.PROFILE);before=history.capacity_predecessor(binding,scope.PROFILE,raw)
    assert hashlib.sha256(before).hexdigest()=='c148ac4b63a280df6289aa73ebd5ed0013ed1b32f5797d03383ca4c11aadcd41'
    old,new=json.loads(before),json.loads(raw)
    prior={r['path']:r for r in old['files']};after={r['path']:r for r in new['files']}
    assert all(new['sources'][k]==v for k,v in old['sources'].items())
    assert all(new[k]==v for k,v in old.items() if k not in {'files','sources'})
    for path,row in prior.items():
        current=after[path]
        assert current['seconds']>=row['seconds']
        for field in ('references','completed_measurements'):
            assert current.get(field,[])[:len(row.get(field,[]))]==row.get(field,[])
    for row in new['files']:
        correction=row.get('nfl_capacity_update')
        if correction:
            assert correction['multiplier']==1.15
            assert row['seconds']==max(correction['prior_profile_floor_seconds'],round(correction['basis_seconds']*1.15,6))
            complete=[m for m in row['completed_measurements'] if m['source'] in correction['completed_sources']]
            assert len(complete)==len(correction['completed_sources'])
            assert all(m['measurement_complete'] and m['phase_basis']=='setup_call_teardown_once' for m in complete)
            assert all(not o['completed_xml_available'] for o in correction['incomplete_observations'])
    assert runner.FALLBACK==old['fallback'] and new['version']==old['version']==1


def test_retained_inventory_exact_once_order_and_capacity_gate():
    ledger=json.loads((ROOT/scope.LEDGER).read_bytes());profile=runner.load_profile(ROOT/scope.PROFILE)
    import base64,zlib
    record=ledger['retained_inventory'];z=zlib.decompressobj()
    raw=z.decompress(base64.b85decode(record['compressed_b85']),8*1024*1024+1)
    assert z.eof and not z.unused_data and len(raw)==record['bytes'] and hashlib.sha256(raw).hexdigest()==record['sha256']
    items=[SimpleNamespace(path=p,nodeid=n) for p,n in json.loads(raw)]
    rows,files,loads=runner.assignment(items,13,profile,ROOT)
    assert len(rows)==len({r['nodeid'] for r in rows})==9135
    assert [r['nodeid'] for r in rows]==[i.nodeid for i in items]
    assert all(len({r['shard'] for r in rows if r['path']==f['path']})==1 for f in files)
    assert len(loads)==13 and all(2700-x-ledger['job_overhead_budget_seconds']>=120 for x in loads)
    assert ledger['old_assignment_sha256']=='52d922d802e924f8f21186bfb9881785071326e9b454f63e04429001e09fa108'
    assert ledger['job_overhead_budget_seconds']>=105.305
    assert ledger['duration_basis']=='setup_call_teardown_once'
    record=ledger['authenticated_observation_payload'];z=zlib.decompressobj()
    raw=z.decompress(base64.b85decode(record['compressed_b85']),8*1024*1024+1)
    assert z.eof and not z.unused_data and len(raw)==record['bytes'] and hashlib.sha256(raw).hexdigest()==record['sha256']
    evidence=json.loads(raw)
    budgets={f['path']:f['scheduling_seconds'] for f in files}
    for label in ('pr','main'):
        for path,observed in evidence['complete_file_totals'][label].items():
            assert budgets[path]>=round(observed['seconds']*1.15,6)


def test_authenticated_parent_generation_and_frozen_assertions():
    binding=history.authenticate(ROOT);raw=history.capacity_parent_binding(binding)
    assert hashlib.sha256(raw).hexdigest()==scope.OLD_STAMP
    assert json.loads(raw)['nfl_storage_diagnostic_predecessors']==binding['nfl_storage_diagnostic_predecessors']
    path='tests/test_ncaaf_filtered_history_scope.py'
    old=history.capacity_predecessor(binding,path,history.physical(ROOT/path))
    new=history.physical(ROOT/path)
    assertions=lambda source:[ast.dump(n) for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Assert)]
    assert assertions(old)==assertions(new)
    assert history.capacity_predecessor(binding,path,new+b'UNREVIEWED')==new+b'UNREVIEWED'


@pytest.mark.parametrize('attack',['binding','profile','parent','cache'])
def test_capacity_generation_rejects_tampering(tmp_path,monkeypatch,attack):
    binding=history.authenticate(ROOT)
    if attack=='cache':
        cache={};monkeypatch.setattr(history,'_CAPACITY_IMAGES',cache)
        history.authenticate(ROOT);key=next(iter(cache));cache[key]+=b'UNREVIEWED'
        with pytest.raises(ValueError,match='NFL_CAPACITY_PREDECESSOR_CHANGED'):history.authenticate(ROOT)
        return
    import shutil
    for relative in set(binding['components'])|{scope.BINDING}:
        p=tmp_path/relative;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/relative,p)
    path=scope.BINDING if attack=='binding' else scope.PROFILE
    if attack=='parent':
        b=json.loads((tmp_path/scope.BINDING).read_bytes());b['nfl_capacity_parent_binding']['before_bytes']+=1
        (tmp_path/scope.BINDING).write_bytes(scope.canonical(b))
    else:
        with (tmp_path/path).open('ab') as f:f.write(b'UNREVIEWED')
    with pytest.raises(ValueError,match='NCAAF_FILTERED_(SOURCE|BINDING)_CHANGED'):history.authenticate(tmp_path)


def test_scheduler_and_workflow_limits_are_unchanged():
    ledger=json.loads((ROOT/scope.LEDGER).read_bytes())
    for path,expected in ledger['unchanged_scheduler_and_workflow_sha256'].items():
        assert hashlib.sha256(reviewed_workflow_bytes(ROOT,path).replace(b'\r\n',b'\n')).hexdigest()==expected
    workflow=(ROOT/'.github/workflows/ci.yml').read_text()
    assert 'shard: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]' in workflow
    assert 'timeout-minutes: 45' in workflow


from tests.nfl_profile_generation_inputs import historical_profile_generation
