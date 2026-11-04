"""Current three-cost contract and exact historical inputs; synthetic only."""
import hashlib,json,subprocess
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts import ncaaf_reviewed_target_scope as scope
from tests.ncaaf_staged_capacity_fixtures import exact_three_cost_review_inputs


def test_only_three_measured_advisory_entries_change_and_metadata_stays_frozen(exact_three_cost_review_inputs):
    root=Path(__file__).resolve().parents[1];binding=json.loads((root/scope.BINDING).read_bytes())
    path='scripts/ci_test_file_costs_v1.json';raw=(root/path).read_bytes().replace(b'\r\n',b'\n')
    prior=json.loads(scope.previous_source(path,raw,binding));current=json.loads(raw)
    assert all(current[k]==v for k,v in prior.items() if k not in ('files','sources'))
    assert all(current['sources'][k]==v for k,v in prior['sources'].items())
    old={r['path']:r for r in prior['files']};now={r['path']:r for r in current['files']}
    changed={p for p in now if now[p]!=old.get(p)}
    assert changed=={'tests/test_ncaaf_reviewed_target.py','tests/test_ncaaf_reviewed_target_pipeline.py','tests/test_ncaaf_reviewed_target_scope.py'}
    for name in changed:
        row=now[name];observations=row['completed_measurements']
        assert row['kind']=='estimated' and row['variance_multiplier']==1.15 and row['latest_measurement_complete']
        assert row['seconds']==round(max(o['seconds'] for o in observations)*1.15,6)
        assert row['profiled_tests']==max(o['count'] for o in observations)
        for observation in observations:
            source=current['sources'][observation['source']]
            assert source['measurement_complete'] and source['run_attempt']==1
            assert len(source['checkout_commit'])==40 and len(source['xml_sha256'])==64
            assert source['release_coverage'] is False and source['complete_cycle'] is False
    pipeline=now['tests/test_ncaaf_reviewed_target_pipeline.py']
    assert pipeline['profiled_tests']==39
    assert all(o['measurement_complete'] is False for o in pipeline['interrupted_observations'])


@pytest.mark.parametrize('attack',['body','rehashed_body','binding','missing_binding'])
def test_timing_predecessor_authenticates_body_and_binding_before_projection(tmp_path,monkeypatch,attack):
    from tests import ncaaf_staged_capacity_fixtures as fixtures
    root=Path(__file__).resolve().parents[1];path=fixtures.PROFILE;raw=(root/path).read_bytes()
    target=tmp_path/fixtures.BINDING;target.parent.mkdir(parents=True);target.write_bytes((root/fixtures.BINDING).read_bytes())
    monkeypatch.setattr(fixtures,'ROOT',tmp_path)
    assert hashlib.sha256(fixtures.reviewed_target_predecessor_profile(raw)).hexdigest()=='7474e2d7f22d4c5ad5dfa6dbc945d2e549a401250e295546303084c6116ad5a7'
    assert fixtures.exact_predecessor_bytes(path,raw)
    if attack=='body':raw+=b'UNREVIEWED'
    elif attack=='rehashed_body':
        changed=json.loads(raw);changed['files'][0]['seconds']+=1;raw=json.dumps(changed).encode()
    elif attack=='binding':
        changed=json.loads(target.read_bytes());changed['source_reconstructions'][path]['before_sha256']='0'*64;target.write_bytes(json.dumps(changed).encode())
    else:target.unlink()
    with pytest.raises((AssertionError,FileNotFoundError)):fixtures.exact_predecessor_bytes(path,raw)


def test_timing_historical_snapshot_is_isolated_while_current_runner_stays_live(tmp_path,monkeypatch):
    from scripts import check_launch_change_scope
    from scripts import run_ci_tests as runner
    from tests import ncaaf_staged_capacity_fixtures as fixtures
    root=Path(__file__).resolve().parents[1];path=fixtures.PROFILE;live=(root/path).read_bytes()
    def forbidden(*a,**k):raise AssertionError('SYNTHETIC_HISTORY_ACCESS_FORBIDDEN')
    monkeypatch.setattr(subprocess,'check_output',forbidden)
    first=SimpleNamespace(ROOT=root);second=SimpleNamespace(ROOT=root)
    with pytest.MonkeyPatch.context() as patch:
        one=fixtures.historical_reviewed_target_timing_profile.__wrapped__(tmp_path/'one',patch,SimpleNamespace(module=first))
        two=fixtures.historical_reviewed_target_timing_profile.__wrapped__(tmp_path/'two',patch,SimpleNamespace(module=second))
        expected='7474e2d7f22d4c5ad5dfa6dbc945d2e549a401250e295546303084c6116ad5a7'
        assert hashlib.sha256((one/path).read_bytes()).hexdigest()==hashlib.sha256((two/path).read_bytes()).hexdigest()==expected
        assert runner.load_profile(runner.DEFAULT_PROFILE)['sha256']==hashlib.sha256(live).hexdigest()!=expected
        (one/path).write_bytes(b'UNREVIEWED');assert hashlib.sha256((two/path).read_bytes()).hexdigest()==expected
        assert not (one/'.git').exists() and not (two/'.git').exists()
    assert first.ROOT==second.ROOT==root


def test_timing_fixture_changes_preserve_all_original_assertions_without_history():
    import ast
    root=Path(__file__).resolve().parents[1];binding=json.loads((root/scope.BINDING).read_bytes())
    path='tests/test_ncaaf_staged_ci_capacity.py';raw=(root/path).read_bytes().replace(b'\r\n',b'\n');prior=scope.previous_source(path,raw,binding)
    def assertions(source):return {n.name:[ast.dump(a) for a in ast.walk(n) if isinstance(a,ast.Assert)] for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name.startswith('test_')}
    assert assertions(raw)==assertions(prior)
    assert scope.previous_source(path,raw+b'UNREVIEWED',binding)==raw+b'UNREVIEWED'


def test_direct_historical_profile_readers_receive_only_exact_prior_bytes(tmp_path,monkeypatch):
    from tests import ncaaf_staged_capacity_fixtures as fixtures
    root=Path(__file__).resolve().parents[1];live=(root/fixtures.PROFILE).read_bytes()
    expected=fixtures.reviewed_target_predecessor_profile(live)
    path=tmp_path/fixtures.PROFILE;path.parent.mkdir(parents=True);path.write_bytes(live)
    monkeypatch.setattr(fixtures,'ROOT',tmp_path)
    for name in ('test_football_document_ci_closure','test_ncaaf_socket_ci_costs','test_ci_timing_profile'):
        request=SimpleNamespace(module=SimpleNamespace(__name__='tests.'+name))
        with pytest.MonkeyPatch.context() as patch:
            fixtures.install_reviewed_target_profile_inputs(request,patch)
            assert path.read_bytes()==(live if name=='test_ci_timing_profile' else expected)
            path.write_bytes(live+b'UNREVIEWED')
            assert path.read_bytes()==live+b'UNREVIEWED'
            path.write_bytes(live)
        assert path.read_bytes()==live


def test_five_live_costs_preserve_historical_measurement_meanings():
    from tests import ncaaf_staged_capacity_fixtures as fixtures
    root=Path(__file__).resolve().parents[1];path=fixtures.PROFILE
    with (root/path).open('rb') as f:raw=f.read()
    with (root/scope.BINDING).open('rb') as f:binding=json.loads(f.read())
    prior=json.loads(scope.previous_source(path,raw,binding));current=json.loads(raw)
    three=json.loads(fixtures.reviewed_target_three_cost_profile(raw))
    old={r['path']:r for r in prior['files']};now={r['path']:r for r in current['files']}
    expected={'tests/test_ncaaf_reviewed_target.py','tests/test_ncaaf_reviewed_target_pipeline.py',
              'tests/test_ncaaf_reviewed_target_scope.py','tests/test_ncaaf_pilot_scope.py',
              'tests/test_ncaaf_spread_boundaries.py'}
    assert {name for name in now if now[name]!=old.get(name)}==expected
    assert all(current[k]==v for k,v in prior.items() if k not in ('files','sources'))
    assert all(current['sources'][k]==v for k,v in three['sources'].items())
    for row in three['files']:
        if row['path']!='tests/test_ncaaf_pilot_scope.py':assert now[row['path']]==row
    pilot=old['tests/test_ncaaf_pilot_scope.py']
    assert all(now['tests/test_ncaaf_pilot_scope.py'][k]==v for k,v in pilot.items() if k not in ('seconds','references'))
    assert now['tests/test_ncaaf_pilot_scope.py']['references'][:len(pilot['references'])]==pilot['references']
    for name,count,maximum,estimate in [('tests/test_ncaaf_pilot_scope.py',15,108.033,124.23795),
                                        ('tests/test_ncaaf_spread_boundaries.py',78,368.012,423.2138)]:
        row=now[name];observations=row['reviewed_target_timing_measurements']
        assert row['kind']=='estimated' and row['profiled_tests']==count
        assert row['seconds']==estimate==round(max(o['seconds'] for o in observations)*1.15,6)
        assert row['reviewed_target_timing_basis_seconds']==maximum
        assert row['reviewed_target_timing_variance_multiplier']==1.15
        for observation in observations:
            source=current['sources'][observation['source']]
            assert observation['count']==count and observation['measurement_complete']
            assert source['measurement_complete'] and source['run_attempt']==1 and not source['release_coverage']
            assert len(source['checkout_commit'])==40 and len(source['xml_sha256'])==64


def test_three_cost_snapshot_is_exact_isolated_and_rejects_tampering(monkeypatch):
    import base64,zlib
    from tests import ncaaf_staged_capacity_fixtures as fixtures
    from scripts import check_launch_change_scope as live
    root=Path(__file__).resolve().parents[1];profile=root/fixtures.PROFILE;binding=root/scope.BINDING
    raw=profile.read_bytes();binding_raw=binding.read_bytes();read=Path.read_bytes
    def no_history(*a,**k):raise AssertionError('SYNTHETIC_GIT_HISTORY_FORBIDDEN')
    monkeypatch.setattr(subprocess,'check_output',no_history)
    with pytest.MonkeyPatch.context() as patch:
        fixtures.exact_three_cost_review_inputs.__wrapped__(patch)
        assert hashlib.sha256(profile.read_bytes()).hexdigest()=='da8276acf04815e9167d44511d54a192572983ac18000e817880d596fb38f3cb'
        assert hashlib.sha256(binding.read_bytes()).hexdigest()=='fa86cc15ffce8027409c2ff10ee6cfd42eb805ca55427e408e4a9e4fc3cb5e24'
    assert profile.read_bytes()==raw and binding.read_bytes()==binding_raw
    for attack in ('profile','binding','rehashed_binding','frozen_binding'):
        with pytest.MonkeyPatch.context() as patch:
            changed=json.loads(binding_raw);changed['base']='UNREVIEWED'
            corrupted=json.dumps(changed,sort_keys=True).encode() if attack=='rehashed_binding' else binding_raw+b'UNREVIEWED'
            changed_profile=json.loads(raw);changed_profile['files'][0]['seconds']+=1
            altered_profile=json.dumps(changed_profile,sort_keys=True).encode()
            def altered(path):
                value=read(path)
                if path.resolve()==profile.resolve() and attack=='profile':return altered_profile
                if path.resolve()==binding.resolve() and attack in ('binding','rehashed_binding'):return corrupted
                return value
            patch.setattr(Path,'read_bytes',altered)
            if attack=='frozen_binding':
                patch.setattr(fixtures,'_THREE_BINDING_ZLIB_B64',base64.b64encode(zlib.compress(b'UNREVIEWED')).decode())
            fixtures.exact_three_cost_review_inputs.__wrapped__(patch)
            with pytest.raises(AssertionError):
                if attack=='profile':test_only_three_measured_advisory_entries_change_and_metadata_stays_frozen(None)
                else:binding.read_bytes()
        assert profile.read_bytes()==raw and binding.read_bytes()==binding_raw
