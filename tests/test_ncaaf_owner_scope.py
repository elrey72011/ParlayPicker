"""Synthetic Git scope fixtures; original guard assertions remain unchanged."""
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import pytest
from scripts import check_launch_change_scope as guard
from scripts import ncaaf_owner_scope as scope


@pytest.fixture
def sealed(tmp_path):
    repo=tmp_path/'SYNTHETIC-git'; repo.mkdir()
    def git(*args): return subprocess.check_output(['git',*args],cwd=repo).decode().strip()
    def raw(*args): return subprocess.check_output(['git',*args],cwd=repo)
    def write(path, value):
        p=repo/path; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(value)
    def commit(message):
        git('add','--all'); git('commit','-qm',message); return git('rev-parse','HEAD')
    git('init','-q'); git('config','user.name','SYNTHETIC test'); git('config','user.email','synthetic@example.invalid'); git('config','core.autocrlf','false')
    previous=b'SYNTHETIC frozen guard\n'; successor=b'SYNTHETIC reviewed guard\n'
    write(guard.GUARD_PATH,previous); write('core/frozen.py',b'SYNTHETIC original assertions\n')
    manifest=b'{"protected_files":["core/frozen.py"]}\n'
    write(guard.MANIFEST_PATH,manifest); write(guard.NCAAF_DISCOVERY_POLICY_PATH,b'{"SYNTHETIC":"predecessor"}\n')
    base=commit('synthetic verified main')
    paths=[guard.GUARD_PATH,scope.BINDING,'app_core/SYNTHETIC_discovery.py']
    write(guard.GUARD_PATH,successor); write(paths[-1],b'SYNTHETIC new implementation\n')
    binding=dict(base=base,base_tree=git('rev-parse',base+'^{tree}'),manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        previous_guard_sha256=hashlib.sha256(previous).hexdigest(),paths=paths,
        approval_reference='SYNTHETIC bounded implementation only',reviewed_blobs={paths[-1]:git('hash-object',paths[-1])},source_reconstructions={})
    write(scope.BINDING,json.dumps(binding).encode()); implementation=commit('synthetic implementation')
    def exists(rev,path): return subprocess.run(['git','cat-file','-e',rev+':'+path],cwd=repo,capture_output=True).returncode==0
    g=SimpleNamespace(ROOT=repo,git=git,git_bytes=raw,exists_at=exists,_require=guard._require,GUARD_PATH=guard.GUARD_PATH,MANIFEST_PATH=guard.MANIFEST_PATH,
        NCAAF_DISCOVERY_POLICY_PATH=guard.NCAAF_DISCOVERY_POLICY_PATH,_owner_guard_matches=lambda source:source==successor,
        _owner_base_source=lambda source:previous,OWNER_SOURCE_RECONSTRUCTIONS={},run=lambda *a:(1,{'status':'SYNTHETIC original baseline report'}))
    policy=scope.make_policy(g,binding,implementation); write(scope.POLICY,json.dumps(policy).encode()); seal=commit('synthetic policy only')
    return g,binding,implementation,seal,policy,write,commit


def assess(fx):
    g,b,*_=fx
    return scope.run(g,g.ROOT/g.MANIFEST_PATH,None,b)


def test_exact_policy_and_ordered_ci_merge(sealed):
    g,b,implementation,seal,*_=sealed
    assert assess(sealed)[0]==0
    merge=g.git('commit-tree',g.git('rev-parse','HEAD^{tree}'),'-p',b['base'],'-p',seal,'-m','synthetic CI merge')
    g.git('checkout','-q',merge)
    assert assess(sealed)[0]==0


@pytest.mark.parametrize('attack',['dirty','frozen','binding','reconstruction','policy','extra_seal','wrong_parent','wrong_tree','extra_commit','shadow'])
def test_scope_rejects_unreviewed_changes(sealed,attack):
    g,b,implementation,seal,policy,write,commit=sealed
    if attack in {'dirty','frozen'}:
        write('core/frozen.py',b'UNREVIEWED\n')
        if attack=='frozen':commit('unreviewed frozen change')
    elif attack=='binding':
        write(scope.BINDING,json.dumps(dict(b,approval_reference='UNREVIEWED')).encode());commit('changed binding')
    elif attack=='reconstruction':
        write(scope.BINDING,json.dumps(dict(b,source_reconstructions={'UNREVIEWED':{}})).encode());commit('changed reconstruction')
    elif attack=='policy':
        write(scope.POLICY,json.dumps(dict(policy,unchanged_bindings={})).encode());commit('changed policy')
    elif attack=='extra_seal':
        g.git('checkout','-q',implementation);write(scope.POLICY,json.dumps(policy).encode());write('extra.py',b'UNREVIEWED');commit('bad seal')
    elif attack in {'wrong_parent','wrong_tree'}:
        tree=g.git('rev-parse',(b['base'] if attack=='wrong_tree' else seal)+'^{tree}')
        parents=[seal,b['base']] if attack=='wrong_parent' else [b['base'],seal]
        merge=g.git('commit-tree',tree,'-p',parents[0],'-p',parents[1],'-m','synthetic invalid merge');g.git('checkout','-q',merge)
    elif attack=='extra_commit':g.git('commit','--allow-empty','-qm','unreviewed extra commit')
    elif attack=='shadow':write('shadow/core/frozen.py',b'UNREVIEWED')
    assert assess(sealed)[0]!=0


def test_old_assertions_and_production_clocks_are_preserved():
    root=Path(__file__).resolve().parents[1]
    # The separate full-history scope gate verifies these exact base bindings.
    # Application shards intentionally have a shallow checkout: no history fetch
    # is necessary to prove whole-file identity, which also preserves assertions.
    policy=json.loads((root/scope.POLICY).read_text())
    for path in ('tests/test_ncaaf_pilot.py','app_core/candidate_chronology.py',
                 'app_core/quote_freshness.py','tests/conftest.py'):
        raw=(root/path).read_bytes().replace(b'\r\n',b'\n')
        blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        assert blob==policy['unchanged_bindings'][path]
    from scripts.ncaaf_synthetic_clock_fixture import applies,TARGET
    assert all(applies(TARGET+'['+case+']') for case in ('spread_home--3.5','spread_away-3.5','total_over-50.5','total_under-50.5'))
    assert not applies('tests/test_ncaaf_pilot.py::test_unaccepted_or_invalid_evidence')
    assert not applies('tests/test_ncaaf_owner_research.py::test_synthetic_clock_isolation_does_not_renew_quotes_or_started_games')
    config=(root/scope.TEST_CONFIG).read_bytes().replace(b'\r\n',b'\n')
    before=guard._owner_previous_source(scope.TEST_CONFIG,config)
    assert config==before.replace(scope.CONFIG_ANCHOR,scope.CONFIG_ANCHOR+scope.CONFIG_CLOCK)


def test_preservation_needs_no_history_fetch(monkeypatch):
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**k:pytest.fail('Shallow preservation cannot fetch/read missing Git history'))
    test_old_assertions_and_production_clocks_are_preserved()


def test_legacy_bootstrap_needs_no_successor_binding(tmp_path):
    import types
    path = tmp_path/'scripts/check_launch_change_scope.py'
    path.parent.mkdir()
    source = Path(guard.__file__).read_bytes().replace(b'\r\n', b'\n')
    path.write_bytes(source)
    legacy = types.ModuleType('isolated_legacy_guard')
    legacy.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), legacy.__dict__)
    assert legacy._owner_base_source(source) == guard._owner_base_source(source)
    assert legacy.OWNER_SOURCE_RECONSTRUCTIONS == guard.OWNER_SOURCE_RECONSTRUCTIONS
    # Bootstrap alone grants no gate exception: actual successor validation
    # still requires its hash-verified external binding and policy.
    with pytest.raises(FileNotFoundError):
        legacy._owner_binding()


def test_exact_source_peel_delegates_older_reconstructions():
    root=Path(__file__).resolve().parents[1]
    current=(root/'app_core/per_game_boards.py').read_bytes().replace(b'\r\n',b'\n')
    earlier=guard._estimate_previous_main_source('app_core/per_game_boards.py',current)
    assert guard._drive_previous_main_source('app_core/per_game_boards.py',earlier)==guard._owner_predecessor()._drive_previous_main_source('app_core/per_game_boards.py',earlier)
    with pytest.raises(ValueError):
        guard._drive_previous_main_source('app_core/per_game_boards.py',earlier+b'\nUNREVIEWED=1\n')


def test_historical_fixture_preimages_remain_exact():
    parent=guard._owner_predecessor()
    bindings=guard._owner_binding()['source_reconstructions']
    for name,group in vars(parent).items():
        if name.endswith('_PRIOR_SOURCE_RECONSTRUCTIONS') and isinstance(group,dict):
            current=getattr(guard,name)
            for path,receipt in group.items():
                assert current[path]['sha256']==receipt['sha256']
                if 'after_sha256' not in receipt: continue
                expected=bindings.get(path,{}).get('after_sha256',receipt['after_sha256'])
                assert current[path]['after_sha256']==expected


def test_merged_cloud_guard_and_bindings_are_preserved():
    root=Path(__file__).resolve().parents[1]
    current=(root/guard.GUARD_PATH).read_bytes().replace(b'\r\n',b'\n')
    previous=guard._owner_base_source(current)
    binding=guard._owner_binding()
    assert hashlib.sha256(previous).hexdigest()==binding['previous_guard_sha256']
    parent=guard._owner_predecessor()
    assert parent._cloud_guard_matches(previous)
    assert guard._cloud_guard_matches(current)
    assert guard._cloud_peel_exact_guard(current)==parent._cloud_peel_exact_guard(previous)
    with pytest.raises(ValueError,match='SUCCESSOR_GUARD_REVIEWED_BYTES_CHANGED'):
        guard._cloud_guard_matches(current+b'\nUNREVIEWED=1\n')
    policy=json.loads((root/scope.POLICY).read_text(encoding='utf-8'))
    cloud_policy=guard.NCAAF_CLOUD_BACKUP_POLICY_PATH
    assert policy['predecessor_policy_blob']==policy['unchanged_bindings'][cloud_policy]
    for path in ('app_core/ncaaf_cloud_backup.py','scripts/ncaaf_cloud_backup.py',
                 'scripts/ncaaf_cloud_backup_scope.py','tests/test_ncaaf_cloud_backup.py',
                 'tests/test_ncaaf_cloud_backup_scope.py','tests/test_ncaaf_pilot.py',
                 'docs/paid-launch/ncaaf-cloud-backup-binding-v1.json',cloud_policy):
        raw=(root/path).read_bytes().replace(b'\r\n',b'\n')
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==policy['unchanged_bindings'][path]
