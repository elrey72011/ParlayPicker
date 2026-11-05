"""Synthetic Git seals; predecessor assertions and genuine clocks preserved."""
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import pytest
from scripts import check_launch_change_scope as guard
from scripts import nfl_odds_scope as scope


@pytest.fixture
def sealed(tmp_path):
    repo=tmp_path/'SYNTHETIC-git'; repo.mkdir()
    def git(*args): return subprocess.check_output(['git',*args],cwd=repo).decode().strip()
    def raw(*args): return subprocess.check_output(['git',*args],cwd=repo)
    def write(path,value):
        p=repo/path; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(value)
    def commit(message):
        git('add','--all'); git('commit','-qm',message); return git('rev-parse','HEAD')
    git('init','-q'); git('config','user.name','SYNTHETIC test'); git('config','user.email','synthetic@example.invalid'); git('config','core.autocrlf','false')
    git('config','core.longpaths','true')  # Synthetic repo only; portable Windows fixture paths.
    previous=b'SYNTHETIC frozen guard\n'; successor=b'SYNTHETIC reviewed guard\n'
    write(guard.GUARD_PATH,previous); write('core/frozen.py',b'SYNTHETIC unchanged assertions\n')
    manifest=b'{"protected_files":["core/frozen.py"]}\n'
    write(guard.MANIFEST_PATH,manifest); write(guard.NFL_SCENARIO_POLICY_PATH,b'{"SYNTHETIC":"predecessor"}\n')
    base=commit('synthetic verified main')
    paths=[guard.GUARD_PATH,scope.BINDING,'app_core/SYNTHETIC_diagnostics.py']
    write(guard.GUARD_PATH,successor); write(paths[-1],b'SYNTHETIC implementation\n')
    binding=dict(base=base,base_tree=git('rev-parse',base+'^{tree}'),manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        previous_guard_sha256=hashlib.sha256(previous).hexdigest(),paths=paths,approval_reference='SYNTHETIC diagnostics only',
        reviewed_blobs={paths[-1]:git('hash-object',paths[-1])},source_reconstructions={})
    write(scope.BINDING,json.dumps(binding).encode()); implementation=commit('synthetic implementation')
    def exists(rev,path): return subprocess.run(['git','cat-file','-e',rev+':'+path],cwd=repo,capture_output=True).returncode==0
    g=SimpleNamespace(ROOT=repo,git=git,git_bytes=raw,exists_at=exists,_require=guard._require,GUARD_PATH=guard.GUARD_PATH,MANIFEST_PATH=guard.MANIFEST_PATH,
        NFL_SCENARIO_POLICY_PATH=guard.NFL_SCENARIO_POLICY_PATH,_nfl_odds_guard_matches=lambda source:source==successor,
        _nfl_odds_base_source=lambda source:previous,NFL_ODDS_SOURCE_RECONSTRUCTIONS={},run=lambda *a:(1,{'status':'SYNTHETIC original baseline report'}))
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
    elif attack=='binding':write(scope.BINDING,json.dumps(dict(b,approval_reference='UNREVIEWED')).encode());commit('changed binding')
    elif attack=='reconstruction':write(scope.BINDING,json.dumps(dict(b,source_reconstructions={'UNREVIEWED':{}})).encode());commit('changed reconstruction')
    elif attack=='policy':write(scope.POLICY,json.dumps(dict(policy,unchanged_bindings={})).encode());commit('changed policy')
    elif attack=='extra_seal':
        g.git('checkout','-q',implementation);write(scope.POLICY,json.dumps(policy).encode());write('extra.py',b'UNREVIEWED');commit('bad seal')
    elif attack in {'wrong_parent','wrong_tree'}:
        tree=g.git('rev-parse',(b['base'] if attack=='wrong_tree' else seal)+'^{tree}')
        parents=[seal,b['base']] if attack=='wrong_parent' else [b['base'],seal]
        merge=g.git('commit-tree',tree,'-p',parents[0],'-p',parents[1],'-m','synthetic invalid merge');g.git('checkout','-q',merge)
    elif attack=='extra_commit':g.git('commit','--allow-empty','-qm','unreviewed extra commit')
    elif attack=='shadow':write('shadow/core/frozen.py',b'UNREVIEWED')
    assert assess(sealed)[0]!=0


def test_original_assertions_and_clocks_preserved_without_git_history(monkeypatch):
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**k:pytest.fail('No missing-history fetch'))
    root=Path(__file__).resolve().parents[1]
    policy=json.loads((root/scope.POLICY).read_text())
    for path in ('tests/test_nfl_scoring_acquisition.py','tests/test_ncaaf_pilot.py',
                 'app_core/candidate_chronology.py','app_core/quote_freshness.py','tests/conftest.py',
                 '.github/workflows/ci.yml','scripts/nfl_scoring_scope.py'):
        content=(root/path).read_bytes().replace(b'\r\n',b'\n')
        content=guard._listing_capacity_previous_source(path,content)
        assert hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()==policy['unchanged_bindings'][path]


def test_predecessor_guard_logic_and_matchers_preserved():
    root=Path(__file__).resolve().parents[1]
    source=(root/guard.GUARD_PATH).read_bytes().replace(b'\r\n',b'\n')
    binding=json.loads((root/scope.BINDING).read_text())
    assert hashlib.sha256(guard._nfl_odds_base_source(source)).hexdigest()==binding['previous_guard_sha256']
    for identity in (guard.NFL_ODDS_REVIEWED_GUARD_SHA256,guard.CI_TIMING_REVIEWED_GUARD_SHA256,
                     guard.NFL_ADMISSION_BINDINGS['successor_guard_sha256']):
        assert guard._dfs_guard_matches(source,identity)
        assert guard._dfs_guard_matches(source+b'\nBYPASS=True\n',identity) is False


def test_bounded_ancestor_source_cache_preserves_rejections():
    source=(Path(__file__).resolve().parents[1]/guard.GUARD_PATH).read_bytes().replace(b'\r\n',b'\n')
    identities=sorted({v for k,v in vars(guard).items()
                       if k.endswith('REVIEWED_GUARD_SHA256') and isinstance(v,str)})
    guard._nfl_odds_base_source.cache_clear()
    for _ in range(2):
        assert all(guard._dfs_guard_matches(source,identity) for identity in identities)
    assert guard._nfl_odds_base_source.cache_info().maxsize == 8
    assert guard._nfl_odds_base_source.cache_info().currsize <= 8
    changed=source+b'\nUNREVIEWED=True\n'
    assert all(guard._dfs_guard_matches(changed,identity) is False for identity in identities)
    assert all(guard._dfs_guard_matches(source,identity) for identity in identities)
