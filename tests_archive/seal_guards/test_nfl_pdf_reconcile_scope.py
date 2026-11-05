"""Synthetic Git seals; predecessor assertions and genuine clocks preserved."""
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import pytest
from scripts import check_launch_change_scope as guard
from scripts import nfl_pdf_reconcile_scope as scope


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
    write(guard.MANIFEST_PATH,manifest); write(guard.FOOTBALL_DOCCLOSE_POLICY_PATH,b'{"SYNTHETIC":"predecessor"}\n')
    write('tests/SYNTHETIC_approved_cardinality.py',b'assert True # SYNTHETIC unchanged assertion\n')
    base=commit('synthetic verified main')
    write('main/evidence.txt',b'SYNTHETIC merged socket evidence\n'); main=commit('synthetic owner-merged main'); git('checkout','-q',base)
    paths=[guard.GUARD_PATH,scope.BINDING,'main/evidence.txt','tests/SYNTHETIC_approved_cardinality.py']
    write(guard.GUARD_PATH,successor); write(paths[-1],b'# SYNTHETIC approved input adaptation\nassert True # SYNTHETIC unchanged assertion\n')
    write('main/evidence.txt',b'SYNTHETIC merged socket evidence\n')
    binding=dict(base=base,base_tree=git('rev-parse',base+'^{tree}'),main_base=main,main_tree=git('rev-parse',main+'^{tree}'),manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        previous_guard_sha256=hashlib.sha256(previous).hexdigest(),paths=paths,approval_reference='SYNTHETIC diagnostics only',
        reviewed_blobs={p:git('hash-object',p) for p in paths if p not in {guard.GUARD_PATH,scope.BINDING}},source_reconstructions={},
        merged_source_blobs={'main/evidence.txt':git('rev-parse',main+':main/evidence.txt')},
        approved_existing_test_changes={paths[-1]:'SYNTHETIC exact authorized adaptation; original assertion unchanged'})
    binding['reconstruction_json_sha256']=hashlib.sha256(b'{}').hexdigest()
    write(scope.BINDING,json.dumps(binding).encode()); git('add','--all'); implementation=git('commit-tree',git('write-tree'),'-p',base,'-p',main,'-m','synthetic exact reconciliation'); git('checkout','-q',implementation)
    def exists(rev,path): return subprocess.run(['git','cat-file','-e',rev+':'+path],cwd=repo,capture_output=True).returncode==0
    g=SimpleNamespace(ROOT=repo,git=git,git_bytes=raw,exists_at=exists,_require=guard._require,GUARD_PATH=guard.GUARD_PATH,MANIFEST_PATH=guard.MANIFEST_PATH,
        FOOTBALL_DOCCLOSE_POLICY_PATH=guard.FOOTBALL_DOCCLOSE_POLICY_PATH,_nfl_reconcile_guard_matches=lambda source:source==successor,
        _nfl_reconcile_base_source=lambda source:previous,NFL_PDF_RECONCILE_RECONSTRUCTION_JSON=b'{}',run=lambda *a:(1,{'status':'SYNTHETIC original baseline report'}))
    policy=scope.make_policy(g,binding,implementation); write(scope.POLICY,json.dumps(policy).encode()); seal=commit('synthetic policy only')
    return g,binding,implementation,seal,policy,write,commit


def assess(fx):
    g,b,*_=fx
    return scope.run(g,g.ROOT/g.MANIFEST_PATH,None,b)


def test_exact_policy_and_ordered_ci_merge(sealed):
    g,b,implementation,seal,*_=sealed
    assert assess(sealed)[0]==0
    merge=g.git('commit-tree',g.git('rev-parse','HEAD^{tree}'),'-p',b['main_base'],'-p',seal,'-m','synthetic CI merge')
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
        parents=[seal,b['main_base']] if attack=='wrong_parent' else [b['main_base'],seal]
        merge=g.git('commit-tree',tree,'-p',parents[0],'-p',parents[1],'-m','synthetic invalid merge');g.git('checkout','-q',merge)
    elif attack=='extra_commit':g.git('commit','--allow-empty','-qm','unreviewed extra commit')
    elif attack=='shadow':write('shadow/core/frozen.py',b'UNREVIEWED')
    assert assess(sealed)[0]!=0


def test_exact_predecessor_source_reconstruction_and_unknown_edits():
    root=Path(__file__).resolve().parents[1]
    source=(root/guard.GUARD_PATH).read_bytes().replace(b'\r\n',b'\n')
    b=json.loads((root/scope.BINDING).read_text())
    assert hashlib.sha256(guard._nfl_reconcile_base_source(source)).hexdigest()==b['previous_guard_sha256']
    assert guard._dfs_guard_matches(source,guard.NFL_PDF_RECONCILE_REVIEWED_GUARD_SHA256)
    assert not guard._dfs_guard_matches(source+b'\nUNREVIEWED',guard.NFL_PDF_RECONCILE_REVIEWED_GUARD_SHA256)
    for path,record in b['source_reconstructions'].items():
        value=(root/path).read_bytes().replace(b'\r\n',b'\n')
        prior=guard._nfl_reconcile_previous_source(path,value)
        assert hashlib.sha256(prior).hexdigest()==record['before_sha256']
        assert guard._nfl_reconcile_previous_source(path,value+b'UNREVIEWED')==value+b'UNREVIEWED'
    assert guard._nfl_reconcile_base_source.cache_info().maxsize==2

def test_staged_mutation_is_not_a_cached_validation_decision(sealed):
    g,*_=sealed
    assert assess(sealed)[0]==0
    (g.ROOT/'core/frozen.py').write_bytes(b'UNREVIEWED staged mutation\n')
    g.git('add','core/frozen.py')
    assert assess(sealed)[0]!=0

def test_length_prefilter_never_grants_trust_or_cross_source_cache_reuse():
    source=(Path(__file__).resolve().parents[1]/guard.GUARD_PATH).read_bytes().replace(b'\r\n',b'\n')
    prior=guard._nfl_reconcile_base_source(source)
    assert guard._nfl_reconcile_base_source(prior)==prior
    for changed in (source.replace(b'NFL_PDF_RECONCILE_BOOTSTRAP = True',b'NFL_PDF_RECONCILE_BOOTSTRAP = None',1),
                    source[:-1],source+b'UNREVIEWED',b'X'*len(source)):
        assert not guard._dfs_guard_matches(changed,guard.NFL_PDF_RECONCILE_REVIEWED_GUARD_SHA256)
        assert not guard._dfs_guard_matches(changed,guard.NFL_OWNER_REVIEWED_GUARD_SHA256)
    assert guard._nfl_reconcile_base_source(source)==prior
    assert guard._nfl_reconcile_base_source.cache_info().maxsize==2


def test_both_reviewed_branch_guards_and_changed_source_reject():
    root=Path(__file__).resolve().parents[1]
    source=(root/guard.GUARD_PATH).read_bytes().replace(b'\r\n',b'\n')
    b=json.loads((root/scope.BINDING).read_bytes())
    assert hashlib.sha256(guard._nfl_reconcile_base_source(source)).hexdigest()==b['previous_guard_sha256']
    socket=guard._nfl_reconcile_socket_source(source)
    assert hashlib.sha256(socket).hexdigest()==b['reviewed_predecessors']['socket_guard_sha256']
    assert guard._nfl_odds_socket_guard_matches(socket)
    assert guard._dfs_guard_matches(source,guard.NFL_ODDS_SOCKET_REVIEWED_GUARD_SHA256)
    assert hashlib.sha256(guard._nfl_odds_socket_base_source(source)).hexdigest()==guard.NFL_ODDS_SOCKET_PREVIOUS_GUARD_SHA256
    changed=source.replace(b'NFL_PDF_RECONCILE_BOOTSTRAP = True',b'NFL_PDF_RECONCILE_BOOTSTRAP = None',1)
    with pytest.raises(ValueError):guard._nfl_reconcile_socket_source(changed)
    assert guard._nfl_reconcile_socket_source(source)==socket
    runtime=guard._scope_reconstruction_for_validation()
    assert runtime.MAX_CONTEXT_FIELDS==512
    for name in runtime.NAMES:
        assert getattr(guard,name).reconstruction_context_fields<=512


def test_merged_main_evidence_cannot_change_after_valid_assessment(sealed):
    g,b,implementation,seal,policy,write,commit=sealed
    assert assess(sealed)[0]==0
    write('main/evidence.txt',b'UNREVIEWED substituted main evidence\n')
    commit('unreviewed main evidence')
    assert assess(sealed)[0]!=0
