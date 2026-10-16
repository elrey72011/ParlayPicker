"""Synthetic Git seals; predecessor assertions and genuine clocks preserved."""
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import pytest
from scripts import check_launch_change_scope as guard
from scripts import football_document_main_scope as scope


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
        FOOTBALL_DOCCLOSE_POLICY_PATH=guard.FOOTBALL_DOCCLOSE_POLICY_PATH,_document_main_guard_matches=lambda source:source==successor,
        _document_main_base_source=lambda source:previous,DOCUMENT_MAIN_RECONSTRUCTION_JSON=b'{}',run=lambda *a:(1,{'status':'SYNTHETIC original baseline report'}))
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
    assert hashlib.sha256(guard._document_main_base_source(source)).hexdigest()==b['previous_guard_sha256']
    assert guard._dfs_guard_matches(source,guard.DOCUMENT_MAIN_REVIEWED_GUARD_SHA256)
    assert not guard._dfs_guard_matches(source+b'\nUNREVIEWED',guard.DOCUMENT_MAIN_REVIEWED_GUARD_SHA256)
    for path,record in b['source_reconstructions'].items():
        value=(root/path).read_bytes().replace(b'\r\n',b'\n')
        prior=guard._document_main_previous_source(path,value)
        assert hashlib.sha256(prior).hexdigest()==record['before_sha256']
        assert guard._document_main_previous_source(path,value+b'UNREVIEWED')==value+b'UNREVIEWED'
    assert guard._document_main_project.cache_info().maxsize==2

def test_staged_mutation_is_not_a_cached_validation_decision(sealed):
    g,*_=sealed
    assert assess(sealed)[0]==0
    (g.ROOT/'core/frozen.py').write_bytes(b'UNREVIEWED staged mutation\n')
    g.git('add','core/frozen.py')
    assert assess(sealed)[0]!=0

def test_length_prefilter_never_grants_trust_or_cross_source_cache_reuse():
    source=(Path(__file__).resolve().parents[1]/guard.GUARD_PATH).read_bytes().replace(b'\r\n',b'\n')
    prior=guard._document_main_base_source(source)
    assert guard._document_main_base_source(prior)==prior
    for changed in (source.replace(b'DOCUMENT_MAIN_BOOTSTRAP = True',b'DOCUMENT_MAIN_BOOTSTRAP = None',1),
                    source[:-1],source+b'UNREVIEWED',b'X'*len(source)):
        assert not guard._dfs_guard_matches(changed,guard.DOCUMENT_MAIN_REVIEWED_GUARD_SHA256)
        assert not guard._dfs_guard_matches(changed,guard.NFL_OWNER_REVIEWED_GUARD_SHA256)
    assert guard._document_main_base_source(source)==prior
    assert guard._document_main_project.cache_info().maxsize==2



def test_main_and_diagnostic_predecessors_exact_and_unknown_edits_reject():
    root=Path(__file__).resolve().parents[1];source=(root/guard.GUARD_PATH).read_bytes()
    b=json.loads((root/scope.BINDING).read_bytes())
    assert hashlib.sha256(guard._document_main_project(source,'main')).hexdigest()==b['main_guard_sha256']
    assert hashlib.sha256(guard._document_main_project(source,'old')).hexdigest()==b['previous_guard_sha256']
    for value in (source+b'UNREVIEWED',b'X'+source[1:],source[:-1]):
        with pytest.raises(ValueError):guard._document_main_project(value,'main')
    runtime=guard._scope_reconstruction_for_validation()
    assert runtime.MAX_CONTEXT_FIELDS==512
    for name in runtime.NAMES:
        assert hasattr(getattr(guard,name),'reconstruction_memo')
        assert getattr(guard,name).reconstruction_context_fields<=512


def test_current_seven_capacity_and_profile_remain_bound_to_main():
    root=Path(__file__).resolve().parents[1];b=json.loads((root/scope.BINDING).read_bytes())
    for p in ('.github/workflows/ci.yml','scripts/run_ci_tests.py','scripts/ci_test_file_costs_v1.json','scripts/scope_reconstruction.py'):
        value=(root/p).read_bytes().replace(b'\r\n',b'\n')
        assert hashlib.sha1(b'blob '+str(len(value)).encode()+b'\0'+value).hexdigest()==b['merged_source_blobs'][p]


def test_flattened_pure_adapter_preserves_complete_original_chain():
    root=Path(__file__).resolve().parents[1];source=(root/guard.GUARD_PATH).read_bytes()
    def outcome(fn,value):
        try:return ('bytes',fn(value))
        except ValueError as exc:return ('rejection',str(exc))
    for value in (source,guard._document_main_project(source,'main'),source+b'UNREVIEWED',b'X'+source[1:],source[:-1]):
        assert outcome(guard._nfl_odds_timing_base_source,value)==outcome(guard._DOCUMENT_MAIN_ORIGINAL_TIMING_SOURCE,value)


@pytest.fixture(autouse=True)
def exact_main_profile_for_historical_assertion(request,tmp_path,monkeypatch):
    if request.node.name!='test_current_seven_capacity_and_profile_remain_bound_to_main':return
    import sys
    original=Path(__file__).resolve().parents[1];root=tmp_path/'SYNTHETIC-exact-main-profile'
    for path in ('.github/workflows/ci.yml','scripts/run_ci_tests.py','scripts/scope_reconstruction.py',scope.BINDING):
        target=root/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((original/path).read_bytes())
    target=root/'scripts/ci_test_file_costs_v1.json'
    value=(original/'docs/paid-launch/football-document-main-profile-v1.json').read_bytes()
    assert hashlib.sha256(value).hexdigest()=='090c444804b9d7941b8e8200f60dd02a7de3e0e97b7df89eca7b153c2a5470f7'
    target.write_bytes(value)
    monkeypatch.setattr(sys.modules[__name__],'__file__',str(root/'tests/test_document_main_reconcile.py'))

def test_current_profile_changes_only_one_completed_xml_backed_cost():
    root=Path(__file__).resolve().parents[1]
    before=json.loads((root/'docs/paid-launch/football-document-main-profile-v1.json').read_bytes())
    after=json.loads((root/'scripts/ci_test_file_costs_v1.json').read_bytes())
    a={r['path']:r for r in after['files']};b={r['path']:r for r in before['files']}
    assert set(a)==set(b)
    assert [p for p in b if a[p]!=b[p]]==['tests/test_slate_audit_scope.py']
    row=a['tests/test_slate_audit_scope.py'];rec=after['sources']['document-main-f980-completed-slate']
    assert rec['complete_for_target_file'] is True and rec['checkout_commit']=='f98015dbd969456990a01f9d45405697c7fc5251'
    assert rec['seconds']==801.868 and row['seconds']==round(rec['seconds']*1.15,6)
    assert row['profiled_tests']==rec['count'] and row['latest_measurement_complete'] is True
    assert after['revision']==before['revision']==3
    for k in before.keys()-{'files','sources'}:assert after[k]==before[k]
    assert {k:v for k,v in after['sources'].items() if k!=rec['source']}==before['sources']
    assert set(after)-set(before)=={'document_main_reconcile_cost_provenance'}

def test_current_profile_exact_predecessor_and_unknown_edit_rejection():
    root=Path(__file__).resolve().parents[1]
    current=(root/'scripts/ci_test_file_costs_v1.json').read_bytes()
    original=(root/'docs/paid-launch/football-document-main-profile-v1.json').read_bytes()
    assert guard._document_main_previous('scripts/ci_test_file_costs_v1.json',current,'main')==original
    assert guard._document_main_previous('scripts/ci_test_file_costs_v1.json',current+b'UNREVIEWED','main')==current+b'UNREVIEWED'

@pytest.mark.parametrize('change',[b'UNREVIEWED',b' ',b'\n'])
def test_historical_capacity_fixture_rejects_changed_current_profile(tmp_path,change):
    from tests.nfl_pdf_capacity_fixtures import predecessor_profile,PROFILE,BINDING
    root=Path(__file__).resolve().parents[1]
    assert isinstance(predecessor_profile(root),bytes)
    for path in (PROFILE,BINDING):
        target=tmp_path/path;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((root/path).read_bytes()+(change if path==PROFILE else b''))
    with pytest.raises(AssertionError):predecessor_profile(tmp_path)
