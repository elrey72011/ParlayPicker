"""Synthetic exact-scope and inherited-guard rejection tests."""
import base64
import copy
import hashlib
import json
import subprocess
import zlib

import pytest
from scripts import nfl_storage_diag_scope as scope


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    root = tmp_path/'SYNTHETIC-diagnostic-scope'; root.mkdir()
    def git(*args): return subprocess.check_output(['git', *args], cwd=root).decode().strip()
    def write(path, raw):
        p = root/path; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(raw)
    def commit(message):
        git('add', '--all'); git('commit', '-qm', message); return git('rev-parse', 'HEAD')
    def digest(raw): return hashlib.sha256(raw).hexdigest()
    git('init', '-q'); git('config', 'user.name', 'Synthetic'); git('config', 'user.email', 'synthetic@example.invalid')
    git('config', 'core.autocrlf', 'false'); git('config', 'core.longpaths', 'true')
    manifest = 'docs/SYNTHETIC-manifest.json'; write(manifest, b'{"base_sha":"SYNTHETIC-older"}\n')
    for path in ('app_core/market_probability_model.py', 'app_core/nfl_scenario_research.py',
                 'app_core/research_replay.py', 'app_core/nfl_workflow_store.py',
                 'tests/test_source_evidence_intake.py', '.github/workflows/ci.yml'):
        write(path, b'SYNTHETIC-frozen\n')
    for path in scope.PATHS - {scope.BINDING, scope.AUTHORIZATION, 'scripts/nfl_storage_diag_scope.py',
                                'app/ui/nfl_storage_diagnostic.py', 'app_core/nfl_storage_diagnostic.py',
                                'tests/test_nfl_storage_diagnostic.py', 'tests/test_nfl_storage_diag_scope.py'}:
        write(path, b'SYNTHETIC-base\n')
    history = (b'SYNTHETIC-stamp '+b'0'*64+b'\n'+scope.HISTORY_CACHE_MARKER
        +b'def authenticate(root):\n'+scope.HISTORY_AUTH_MARKER+b'root,binding,relative,expected):\n    pass\n\n'
        +b'def predecessor_source(root,relative,raw):\n'+scope.HISTORY_OLD_CALL
        +b'    supplied=raw\n'+scope.HISTORY_OLD_COND+b'    return raw\n')
    dispatch = b'SYNTHETIC-dispatch\ndef make_policy(root, binding, implementation, binding_sha):\n    return None\n\ndef validate(root, binding, comparison_base, manifest, *, binding_sha):\n    return None\n'
    prior = {'baseline_manifest': manifest, 'components': {}, 'component_predecessors': {}, 'historical':'unchanged'}
    for path in scope.PATHS - {scope.BINDING, scope.AUTHORIZATION, 'scripts/nfl_storage_diag_scope.py',
                                'app/ui/nfl_storage_diagnostic.py', 'app_core/nfl_storage_diagnostic.py',
                                'tests/test_nfl_storage_diagnostic.py', 'tests/test_nfl_storage_diag_scope.py'}:
        raw = history if path == 'scripts/ncaaf_filtered_history_scope.py' else dispatch if path == 'scripts/nfl_saved_zip_scope.py' else (b'SYNTHETIC-stamp '+b'0'*64+b'\n' if path in scope.NORMALIZED else b'SYNTHETIC-base\n')
        prior['components'][path] = digest(raw)
    prior_raw = (json.dumps(prior, sort_keys=True, indent=2)+'\n').encode(); prior_stamp = digest(prior_raw).encode()
    for path in scope.NORMALIZED:
        p = root/path; raw = history if path.endswith('ncaaf_filtered_history_scope.py') else b'SYNTHETIC-stamp '+b'0'*64+b'\n'
        write(path, raw.replace(b'0'*64, prior_stamp))
    write('scripts/nfl_saved_zip_scope.py', dispatch); write(scope.BINDING, prior_raw)
    base = commit('synthetic base'); tree = git('rev-parse', base+'^{tree}')
    monkeypatch.setattr(scope, 'BASE', base); monkeypatch.setattr(scope, 'TREE', tree)
    monkeypatch.setattr(scope, 'PARENTS', []); monkeypatch.setattr(scope, 'OLD_STAMP', prior_stamp)
    for path in scope.PATHS - {scope.BINDING, *scope.NORMALIZED, 'scripts/nfl_saved_zip_scope.py'}:
        write(path, b'SYNTHETIC-diagnostic\n')
    old_history = (root/'scripts/ncaaf_filtered_history_scope.py').read_bytes()
    new_history = old_history.replace(scope.HISTORY_OLD_CALL, scope.HISTORY_NEW_CALL, 1).replace(scope.HISTORY_OLD_COND, scope.HISTORY_NEW_COND, 1)
    new_history = new_history.replace(scope.HISTORY_CACHE_MARKER,scope.HISTORY_CACHE_MARKER+scope.HISTORY_CACHE_ADDITION,1)
    new_history = new_history.replace(scope.HISTORY_AUTH_MARKER,scope.HISTORY_AUTH_ADDITION+scope.HISTORY_AUTH_MARKER,1)
    new_history = new_history.replace(b'def predecessor_source(', scope.HISTORY_HELPER+b'def predecessor_source(', 1)
    write('scripts/ncaaf_filtered_history_scope.py', new_history)
    policy_line=b'def make_policy(root, binding, implementation, binding_sha):\n'
    validate_line=b'def validate(root, binding, comparison_base, manifest, *, binding_sha):\n'
    new_dispatch=dispatch.replace(policy_line,scope.DISPATCH_HELPER+policy_line+scope.DISPATCH_POLICY,1).replace(validate_line,validate_line+scope.DISPATCH_VALIDATE,1)
    write('scripts/nfl_saved_zip_scope.py',new_dispatch)
    binding=copy.deepcopy(prior); predecessors={}
    for path in scope.PATHS-{scope.BINDING}:
        raw=(root/path).read_bytes()
        normalized=raw.replace(prior_stamp,b'0'*64) if path in scope.NORMALIZED else raw
        new_hash=digest(normalized);binding['components'][path]=new_hash
        if path in prior['components']:
            old=subprocess.check_output(['git','show',base+':'+path],cwd=root)
            predecessors[path]=dict(before_sha256=digest(old),before_bytes=len(old),
                before_b85=base64.b85encode(zlib.compress(old,9)).decode(),after_sha256=new_hash)
    binding['nfl_storage_diagnostic_predecessors']=predecessors
    binding['nfl_storage_diagnostic']=dict(base=base,base_tree=tree,prior_binding_sha256=prior_stamp.decode(),
        authorization_sha256=binding['components'][scope.AUTHORIZATION],implementation_paths=sorted(scope.PATHS))
    raw=(json.dumps(binding,sort_keys=True,indent=2)+'\n').encode(); stamp=digest(raw)
    write(scope.BINDING,raw)
    for path in scope.NORMALIZED:
        p=root/path;write(path,p.read_bytes().replace(prior_stamp,stamp.encode()))
    implementation=commit('synthetic diagnostic')
    write(scope.POLICY,(json.dumps(scope.make_policy(root,binding,implementation,stamp),sort_keys=True)+'\n').encode())
    seal=commit('synthetic seal')
    return root,binding,stamp,git,write,commit,seal


def assess(fixture):
    root,binding,stamp,*_=fixture
    return scope.validate(root,binding,None,None,binding_sha=stamp)


def test_exact_successor_and_tested_merge(sealed):
    assert assess(sealed)['status']=='PASS'
    root,b,stamp,git,write,commit,seal=sealed
    assert scope.validate(root,b,'SYNTHETIC-older',None,binding_sha=stamp)['status']=='PASS'
    with pytest.raises(ValueError, match='COMPARISON_BASE_NOT_APPROVED'):
        scope.validate(root,b,'SYNTHETIC-unreviewed',None,binding_sha=stamp)
    merge=git('commit-tree',git('rev-parse',seal+'^{tree}'),'-p',scope.BASE,'-p',seal,'-m','synthetic merge')
    git('checkout','-q',merge)
    assert assess(sealed)['status']=='PASS'


@pytest.mark.parametrize('attack',['dirty','untracked','model','reader','replay','store','owner_test',
    'ci','policy','paths','authority','history','predecessor','extra_commit','wrong_merge'])
def test_successor_rejects_unreviewed_changes(sealed,attack):
    root,b,stamp,git,write,commit,seal=sealed
    immutable={'model':'app_core/market_probability_model.py','reader':'app_core/nfl_scenario_research.py',
        'replay':'app_core/research_replay.py','store':'app_core/nfl_workflow_store.py',
        'owner_test':'tests/test_source_evidence_intake.py','ci':'.github/workflows/ci.yml'}
    if attack=='dirty':write('app/ui/nfl_private_workflow.py',b'UNREVIEWED\n')
    elif attack=='untracked':write('UNREVIEWED.py',b'UNREVIEWED\n')
    elif attack in immutable:write(immutable[attack],b'UNREVIEWED\n');commit('unreviewed')
    elif attack=='policy':
        p=json.loads((root/scope.POLICY).read_bytes());p['unchanged_blobs']={};write(scope.POLICY,json.dumps(p).encode());git('add','--all');git('commit','--amend','--no-edit','-q')
    elif attack=='paths':b['nfl_storage_diagnostic']['implementation_paths'].append('UNREVIEWED.py')
    elif attack=='authority':b['nfl_storage_diagnostic']['authorization_sha256']='UNREVIEWED'
    elif attack=='history':b['historical']='UNREVIEWED'
    elif attack=='predecessor':next(iter(b['nfl_storage_diagnostic_predecessors'].values()))['before_sha256']='UNREVIEWED'
    elif attack=='extra_commit':git('commit','--allow-empty','-qm','unreviewed')
    else:
        merge=git('commit-tree',git('rev-parse',seal+'^{tree}'),'-p',seal,'-p',scope.BASE,'-m','wrong parents');git('checkout','-q',merge)
    with pytest.raises(ValueError):assess(sealed)


def test_physical_successor_views_reconstruct_exact_released_predecessors():
    from pathlib import Path
    from scripts import ncaaf_filtered_history_scope as boundary
    root=Path(__file__).resolve().parents[1]
    binding=boundary.authenticate(root)
    for path in binding['nfl_storage_diagnostic_predecessors']:
        current=(root/path).read_bytes()
        previous=subprocess.check_output(['git','show',scope.BASE+':'+path],cwd=root)
        assert boundary.diagnostic_predecessor(binding,path,current)==previous


def test_diagnostic_predecessor_cache_rechecks_bytes_and_physical_sources(monkeypatch,tmp_path):
    from pathlib import Path
    from scripts import ncaaf_filtered_history_scope as boundary
    root=Path(__file__).resolve().parents[1]
    cache={}
    monkeypatch.setattr(boundary,'_DIAGNOSTIC_IMAGES',cache)
    boundary.authenticate(root)
    assert cache and len(cache)<=16 and sum(map(len,cache.values()))<=boundary._LEGACY_IMAGE_LIMIT
    key=next(iter(cache));original=cache[key]
    cache[key]=original+b'SYNTHETIC corrupt cached bytes'
    with pytest.raises(ValueError,match='NFL_DIAGNOSTIC_PREDECESSOR_CHANGED'):
        boundary.authenticate(root)
    cache[key]=original
    assert boundary.authenticate(root)
