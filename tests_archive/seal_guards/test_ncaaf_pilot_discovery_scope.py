"""Exact synthetic Git seals and frozen-reader assertions; no external IO."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest
from scripts import check_launch_change_scope as guard, ncaaf_discovery_scope as scope
from app_core import ncaaf_pilot as capture, ncaaf_pilot_discovery as discovery

SOURCE = Path(__file__).resolve().parents[1]


def test_predecessor_bytes_and_historical_readers_preserved():
    current = (SOURCE/guard.GUARD_PATH).read_bytes().replace(b'\r\n', b'\n')
    original = guard._discovery_parent_guard_source(current)
    binding = json.loads((SOURCE/scope.BINDING).read_text())
    assert hashlib.sha256(original).hexdigest() == binding['previous_guard_sha256']
    assert guard._dfs_guard_matches(current, guard.NCAAF_PILOT_BINDINGS['successor_guard_sha256'])
    for name in dir(guard):
        if name.endswith(('_previous_guard_source', '_parent_guard_source')) and not name.startswith('_discovery_'):
            reader = getattr(guard, name)
            assert reader(current) == reader(original), name
            with pytest.raises(ValueError): reader(current+b'\nUNREVIEWED=True\n')
    with pytest.raises(ValueError): guard._discovery_parent_guard_source(current+b'\nUNREVIEWED=True\n')


def test_proposals_are_distinct_blocked_and_seven_requests_already_supported():
    root = SOURCE/'docs/paid-launch'
    d = json.loads((root/'ncaaf-pilot-discovery-army-fau-proposal-v1.json').read_text())
    c = json.loads((root/'ncaaf-pilot-army-fau-capture-proposal-v1.json').read_text())
    old = (root/'ncaaf-pilot-army-proposal-v1.json').read_bytes().replace(b'\r\n', b'\n')
    b = json.loads((SOURCE/scope.BINDING).read_text())
    assert hashlib.sha1(b'blob '+str(len(old)).encode()+b'\0'+old).hexdigest() == b['preserved_original_proposal_blob']
    assert discovery.planning(d)['status'] == 'BLOCKED' and capture.planning(c)['status'] == 'BLOCKED'
    assert len(d['payload']['requests']) == 1 and len(c['payload']['requests']) == c['payload']['limits']['max_attempts'] == 7
    assert [r['params'].get('week') for r in c['payload']['requests'][1:-1]] == [1,2,3,4,5]
    assert sum(r['provider']=='cfbd' for r in c['payload']['requests']) == 6
    assert len(d['payload']['requests'])+len(c['payload']['requests']) == 8
    capture.validate_plan(c)
    with pytest.raises(ValueError, match='PREREQUISITES'): capture.validate_plan(c, executable=True)
    with pytest.raises(ValueError, match='PREREQUISITES'): discovery.validate_plan(d, executable=True)
    too_small = deepcopy(c['payload']); too_small['limits']['max_attempts'] = 6
    with pytest.raises(ValueError, match='NCAAF_PILOT_BUDGET'): capture.validate_plan(capture.seal(too_small))


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
    write(guard.MANIFEST_PATH,manifest); write(guard.NCAAF_PILOT_POLICY_PATH,b'{"SYNTHETIC":"predecessor"}\n')
    base=commit('synthetic verified main')
    paths=[guard.GUARD_PATH,scope.BINDING,'app_core/SYNTHETIC_discovery.py']
    write(guard.GUARD_PATH,successor); write(paths[-1],b'SYNTHETIC new implementation\n')
    binding=dict(base=base,base_tree=git('rev-parse',base+'^{tree}'),manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        previous_guard_sha256=hashlib.sha256(previous).hexdigest(),paths=paths,
        approval_reference='SYNTHETIC bounded implementation only',reviewed_blobs={paths[-1]:git('hash-object',paths[-1])})
    write(scope.BINDING,json.dumps(binding).encode()); implementation=commit('synthetic implementation')
    def exists(rev,path): return subprocess.run(['git','cat-file','-e',rev+':'+path],cwd=repo,capture_output=True).returncode==0
    g=SimpleNamespace(ROOT=repo,git=git,git_bytes=raw,exists_at=exists,_require=guard._require,GUARD_PATH=guard.GUARD_PATH,MANIFEST_PATH=guard.MANIFEST_PATH,
        NCAAF_PILOT_POLICY_PATH=guard.NCAAF_PILOT_POLICY_PATH,_discovery_guard_matches=lambda source:source==successor,
        _discovery_parent_guard_source=lambda source:previous,run=lambda *a:(1,{'status':'SYNTHETIC original baseline report'}))
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


@pytest.mark.parametrize('attack',['dirty','frozen','binding','policy','extra_seal','wrong_parent','wrong_tree','extra_commit','shadow'])
def test_scope_rejects_unreviewed_changes(sealed,attack):
    g,b,implementation,seal,policy,write,commit=sealed
    if attack in {'dirty','frozen'}:
        write('core/frozen.py',b'UNREVIEWED\n')
        if attack=='frozen':commit('unreviewed frozen change')
    elif attack=='binding':
        write(scope.BINDING,json.dumps(dict(b,approval_reference='UNREVIEWED')).encode());commit('changed binding')
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
