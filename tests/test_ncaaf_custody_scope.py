"""Bounded exact successor seals; Git fixtures contain synthetic evidence only."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest
from scripts import check_launch_change_scope as guard
from scripts import ncaaf_custody_scope as scope
from scripts.benchmark_drive_history_loading import blocked_network

SOURCE=Path(__file__).resolve().parents[1]
def git(repo,*args):return subprocess.check_output(['git',*args],cwd=repo).decode().strip()
def write(repo,path,raw):
    p=repo/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
def commit(repo,message):
    git(repo,'add','--all');git(repo,'commit','--allow-empty','-qm',message);return git(repo,'rev-parse','HEAD')


@pytest.fixture(scope='session')
def prepared(tmp_path_factory):
    repo=tmp_path_factory.mktemp('ncaaf-new-scope')/'r';repo.mkdir()
    current=(SOURCE/guard.GUARD_PATH).read_bytes().replace(b'\r\n',b'\n')
    previous=guard._ncaaf_custody_parent_guard_source(current)
    original=current.split(b'\nPOLICY_PATH =',1)[0]+b'\n'
    old_root=guard.ROOT
    with blocked_network():
        git(repo,'init','-q');git(repo,'config','user.name','Offline Test');git(repo,'config','user.email','offline@example.invalid');git(repo,'config','core.autocrlf','false')
        write(repo,guard.GUARD_PATH,original);write(repo,'core/protected.py',b'FROZEN = True\n')
        write(repo,'README.md',b'Synthetic exact successor fixture\n')
        write(repo,'.github/workflows/paid-launch.yml',b'name: frozen synthetic workflow\n')
        write(repo,guard.CLOCK_TEST,b'assert future.expired == 1\n')
        recorded=commit(repo,'synthetic original baseline')
        manifest=dict(base_sha=recorded,required_ancestry={'pr_2349_merge_commit':recorded},protected_files=['core/protected.py'],
            protected_git_blobs={'core/protected.py':git(repo,'rev-parse',recorded+':core/protected.py')},
            tooling_sha256={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in [guard.GUARD_PATH,'.github/workflows/paid-launch.yml']})
        raw_manifest=json.dumps(manifest).encode()+b'\n'
        old_policy=json.loads((SOURCE/guard.NCAAF_CHRONOLOGY_POLICY_PATH).read_text())
        retained=set(old_policy['unchanged_bindings'])|set(old_policy['implementation_changes'])|{guard.NCAAF_CHRONOLOGY_POLICY_PATH}|set(guard.NCAAF_CUSTODY_FROZEN_PATHS)
        for path in retained|set(guard.NCAAF_CUSTODY_PATHS):
            if path in {guard.GUARD_PATH,guard.MANIFEST_PATH,'.github/workflows/paid-launch.yml'} or not (SOURCE/path).is_file():continue
            data=(SOURCE/path).read_bytes().replace(b'\r\n',b'\n')
            if path in guard.NCAAF_CUSTODY_PATHS:
                if path not in guard.NCAAF_CUSTODY_PRIOR_SOURCE_RECONSTRUCTIONS:continue
                data=guard._ncaaf_custody_parent_main_source(path,data)
            write(repo,path,data)
        fixture_policy=dict(old_policy,unchanged_bindings={p:git(repo,'hash-object',p) for p in retained if (repo/p).exists()})
        write(repo,guard.NCAAF_CHRONOLOGY_POLICY_PATH,json.dumps(fixture_policy).encode()+b'\n')
        write(repo,guard.MANIFEST_PATH,raw_manifest);write(repo,guard.GUARD_PATH,previous)
        base=commit(repo,'verified synthetic main');guard.ROOT=repo
        binding=dict(guard.NCAAF_CUSTODY_BINDINGS,base=base,base_tree=git(repo,'rev-parse','HEAD^{tree}'),
            manifest_sha256=hashlib.sha256(raw_manifest).hexdigest(),previous_guard_sha256=hashlib.sha256(previous).hexdigest(),
            previous_policy_blob=guard.blob(base,guard.NCAAF_CHRONOLOGY_POLICY_PATH),previous_ci_policy_blob=guard.blob(base,guard.CI_SCHEDULING_POLICY_PATH),
            previous_compatibility_policy_blob=guard.blob(base,guard.NCAAF_COMPAT_POLICY_PATH))
        for path in guard.NCAAF_CUSTODY_PATHS:write(repo,path,(SOURCE/path).read_bytes().replace(b'\r\n',b'\n'))
        binding['reviewed_blobs']={p:git(repo,'hash-object',p) for p in guard.NCAAF_CUSTODY_PATHS if p!=guard.GUARD_PATH}
        implementation=commit(repo,'reviewed synthetic integration')
        policy=scope.make_policy(guard,binding,implementation)
        write(repo,guard.NCAAF_CUSTODY_POLICY_PATH,json.dumps(policy,indent=2).encode()+b'\n')
        candidate=commit(repo,'policy-only seal');guard.ROOT=old_root
    return repo,binding,implementation,candidate,policy


@pytest.fixture
def fx(tmp_path,monkeypatch,prepared):
    template,*rest=prepared;repo=tmp_path/'r';shutil.copytree(template,repo)
    monkeypatch.setattr(guard,'ROOT',repo)
    with blocked_network():yield (repo,*rest)


def assess(fx):
    return scope.run(guard,fx[0]/guard.MANIFEST_PATH,None,fx[1])


def test_exact_successor_and_ordered_ci_merge(fx):
    repo,binding,implementation,candidate,policy=fx
    code,report=assess(fx);assert code==0,{k:report.get(k) for k in ('reason_codes','protected_changes','existing_test_changes','tooling_hash_mismatches','runtime_shadowing','self_protected_changes')}
    assert policy['schema_version']==31 and not report['new_existing_test_exceptions']
    assert git(repo,'diff','--name-status',implementation,candidate).splitlines()==['A\t'+guard.NCAAF_CUSTODY_POLICY_PATH]
    tree=git(repo,'rev-parse','HEAD^{tree}')
    merge=git(repo,'commit-tree',tree,'-p',binding['base'],'-p',candidate,'-m','synthetic CI merge')
    git(repo,'checkout','-q',merge)
    assert assess(fx)[0]==0
    for path,expected in policy['unchanged_bindings'].items():assert guard.blob('HEAD',path)==expected


def test_exact_shared_source_and_predecessor_reconstruction():
    current=(SOURCE/guard.GUARD_PATH).read_bytes().replace(b'\r\n',b'\n')
    previous=guard._ncaaf_custody_parent_guard_source(current)
    assert hashlib.sha256(previous).hexdigest()==guard.NCAAF_CUSTODY_BINDINGS['previous_guard_sha256']
    for old in (guard.NCAAF_NORMAL_BINDINGS,guard.READINESS_DASHBOARD_BINDINGS,guard.NCAAF_COMPAT_BINDINGS,guard.CI_SCHEDULING_BINDINGS,guard.SLATE_AUDIT_BINDINGS):
        assert guard._dfs_guard_matches(current,old['successor_guard_sha256'])
    for path,receipt in guard.NCAAF_CUSTODY_PRIOR_SOURCE_RECONSTRUCTIONS.items():
        after=(SOURCE/path).read_bytes().replace(b'\r\n',b'\n')
        before=guard._ncaaf_custody_parent_main_source(path,after)
        assert hashlib.sha256(before).hexdigest()==receipt['sha256']
        with pytest.raises(ValueError):guard._ncaaf_custody_parent_main_source(path,after+b'\nUNREVIEWED=True\n')
    with pytest.raises(ValueError):guard._ncaaf_custody_parent_guard_source(current+b'\nUNREVIEWED=True\n')


@pytest.mark.parametrize('path',['app_core/ncaaf_compatible_pipeline.py','app_core/ncaaf_model_compatibility.py',
    'app_core/prediction_evidence.py','.github/workflows/ci.yml','core/protected.py',
    'app/ui/readiness_dashboard.py','tests/test_readiness_dashboard.py'])
def test_unreviewed_dirty_staged_and_committed_changes_reject(fx,path):
    repo,*_=fx
    write(repo,path,(repo/path).read_bytes()+b'\nUNREVIEWED = True\n')
    assert assess(fx)[0]!=0
    git(repo,'add',path);assert assess(fx)[0]!=0
    commit(repo,'unreviewed change');assert assess(fx)[0]!=0


@pytest.mark.parametrize('attack',['policy','extra_seal_path','extra_commit','wrong_parent','wrong_tree'])
def test_policy_and_ancestry_reject(fx,attack):
    repo,binding,implementation,candidate,policy=fx
    if attack=='policy':
        git(repo,'checkout','-q',implementation)
        write(repo,guard.NCAAF_CUSTODY_POLICY_PATH,json.dumps(dict(policy,unchanged_bindings={})).encode()+b'\n')
        commit(repo,'altered seal')
    elif attack=='extra_seal_path':
        git(repo,'checkout','-q',implementation);write(repo,'extra.py',b'BYPASS=True\n');commit(repo,'bad seal')
    elif attack=='extra_commit':commit(repo,'extra commit')
    else:
        tree=git(repo,'rev-parse',(binding['base'] if attack=='wrong_tree' else candidate)+'^{tree}')
        parents=[candidate,binding['base']] if attack=='wrong_parent' else [binding['base'],candidate]
        merge=git(repo,'commit-tree',tree,'-p',parents[0],'-p',parents[1],'-m','bad synthetic merge');git(repo,'checkout','-q',merge)
    assert assess(fx)[0]!=0


def test_current_parent_and_inherited_historical_reader_contracts_are_separate():
    current=(SOURCE/guard.GUARD_PATH).read_bytes().replace(b"\r\n",b"\n")
    parent=guard._ncaaf_custody_parent_guard_source(current)
    assert hashlib.sha256(parent).hexdigest()==guard.NCAAF_CUSTODY_BINDINGS["previous_guard_sha256"]
    historical=guard._readiness_dashboard_previous_guard_source(current)
    assert hashlib.sha256(historical).hexdigest()==guard.READINESS_DASHBOARD_BINDINGS["previous_guard_sha256"]
    for name in dir(guard):
        if name.endswith("_previous_guard_source") and not name.startswith("_readiness_dashboard_"):
            reader=getattr(guard,name)
            assert reader(current)==reader(historical),name
            with pytest.raises(ValueError,match="SUCCESSOR_GUARD_REVIEWED_BYTES_CHANGED"):
                reader(current+b"\nUNREVIEWED=True\n")
