"""Exact-scope synthetic Git fixtures; no private records or authority."""
import json
import subprocess

import pytest
from scripts import nfl_workflow_scope as scope


@pytest.fixture
def sealed(tmp_path,monkeypatch):
    root=tmp_path/'SYNTHETIC-workflow-scope';root.mkdir()
    def git(*args):return subprocess.check_output(['git',*args],cwd=root).decode().strip()
    def write(p,raw):
        path=root/p;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    def commit(message):
        git('add','--all');git('commit','-qm',message);return git('rev-parse','HEAD')
    git('init','-q');git('config','user.name','SYNTHETIC fixture');git('config','user.email','synthetic@example.invalid');git('config','core.autocrlf','false');git('config','core.longpaths','true')
    manifest='docs/SYNTHETIC-baseline.json';write(manifest,b'{"base_sha":"SYNTHETIC-older"}\n')
    for path in ('app_core/nfl_scenario_research.py','app_core/research_replay.py',
            'app_core/market_probability_model.py','tests/test_source_evidence_intake.py',
            'tests/test_remote_canonical_download.py','.github/workflows/ci.yml',
            'docs/research/nfl-scenario-static-replay-bindings-v1.json'):
        write(path,b'SYNTHETIC frozen original\n')
    for path in scope.NORMALIZED:write(path,b'SYNTHETIC guard '+scope.OLD_STAMP+b'\n')
    write('scripts/nfl_saved_zip_scope.py',b'SYNTHETIC inherited scope\n')
    write('requirements.txt',b'SYNTHETIC dependencies\n# Direct cPanel publication\n')
    ci='.github/workflows/subscriber-completion-postgres.yml'
    old_ci=b'SYNTHETIC unchanged original tests and limits\nRun the three subscriber completion cases\n          --junitxml=subscriber-completion-tests.xml\n'
    write(ci,old_ci)
    base=commit('synthetic main');tree=git('rev-parse',base+'^{tree}')
    monkeypatch.setattr(scope,'BASE',base);monkeypatch.setattr(scope,'TREE',tree)
    stamp='a'*64
    for path in scope.PATHS:
        raw=b'SYNTHETIC authorized implementation\n'
        if path in scope.NORMALIZED:raw=b'SYNTHETIC guard '+stamp.encode()+b'\n'
        elif path=='scripts/nfl_saved_zip_scope.py':raw=b'SYNTHETIC inherited scope\n'+scope.DISPATCH_HELPER+b''.join(scope.DISPATCH_BRANCHES)
        elif path=='requirements.txt':raw=b'SYNTHETIC dependencies\n# Private NFL workflow: durable atomic claims/results; same existing service pin.\npsycopg[binary]==3.2.10\n\n# Direct cPanel publication\n'
        elif path==ci:raw=old_ci.replace(b'Run the three subscriber completion cases',b'Run subscriber completion and private NFL storage cases').replace(b'          --junitxml=',b'          tests/paid_launch/case_nfl_workflow_store.py\n          --junitxml=')
        write(path,raw)
    b=dict(baseline_manifest=manifest,components={scope.AUTHORIZATION:'SYNTHETIC-authority'},
        private_nfl_streamlit_workflow=dict(base=base,base_tree=tree,authorization_sha256='SYNTHETIC-authority',implementation_paths=sorted(scope.PATHS)))
    implementation=commit('synthetic implementation')
    write(scope.POLICY,json.dumps(scope.make_policy(root,b,implementation,stamp)).encode())
    seal=commit('synthetic policy only')
    return root,b,stamp,git,write,commit,seal


def assess(fx):
    root,b,stamp,*_=fx
    return scope.validate(root,b,None,None,binding_sha=stamp)


def test_exact_workflow_scope_and_tested_tree(sealed):
    assert assess(sealed)['status']=='PASS'
    root,b,stamp,git,write,commit,seal=sealed
    merge=git('commit-tree',git('rev-parse',seal+'^{tree}'),'-p',scope.BASE,'-p',seal,'-m','synthetic merge')
    git('checkout','-q',merge)
    assert assess(sealed)['status']=='PASS'


@pytest.mark.parametrize('attack',['dirty','untracked','model','reader','replay','original_owner_test',
    'original_remote_test','shards','zip_trust','policy','paths','authority','extra_commit','wrong_merge'])
def test_exact_workflow_rejects_unreviewed_changes(sealed,attack):
    root,b,stamp,git,write,commit,seal=sealed
    immutable=dict(model='app_core/market_probability_model.py',reader='app_core/nfl_scenario_research.py',
        replay='app_core/research_replay.py',original_owner_test='tests/test_source_evidence_intake.py',
        original_remote_test='tests/test_remote_canonical_download.py',shards='.github/workflows/ci.yml',
        zip_trust='docs/research/nfl-scenario-static-replay-bindings-v1.json')
    if attack=='dirty':write('app/ui/publish_panel.py',b'UNREVIEWED\n')
    elif attack=='untracked':write('UNREVIEWED.py',b'UNREVIEWED\n')
    elif attack in immutable:write(immutable[attack],b'UNREVIEWED\n');commit('unreviewed')
    elif attack=='policy':
        p=json.loads((root/scope.POLICY).read_bytes());p['unchanged_blobs']={};write(scope.POLICY,json.dumps(p).encode());git('add','--all');git('commit','--amend','--no-edit','-q')
    elif attack=='paths':b['private_nfl_streamlit_workflow']['implementation_paths'].append('UNREVIEWED.py')
    elif attack=='authority':b['private_nfl_streamlit_workflow']['authorization_sha256']='UNREVIEWED'
    elif attack=='extra_commit':git('commit','--allow-empty','-qm','unreviewed')
    else:
        merge=git('commit-tree',git('rev-parse',seal+'^{tree}'),'-p',seal,'-p',scope.BASE,'-m','wrong parents');git('checkout','-q',merge)
    with pytest.raises(ValueError):assess(sealed)


@pytest.mark.parametrize('path',['requirements.txt','.github/workflows/subscriber-completion-postgres.yml',
    'scripts/nfl_saved_zip_scope.py','scripts/ncaaf_filtered_history_scope.py'])
def test_resealed_dependency_guard_and_ci_changes_are_rejected(sealed,path):
    root,b,stamp,git,write,commit,seal=sealed
    implementation=json.loads((root/scope.POLICY).read_bytes())['implementation_commit']
    git('checkout','-q',implementation)
    write(path,(root/path).read_bytes()+b'UNREVIEWED\n');git('add','--all');git('commit','--amend','--no-edit','-q')
    new=git('rev-parse','HEAD');write(scope.POLICY,json.dumps(scope.make_policy(root,b,new,stamp)).encode());commit('unreviewed seal')
    with pytest.raises(ValueError):assess(sealed)
