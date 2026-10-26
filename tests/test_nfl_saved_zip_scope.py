"""Synthetic Git scope fixtures; no private bodies or saved-result auth patches."""
import json
import subprocess

import pytest
from scripts import nfl_saved_zip_scope as scope


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    root = tmp_path/'SYNTHETIC-zip-git'; root.mkdir()
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=root).decode().strip()
    def write(path, raw):
        p=root/path; p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
    def commit(message):
        git('add','--all');git('commit','-qm',message);return git('rev-parse','HEAD')
    git('init','-q');git('config','user.name','SYNTHETIC fixture');git('config','user.email','synthetic@example.invalid');git('config','core.autocrlf','false')
    manifest='docs/SYNTHETIC-baseline.json'
    write(manifest,b'{"base_sha":"SYNTHETIC-older"}\n')
    write('app_core/nfl_scenario_research.py',b'SYNTHETIC frozen live reader\n')
    write('app_core/research_replay.py',b'SYNTHETIC frozen replay reader\n')
    write('.github/workflows/ci.yml',b'SYNTHETIC unchanged checks and timeouts\n')
    write('docs/paid-launch/launch-scope-policy-nfl-saved-scenarios-v1.json',b'SYNTHETIC preserved prior seal\n')
    for p in ('scripts/check_launch_change_scope.py','scripts/ncaaf_private_refresh_scope.py'):
        write(p,b'SYNTHETIC unchanged guard '+scope.OLD_STAMP+b'\n')
    filtered=b'SYNTHETIC unchanged filtered guard '+scope.OLD_STAMP+b'\n'
    write('scripts/ncaaf_filtered_history_scope.py',filtered)
    base=commit('synthetic cleared main');tree=git('rev-parse',base+'^{tree}')
    monkeypatch.setattr(scope,'BASE',base);monkeypatch.setattr(scope,'TREE',tree)
    stamp='a'*64
    for p in scope.PATHS:
        raw=(b'SYNTHETIC unchanged guard '+stamp.encode()+b'\n') if p in {'scripts/check_launch_change_scope.py','scripts/ncaaf_private_refresh_scope.py'} else b'SYNTHETIC implementation\n'
        if p=='scripts/ncaaf_filtered_history_scope.py':
            raw=filtered.replace(scope.OLD_STAMP,stamp.encode())+scope.DISPATCH_HELPER+b''.join(scope.DISPATCH_BRANCHES)
        write(p,raw)
    binding=dict(baseline_manifest=manifest,components={scope.AUTHORIZATION:'SYNTHETIC-authority',scope.CATALOG:'SYNTHETIC-catalog'},
        saved_nfl_scenario_zip_viewer=dict(base=base,base_tree=tree,authorization_sha256='SYNTHETIC-authority',
            catalog_sha256='SYNTHETIC-catalog',implementation_paths=sorted(scope.PATHS)))
    implementation=commit('synthetic implementation')
    write(scope.POLICY,json.dumps(scope.make_policy(root,binding,implementation,stamp)).encode())
    seal=commit('synthetic policy only')
    return root,binding,stamp,git,write,commit,seal


def assess(fx):
    root,binding,stamp,*_=fx
    return scope.validate(root,binding,None,None,binding_sha=stamp)


def test_exact_zip_scope_and_reviewed_merge_tree(sealed):
    assert assess(sealed)['status']=='PASS'
    root,b,stamp,git,write,commit,seal=sealed
    merge=git('commit-tree',git('rev-parse',seal+'^{tree}'),'-p',b['saved_nfl_scenario_zip_viewer']['base'],'-p',seal,'-m','synthetic merge')
    git('checkout','-q',merge)
    assert assess(sealed)['status']=='PASS'


@pytest.mark.parametrize('attack',['dirty','untracked','live_reader','replay_reader','workflow','prior_seal','policy','catalog','paths','extra_commit','wrong_merge'])
def test_scope_rejects_changes_outside_exact_zip_authority(sealed,attack):
    root,b,stamp,git,write,commit,seal=sealed
    if attack=='dirty':write('app/ui/nfl_saved_scenarios.py',b'UNREVIEWED\n')
    elif attack=='untracked':write('UNREVIEWED.py',b'UNREVIEWED\n')
    elif attack in {'live_reader','replay_reader','workflow','prior_seal'}:
        path={'live_reader':'app_core/nfl_scenario_research.py','replay_reader':'app_core/research_replay.py',
              'workflow':'.github/workflows/ci.yml','prior_seal':'docs/paid-launch/launch-scope-policy-nfl-saved-scenarios-v1.json'}[attack]
        write(path,b'UNREVIEWED\n');commit('unreviewed change')
    elif attack=='policy':
        p=json.loads((root/scope.POLICY).read_bytes());p['unchanged_blobs']={};write(scope.POLICY,json.dumps(p).encode());git('add','--all');git('commit','--amend','--no-edit','-q')
    elif attack=='catalog':b['saved_nfl_scenario_zip_viewer']['catalog_sha256']='UNREVIEWED'
    elif attack=='paths':b['saved_nfl_scenario_zip_viewer']['implementation_paths'].append('UNREVIEWED.py')
    elif attack=='extra_commit':git('commit','--allow-empty','-qm','unreviewed')
    else:
        merge=git('commit-tree',git('rev-parse',seal+'^{tree}'),'-p',seal,'-p',b['saved_nfl_scenario_zip_viewer']['base'],'-m','wrong parents');git('checkout','-q',merge)
    with pytest.raises(ValueError):assess(sealed)
