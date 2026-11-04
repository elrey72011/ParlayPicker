"""Synthetic ordered-successor fixtures; preserve the failed seal and readers."""
import hashlib
import json
import subprocess

import pytest
from scripts import nfl_zip_owner_gate_scope as scope


@pytest.fixture
def sealed(tmp_path, monkeypatch, request):
    root = tmp_path/'SYNTHETIC-owner-gate'; root.mkdir()
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=root).decode().strip()
    def write(path, value):
        p=root/path; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(value)
    def commit(message):
        git('add','--all');git('commit','-qm',message);return git('rev-parse','HEAD')
    git('init','-q');git('config','user.name','SYNTHETIC fixture');git('config','user.email','synthetic@example.invalid');git('config','core.autocrlf','false')
    manifest='docs/SYNTHETIC-baseline.json';write(manifest,b'{"base_sha":"SYNTHETIC-older"}\n')
    write('app_core/nfl_scenario_research.py',b'SYNTHETIC frozen model reader\n')
    write('app_core/research_replay.py',b'SYNTHETIC frozen replay reader\n')
    write('tests/test_source_evidence_intake.py',b'SYNTHETIC unchanged original assertions\n')
    write('.github/workflows/ci.yml',b'SYNTHETIC unchanged checks and timeouts\n')
    write('docs/research/nfl-scenario-static-replay-bindings-v1.json',b'SYNTHETIC unchanged trust catalog\n')
    write('docs/paid-launch/launch-scope-policy-nfl-saved-scenario-zip-v1.json',b'SYNTHETIC preserved failed policy\n')
    main=commit('synthetic main')
    old_binding=b'{"SYNTHETIC":"preserved failed binding"}\n'
    parent=hashlib.sha256(old_binding).hexdigest();write(scope.BINDING,old_binding)
    old_pub=b'revoke_saved_viewer(st.session_state)\nSYNTHETIC unchanged token gate\nrevoke_saved_viewer(st.session_state)\n'
    write('app/ui/publish_panel.py',old_pub)
    for p in ('scripts/check_launch_change_scope.py','scripts/ncaaf_private_refresh_scope.py','scripts/ncaaf_filtered_history_scope.py'):
        write(p,b'SYNTHETIC unchanged guard '+parent.encode()+b'\n')
    write('scripts/nfl_saved_zip_scope.py',b'SYNTHETIC unchanged ZIP scope\n')
    base=commit('synthetic failed head');tree=git('rev-parse',base+'^{tree}')
    monkeypatch.setattr(scope,'BASE',base);monkeypatch.setattr(scope,'TREE',tree)
    monkeypatch.setattr(scope,'MAIN',main);monkeypatch.setattr(scope,'PARENT_BINDING',parent)
    stamp='a'*64
    for p in scope.PATHS:
        raw=b'SYNTHETIC authorized correction\n'
        if p=='app/ui/publish_panel.py':raw=old_pub.replace(b'revoke_saved_viewer(st.session_state)',b"revoke_saved_viewer(getattr(st, 'session_state', {}))")
        elif p in ('scripts/check_launch_change_scope.py','scripts/ncaaf_private_refresh_scope.py','scripts/ncaaf_filtered_history_scope.py'):
            raw=b'SYNTHETIC unchanged guard '+stamp.encode()+b'\n'
        elif p=='scripts/nfl_saved_zip_scope.py':raw=b'SYNTHETIC unchanged ZIP scope\n'+scope.DISPATCH_HELPER+b''.join(scope.DISPATCH_BRANCHES)
        write(p,raw)
    mode=getattr(request,'param',None)
    if mode=='pub':write('app/ui/publish_panel.py',(root/'app/ui/publish_panel.py').read_bytes()+b'UNAUTHORIZED\n')
    if mode=='guard':write('scripts/ncaaf_filtered_history_scope.py',b'UNAUTHORIZED '+stamp.encode()+b'\n')
    if mode=='dispatch':write('scripts/nfl_saved_zip_scope.py',(root/'scripts/nfl_saved_zip_scope.py').read_bytes()+b'UNAUTHORIZED\n')
    binding=dict(baseline_manifest=manifest,components={scope.AUTHORIZATION:'SYNTHETIC-authority'},
        saved_nfl_zip_owner_gate_correction=dict(base=base,base_tree=tree,main=main,patch_sha256=scope.PATCH,
            authorization_sha256='SYNTHETIC-authority',parent_binding_sha256=parent,implementation_paths=sorted(scope.PATHS)))
    implementation=commit('synthetic correction')
    write(scope.POLICY,json.dumps(scope.make_policy(root,binding,implementation,stamp)).encode())
    seal=commit('synthetic policy-only seal')
    return root,binding,stamp,git,write,commit,seal


def assess(fx):
    root,binding,stamp,*_=fx
    return scope.validate(root,binding,None,None,binding_sha=stamp)


def test_exact_correction_and_main_merge_preserve_failed_head(sealed):
    assert assess(sealed)['status']=='PASS'
    root,binding,stamp,git,write,commit,seal=sealed
    merge=git('commit-tree',git('rev-parse',seal+'^{tree}'),'-p',scope.MAIN,'-p',seal,'-m','synthetic CI merge')
    git('checkout','-q',merge)
    assert assess(sealed)['status']=='PASS'


@pytest.mark.parametrize('sealed',['pub','guard','dispatch'],indirect=True)
def test_resealed_changes_to_approved_runtime_or_guard_are_rejected(sealed):
    with pytest.raises(ValueError):assess(sealed)


@pytest.mark.parametrize('attack',['dirty','untracked','reader','replay','original_test','workflow','catalog','prior_policy','paths','extra_commit','wrong_merge'])
def test_correction_scope_rejects_outside_changes(sealed,attack):
    root,binding,stamp,git,write,commit,seal=sealed
    if attack=='dirty':write('app/ui/publish_panel.py',b'UNAUTHORIZED\n')
    elif attack=='untracked':write('UNAUTHORIZED.py',b'UNAUTHORIZED\n')
    elif attack in {'reader','replay','original_test','workflow','catalog','prior_policy'}:
        p={'reader':'app_core/nfl_scenario_research.py','replay':'app_core/research_replay.py',
           'original_test':'tests/test_source_evidence_intake.py','workflow':'.github/workflows/ci.yml',
           'catalog':'docs/research/nfl-scenario-static-replay-bindings-v1.json',
           'prior_policy':'docs/paid-launch/launch-scope-policy-nfl-saved-scenario-zip-v1.json'}[attack]
        write(p,b'UNAUTHORIZED\n');commit('unreviewed change')
    elif attack=='paths':binding['saved_nfl_zip_owner_gate_correction']['implementation_paths'].append('UNAUTHORIZED.py')
    elif attack=='extra_commit':git('commit','--allow-empty','-qm','unreviewed')
    else:
        merge=git('commit-tree',git('rev-parse',seal+'^{tree}'),'-p',seal,'-p',scope.MAIN,'-m','wrong CI merge parents');git('checkout','-q',merge)
    with pytest.raises(ValueError):assess(sealed)
