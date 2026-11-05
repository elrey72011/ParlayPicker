"""Labelled synthetic workflow contexts; no provider access or owner acceptance."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

PUSH_BASE = '5fb8e13577c3092f1eda4ad9b787368d3f691c71'
PR_BASE = 'ab0049adc88e913b8a2d7bda123bfeecd403dc52'


@pytest.fixture
def workflow_repo(tmp_path):
    from scripts import check_launch_change_scope as live
    from scripts import ncaaf_future_spread_scope as scope
    root = tmp_path/'SYNTHETIC-workflow-repository'; root.mkdir()
    def raw(*args): return subprocess.check_output(['git', *args], cwd=root)
    def git(*args): return raw(*args).decode().strip()
    def write(path, value):
        dest = root/path; dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(value)
    def commit(message):
        git('add', '--all'); git('-c', 'gc.auto=0', '-c', 'maintenance.auto=false', 'commit', '-qm', message)
        return git('rev-parse', 'HEAD')
    git('init', '-q'); git('config', 'user.name', 'SYNTHETIC fixture')
    git('config', 'user.email', 'synthetic@example.invalid'); git('config', 'core.autocrlf', 'false')
    previous = b'SYNTHETIC predecessor guard\n'; successor = b'SYNTHETIC successor guard\n'
    manifest = json.dumps(dict(base_sha=PUSH_BASE, protected_files=['core/frozen.py'])).encode()
    write(live.GUARD_PATH, previous); write('core/frozen.py', b'SYNTHETIC frozen assertions\n')
    write(live.MANIFEST_PATH, manifest); write(live.NCAAF_STAGED_CAPACITY_POLICY_PATH, b'{}\n')
    base = commit('SYNTHETIC baseline')
    scope_path = 'scripts/ncaaf_future_spread_scope.py'
    paths = [live.GUARD_PATH, scope.BINDING, scope_path]
    write(live.GUARD_PATH, successor)
    write(scope_path, Path(scope.__file__).read_bytes().replace(b'\r\n', b'\n'))
    binding = dict(base=base, base_tree=git('rev-parse', base+'^{tree}'), main_base=base,
        main_tree=git('rev-parse', base+'^{tree}'), manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        previous_guard_sha256=hashlib.sha256(previous).hexdigest(), paths=paths,
        approval_reference='SYNTHETIC exact fixture, no human review',
        reviewed_blobs={scope_path: git('hash-object', scope_path)}, source_reconstructions={},
        reconstruction_json_sha256=hashlib.sha256(b'{}').hexdigest())
    write(scope.BINDING, json.dumps(binding).encode()); implementation = commit('SYNTHETIC implementation')
    def exists(rev, path):
        return subprocess.run(['git','cat-file','-e',rev+':'+path],cwd=root,capture_output=True).returncode==0
    g = SimpleNamespace(ROOT=root, git=git, git_bytes=raw, exists_at=exists, _require=live._require,
        GUARD_PATH=live.GUARD_PATH, MANIFEST_PATH=live.MANIFEST_PATH,
        NCAAF_STAGED_CAPACITY_POLICY_PATH=live.NCAAF_STAGED_CAPACITY_POLICY_PATH,
        _ncaaf_future_guard_matches=lambda value: value==successor,
        _ncaaf_future_base_source=lambda value: previous,
        NCAAF_FUTURE_RECONSTRUCTION_JSON=b'{}',
        NCAAF_FUTURE_BINDING_SHA256=hashlib.sha256((root/scope.BINDING).read_bytes()).hexdigest(),
        run=lambda *args:(1,dict(status='SYNTHETIC unchanged baseline report')))
    policy = scope.make_policy(g,binding,implementation)
    write(scope.POLICY,json.dumps(policy).encode()); seal=commit('SYNTHETIC policy seal')
    return SimpleNamespace(g=g,binding=binding,scope=scope,write=write,commit=commit,
        implementation=implementation,seal=seal,manifest=manifest)


@pytest.mark.parametrize('context',['push','original-pr','current-pr','default'])
def test_actual_cli_entrypoint_uses_workflow_environment_and_preserves_contexts(workflow_repo,monkeypatch,capsys,context):
    from scripts import check_launch_change_scope as entry
    fx=workflow_repo;g=fx.g
    for name,value in vars(g).items(): monkeypatch.setattr(entry,name,value,raising=False)
    monkeypatch.setattr(entry,'DEFAULT_MANIFEST',g.ROOT/g.MANIFEST_PATH)
    monkeypatch.setattr(entry,'_scope_git_reads_for_validation',
        lambda:SimpleNamespace(validation_stream=lambda root:contextlib.nullcontext()))
    value={'push':PUSH_BASE,'original-pr':PR_BASE,'current-pr':fx.binding['main_base']}.get(context)
    argv=['scripts/check_launch_change_scope.py']
    if context!='default':
        monkeypatch.setenv('BASE_SHA',value)
        argv+=['--base',os.environ['BASE_SHA']]
    monkeypatch.setattr(sys,'argv',argv)
    assert entry.main()==0
    report=json.loads(capsys.readouterr().out)
    assert report['status']=='PASS' and report['policy_valid'] is True
    assert report['protected_changes']==report['existing_test_changes']==[]
    assert report['original_guard_report']=={'status':'SYNTHETIC unchanged baseline report'}


def test_workflow_fallback_and_command_are_the_real_unchanged_values():
    import yaml
    root=Path(__file__).resolve().parents[1]
    steps=yaml.safe_load((root/'.github/workflows/paid-launch.yml').read_text())['jobs']['protected-scope']['steps']
    step=next(s for s in steps if 'BASE_SHA' in s.get('env',{}))
    assert step['env']['BASE_SHA']=="${{ github.event.pull_request.base.sha || '"+PUSH_BASE+"' }}"
    assert step['run']=='python scripts/check_launch_change_scope.py --base "$BASE_SHA" --json-output protected-code-diff-report.json'


@pytest.mark.parametrize('base',['0'*40,'f'*40,'HEAD','main','--help','5fb8e135',''])
def test_arbitrary_comparison_bases_reject(workflow_repo,base):
    fx=workflow_repo
    code,report=fx.scope.run(fx.g,fx.g.ROOT/fx.g.MANIFEST_PATH,base,fx.binding)
    assert code==1 and report['reason_codes']==['COMPARISON_BASE_NOT_APPROVED']


@pytest.mark.parametrize('attack',['changed-manifest','substituted-base','invalid-manifest-hash','missing-manifest-base'])
def test_manifest_is_authenticated_before_its_comparison_base_is_permitted(workflow_repo,monkeypatch,attack):
    fx=workflow_repo;g=fx.g;raw=g.git_bytes
    bad_base='e'*40
    value=json.loads(fx.manifest)
    if attack=='substituted-base':value['base_sha']=bad_base
    elif attack=='missing-manifest-base':value.pop('base_sha')
    else:value['UNREVIEWED']='SYNTHETIC changed document'
    changed=json.dumps(value).encode()
    if attack=='invalid-manifest-hash':changed=b'not JSON, not authenticated'
    monkeypatch.setattr(g,'git_bytes',lambda *a:changed if a==('show',fx.binding['base']+':'+g.MANIFEST_PATH) else raw(*a))
    code,report=fx.scope.run(g,g.ROOT/g.MANIFEST_PATH,bad_base,fx.binding)
    assert code==1 and report['reason_codes']==['BASELINE_CHANGED']


@pytest.mark.parametrize('attack',['dirty','staged','committed','resealed','changed-binding','changed-scope'])
def test_protected_source_and_rehashed_metadata_changes_reject(workflow_repo,attack):
    fx=workflow_repo;g=fx.g
    assert fx.scope.run(g,g.ROOT/g.MANIFEST_PATH,PUSH_BASE,fx.binding)[0]==0
    if attack=='changed-binding':
        changed=dict(fx.binding,manifest_sha256='0'*64)
        fx.write(fx.scope.BINDING,json.dumps(changed).encode());fx.commit('SYNTHETIC changed binding')
    elif attack=='changed-scope':
        fx.write('scripts/ncaaf_future_spread_scope.py',b'# SYNTHETIC substitution\n');fx.commit('SYNTHETIC changed scope')
    else:
        fx.write('core/frozen.py',b'SYNTHETIC unauthorized mutation\n')
        if attack=='staged':g.git('add','core/frozen.py')
        if attack=='committed':fx.commit('SYNTHETIC unauthorized mutation')
        if attack=='resealed':
            g.git('checkout','-q',fx.implementation)
            fx.write('core/frozen.py',b'SYNTHETIC unauthorized mutation\n')
            impl=fx.commit('SYNTHETIC unauthorized implementation')
            fx.write(fx.scope.POLICY,json.dumps(fx.scope.make_policy(g,fx.binding,impl)).encode())
            fx.commit('SYNTHETIC rehashed policy')
    assert fx.scope.run(g,g.ROOT/g.MANIFEST_PATH,PUSH_BASE,fx.binding)[0]!=0


def test_current_successor_retains_exact_predecessor_reconstruction_and_bounds():
    from scripts import check_launch_change_scope as guard
    from scripts import ncaaf_future_spread_scope as scope
    root=Path(__file__).resolve().parents[1]
    binding=json.loads((root/scope.BINDING).read_bytes())
    current=(root/guard.GUARD_PATH).read_bytes().replace(b'\r\n',b'\n')
    assert guard._ncaaf_future_guard_matches(current)
    assert len(current)==8388490<=8*1024*1024
    original=guard._ncaaf_future_base_source(current)
    assert hashlib.sha256(original).hexdigest()==binding['previous_guard_sha256']
    assert not guard._ncaaf_future_guard_matches(current+b'UNREVIEWED')
    assert not hasattr(guard._ncaaf_future_base_source,'cache_info')
    assert binding['postmerge_correction']['proposal_sha256']=='08d51c2b440090e5adcf3ef87bd9a7cea2d9587da1c6c091cb59d6bcebd0bc16'
    from scripts import scope_reconstruction
    assert scope_reconstruction.MAX_CONTEXT_FIELDS==512
    assert scope_reconstruction.MAX_SOURCE_BYTES==8*1024*1024


@pytest.mark.parametrize('attack',[None,'proposal','base','tree','parents','binding-hash','policy-hash',
    'guard-hash','inherited-binding','extra-path','implementation-parent','policy-in-implementation','immutable-file'])
def test_postmerge_successor_binds_merged_predecessor_and_rejects_rehashed_substitutions(attack):
    """Synthetic Git responses exercise the successor envelope without Git history."""
    import copy
    from scripts import check_launch_change_scope as guard
    from scripts import ncaaf_future_spread_scope as scope
    base='f8ee244cea78c96ec244a15cda0753e1b7a09c11'
    tree='6658abdb587fca5f6053f94a4d4e8bb3447ba84f';impl='1'*40
    parents=[PR_BASE,'8d474ce47deab00810441cdb141ab5e306953dee']
    paths=[guard.GUARD_PATH,scope.BINDING,'scripts/ncaaf_future_spread_scope.py',
        'tests/test_ncaaf_postmerge_scope.py','docs/paid-launch/ncaaf-postmerge-scope-correction-v1.txt']
    previous=dict(base=PR_BASE,main_base=PR_BASE,main_tree='SYNTHETIC previous tree',
        paths=['SYNTHETIC/prior.py'],approval_reference='SYNTHETIC prior',
        reviewed_blobs={},manifest_sha256='SYNTHETIC immutable manifest hash')
    raw_binding=json.dumps(previous).encode();raw_policy=b'SYNTHETIC prior policy';raw_guard=b'SYNTHETIC prior guard'
    sha=lambda value:hashlib.sha256(value).hexdigest()
    correction=dict(proposal_sha256='08d51c2b440090e5adcf3ef87bd9a7cea2d9587da1c6c091cb59d6bcebd0bc16',
        base=base,base_tree=tree,ordered_parents=parents,previous_binding_sha256=sha(raw_binding),
        previous_policy_sha256=sha(raw_policy),previous_guard_sha256=sha(raw_guard),paths=paths)
    binding=dict(previous,main_base=base,main_tree=tree,paths=previous['paths']+paths,
        postmerge_correction=correction)
    binding=copy.deepcopy(binding);correction=binding['postmerge_correction']
    if attack in {'proposal','base','tree'}:correction[{'proposal':'proposal_sha256','base':'base','tree':'base_tree'}[attack]]='0'*40
    if attack=='parents':correction['ordered_parents']=list(reversed(parents))
    for name in ['binding','policy','guard']:
        if attack==name+'-hash':correction['previous_'+name+'_sha256']='0'*64
    if attack=='inherited-binding':binding['manifest_sha256']='SYNTHETIC rehashed substitute'
    if attack=='extra-path':correction['paths'].append('UNREVIEWED.py')
    def git(*args):
        if args==('rev-parse',base+'^{tree}'):return tree
        if args==('show','-s','--format=%P',base):return ' '.join(parents)
        if args==('show','-s','--format=%P',impl):return '0'*40 if attack=='implementation-parent' else base
        if args==('diff','--name-only',base,impl):return '\n'.join(paths)
        if args==('diff','--name-only',PR_BASE,impl):return '\n'.join(set(binding['paths'])|{scope.POLICY})
        if args[:2]==('ls-tree','-r'):
            oid=('e' if attack=='immutable-file' and args[2]==impl else 'a')*40
            return '100644 blob '+oid+'\tcore/frozen.py'
        raise AssertionError(('Unexpected synthetic Git request',args))
    def raw(*args):
        if args==('show',base+':'+scope.BINDING):return raw_binding
        if args==('show',base+':'+scope.POLICY):return raw_policy
        if args==('show',base+':'+guard.GUARD_PATH):return raw_guard
        if args==('show',impl+':'+scope.POLICY):return b'UNREVIEWED' if attack=='policy-in-implementation' else raw_policy
        raise AssertionError(('Unexpected synthetic Git bytes request',args))
    g=SimpleNamespace(_require=guard._require,GUARD_PATH=guard.GUARD_PATH,git=git,git_bytes=raw)
    if attack is None:
        assert scope.validate_postmerge_successor(g,binding,impl,correction) is None
    else:
        with pytest.raises(ValueError):scope.validate_postmerge_successor(g,binding,impl,correction)
