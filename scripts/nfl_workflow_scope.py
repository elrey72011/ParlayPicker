"""Exact owner-authorized integrated private NFL successor; no live authority."""
import hashlib
import json
from pathlib import Path
import subprocess

BASE = '22f4dc219d8360001d54202418abcb78f5c72d98'
TREE = 'b9153afba757718cc48fef853da75a3f99f19049'
OLD_STAMP = b'ae4cdb0b5000dd4e65f7cd0ec1eb067d8ce7f3b29f123409231a073dd5a755e3'
AUTHORIZATION = 'docs/paid-launch/private-nfl-streamlit-workflow-v1.md'
POLICY = 'docs/paid-launch/launch-scope-policy-private-nfl-workflow-v1.json'
BINDING = 'docs/research/ncaaf-filtered-history-bindings-v1.json'
NORMALIZED = {'scripts/check_launch_change_scope.py', 'scripts/ncaaf_private_refresh_scope.py', 'scripts/ncaaf_filtered_history_scope.py'}
PATHS = {
    'app/ui/publish_panel.py', 'app/ui/nfl_private_workflow.py', 'app/ui/nfl_workflow_state.py',
    'app_core/nfl_private_workflow.py', 'app_core/nfl_workflow_acquisition.py',
    'app_core/nfl_workflow_inputs.py', 'app_core/nfl_workflow_store.py',
    'requirements.txt', '.github/workflows/subscriber-completion-postgres.yml',
    'docs/research/nfl-private-workflow-store-v1.sql', AUTHORIZATION, BINDING,
    'tests/nfl_private_workflow_browser.py', 'tests/nfl_workflow_fixtures.py',
    'tests/paid_launch/case_nfl_workflow_store.py', 'tests/test_nfl_private_workflow.py',
    'tests/test_nfl_private_workflow_ui.py', 'tests/test_nfl_workflow_scope.py',
    'scripts/nfl_workflow_scope.py', 'scripts/nfl_saved_zip_scope.py', *NORMALIZED,
}
DISPATCH_HELPER = '''def _workflow_module(root, binding):
    import io
    import types
    path = Path(root) / 'scripts/nfl_workflow_scope.py'
    with io.FileIO(path, 'r') as stream:
        raw = stream.read(8 * 1024 * 1024 + 1)
    require(len(raw) <= 8 * 1024 * 1024, 'NFL_WORKFLOW_SCOPE_LIMIT')
    raw = raw.replace(b'\\r\\n', b'\\n')
    require(hashlib.sha256(raw).hexdigest() == binding['components']['scripts/nfl_workflow_scope.py'], 'NFL_WORKFLOW_SCOPE_CHANGED')
    module = types.ModuleType('exact_nfl_workflow_scope'); module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


'''.encode()
DISPATCH_BRANCHES = (
    b"    if binding.get('private_nfl_streamlit_workflow') is not None:\n        return _workflow_module(root, binding).make_policy(root, binding, implementation, binding_sha)\n",
    b"    if binding.get('private_nfl_streamlit_workflow') is not None:\n        return _workflow_module(root, binding).validate(root, binding, comparison_base, manifest, binding_sha=binding_sha)\n",
)


def require(ok, code):
    if not ok:
        raise ValueError(code)


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root)


def text(root, *args):
    return git(root, *args).decode().strip()


def blobs(root, rev):
    return {p: m.split()[2] for line in text(root, 'ls-tree', '-r', rev).splitlines()
            for m, p in [line.split('\t', 1)]}


def make_policy(root, binding, implementation, binding_sha):
    c = binding['private_nfl_streamlit_workflow']
    return dict(version='private-nfl-streamlit-workflow-seal-v1', base=c['base'], base_tree=c['base_tree'],
        authorization_sha256=c['authorization_sha256'], binding_sha256=binding_sha,
        implementation_commit=implementation, implementation_tree=text(root, 'rev-parse', implementation+'^{tree}'),
        implementation_paths=c['implementation_paths'],
        unchanged_blobs={p:v for p,v in blobs(root,c['base']).items() if p not in PATHS})


def validate(root, binding, comparison_base, manifest, *, binding_sha):
    root=Path(root); c=binding['private_nfl_streamlit_workflow']
    require(set(c)=={'base','base_tree','authorization_sha256','implementation_paths'}, 'NFL_WORKFLOW_SCOPE_CHANGED')
    require(c['base']==BASE and c['base_tree']==TREE and text(root,'rev-parse',BASE+'^{tree}')==TREE,'NFL_WORKFLOW_BASE_CHANGED')
    require(set(c['implementation_paths'])==PATHS and len(c['implementation_paths'])==len(PATHS),'NFL_WORKFLOW_PATHS_CHANGED')
    prior=json.loads(git(root,'show',BASE+':'+binding['baseline_manifest']))
    require(comparison_base in {None,BASE,prior['base_sha']},'COMPARISON_BASE_NOT_APPROVED')
    require(manifest is None or Path(manifest).resolve()==(root/binding['baseline_manifest']).resolve(),'MANIFEST_PATH_NOT_APPROVED')
    require(not text(root,'diff','--name-only') and not text(root,'diff','--cached','--name-only')
            and not text(root,'ls-files','--others','--exclude-standard'),'UNAUTHORIZED_CHECKOUT_CHANGE')
    require(c['authorization_sha256']==binding['components'][AUTHORIZATION],'NFL_WORKFLOW_AUTHORITY_CHANGED')
    policy=json.loads(git(root,'show','HEAD:'+POLICY)); implementation=policy['implementation_commit']
    require(text(root,'show','-s','--format=%P',implementation).split()==[BASE],'IMPLEMENTATION_PARENT_NOT_APPROVED')
    require(set(text(root,'diff','--name-only',BASE,implementation).splitlines())==PATHS,'IMPLEMENTATION_CHANGE_SET_NOT_APPROVED')
    require(policy==make_policy(root,binding,implementation,binding_sha),'EXACT_POLICY_BINDINGS_CHANGED')
    for path in NORMALIZED:
        old=git(root,'show',BASE+':'+path);new=git(root,'show',implementation+':'+path)
        require(old.count(OLD_STAMP)==new.count(binding_sha.encode())==1
                and old.replace(OLD_STAMP,b'0'*64)==new.replace(binding_sha.encode(),b'0'*64),'INHERITED_GUARD_LOGIC_CHANGED')
    path='scripts/nfl_saved_zip_scope.py';old=git(root,'show',BASE+':'+path);new=git(root,'show',implementation+':'+path)
    for addition in (DISPATCH_HELPER,*DISPATCH_BRANCHES):
        require(new.count(addition)==1,'NFL_WORKFLOW_DISPATCH_CHANGED');new=new.replace(addition,b'')
    require(new==old,'INHERITED_GUARD_LOGIC_CHANGED')
    old=git(root,'show',BASE+':requirements.txt');new=git(root,'show',implementation+':requirements.txt')
    require(new==old.replace(b'# Direct cPanel publication',b'# Private NFL workflow: durable atomic claims/results; same existing service pin.\npsycopg[binary]==3.2.10\n\n# Direct cPanel publication'),'NFL_WORKFLOW_DEPENDENCIES_CHANGED')
    path='.github/workflows/subscriber-completion-postgres.yml';old=git(root,'show',BASE+':'+path);new=git(root,'show',implementation+':'+path)
    require(new==old.replace(b'Run the three subscriber completion cases',b'Run subscriber completion and private NFL storage cases').replace(
        b'          --junitxml=subscriber-completion-tests.xml',b'          tests/paid_launch/case_nfl_workflow_store.py\n          --junitxml=subscriber-completion-tests.xml'),'NFL_WORKFLOW_CI_LIMITS_CHANGED')
    head=text(root,'rev-parse','HEAD');parents=text(root,'show','-s','--format=%P',head).split()
    if len(parents)==2:
        require(parents[0]==BASE,'TESTED_MERGE_PARENTS_CHANGED');seal=parents[1]
        require(text(root,'rev-parse',seal+'^{tree}')==text(root,'rev-parse',head+'^{tree}'),'TESTED_TREE_CHANGED')
    else:
        seal=head
    require(text(root,'show','-s','--format=%P',seal).split()==[implementation]
        and text(root,'diff','--name-status',implementation,seal)=='A\t'+POLICY,'SEAL_CHANGE_SET_NOT_APPROVED')
    before=blobs(root,BASE);actual=blobs(root,'HEAD')
    require(POLICY not in before and set(actual)==set(before)|(PATHS-set(before))|{POLICY},'UNREVIEWED_PATH')
    require(all(actual.get(p)==v for p,v in policy['unchanged_blobs'].items()),'IMMUTABLE_FILE_CHANGED')
    for path,value in actual.items():
        raw=(root/path).read_bytes()
        def blob(b):return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
        require(value in {blob(raw),blob(raw.replace(b'\r\n',b'\n'))},'UNAUTHORIZED_CHECKOUT_CHANGE')
    return dict(schema_version=94,status='PASS',base=BASE,implementation=implementation,seal=seal,candidate=head,
        tree=text(root,'rev-parse',head+'^{tree}'),changed_files=sorted(PATHS|{POLICY}),protected_changes=[],
        existing_test_changes=[],unchanged_file_count=len(policy['unchanged_blobs']),
        inherited_runtime_and_readers_unchanged=True,accepted=False,inference=False)
