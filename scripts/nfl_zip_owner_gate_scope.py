"""Exact linear correction of the preserved failed ZIP owner-gate head."""
import hashlib
import json
from pathlib import Path
import subprocess

BASE = 'd46528beb86752c81846c81aa944f603ff985a64'
TREE = '5cf5876d9af71f52aa02cb566dafcb0a6f3dc593'
MAIN = 'c273e3fa072cd6f161fb02abede159593ad07ce6'
PATCH = 'bd8829057e72c4de09e73477a688a0d3e531eb6759d6b9f488dd7499a74dc9b5'
PARENT_BINDING = '84a97bcca548455c4aa3a195fee16fd05e9a0e54c913439bb604d19985efa962'
AUTHORIZATION = 'docs/paid-launch/nfl-saved-zip-owner-gate-correction-v1.md'
POLICY = 'docs/paid-launch/launch-scope-policy-nfl-saved-zip-owner-gate-v1.json'
BINDING = 'docs/research/ncaaf-filtered-history-bindings-v1.json'
PATHS = {
    'app/ui/publish_panel.py', 'tests/test_nfl_saved_zip_owner_gate.py',
    'tests/test_nfl_zip_owner_gate_scope.py', AUTHORIZATION, BINDING,
    'scripts/nfl_saved_zip_scope.py', 'scripts/nfl_zip_owner_gate_scope.py',
    'scripts/check_launch_change_scope.py', 'scripts/ncaaf_private_refresh_scope.py',
    'scripts/ncaaf_filtered_history_scope.py',
}
DISPATCH_HELPER = """def _owner_gate_module(root, binding):
    import io
    import types
    path = Path(root) / 'scripts/nfl_zip_owner_gate_scope.py'
    with io.FileIO(path, 'r') as stream:
        raw = stream.read(8 * 1024 * 1024 + 1)
    require(len(raw) <= 8 * 1024 * 1024, 'OWNER_GATE_SCOPE_LIMIT')
    raw = raw.replace(b'\\r\\n', b'\\n')
    require(hashlib.sha256(raw).hexdigest() == binding['components']['scripts/nfl_zip_owner_gate_scope.py'], 'OWNER_GATE_SCOPE_CHANGED')
    module = types.ModuleType('exact_zip_owner_gate_scope'); module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


""".encode()
DISPATCH_BRANCHES = (
    b"    if binding.get('saved_nfl_zip_owner_gate_correction') is not None:\n        return _owner_gate_module(root, binding).make_policy(root, binding, implementation, binding_sha)\n",
    b"    if binding.get('saved_nfl_zip_owner_gate_correction') is not None:\n        return _owner_gate_module(root, binding).validate(root, binding, comparison_base, manifest, binding_sha=binding_sha)\n",
)


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root)


def text(root, *args):
    return git(root, *args).decode().strip()


def blobs(root, rev):
    return {p: m.split()[2] for line in text(root, 'ls-tree', '-r', rev).splitlines()
            for m, p in [line.split('\t', 1)]}


def make_policy(root, binding, implementation, binding_sha):
    c = binding['saved_nfl_zip_owner_gate_correction']
    return dict(version='nfl-saved-zip-owner-gate-seal-v1', base=c['base'], base_tree=c['base_tree'],
        patch_sha256=c['patch_sha256'], authorization_sha256=c['authorization_sha256'],
        parent_binding_sha256=c['parent_binding_sha256'], binding_sha256=binding_sha,
        implementation_commit=implementation, implementation_tree=text(root, 'rev-parse', implementation+'^{tree}'),
        implementation_paths=c['implementation_paths'],
        unchanged_blobs={p: v for p, v in blobs(root, c['base']).items() if p not in PATHS})


def validate(root, binding, comparison_base, manifest, *, binding_sha):
    root = Path(root); c = binding['saved_nfl_zip_owner_gate_correction']
    require(set(c) == {'base','base_tree','main','patch_sha256','authorization_sha256','parent_binding_sha256','implementation_paths'}, 'OWNER_GATE_CONTRACT_CHANGED')
    require(c['base'] == BASE and c['base_tree'] == TREE and c['main'] == MAIN
            and text(root, 'rev-parse', BASE+'^{tree}') == TREE
            and text(root, 'merge-base', MAIN, BASE) == MAIN, 'OWNER_GATE_BASE_CHANGED')
    require(c['patch_sha256'] == PATCH and c['parent_binding_sha256'] == PARENT_BINDING, 'OWNER_GATE_AUTHORITY_CHANGED')
    require(hashlib.sha256(git(root, 'show', BASE+':'+BINDING)).hexdigest() == PARENT_BINDING, 'OWNER_GATE_PARENT_BINDING_CHANGED')
    old_manifest = json.loads(git(root, 'show', BASE+':'+binding['baseline_manifest']))
    require(comparison_base in {None, BASE, MAIN, old_manifest['base_sha']}, 'COMPARISON_BASE_NOT_APPROVED')
    require(manifest is None or Path(manifest).resolve() == (root/binding['baseline_manifest']).resolve(), 'MANIFEST_PATH_NOT_APPROVED')
    require(set(c['implementation_paths']) == PATHS and len(c['implementation_paths']) == len(PATHS), 'OWNER_GATE_PATHS_CHANGED')
    require(c['authorization_sha256'] == binding['components'][AUTHORIZATION], 'OWNER_GATE_AUTHORITY_CHANGED')
    require(not text(root, 'diff', '--name-only') and not text(root, 'diff', '--cached', '--name-only')
            and not text(root, 'ls-files', '--others', '--exclude-standard'), 'UNAUTHORIZED_CHECKOUT_CHANGE')
    policy = json.loads(git(root, 'show', 'HEAD:'+POLICY)); implementation = policy['implementation_commit']
    require(text(root, 'show', '-s', '--format=%P', implementation).split() == [BASE], 'IMPLEMENTATION_PARENT_NOT_APPROVED')
    require(set(text(root, 'diff', '--name-only', BASE, implementation).splitlines()) == PATHS, 'IMPLEMENTATION_CHANGE_SET_NOT_APPROVED')
    require(policy == make_policy(root, binding, implementation, binding_sha), 'EXACT_POLICY_BINDINGS_CHANGED')
    before = blobs(root, BASE); require(POLICY not in before, 'SUCCESSOR_POLICY_ALREADY_EXISTS')
    path = 'app/ui/publish_panel.py'; old = git(root, 'show', BASE+':'+path)
    needle = b'revoke_saved_viewer(st.session_state)'
    require(old.count(needle) == 2 and git(root, 'show', implementation+':'+path) ==
            old.replace(needle, b"revoke_saved_viewer(getattr(st, 'session_state', {}))"), 'OWNER_GATE_CORRECTION_CHANGED')
    for path in ('scripts/check_launch_change_scope.py','scripts/ncaaf_private_refresh_scope.py','scripts/ncaaf_filtered_history_scope.py'):
        old = git(root, 'show', BASE+':'+path); new = git(root, 'show', implementation+':'+path)
        require(old.count(PARENT_BINDING.encode()) == new.count(binding_sha.encode()) == 1
                and old.replace(PARENT_BINDING.encode(), b'0'*64) == new.replace(binding_sha.encode(), b'0'*64), 'INHERITED_GUARD_LOGIC_CHANGED')
    path = 'scripts/nfl_saved_zip_scope.py'; old = git(root, 'show', BASE+':'+path); new = git(root, 'show', implementation+':'+path)
    for addition in (DISPATCH_HELPER, *DISPATCH_BRANCHES):
        require(new.count(addition) == 1, 'OWNER_GATE_DISPATCH_CHANGED'); new = new.replace(addition, b'')
    require(new == old, 'INHERITED_GUARD_LOGIC_CHANGED')
    head = text(root, 'rev-parse', 'HEAD'); parents = text(root, 'show', '-s', '--format=%P', head).split()
    if len(parents) == 2:
        require(parents[0] == MAIN, 'TESTED_MERGE_PARENTS_CHANGED'); seal = parents[1]
        require(text(root, 'rev-parse', seal+'^{tree}') == text(root, 'rev-parse', head+'^{tree}'), 'TESTED_TREE_CHANGED')
    else:
        seal = head
    require(text(root, 'show', '-s', '--format=%P', seal).split() == [implementation]
            and text(root, 'diff', '--name-status', implementation, seal) == 'A\t'+POLICY, 'SEAL_CHANGE_SET_NOT_APPROVED')
    actual = blobs(root, 'HEAD')
    require(set(actual) == set(before) | (PATHS-set(before)) | {POLICY}, 'UNREVIEWED_PATH')
    require(all(actual.get(p) == v for p, v in policy['unchanged_blobs'].items()), 'IMMUTABLE_FILE_CHANGED')
    for path, value in actual.items():
        raw = (root/path).read_bytes()
        blob = lambda b: hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
        require(value in {blob(raw), blob(raw.replace(b'\r\n', b'\n'))}, 'UNAUTHORIZED_CHECKOUT_CHANGE')
    return dict(schema_version=93, status='PASS', base=MAIN, preserved_failed_head=BASE,
        implementation=implementation, seal=seal, candidate=head, tree=text(root,'rev-parse',head+'^{tree}'),
        changed_files=sorted(PATHS|{POLICY}), protected_changes=[], existing_test_changes=[],
        inherited_runtime_and_readers_unchanged=True, unchanged_file_count=len(policy['unchanged_blobs']),
        accepted=False, inference=False)
