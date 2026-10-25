"""Exact read-only saved-viewer successor; all inherited runtime stays frozen."""
import hashlib
import json
from pathlib import Path
import subprocess

BASE = '792abe61a8e00992290abe752eb17825b7bbd0b6'
TREE = 'de0f0762b784321cdda4fc8a72a6bfc773a96f13'
POLICY = 'docs/paid-launch/launch-scope-policy-nfl-saved-scenarios-v1.json'
PATHS = {
    'app/ui/nfl_private_research.py', 'app/ui/nfl_saved_scenarios.py',
    'app_core/nfl_saved_scenarios.py', 'tests/test_nfl_saved_scenarios.py',
    'tests/test_nfl_saved_scenario_scope.py',
    'docs/paid-launch/nfl-saved-scenarios-v1.md',
    'docs/research/ncaaf-filtered-history-bindings-v1.json',
    'scripts/check_launch_change_scope.py', 'scripts/ncaaf_filtered_history_scope.py',
    'scripts/ncaaf_private_refresh_scope.py', 'scripts/nfl_saved_scenario_scope.py',
}
OLD_STAMP = b'2120605e490bb310f9d39e3256d43c8f4c3f3f28b836d21d98b274657271569c'


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
    c = binding['saved_nfl_scenario_viewer']
    return dict(version='nfl-saved-scenario-viewer-seal-v1', base=c['base'], base_tree=c['base_tree'],
        authorization_sha256=c['authorization_sha256'], binding_sha256=binding_sha,
        implementation_commit=implementation, implementation_tree=text(root, 'rev-parse', implementation + '^{tree}'),
        implementation_paths=c['implementation_paths'],
        unchanged_blobs={p: v for p, v in blobs(root, c['base']).items() if p not in c['implementation_paths']})


def validate(root, binding, comparison_base, manifest, *, binding_sha):
    root = Path(root)
    c = binding['saved_nfl_scenario_viewer']
    require(set(c) == {'base', 'base_tree', 'implementation_paths', 'authorization_sha256'}, 'SAVED_VIEWER_SCOPE_CHANGED')
    require(c['base'] == BASE and c['base_tree'] == TREE
            and text(root, 'rev-parse', BASE + '^{tree}') == TREE, 'SAVED_VIEWER_BASE_CHANGED')
    require(set(c['implementation_paths']) == PATHS and len(c['implementation_paths']) == len(PATHS), 'SAVED_VIEWER_PATHS_CHANGED')
    require(comparison_base in {None, BASE, json.loads(git(root, 'show', BASE + ':' + binding['baseline_manifest']))['base_sha']},
            'COMPARISON_BASE_NOT_APPROVED')
    require(manifest is None or Path(manifest).resolve() == (root / binding['baseline_manifest']).resolve(), 'MANIFEST_PATH_NOT_APPROVED')
    require(not text(root, 'diff', '--name-only') and not text(root, 'diff', '--cached', '--name-only')
            and not text(root, 'ls-files', '--others', '--exclude-standard'), 'UNAUTHORIZED_CHECKOUT_CHANGE')
    require(c['authorization_sha256'] == binding['components']['docs/paid-launch/nfl-saved-scenarios-v1.md'], 'SAVED_VIEWER_AUTHORIZATION_CHANGED')
    policy = json.loads(git(root, 'show', 'HEAD:' + POLICY))
    implementation = policy['implementation_commit']
    require(text(root, 'show', '-s', '--format=%P', implementation).split() == [BASE], 'IMPLEMENTATION_PARENT_NOT_APPROVED')
    require(set(text(root, 'diff', '--name-only', BASE, implementation).splitlines()) == PATHS, 'IMPLEMENTATION_CHANGE_SET_NOT_APPROVED')
    require(policy == make_policy(root, binding, implementation, binding_sha), 'EXACT_POLICY_BINDINGS_CHANGED')
    before = blobs(root, BASE)
    require(POLICY not in before, 'SUCCESSOR_POLICY_ALREADY_EXISTS')
    # Guard dispatch and private-Refresh authentication change only their stamp.
    for name in ('scripts/check_launch_change_scope.py', 'scripts/ncaaf_private_refresh_scope.py'):
        old = git(root, 'show', BASE + ':' + name)
        new = git(root, 'show', implementation + ':' + name)
        require(old.count(OLD_STAMP) == new.count(binding_sha.encode()) == 1
                and old.replace(OLD_STAMP, b'0' * 64) == new.replace(binding_sha.encode(), b'0' * 64), 'INHERITED_GUARD_LOGIC_CHANGED')
    head = text(root, 'rev-parse', 'HEAD')
    parents = text(root, 'show', '-s', '--format=%P', head).split()
    if len(parents) == 2:
        require(parents[0] == BASE, 'TESTED_MERGE_PARENTS_CHANGED')
        seal = parents[1]
        require(text(root, 'rev-parse', seal + '^{tree}') == text(root, 'rev-parse', head + '^{tree}'), 'TESTED_TREE_CHANGED')
    else:
        seal = head
    require(text(root, 'show', '-s', '--format=%P', seal).split() == [implementation]
            and text(root, 'diff', '--name-status', implementation, seal) == 'A\t' + POLICY, 'SEAL_CHANGE_SET_NOT_APPROVED')
    actual = blobs(root, 'HEAD')
    require(set(actual) == set(before) | (PATHS - set(before)) | {POLICY}, 'UNREVIEWED_PATH')
    require(all(actual.get(p) == v for p, v in policy['unchanged_blobs'].items()), 'IMMUTABLE_FILE_CHANGED')
    for path, value in actual.items():
        raw = (root / path).read_bytes()
        blob = lambda b: hashlib.sha1(b'blob ' + str(len(b)).encode() + b'\0' + b).hexdigest()
        require(value in {blob(raw), blob(raw.replace(b'\r\n', b'\n'))}, 'UNAUTHORIZED_CHECKOUT_CHANGE')
    return dict(schema_version=91, status='PASS', base=BASE, implementation=implementation, seal=seal,
        candidate=head, tree=text(root, 'rev-parse', head + '^{tree}'), changed_files=sorted(PATHS | {POLICY}),
        protected_changes=[], existing_test_changes=[], inherited_runtime_and_readers_unchanged=True,
        unchanged_file_count=len(policy['unchanged_blobs']), accepted=False, inference=False)
