"""Exact successor seal for the owner-only NFL storage diagnostic."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import zlib

BASE = 'd55706bfdd9cb182f121bcb352fd9f7a2a30f436'
TREE = '42d65b04ddbcd18d25b5a30f8c483595211c8090'
PARENTS = ['22f4dc219d8360001d54202418abcb78f5c72d98', 'af8d2c399d74cb2cbff6415abf68cd93d18f4597']
OLD_STAMP = b'a2a84b3f99a1db55fd72a1ea68368656601a5c42ed19b63ab7e3a935c1ccafef'
BINDING = 'docs/research/ncaaf-filtered-history-bindings-v1.json'
AUTHORIZATION = 'docs/paid-launch/nfl-storage-diagnostic-v1.md'
POLICY = 'docs/paid-launch/launch-scope-policy-nfl-storage-diagnostic-v1.json'
NORMALIZED = {'scripts/check_launch_change_scope.py', 'scripts/ncaaf_private_refresh_scope.py',
              'scripts/ncaaf_filtered_history_scope.py'}
PATHS = {'app/ui/nfl_private_workflow.py', 'app/ui/nfl_workflow_state.py',
         'app/ui/nfl_storage_diagnostic.py', 'app_core/nfl_storage_diagnostic.py',
         AUTHORIZATION, BINDING, 'scripts/nfl_saved_zip_scope.py',
         'scripts/nfl_storage_diag_scope.py', 'tests/test_nfl_storage_diagnostic.py',
         'tests/test_nfl_storage_diag_scope.py', *NORMALIZED}
MAX_SOURCE = 8 * 1024 * 1024
HISTORY_HELPER = b'def diagnostic_predecessor(binding,relative,raw):\n    """Peel authenticated predecessor bytes; never cache a validation verdict."""\n    record=binding.get(\'nfl_storage_diagnostic_predecessors\',{}).get(relative)\n    if record is None:return raw\n    supplied=raw.replace(b\'\\r\\n\',b\'\\n\')\n    if relative in NORMALIZED and supplied.count(_BINDING_SHA256.encode())!=1:return raw\n    if sha(normalized(relative,supplied))!=record[\'after_sha256\']:return raw\n    key=(_BINDING_SHA256,relative,record[\'after_sha256\'],record[\'before_sha256\'],record[\'before_bytes\'],record[\'before_b85\'])\n    old=_DIAGNOSTIC_IMAGES.get(key)\n    if old is None:\n        z=zlib.decompressobj();old=z.decompress(base64.b85decode(record[\'before_b85\']),MAX_SOURCE+1)\n        require(z.eof and not z.unused_data and type(old) is bytes and len(old)==record[\'before_bytes\']<=MAX_SOURCE\n            and sha(old)==record[\'before_sha256\'],\'NFL_DIAGNOSTIC_PREDECESSOR_CHANGED\')\n        while _DIAGNOSTIC_IMAGES and (len(_DIAGNOSTIC_IMAGES)>=16 or\n                sum(len(value) for value in _DIAGNOSTIC_IMAGES.values())+len(old)>_LEGACY_IMAGE_LIMIT):\n            _DIAGNOSTIC_IMAGES.pop(next(iter(_DIAGNOSTIC_IMAGES)))\n        if len(old)<=_LEGACY_IMAGE_LIMIT:_DIAGNOSTIC_IMAGES[key]=old\n    require(type(old) is bytes and len(old)==record[\'before_bytes\']<=MAX_SOURCE\n        and sha(old)==record[\'before_sha256\'],\'NFL_DIAGNOSTIC_PREDECESSOR_CHANGED\')\n    return old\n\n\n'
HISTORY_CACHE_MARKER = b'_PREDECESSOR_IMAGES={} # Bounded immutable bytes only; never validation decisions or modules.\n'
HISTORY_CACHE_ADDITION = b'_DIAGNOSTIC_IMAGES={} # Same bounded, reauthenticated predecessor bytes; no cached verdicts.\n'
HISTORY_AUTH_MARKER = b'    return b\n\n\n\ndef predecessor_generation('
HISTORY_AUTH_ADDITION = b"    for relative in b.get('nfl_storage_diagnostic_predecessors',{}):\n        require(relative in b['components'],'NFL_DIAGNOSTIC_PREDECESSOR_CHANGED')\n        diagnostic_predecessor(b,relative,physical(root/relative))\n"
HISTORY_OLD_CALL = b"    b=authenticate(root);record=b['component_predecessors'].get(relative)\n"
HISTORY_NEW_CALL = b"    b=authenticate(root);raw=diagnostic_predecessor(b,relative,raw);record=b['component_predecessors'].get(relative)\n"
HISTORY_OLD_COND = b"    if relative in NORMALIZED and supplied.count(_BINDING_SHA256.encode())!=1:return raw\n"
HISTORY_NEW_COND = b"    if relative in NORMALIZED and supplied.count(_BINDING_SHA256.encode())!=1:\n        prior=b.get('nfl_storage_diagnostic',{}).get('prior_binding_sha256','').encode()\n        if not prior or supplied.count(prior)!=1:return raw\n        supplied=supplied.replace(prior,_BINDING_SHA256.encode())\n"
DISPATCH_HELPER = '''def _storage_diagnostic_module(root, binding):
    import io
    import types
    path = Path(root) / 'scripts/nfl_storage_diag_scope.py'
    with io.FileIO(path, 'r') as stream:
        raw = stream.read(8 * 1024 * 1024 + 1)
    require(len(raw) <= 8 * 1024 * 1024, 'NFL_DIAGNOSTIC_SCOPE_LIMIT')
    raw = raw.replace(b'\\r\\n', b'\\n')
    require(hashlib.sha256(raw).hexdigest() == binding['components']['scripts/nfl_storage_diag_scope.py'], 'NFL_DIAGNOSTIC_SCOPE_CHANGED')
    module = types.ModuleType('exact_nfl_storage_diag_scope'); module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


'''.encode()
DISPATCH_POLICY = b"    if binding.get('nfl_storage_diagnostic') is not None:\n        return _storage_diagnostic_module(root, binding).make_policy(root, binding, implementation, binding_sha)\n"
DISPATCH_VALIDATE = b"    if binding.get('nfl_storage_diagnostic') is not None:\n        return _storage_diagnostic_module(root, binding).validate(root, binding, comparison_base, manifest, binding_sha=binding_sha)\n"


def require(ok, code):
    if not ok:
        raise ValueError(code)


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root)


def text(root, *args):
    return git(root, *args).decode().strip()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def blobs(root, revision):
    return {p: m.split()[2] for line in text(root, 'ls-tree', '-r', revision).splitlines()
            for m, p in [line.split('\t', 1)]}


def normal(path, raw, stamp):
    raw = raw.replace(b'\r\n', b'\n')
    if path in NORMALIZED:
        require(raw.count(stamp.encode()) == 1, 'NFL_DIAGNOSTIC_STAMP_CHANGED')
        raw = raw.replace(stamp.encode(), b'0' * 64)
    return raw


def make_policy(root, binding, implementation, binding_sha):
    c = binding['nfl_storage_diagnostic']
    return dict(version='nfl-storage-diagnostic-seal-v1', base=BASE, base_tree=TREE,
        authorization_sha256=c['authorization_sha256'], binding_sha256=binding_sha,
        implementation_commit=implementation, implementation_tree=text(root, 'rev-parse', implementation+'^{tree}'),
        implementation_paths=c['implementation_paths'],
        unchanged_blobs={p:v for p,v in blobs(root, BASE).items() if p not in PATHS})


def validate(root, binding, comparison_base, manifest, *, binding_sha):
    root = Path(root)
    c = binding['nfl_storage_diagnostic']
    require(set(c) == {'base', 'base_tree', 'prior_binding_sha256', 'authorization_sha256', 'implementation_paths'},
            'NFL_DIAGNOSTIC_SCOPE_CHANGED')
    require(c['base'] == BASE and c['base_tree'] == TREE and c['prior_binding_sha256'] == OLD_STAMP.decode(),
            'NFL_DIAGNOSTIC_BASE_CHANGED')
    require(text(root, 'rev-parse', BASE+'^{tree}') == TREE and
            text(root, 'show', '-s', '--format=%P', BASE).split() == PARENTS, 'NFL_DIAGNOSTIC_BASE_CHANGED')
    inherited_manifest = json.loads(git(root, 'show', BASE+':'+binding['baseline_manifest']))
    require(comparison_base in (None, BASE, inherited_manifest['base_sha']), 'COMPARISON_BASE_NOT_APPROVED')
    require(manifest is None or Path(manifest).resolve() == (root/binding['baseline_manifest']).resolve(),
            'MANIFEST_PATH_NOT_APPROVED')
    require(not text(root, 'diff', '--name-only') and not text(root, 'diff', '--cached', '--name-only')
            and not text(root, 'ls-files', '--others', '--exclude-standard'), 'UNAUTHORIZED_CHECKOUT_CHANGE')
    require(set(c['implementation_paths']) == PATHS and len(c['implementation_paths']) == len(PATHS),
            'NFL_DIAGNOSTIC_PATHS_CHANGED')
    prior_raw = git(root, 'show', BASE+':'+BINDING)
    require(sha(prior_raw) == OLD_STAMP.decode(), 'NFL_DIAGNOSTIC_PREDECESSOR_CHANGED')
    prior = json.loads(prior_raw)
    unchanged = copy.deepcopy(binding)
    for field in ('components', 'nfl_storage_diagnostic', 'nfl_storage_diagnostic_predecessors'):
        unchanged.pop(field, None)
    expected = copy.deepcopy(prior); expected.pop('components')
    require(unchanged == expected, 'NFL_DIAGNOSTIC_HISTORY_CHANGED')
    expected_components = copy.deepcopy(prior['components'])
    before = blobs(root, BASE)
    predecessors = {}
    for path in PATHS - {BINDING}:
        raw = (root/path).read_bytes().replace(b'\r\n', b'\n')
        require(len(raw) <= MAX_SOURCE, 'NFL_DIAGNOSTIC_SCOPE_LIMIT')
        digest = sha(normal(path, raw, binding_sha))
        expected_components[path] = digest
        if path in prior['components'] and path in before:
            old = git(root, 'show', BASE+':'+path)
            predecessors[path] = dict(before_sha256=sha(old), before_bytes=len(old),
                before_b85=base64.b85encode(zlib.compress(old, 9)).decode(), after_sha256=digest)
    require(binding['components'] == expected_components, 'NFL_DIAGNOSTIC_COMPONENTS_CHANGED')
    require(binding['nfl_storage_diagnostic_predecessors'] == predecessors, 'NFL_DIAGNOSTIC_PREDECESSOR_CHANGED')
    require(c['authorization_sha256'] == binding['components'][AUTHORIZATION], 'NFL_DIAGNOSTIC_AUTHORITY_CHANGED')
    for path in NORMALIZED - {'scripts/ncaaf_filtered_history_scope.py'}:
        old = git(root, 'show', BASE+':'+path)
        new = (root/path).read_bytes().replace(b'\r\n', b'\n')
        require(old.count(OLD_STAMP) == new.count(binding_sha.encode()) == 1 and
                old.replace(OLD_STAMP, b'0'*64) == new.replace(binding_sha.encode(), b'0'*64),
                'INHERITED_GUARD_LOGIC_CHANGED')
    history = 'scripts/ncaaf_filtered_history_scope.py'
    old = git(root, 'show', BASE+':'+history).replace(OLD_STAMP, binding_sha.encode())
    require(old.count(b'def predecessor_source(') == old.count(HISTORY_OLD_CALL) == old.count(HISTORY_OLD_COND)
            == old.count(HISTORY_CACHE_MARKER) == old.count(HISTORY_AUTH_MARKER) == 1,
            'INHERITED_GUARD_LOGIC_CHANGED')
    expected_history = old.replace(HISTORY_OLD_CALL, HISTORY_NEW_CALL, 1).replace(HISTORY_OLD_COND, HISTORY_NEW_COND, 1)
    expected_history = expected_history.replace(HISTORY_CACHE_MARKER, HISTORY_CACHE_MARKER+HISTORY_CACHE_ADDITION, 1)
    expected_history = expected_history.replace(HISTORY_AUTH_MARKER, HISTORY_AUTH_ADDITION+HISTORY_AUTH_MARKER, 1)
    expected_history = expected_history.replace(b'def predecessor_source(', HISTORY_HELPER+b'def predecessor_source(', 1)
    require((root/history).read_bytes().replace(b'\r\n', b'\n') == expected_history, 'INHERITED_GUARD_LOGIC_CHANGED')
    dispatch = 'scripts/nfl_saved_zip_scope.py'
    old = git(root, 'show', BASE+':'+dispatch)
    policy_line = b'def make_policy(root, binding, implementation, binding_sha):\n'
    validate_line = b'def validate(root, binding, comparison_base, manifest, *, binding_sha):\n'
    require(old.count(policy_line) == old.count(validate_line) == 1, 'INHERITED_GUARD_LOGIC_CHANGED')
    expected_dispatch = old.replace(policy_line, DISPATCH_HELPER+policy_line+DISPATCH_POLICY, 1)
    expected_dispatch = expected_dispatch.replace(validate_line, validate_line+DISPATCH_VALIDATE, 1)
    require((root/dispatch).read_bytes().replace(b'\r\n', b'\n') == expected_dispatch, 'INHERITED_GUARD_LOGIC_CHANGED')
    policy = json.loads(git(root, 'show', 'HEAD:'+POLICY))
    implementation = policy['implementation_commit']
    require(text(root, 'show', '-s', '--format=%P', implementation).split() == [BASE],
            'IMPLEMENTATION_PARENT_NOT_APPROVED')
    require(set(text(root, 'diff', '--name-only', BASE, implementation).splitlines()) == PATHS,
            'IMPLEMENTATION_CHANGE_SET_NOT_APPROVED')
    require(policy == make_policy(root, binding, implementation, binding_sha), 'EXACT_POLICY_BINDINGS_CHANGED')
    head = text(root, 'rev-parse', 'HEAD')
    parents = text(root, 'show', '-s', '--format=%P', head).split()
    if len(parents) == 2:
        require(parents[0] == BASE, 'TESTED_MERGE_PARENTS_CHANGED')
        seal = parents[1]
        require(text(root, 'rev-parse', seal+'^{tree}') == text(root, 'rev-parse', head+'^{tree}'),
                'TESTED_TREE_CHANGED')
    else:
        seal = head
    require(text(root, 'show', '-s', '--format=%P', seal).split() == [implementation] and
            text(root, 'diff', '--name-status', implementation, seal) == 'A\t'+POLICY,
            'SEAL_CHANGE_SET_NOT_APPROVED')
    actual = blobs(root, 'HEAD')
    require(POLICY not in before and set(actual) == set(before) | (PATHS-set(before)) | {POLICY}, 'UNREVIEWED_PATH')
    require(all(actual.get(p) == v for p,v in policy['unchanged_blobs'].items()), 'IMMUTABLE_FILE_CHANGED')
    for path, value in actual.items():
        raw = (root/path).read_bytes()
        def blob(data): return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        require(value in {blob(raw), blob(raw.replace(b'\r\n', b'\n'))}, 'UNAUTHORIZED_CHECKOUT_CHANGE')
    return dict(schema_version=95, status='PASS', base=BASE, implementation=implementation, seal=seal,
        candidate=head, tree=text(root, 'rev-parse', head+'^{tree}'), changed_files=sorted(PATHS|{POLICY}),
        protected_changes=[], existing_test_changes=[], unchanged_file_count=len(policy['unchanged_blobs']),
        inherited_runtime_and_readers_unchanged=True, accepted=False, inference=False)
