"""Authenticated data-only timing successor; predecessor expectations stay frozen."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import zlib

BASE = '95d227f82984737d7c822a8c0fb905cbe39e69d3'
TREE = '01dd7973b28ca3c66a4c61ddd883d13dd0a07f44'
PARENTS = ['00e666a8bd33394c7cb36b7d3bd3e920d0b050c6']
MAIN = '6de0aa70413f96f42b174e9ad6973ad33a2bb913'
OLD_STAMP = 'ef2c2b14098e0eef8bc2d9626c4d748b1d565d7b961ff311f7ea5c1ee612fa5f'
BINDING = 'docs/research/ncaaf-filtered-history-bindings-v1.json'
AUTHORIZATION = 'docs/paid-launch/nfl-first-driver-capacity-correction-v1.md'
POLICY = 'docs/paid-launch/launch-scope-policy-nfl-first-driver-capacity-v1.json'
PROFILE = 'scripts/ci_test_file_costs_v1.json'
PREVIOUS_SCOPE = 'scripts/nfl_first_driver_scope.py'
NORMALIZED = {'scripts/check_launch_change_scope.py', 'scripts/ncaaf_private_refresh_scope.py',
              'scripts/ncaaf_filtered_history_scope.py'}
PATHS = {BINDING, AUTHORIZATION, PROFILE, PREVIOUS_SCOPE,
         'scripts/nfl_first_driver_capacity_scope.py', 'tests/nfl_profile_generation_inputs.py',
         'tests/test_nfl_capacity_profile.py', 'tests/test_nfl_capacity_context_profile.py', *NORMALIZED}
APPROVED = {'docs/paid-launch/nfl-first-driver-capacity-correction-v1.md': 'c9e444d28d44e8f5bcc3b4f5900bbe93791798c42d6e2a0115549516130a2100', 'scripts/check_launch_change_scope.py': '4fce27df09b797820c1d551eaabbed01522b2529096acf951f15a8494ef4d9e2', 'scripts/ci_test_file_costs_v1.json': '599043bc08bce3f668613ef984c94bf485d1a92cc5eb4449bc20d413117d559e', 'scripts/ncaaf_filtered_history_scope.py': 'ccdbe44641108739798fe0a628b89e0a453aa8b4fe8854c230f172d9f867aa74', 'scripts/ncaaf_private_refresh_scope.py': '0decd4f5ec6d7c8f652e1b2a9eb4d564f66b6719aa266221249b9d9a7f2543d0', 'scripts/nfl_first_driver_scope.py': '69612a3159b5aea7965e510748cb01607f28c7c9090e21a58886bb40500048d4', 'tests/nfl_profile_generation_inputs.py': '787441c9794893e2df413a874bdb1d9064b101c0ec442d529df4403e14124c34', 'tests/test_nfl_capacity_context_profile.py': 'aeb151fe62c250ed75daecb9935cd97bf7b78bdaa41bfacd9731947988924f4a', 'tests/test_nfl_capacity_profile.py': '2ad84c44616450aca18e7c9a0d5230de47ccebffe18164b18465e3364df0ad04'}


def require(ok, code):
    if not ok: raise ValueError(code)


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root)


def text(root, *args):
    return git(root, *args).decode().strip()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()


def normalized(path, raw, stamp):
    raw = raw.replace(b'\r\n', b'\n')
    if path in NORMALIZED:
        require(raw.count(stamp.encode()) == 1, 'CAPACITY_BINDING_STAMP_CHANGED')
        raw = raw.replace(stamp.encode(), b'0' * 64)
    return raw


def blobs(root, revision):
    return {p: m.split()[2] for line in text(root, 'ls-tree', '-r', revision).splitlines()
            for m, p in [line.split('\t', 1)]}


def binding_successor(root, stamp):
    raw = git(root, 'show', BASE + ':' + BINDING)
    require(sha(raw) == OLD_STAMP, 'CAPACITY_PREDECESSOR_BINDING_CHANGED')
    binding = json.loads(raw)
    parent_components = dict(binding['components'])
    parent_edges = {p:r['after_sha256'] for p,r in binding['nfl_capacity_predecessors'].items()}
    parent_bytes = len(raw)
    for path in PATHS - {BINDING}:
        digest = sha(normalized(path, (Path(root) / path).read_bytes(), stamp))
        if path in binding['nfl_capacity_predecessors']:
            binding['nfl_capacity_predecessors'][path]['after_sha256'] = digest
        elif path in binding['components']:
            before = git(root, 'show', BASE + ':' + path)
            binding['nfl_capacity_predecessors'][path] = dict(before_bytes=len(before), before_sha256=sha(before),
                before_b85=base64.b85encode(zlib.compress(before, 9)).decode(), after_sha256=digest)
        binding['components'][path] = digest
    parent = binding['qualification_release_repair']['build_preflight']['receipt_successor']['first_driver_successor']
    parent['capacity_context_successor'] = dict(version=1, base=BASE, tree=TREE, main=MAIN,
        prior_binding_sha256=OLD_STAMP, authorization_sha256=binding['components'][AUTHORIZATION],
        profile_sha256=binding['components'][PROFILE], paths=sorted(PATHS),
        historical_parent_binding=dict(bytes=parent_bytes, sha256=OLD_STAMP,
            components=parent_components, predecessor_after_sha256=parent_edges),
        historical_parent_sources={p:dict(bytes=len(source),sha256=sha(source),
            compressed_b85=base64.b85encode(zlib.compress(source,9)).decode())
            for p in (PROFILE, PREVIOUS_SCOPE, 'tests/test_nfl_capacity_profile.py',
                'scripts/run_ci_tests.py', 'tests/qualification_ops/reviewed_profile_inputs.py')
            for source in [git(root,'show',BASE+':'+p)]})
    return binding



def _tls_scope(root, binding):
    import types
    path = Path(root)/'scripts/nfl_tls_trust_scope.py'
    with path.open('rb') as stream: raw = stream.read(8*1024*1024+1)
    require(len(raw) <= 8*1024*1024 and sha(raw.replace(b'\r\n',b'\n')) ==
        binding['components']['scripts/nfl_tls_trust_scope.py'], 'TLS_SCOPE_CHANGED')
    module = types.ModuleType('exact_nfl_tls_trust_scope'); module.__file__ = str(path)
    exec(compile(raw,str(path),'exec'),module.__dict__)
    return module


def _tls_metadata(binding):
    return binding.get('qualification_release_repair',{}).get('build_preflight',{}).get('receipt_successor',{}).get('first_driver_successor',{}).get('capacity_context_successor',{}).get('tls_trust_successor')


def make_policy(root, binding, implementation, binding_sha):
    if _tls_metadata(binding) is not None:
        return _tls_scope(root,binding).make_policy(root,binding,implementation,binding_sha)
    return dict(version='nfl-first-driver-capacity-seal-v1', base=BASE, base_tree=TREE, main=MAIN,
        ordered_predecessor_parents=PARENTS, binding_sha256=binding_sha,
        predecessor_policy_sha256=sha(git(root, 'show', BASE + ':docs/paid-launch/launch-scope-policy-nfl-first-driver-failure-v1.json')),
        authorization_sha256=binding['components'][AUTHORIZATION], implementation_commit=implementation,
        implementation_tree=text(root, 'rev-parse', implementation + '^{tree}'), implementation_paths=sorted(PATHS),
        unchanged_blobs={p:v for p,v in blobs(root, BASE).items() if p not in PATHS})


def validate(root, binding, comparison_base, manifest, *, binding_sha):
    if _tls_metadata(binding) is not None:
        return _tls_scope(root,binding).validate(root,binding,comparison_base,manifest,binding_sha=binding_sha)
    root = Path(root)
    require(text(root, 'rev-parse', BASE + '^{tree}') == TREE and
            text(root, 'show', '-s', '--format=%P', BASE).split() == PARENTS and
            text(root, 'merge-base', MAIN, BASE) == MAIN, 'CAPACITY_BASE_CHANGED')
    inherited = json.loads(git(root, 'show', BASE + ':' + binding['baseline_manifest']))
    require(comparison_base in (None, MAIN, BASE, inherited['base_sha']), 'COMPARISON_BASE_NOT_APPROVED')
    require(manifest is None or Path(manifest).resolve() == (root / binding['baseline_manifest']).resolve(),
            'MANIFEST_PATH_NOT_APPROVED')
    require(not text(root, 'diff', '--name-only') and not text(root, 'diff', '--cached', '--name-only') and
            not text(root, 'ls-files', '--others', '--exclude-standard'), 'UNAUTHORIZED_CHECKOUT_CHANGE')
    require(binding == binding_successor(root, binding_sha), 'CAPACITY_BINDING_CHANGED')
    require(APPROVED and all(sha(normalized(p, (root / p).read_bytes(), binding_sha)) == v
                             for p,v in APPROVED.items()), 'CAPACITY_SOURCE_NOT_APPROVED')
    for path in NORMALIZED:
        require(normalized(path, git(root, 'show', BASE + ':' + path), OLD_STAMP) ==
                normalized(path, (root / path).read_bytes(), binding_sha), 'INHERITED_GUARD_LOGIC_CHANGED')
    old_consumer = git(root, 'show', BASE + ':tests/test_nfl_capacity_profile.py')
    require((root/'tests/test_nfl_capacity_profile.py').read_bytes() == old_consumer +
        b'\n\nfrom tests.nfl_profile_generation_inputs import historical_profile_generation\n',
        'UNAUTHORIZED_EXISTING_TEST_CHANGE')
    policy = json.loads(git(root, 'show', 'HEAD:' + POLICY)); implementation = policy['implementation_commit']
    require(text(root, 'show', '-s', '--format=%P', implementation).split() == [BASE], 'IMPLEMENTATION_PARENT_NOT_APPROVED')
    require(set(text(root, 'diff', '--name-only', BASE, implementation).splitlines()) == PATHS | {POLICY},
            'IMPLEMENTATION_CHANGE_SET_NOT_APPROVED')
    require(policy == make_policy(root, binding, implementation, binding_sha), 'EXACT_POLICY_BINDINGS_CHANGED')
    head = text(root, 'rev-parse', 'HEAD'); parents = text(root, 'show', '-s', '--format=%P', head).split(); seal = head
    if len(parents) == 2:
        require(parents[0] == MAIN, 'TESTED_MERGE_PARENTS_CHANGED'); seal = parents[1]
        require(text(root, 'rev-parse', seal + '^{tree}') == text(root, 'rev-parse', head + '^{tree}'), 'TESTED_TREE_CHANGED')
    require(text(root, 'show', '-s', '--format=%P', seal).split() == [implementation] and
            text(root, 'diff', '--name-status', implementation, seal) == 'M\t' + POLICY, 'SEAL_CHANGE_SET_NOT_APPROVED')
    actual = blobs(root, 'HEAD')
    require(actual.keys() == blobs(root, BASE).keys() | PATHS | {POLICY}, 'UNREVIEWED_PATH')
    require(all(actual.get(p) == v for p,v in policy['unchanged_blobs'].items()), 'IMMUTABLE_FILE_CHANGED')
    for path, value in actual.items():
        raw = (root / path).read_bytes()
        blob = lambda b: hashlib.sha1(b'blob ' + str(len(b)).encode() + b'\0' + b).hexdigest()
        require(value in (blob(raw), blob(raw.replace(b'\r\n', b'\n'))), 'UNAUTHORIZED_CHECKOUT_CHANGE')
    return dict(schema_version=103, status='PASS', base=MAIN, preserved_head=BASE,
        implementation=implementation, seal=seal, candidate=head, tree=text(root, 'rev-parse', head + '^{tree}'),
        changed_files=sorted(text(root, 'diff', '--name-only', MAIN, 'HEAD').splitlines()),
        protected_changes=[], existing_test_changes=[], approved_fixture_test_adaptations={
            'tests/test_nfl_capacity_profile.py': dict(
                before_sha256=sha(git(root,'show',BASE+':tests/test_nfl_capacity_profile.py')),
                after_sha256=sha((root/'tests/test_nfl_capacity_profile.py').read_bytes()),
                before_blob=blobs(root,BASE)['tests/test_nfl_capacity_profile.py'],
                after_blob=actual['tests/test_nfl_capacity_profile.py'], authorization=AUTHORIZATION,
                authorization_sha256=binding['components'][AUTHORIZATION],
                exact_append='module-wide authenticated predecessor fixture import; original bodies/assertions unchanged')},
        timing_profile_and_authenticated_fixture_only_since_preserved_head=True, diagnostic_implementation_unchanged=True,
        production_store_model_transport_unchanged=True, accepted=False, inference=False)
