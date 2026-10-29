"""Exact owner-only TLS inspection successor; historical expectations remain frozen."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import zlib

BASE = 'c548ddb2693b64c94942ff315eed93e3fcb2f56e'
TREE = 'e9c9ab85b66576cb9a04daa83ed956af69955b26'
PARENTS = ['6de0aa70413f96f42b174e9ad6973ad33a2bb913', '9b2f481be7333fc2d84564109ae19f498c5ba253']
OLD_STAMP = 'ac2abfe85d86f20c9cbbb10be72fdba9a4a61c6c54ab588e4a12bed45fb0167b'
BINDING = 'docs/research/ncaaf-filtered-history-bindings-v1.json'
AUTHORIZATION = 'docs/paid-launch/nfl-tls-trust-preflight-v1.md'
POLICY = 'docs/paid-launch/launch-scope-policy-nfl-tls-trust-v1.json'
NORMALIZED = {'scripts/check_launch_change_scope.py', 'scripts/ncaaf_private_refresh_scope.py',
              'scripts/ncaaf_filtered_history_scope.py'}
PATHS = {BINDING, AUTHORIZATION, 'app_core/nfl_build_preflight.py', 'app_core/nfl_tls_trust.py',
         'scripts/nfl_first_driver_capacity_scope.py', 'scripts/nfl_tls_trust_scope.py',
         'tests/nfl_profile_generation_inputs.py', 'tests/test_nfl_tls_trust.py',
         'tests/test_nfl_tls_trust_scope.py', *NORMALIZED}
APPROVED = {'app_core/nfl_build_preflight.py': '6c0c97f20cbb4dddcedb4689088ca87e80d39952fcb31c6085560cfcefc23c0f', 'app_core/nfl_tls_trust.py': '9161ab15d0a0f990e5a34954d55c9a4bdfc5aa3ac5eb594ff14994ebfb7f88f2', 'docs/paid-launch/nfl-tls-trust-preflight-v1.md': '2014302f8482b54d545a67fb7b65f18fa2653241e7e61dc8bf7cffbc6e6a9ed7', 'scripts/check_launch_change_scope.py': '4fce27df09b797820c1d551eaabbed01522b2529096acf951f15a8494ef4d9e2', 'scripts/ncaaf_filtered_history_scope.py': 'ccdbe44641108739798fe0a628b89e0a453aa8b4fe8854c230f172d9f867aa74', 'scripts/ncaaf_private_refresh_scope.py': '0decd4f5ec6d7c8f652e1b2a9eb4d564f66b6719aa266221249b9d9a7f2543d0', 'scripts/nfl_first_driver_capacity_scope.py': '341b9a45ca23dfc797c52db2efd0fd7c10e21c635031f921aefd5ef38b1cc46a', 'tests/nfl_profile_generation_inputs.py': 'aac94ff3b7b012e39737d8e70872b2f86a52b6225acf8ff951658b126e7df841', 'tests/test_nfl_tls_trust.py': '684efbc348463deeff7e0e6dadacaead45e342533aa9bf9d864f8c0fbe1a58b9', 'tests/test_nfl_tls_trust_scope.py': '8d85cf0afdc3a7c5168eb9c919a4c02d3610b87135ac433d6b54e28e64647fec'}


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
        require(raw.count(stamp.encode()) == 1, 'RECEIPT_BINDING_STAMP_CHANGED')
        raw = raw.replace(stamp.encode(), b'0' * 64)
    return raw


def blobs(root, revision):
    return {p: m.split()[2] for line in text(root, 'ls-tree', '-r', revision).splitlines()
            for m, p in [line.split('\t', 1)]}



def metadata(binding):
    return binding['qualification_release_repair']['build_preflight']['receipt_successor']['first_driver_successor']['capacity_context_successor']


def binding_successor(root, stamp):
    raw = git(root, 'show', BASE + ':' + BINDING)
    require(sha(raw) == OLD_STAMP, 'TLS_PREDECESSOR_BINDING_CHANGED')
    binding = json.loads(raw)
    historical = {}
    for path in PATHS - {BINDING}:
        digest = sha(normalized(path, (Path(root) / path).read_bytes(), stamp))
        if path in binding['components'] and path not in NORMALIZED:
            before = git(root, 'show', BASE + ':' + path)
            historical[path] = dict(bytes=len(before), sha256=sha(before),
                compressed_b85=base64.b85encode(zlib.compress(before, 9)).decode())
        if path in binding['nfl_capacity_predecessors']:
            binding['nfl_capacity_predecessors'][path]['after_sha256'] = digest
        elif path in binding['components']:
            before = git(root, 'show', BASE + ':' + path)
            binding['nfl_capacity_predecessors'][path] = dict(before_bytes=len(before),
                before_sha256=sha(before), before_b85=base64.b85encode(zlib.compress(before, 9)).decode(),
                after_sha256=digest)
        binding['components'][path] = digest
    metadata(binding)['tls_trust_successor'] = dict(version=1, base=BASE, tree=TREE,
        prior_binding_sha256=OLD_STAMP, authorization_sha256=binding['components'][AUTHORIZATION],
        paths=sorted(PATHS), historical_sources=historical)
    return binding


def make_policy(root, binding, implementation, binding_sha):
    return dict(version='nfl-tls-trust-preflight-seal-v1', base=BASE, base_tree=TREE,
        ordered_parents=PARENTS, binding_sha256=binding_sha,
        authorization_sha256=binding['components'][AUTHORIZATION],
        predecessor_policy_sha256=sha(git(root, 'show', BASE + ':docs/paid-launch/launch-scope-policy-nfl-first-driver-capacity-v1.json')),
        implementation_commit=implementation, implementation_tree=text(root,'rev-parse',implementation+'^{tree}'),
        implementation_paths=sorted(PATHS),
        unchanged_blobs={p:v for p,v in blobs(root,BASE).items() if p not in PATHS})


def validate(root, binding, comparison_base, manifest, *, binding_sha):
    root = Path(root)
    require(text(root,'rev-parse',BASE+'^{tree}') == TREE and
        text(root,'show','-s','--format=%P',BASE).split() == PARENTS, 'TLS_BASE_CHANGED')
    inherited = json.loads(git(root,'show',BASE+':'+binding['baseline_manifest']))
    require(comparison_base in (None,BASE,inherited['base_sha']), 'COMPARISON_BASE_NOT_APPROVED')
    require(manifest is None or Path(manifest).resolve() == (root/binding['baseline_manifest']).resolve(),
        'MANIFEST_PATH_NOT_APPROVED')
    require(not text(root,'diff','--name-only') and not text(root,'diff','--cached','--name-only') and
        not text(root,'ls-files','--others','--exclude-standard'), 'UNAUTHORIZED_CHECKOUT_CHANGE')
    require(binding == binding_successor(root,binding_sha), 'TLS_BINDING_CHANGED')
    require(APPROVED and all(sha(normalized(p,(root/p).read_bytes(),binding_sha)) == digest
        for p,digest in APPROVED.items()), 'TLS_SOURCE_NOT_APPROVED')
    for path in NORMALIZED:
        require(normalized(path,git(root,'show',BASE+':'+path),OLD_STAMP) ==
            normalized(path,(root/path).read_bytes(),binding_sha), 'INHERITED_GUARD_LOGIC_CHANGED')
    policy = json.loads(git(root,'show','HEAD:'+POLICY)); implementation = policy['implementation_commit']
    require(text(root,'show','-s','--format=%P',implementation).split() == [BASE], 'IMPLEMENTATION_PARENT_NOT_APPROVED')
    require(set(text(root,'diff','--name-only',BASE,implementation).splitlines()) == PATHS|{POLICY},
        'IMPLEMENTATION_CHANGE_SET_NOT_APPROVED')
    require(policy == make_policy(root,binding,implementation,binding_sha), 'EXACT_POLICY_BINDINGS_CHANGED')
    head = text(root,'rev-parse','HEAD'); parents = text(root,'show','-s','--format=%P',head).split(); seal = head
    if len(parents) == 2:
        require(parents[0] == BASE, 'TESTED_MERGE_PARENTS_CHANGED'); seal = parents[1]
        require(text(root,'rev-parse',seal+'^{tree}') == text(root,'rev-parse',head+'^{tree}'), 'TESTED_TREE_CHANGED')
    require(text(root,'show','-s','--format=%P',seal).split() == [implementation] and
        text(root,'diff','--name-status',implementation,seal) == 'M\t'+POLICY, 'SEAL_CHANGE_SET_NOT_APPROVED')
    actual = blobs(root,'HEAD')
    require(actual.keys() == blobs(root,BASE).keys()|PATHS|{POLICY}, 'UNREVIEWED_PATH')
    require(all(actual.get(p) == v for p,v in policy['unchanged_blobs'].items()), 'IMMUTABLE_FILE_CHANGED')
    for path, value in actual.items():
        raw = (root/path).read_bytes()
        blob = lambda b: hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
        require(value in (blob(raw),blob(raw.replace(b'\r\n',b'\n'))), 'UNAUTHORIZED_CHECKOUT_CHANGE')
    return dict(schema_version=104,status='PASS',base=BASE,implementation=implementation,seal=seal,candidate=head,
        tree=text(root,'rev-parse',head+'^{tree}'),changed_files=sorted(PATHS|{POLICY}),
        protected_changes=[],existing_test_changes=[],approved_fixture_test_adaptations={},
        approved_test_helper_changes=['tests/nfl_profile_generation_inputs.py'],
        production_store_model_transport_and_connection_diagnostic_unchanged=True,
        owner_only_connection_free_tls_inspection=True,accepted=False,inference=False)
