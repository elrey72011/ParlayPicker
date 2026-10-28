"""Exact receipt-only successor; keep authenticated historical generations intact."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import zlib

BASE = 'ad5662c99b6529fddc1f40dbaac6853563e01c8e'
TREE = 'de1ab55c0e4dce2a539ebc859df8f2acfd069ce3'
PARENTS = ['3fb9302463f44589ea0d2c75b2ed100bf2f596d0', 'e016e7dadea7b45d8baa7b0617ffef78a657be89']
OLD_STAMP = '71fa1b1b343bf869d56f84e6604e31abaeedc697e544b9e497029e5c158dcd3d'
BINDING = 'docs/research/ncaaf-filtered-history-bindings-v1.json'
AUTHORIZATION = 'docs/paid-launch/nfl-diagnostic-receipts-v1.md'
POLICY = 'docs/paid-launch/launch-scope-policy-nfl-diagnostic-receipts-v1.json'
NORMALIZED = {'scripts/check_launch_change_scope.py', 'scripts/ncaaf_private_refresh_scope.py',
              'scripts/ncaaf_filtered_history_scope.py'}
PATHS = {BINDING, AUTHORIZATION, 'app/ui/nfl_storage_diagnostic.py',
         'app_core/nfl_diagnostic_receipts.py', 'app_core/nfl_build_preflight.py',
         'scripts/nfl_build_preflight_scope.py', 'scripts/nfl_diagnostic_receipt_scope.py',
         'tests/test_nfl_diagnostic_receipts.py', 'scripts/ci_test_file_costs_v1.json', *NORMALIZED}
APPROVED = {'app/ui/nfl_storage_diagnostic.py': '3f9aaf584072a83123dc73c0b86f483705d0465db632cf0d9b8142378fcbe952', 'app_core/nfl_build_preflight.py': '2e8120a76062e16efb7270e304f54f6d7d08aff3e50ba2d4ccc821e1047cfd92', 'app_core/nfl_diagnostic_receipts.py': '397c11f37098930ae520846d850952bd566c451db670f5b2ed41a2d11dd13dec', 'docs/paid-launch/nfl-diagnostic-receipts-v1.md': 'da9a7548a6001c1b0b51c6c5289e709b8ba1bf5ccd7a1168963be4243dfe8378', 'scripts/check_launch_change_scope.py': '4fce27df09b797820c1d551eaabbed01522b2529096acf951f15a8494ef4d9e2', 'scripts/ci_test_file_costs_v1.json': '5a46c9ea83d88f29f01bcca6eea068c3a1999c19dcf0f1164de65ccf9fb8f94c', 'scripts/ncaaf_filtered_history_scope.py': 'ccdbe44641108739798fe0a628b89e0a453aa8b4fe8854c230f172d9f867aa74', 'scripts/ncaaf_private_refresh_scope.py': '0decd4f5ec6d7c8f652e1b2a9eb4d564f66b6719aa266221249b9d9a7f2543d0', 'scripts/nfl_build_preflight_scope.py': '89be0a5e3ba2238a19f0508845e93040aa94a8d38bbb89a06404a417351d6855', 'tests/test_nfl_diagnostic_receipts.py': '98a391e1717e1aef85a2f0a9ae137108ac4218e7e503aedf5092591272458d45'}  # Independent final reviewed identities.


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


def binding_successor(root, stamp):
    raw = git(root, 'show', BASE + ':' + BINDING)
    require(sha(raw) == OLD_STAMP, 'RECEIPT_PREDECESSOR_BINDING_CHANGED')
    binding = json.loads(raw)
    for path in PATHS - {BINDING}:
        digest = sha(normalized(path, (Path(root) / path).read_bytes(), stamp))
        if path in binding['nfl_capacity_predecessors']:
            # Preserve the exact frozen pre-capacity predecessor; only add this
            # authorized generation to the current side of that existing edge.
            binding['nfl_capacity_predecessors'][path]['after_sha256'] = digest
        elif path in binding['components']:
            before = git(root, 'show', BASE + ':' + path)
            binding['nfl_capacity_predecessors'][path] = dict(before_bytes=len(before),
                before_sha256=sha(before), before_b85=base64.b85encode(zlib.compress(before, 9)).decode(),
                after_sha256=digest)
        binding['components'][path] = digest
    binding['qualification_release_repair']['build_preflight']['receipt_successor'] = dict(
        version=1, base=BASE, tree=TREE, prior_binding_sha256=OLD_STAMP,
        authorization_sha256=binding['components'][AUTHORIZATION], paths=sorted(PATHS))
    return binding


def make_policy(root, binding, implementation, binding_sha):
    return dict(version='nfl-diagnostic-receipt-seal-v1', base=BASE, base_tree=TREE,
        ordered_parents=PARENTS, binding_sha256=binding_sha,
        authorization_sha256=binding['components'][AUTHORIZATION],
        implementation_commit=implementation, implementation_tree=text(root, 'rev-parse', implementation + '^{tree}'),
        implementation_paths=sorted(PATHS),
        unchanged_blobs={p: v for p, v in blobs(root, BASE).items() if p not in PATHS})


def validate(root, binding, comparison_base, manifest, *, binding_sha):
    root = Path(root)
    require(text(root, 'rev-parse', BASE + '^{tree}') == TREE and
            text(root, 'show', '-s', '--format=%P', BASE).split() == PARENTS, 'RECEIPT_BASE_CHANGED')
    inherited = json.loads(git(root, 'show', BASE + ':' + binding['baseline_manifest']))
    require(comparison_base in (None, BASE, inherited['base_sha']), 'COMPARISON_BASE_NOT_APPROVED')
    require(manifest is None or Path(manifest).resolve() == (root / binding['baseline_manifest']).resolve(),
            'MANIFEST_PATH_NOT_APPROVED')
    require(not text(root, 'diff', '--name-only') and not text(root, 'diff', '--cached', '--name-only') and
            not text(root, 'ls-files', '--others', '--exclude-standard'), 'UNAUTHORIZED_CHECKOUT_CHANGE')
    require(binding == binding_successor(root, binding_sha), 'RECEIPT_BINDING_CHANGED')
    require(APPROVED and all(sha(normalized(p, (root / p).read_bytes(), binding_sha)) == v
                             for p, v in APPROVED.items()), 'RECEIPT_SOURCE_NOT_APPROVED')
    for path in NORMALIZED:
        require(normalized(path, git(root, 'show', BASE + ':' + path), OLD_STAMP) ==
                normalized(path, (root / path).read_bytes(), binding_sha), 'INHERITED_GUARD_LOGIC_CHANGED')
    policy = json.loads(git(root, 'show', 'HEAD:' + POLICY)); implementation = policy['implementation_commit']
    require(text(root, 'show', '-s', '--format=%P', implementation).split() == [BASE], 'IMPLEMENTATION_PARENT_NOT_APPROVED')
    require(set(text(root, 'diff', '--name-only', BASE, implementation).splitlines()) == PATHS | {POLICY},
            'IMPLEMENTATION_CHANGE_SET_NOT_APPROVED')
    require(policy == make_policy(root, binding, implementation, binding_sha), 'EXACT_POLICY_BINDINGS_CHANGED')
    head = text(root, 'rev-parse', 'HEAD'); parents = text(root, 'show', '-s', '--format=%P', head).split(); seal = head
    if len(parents) == 2:
        require(parents[0] == BASE, 'TESTED_MERGE_PARENTS_CHANGED'); seal = parents[1]
        require(text(root, 'rev-parse', seal + '^{tree}') == text(root, 'rev-parse', head + '^{tree}'), 'TESTED_TREE_CHANGED')
    require(text(root, 'show', '-s', '--format=%P', seal).split() == [implementation] and
            text(root, 'diff', '--name-status', implementation, seal) == 'M\t' + POLICY, 'SEAL_CHANGE_SET_NOT_APPROVED')
    actual = blobs(root, 'HEAD')
    require(actual.keys() == blobs(root, BASE).keys() | PATHS | {POLICY}, 'UNREVIEWED_PATH')
    require(all(actual.get(p) == v for p, v in policy['unchanged_blobs'].items()), 'IMMUTABLE_FILE_CHANGED')
    for path, value in actual.items():
        raw = (root / path).read_bytes()
        blob = lambda b: hashlib.sha1(b'blob ' + str(len(b)).encode() + b'\0' + b).hexdigest()
        require(value in (blob(raw), blob(raw.replace(b'\r\n', b'\n'))), 'UNAUTHORIZED_CHECKOUT_CHANGE')
    return dict(schema_version=101, status='PASS', base=BASE, implementation=implementation, seal=seal,
        candidate=head, tree=text(root, 'rev-parse', head + '^{tree}'), changed_files=sorted(PATHS | {POLICY}),
        protected_changes=[], existing_test_changes=[], approved_fixture_test_adaptations={},
        production_store_model_transport_and_diagnostic_unchanged=True, accepted=False, inference=False)
