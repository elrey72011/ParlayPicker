"""Exact qualification-harness successor; production and prior seals unchanged."""
import ast
import base64
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import zlib

BASE = '2472bc05ee9f6c3ce30c387ee4695583a2a04281'
TREE = '1c50555e222e8161a24e1550c25ba2d947b34931'
PARENTS = ['f6a7506dc3300fbcf390dbcdd00f0688dba073a4', '53726da34164b7f2ede89a545bef669bcfc999a2']
OLD_STAMP = 'ebf3be5fadba750afd2e99465d7d6d0caef579d559bafc522f1d619e581facf7'
BINDING = 'docs/research/ncaaf-filtered-history-bindings-v1.json'
POLICY = 'docs/paid-launch/launch-scope-policy-qualification-repair-v1.json'
AUTHORIZATION = 'docs/paid-launch/qualification-failure-evidence-repair-v1.md'
NORMALIZED = {'scripts/check_launch_change_scope.py', 'scripts/ncaaf_private_refresh_scope.py', 'scripts/ncaaf_filtered_history_scope.py'}
PATHS = {BINDING, AUTHORIZATION, *NORMALIZED, 'scripts/qualification_repair_scope.py',
         '.github/workflows/ci.yml', '.github/workflows/qualification-operations.yml',
         'tests/requirements-ci.txt', 'tests/test_nfl_capacity_profile.py',
         'tests/qualification_ops/functional_suite.py', 'tests/qualification_ops/run_offline.py',
         'tests/qualification_ops/logical_watchdog.py', 'tests/qualification_ops/child_runner.py',
         'tests/qualification_ops/failure_reports.py', 'tests/qualification_ops/reporting_regressions.py',
         'tests/qualification_ops/reviewed_profile_inputs.py'}
# Independently pinned final source expectations are populated by the reviewed
# local preparation, before binding/seal generation. They are not read from a
# candidate binding, and rehashing a modified source cannot satisfy them.
APPROVED = {'.github/workflows/ci.yml': 'fbdeed4c90de527232293ced8caa2e56a116b931bb99475ea8e78ca4a4eb69a0', '.github/workflows/qualification-operations.yml': '9eeeb16d4a7bf256589e53f72a748f86db5eea0ffdc34203eb7dcbbc6a78067d', 'docs/paid-launch/qualification-failure-evidence-repair-v1.md': '8cabe47598f1e19d8f28fda632430a85443291374921a67fef124a6aab922285', 'scripts/check_launch_change_scope.py': '4fce27df09b797820c1d551eaabbed01522b2529096acf951f15a8494ef4d9e2', 'scripts/ncaaf_filtered_history_scope.py': 'ccdbe44641108739798fe0a628b89e0a453aa8b4fe8854c230f172d9f867aa74', 'scripts/ncaaf_private_refresh_scope.py': '0decd4f5ec6d7c8f652e1b2a9eb4d564f66b6719aa266221249b9d9a7f2543d0', 'tests/qualification_ops/child_runner.py': 'd44c25db8d739c3f332bf10250339c41bde00fb394d5db911a3ed1d0f171e1dd', 'tests/qualification_ops/failure_reports.py': '1573b1e0dfd8df748391b0f0ad1a765d5c04ed3e62528da298b83c7fd5f55837', 'tests/qualification_ops/functional_suite.py': 'f2777b80f89fb01c44523419ce005272512027160ab6261f5249ef23b9a1e578', 'tests/qualification_ops/logical_watchdog.py': 'dd40b31f0c850bb77c3df6502abf2431da4e6d90760acf2196d5e46b537cc11b', 'tests/qualification_ops/reporting_regressions.py': '427eb73efcbc93dbf2305f60387ea96ad66469b8f92df1e7f4684bfc94239482', 'tests/qualification_ops/reviewed_profile_inputs.py': 'e36db16d8dbb636d4fd7ba87a14f3481e50f616c2c9a79f13a2c090fa81e0926', 'tests/qualification_ops/run_offline.py': 'afcea19e06c6020e2c94605cc9c0c727c5bb4d903bda5de2dccbd4d361d9071b', 'tests/requirements-ci.txt': 'd3bf3fd2125c27db8762bc620c29bfe6830310ff31d834ee12b668493b9a3971', 'tests/test_nfl_capacity_profile.py': '3453ed0af3855ff90acc641c608a5aaa67c9e9951e64cf0bcfd42a33ca0b8536'}  # QUALIFICATION_APPROVED_SOURCES


def require(ok, code):
    if not ok:
        raise ValueError(code)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root)


def text(root, *args):
    return git(root, *args).decode().strip()


def blobs(root, revision):
    return {path: meta.split()[2] for line in text(root, 'ls-tree', '-r', revision).splitlines()
            for meta, path in [line.split('\t', 1)]}


def normalized(path, raw, stamp):
    raw = raw.replace(b'\r\n', b'\n')
    if path in NORMALIZED:
        require(raw.count(stamp.encode()) == 1, 'QUALIFICATION_BINDING_CHANGED')
        raw = raw.replace(stamp.encode(), b'0' * 64)
    return raw


def binding_successor(root, stamp):
    raw = git(root, 'show', BASE + ':' + BINDING)
    require(sha(raw) == OLD_STAMP, 'QUALIFICATION_PREDECESSOR_CHANGED')
    binding = json.loads(raw)
    for path in PATHS - {BINDING}:
        digest = sha(normalized(path, (root / path).read_bytes(), stamp))
        binding['components'][path] = digest
        if path in binding['nfl_capacity_predecessors']:
            binding['nfl_capacity_predecessors'][path]['after_sha256'] = digest
    # The earlier capacity assertion consumes the exact independently retained
    # CI workflow, not this successor's test-runner pin. No expectation changes.
    path = '.github/workflows/ci.yml'
    before = git(root, 'show', BASE + ':' + path)
    binding['nfl_capacity_predecessors'][path] = dict(before_bytes=len(before), before_sha256=sha(before),
        before_b85=base64.b85encode(zlib.compress(before, 9)).decode(), after_sha256=binding['components'][path])
    binding['qualification_release_repair'] = dict(version=1, base=BASE, base_tree=TREE,
        ordered_parents=PARENTS, predecessor_binding_sha256=OLD_STAMP,
        authorization_sha256=binding['components'][AUTHORIZATION], paths=sorted(PATHS))
    return binding


def make_policy(root, binding, implementation, binding_sha):
    return dict(version='qualification-failure-evidence-seal-v1', base=BASE, base_tree=TREE,
        ordered_parents=PARENTS, authorization_sha256=binding['components'][AUTHORIZATION],
        binding_sha256=binding_sha, implementation_commit=implementation,
        implementation_tree=text(root, 'rev-parse', implementation + '^{tree}'), implementation_paths=sorted(PATHS),
        unchanged_blobs={p: value for p, value in blobs(root, BASE).items() if p not in PATHS})


def validate(root, binding, comparison_base, manifest, *, binding_sha):
    root = Path(root)
    require(text(root, 'rev-parse', BASE + '^{tree}') == TREE
        and text(root, 'show', '-s', '--format=%P', BASE).split() == PARENTS, 'QUALIFICATION_BASE_CHANGED')
    inherited = json.loads(git(root, 'show', BASE + ':' + binding['baseline_manifest']))
    require(comparison_base in (None, BASE, inherited['base_sha']), 'COMPARISON_BASE_NOT_APPROVED')
    require(manifest is None or Path(manifest).resolve() == (root / binding['baseline_manifest']).resolve(), 'MANIFEST_PATH_NOT_APPROVED')
    require(not text(root, 'diff', '--name-only') and not text(root, 'diff', '--cached', '--name-only')
        and not text(root, 'ls-files', '--others', '--exclude-standard'), 'UNAUTHORIZED_CHECKOUT_CHANGE')
    require(binding == binding_successor(root, binding_sha), 'QUALIFICATION_PREDECESSOR_CHANGED')
    for path, expected in APPROVED.items():
        require(sha(normalized(path, (root / path).read_bytes(), binding_sha)) == expected, 'QUALIFICATION_PATCH_NOT_APPROVED')
    policy = json.loads(git(root, 'show', 'HEAD:' + POLICY)); implementation = policy['implementation_commit']
    require(text(root, 'show', '-s', '--format=%P', implementation).split() == [BASE], 'IMPLEMENTATION_PARENT_NOT_APPROVED')
    require(set(text(root, 'diff', '--name-only', BASE, implementation).splitlines()) == PATHS, 'IMPLEMENTATION_CHANGE_SET_NOT_APPROVED')
    require(policy == make_policy(root, binding, implementation, binding_sha), 'EXACT_POLICY_BINDINGS_CHANGED')
    head = text(root, 'rev-parse', 'HEAD'); parents = text(root, 'show', '-s', '--format=%P', head).split(); seal = head
    if len(parents) == 2:
        require(parents[0] == BASE, 'TESTED_MERGE_PARENTS_CHANGED'); seal = parents[1]
        require(text(root, 'rev-parse', seal + '^{tree}') == text(root, 'rev-parse', head + '^{tree}'), 'TESTED_TREE_CHANGED')
    require(text(root, 'show', '-s', '--format=%P', seal).split() == [implementation]
        and text(root, 'diff', '--name-status', implementation, seal) == 'A\t' + POLICY, 'SEAL_CHANGE_SET_NOT_APPROVED')
    before = blobs(root, BASE); actual = blobs(root, 'HEAD')
    require(set(actual) == set(before) | PATHS | {POLICY}, 'UNREVIEWED_PATH')
    require(all(actual.get(p) == value for p, value in policy['unchanged_blobs'].items()), 'IMMUTABLE_FILE_CHANGED')
    for path, value in actual.items():
        raw = (root / path).read_bytes()
        blob = lambda b: hashlib.sha1(b'blob ' + str(len(b)).encode() + b'\0' + b).hexdigest()
        require(value in (blob(raw), blob(raw.replace(b'\r\n', b'\n'))), 'UNAUTHORIZED_CHECKOUT_CHANGE')
    adaptations = {}
    preserved = 0
    for path in ('tests/qualification_ops/functional_suite.py', 'tests/test_nfl_capacity_profile.py'):
        old = git(root, 'show', BASE + ':' + path); current = (root / path).read_bytes()
        # Every original test remains; every original assertion is retained.
        original_tree, current_tree = ast.parse(old), ast.parse(current)
        methods = lambda tree: {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name.startswith('test_')}
        require(methods(original_tree) == methods(current_tree), 'QUALIFICATION_TEST_MEMBERSHIP_CHANGED')
        preserved += len(methods(original_tree))
        adaptations[path] = dict(before_sha256=sha(old), after_sha256=sha(current), before_blob=before[path],
            after_blob=actual[path], authorization=AUTHORIZATION, authorization_sha256=binding['components'][AUTHORIZATION])
    return dict(schema_version=99, status='PASS', base=BASE, implementation=implementation, seal=seal, candidate=head,
        tree=text(root, 'rev-parse', head + '^{tree}'), changed_files=sorted(PATHS | {POLICY}), protected_changes=[],
        existing_test_changes=[], approved_fixture_test_adaptations=adaptations,
        approved_test_helper_changes=sorted(p for p in PATHS if p.startswith('tests/')),
        preserved_original_test_bodies=preserved, accepted=False, inference=False,
        production_model_store_and_driver_unchanged=True)
