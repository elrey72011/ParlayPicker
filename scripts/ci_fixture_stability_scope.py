"""Exact test-fixture successor. Authenticate current state; never cache validation."""
import argparse
import base64
import copy
import hashlib
import importlib.util
import json
import re
from pathlib import Path
import subprocess
import sys
import zlib

PROPOSAL = '261bb913276725605a4b142458b0be536002cac9af327ec17187da2dde1b359d'
CONSOLIDATED_AMENDMENT = 'b7175b51176334e8b31967bac836e38ebea307a0f90d5e1f7af24b850b36fd7e'
COMPLETE_LEDGER_SHA256 = '5eec77cd6d43fbd41b3839d7b99c135bfd0cb2603e92b794874385f8432261b7'
BASE = '8b79b0a0eead4f63a041a551283a4b6be2980d72'
BASE_TREE = '028d3daf56d71596663a10714495a7b48d53c7ab'
GUARD = 'scripts/check_launch_change_scope.py'
BINDING = 'docs/paid-launch/ci-fixture-stability-bindings-v1.json'
POLICY = 'docs/paid-launch/launch-scope-policy-ci-fixture-stability-v1.json'
MAX_SOURCE = 8 * 1024 * 1024
GUARD_TEMPLATE_SHA256 = '43b5a27efbe0239e5cc8b26f4b09fcfbe221e94ec8328184c610490a06b4c71f'


def sha(value):
    return hashlib.sha256(value).hexdigest()


def blob(value):
    return hashlib.sha1(b'blob ' + str(len(value)).encode() + b'\0' + value).hexdigest()


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def bytes_at(root, relative):
    with (root / relative).open('rb') as stream:
        return stream.read()


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def previous_guard(source, binding):
    d = binding['predecessor_guard']
    if sha(source) == d['sha256']:
        return source
    require(type(source) is bytes and len(source) <= MAX_SOURCE, 'SUCCESSOR_GUARD_LIMIT_CHANGED')
    identity = re.search(rb"_CI_STABILITY_GUARD_SHA256='([0-9a-f]{64})'", source)
    stamp = re.search(rb"_CI_STABILITY_BINDING_SHA256='([0-9a-f]{64})'", source)
    require(identity is not None and stamp is not None, 'SUCCESSOR_GUARD_REVIEWED_BYTES_CHANGED')
    identity, stamp = identity.group(1), stamp.group(1)
    require(source.count(identity) == 1 and sha(source.replace(identity, b'0'*64)) == identity.decode(), 'SUCCESSOR_GUARD_REVIEWED_BYTES_CHANGED')
    require(stamp.decode() == sha(canonical(binding)), 'REVIEWED_BINDINGS_CHANGED')
    template = source.replace(identity, b'0'*64).replace(stamp, b'0'*64)
    require(sha(template) == GUARD_TEMPLATE_SHA256, 'SUCCESSOR_GUARD_REVIEWED_BYTES_CHANGED')
    z = zlib.decompressobj()
    value = z.decompress(base64.b85decode(d['source_b85']), MAX_SOURCE + 1)
    require(z.eof and not z.unused_data and len(value) == d['bytes'] <= MAX_SOURCE
            and sha(value) == d['sha256'], 'PREVIOUS_GUARD_LOGIC_CHANGED')
    return value


def trees(g, revision):
    return {path: meta.split()[2] for row in g.git('ls-tree', '-r', revision).splitlines()
            for meta, path in [row.split('\t', 1)]}


def make_policy(g, binding, implementation):
    before, after = trees(g, binding['base']), trees(g, implementation)
    return dict(schema_version=85, policy_version='ci-fixture-stability-v1',
        proposal_sha256=PROPOSAL, advisory_amendment_sha256='c7597d04525eca38a52c7207f75f45829560bd6fa1ab0ce1f5ba2a1bfa7dcf45',
        second_advisory_amendment_sha256='6cf28cd1db29cd0ba101cce67b8ac3bcc5d6857b61ba555b6aa382298fda7295',
        consolidated_advisory_amendment_sha256=CONSOLIDATED_AMENDMENT,complete_profile_ledger_sha256=COMPLETE_LEDGER_SHA256, base_sha=binding['base'], base_tree=binding['base_tree'],
        implementation_commit=implementation, implementation_tree=g.git('rev-parse', implementation+'^{tree}'),
        implementation_changes={p: dict(before_blob=before.get(p), after_blob=after.get(p)) for p in binding['paths']},
        unchanged_bindings={p: v for p, v in before.items() if p not in binding['paths']})


def validate(g, manifest_path, comparison_base, binding):
    require(binding['proposal_sha256'] == PROPOSAL, 'PROPOSAL_NOT_APPROVED')
    require(binding.get('consolidated_advisory_amendment_sha256') == CONSOLIDATED_AMENDMENT
            and binding.get('complete_profile_ledger_sha256') == COMPLETE_LEDGER_SHA256, 'COMPLETE_PROFILE_PROPOSAL_NOT_APPROVED')
    require(binding['base'] == BASE and binding['base_tree'] == BASE_TREE, 'STARTING_MAIN_NOT_APPROVED')
    require(manifest_path.resolve() == (g.ROOT/g.MANIFEST_PATH).resolve(), 'MANIFEST_PATH_NOT_APPROVED')
    raw = g.git_bytes('show', BASE+':'+g.MANIFEST_PATH)
    require(sha(raw) == binding['manifest_sha256'], 'BASELINE_CHANGED')
    # The recorded comparison base is usable only after the original manifest is authenticated.
    require(comparison_base in (None, BASE, json.loads(raw)['base_sha']), 'COMPARISON_BASE_NOT_APPROVED')
    require(g.git('rev-parse', BASE+'^{tree}') == BASE_TREE, 'STARTING_MAIN_TREE_CHANGED')
    parents = [line.split()[1] for line in g.git('cat-file', '-p', BASE).splitlines() if line.startswith('parent ')]
    require(parents == binding['ordered_base_parents'], 'PREDECESSOR_PARENTS_CHANGED')
    require(not g.git('diff', '--name-only') and not g.git('diff', '--cached', '--name-only'), 'UNAUTHORIZED_CHECKOUT_CHANGE')
    policy_raw = g.git_bytes('show', 'HEAD:'+POLICY)
    policy = json.loads(policy_raw)
    implementation = policy['implementation_commit']
    require(g.git('show', '-s', '--format=%P', implementation).split() == [BASE], 'IMPLEMENTATION_PARENT_NOT_APPROVED')
    require(not g.exists_at(BASE, POLICY) and not g.exists_at(implementation, POLICY), 'POLICY_SEAL_MUST_FOLLOW_IMPLEMENTATION')
    require(set(g.git('diff', '--name-only', BASE, implementation).splitlines()) == set(binding['paths']), 'IMPLEMENTATION_CHANGE_SET_NOT_APPROVED')
    require(len(binding['paths']) == len(set(binding['paths'])) == 33, 'IMPLEMENTATION_PATH_SET_CHANGED')
    require(policy == make_policy(g, binding, implementation), 'EXACT_POLICY_BINDINGS_CHANGED')
    impl = trees(g, implementation)
    require(set(binding['reviewed_blobs']) == set(binding['paths'])-{GUARD, BINDING}, 'REVIEWED_BINDING_SET_CHANGED')
    require(all(impl.get(p) == value for p, value in binding['reviewed_blobs'].items()), 'REVIEWED_IMPLEMENTATION_BYTES_CHANGED')
    require(g.git_bytes('show', implementation+':'+BINDING) == canonical(binding), 'REVIEWED_BINDINGS_CHANGED')
    current_guard = g.git_bytes('show', implementation+':'+GUARD)
    require(previous_guard(current_guard, binding) == g.git_bytes('show', BASE+':'+GUARD), 'PREVIOUS_GUARD_LOGIC_CHANGED')
    require(sha(g.git_bytes('show', BASE+':'+GUARD)) == binding['predecessor_guard']['sha256'], 'PREVIOUS_GUARD_CHANGED')
    # Verify every exact predecessor input, not just the historical test's asserted result.
    helper_path = 'tests/ci_fixture_stability_inputs.py'
    helper_raw = g.git_bytes('show', implementation+':'+helper_path)
    require(blob(helper_raw) == binding['reviewed_blobs'][helper_path], 'PREDECESSOR_INPUT_MODULE_CHANGED')
    namespace = {'__file__': str(g.ROOT/helper_path), '__name__': 'exact_ci_fixture_preimages'}
    exec(compile(helper_raw, namespace['__file__'], 'exec'), namespace)
    require(namespace['BASE'] == BASE and len(namespace['RECORDS']) == 25, 'PREDECESSOR_INPUT_SET_CHANGED')
    for path, record in namespace['RECORDS'].items():
        before = g.git_bytes('show', BASE+':'+path)
        after = g.git_bytes('show', implementation+':'+path)
        require(sha(before) == record['before_sha256'] and blob(before) == record['before_git_blob'], 'PREDECESSOR_INPUT_CHANGED')
        require(sha(after) == record['after_sha256'] and namespace['original_source'](path, after) == before, 'PREDECESSOR_INPUT_CHANGED')
    # The fixture edits and the one harness adjustment have exact, reversible semantics.
    family = binding['fixture_paths']
    require(len(family) == len(set(family)) == 22, 'FIXTURE_FAMILY_CHANGED')
    for path in family:
        before = g.git_bytes('show', BASE+':'+path)
        after = g.git_bytes('show', implementation+':'+path)
        edits = binding['fixture_insertions'][path]
        value = after
        for insertion in reversed(edits):
            at = insertion['offset']; added = insertion['bytes'].encode()
            require(value[at:at+len(added)] == added, 'FIXTURE_SAFEGUARD_CHANGED')
            value = value[:at] + value[at+len(added):]
        require(value == before and len(edits) == 3, 'PRIOR_ASSERTIONS_CHANGED')
    ui = 'tests/test_drive_history_loading.py'
    before = g.git_bytes('show', BASE+':'+ui)
    after = g.git_bytes('show', implementation+':'+ui)
    require(after.replace(b'source+"\\nmain()", default_timeout=10', b'source+"\\nmain()"') == before, 'PRIOR_ASSERTIONS_CHANGED')
    parents = g.git('show', '-s', '--format=%P', 'HEAD').split()
    candidate = g.git('rev-parse', 'HEAD')
    if len(parents) == 2:
        require(parents[0] == BASE, 'CI_BASE_PARENT_NOT_APPROVED')
        candidate = parents[1]
        require(g.git('rev-parse', 'HEAD^{tree}') == g.git('rev-parse', candidate+'^{tree}'), 'CI_MERGE_TREE_CHANGED')
    require(g.git('show', '-s', '--format=%P', candidate).split() == [implementation], 'CANDIDATE_NOT_POLICY_SEAL')
    require(g.git('diff', '--name-status', implementation, candidate).splitlines() == ['A\t'+POLICY], 'SEAL_CHANGE_SET_NOT_APPROVED')
    head = trees(g, 'HEAD')
    require(set(head) == set(policy['unchanged_bindings'])|set(binding['paths'])|{POLICY}, 'UNREVIEWED_PATH')
    require(all(head.get(p) == v for p, v in policy['unchanged_bindings'].items()), 'IMMUTABLE_FILE_CHANGED')
    for path, value in head.items():
        content = bytes_at(g.ROOT, path)
        require(blob(content) == value or blob(content.replace(b'\r\n', b'\n')) == value, 'UNAUTHORIZED_CHECKOUT_CHANGE')
    for path in g.ROOT.rglob('*'):
        if not path.is_file() or '.git' in path.relative_to(g.ROOT).parts:
            continue
        relative = path.relative_to(g.ROOT).as_posix()
        require(path.name not in {'sitecustomize.py', 'usercustomize.py'} and path.suffix != '.pth', 'PROTECTED_RUNTIME_SHADOWING_RISK')
        require(not any(relative.endswith('/'+p) for p in json.loads(raw)['protected_files']), 'PROTECTED_RUNTIME_SHADOWING_RISK')
    return policy


def run(g, manifest_path, base, binding):
    try:
        policy = validate(g, manifest_path, base, binding)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.CalledProcessError) as exc:
        return 1, dict(schema_version=85, status='FAIL', policy_valid=False, reason_codes=[str(exc)])
    # The successor compares exact snapshots against its authenticated starting
    # main. Do not require older Git ancestry merely to render the legacy report
    # in a shallow checkout. Its actual comparison context is explicit below.
    _, original = g.run(manifest_path, binding['base'])
    return 0, dict(schema_version=85, status='PASS', policy_valid=True, reason_codes=[],
        base_sha=binding['base'], head_sha=g.git('rev-parse', 'HEAD'), original_guard_report=original,
        original_guard_report_comparison_base=binding['base'], requested_workflow_comparison_base=base,
        approved_integration_changes=policy['implementation_changes'], unchanged_bindings=policy['unchanged_bindings'],
        protected_changes=[], existing_test_changes=[],
        approved_fixture_inputs='Authenticated exact predecessor sources; assertions and current validation unchanged.',
        approved_test_budget='Only the completed-analysis deferred-history AppTest uses ten seconds; production deadlines unchanged.',
        historical_git_deleting_process='UNKNOWN', historical_Streamlit_timeout_cause='UNKNOWN')
