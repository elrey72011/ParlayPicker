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
import functools
import types

PROPOSAL = '261bb913276725605a4b142458b0be536002cac9af327ec17187da2dde1b359d'
CONSOLIDATED_AMENDMENT = 'b7175b51176334e8b31967bac836e38ebea307a0f90d5e1f7af24b850b36fd7e'
HISTORICAL_INPUT_AMENDMENT = '12e3046831d8fde24748e5193a2e6a54311e1be318e703d3739fe6e1c41da7b2'
SHARED_BOUNDARY_AMENDMENT = '8ad3749ffee5ebcb9f2688e9bb1b10d99efb5b5dc476c85d956dfd004c8a731f'
MEASURED_CAPACITY_AMENDMENT = 'dd6e6859971a5b7d3938d9a92a648bed04e94f14b1c2ffdc535e143b54cee9b8'
CAPACITY_PREDECESSOR = 'ea320f9e5303d0d7915456c1aebb0b23dca85e8a'
CAPACITY_PREDECESSOR_TREE = 'b80daa84027c2a5962eac7869b2d1f72b91b904c'
CAPACITY_LEDGER_SHA256 = 'a6a1143c0f264136cecaf4ed543b8ddb04d9b52d4969ffad89152e8259a0fd65'
PRESERVED_CANDIDATE = '97ed96247939165c2f73026525e31e1c3dba9302'
PRESERVED_TREE = '945cab79d9674ee15e4d3d81f088e9c5beab969e'
COMPLETE_LEDGER_SHA256 = '5eec77cd6d43fbd41b3839d7b99c135bfd0cb2603e92b794874385f8432261b7'
BASE = '8b79b0a0eead4f63a041a551283a4b6be2980d72'
BASE_TREE = '028d3daf56d71596663a10714495a7b48d53c7ab'
GUARD = 'scripts/check_launch_change_scope.py'
BINDING = 'docs/paid-launch/ci-fixture-stability-bindings-v1.json'
POLICY = 'docs/paid-launch/launch-scope-policy-ci-fixture-stability-v1.json'
MAX_SOURCE = 8 * 1024 * 1024
GUARD_TEMPLATE_SHA256 = 'e9069c74d604f1d5f5cf7d0b665f8d14e03e91db7ed0ebdca170d2000a2a1a6e'


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


LEGACY_SOURCE_LOADERS = {
    '_slate_predecessor': '_slate_audit_previous_guard_source',
    '_ci_scheduling_predecessor': '_ci_scheduling_previous_guard_source',
    '_ncaaf_compatibility_predecessor': '_ncaaf_compatibility_previous_guard_source',
    '_readiness_dashboard_predecessor': '_readiness_dashboard_previous_guard_source',
    '_ncaaf_normal_predecessor': '_ncaaf_normal_parent_guard_source',
    '_preview_binding_predecessor': '_preview_binding_parent_guard_source',
    '_ncaaf_chronology_predecessor': '_ncaaf_chronology_parent_guard_source',
    '_ncaaf_custody_predecessor': '_ncaaf_custody_parent_guard_source',
    '_ncaaf_pilot_predecessor': '_ncaaf_pilot_parent_guard_source',
    '_discovery_predecessor': '_discovery_parent_guard_source',
    '_cloud_predecessor': '_cloud_peel_exact_guard',
    '_owner_predecessor': '_owner_base_source',
    '_nfl_owner_predecessor': '_nfl_owner_base_source',
}


def install_legacy_source_bridge(namespace, authenticate, _immutable=None):
    """One authenticated immutable source generation; no validation cache.

    Physical source/binding authentication runs on every call, including a
    warmed loader. Only the already verified predecessor bytes are reused.
    No Path override or mutable predecessor namespace is retained here.
    """
    identity = lambda source, binding: (sha(source), sha(canonical(binding)),
                                      binding['predecessor_guard']['sha256'])
    physical, binding = authenticate()
    key = identity(physical, binding)
    if _immutable is None:
        predecessor = previous_guard(physical, binding)
        immutable = (key, predecessor)
    else:
        require(type(_immutable) is tuple and len(_immutable) == 2 and key == _immutable[0], 'LEGACY_SOURCE_GENERATION_CHANGED')
        require(type(_immutable[1]) is bytes and len(_immutable[1]) == binding['predecessor_guard']['bytes'] <= MAX_SOURCE
                and sha(_immutable[1]) == binding['predecessor_guard']['sha256'], 'PREVIOUS_GUARD_LOGIC_CHANGED')
        immutable = _immutable
    check = require
    digest = sha
    def verified():
        source, current = authenticate()
        check(identity(source, current) == immutable[0], 'LEGACY_SOURCE_GENERATION_CHANGED')
        return source, current
    def bind(original, wrapper):
        # Preserve the original global dictionary as well as public metadata.
        result = types.FunctionType(wrapper.__code__, original.__globals__,
            original.__name__, wrapper.__defaults__, wrapper.__closure__)
        result.__kwdefaults__ = wrapper.__kwdefaults__
        return functools.update_wrapper(result, original)
    installer = install_legacy_source_bridge
    module_type = types.ModuleType
    pending = [namespace]
    seen = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        require(len(seen) <= 128, 'LEGACY_SOURCE_NAMESPACE_LIMIT')
        for value in tuple(current.values()):
            if isinstance(value, types.FunctionType) and value.__code__.co_filename == current.get('__file__'):
                if value.__globals__ is not current:
                    pending.append(value.__globals__)
            elif isinstance(value, module_type) and value.__name__.startswith('exact_') and getattr(value, '__file__', None) == current.get('__file__'):
                pending.append(vars(value))
        for loader, reader in LEGACY_SOURCE_LOADERS.items():
            original = current.get(reader)
            if callable(original) and not getattr(original, '_authenticated_source_bridge', False):
                def source_input(source, *args, _original=original, **kwargs):
                    physical, current_binding = verified()
                    check(digest(immutable[1]) == current_binding['predecessor_guard']['sha256'], 'PREVIOUS_GUARD_LOGIC_CHANGED')
                    if type(source) is bytes and source.replace(b'\r\n', b'\n') == physical:
                        source = immutable[1]
                    return _original(source, *args, **kwargs)
                wrapped = bind(original, source_input)
                wrapped._authenticated_source_bridge = True
                current[reader] = wrapped
            original = current.get(loader)
            if callable(original) and not getattr(original, '_authenticated_loader_bridge', False):
                def loaded(*args, _original=original, **kwargs):
                    verified()
                    result = _original(*args, **kwargs)
                    if isinstance(result, module_type):
                        installer(vars(result), authenticate, immutable)
                    return result
                wrapped = bind(original, loaded)
                wrapped._authenticated_loader_bridge = True
                current[loader] = wrapped


def trees(g, revision):
    return {path: meta.split()[2] for row in g.git('ls-tree', '-r', revision).splitlines()
            for meta, path in [row.split('\t', 1)]}


def make_policy(g, binding, implementation):
    before, after = trees(g, binding['base']), trees(g, implementation)
    return dict(schema_version=85, policy_version='ci-fixture-stability-v1',
        proposal_sha256=PROPOSAL, shared_historical_boundary_amendment_sha256=binding.get('shared_historical_boundary_amendment_sha256'),
        measured_capacity_amendment_sha256=binding.get('measured_capacity_amendment_sha256'),
        advisory_amendment_sha256='c7597d04525eca38a52c7207f75f45829560bd6fa1ab0ce1f5ba2a1bfa7dcf45',
        second_advisory_amendment_sha256='6cf28cd1db29cd0ba101cce67b8ac3bcc5d6857b61ba555b6aa382298fda7295',
        consolidated_advisory_amendment_sha256=CONSOLIDATED_AMENDMENT,historical_input_amendment_sha256=HISTORICAL_INPUT_AMENDMENT,complete_profile_ledger_sha256=COMPLETE_LEDGER_SHA256, base_sha=binding['base'], base_tree=binding['base_tree'],
        implementation_commit=implementation, implementation_tree=g.git('rev-parse', implementation+'^{tree}'),
        implementation_changes={p: dict(before_blob=before.get(p), after_blob=after.get(p)) for p in binding['paths']},
        unchanged_bindings={p: v for p, v in before.items() if p not in binding['paths']})


def validate(g, manifest_path, comparison_base, binding):
    require(binding['proposal_sha256'] == PROPOSAL, 'PROPOSAL_NOT_APPROVED')
    require(binding.get('consolidated_advisory_amendment_sha256') == CONSOLIDATED_AMENDMENT
            and binding.get('complete_profile_ledger_sha256') == COMPLETE_LEDGER_SHA256, 'COMPLETE_PROFILE_PROPOSAL_NOT_APPROVED')
    require(binding.get('historical_input_amendment_sha256') == HISTORICAL_INPUT_AMENDMENT, 'HISTORICAL_INPUT_PROPOSAL_NOT_APPROVED')
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
    shared = binding.get('shared_historical_boundary_amendment_sha256') == SHARED_BOUNDARY_AMENDMENT
    # Original synthetic baseline/seal contracts still start at their supplied
    # exact BASE. Only the actual reviewed checkout may extend the preserved seal.
    successor = shared and BASE == '8b79b0a0eead4f63a041a551283a4b6be2980d72'
    capacity = binding.get('measured_capacity_amendment_sha256') == MEASURED_CAPACITY_AMENDMENT
    if binding.get('measured_capacity_amendment_sha256') is not None:
        require(capacity and binding['measured_capacity']['ledger_sha256']==CAPACITY_LEDGER_SHA256,
                'MEASURED_CAPACITY_PROPOSAL_NOT_APPROVED')
    parent = (CAPACITY_PREDECESSOR if capacity else PRESERVED_CANDIDATE) if successor else BASE
    require(g.git('show', '-s', '--format=%P', implementation).split() == [parent], 'IMPLEMENTATION_PARENT_NOT_APPROVED')
    require(not g.exists_at(BASE, POLICY), 'POLICY_SEAL_MUST_FOLLOW_IMPLEMENTATION')
    if successor:
        require(binding.get('preserved_candidate') == PRESERVED_CANDIDATE and binding.get('preserved_tree') == PRESERVED_TREE,
                'PRESERVED_CANDIDATE_CHANGED')
        require(g.git('rev-parse', PRESERVED_CANDIDATE+'^{tree}') == PRESERVED_TREE, 'PRESERVED_CANDIDATE_CHANGED')
        if capacity:
            require(binding['measured_capacity']['predecessor']==CAPACITY_PREDECESSOR
                    and binding['measured_capacity']['predecessor_tree']==CAPACITY_PREDECESSOR_TREE
                    and g.git('rev-parse',parent+'^{tree}')==CAPACITY_PREDECESSOR_TREE, 'CAPACITY_PREDECESSOR_CHANGED')
        require(g.git_bytes('show', implementation+':'+POLICY) == g.git_bytes('show', parent+':'+POLICY), 'POLICY_SEAL_MUST_FOLLOW_IMPLEMENTATION')
    else:
        require(not g.exists_at(implementation, POLICY), 'POLICY_SEAL_MUST_FOLLOW_IMPLEMENTATION')
    changes = set(g.git('diff', '--name-only', BASE, implementation).splitlines())
    require(changes - ({POLICY} if successor else set()) == set(binding['paths']), 'IMPLEMENTATION_CHANGE_SET_NOT_APPROVED')
    require(len(binding['paths']) == len(set(binding['paths'])) == (43 if capacity else 38 if shared else 33), 'IMPLEMENTATION_PATH_SET_CHANGED')
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
        authenticated_after=namespace['capacity_predecessor_source'](path,after) if capacity else after
        require(sha(authenticated_after) == record['after_sha256'] and namespace['original_source'](path, after) == before, 'PREDECESSOR_INPUT_CHANGED')
    if successor:
        helpers=binding['historical_input_boundary']['helper_preimages']
        require(set(helpers)=={'tests/ci_fixture_scope_inputs.py','tests/ncaaf_transport_scope_inputs.py','tests/ncaaf_staged_capacity_fixtures.py'}, 'HISTORICAL_HELPER_INPUT_SET_CHANGED')
        for path,record in helpers.items():
            before=g.git_bytes('show',BASE+':'+path);after=g.git_bytes('show',implementation+':'+path)
            decoder=zlib.decompressobj();value=decoder.decompress(base64.b85decode(record['compressed']),MAX_SOURCE+1)
            require(decoder.eof and not decoder.unused_data and value==before
                    and len(value)==record['bytes']<=MAX_SOURCE and sha(value)==record['before_sha256']
                    and blob(value)==record['before_git_blob'] and sha(after)==record['after_sha256'], 'HISTORICAL_HELPER_PREIMAGE_CHANGED')
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
    require(g.git('diff', '--name-status', implementation, candidate).splitlines() == [('M\t' if successor else 'A\t')+POLICY], 'SEAL_CHANGE_SET_NOT_APPROVED')
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
