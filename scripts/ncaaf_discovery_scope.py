"""Exact append-only discovery/setup successor. No inherited scope is reset."""
import hashlib
import json
from pathlib import Path
import subprocess

POLICY = 'docs/paid-launch/launch-scope-policy-ncaaf-discovery-v1.json'
BINDING = 'docs/paid-launch/ncaaf-pilot-discovery-binding-v1.json'


def blobs(g, revision):
    return {path: meta.split()[2] for line in g.git('ls-tree', '-r', revision).splitlines()
            for meta, path in [line.split('\t', 1)]}


def make_policy(g, binding, implementation):
    before = blobs(g, binding['base']); after = blobs(g, implementation)
    return dict(schema_version=33, policy_version='ncaaf-discovery-local-setup-v1',
        approval_reference=binding['approval_reference'], base_sha=binding['base'], base_tree=binding['base_tree'],
        original_manifest_sha256=binding['manifest_sha256'], predecessor_policy_blob=before[g.NCAAF_PILOT_POLICY_PATH],
        implementation_commit=implementation, implementation_tree=g.git('rev-parse', implementation+'^{tree}'),
        implementation_changes={p: {'before_blob': before.get(p), 'after_blob': after.get(p)} for p in binding['paths']},
        unchanged_bindings={p: value for p, value in before.items() if p not in binding['paths']})


def validate(g, manifest_path, base, binding):
    require = g._require
    require(manifest_path.resolve() == (g.ROOT/g.MANIFEST_PATH).resolve(), 'MANIFEST_PATH_NOT_APPROVED')
    require(base in (None, binding['base'], '5fb8e13577c3092f1eda4ad9b787368d3f691c71'), 'COMPARISON_BASE_NOT_APPROVED')
    require(not g.git('diff', '--name-only') and not g.git('diff', '--cached', '--name-only'), 'UNAUTHORIZED_CHECKOUT_CHANGE')
    require(g.git('rev-parse', binding['base']+'^{tree}') == binding['base_tree'], 'STARTING_MAIN_TREE_CHANGED')
    original = g.git_bytes('show', binding['base']+':'+g.GUARD_PATH)
    require(hashlib.sha256(original).hexdigest() == binding['previous_guard_sha256'], 'PREVIOUS_GUARD_CHANGED')
    manifest = g.git_bytes('show', binding['base']+':'+g.MANIFEST_PATH)
    require(hashlib.sha256(manifest).hexdigest() == binding['manifest_sha256'], 'BASELINE_CHANGED')
    require(json.loads(g.git_bytes('show', 'HEAD:'+BINDING)) == binding, 'REVIEWED_BINDINGS_CHANGED')
    require(not g.exists_at(binding['base'], POLICY), 'SUCCESSOR_POLICY_ALREADY_EXISTS')
    raw = g.git_bytes('show', 'HEAD:'+POLICY); policy = json.loads(raw)
    implementation = policy['implementation_commit']
    require(g.git('show', '-s', '--format=%P', implementation).split() == [binding['base']], 'IMPLEMENTATION_PARENT_NOT_APPROVED')
    require(not g.exists_at(implementation, POLICY), 'POLICY_SEAL_MUST_FOLLOW_IMPLEMENTATION')
    require(set(g.git('diff', '--name-only', binding['base'], implementation).splitlines()) == set(binding['paths']), 'IMPLEMENTATION_CHANGE_SET_NOT_APPROVED')
    require(policy == make_policy(g, binding, implementation), 'EXACT_POLICY_BINDINGS_CHANGED')
    reviewed = blobs(g, implementation)
    require(set(binding['reviewed_blobs']) == set(binding['paths']) - {g.GUARD_PATH, BINDING}, 'REVIEWED_BINDING_SET_CHANGED')
    require(all(reviewed.get(p) == value for p, value in binding['reviewed_blobs'].items()), 'REVIEWED_IMPLEMENTATION_BYTES_CHANGED')
    successor = g.git_bytes('show', implementation+':'+g.GUARD_PATH)
    require(g._discovery_guard_matches(successor), 'SUCCESSOR_GUARD_REVIEWED_BYTES_CHANGED')
    require(g._discovery_parent_guard_source(successor) == original, 'PREVIOUS_GUARD_LOGIC_CHANGED')
    parents = g.git('show', '-s', '--format=%P', 'HEAD').split()
    if len(parents) == 2:
        require(parents[0] == binding['base'], 'CI_BASE_PARENT_NOT_APPROVED')
        candidate = parents[1]
        require(g.git('rev-parse', 'HEAD^{tree}') == g.git('rev-parse', candidate+'^{tree}'), 'CI_MERGE_TREE_CHANGED')
    else:
        candidate = g.git('rev-parse', 'HEAD')
    require(g.git('show', '-s', '--format=%P', candidate).split() == [implementation], 'CANDIDATE_NOT_POLICY_SEAL')
    require(g.git('diff', '--name-status', implementation, candidate).splitlines() == ['A\t'+POLICY], 'SEAL_CHANGE_SET_NOT_APPROVED')
    head = blobs(g, 'HEAD')
    require(all(head.get(p) == value for p, value in policy['unchanged_bindings'].items()), 'IMMUTABLE_FILE_CHANGED')
    require(set(head) == set(policy['unchanged_bindings']) | set(binding['paths']) | {POLICY}, 'UNREVIEWED_PATH')
    conversions = []
    for path, expected in head.items():
        checkout = (g.ROOT/path).read_bytes()
        def blob(raw): return hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if blob(checkout) != expected:
            require(blob(checkout.replace(b'\r\n', b'\n')) == expected, 'UNAUTHORIZED_CHECKOUT_CHANGE')
            conversions.append(path)
    # Untracked duplicates cannot shadow frozen runtime, nor inject startup hooks.
    protected = json.loads(manifest)['protected_files']
    for other in g.ROOT.rglob('*'):
        if not other.is_file(): continue
        rel = other.relative_to(g.ROOT).as_posix()
        require(other.name not in {'sitecustomize.py', 'usercustomize.py'} and other.suffix != '.pth', 'PROTECTED_RUNTIME_SHADOWING_RISK')
        for path in protected:
            if rel.endswith('/'+path) and rel not in policy['unchanged_bindings']:
                require(False, 'PROTECTED_RUNTIME_SHADOWING_RISK')
    return policy, conversions


def run(g, manifest_path, base, binding):
    try:
        policy, conversions = validate(g, manifest_path, base, binding)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.CalledProcessError) as exc:
        return 1, dict(schema_version=33, status='FAIL', policy_valid=False, reason_codes=[str(exc)])
    # Retain the original baseline report verbatim. Exact frozen-main bindings
    # account for inherited changes without granting new test exceptions.
    _, original = g.run(manifest_path, base)
    return 0, dict(schema_version=33, status='PASS', policy_valid=True, reason_codes=[],
        base_sha=binding['base'], head_sha=g.git('rev-parse', 'HEAD'),
        original_guard_report=original, approved_integration_changes=policy['implementation_changes'],
        unchanged_bindings=policy['unchanged_bindings'], protected_changes=[], existing_test_changes=[],
        new_existing_test_exceptions=[], checkout_line_ending_conversions=conversions)
