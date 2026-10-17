"""Exact append-only owner-reviewed private-research successor. No inherited scope is reset."""
import hashlib
import json
from pathlib import Path
import subprocess

POLICY = 'docs/paid-launch/launch-scope-policy-ncaaf-future-spread-v1.json'
BINDING = 'docs/paid-launch/ncaaf-future-spread-binding-v1.json'
TEST_CONFIG = 'pytest.ini'
CONFIG_ANCHOR = b'testpaths = tests\n'
CONFIG_CLOCK = b''


def blobs(g, revision):
    return {path: meta.split()[2] for line in g.git('ls-tree', '-r', revision).splitlines()
            for meta, path in [line.split('\t', 1)]}


def make_policy(g, binding, implementation):
    before = blobs(g, binding['base']); after = blobs(g, implementation)
    return dict(schema_version=79, policy_version='ncaaf-future-spread-v1',
        approval_reference=binding['approval_reference'], base_sha=binding['base'], base_tree=binding['base_tree'],
        original_manifest_sha256=binding['manifest_sha256'], predecessor_policy_blob=before[g.NCAAF_STAGED_CAPACITY_POLICY_PATH],
        implementation_commit=implementation, implementation_tree=g.git('rev-parse', implementation+'^{tree}'),
        implementation_changes={p: {'before_blob': before.get(p), 'after_blob': after.get(p)} for p in binding['paths']},
        unchanged_bindings={p: value for p, value in before.items() if p not in binding['paths']})



def validate_postmerge_successor(g, binding, implementation, correction):
    """Authenticate the exact merged predecessor; never waive its protections."""
    require = g._require
    require(set(correction) == {'proposal_sha256', 'base', 'base_tree', 'ordered_parents',
            'previous_binding_sha256', 'previous_policy_sha256', 'previous_guard_sha256',
            'paths'}, 'POSTMERGE_BINDING_FIELDS_CHANGED')
    require(correction['proposal_sha256'] == '08d51c2b440090e5adcf3ef87bd9a7cea2d9587da1c6c091cb59d6bcebd0bc16'
            and correction['base'] == binding['main_base'] == 'f8ee244cea78c96ec244a15cda0753e1b7a09c11'
            and correction['base_tree'] == binding['main_tree'] == '6658abdb587fca5f6053f94a4d4e8bb3447ba84f',
            'POSTMERGE_BASE_NOT_APPROVED')
    base = correction['base']
    require(g.git('rev-parse', base+'^{tree}') == correction['base_tree']
            and g.git('show', '-s', '--format=%P', base).split() == correction['ordered_parents']
            == ['ab0049adc88e913b8a2d7bda123bfeecd403dc52', '8d474ce47deab00810441cdb141ab5e306953dee'],
            'POSTMERGE_PREDECESSOR_CHANGED')
    prior_binding_raw = g.git_bytes('show', base+':'+BINDING)
    prior_policy = g.git_bytes('show', base+':'+POLICY)
    prior_guard = g.git_bytes('show', base+':'+g.GUARD_PATH)
    require(hashlib.sha256(prior_binding_raw).hexdigest() == correction['previous_binding_sha256']
            and hashlib.sha256(prior_policy).hexdigest() == correction['previous_policy_sha256']
            and hashlib.sha256(prior_guard).hexdigest() == correction['previous_guard_sha256'],
            'POSTMERGE_PREDECESSOR_HASH_CHANGED')
    previous = json.loads(prior_binding_raw)
    adaptations = {'main_base', 'main_tree', 'approval_reference', 'reviewed_blobs', 'paths', 'postmerge_correction'}
    require(set(binding) == set(previous) | {'postmerge_correction'}
            and all(binding[k] == value for k, value in previous.items() if k not in adaptations),
            'POSTMERGE_INHERITED_BINDINGS_CHANGED')
    changed = set(correction['paths'])
    require(changed == {g.GUARD_PATH, BINDING, 'scripts/ncaaf_future_spread_scope.py',
            'tests/test_ncaaf_postmerge_scope.py', 'docs/paid-launch/ncaaf-postmerge-scope-correction-v1.txt'}
            and set(binding['paths']) == set(previous['paths']) | changed,
            'POSTMERGE_CHANGE_SET_NOT_APPROVED')
    require(g.git('show', '-s', '--format=%P', implementation).split() == [base]
            and set(g.git('diff', '--name-only', base, implementation).splitlines()) == changed,
            'IMPLEMENTATION_CHANGE_SET_NOT_APPROVED')
    require(g.git_bytes('show', implementation+':'+POLICY) == prior_policy,
            'POLICY_SEAL_MUST_FOLLOW_IMPLEMENTATION')
    require(set(g.git('diff', '--name-only', binding['base'], implementation).splitlines())
            == set(binding['paths']) | {POLICY}, 'IMPLEMENTATION_CHANGE_SET_NOT_APPROVED')
    # Historical paths not named by this correction must retain their exact blobs.
    before = blobs(g, base); after = blobs(g, implementation)
    require(all(after.get(p) == value for p, value in before.items() if p not in changed),
            'IMMUTABLE_FILE_CHANGED')


def validate(g, manifest_path, base, binding):
    require = g._require
    require(manifest_path.resolve() == (g.ROOT/g.MANIFEST_PATH).resolve(), 'MANIFEST_PATH_NOT_APPROVED')
    require(not g.git('diff', '--name-only') and not g.git('diff', '--cached', '--name-only'), 'UNAUTHORIZED_CHECKOUT_CHANGE')
    require(g.git('rev-parse', binding['base']+'^{tree}') == binding['base_tree'], 'STARTING_MAIN_TREE_CHANGED')
    original = g.git_bytes('show', binding['base']+':'+g.GUARD_PATH)
    require(hashlib.sha256(original).hexdigest() == binding['previous_guard_sha256'], 'PREVIOUS_GUARD_CHANGED')
    manifest = g.git_bytes('show', binding['base']+':'+g.MANIFEST_PATH)
    require(hashlib.sha256(manifest).hexdigest() == binding['manifest_sha256'], 'BASELINE_CHANGED')
    require(base in (None, binding['base'], binding['main_base'], 'ab0049adc88e913b8a2d7bda123bfeecd403dc52',
                     json.loads(manifest).get('base_sha')), 'COMPARISON_BASE_NOT_APPROVED')
    binding_bytes = g.git_bytes('show', 'HEAD:'+BINDING)
    require(hashlib.sha256(binding_bytes).hexdigest() == g.NCAAF_FUTURE_BINDING_SHA256 and json.loads(binding_bytes) == binding, 'REVIEWED_BINDINGS_CHANGED')
    require(hashlib.sha256(g.NCAAF_FUTURE_RECONSTRUCTION_JSON).hexdigest() == binding['reconstruction_json_sha256'] and json.loads(g.NCAAF_FUTURE_RECONSTRUCTION_JSON) == binding['source_reconstructions'], 'REVIEWED_RECONSTRUCTIONS_CHANGED')
    require(not g.exists_at(binding['base'], POLICY), 'SUCCESSOR_POLICY_ALREADY_EXISTS')
    raw = g.git_bytes('show', 'HEAD:'+POLICY); policy = json.loads(raw)
    implementation = policy['implementation_commit']
    correction = binding.get('postmerge_correction')
    if correction is not None:
        validate_postmerge_successor(g, binding, implementation, correction)
    else:
        ancestors = binding.get('implementation_ancestors', [])
        require(isinstance(ancestors, list) and len(ancestors) <= 2, 'IMPLEMENTATION_PARENT_NOT_APPROVED')
        parent = binding['base']
        for ancestor in ancestors:
            require(isinstance(ancestor, str) and len(ancestor) == 40 and
                    g.git('show', '-s', '--format=%P', ancestor).split() == [parent],
                    'IMPLEMENTATION_PARENT_NOT_APPROVED')
            require(not g.exists_at(ancestor, POLICY), 'POLICY_SEAL_MUST_FOLLOW_IMPLEMENTATION')
            require(set(g.git('diff', '--name-only', binding['base'], ancestor).splitlines()) <= set(binding['paths']),
                    'IMPLEMENTATION_CHANGE_SET_NOT_APPROVED')
            parent = ancestor
        require(g.git('show', '-s', '--format=%P', implementation).split() == [parent], 'IMPLEMENTATION_PARENT_NOT_APPROVED')
        require(not g.exists_at(implementation, POLICY), 'POLICY_SEAL_MUST_FOLLOW_IMPLEMENTATION')
        require(set(g.git('diff', '--name-only', binding['base'], implementation).splitlines()) == set(binding['paths']), 'IMPLEMENTATION_CHANGE_SET_NOT_APPROVED')
    require(policy == make_policy(g, binding, implementation), 'EXACT_POLICY_BINDINGS_CHANGED')
    require(g.git('rev-parse', binding['main_base']+'^{tree}') == binding['main_tree'], 'STARTING_MAIN_TREE_CHANGED')
    if TEST_CONFIG in binding['paths']:
        original_config = g.git_bytes('show', binding['base']+':'+TEST_CONFIG)
        require(original_config.count(CONFIG_ANCHOR) == 1, 'SYNTHETIC_CLOCK_FIXTURE_CHANGED')
        require(g.git_bytes('show', implementation+':'+TEST_CONFIG) ==
                original_config.replace(CONFIG_ANCHOR, CONFIG_ANCHOR+CONFIG_CLOCK),
                'TEST_CONFIGURATION_NOT_APPROVED')
    reviewed = blobs(g, implementation)
    require(set(binding['reviewed_blobs']) == set(binding['paths']) - {g.GUARD_PATH, BINDING}, 'REVIEWED_BINDING_SET_CHANGED')
    require(all(reviewed.get(p) == value for p, value in binding['reviewed_blobs'].items()), 'REVIEWED_IMPLEMENTATION_BYTES_CHANGED')
    successor = g.git_bytes('show', implementation+':'+g.GUARD_PATH)
    require(g._ncaaf_future_guard_matches(successor), 'SUCCESSOR_GUARD_REVIEWED_BYTES_CHANGED')
    require(g._ncaaf_future_base_source(successor) == original, 'PREVIOUS_GUARD_LOGIC_CHANGED')
    parents = g.git('show', '-s', '--format=%P', 'HEAD').split()
    if len(parents) == 2:
        require(parents[0] == binding['main_base'], 'CI_BASE_PARENT_NOT_APPROVED')
        candidate = parents[1]
        require(g.git('rev-parse', 'HEAD^{tree}') == g.git('rev-parse', candidate+'^{tree}'), 'CI_MERGE_TREE_CHANGED')
    else:
        candidate = g.git('rev-parse', 'HEAD')
    require(g.git('show', '-s', '--format=%P', candidate).split() == [implementation], 'CANDIDATE_NOT_POLICY_SEAL')
    seal_status = 'M' if correction is not None else 'A'
    require(g.git('diff', '--name-status', implementation, candidate).splitlines() == [seal_status+'\t'+POLICY], 'SEAL_CHANGE_SET_NOT_APPROVED')
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
        return 1, dict(schema_version=79, status='FAIL', policy_valid=False, reason_codes=[str(exc)])
    # Retain the original baseline report verbatim. Exact frozen-main bindings
    # account for inherited changes without granting new test exceptions.
    _, original = g.run(manifest_path, base)
    return 0, dict(schema_version=79, status='PASS', policy_valid=True, reason_codes=[],
        base_sha=binding['base'], head_sha=g.git('rev-parse', 'HEAD'),
        original_guard_report=original, approved_integration_changes=policy['implementation_changes'],
        unchanged_bindings=policy['unchanged_bindings'], protected_changes=[],
        existing_test_changes=[], new_existing_test_exceptions=[],
        approved_successor_fixture_changes=binding.get('approved_existing_test_changes', {}),
        checkout_line_ending_conversions=conversions,
        scope='Explicit reviewed future target, unchanged native integer masses and exact two-side private comparison. No live collection, admission catalog, mathematics, workflow, profile or operational changes.')


def verify_artifact_provenance(run,artifact,archive_sha256,*,run_id,head,tree,tested_tree):
    """Read-only closeout check. GitHub run/ZIP evidence, not manifest inference."""
    if not (run['id']==run_id and run['head_sha']==head and artifact['workflow_run']['id']==run_id
            and artifact['workflow_run']['head_sha']==head and tree==tested_tree
            and artifact['digest']=='sha256:'+archive_sha256 and not artifact['expired']):
        raise ValueError('CAPACITY_ARTIFACT_REVISION_OR_HASH_MISMATCH')
    return True
