"""One explicitly reviewed missing response; never a resume of a spent capture.

The parent stays INCOMPLETE. Its originals, clocks, attempts and full-precision
active time are authenticated separately and charged to this version's budget.
No trust catalog, credential, mutable cache or acquisition authority lives here.
"""
import base64
from copy import deepcopy
from decimal import Decimal
import io
import json
from pathlib import Path

from app_core import ncaaf_filtered_history as filtered, ncaaf_fresh_history as fresh
from app_core import ncaaf_history as history, ncaaf_reviewed_target as target

HISTORY_PLAN = 'ncaaf-history-continuation-stage-plan-v6'
QUOTE_PLAN = 'ncaaf-continuation-final-quote-stage-plan-v6'
HISTORY_BUNDLE = 'ncaaf-history-continuation-unaccepted-v6'
QUOTE_BUNDLE = 'ncaaf-continuation-final-quote-unaccepted-v6'
PREQUOTE = 'ncaaf-continuation-prequote-features-v6'
BINDING = 'ncaaf-continuation-staged-evidence-v6'
CONFIGURATION = 'ncaaf-continuation-controlled-bindings-v6'
POLICY = dict(cumulative_active_seconds=120, market_reserve_seconds=20, header_ceiling_seconds=30)
MATERIALS = frozenset(('bundle', 'journal', 'diagnostics', 'claim', 'report',
    'configuration', 'authorization', 'operation', 'method', 'approval',
    'derivation', 'template', 'application_readback', 'release'))
require = filtered.require


def selected(p):
    return p.get('version') in {HISTORY_PLAN, QUOTE_PLAN}


def decode(raw):
    return json.loads(raw, object_pairs_hook=filtered.custody._unique,
        parse_constant=filtered.custody._constant)


def material(p):
    c = p['continuation']; m = c['parent']
    require(set(c) == {'version', 'parent', 'review', 'budget'}
        and c['version'] == 'ncaaf-spent-history-parent-v6' and c['budget'] == POLICY
        and set(m) == {'blobs_b64', 'pins', 'custody_names'}
        and set(m['blobs_b64']) == set(m['pins']) == set(m['custody_names']) == MATERIALS,
        'NCAAF_CONTINUATION_PARENT_SCHEMA')
    require(all(type(v) is str and len(v) <= filtered.MAX_BUNDLE*4//3+4
        for v in m['blobs_b64'].values()), 'FILTERED_BODY_LIMIT')
    blobs = {k: base64.b64decode(v, validate=True) for k, v in m['blobs_b64'].items()}
    require(sum(map(len, blobs.values())) <= filtered.MAX_BUNDLE
        and all(filtered.sha(v) == m['pins'][k] for k, v in blobs.items()),
        'NCAAF_CONTINUATION_PARENT_INTEGRITY')
    return blobs, m['pins']


def subject(p):
    return filtered.digest(dict(version='ncaaf-spent-history-reuse-subject-v6',
        pins=p['continuation']['parent']['pins'], budget=POLICY,
        target_manifest=p['target_evidence'], source=p['filtered_history']))


def _read_parent(p, *, at, review_required):
    from app_core import ncaaf_staged_acquisition as stage
    from app_core import ncaaf_transport_diagnostics as diagnostics
    blobs, pins = material(p)
    envelope = decode(blobs['bundle']); b = stage.read_bundle(envelope, complete=False)
    old = b['plan']['payload']; c = decode(blobs['configuration'])
    require(old['version'] == fresh.HISTORY_PLAN and b['status'] == 'INCOMPLETE'
        and b['reason'] == 'NCAAF_STAGED_DEADLINE' and b['attempts'] == 3
        and len(b['objects']) == 2 and b['prior_active_seconds'] == b['prior_original_bytes'] == 0
        and old['requests'] == fresh.requests(old['filtered_history']), 'NCAAF_CONTINUATION_PARENT_STATE')
    require(target.canonical(old['target']) == target.canonical(p['target'])
        and old['target_evidence'] == p['target_evidence']
        and old['filtered_history'] == p['filtered_history']
        and old['custody_id'] == p['custody_id'], 'NCAAF_CONTINUATION_TARGET')
    require(base64.b64decode(b['journal_b64'], validate=True) == blobs['journal']
        and filtered.sha(blobs['journal']) == b['journal_sha256']
        and filtered.sha(blobs['configuration']) == b['control_receipt']['file_sha256']
        and base64.b64decode(b['control_receipt']['file_b64'], validate=True) == blobs['configuration']
        and decode(blobs['authorization']) == b['authorization'], 'NCAAF_CONTINUATION_LINKAGE')
    report, claim, method, approval = [decode(blobs[k]) for k in ('report', 'claim', 'method', 'approval')]
    operation = decode(blobs['operation'])
    require(stage._envelope(operation)['runtime'] == c['payload']['runtime']
        and c['payload']['runtime']['operation_manifest_sha256'] == filtered.digest(
            {k:v for k,v in operation['payload'].items() if k != 'runtime'})
        and method['native_operation_file_sha256'] == pins['operation']
        and method['derivation_sha256'] == pins['derivation']
        and method['release_clearance_receipt_sha256'] == pins['release'], 'NCAAF_CONTINUATION_METHOD')
    require(approval['package_sha256'] == pins['method']
        and approval['derivation_template_sha256'] == pins['template']
        and approval['owner'] == target.OWNER
        and approval['decision'] == 'APPROVED_CONSOLIDATED_FRESH_HISTORY_CAPTURE'
        and target.instant(approval['received_at']) <= target.instant(b['started_at']), 'NCAAF_CONTINUATION_APPROVAL')
    readback, release = [decode(blobs[k]) for k in ('application_readback', 'release')]
    require(readback['status'] == 'INDEPENDENT_READBACK_PASS'
        and readback['configuration_file_sha256'] == pins['configuration']
        and readback['plan_sha256'] == b['plan']['sha256']
        and readback['package_sha256'] == pins['method']
        and target.instant(readback['actual_approval_at']) == target.instant(approval['received_at'])
        and target.instant(readback['at']) <= target.instant(b['started_at'])
        and release['status'] == 'RELEASE_CLEARANCE_PASS'
        and release['main'] == c['payload']['runtime']['code_head']
        and release['checks']['expected'] == release['checks']['passed']
        and release['artifacts']['expected'] == release['artifacts']['verified']
        and release['application_outcomes']['native_reconciliation']['status'] == 'PASS',
        'NCAAF_CONTINUATION_READBACK')
    require(claim['version'] == report['version'] == 'ncaaf-reviewed-target-operation-v1'
        and claim['configuration_file_sha256'] == report['configuration_file_sha256'] == pins['configuration']
        and claim['manifest_file_sha256'] == report['manifest_file_sha256'] == pins['operation']
        and claim['plan_sha256'] == report['plan_sha256'] == b['plan']['sha256']
        and report['claim_sha256'] == pins['claim'] and report['bundle_file_sha256'] == pins['bundle']
        and report['bundle_sha256'] == envelope['sha256'] and report['journal_sha256'] == pins['journal']
        and report['diagnostics_file_sha256'] == pins['diagnostics']
        and report['provider_attempts'] == 3 and report['status'] == b['status']
        and report['reason'] == b['reason'] and report['retry'] is False and report['readback'] == 'PASS',
        'NCAAF_CONTINUATION_REPORT')
    # Old diagnostics authenticate against the old, already bound runtime. The
    # ordinary diagnostics reader still requires current physical sources.
    d = stage._envelope(decode(blobs['diagnostics']))
    expected = {k:c['payload']['runtime']['sources'][k] for k in diagnostics.SOURCE_PATHS}
    diagnostics._checked(d, expected_sources=expected)
    require(d['plan_sha256'] == b['plan']['sha256'] and d['authorization_sha256'] == b['authorization']['sha256']
        and d['configuration_file_sha256'] == pins['configuration'] and d['bundle_sha256'] == envelope['sha256']
        and d['bundle_file_sha256'] == pins['bundle'] and d['journal_sha256'] == pins['journal']
        and d['active_seconds'] == b['active_seconds'] and len(d['attempts']) == 3
        and d['status'] == b['status'] and d['reason'] == b['reason']
        and [a['complete_original'] for a in d['attempts']] == [True, True, False], 'NCAAF_CONTINUATION_DIAGNOSTICS')
    now = target.instant(at)
    if review_required:
        r = p['continuation']['review']
        require(set(r) == {'reviewer', 'reviewed_at', 'subject_sha256', 'finding'}
            and r['reviewer'] == target.OWNER and r['subject_sha256'] == subject(p)
            and target.instant(report['finished_at']) <= target.instant(r['reviewed_at']) < now
            and type(r['finding']) is str and len(r['finding'].strip()) >= 32, 'NCAAF_CONTINUATION_REVIEW')
    require(all(0 <= (now-target.instant(o['metadata']['received_at'])).total_seconds() <= 86400
        for o in b['objects']) and now < target.instant(p['target']['start_utc']), 'NCAAF_STAGED_INPUT_CLOCK')
    scoring, contract = fresh.scoring_contract(old, b['objects'][0], at=at)
    filtered.validate_statistics(b['objects'][1], contract, scoring)
    # Authenticate the failed request too; no lost attempt or reset accounting.
    events = [decode(line) for line in blobs['journal'].splitlines()]
    require([e['request'] for e in events if e['kind'] == 'attempt'] == old['requests']
        and events[-1]['stage_active_seconds'] == b['active_seconds'], 'NCAAF_CONTINUATION_ACCOUNTING')
    remaining(b['active_seconds'], prior_bytes=b['original_bytes'])
    return b, dict(original_bytes=b['original_bytes'], charged_active_seconds=b['active_seconds'])


def parent(p, *, at):
    """Operational readers always require the exact separately recorded review."""
    return _read_parent(p,at=at,review_required=True)


def inspect_parent(p, *, at):
    """Read-only preparation facts; supplies no review or acquisition authority."""
    b,facts=_read_parent(p,at=at,review_required=False)
    return dict(facts,plan_sha256=b['plan']['sha256'],attempts=b['attempts'],status=b['status'],reason=b['reason'],
        original_receipts=[o['metadata']['received_at'] for o in b['objects']],
        original_sha256=[o['body_sha256'] for o in b['objects']],
        continuation_seconds=str(remaining(facts['charged_active_seconds'],prior_bytes=facts['original_bytes'])),
        provider_receipt='UNKNOWN',credit_consumption='UNKNOWN',configuration_applied=False,provider_requests=0,inference=False)


def remaining(prior_active, added_active=0, *, prior_bytes=0, added_bytes=0, reserve_seconds=20):
    values = [Decimal(str(v)) for v in (prior_active, added_active, reserve_seconds)]
    require(all(v.is_finite() and v >= 0 for v in values)
        and type(prior_bytes) is int and type(added_bytes) is int
        and min(prior_bytes, added_bytes) >= 0
        and prior_bytes + added_bytes <= 2*1024*1024, 'NCAAF_STAGED_COMBINED_LIMIT')
    value = Decimal(120)-sum(values)
    require(value > 0, 'NCAAF_STAGED_COMBINED_LIMIT')
    return value


def validate_plan(plan, *, executable=False):
    from app_core import ncaaf_staged_acquisition as stage
    p = stage._envelope(plan)
    require(selected(p) and 'continuation' in p, 'NCAAF_CONTINUATION_SCHEMA')
    parent_value, _ = parent(p, at=p['execution_window']['not_before'])
    old = parent_value['plan']['payload']
    expected = [old['requests'][2]] if p['version'] == HISTORY_PLAN else target.requests(p['target'], target.QUOTE_PLAN, p['target_evidence'])
    require(p['requests'] == expected and p['limits'] == dict(stage.LIMITS, shared_active_seconds=120),
        'NCAAF_STAGED_REQUEST_OR_LIMIT_CONFLICT')
    # Reuse the exact structural/target checks, with an explicit validation-only
    # view. This never changes an original, actual request or persisted plan.
    view = deepcopy(p); del view['continuation']
    view['version'] = fresh.HISTORY_PLAN if p['version'] == HISTORY_PLAN else fresh.QUOTE_PLAN
    view['limits'] = deepcopy(stage.LIMITS)
    if p['version'] == HISTORY_PLAN:view['requests'] = deepcopy(old['requests'])
    fresh.validate_plan(stage.seal(view), executable=executable)
    return deepcopy(p)


def verify_parent_files(root, p, *, security=True):
    blobs, _ = material(p); paths = []
    for key, name in p['continuation']['parent']['custody_names'].items():
        require(type(name) is str and name not in {'.', '..', ''} and Path(name).name == name
            and '/' not in name and '\\' not in name, 'FILTERED_PARENT_CUSTODY')
        path = Path(root)/name
        require(path.is_file() and not path.is_symlink(), 'FILTERED_PARENT_CUSTODY')
        with io.FileIO(path, 'r') as stream: raw = stream.read(filtered.MAX_BUNDLE+1)
        require(raw == blobs[key], 'FILTERED_PARENT_CUSTODY'); paths.append(path)
    if security:require(filtered.private_files(paths), 'NCAAF_STAGED_SECURITY')
    return True


def claim_name(p):
    return p['continuation']['parent']['pins']['bundle']+'.history-continuation-claim-v6.json'


def history_objects(b, *, at):
    old, facts = parent(b['plan']['payload'], at=at)
    require(b['prior_active_seconds'] == facts['charged_active_seconds']
        and b['prior_original_bytes'] == facts['original_bytes'], 'NCAAF_CONTINUATION_ACCOUNTING')
    return deepcopy(old['objects'])+deepcopy(b['objects'])


def validate_response(obj, p, *, at, objects):
    old, _ = parent(p, at=at)
    require(len(objects) == 1 and objects == [obj] and obj['request_id'] == old['plan']['payload']['requests'][2]['id'], 'FILTERED_SCOPE')
    fresh.validate_response(obj, old['plan']['payload'], at=at, objects=old['objects']+objects)


def project_bundle(b, *, at):
    require(len(b['objects']) == 1, 'FILTERED_YARDAGE_COVERAGE')
    old, facts = parent(b['plan']['payload'], at=at)
    require(b['prior_active_seconds'] == facts['charged_active_seconds']
        and b['prior_original_bytes'] == facts['original_bytes'], 'NCAAF_CONTINUATION_ACCOUNTING')
    # The projection uses the unchanged original reader; its inputs are the
    # three authentic objects with their own original metadata and clocks.
    return fresh.project_bundle(dict(plan=old['plan'], objects=deepcopy(old['objects'])+deepcopy(b['objects'])), at=at)


def history_ceiling(p):
    _, facts = parent(p, at=p['execution_window']['not_before'])
    return float(remaining(facts['charged_active_seconds'], prior_bytes=facts['original_bytes']))
