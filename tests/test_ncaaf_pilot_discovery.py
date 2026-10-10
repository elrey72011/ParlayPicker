"""Labelled SYNTHETIC bytes/clocks only; external transports blocked by conftest."""
import base64
from copy import deepcopy
import gzip
import hashlib
import json
from unittest.mock import Mock

import pytest
from app_core import ncaaf_pilot as capture, ncaaf_pilot_discovery as discovery
from app_core import ncaaf_model_compatibility as model, ncaaf_response_custody as custody
from test_ncaaf_pilot import Clock, Stream, fixture as capture_fixture
from test_ncaaf_model_compatibility import synthetic


def fixture(monkeypatch, root):
    target = dict(canonical_event_id='cfbd:9', home_team='SYNTHETIC Home', home_id=1,
        away_team='SYNTHETIC Away', away_id=2, start_utc='2026-10-10T16:00:00Z', neutral_site=False,
        schedule_reference='SYNTHETIC independently retained schedule receipt')
    req = dict(id='SYNTHETIC-one-discovery', provider='odds_api', host='api.the-odds-api.com',
        endpoint=custody.ODDS_ENDPOINT, params=dict(regions='us', markets='spreads', bookmakers='draftkings',
        oddsFormat='american', dateFormat='iso', commenceTimeFrom='2026-10-10T00:00:00Z', commenceTimeTo='2026-10-11T00:00:00Z'), max_attempts=1)
    permission = dict(reference='SYNTHETIC-permission', provider='odds_api', reviewer='SYNTHETIC-advance-reviewer',
        license_holder='SYNTHETIC-license-holder', evidence_reference='SYNTHETIC-private-retention-permission',
        reviewed_at='2026-10-01T00:00:00Z', effective_from='2026-10-01T00:00:00Z', effective_until='2026-11-01T00:00:00Z',
        permitted_use='private_identifier_discovery_retention', requests_sha256=model.digest([req]), target_sha256=model.digest(target))
    p = dict(version=discovery.VERSION, evidence_label='SYNTHETIC', target=target,
        offer_scope=dict(market='spreads', side='home', bookmaker='draftkings'), requests=[req],
        limits=dict(max_attempts=1, max_objects=16, max_body_bytes=512*1024, max_aggregate_bytes=2*1024*1024,
            connect_seconds=3, read_seconds=5, total_seconds=60),
        execution_window=dict(not_before='2026-10-09T11:59:58Z', not_after='2026-10-09T12:00:10Z', authorization_expires_at='2026-10-09T12:00:11Z'),
        advance_permission=permission, collection_authorization_ref='SYNTHETIC-discovery-auth', custody_id='SYNTHETIC-private-root', unresolved=[])
    plan = discovery.seal(p); auth = trust(plan, monkeypatch)
    monkeypatch.setattr(discovery, 'AUTHORIZED_CUSTODY_ROOTS', {p['custody_id']: str(root.resolve())})
    return plan, auth


def trust(plan, monkeypatch):
    p = plan['payload']
    auth = discovery.seal(dict(version=discovery.AUTH_VERSION, reference=p['collection_authorization_ref'],
        plan_sha256=plan['sha256'], custody_id=p['custody_id'], authorized_at='2026-10-09T11:00:00Z', expires_at='2026-10-09T12:00:11Z',
        owner='SYNTHETIC-owner', max_attempts=1, max_provider_credits=1, max_cost_usd=0, quota_reference='SYNTHETIC-prepaid-account'))
    monkeypatch.setattr(discovery, 'AUTHORIZED_DISCOVERIES', {auth['payload']['reference']: auth['sha256']})
    monkeypatch.setattr(discovery, 'ACCEPTED_DISCOVERY_PERMISSIONS', {p['advance_permission']['reference']: model.digest(p['advance_permission'])})
    return auth


def event(eid='SYNTHETIC-provider-id', **changes):
    return dict(id=eid, sport_key='americanfootball_ncaaf', home_team='SYNTHETIC Home', away_team='SYNTHETIC Away',
        commence_time='2026-10-10T16:00:00Z', bookmakers=[], private_canary='SYNTHETIC_ORIGINAL_BODY', **changes)


def body(events):
    return json.dumps(events, indent=2).encode() + b'\n'


def guards(monkeypatch):
    from app_core import ncaaf_research, ncaaf_pipeline_evidence, research_replay, prediction_evidence
    from test_ncaaf_pilot import forbidden
    spies = forbidden(monkeypatch)
    for obj, name in ((ncaaf_research, 'centers'), (ncaaf_research, 'distribution'),
                      (capture, 'admitted_analysis'), (capture, 'retain_analysis'),
                      (prediction_evidence, 'connect'), (research_replay, 'retain_export')):
        if hasattr(obj, name):
            spy = Mock(side_effect=AssertionError('Discovery cannot infer, admit or initialize evidence'))
            monkeypatch.setattr(obj, name, spy); spies.append(spy)
    return spies


def test_original_capture_still_rejects_unknown_provider_id(synthetic, monkeypatch, tmp_path):
    plan, *_ = capture_fixture(synthetic, monkeypatch, root=tmp_path)
    plan['payload']['target']['provider_event_id'] = None
    plan = capture.seal(plan['payload'])
    with pytest.raises(ValueError, match='NCAAF_PILOT_TARGET'):
        capture.validate_plan(plan, executable=True)


def test_planning_and_dispatch_are_explicit_and_separate(monkeypatch, tmp_path):
    plan, _ = fixture(monkeypatch, tmp_path); spies = guards(monkeypatch)
    before = list(tmp_path.iterdir())
    r = discovery.planning(plan)
    assert r['network_requests'] == r['credential_loads'] == r['storage_operations'] == 0
    assert list(tmp_path.iterdir()) == before and all(not s.called for s in spies)
    assert 'provider_event_id' not in plan['payload']['target']
    assert 'quote_sha256' not in model.encode(plan).decode()


@pytest.mark.parametrize('change,code', [
    ('no_auth', 'AUTHORIZATION_MISSING'), ('untrusted', 'AUTHORIZATION_UNTRUSTED'),
    ('wrong_auth', 'AUTHORIZATION_CONFLICT'), ('expiry', 'AUTHORIZATION_EXPIRED'),
    ('owner', 'AUTHORIZATION_CONFLICT'), ('window', 'WINDOW_EXPIRED'),
    ('permission', 'PERMISSION_UNTRUSTED'), ('permission_expiry', 'PERMISSION_CLOCK'),
    ('permission_future', 'PERMISSION_CLOCK'), ('permission_scope', 'PERMISSION_SCOPE'),
    ('permission_target', 'PERMISSION_SCOPE'), ('host', 'REQUEST'), ('endpoint', 'REQUEST'),
    ('params', 'NCAAF_CUSTODY_SCOPE_CONFLICT'), ('extra_request', 'BUDGET'), ('extra_target', 'TARGET'),
    ('plan_integrity', 'PLAN_INTEGRITY'), ('unresolved', 'PREREQUISITES'), ('custody', 'CUSTODY_CONFLICT'),
    ('limit', 'LIMITS'), ('unknown_clock', 'CLOCK'), ('proposal', 'PREREQUISITES')])
def test_stop_before_request(monkeypatch, tmp_path, change, code):
    plan, auth = fixture(monkeypatch, tmp_path); clock = Clock(); transport = Mock()
    p = plan['payload']; a = auth['payload']
    if change == 'no_auth': auth = None
    elif change == 'untrusted': monkeypatch.setattr(discovery, 'AUTHORIZED_DISCOVERIES', {})
    elif change == 'permission': monkeypatch.setattr(discovery, 'ACCEPTED_DISCOVERY_PERMISSIONS', {})
    elif change == 'window': clock.value = '2026-10-09T12:01:00Z'
    elif change == 'custody': monkeypatch.setattr(discovery, 'AUTHORIZED_CUSTODY_ROOTS', {})
    elif change in {'wrong_auth', 'expiry', 'owner'}:
        if change == 'wrong_auth': a['plan_sha256'] = 'a'*64
        elif change == 'expiry': a['expires_at'] = '2026-10-09T11:01:00Z'
        else: a['owner'] = ''
        auth = discovery.seal(a); monkeypatch.setattr(discovery, 'AUTHORIZED_DISCOVERIES', {a['reference']: auth['sha256']})
    else:
        if change == 'permission_expiry': p['advance_permission']['effective_until'] = '2026-10-09T11:00:00Z'
        elif change == 'permission_future': p['advance_permission']['reviewed_at'] = '2026-10-09T12:00:00Z'
        elif change == 'permission_scope': p['advance_permission']['requests_sha256'] = 'a'*64
        elif change == 'permission_target': p['advance_permission']['target_sha256'] = 'a'*64
        elif change == 'host': p['requests'][0]['host'] = 'evil.example'
        elif change == 'endpoint': p['requests'][0]['endpoint'] = 'v4/sports/americanfootball_ncaaf/events'
        elif change == 'params': p['requests'][0]['params']['apiKey'] = 'SYNTHETIC_KEY'
        elif change == 'extra_request': p['requests'].append(deepcopy(p['requests'][0]))
        elif change == 'extra_target': p['target']['provider_event_id'] = 'borrowed'
        elif change == 'unresolved': p['unresolved'] = ['UNKNOWN_RIGHTS']
        elif change == 'limit': p['limits']['max_attempts'] = 2
        elif change == 'unknown_clock': p['target']['start_utc'] = '2026-10-10T16:00:00'
        elif change == 'proposal': p['evidence_label'] = 'PROPOSAL'
        plan = discovery.seal(p); auth = trust(plan, monkeypatch)
        if change == 'plan_integrity': plan['sha256'] = 'a'*64
    if change == 'params': code = 'NCAAF_CUSTODY_CREDENTIAL_FIELD'
    with pytest.raises(ValueError, match=code):
        discovery.acquire(plan, auth, root=tmp_path, transport=transport, wall=clock.wall, monotonic=clock.monotonic)
    assert transport.call_count == 0 and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('case,state', [('one', 'CANDIDATE_UNVERIFIED'), ('empty', 'NO_EXACT_DESCRIPTOR_MATCH'),
    ('alias', 'NO_EXACT_DESCRIPTOR_MATCH'), ('two', 'AMBIGUOUS'), ('duplicate', 'AMBIGUOUS'),
    ('reversed', 'CONFLICTING'), ('kickoff', 'CONFLICTING'), ('id_conflict', 'CONFLICTING')])
def test_unknown_identifiers_retained_not_admitted(monkeypatch, tmp_path, case, state):
    plan, auth = fixture(monkeypatch, tmp_path); spies = guards(monkeypatch)
    rows = [event()]
    if case == 'empty': rows = []
    elif case == 'alias': rows[0]['home_team'] = 'SYNTHETIC Home Mascot'
    elif case == 'two': rows.append(event('SYNTHETIC-second-id'))
    elif case == 'duplicate': rows.append(deepcopy(rows[0]))
    elif case == 'reversed': rows[0]['home_team'], rows[0]['away_team'] = rows[0]['away_team'], rows[0]['home_team']
    elif case == 'kickoff': rows[0]['commence_time'] = '2026-10-10T17:00:00Z'
    elif case == 'id_conflict': rows.append(deepcopy(rows[0])); rows[1]['commence_time'] = '2026-10-10T17:00:00Z'
    raw = body(rows); clock = Clock(); calls = []
    def fake(request, *args):
        calls.append(deepcopy(request)); clock.value = '2026-10-09T12:00:01Z'; return Stream(raw)
    b = discovery.acquire(plan, auth, root=tmp_path, transport=fake, wall=clock.wall, monotonic=clock.monotonic)
    assert calls == plan['payload']['requests'] and b['payload']['attempted_requests'] == 1
    assert b['payload']['status'] == 'DISCOVERED_UNACCEPTED' and b['payload']['identifiers']['state'] == state
    assert base64.b64decode(b['payload']['objects'][0]['body_b64']) == raw
    assert b['payload']['objects'][0]['metadata']['received_at'] == '2026-10-09T12:00:01Z'
    assert b['payload']['objects'][0]['requested_at'] == '2026-10-09T11:59:59Z'
    assert not any(b['payload'][k] for k in ('accepted', 'inference', 'scientific_acceptance', 'wagering_authority'))
    path = tmp_path / (plan['sha256'] + '.discovery-unaccepted.json')
    assert discovery.read_bundle(path) == b and all(not s.called for s in spies)
    with pytest.raises(ValueError, match='ALREADY_ATTEMPTED'):
        discovery.acquire(plan, auth, root=tmp_path, transport=fake, wall=clock.wall, monotonic=clock.monotonic)
    assert len(calls) == 1
    assert 'probability' not in b['payload']['identifiers'] and 'stake' not in b['payload']['identifiers']


@pytest.mark.parametrize('failure,code', [('redirect', 'HTTP_STATUS'), ('pagination', 'REDIRECT_OR_PAGINATION'),
    ('oversize', 'BODY_LIMIT'), ('compressed', 'BODY_LIMIT'), ('incomplete', 'INCOMPLETE_BODY'),
    ('timeout', 'TRANSPORT_FAILURE'), ('slow', 'DEADLINE'), ('credential', 'CREDENTIAL_FIELD'),
    ('schema', 'BODY_SCHEMA'), ('rate', 'AUTH_OR_RATE_LIMIT'), ('auth', 'AUTH_OR_RATE_LIMIT')])
def test_one_spent_attempt_failures_no_retry(monkeypatch, tmp_path, failure, code):
    plan, auth = fixture(monkeypatch, tmp_path); clock = Clock(); calls = []
    def fake(request, *args):
        calls.append(request); s = Stream(body([event()]))
        if failure == 'redirect': s.status = 302
        elif failure == 'pagination': s.headers['x-next-page'] = 'next'
        elif failure == 'oversize': s = Stream(b' '*(custody.MAX_BODY_BYTES+1))
        elif failure == 'compressed': s = Stream(gzip.compress(b' '*(custody.MAX_BODY_BYTES+1))); s.headers['content-encoding'] = 'gzip'
        elif failure == 'incomplete': s.headers['content-length'] = str(len(s.raw)+1)
        elif failure == 'timeout': raise TimeoutError('SYNTHETIC_SECRET_DO_NOT_LOG')
        elif failure == 'slow': clock.elapsed = 61
        elif failure == 'credential': s = Stream(b'[{"apiKey":"SYNTHETIC_SECRET_DO_NOT_LOG"}]')
        elif failure == 'schema': s = Stream(b'[{}]')
        elif failure == 'rate': s.status = 429
        elif failure == 'auth': s.status = 403
        return s
    b = discovery.acquire(plan, auth, root=tmp_path, transport=fake, wall=clock.wall, monotonic=clock.monotonic)
    assert b['payload']['status'] == 'INCOMPLETE' and b['payload']['reason'].endswith(code)
    assert len(calls) == b['payload']['attempted_requests'] == 1 and b['payload']['objects'] == []
    assert 'SYNTHETIC_SECRET' not in model.encode(b).decode()
    with pytest.raises(ValueError, match='ALREADY_ATTEMPTED'):
        discovery.acquire(plan, auth, root=tmp_path, transport=fake, wall=clock.wall, monotonic=clock.monotonic)
    assert len(calls) == 1


def test_interruption_is_not_resumable(monkeypatch, tmp_path):
    plan, auth = fixture(monkeypatch, tmp_path); calls = Mock(side_effect=KeyboardInterrupt()); clock = Clock()
    with pytest.raises(KeyboardInterrupt): discovery.acquire(plan, auth, root=tmp_path, transport=calls, wall=clock.wall, monotonic=clock.monotonic)
    journal = (tmp_path / (plan['sha256'] + '.discovery-attempts.jsonl')).read_text()
    assert '"attempt":1' in journal and calls.call_count == 1
    with pytest.raises(ValueError, match='ALREADY_ATTEMPTED'): discovery.acquire(plan, auth, root=tmp_path, transport=calls, wall=clock.wall, monotonic=clock.monotonic)
    assert calls.call_count == 1


@pytest.mark.parametrize('encoded', ['literal', 'escaped', 'url'])
def test_existing_https_credential_echo_guard_reused(monkeypatch, tmp_path, encoded):
    plan, auth = fixture(monkeypatch, tmp_path); clock = Clock()
    secret = 'SYNTHETIC_KEY_CANARY'
    class FakeHttps(capture.HttpsTransport):
        def __call__(self, *args):
            self._secret = secret
            row = event(); row['description'] = secret
            raw = body([row])
            if encoded == 'escaped': raw = raw.replace(secret.encode(), b'\\u0053YNTHETIC_KEY_CANARY')
            elif encoded == 'url': raw = raw.replace(secret.encode(), b'%53YNTHETIC_KEY_CANARY')
            return Stream(raw)
    b = discovery.acquire(plan, auth, root=tmp_path, transport=FakeHttps(lambda *a: pytest.fail('No real credentials')),
        wall=clock.wall, monotonic=clock.monotonic)
    assert b['payload']['status'] == 'INCOMPLETE' and b['payload']['reason'] == 'NCAAF_PILOT_CREDENTIAL_IN_BODY'
    assert b['payload']['objects'] == [] and secret not in model.encode(b).decode()


def test_slow_stream_and_changed_root_cannot_repeat(monkeypatch, tmp_path):
    plan, auth = fixture(monkeypatch, tmp_path); clock = Clock(); calls = []
    class Slow(Stream):
        def chunks(self):
            clock.elapsed = 61
            yield self.raw
    def fake(request, *args): calls.append(request); return Slow(body([event()]))
    b = discovery.acquire(plan, auth, root=tmp_path, transport=fake, wall=clock.wall, monotonic=clock.monotonic)
    assert b['payload']['reason'] == 'NCAAF_PILOT_DEADLINE' and b['payload']['objects'] == []
    other = tmp_path/'other'; other.mkdir()
    with pytest.raises(ValueError, match='CUSTODY_CONFLICT'): discovery.acquire(plan, auth, root=other, transport=fake, wall=clock.wall, monotonic=clock.monotonic)
    assert len(calls) == 1 and list(other.iterdir()) == []


def test_exception_prefix_cannot_smuggle_credential_into_diagnostic(monkeypatch, tmp_path):
    plan, auth = fixture(monkeypatch, tmp_path); clock = Clock()
    secret = 'SYNTHETIC_SECRET_CALLBACK_CANARY'
    def fake(*args): raise ValueError('NCAAF_PILOT_' + secret)
    b = discovery.acquire(plan, auth, root=tmp_path, transport=fake, wall=clock.wall, monotonic=clock.monotonic)
    assert secret not in model.encode(b).decode()
    assert b['payload']['reason'] == 'NCAAF_DISCOVERY_STOPPED'


def test_discovery_cannot_borrow_capture_or_acceptance(synthetic, monkeypatch, tmp_path):
    plan, auth = fixture(monkeypatch, tmp_path); clock = Clock()
    b = discovery.acquire(plan, auth, root=tmp_path, transport=lambda *a: Stream(body([event()])), wall=clock.wall, monotonic=clock.monotonic)
    with pytest.raises(ValueError, match='NCAAF_PILOT_UNACCEPTED_BUNDLE'):
        capture.project_capture(b, native_batches={}, quote={}, terms={})
    _, _, packet, row, _, _, _, _ = capture_fixture(synthetic, monkeypatch, root=tmp_path)
    from app_core import ncaaf_research as research
    numeric = Mock(side_effect=AssertionError('Must reject before numerical inference'))
    monkeypatch.setattr(research, 'centers', numeric)
    result = capture.admitted_analysis(packet, row, inventory=None, bundle=b)
    assert result.iloc[0].ml_inference_status == 'unavailable' and not numeric.called
    assert result.iloc[0].ml_unavailable_reason == 'NCAAF_PACKET_INTEGRITY'
    rejected = json.loads(result.iloc[0].ml_estimate_metadata)['ncaaf_inputs']['payload']
    assert rejected['pilot_capture']['payload']['reason'] == 'NCAAF_PILOT_UNACCEPTED_BUNDLE'
    assert rejected['live_stake'] == 0 and not rejected['wagering_authority']


def test_corrupt_and_rehashed_clock_readback_reject(monkeypatch, tmp_path):
    plan, auth = fixture(monkeypatch, tmp_path); clock = Clock()
    b = discovery.acquire(plan, auth, root=tmp_path, transport=lambda *a: Stream(body([event()])), wall=clock.wall, monotonic=clock.monotonic)
    path = tmp_path / 'SYNTHETIC-corrupt.json'
    b['payload']['objects'][0]['requested_at'] = '2026-10-09T12:00:01Z'
    path.write_bytes(model.encode(b))
    with pytest.raises(ValueError, match='BUNDLE_INTEGRITY'): discovery.read_bundle(path)
    path.write_bytes(model.encode(discovery.seal(b['payload'])))
    with pytest.raises(ValueError, match='BUNDLE_INTEGRITY'): discovery.read_bundle(path)


def test_production_trust_catalogs_remain_empty():
    assert discovery.AUTHORIZED_DISCOVERIES == discovery.ACCEPTED_DISCOVERY_PERMISSIONS == discovery.AUTHORIZED_CUSTODY_ROOTS == {}
