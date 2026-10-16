"""Labelled SYNTHETIC staged bodies; external transports remain blocked."""
import base64
from copy import deepcopy
import hashlib
import json
import pytest
from app_core import ncaaf_staged_acquisition as stage, ncaaf_staged_evidence as evidence
from app_core import ncaaf_model_compatibility as model, ncaaf_research as math
from scripts.benchmark_drive_history_loading import blocked_network
from test_ncaaf_model_compatibility import synthetic
from test_ncaaf_pilot import Stream

TARGET = dict(provider_event_id='SYNTHETIC-MEM-TUL', home_team='Tulane Green Wave', away_team='Memphis Tigers',
              start_utc='2026-10-16T23:30:00Z', neutral_site=False)


@pytest.fixture(autouse=True)
def offline(request):
    if request.node.name.startswith('test_real_socket_'):
        # Existing local_http creates Windows loopback pairs first, then blocks
        # external application transports for the complete acquisition/CLI path.
        yield
        return
    with blocked_network(): yield


class Clock:
    value = '2026-10-16T18:00:01Z'
    elapsed = 0.
    def wall(self): return self.value
    def mono(self): return self.elapsed


def bodies():
    target = dict(id=401862801, season=2026, week=7, seasonType='regular', startDate=TARGET['start_utc'], startTimeTBD=False,
                  completed=False, neutralSite=False, homeId=2655, homeTeam='Tulane', homePoints=None, awayId=235, awayTeam='Memphis', awayPoints=None)
    games = [target]; stats = {w: [] for w in (1, 2, 3)}
    for week in (1, 2, 3):
        for tid, name in ((2655, 'Tulane'), (235, 'Memphis')):
            g = dict(target, id=tid*10+week, week=week, startDate=f'2026-09-{week*7:02}T17:00:00Z', completed=True,
                     homeId=tid, homeTeam=name, homePoints=17, awayId=9000+tid, awayTeam='SYNTHETIC opponent '+name, awayPoints=14)
            games.append(g)
            stats[week].append(dict(id=g['id'], season=2026, week=week, seasonType='regular', teams=[
                dict(teamId=g['homeId'], points=17, stats=[dict(category='totalYards', stat='400')]),
                dict(teamId=g['awayId'], points=14, stats=[dict(category='totalYards', stat='300')])]))
    return [model.encode(games)]+[model.encode(stats[w]) for w in (1, 2, 3)]


def odds(line=-3.5):
    return model.encode([dict(id=TARGET['provider_event_id'], sport_key='americanfootball_ncaaf', home_team=TARGET['home_team'],
        away_team=TARGET['away_team'], commence_time=TARGET['start_utc'], bookmakers=[dict(key='draftkings', last_update='2026-10-16T18:30:00Z',
        markets=[dict(key='spreads', last_update='2026-10-16T18:30:00Z', outcomes=[dict(name=TARGET['home_team'], point=line, price=-110),
        dict(name=TARGET['away_team'], point=-line, price=-110)])])])])


def plan(kind=stage.HISTORY_PLAN, parent=None, prequote=None):
    quote = kind == stage.QUOTE_PLAN; req = stage.requests(TARGET, kind)
    return stage.seal(dict(version=kind, evidence_label='SYNTHETIC', target=deepcopy(TARGET), requests=req, limits=deepcopy(stage.LIMITS),
        execution_window=dict(not_before='2026-10-16T18:30:00Z' if quote else '2026-10-16T18:00:00Z',
            not_after='2026-10-16T18:45:00Z' if quote else '2026-10-16T18:15:00Z', authorization_expires_at='2026-10-16T18:45:30Z' if quote else '2026-10-16T18:15:30Z'),
        advance_permissions=[dict(reference='SYNTHETIC-permission-'+kind, provider='odds_api' if quote else 'cfbd', reviewed_at='2026-10-16T17:00:00Z',
            effective_from='2026-10-16T17:00:00Z', effective_until='2026-10-16T23:00:00Z', permitted_use='private_prospective_capture', requests_sha256=model.digest(req))],
        collection_authorization_ref='SYNTHETIC-authorization-'+kind, custody_id='SYNTHETIC-private', parent_history_sha256=parent['sha256'] if parent else None,
        prequote_sha256=prequote['sha256'] if prequote else None, unresolved=[]))


def authorize(p, monkeypatch, root):
    r = p['payload'];day=r['execution_window']['not_before'][:10]
    a = stage.seal(dict(version=stage.AUTH_VERSION, reference=r['collection_authorization_ref'], plan_sha256=p['sha256'],
        custody_id=r['custody_id'], authorized_at=day+('T18:20:00Z' if r['version']==stage.QUOTE_PLAN else 'T17:30:00Z'), expires_at=r['execution_window']['authorization_expires_at'], owner='Robert Velarde'))
    monkeypatch.setattr(stage, 'AUTHORIZED_COLLECTIONS', {a['payload']['reference']:a['sha256']})
    monkeypatch.setattr(stage, 'ACCEPTED_ADVANCE_PERMISSIONS', {v['reference']:model.digest(v) for v in r['advance_permissions']})
    monkeypatch.setattr(stage, 'AUTHORIZED_CUSTODY_ROOTS', {r['custody_id']:str(root.resolve())})
    monkeypatch.setattr(stage, 'AUTHORIZED_ACCOUNT_EVIDENCE', {a['payload']['reference']:dict(provider=r['requests'][0]['provider'], included_remaining=10,
        observed_at=day+'T17:00:00Z', valid_until=day+'T23:00:00Z', evidence_sha256='a'*64, no_paid_overage=True, reviewer='Robert Velarde')})
    monkeypatch.setattr(stage, 'inspect_security', lambda p:dict(status='VERIFIED'))  # SYNTHETIC root only.
    cfg=configuration(p,a,root,stage.AUTHORIZED_ACCOUNT_EVIDENCE[a['payload']['reference']]);raw=model.encode(cfg)
    monkeypatch.setattr(stage,'AUTHORIZED_CONFIGURATIONS',{p['sha256']:dict(file_sha256=hashlib.sha256(raw).hexdigest(),file_b64=base64.b64encode(raw).decode())})
    return a


def configuration(p,a,root,account):
    cfg=dict(version='ncaaf-staged-controlled-bindings-v1',plan_sha256=p['sha256'],authorization=a,
        permission=p['payload']['advance_permissions'][0],custody=dict(id=p['payload']['custody_id'],root=str(root),storage_receipt_sha256='c'*64),account=deepcopy(account),review={})
    cfg['review']=dict(reviewer='Robert Velarde',reviewed_at=p['payload']['execution_window']['not_before'][:10]+('T18:25:00Z' if p['payload']['version']==stage.QUOTE_PLAN else 'T17:45:00Z'),subject_sha256=model.digest({k:v for k,v in cfg.items() if k!='review'}),finding='SYNTHETIC exact configuration review for isolated fixture only; no actual owner review.')
    return stage.seal(cfg)


def review_prequote(p,h):
    p=deepcopy(p);v=p['payload']
    v['history_review']=dict(version='ncaaf-owner-prequote-review-v1',reviewer='Robert Velarde',reviewed_at=v['prepared_at'][:10]+'T18:03:00Z',
        subject_sha256=evidence.prequote_subject(p),qualifications='SYNTHETIC documented verification basis, no personal technical audit or acceptance.',
        conclusions={k:dict(conclusion='VERIFIED',evidence_sha256=sha,finding='SYNTHETIC owner reviewed the exact prequote '+k+' evidence; no authentic review is recorded.') for k,sha in evidence.prequote_review_subjects(v,h).items()})
    return stage.seal(v)


def capture(monkeypatch, root, p=None, raw=None, clock=None, parent=None, prequote=None):
    p = p or plan(); a = authorize(p, monkeypatch, root); clock = clock or Clock(); sent = []
    raw = bodies() if raw is None else raw
    def transport(req, limits, deadline, mono):
        sent.append(deepcopy(req)); clock.elapsed += .01
        return Stream(raw[len(sent)-1])
    b = stage.acquire(p, a, root=root, transport=transport, history_bundle=parent, prequote=prequote, wall=clock.wall, monotonic=clock.mono)
    return b, sent, clock


def prepared(synthetic, monkeypatch, root):
    from test_ncaaf_owner_terms import fixture
    packet, _ = fixture(synthetic, monkeypatch)
    m = packet['payload']['custody_packet']['payload']['native_packet']['payload']['observation']['payload']['model']
    h, _, _ = capture(monkeypatch, root)
    pref = evidence.prepare(h, model_packet=m, prepared_at='2026-10-16T18:01:00Z')
    return h, review_prequote(pref,h)


def test_network_free_side_neutral_planning(monkeypatch, tmp_path):
    p = plan(); before = list(tmp_path.iterdir()); result = stage.planning(p)
    assert result['network_requests'] == result['credential_loads'] == result['storage_operations'] == 0
    assert len(result['requests']) == 4 and all(r['provider'] == 'cfbd' for r in result['requests'])
    assert not result['accepted'] and not result['inference'] and list(tmp_path.iterdir()) == before
    assert 'offer' not in p['payload'] and 'selection' not in p['payload']


def test_history_originals_journal_readback_and_spent(monkeypatch, tmp_path):
    b, sent, clock = capture(monkeypatch, tmp_path)
    assert b['payload']['status'] == 'CAPTURED_UNREVIEWED' and len(sent) == 4
    assert stage.read_bundle(b)['attempts'] == 4 and stage.verify_journal(tmp_path, b)
    assert [base64.b64decode(o['body_b64']) for o in b['payload']['objects']] == bodies()
    assert (tmp_path/(b['payload']['plan']['sha256']+'.stage.json')).read_bytes() == model.encode(b)
    with pytest.raises(ValueError, match='ALREADY_SPENT'): capture(monkeypatch, tmp_path, clock=clock)


def test_separate_quote_authority_claim_and_shared_accounting(synthetic, monkeypatch, tmp_path):
    h, pref = prepared(synthetic, monkeypatch, tmp_path)
    p = plan(stage.QUOTE_PLAN, h, pref); clock = Clock(); clock.value = '2026-10-16T18:30:01Z'
    q, sent, _ = capture(monkeypatch, tmp_path, p, [odds()], clock, h, pref)
    joined = evidence.join(h, pref, q)
    assert len(sent) == 1 and q['payload']['prior_original_bytes'] == h['payload']['original_bytes']
    assert q['payload']['prior_active_seconds'] == h['payload']['active_seconds'] and q['payload']['status'] == 'CAPTURED_UNREVIEWED'
    assert joined['payload']['prequote'] == pref and not q['payload']['accepted'] and not q['payload']['inference']
    p['payload']['collection_authorization_ref'] += '-second'; p = stage.seal(p['payload'])
    q2, sent, _ = capture(monkeypatch, tmp_path, p, [odds()], clock, h, pref)
    assert q2['payload']['status'] == 'INCOMPLETE' and q2['payload']['reason'] == 'NCAAF_STAGED_HISTORY_ALREADY_CLAIMED' and sent == []


@pytest.mark.parametrize('change', ['host', 'endpoint', 'params', 'extra', 'limit', 'side', 'permission', 'untrusted', 'expired', 'account'])
def test_rejection_before_dispatch(monkeypatch, tmp_path, change):
    p = plan(); a = authorize(p, monkeypatch, tmp_path); clock = Clock()
    if change == 'host': p['payload']['requests'][0]['host'] = 'example.org'
    elif change == 'endpoint': p['payload']['requests'][0]['endpoint'] = 'teams'
    elif change == 'params': p['payload']['requests'][1]['params']['week'] = 4
    elif change == 'extra': p['payload']['requests'][0]['params']['token'] = 'SYNTHETIC'
    elif change == 'limit': p['payload']['limits']['shared_active_seconds'] = 61
    elif change == 'side': p['payload']['selected_team'] = 'Memphis'
    elif change == 'permission': monkeypatch.setattr(stage, 'ACCEPTED_ADVANCE_PERMISSIONS', {})
    elif change == 'untrusted': monkeypatch.setattr(stage, 'AUTHORIZED_COLLECTIONS', {})
    elif change == 'expired': clock.value = '2026-10-16T18:16:00Z'
    else: monkeypatch.setattr(stage, 'AUTHORIZED_ACCOUNT_EVIDENCE', {})
    p = stage.seal(p['payload'])
    with pytest.raises(ValueError): stage.acquire(p, a, root=tmp_path, transport=lambda *a:pytest.fail('INVALID DISPATCH'), wall=clock.wall, monotonic=clock.mono)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('attack', ['oversize', 'truncated', 'http', 'redirect', 'secret', 'read', 'deadline'])
def test_failure_is_spent_sanitized_and_never_retries(monkeypatch, tmp_path, attack):
    p = plan(); a = authorize(p, monkeypatch, tmp_path); clock = Clock(); sent = []
    secret = 'SYNTHETIC_NOT_A_REAL_CREDENTIAL_112233'
    def transport(req, limits, deadline, mono):
        sent.append(req)
        if attack == 'read': raise OSError(secret)
        if attack == 'deadline': clock.elapsed = 61
        if attack == 'oversize': return Stream(b'['+b' '*stage.LIMITS['max_body_bytes']+b']')
        if attack == 'truncated': return Stream(b'[]', headers={'content-type':'application/json', 'content-length':'20'})
        if attack == 'http': return Stream(b'[]', status=500)
        if attack == 'redirect': return Stream(b'[]', headers={'content-type':'application/json', 'location':'https://example.org'})
        if attack == 'secret': return Stream(model.encode([dict(apiKey=secret)]))
        return Stream(b'[]')
    b = stage.acquire(p, a, root=tmp_path, transport=transport, wall=clock.wall, monotonic=clock.mono)
    assert b['payload']['status'] == 'INCOMPLETE' and b['payload']['attempts'] == len(sent) == 1
    assert secret not in model.encode(b).decode() and not b['payload']['objects']
    stage.read_bundle(b, complete=False); stage.verify_journal(tmp_path, b)
    with pytest.raises(ValueError, match='ALREADY_SPENT'): stage.acquire(p, a, root=tmp_path, transport=transport, wall=lambda:'2026-10-16T18:00:01Z', monotonic=clock.mono)


def test_cli_plan_actual_path_and_runtime_blocker(monkeypatch, tmp_path, capsys):
    from scripts import ncaaf_staged_acquisition as cli
    path = tmp_path/'SYNTHETIC-plan.json'; path.write_bytes(model.encode(plan()))
    assert cli.main(['plan', str(path)]) == 0 and json.loads(capsys.readouterr().out)['network_requests'] == 0
    with pytest.raises(ValueError, match='RUNTIME_UNSUPPORTED'): cli.verify_runtime()


@pytest.mark.parametrize('attack', ['body', 'metadata', 'journal', 'attempts', 'elapsed', 'parent', 'hash'])
def test_durable_readback_tampering(monkeypatch, tmp_path, attack):
    b, _, _ = capture(monkeypatch, tmp_path); p = b['payload']
    if attack == 'body': p['objects'][0]['body_b64'] = 'W10='
    elif attack == 'metadata': p['objects'][0]['metadata']['received_at'] = '2026-10-16T18:20:00Z'
    elif attack == 'journal': p['journal_b64'] = base64.b64encode(b'{}\n').decode()
    elif attack == 'attempts': p['attempts'] = 3
    elif attack == 'elapsed': p['active_seconds'] = 0
    elif attack == 'parent': p['prior_active_seconds'] = 1
    else: p['objects'][0]['body_sha256'] = 'a'*64
    b = stage.seal(p)
    with pytest.raises((ValueError, KeyError, IndexError)): stage.read_bundle(b)


@pytest.mark.parametrize('closing,framing', [(False,'length'), (True,'length'), (True,'chunked')])
def test_real_socket_cli_transport_journal_and_readback(monkeypatch,tmp_path,capsys,closing,framing):
    from scripts import ncaaf_staged_acquisition as cli
    from test_ncaaf_socket_lifecycle import local_http
    p=plan();a=authorize(p,monkeypatch,tmp_path);clock=Clock()
    config=configuration(p,a,tmp_path,stage.AUTHORIZED_ACCOUNT_EVIDENCE[a['payload']['reference']]);raw=model.encode(config)
    for name,value in [('plan.json',p),('auth.json',a),('config.json',config)]: (tmp_path/name).write_bytes(model.encode(value))
    monkeypatch.setattr(cli,'verify_runtime',lambda:None) # Synthetic instrumentation; production gate separately rejects pytest.
    monkeypatch.setattr(cli.sys.stdin,'isatty',lambda:True)
    secret='SYNTHETIC_MEMORY_ONLY_PRIVATE_INPUT'
    monkeypatch.setattr(cli.getpass,'getpass',lambda prompt:secret)
    monkeypatch.setattr(stage.transport,'utc',clock.wall)
    with local_http(monkeypatch,bodies(),closing=closing,framing=framing) as wire:
        assert cli.main(['execute',str(tmp_path/'plan.json'),'--authorization',str(tmp_path/'auth.json'),'--controlled-bindings',str(tmp_path/'config.json'),
            '--bindings-sha256',hashlib.sha256(raw).hexdigest(),'--custody-root',str(tmp_path)])==0
        report=json.loads(capsys.readouterr().out);assert report['status']=='CAPTURED_UNREVIEWED' and report['attempts']==4
        b=json.loads((tmp_path/(p['sha256']+'.stage.json')).read_bytes());stage.read_bundle(b);stage.verify_journal(tmp_path,b)
        assert len(wire['connected'])==len(wire['sent'])==4
        assert secret not in model.encode(b).decode() and stage.CONTROLLED.get() is None


@pytest.mark.parametrize('budget',['active','originals'])
def test_final_stage_cannot_reset_shared_budget(synthetic,monkeypatch,tmp_path,budget):
    from test_ncaaf_owner_terms import fixture as old
    packet,_=old(synthetic,monkeypatch);m=packet['payload']['custody_packet']['payload']['native_packet']['payload']['observation']['payload']['model']
    p=plan();a=authorize(p,monkeypatch,tmp_path);clock=Clock();raw=bodies();calls=[]
    if budget=='originals':raw=[r+b' '*(512*1024-len(r)-80) for r in raw]
    def transport(req,*args):
        calls.append(req);clock.elapsed+=14 if budget=='active' else .01
        return Stream(raw[len(calls)-1])
    h=stage.acquire(p,a,root=tmp_path,transport=transport,wall=clock.wall,monotonic=clock.mono)
    assert h['payload']['status']=='CAPTURED_UNREVIEWED'
    pref=review_prequote(evidence.prepare(h,model_packet=m,prepared_at='2026-10-16T18:01:00Z'),h);qp=plan(stage.QUOTE_PLAN,h,pref);qa=authorize(qp,monkeypatch,tmp_path)
    clock=Clock();clock.value='2026-10-16T18:30:01Z';sent=[]
    def quote(req,*args):
        sent.append(req);clock.elapsed+=5 if budget=='active' else .01
        return Stream(odds())
    q=stage.acquire(qp,qa,root=tmp_path,transport=quote,history_bundle=h,prequote=pref,wall=clock.wall,monotonic=clock.mono)
    assert q['payload']['status']=='INCOMPLETE' and q['payload']['attempts']==len(sent)==1
    assert q['payload']['prior_active_seconds']==h['payload']['active_seconds'] and q['payload']['prior_original_bytes']==h['payload']['original_bytes']
    assert q['payload']['objects']==[] and stage.verify_journal(tmp_path,q)
    with pytest.raises(ValueError):evidence.join(h,pref,q)


def test_rehashed_reordered_journal_is_not_a_valid_receipt(monkeypatch,tmp_path):
    b,_,_=capture(monkeypatch,tmp_path);p=b['payload'];events=[json.loads(x) for x in base64.b64decode(p['journal_b64']).splitlines()]
    events[1],events[2]=events[2],events[1];raw=b''.join(model.encode(e)+b'\n' for e in events)
    p.update(journal_b64=base64.b64encode(raw).decode(),journal_sha256=hashlib.sha256(raw).hexdigest());b=stage.seal(p)
    with pytest.raises(ValueError,match='JOURNAL'):stage.read_bundle(b)


@pytest.mark.parametrize('change',['missing','subject','finding','facet','late','qualifications'])
def test_substantive_prequote_review_required_before_any_quote_attempt(synthetic,monkeypatch,tmp_path,change):
    h,p=prepared(synthetic,monkeypatch,tmp_path);r=p['payload']['history_review']
    if change=='missing':p['payload']['history_review']=None
    elif change=='subject':r['subject_sha256']='f'*64
    elif change=='finding':r['conclusions']['confirmed_event']['finding']='yes'
    elif change=='facet':r['conclusions'].pop('original_histories')
    elif change=='late':r['reviewed_at']='2026-10-16T18:30:01Z'
    else:r['qualifications']=''
    p=stage.seal(p['payload']);qp=plan(stage.QUOTE_PLAN,h,p);a=authorize(qp,monkeypatch,tmp_path);clock=Clock();clock.value='2026-10-16T18:30:01Z'
    before={f.name:f.read_bytes() for f in tmp_path.iterdir()}
    with pytest.raises(ValueError,match='PREQUOTE_REVIEW'):
        stage.acquire(qp,a,root=tmp_path,transport=lambda *a:pytest.fail('UNREVIEWED QUOTE DISPATCH'),history_bundle=h,prequote=p,wall=clock.wall,monotonic=clock.mono)
    assert {f.name:f.read_bytes() for f in tmp_path.iterdir()}==before


def test_actual_configuration_receipt_is_required_and_retained(monkeypatch,tmp_path):
    p=plan();a=authorize(p,monkeypatch,tmp_path);clock=Clock();receipt=deepcopy(stage.AUTHORIZED_CONFIGURATIONS[p['sha256']])
    monkeypatch.setattr(stage,'AUTHORIZED_CONFIGURATIONS',{})
    with pytest.raises(ValueError,match='CONFIGURATION_REQUIRED'):
        stage.acquire(p,a,root=tmp_path,transport=lambda *a:pytest.fail('UNBOUND CONFIGURATION DISPATCH'),wall=clock.wall,monotonic=clock.mono)
    assert not list(tmp_path.iterdir())
    b,_,_=capture(monkeypatch,tmp_path,p)
    assert b['payload']['control_receipt']==receipt
    raw=base64.b64decode(receipt['file_b64']);assert hashlib.sha256(raw).hexdigest()==receipt['file_sha256']
    cfg=json.loads(raw);assert cfg['payload']['authorization']==a and cfg['payload']['review']['subject_sha256']==model.digest({k:v for k,v in cfg['payload'].items() if k!='review'})
    assert stage.read_bundle(b)['control_receipt']==receipt


@pytest.mark.parametrize('change',['missing','bytes','hash','swapped','review_clock','review_before_authorization','account_after_review'])
def test_config_receipt_tampering_cannot_borrow_attempt_journal(monkeypatch,tmp_path,change):
    b,_,_=capture(monkeypatch,tmp_path);p=b['payload'];c=p['control_receipt']
    if change=='missing':p.pop('control_receipt')
    elif change=='bytes':c['file_b64']='e30='
    elif change=='hash':c['file_sha256']='f'*64
    else:
        cfg=json.loads(base64.b64decode(c['file_b64']));v=cfg['payload']
        if change=='swapped':v['plan_sha256']='f'*64
        elif change=='review_clock':v['review']['reviewed_at']='2026-10-16T18:00:02Z'
        elif change=='review_before_authorization':v['review']['reviewed_at']='2026-10-16T17:29:59Z'
        else:v['account']['observed_at']='2026-10-16T17:50:00Z'
        cfg=stage.seal(v);raw=model.encode(cfg);c.update(file_b64=base64.b64encode(raw).decode(),file_sha256=hashlib.sha256(raw).hexdigest())
    b=stage.seal(p)
    with pytest.raises((ValueError,KeyError)):stage.read_bundle(b)


def test_copied_history_cannot_reset_claim_in_another_custody(synthetic,monkeypatch,tmp_path):
    import shutil
    original=tmp_path/'original';original.mkdir();other=tmp_path/'other';other.mkdir()
    h,p=prepared(synthetic,monkeypatch,original)
    for f in original.iterdir():shutil.copyfile(f,other/f.name)
    qp=plan(stage.QUOTE_PLAN,h,p);a=authorize(qp,monkeypatch,other);clock=Clock();clock.value='2026-10-16T18:30:01Z'
    before={f.name:f.read_bytes() for f in other.iterdir()}
    with pytest.raises(ValueError,match='PARENT_CUSTODY'):
        stage.acquire(qp,a,root=other,transport=lambda *a:pytest.fail('CROSS-CUSTODY QUOTE DISPATCH'),history_bundle=h,prequote=p,wall=clock.wall,monotonic=clock.mono)
    assert {f.name:f.read_bytes() for f in other.iterdir()}==before


def test_24_hour_original_age_blocks_quote_while_target_is_still_pregame(synthetic,monkeypatch,tmp_path):
    from test_ncaaf_owner_terms import fixture as old
    packet,_=old(synthetic,monkeypatch);m=packet['payload']['custody_packet']['payload']['native_packet']['payload']['observation']['payload']['model']
    p=plan();v=p['payload']
    v['execution_window']={k:t.replace('2026-10-16','2026-10-15') for k,t in v['execution_window'].items()}
    for r in v['advance_permissions']:
        for k in ('reviewed_at','effective_from','effective_until'):r[k]=r[k].replace('2026-10-16','2026-10-15')
    p=stage.seal(v);clock=Clock();clock.value='2026-10-15T18:00:01Z';h,_,_=capture(monkeypatch,tmp_path,p,clock=clock)
    pref=review_prequote(evidence.prepare(h,model_packet=m,prepared_at='2026-10-15T18:01:00Z'),h)
    qp=plan(stage.QUOTE_PLAN,h,pref);a=authorize(qp,monkeypatch,tmp_path);clock.value='2026-10-16T18:30:01Z'
    before={f.name:f.read_bytes() for f in tmp_path.iterdir()}
    with pytest.raises(ValueError,match='INPUT_CLOCK'):
        stage.acquire(qp,a,root=tmp_path,transport=lambda *a:pytest.fail('EXPIRED ORIGINALS DISPATCH'),history_bundle=h,prequote=pref,wall=clock.wall,monotonic=clock.mono)
    assert {f.name:f.read_bytes() for f in tmp_path.iterdir()}==before


@pytest.mark.parametrize('point',['before_attempt','during_read'])
def test_interrupted_quote_keeps_history_claim_and_exact_spent_accounting(synthetic,monkeypatch,tmp_path,point):
    h,pref=prepared(synthetic,monkeypatch,tmp_path);qp=plan(stage.QUOTE_PLAN,h,pref);a=authorize(qp,monkeypatch,tmp_path)
    clock=Clock();clock.value='2026-10-16T18:30:01Z';sent=[]
    claim=tmp_path/(h['sha256']+'.quote-claim.json')
    def wall():
        if point=='before_attempt' and claim.exists():raise KeyboardInterrupt()
        return clock.wall()
    def interrupted(*args):
        sent.append(args[0]);raise KeyboardInterrupt()
    q=stage.acquire(qp,a,root=tmp_path,transport=interrupted,history_bundle=h,prequote=pref,wall=wall,monotonic=clock.mono)
    assert q['payload']['status']=='INCOMPLETE' and q['payload']['reason']=='NCAAF_STAGED_ACQUISITION_FAILURE'
    assert q['payload']['attempts']==len(sent)==(0 if point=='before_attempt' else 1)
    assert not q['payload']['objects'] and stage.verify_journal(tmp_path,q)
    stage.read_bundle(q,complete=False)
    with pytest.raises(ValueError,match='ALREADY_SPENT'):
        stage.acquire(qp,a,root=tmp_path,transport=lambda *a:pytest.fail('INTERRUPTED PLAN RESUMED'),history_bundle=h,prequote=pref,wall=clock.wall,monotonic=clock.mono)
    newer=deepcopy(qp['payload']);newer['collection_authorization_ref']+='-replacement';newer=stage.seal(newer)
    r,again,_=capture(monkeypatch,tmp_path,newer,[odds()],clock,h,pref)
    assert not again and r['payload']['reason']=='NCAAF_STAGED_HISTORY_ALREADY_CLAIMED' and r['payload']['attempts']==0
    assert claim.read_bytes()==model.encode(dict(plan_sha256=qp['sha256']))


def test_process_exit_after_durable_attempt_cannot_resume_or_reclaim_history(synthetic,monkeypatch,tmp_path):
    import subprocess,sys
    h,pref=prepared(synthetic,monkeypatch,tmp_path);qp=plan(stage.QUOTE_PLAN,h,pref);a=authorize(qp,monkeypatch,tmp_path)
    inputs=tmp_path/'SYNTHETIC-child-inputs.json'
    inputs.write_bytes(model.encode(dict(history=h,prequote=pref,plan=qp,authorization=a,control=stage.AUTHORIZED_CONFIGURATIONS[qp['sha256']],model_binding=synthetic[2])))
    code='\n'.join([
        'import sys;sys.path[:0]='+repr(sys.path),
        'import os,json,base64,pathlib',
        'from app_core import ncaaf_staged_acquisition as s',
        'from app_core import ncaaf_model_compatibility as m',
        'from copy import deepcopy',
        'from scripts.benchmark_drive_history_loading import blocked_network',
        'v=json.loads(pathlib.Path('+repr(str(inputs))+').read_bytes())',
        'm._binding=lambda:deepcopy(v["model_binding"]) # Same labelled SYNTHETIC lineage fixture as parent',
        'root=pathlib.Path('+repr(str(tmp_path))+')',
        's.inspect_security=lambda p:dict(status="VERIFIED") # SYNTHETIC only',
        'with blocked_network(),s.controlled(v["plan"],base64.b64decode(v["control"]["file_b64"]),v["control"]["file_sha256"],root=root,at="2026-10-16T18:30:01Z") as authorization:',
        ' s.acquire(v["plan"],authorization,root=root,transport=lambda *args:os._exit(74),history_bundle=v["history"],prequote=v["prequote"],wall=lambda:"2026-10-16T18:30:01Z",monotonic=lambda:0.)'])
    completed=subprocess.run([sys.executable,'-I','-B','-c',code],capture_output=True,timeout=15)
    assert completed.returncode==74,completed.stderr.decode(errors='replace')
    journal=tmp_path/(qp['sha256']+'.attempts.jsonl');events=[json.loads(x) for x in journal.read_bytes().splitlines()]
    assert [e['kind'] for e in events]==['start','attempt'] and events[1]['attempt']==1
    assert not (tmp_path/(qp['sha256']+'.stage.json')).exists()
    assert stage.verify_journal(tmp_path,h)
    clock=Clock();clock.value='2026-10-16T18:30:01Z'
    with pytest.raises(ValueError,match='ALREADY_SPENT'):
        stage.acquire(qp,a,root=tmp_path,transport=lambda *a:pytest.fail('PROCESS-EXIT PLAN RESUMED'),history_bundle=h,prequote=pref,wall=clock.wall,monotonic=clock.mono)
    newer=deepcopy(qp['payload']);newer['collection_authorization_ref']+='-replacement';newer=stage.seal(newer)
    q,sent,_=capture(monkeypatch,tmp_path,newer,[odds()],clock,h,pref)
    assert not sent and q['payload']['reason']=='NCAAF_STAGED_HISTORY_ALREADY_CLAIMED'
    assert journal.read_bytes()==b''.join(model.encode(e)+b'\n' for e in events)


def test_quote_authority_cannot_predate_the_actual_review_it_binds(synthetic,monkeypatch,tmp_path):
    h,pref=prepared(synthetic,monkeypatch,tmp_path);qp=plan(stage.QUOTE_PLAN,h,pref);a=authorize(qp,monkeypatch,tmp_path)
    a['payload']['authorized_at']='2026-10-16T18:02:59Z';a=stage.seal(a['payload'])
    account=stage.AUTHORIZED_ACCOUNT_EVIDENCE[a['payload']['reference']]
    cfg=configuration(qp,a,tmp_path,account);raw=model.encode(cfg)
    monkeypatch.setattr(stage,'AUTHORIZED_COLLECTIONS',{a['payload']['reference']:a['sha256']})
    monkeypatch.setattr(stage,'AUTHORIZED_CONFIGURATIONS',{qp['sha256']:dict(file_sha256=hashlib.sha256(raw).hexdigest(),file_b64=base64.b64encode(raw).decode())})
    before={f.name:f.read_bytes() for f in tmp_path.iterdir()};clock=Clock();clock.value='2026-10-16T18:30:01Z'
    with pytest.raises(ValueError,match='PREQUOTE_AUTHORIZATION_CLOCK'):
        stage.acquire(qp,a,root=tmp_path,transport=lambda *a:pytest.fail('FUTURE REVIEW AUTHORIZATION DISPATCH'),history_bundle=h,prequote=pref,wall=clock.wall,monotonic=clock.mono)
    assert {f.name:f.read_bytes() for f in tmp_path.iterdir()}==before
