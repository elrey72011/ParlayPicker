"""Labelled synthetic response bodies/artifacts; external transports blocked."""
from copy import deepcopy
import base64
import hashlib
import json
from unittest.mock import Mock
import pytest
from app_core import ncaaf_response_custody as custody
from app_core import ncaaf_compatible_pipeline as native, ncaaf_pipeline_evidence as adapter
from app_core import ncaaf_model_compatibility as model, ncaaf_research as research
from scripts.benchmark_drive_history_loading import blocked_network
import test_ncaaf_prospective_chronology as prior
import test_ncaaf_compatible_pipeline as legacy
from test_ncaaf_model_compatibility import synthetic

NOW = prior.NOW

@pytest.fixture(autouse=True)
def offline():
    with blocked_network():
        yield

def fixture(synthetic, monkeypatch, kind='spread_home', line=-3.5):
    packet,row = prior.fixture(synthetic,monkeypatch,kind,line)
    approval=deepcopy(adapter.ACCEPTED_PACKETS[packet['sha256']])
    p=packet['payload']['observation']['payload'];q=p['quote']
    objects=[]
    for obj in packet['payload']['dependency_objects']:
        b=json.loads(base64.b64decode(obj['bytes_b64']))
        request=b['request']; endpoint='games' if request['kind']=='games' else 'games/teams'
        params={'year':request['year'],'seasonType':'both','classification':'fbs'} if endpoint=='games' else {
            'year':request['year'],'week':request['week'],'seasonType':request['season_type']}
        rows=deepcopy(b['records'])
        for r in rows:r['SYNTHETIC_PRIVATE_BODY_CANARY']='discarded by frozen cleaner'
        raw=json.dumps(rows,indent=3).encode()
        meta=dict(provider='cfbd',endpoint=endpoint,request_scope=params,status=200,
            representation=custody.REPRESENTATION,content_type='application/json',complete=True,
            received_at='2026-10-09T11:59:59.500000Z',receipt_clock_meaning=custody.RECEIPT_MEANING)
        objects.append(custody.capture(raw,meta,native_batch=b))
    kind=q['market_type']
    outcome=dict(name=q['event_home_team'] if kind=='spread_home' else q['event_away_team'] if kind=='spread_away' else kind.split('_')[1].title(),
        point=q['point'],price=q['price'])
    event=dict(id=q['provider_event_id'],sport_key='americanfootball_ncaaf',home_team=q['event_home_team'],away_team=q['event_away_team'],
        commence_time=q['event_start_utc'],bookmakers=[dict(key=q['book'],last_update=q['recorded_at'],markets=[dict(
            key='spreads' if kind.startswith('spread') else 'totals',period=q['period'],settlement_rules=q['rules'],outcomes=[outcome])])])
    raw=json.dumps([event],indent=2).encode()
    meta=dict(provider='odds_api',endpoint=custody.ODDS_ENDPOINT,request_scope=dict(regions='us',markets='spreads' if kind.startswith('spread') else 'totals',
        bookmakers=q['book'],oddsFormat='american',dateFormat='iso',commenceTimeFrom='2026-10-10T00:00:00Z',commenceTimeTo='2026-10-11T00:00:00Z'),
        status=200,representation=custody.REPRESENTATION,content_type='application/json',complete=True,
        received_at='2026-10-09T12:00:00.500000Z',receipt_clock_meaning=custody.RECEIPT_MEANING)
    objects.append(custody.capture(raw,meta,quote=q,terms=p['source_review']['terms_review']))
    envelope=dict(payload=dict(version=custody.VERSION,evidence_label='SYNTHETIC',native_packet=packet,
        response_objects=objects,custody_admission={}),sha256='')
    trust(envelope,approval,monkeypatch)
    return envelope,row,approval

def trust(packet,approval,monkeypatch):
    p=packet['payload'];n=p['native_packet']['payload']['observation']['payload']
    v=dict(verified_at='2026-10-09T12:00:02Z',verifier='SYNTHETIC-custody-verifier',
        subject_version=custody.SUBJECT_VERSION,subject_sha256=custody.subject_hash(packet),
        permissions_sha256=model.digest(approval['dependency_source_review']['permissions_review']),
        terms_sha256=model.digest(n['source_review']['terms_review']),
        native_verification_sha256=model.digest(approval['dependency_source_review']['dependency_verification']))
    a=dict(review_id='SYNTHETIC-custody-acceptance',reviewer='SYNTHETIC-independent-reviewer',accepted_at='2026-10-09T12:00:03Z',
        subject_version=custody.SUBJECT_VERSION,subject_sha256=v['subject_sha256'],verification_sha256=model.digest(v))
    p['custody_admission']=dict(version=custody.ADMISSION_VERSION,verification=v,acceptance=a)
    packet['sha256']=model.digest(p)
    monkeypatch.setattr(custody,'ACCEPTED_ADMISSIONS',{a['review_id']:model.digest(a)})
    monkeypatch.setattr(adapter,'ACCEPTED_PACKETS',{packet['sha256']:deepcopy(approval)})

def actual(monkeypatch,packet):
    original=native.view
    monkeypatch.setattr(native,'view',lambda value:original(value['payload']['native_packet']) if value['payload']['version']==custody.VERSION else original(value))
    return legacy.actual(monkeypatch,packet)

def test_actual_caller_requires_original_response_custody(synthetic,monkeypatch):
    packet,row,_=fixture(synthetic,monkeypatch)
    before=deepcopy(packet)
    analysis,_=actual(monkeypatch,packet)
    result=analysis.loc[analysis.market_type.eq('spread_home')].iloc[0]
    assert result.ml_inference_status=='success',result.ml_unavailable_reason
    assert result.ml_probability==pytest.approx(0.636830651175619)
    assert adapter.diagnose(result)==dict(status='COMPLETE',reason='AVAILABLE')
    assert packet==before

def rehash(packet):
    for item in packet['payload']['response_objects']:
        item['sha256']=model.digest(item['payload'])
    packet['sha256']=model.digest(packet['payload'])

def body_change(packet,index,change):
    item=packet['payload']['response_objects'][index]['payload']
    rows=json.loads(base64.b64decode(item['body_b64']))
    change(rows)
    raw=json.dumps(rows,indent=5).encode()
    item.update(body_b64=base64.b64encode(raw).decode(),body_bytes=len(raw),body_sha256=hashlib.sha256(raw).hexdigest())

@pytest.mark.parametrize('change,reason',[
    ('missing','NCAAF_ORIGINAL_RESPONSE_UNAVAILABLE'),('missing_quote','NCAAF_ORIGINAL_RESPONSE_UNAVAILABLE'),
    ('corrupt','NCAAF_CUSTODY_INTEGRITY'),('swapped','NCAAF_CUSTODY_PROJECTION_CONFLICT'),
    ('projection','NCAAF_CUSTODY_PROJECTION_CONFLICT'),('locator','NCAAF_CUSTODY_PROJECTION_CONFLICT'),
    ('provider','NCAAF_CUSTODY_SCOPE_CONFLICT'),('endpoint','NCAAF_CUSTODY_SCOPE_CONFLICT'),
    ('season','NCAAF_CUSTODY_SCOPE_CONFLICT'),('request_week','NCAAF_CUSTODY_SCOPE_CONFLICT'),
    ('unrelated_game','NCAAF_CUSTODY_PROJECTION_CONFLICT'),('unrelated_team','NCAAF_CUSTODY_PROJECTION_CONFLICT'),
    ('orientation','NCAAF_CUSTODY_QUOTE_CONFLICT'),('event','NCAAF_CUSTODY_QUOTE_CONFLICT'),
    ('listing','NCAAF_CUSTODY_QUOTE_CONFLICT'),('product','NCAAF_CUSTODY_QUOTE_CONFLICT'),
    ('side','NCAAF_CUSTODY_QUOTE_CONFLICT'),('line','NCAAF_CUSTODY_QUOTE_CONFLICT'),('price','NCAAF_CUSTODY_QUOTE_CONFLICT'),
    ('provider_clock','NCAAF_CUSTODY_QUOTE_CONFLICT'),('duplicate_quote','NCAAF_CUSTODY_QUOTE_CONFLICT'),
    ('wrong_window','NCAAF_CUSTODY_SCOPE_CONFLICT'),('wrong_book_scope','NCAAF_CUSTODY_SCOPE_CONFLICT'),
    ('status','NCAAF_CUSTODY_HTTP_STATUS'),('incomplete','NCAAF_CUSTODY_HTTP_STATUS'),
    ('representation','NCAAF_CUSTODY_REPRESENTATION'),('meaning','NCAAF_CUSTODY_CLOCK_CONFLICT'),
    ('missing_clock','NCAAF_CUSTODY_CLOCK_CONFLICT'),('future_clock','NCAAF_CUSTODY_CLOCK_CONFLICT'),
    ('projection_before_receipt','NCAAF_CUSTODY_CLOCK_CONFLICT'),('permission_after_receipt','NCAAF_CUSTODY_PERMISSION_CLOCK_CONFLICT'),
    ('verification_before_facts','NCAAF_CUSTODY_SUBJECT_FUTURE_FACT'),('late_acceptance','NCAAF_CUSTODY_CLOCK_CONFLICT'),
    ('untrusted','NCAAF_CUSTODY_ADMISSION_NOT_TRUSTED'),('same_reviewer','NCAAF_CUSTODY_ADMISSION_CONFLICT'),
    ('borrowed','NCAAF_CUSTODY_SCHEMA'),('runtime','NCAAF_CUSTODY_RUNTIME_CHANGED'),
    ('credentials','NCAAF_CUSTODY_CREDENTIAL_FIELD'),('too_many','NCAAF_CUSTODY_LIMIT'),
])
def test_rejection_before_actual_numeric_inference(synthetic,monkeypatch,change,reason):
    packet,row,approval=fixture(synthetic,monkeypatch)
    p=packet['payload'];r=p['response_objects'][0]['payload'];q=p['response_objects'][-1]['payload']
    if change=='missing':p['response_objects']=[]
    elif change=='missing_quote':p['response_objects'].pop()
    elif change=='corrupt':r['body_b64']=base64.b64encode(b'[]').decode()
    elif change=='swapped':r.update({k:q[k] for k in ('body_b64','body_bytes','body_sha256')})
    elif change=='projection':r['projection']['native_sha256']='e'*64
    elif change=='locator':r['projection']['record_locators'][0]['game_id']=999
    elif change=='provider':r['metadata']['provider']='espn'
    elif change=='endpoint':r['metadata']['endpoint']='ratings'
    elif change=='season':r['metadata']['request_scope']['year']=2025
    elif change=='request_week':p['response_objects'][1]['payload']['metadata']['request_scope']['week']=9
    elif change=='unrelated_game':body_change(packet,0,lambda rows:rows[0].update(id=999))
    elif change=='unrelated_team':body_change(packet,0,lambda rows:rows[0].update(homeId=999))
    elif change=='orientation':body_change(packet,-1,lambda rows:rows[0].update(home_team='Georgia',away_team='Alabama'))
    elif change=='event':body_change(packet,-1,lambda rows:rows[0].update(id='SYNTHETIC-other'))
    elif change in {'listing','product','side','line','price','provider_clock','duplicate_quote'}:
        def alter(rows):
            b=rows[0]['bookmakers'][0];o=b['markets'][0]['outcomes'][0]
            if change=='listing':o['listing_id']='other'
            elif change=='product':o['product']='other'
            elif change=='side':o['name']='Georgia'
            elif change=='line':o['point']=-4.5
            elif change=='price':o['price']=-120
            elif change=='provider_clock':b['last_update']='2026-10-09T12:00:01Z'
            else:b['markets'][0]['outcomes'].append(deepcopy(o))
        body_change(packet,-1,alter)
    elif change=='wrong_window':q['metadata']['request_scope']['commenceTimeFrom']='2026-10-10T13:00:00Z'
    elif change=='wrong_book_scope':q['metadata']['request_scope']['bookmakers']='fanduel'
    elif change=='status':r['metadata']['status']=403
    elif change=='incomplete':r['metadata']['complete']=False
    elif change=='representation':r['metadata']['representation']='reconstructed-json'
    elif change=='meaning':r['metadata']['receipt_clock_meaning']='UNKNOWN'
    elif change=='missing_clock':r['metadata']['received_at']=None
    elif change=='future_clock':r['metadata']['received_at']='2026-10-09T12:00:05Z'
    elif change=='projection_before_receipt':r['metadata']['received_at']='2026-10-09T12:00:01Z'
    elif change=='permission_after_receipt':r['metadata']['received_at']='2026-10-09T10:00:00Z'
    elif change=='borrowed':p['native_packet']['payload']['version']=native.VERSION;p['native_packet']['sha256']=model.digest(p['native_packet']['payload'])
    elif change=='runtime':r['projection_implementation']['app_core/ncaaf_history.py']='e'*64
    elif change=='credentials':r['metadata']['request_scope']['apiKey']='SYNTHETIC_SECRET_CANARY'
    elif change=='too_many':p['response_objects']*=4
    rehash(packet);trust(packet,approval,monkeypatch)
    v,a=p['custody_admission']['verification'],p['custody_admission']['acceptance']
    if change=='verification_before_facts':v['verified_at']='2026-10-09T12:00:01.500000Z'
    elif change=='late_acceptance':a['accepted_at']='2026-10-09T12:00:05Z'
    elif change=='same_reviewer':a['reviewer']=v['verifier']
    a['verification_sha256']=model.digest(v)
    monkeypatch.setattr(custody,'ACCEPTED_ADMISSIONS',{} if change=='untrusted' else {a['review_id']:model.digest(a)})
    packet['sha256']=model.digest(p);monkeypatch.setattr(adapter,'ACCEPTED_PACKETS',{packet['sha256']:approval})
    original=deepcopy(packet);original_review=deepcopy(approval)
    monkeypatch.setattr(research,'centers',Mock(side_effect=AssertionError('Must reject before inference')))
    if change in {'missing','borrowed','too_many','credentials'}:
        with pytest.raises(ValueError,match='^'+reason+'$'):adapter.load(json.dumps(packet).encode())
    else:
        with adapter.selected([packet]):result=adapter.predict(row)
        assert result['ml_inference_status']=='unavailable'
        assert result['ml_unavailable_reason']==reason
        assert adapter.diagnose(dict(row,**result))==dict(status='INCOMPLETE',reason=reason)
    research.centers.assert_not_called()
    assert packet==original and approval==original_review

@pytest.mark.parametrize('raw',[b'{',b'[{"id":1,"id":2}]',b'[NaN]',b'[Infinity]',b'[1e999]',b'{}',b'[null]',b'\xff'])
def test_strict_body_parsing(raw,synthetic,monkeypatch):
    packet,_,_=fixture(synthetic,monkeypatch)
    r=packet['payload']['response_objects'][0]['payload']
    batch=json.loads(base64.b64decode(packet['payload']['native_packet']['payload']['dependency_objects'][0]['bytes_b64']))
    with pytest.raises(ValueError,match='^NCAAF_CUSTODY_BODY_SCHEMA$'):custody.capture(raw,r['metadata'],native_batch=batch)

def test_normalization_collision_and_exact_bytes_roundtrip(synthetic,monkeypatch):
    packet,_,approval=fixture(synthetic,monkeypatch)
    item=packet['payload']['response_objects'][0]
    b=json.loads(base64.b64decode(packet['payload']['native_packet']['payload']['dependency_objects'][0]['bytes_b64']))
    raw=base64.b64decode(item['payload']['body_b64']);rows=json.loads(raw)
    rows[0]['SYNTHETIC_PRIVATE_BODY_CANARY']='different discarded value'
    different=json.dumps(rows,sort_keys=True,separators=(',',':')).encode()
    other=custody.capture(different,item['payload']['metadata'],native_batch=b)
    assert raw!=different and item['payload']['body_sha256']!=other['payload']['body_sha256']
    assert item['payload']['projection']==other['payload']['projection']
    assert custody.capture(raw,item['payload']['metadata'],native_batch=b)==item
    restored=custody.load(json.loads(json.dumps(packet)))
    assert base64.b64decode(restored['payload']['response_objects'][0]['payload']['body_b64'])==raw
    assert custody.verify(packet,approval,NOW.isoformat())['original_response_available']
    # Acceptance/checkpoint are later facts, excluded from exact custody subject.
    before=custody.subject_hash(packet)
    del packet['payload']['native_packet']['payload']['observation']['payload']['source_review']['acceptance']
    del packet['payload']['native_packet']['payload']['observation']['payload']['as_of']
    assert custody.subject_hash(packet)==before

def test_limits_are_not_truncated(synthetic,monkeypatch):
    packet,_,approval=fixture(synthetic,monkeypatch)
    v=packet['payload']['response_objects'][0]['payload'];b=json.loads(base64.b64decode(packet['payload']['native_packet']['payload']['dependency_objects'][0]['bytes_b64']))
    with pytest.raises(ValueError,match='NCAAF_CUSTODY_LIMIT'):custody.capture(b' '*(custody.MAX_BODY_BYTES+1),v['metadata'],native_batch=b)
    monkeypatch.setattr(custody,'MAX_TOTAL_BODY_BYTES',1)
    with pytest.raises(ValueError,match='NCAAF_CUSTODY_LIMIT'):custody.verify(packet,approval,NOW.isoformat())

def test_old_readers_and_production_catalogs_unchanged(synthetic,monkeypatch):
    assert custody.ACCEPTED_ADMISSIONS=={} and adapter.ACCEPTED_PACKETS=={}
    packet,_,_=fixture(synthetic,monkeypatch)
    with pytest.raises(ValueError,match='NCAAF_COMPAT_OBSERVATION_SCHEMA'):native.load(packet)
    old=packet['payload']['native_packet']
    native.load(old)
    before=deepcopy(old)
    native.infer(old,NOW.isoformat())
    assert old==before

@pytest.mark.parametrize('kind,line',[('spread_home',-3.5),('spread_away',3.5),('total_over',50.5),('total_under',50.5)])
def test_actual_capture_export_replay_and_display(synthetic,monkeypatch,tmp_path,kind,line):
    from datetime import timedelta
    from app_core import research_replay
    from app_core.per_game_boards import per_game_board
    from app_core.public_board import build_package,validate_package
    from scripts.publish_board import render,assets_from_html
    from test_research_probability_browser import inspect_browser
    import test_source_contract_pipeline as ef
    packet,_,_=fixture(synthetic,monkeypatch,kind,line);before=deepcopy(packet)
    analysis,diagnostics=actual(monkeypatch,packet)
    row=analysis.loc[analysis.market_type.eq(kind)].iloc[0].to_dict()
    assert row['ml_inference_status']=='success',row['ml_unavailable_reason']
    expected=native.infer(packet['payload']['native_packet'],NOW.isoformat())[2]
    assert row['ml_probability']==expected and adapter.diagnose(row)['status']=='COMPLETE'
    monkeypatch.setattr(ef,'NOW',NOW);monkeypatch.setattr(ef,'CAPTURE',(NOW+timedelta(seconds=1)).isoformat())
    result=legacy.exported(monkeypatch,tmp_path/'export',analysis)
    frames=[per_game_board(result['card'],result['captured'],family=f,novig_only=True,college_fallback=True) for f in ('overall','sides','totals')]
    package=build_package(*frames);validate_package(package)
    family='sides' if kind.startswith('spread') else 'totals'
    shown=package['games'][family][0]['research_display']
    assert shown['availability_reason']=='AVAILABLE' and shown['probability']==expected
    assert shown['ev'] is None and shown['edge'] is None
    receipt=research_replay.retain_export(frames,package,result['card'],result['captured'],path=result['db'])
    _,sources=research_replay.read_export(receipt['export_id'],path=result['db'])
    retained=research_replay.frame_from_payload(next(iter(sources.values()))['original']['producer'])
    row2=retained.loc[retained.market_type.eq(kind)].iloc[0]
    saved=json.loads(row2.ml_estimate_metadata)['ncaaf_inputs']['payload']
    assert saved['original_packet']==before and packet==before
    assert saved['original_blend']['probability']['value']==row2.calibrated_probability and saved['ui_refresh'] is None
    for a,b in zip(before['payload']['response_objects'],saved['original_packet']['payload']['response_objects']):
        assert base64.b64decode(a['payload']['body_b64'])==base64.b64decode(b['payload']['body_b64'])
    assert not saved['scientific_acceptance'] and not saved['wagering_authority'] and saved['live_stake']==0
    assert all(c['production_bet_amount']==0 for c in result['card'].wager_contract)
    assert all(r['status']=='PASS' for r in package['games'][family])
    public=json.dumps(package)+render(package)
    assert all(value not in public for value in ('body_b64','response_objects','native_packet','SYNTHETIC_PRIVATE_BODY_CANARY','SYNTHETIC_UNUSED_SECRET_CANARY','custody_admission'))
    assert 'SYNTHETIC' in shown['basis'] and 'uncalibrated' in shown['basis']
    if kind=='spread_home':
        browser=inspect_browser(json.loads(assets_from_html(render(package))['board-data.json']),tmp_path/'browser',NOW)
        assert browser['initial']['shown'][0]['probability']==expected and browser['initial']['current']==0

def test_actual_rejection_retains_coverage_and_no_estimate(synthetic,monkeypatch):
    packet,_,approval=fixture(synthetic,monkeypatch)
    packet['payload']['response_objects'].pop();trust(packet,approval,monkeypatch)
    monkeypatch.setattr(research,'centers',Mock(side_effect=AssertionError('No rejected inference')))
    analysis,diagnostics=actual(monkeypatch,packet)
    row=analysis.loc[analysis.market_type.eq('spread_home')].iloc[0]
    assert row.ml_unavailable_reason=='NCAAF_ORIGINAL_RESPONSE_UNAVAILABLE'
    from app_core.slate_coverage import build_coverage,native_ncaaf
    run='20261009T120004.000000Z'
    coverage=build_coverage([native_ncaaf(diagnostics['ncaaf_schedule'],'2026-10-10')],selected_date='2026-10-10',
        as_of=NOW.isoformat(),run_id=run,candidates=analysis.assign(export_run_id=run).to_dict('records'),
        provider_health={'sports':{'americanfootball_ncaaf':{'outcome':'SUCCESS','processing':'SUCCESS'}}})
    decisions=coverage['decisions']
    assert len(decisions)==2 and decisions[0]['coverage_decision_state']=='UNVERIFIED'
    assert 'NCAAF_ORIGINAL_RESPONSE_UNAVAILABLE' in json.dumps(decisions[0])
    assert not research.centers.called

def test_consistently_rehashed_unrelated_native_chain_rejects(synthetic,monkeypatch):
    packet,row,_=fixture(synthetic,monkeypatch)
    n=packet['payload']['native_packet']
    legacy.mutate_batch(n,0,lambda b:b['records'][0].update(homeId=999),rehash=True)
    prior.seal(n,monkeypatch)
    approval=deepcopy(adapter.ACCEPTED_PACKETS[n['sha256']])
    b=json.loads(base64.b64decode(n['payload']['dependency_objects'][0]['bytes_b64']))
    prior_object=packet['payload']['response_objects'][0]['payload']
    raw=json.dumps(b['records']).encode()
    packet['payload']['response_objects'][0]=custody.capture(raw,prior_object['metadata'],native_batch=b)
    trust(packet,approval,monkeypatch)
    monkeypatch.setattr(research,'centers',Mock(side_effect=AssertionError('Unrelated identity must not infer')))
    with adapter.selected([packet]):result=adapter.predict(row)
    assert result['ml_unavailable_reason']=='NCAAF_COMPAT_DEPENDENCY_IDENTITY_CONFLICT'
    research.centers.assert_not_called()

def test_rehashed_body_cannot_reuse_trusted_acceptance(synthetic,monkeypatch):
    packet,row,approval=fixture(synthetic,monkeypatch)
    body_change(packet,0,lambda rows:rows[0].update(SYNTHETIC_PRIVATE_BODY_CANARY='different original body'))
    rehash(packet)
    # Even an otherwise matching projection has a different custody subject.
    monkeypatch.setattr(adapter,'ACCEPTED_PACKETS',{packet['sha256']:approval})
    with pytest.raises(ValueError,match='NCAAF_CUSTODY_ADMISSION_CONFLICT'):custody.verify(packet,approval,NOW.isoformat())

@pytest.mark.parametrize('where',['body_key','body_value','metadata'])
def test_credential_exclusion_rejects_before_retention(synthetic,monkeypatch,where):
    packet,_,_=fixture(synthetic,monkeypatch)
    obj=packet['payload']['response_objects'][0]['payload']
    metadata=deepcopy(obj['metadata']);rows=json.loads(base64.b64decode(obj['body_b64']))
    if where=='body_key':rows[0]['Authorization']='SYNTHETIC_SECRET_CANARY'
    elif where=='body_value':rows[0]['provider_note']='apiKey=SYNTHETIC_SECRET_CANARY'
    else:metadata['request_scope']['apiKey']='SYNTHETIC_SECRET_CANARY'
    batch=json.loads(base64.b64decode(packet['payload']['native_packet']['payload']['dependency_objects'][0]['bytes_b64']))
    with pytest.raises(ValueError,match='NCAAF_CUSTODY_CREDENTIAL_FIELD'):custody.capture(json.dumps(rows).encode(),metadata,native_batch=batch)

def test_missing_custody_selection_never_uses_default_inference(synthetic,monkeypatch):
    _,row,_=fixture(synthetic,monkeypatch)
    monkeypatch.setattr(research,'centers',Mock(side_effect=AssertionError('No implicit custody selection')))
    with adapter.selected([]):result=adapter.predict(row)
    assert result['ml_unavailable_reason']=='NCAAF_EXACT_OFFER_NOT_SELECTED'
    research.centers.assert_not_called()

@pytest.mark.parametrize('offset',[-10,16*60,2*86400])
def test_successor_preserves_future_stale_started_rejection(synthetic,monkeypatch,offset):
    from datetime import timedelta
    packet,row,_=fixture(synthetic,monkeypatch)
    monkeypatch.setattr(adapter,'generated_time',lambda:(NOW+timedelta(seconds=offset)).isoformat())
    monkeypatch.setattr(research,'centers',Mock(side_effect=AssertionError('Freshness rejection precedes math')))
    with adapter.selected([packet]):result=adapter.predict(row)
    assert result['ml_inference_status']=='unavailable' and result['ml_unavailable_reason'] in custody.REASONS
    research.centers.assert_not_called()

def test_corrupt_saved_computation_is_static_rejection(synthetic,monkeypatch):
    packet,row,_=fixture(synthetic,monkeypatch)
    with adapter.selected([packet]):result=adapter.predict(row)
    source=dict(row,**result);item=json.loads(source['ml_estimate_metadata'])
    saved=item['ncaaf_inputs']['payload'];saved['computation']['payload']['raw_probability']=.99
    item['ncaaf_inputs']['sha256']=model.digest(saved)
    monkeypatch.setattr(research,'centers',Mock(side_effect=AssertionError('Static diagnosis cannot infer')))
    assert adapter.diagnose(source,item)==dict(status='REJECTED',reason='NCAAF_CUSTODY_COMPUTATION_CONFLICT')
    research.centers.assert_not_called()

@pytest.mark.parametrize('age',[86400,86400.001,93604])
def test_original_body_age_cannot_be_reset_by_native_projection(synthetic,monkeypatch,age):
    from datetime import timedelta
    packet,row,_=fixture(synthetic,monkeypatch)
    n=packet['payload']['native_packet'];permission=prior.advance_dependency_permission()
    permission['reviewed_at']='2026-10-01T00:00:00Z'
    prior.seal(n,monkeypatch,permission=permission)
    approval=adapter.ACCEPTED_PACKETS[n['sha256']]
    for obj in packet['payload']['response_objects'][:-1]:
        obj['payload']['metadata']['received_at']=(NOW-timedelta(seconds=age)).isoformat()
    rehash(packet);trust(packet,approval,monkeypatch)
    immutable=deepcopy(packet)
    if age>86400:monkeypatch.setattr(research,'centers',Mock(side_effect=AssertionError('Stale body must reject before math')))
    with adapter.selected([packet]):result=adapter.predict(row)
    if age<=86400:assert result['ml_inference_status']=='success'
    else:
        assert result['ml_unavailable_reason']=='NCAAF_CUSTODY_INPUT_STALE'
        research.centers.assert_not_called()
    assert packet==immutable
