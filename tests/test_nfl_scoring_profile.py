"""Labelled synthetic full-source CSVs; no authentic body or inference."""
from copy import deepcopy
import csv
import io
import json
import pytest
from scripts.benchmark_drive_history_loading import blocked_network


@pytest.fixture(autouse=True)
def offline():
    with blocked_network():
        yield


def sized_csv(size,original=()):
    header=b'game_id,season,gameday,home_team,away_team,home_score,away_score,result,stadium\n'
    out=bytearray(header); index=0
    for row in original:
        value=io.StringIO(newline=''); writer=csv.DictWriter(value,fieldnames=header.decode().strip().split(','))
        writer.writerow({k:row.get(k,'') for k in writer.fieldnames})
        out.extend(value.getvalue().encode())
    while size-len(out)>256:
        row=f'SYNTHETIC-{index},2025,2025-10-01,DAL,BUF,21,14,7,SYNTHETIC stadium'.encode()
        out.extend(row+b'\n'); index+=1
    final=f'SYNTHETIC-{index},2025,2025-10-01,DAL,BUF,21,14,7,'.encode()
    out.extend(final+b'X'*(size-len(out)-len(final)-1)+b'\n')
    assert len(out)==size
    return bytes(out)


def test_original_size_rejection_is_preserved():
    from app_core import nfl_observed_custody as old, nfl_bounded_acquisition as lane
    from test_nfl_bounded_acquisition import Stream
    raw=sized_csv(2183999)
    with pytest.raises(ValueError,match='NFL_PRIVATE_RESPONSE_SIZE'):
        old.retained(raw,receipt={})
    with pytest.raises(ValueError,match='NFL_ACQUIRE_BODY_LIMIT'):
        lane.read(Stream(raw),old.MAX_BODY,10,lambda:0)
    assert old.MAX_BODY==512*1024 and old.MAX_TOTAL==2*1024*1024


def transfer(raw,receipt):
    import hashlib
    from app_core.nfl_scoring_acquisition import PRIMARY
    return dict(wire_bytes=len(raw),wire_sha256=hashlib.sha256(raw).hexdigest(),encoding='identity',
        requests=[dict(attempt=1,request=deepcopy(PRIMARY),started_at=receipt['request_started_at'],
            received_at=receipt['received_at'],status=200,location_sha256=None)])


def sign_new(p):
    from app_core import nfl_scoring_research as reader
    from test_nfl_owner_research import sign
    sign(p); p['owner_review']['version']='nfl-owner-scoring-verification-v3'
    p['owner_review']['subject_sha256']=reader.subject(p)


def reproject(p):
    from app_core import nfl_scoring_custody as bodies,nfl_source_projection as projection
    identities={k:dict(body_sha256=o['body_sha256'],receipt_sha256=bodies.digest(o['receipt'])) for k,o in p['objects'].items()}
    for name in ('schedule','listing','clock_document'):
        value=p['projections'][name]['payload']; value['original_objects']=deepcopy(identities)
        p['projections'][name]=projection.seal(value)
    rows=bodies.read_score(p['objects']['scores'])[1]; e=p['event']
    p['projections']['scores']=projection.seal(dict(version=projection.VERSION,
        body_sha256=p['objects']['scores']['body_sha256'],receipt_sha256=bodies.digest(p['objects']['scores']['receipt']),
        season=e['season'],selected_row_indices=[i for i,r in enumerate(rows) if int(r['season'])==e['season']],
        selection='exact_season_source_order_then_native_completed_before_utc_day',home=e['home'],away=e['away'],publisher_available_at=None,publisher_availability='UNKNOWN'))
    sign_new(p)
    return projection.seal(p)


def fixture(monkeypatch,size=2183999):
    from test_nfl_observed_research import source_fixture
    from app_core import nfl_observed_custody as old,nfl_scoring_custody as bodies,nfl_scoring_profile as profile,nfl_scoring_research as reader
    packet,baseline=source_fixture(monkeypatch); p=packet['payload']
    rows=old.read(p['objects']['scores'])[1]; raw=sized_csv(size,rows)
    receipt=deepcopy(p['objects']['scores']['receipt']);receipt.update(source_id=profile.SOURCE_ID,endpoint=profile.ENDPOINT)
    p['objects']['scores']=bodies.retained_score(raw,receipt=receipt,transfer=transfer(raw,receipt))
    p['permissions']['scores'].update(source_id=profile.SOURCE_ID,endpoint=profile.ENDPOINT)
    p.update(version=reader.VERSION,profile=profile.PROFILE)
    return reproject(p),baseline


def acquire_synthetic_original(monkeypatch,tmp_path,packet):
    """Same fake-acquired complete bytes become the later exact review subject."""
    from app_core import nfl_scoring_acquisition as lane,nfl_scoring_custody as bodies
    from test_nfl_scoring_acquisition import fixture as acquisition_fixture
    from test_nfl_bounded_acquisition import Stream
    root=tmp_path/'SYNTHETIC-scoring-acquisition';root.mkdir()
    plan,authorization=acquisition_fixture(monkeypatch,root)
    value=plan['payload'];p=packet['payload'];permission=p['permissions']['scores']
    observed=p['objects']['scores']['receipt']['observed_at']
    value['window']=dict(not_before='2026-10-06T19:25:00Z',not_after='2026-10-06T19:26:00Z',expires_at='2026-10-06T19:26:30Z')
    value['permission'].update(reviewed_at=permission['reviewed_at'],effective_from=permission['effective_from'],effective_until=permission['effective_until'])
    plan=lane.seal(value);a=authorization['payload']
    a.update(plan_sha256=plan['sha256'],authorized_at='2026-10-06T19:00:00Z',expires_at=value['window']['expires_at'])
    authorization=lane.seal(a)
    monkeypatch.setattr(lane,'AUTHORIZED_COLLECTIONS',{a['reference']:authorization['sha256']})
    monkeypatch.setattr(lane,'REVIEWED_PERMISSIONS',{value['permission']['reference']:lane.digest(value['permission'])})
    raw=bodies.read_score(p['objects']['scores'])[0]
    with monkeypatch.context() as untouched:
        untouched.setattr('app_core.market_probability_model._normal_cdf',lambda *a,**k:pytest.fail('acquisition cannot infer'))
        bundle=lane.acquire(plan,authorization,root=root,transport=lambda *a:Stream(raw),wall=lambda:observed)
    assert bundle['payload']['status']=='CAPTURED_UNREVIEWED' and bundle['payload']['attempted']==1
    assert bundle['payload']['inference'] is bundle['payload']['source_acceptance'] is bundle['payload']['owner_reviewed'] is False
    p['objects']['scores']=bundle['payload']['object']
    assert bodies.read_score(p['objects']['scores'])[0]==raw
    # Later labelled owner review rebinds the actual receipt and bytes, unchanged.
    return reproject(p)


@pytest.mark.parametrize('size',[2183999,4*1024*1024])
def test_full_original_score_size_profile_and_old_reader_separation(monkeypatch,size):
    from app_core import nfl_scoring_research as new,nfl_observed_research as old,nfl_scoring_custody as bodies
    from app_core.research_estimate_trace import encode
    from test_source_contract_pipeline import INFERENCE
    packet,_=fixture(monkeypatch,size)
    assert len(bodies.read_score(packet['payload']['objects']['scores'])[0])==size
    assert new.checked(packet,INFERENCE)[0]==packet['payload']['features']
    with pytest.raises(ValueError,match='NFL_PRIVATE_PACKET_SCHEMA'):old.checked(packet,INFERENCE)
    assert new.load(encode(packet).encode())==packet


def test_one_byte_over_and_other_bodies_do_not_borrow_profile(monkeypatch):
    from app_core import nfl_scoring_custody as bodies,nfl_observed_custody as old,nfl_scoring_profile as profile
    packet,_=fixture(monkeypatch,1024); receipt=packet['payload']['objects']['scores']['receipt']
    raw=sized_csv(profile.MAX_BODY+1)
    with pytest.raises(ValueError,match='NFL_PRIVATE_RESPONSE_SIZE'):bodies.retained_score(raw,receipt=receipt,transfer=transfer(raw,receipt))
    with pytest.raises(ValueError,match='NFL_PRIVATE_RESPONSE_SIZE'):old.retained(b'x'*(512*1024+1),receipt={})
    changed=deepcopy(packet['payload']['objects']);changed['odds']=changed['scores']
    with pytest.raises(ValueError,match='NFL_PRIVATE_SCORING_PROFILE_CONFLICT'):bodies.objects(changed)


def test_malformed_wide_csv_rejects_before_padding_or_remaining_rows(monkeypatch):
    from app_core import nfl_scoring_custody as bodies
    packet,_=fixture(monkeypatch,1024);obj=packet['payload']['objects']['scores']
    fields=['game_id','season','gameday','home_team','away_team','home_score','away_score','result']+[f'SYNTHETIC-extra-{i}' for i in range(1000)]
    raw=(','.join(fields)+'\nSYNTHETIC,2026\n').encode()
    class Reader:
        line_num=0
        def __iter__(self):return self
        def __next__(self):
            self.line_num+=1
            if self.line_num==1:return fields
            if self.line_num==2:return ['SYNTHETIC','2026']
            pytest.fail('Parser must reject malformed width before padding and reading remaining rows')
    monkeypatch.setattr(bodies.csv,'reader',lambda *a,**k:Reader())
    with pytest.raises(ValueError,match='NFL_PRIVATE_RESPONSE_SCHEMA'):
        bodies.retained_score(raw,receipt=obj['receipt'],transfer=transfer(raw,obj['receipt']))


def test_aggregate_exact_limit_and_one_over(monkeypatch):
    from app_core import nfl_scoring_custody as bodies,nfl_observed_custody as old,nfl_scoring_profile as profile
    packet,_=fixture(monkeypatch,profile.MAX_BODY); items={'scores':packet['payload']['objects']['scores']}
    r=deepcopy(packet['payload']['objects']['clock_document']['receipt']);r['format']='text'
    for index in range(4):
        raw=f'SYNTHETIC-{index}'.encode()+b'x'*(old.MAX_BODY-len(f'SYNTHETIC-{index}'))
        items[str(index)]=old.retained(raw,receipt=r)
    assert len(bodies.objects(items))==5
    items['over']=old.retained(b'x',receipt=r)
    with pytest.raises(ValueError,match='NFL_PRIVATE_RESPONSE_SIZE'):bodies.objects(items)


def test_serialized_packet_exact_limit_and_selection_ceiling(monkeypatch):
    from app_core import nfl_scoring_research as new,nfl_private_routes as routes,nfl_source_projection as projection
    from app_core.research_estimate_trace import encode
    packet,baseline=fixture(monkeypatch,1024);p=packet['payload']
    baseline=len(encode(packet).encode());p['owner_review']['findings']['permissions']['basis']+='X'*(new.MAX_PACKET-baseline)
    packet=projection.seal(p)
    assert len(encode(packet).encode())==new.MAX_PACKET and routes.load(encode(packet).encode())==packet
    p['owner_review']['findings']['permissions']['basis']+='X'
    with pytest.raises(ValueError,match='NFL_PRIVATE_PACKET_SCHEMA'):routes.load(encode(projection.seal(p)).encode())
    packet2=deepcopy(packet);packet2['payload']['owner_review']['findings']['permissions']['basis']='Y'+packet2['payload']['owner_review']['findings']['permissions']['basis'][1:]
    packet2=projection.seal(packet2['payload'])
    with pytest.raises(ValueError,match='NFL_PRIVATE_PACKET_AMBIGUOUS'):
        with routes.selected([packet,packet2]):pytest.fail('selection >16MiB accepted')


@pytest.mark.parametrize('attack,code',[
    ('body','NFL_PRIVATE_CUSTODY_CORRUPT'),('source','NFL_PRIVATE_SCORING_SOURCE_CONFLICT'),
    ('profile','NFL_PRIVATE_PACKET_SCHEMA'),('clock','NFL_PRIVATE_CUSTODY_CLOCK_CONFLICT'),
    ('stale','NFL_PRIVATE_QUOTE_CLOCK_CONFLICT'),('future','NFL_PRIVATE_QUOTE_CLOCK_CONFLICT'),
    ('started','NFL_PRIVATE_QUOTE_CLOCK_CONFLICT'),('identity','NFL_PRIVATE_EVENT_MAPPING_CONFLICT'),
    ('review','NFL_PRIVATE_OWNER_REVIEW_MISSING')])
def test_rejections_precede_numerical_inference(monkeypatch,attack,code):
    from app_core import nfl_scoring_research as new,nfl_private_routes as routes,nfl_source_projection as projection
    from test_nfl_owner_research import actual_pipeline
    packet,baseline=fixture(monkeypatch,1024);p=packet['payload']
    if attack=='body':p['objects']['scores']['body_sha256']='0'*64
    elif attack=='source':p['objects']['scores']['receipt']['source_id']='nflverse:nfldata:games'
    elif attack=='profile':p['profile']='borrow-old-profile'
    elif attack=='clock':p['objects']['scores']['receipt']['observed_at']='2026-10-07T00:00:00Z'
    elif attack=='stale':p['quote']['recorded_at']='2026-10-06T18:00:00Z'
    elif attack=='future':p['quote']['recorded_at']='2026-10-07T00:00:00Z'
    elif attack=='started':p['event']['start']='2026-10-06T18:00:00Z'
    elif attack=='identity':p['event']['home'],p['event']['away']=p['event']['away'],p['event']['home']
    if attack in ('clock','identity'):reproject(p)
    sign_new(p)
    if attack=='review':p['owner_review']['version']='nfl-owner-observed-verification-v2'
    packet=projection.seal(p)
    # Malformed original-body/schema packets cannot even be staged.
    from test_source_contract_pipeline import INFERENCE
    with pytest.raises(ValueError,match=code):new.checked(packet,INFERENCE)
    expected=deepcopy(p['model']);calls=[]
    monkeypatch.setattr(new,'model_binding',lambda:expected)
    monkeypatch.setattr('app_core.market_probability_model._normal_cdf',lambda *a,**k:calls.append(a) or pytest.fail('inference forbidden'))
    from app_core.market_probability_model import predict_market_probabilities
    import pandas as pd
    try:
        with routes.selected([packet]):result=predict_market_probabilities(pd.DataFrame([baseline.iloc[0].to_dict()]))
    except ValueError:pass
    else:assert result.iloc[0].ml_inference_status=='unavailable'
    assert calls==[]


@pytest.mark.parametrize('side,size',[(0,2183999),(1,2183999),(0,4*1024*1024)])
def test_actual_caller_capture_export_replay_private_browser(monkeypatch,tmp_path,side,size):
    from app_core import nfl_scoring_research as new,nfl_private_routes as routes,nfl_scoring_custody as bodies,nfl_observed_custody as small,nfl_source_projection as projection,research_replay
    from app_core.research_estimate_trace import encode
    from test_nfl_owner_research import actual_pipeline,private_app
    from test_source_contract_pipeline import exported
    packet,baseline=fixture(monkeypatch,size)
    packet=acquire_synthetic_original(monkeypatch,tmp_path,packet)
    if side:
        p=packet['payload'];q=p['quote'];q.update(market_type='spread_away',point=-q['point'],price=baseline.iloc[1].odds_american)
        _,games=small.read(p['objects']['odds']);games[0]['bookmakers'][0]['markets'][0]['outcomes'][0].update(name='Buffalo Bills',point=q['point'],price=q['price'])
        p['objects']['odds']=small.retained(encode(games).encode(),receipt=p['objects']['odds']['receipt']);packet=reproject(p)
    original=deepcopy(packet)
    with routes.selected([packet]):analysis=actual_pipeline(monkeypatch)
    raw=analysis.iloc[side].ml_probability
    assert raw==baseline.iloc[side].ml_probability and routes.diagnose(analysis.iloc[side])['status']=='COMPLETE'
    result=exported(monkeypatch,tmp_path,analysis)
    receipt=research_replay.retain_export(result['frames'],result['package'],result['card'],result['captured'],path=result['db'])
    saved,sources=research_replay.read_export(receipt['export_id'],path=result['db'])
    retained=research_replay.frame_from_payload(next(iter(sources.values()))['original']['producer'])
    assert routes.diagnose(retained.iloc[side])['status']=='COMPLETE'
    p=json.loads(retained.iloc[side].ml_estimate_metadata)['nfl_private_inputs']['payload']
    assert p['original_packet']==original and p['profile']==new.profile.PROFILE and p['publisher_availability']=='UNKNOWN'
    assert bodies.read_score(p['original_packet']['payload']['objects']['scores'])[0]==bodies.read_score(original['payload']['objects']['scores'])[0]
    assert 'bytes_base64' not in encode(saved['package']) and 'OWNER_REVIEWED' not in encode(saved['package'])
    assert all(r['status']=='PASS' and r['win_estimate'] is None and r['odds'] is None for f in saved['package']['games'].values() for r in f)
    assert all(c['production_bet_amount']==0 for c in result['card'].wager_contract)
    from streamlit.testing.v1 import AppTest
    at=AppTest.from_function(private_app);at.session_state['private_candidates']=retained;at.run()
    assert not at.exception and any(m.value==f'{raw:.2%}' for m in at.metric)
    assert any('OWNER_REVIEWED / PRIVATE_RESEARCH' in m.value for m in at.markdown)


def test_maximum_synthetic_parsing_and_serialization_measurements(monkeypatch,tmp_path):
    import time,tracemalloc
    from app_core import nfl_scoring_custody as bodies,nfl_scoring_research as reader
    from app_core.research_estimate_trace import encode
    packet,_=fixture(monkeypatch,4*1024*1024)
    tracemalloc.start();start=time.perf_counter();raw,rows=bodies.read_score(packet['payload']['objects']['scores'])
    parse_seconds=time.perf_counter()-start;_,parse_peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    tracemalloc.start();start=time.perf_counter();serialized=encode(packet).encode()
    seconds=time.perf_counter()-start;_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    assert len(raw)==reader.profile.MAX_BODY and len(serialized)<reader.MAX_PACKET
    receipt=dict(synthetic_only=True,body_bytes=len(raw),rows=len(rows),packet_bytes=len(serialized),
        parse_seconds=parse_seconds,parse_peak_bytes=parse_peak,serialization_seconds=seconds,serialization_peak_bytes=peak,
        inference_calls=0,network_requests=0)
    (tmp_path/'SYNTHETIC-overhead.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,sort_keys=True))
