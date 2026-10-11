"""SYNTHETIC source bodies; no authentic historical numerical execution."""
from copy import deepcopy
import csv
import io
import json
import pytest
from app_core import nfl_owner_research as old, nfl_response_custody as old_custody
from app_core.research_estimate_trace import encode
from test_nfl_owner_research import packet_fixture, sign, actual_pipeline, private_app
from test_source_contract_pipeline import INFERENCE
from scripts.benchmark_drive_history_loading import blocked_network


@pytest.fixture(autouse=True)
def offline():
    with blocked_network():
        yield


def upstream_body(packet, *, publication=False, other_season=True):
    _, rows = old_custody.read(packet['payload']['objects']['scores'])
    if not publication:
        for r in rows:
            r.pop('source_available_at')
    if other_season:
        row = deepcopy(rows[0]); row.update(game_id='2025_01_BUF_DAL', season=2025)
        rows.insert(0, row)
    body = io.StringIO(newline=''); w = csv.DictWriter(body, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
    return body.getvalue().encode()


@pytest.mark.parametrize('publication,code', [(False,'NFL_PRIVATE_PUBLISHER_AVAILABILITY_MISSING'),
    (True,'NFL_PRIVATE_DEPENDENCY_IDENTITY_CONFLICT')])
def test_original_reader_reproduces_upstream_contract_rejections(monkeypatch, publication, code):
    packet, _ = packet_fixture(monkeypatch)
    p = packet['payload']; obj = p['objects']['scores']
    p['objects']['scores'] = old_custody.retained(upstream_body(packet, publication=publication), receipt=obj['receipt'])
    sign(p); packet['sha256'] = old.digest(p)
    with pytest.raises(ValueError, match=code):
        old.checked(packet, INFERENCE)


def source_fixture(monkeypatch):
    from app_core import nfl_observed_research as new, nfl_observed_custody as bodies, nfl_source_projection as projection
    packet, baseline = packet_fixture(monkeypatch); p = packet['payload']
    scoring = upstream_body(packet)
    raw_listing, listing = old_custody.read(p['objects']['listing'])
    e = p['event']
    schedule = dict(events=[dict(id=e['canonical_event_id'], home='Dallas Cowboys', away='Buffalo Bills',
        kickoff=e['start'], season=e['season'], neutral_site=False)])
    listing = {k:v for k,v in listing.items() if k not in {'event','quote'}}
    text = '\n'.join(k+'='+ ('true' if v is True else str(v)) for k,v in listing.items())
    clocks = dict(field='markets.last_update', meaning='provider_market_last_update', reference='SYNTHETIC original provider clock documentation')
    old_objects = p['objects']
    raw = dict(scores=scoring,schedule=encode(schedule).encode(),listing=text.encode(),clock_document=encode(clocks).encode(),
        odds=old_custody.read(old_objects['odds'])[0])
    p['objects'] = {}
    for name, value in raw.items():
        r = deepcopy(old_objects[name]['receipt'])
        r['format'] = 'csv' if name == 'scores' else 'text' if name == 'listing' else 'json'
        if name == 'scores': r['provider_clock_meaning'] = 'publisher_availability_unknown'
        if name in {'listing','clock_document'}: r['provider_clock_meaning'] = 'attributed_document_no_publication_clock'
        if name == 'odds':
            r['source_id']='the-odds-api:v4:americanfootball_nfl'
            p['permissions'][name]['source_id']=r['source_id']
        p['objects'][name] = bodies.retained(value,receipt=r)
    identities = {k:dict(body_sha256=o['body_sha256'],receipt_sha256=bodies.digest(o['receipt'])) for k,o in p['objects'].items()}
    def field(obj,path,op='identity',text=None): return dict(object=obj,path=path,text=text,operation=op)
    def doc(facts,values): return projection.seal(dict(version=projection.VERSION,facts=facts,values=values,original_objects=identities))
    sf = {k:field('schedule',['events',0,path],op) for k,path,op in (
        ('canonical_event_id','id','identity'),('home','home','team'),('away','away','team'),('start','kickoff','clock'),
        ('season','season','integer'),('neutral_site','neutral_site','identity'))}
    sf.update(provider_event_id=field('odds',[0,'id']),provider_namespace=field('odds',None,'receipt_source_namespace'))
    lf = {}
    for k,v in listing.items():
        value = 'true' if v is True else str(v); start=text.index(k+'=')+len(k)+1
        lf[k]=field('listing',None,'boolean' if v is True else 'identity',dict(start=start,end=start+len(value),exact=value))
    cf = {k:field('clock_document',[path]) for k,path in (
        ('provider_quote_field','field'),('quote_meaning','meaning'),('source_reference','reference'))}
    csv_rows = bodies.read(p['objects']['scores'])[1]
    score = dict(version=projection.VERSION,body_sha256=p['objects']['scores']['body_sha256'],receipt_sha256=bodies.digest(p['objects']['scores']['receipt']),
        season=e['season'],selected_row_indices=[i for i,r in enumerate(csv_rows) if int(r['season'])==e['season']],
        selection='exact_season_source_order_then_native_completed_before_utc_day',home=e['home'],away=e['away'],publisher_available_at=None,publisher_availability='UNKNOWN')
    p['projections']=dict(schedule=doc(sf,e),listing=doc(lf,listing),clock_document=doc(cf,dict(provider_quote_field='markets.last_update',
        quote_meaning='provider_market_last_update',source_reference=clocks['reference'],publisher_field=None,publisher_meaning='UNKNOWN')),scores=projection.seal(score),odds_path=[0])
    p['version']=new.VERSION
    sign_new(p)
    return projection.seal(p),baseline


def sign_new(p):
    from app_core import nfl_observed_research as new
    sign(p); p['owner_review']['version']='nfl-owner-observed-verification-v2'
    p['owner_review']['subject_sha256']=new.subject(p)


def test_source_projection_and_old_reader_remain_separate(monkeypatch):
    from app_core import nfl_observed_research as new
    packet,_=source_fixture(monkeypatch)
    features,listing=new.checked(packet,INFERENCE)
    assert features == packet['payload']['features'] and listing['payoff']=='novig_fvs_decided_game'
    assert packet['payload']['projections']['scores']['payload']['publisher_availability']=='UNKNOWN'
    with pytest.raises(ValueError,match='NFL_PRIVATE_PACKET_SCHEMA'): old.checked(packet,INFERENCE)
    with pytest.raises(ValueError,match='NFL_PRIVATE_PACKET_SCHEMA'):
        new.checked(packet_fixture(monkeypatch)[0],INFERENCE)


@pytest.mark.parametrize('side',[0,1])
def test_actual_caller_capture_export_replay_private_display(monkeypatch,tmp_path,side):
    from app_core import nfl_observed_research as new, nfl_private_routes as routes, research_replay
    from test_nfl_owner_research import opposite_packet
    from test_source_contract_pipeline import exported
    from app_core.per_game_boards import per_game_board
    packet,baseline=source_fixture(monkeypatch)
    if side:
        p=deepcopy(packet['payload']); q=p['quote']; e=p['event']
        q['market_type']='spread_away'; q['point']=-q['point']; q['price']=baseline.iloc[1].odds_american
        body,game=new.custody.read(p['objects']['odds']);outcome=game[0]['bookmakers'][0]['markets'][0]['outcomes'][0]
        outcome.update(name='Buffalo Bills',point=q['point'],price=q['price'])
        p['objects']['odds']=new.custody.retained(encode(game).encode(),receipt=p['objects']['odds']['receipt'])
        for name in ('schedule','listing','clock_document'):
            proj=p['projections'][name]['payload'];proj['original_objects']['odds']['body_sha256']=p['objects']['odds']['body_sha256']
            p['projections'][name]['sha256']=new.digest(proj)
        sign_new(p);packet=dict(payload=p,sha256=new.digest(p))
    original=deepcopy(packet)
    with routes.selected([packet]): analysis=actual_pipeline(monkeypatch)
    assert analysis.iloc[side].ml_probability==baseline.iloc[side].ml_probability
    assert routes.diagnose(analysis.iloc[side])['status']=='COMPLETE'
    # Explicit supplied-input entry point cannot invoke the general fetch lane.
    from app_core.nfl_bounded_acquisition import admitted_analysis
    from app_core import feature_processing as fp
    monkeypatch.setattr(fp,'fetch_nfl_stats',lambda *a,**k:pytest.fail('general fetch forbidden'))
    e=packet['payload']['event']
    inventory=dict(version='slate-inventory-v1',league='NFL',source='SYNTHETIC-independent-retained-schedule',selected_date='2026-10-06',
        timezone='America/New_York',observed_at='2026-10-06T19:25:02Z',status='PARTIAL',completeness_basis='SYNTHETIC one target, no full-slate claim',
        reasons=['SYNTHETIC_TARGET_ONLY'],events=[dict(canonical_event_id=e['canonical_event_id'],home_team=e['home'],away_team=e['away'],
        original_start=e['start'],provider_ids={'odds_api':e['provider_event_id']},schedule_status='SCHEDULED')])
    isolated=admitted_analysis(packet,baseline.iloc[side].to_dict(),inventories=[inventory])
    assert isolated.iloc[0].ml_probability==baseline.iloc[side].ml_probability
    assert len(isolated.attrs['slate_coverage']['decisions'])==1
    result=exported(monkeypatch,tmp_path,analysis)
    from app_core.nfl_bounded_acquisition import retain_analysis
    private_retained=retain_analysis(isolated,path=result['db'])
    assert len(private_retained['coverage']['decisions'])==1
    assert routes.diagnose(private_retained['captured'].iloc[0])['status']=='COMPLETE'
    assert any(json.loads(v)['raw_probability']==baseline.iloc[side].ml_probability for v in private_retained['frames'][1].nfl_private_research_display)
    assert json.loads(isolated.iloc[0].ml_estimate_metadata)['nfl_private_inputs']['payload']['original_blend'] is None
    receipt=research_replay.retain_export(result['frames'],result['package'],result['card'],result['captured'],path=result['db'])
    saved,sources=research_replay.read_export(receipt['export_id'],path=result['db'])
    retained=research_replay.frame_from_payload(next(iter(sources.values()))['original']['producer'])
    assert routes.diagnose(retained.iloc[side])['status']=='COMPLETE'
    saved_packet=json.loads(retained.iloc[side].ml_estimate_metadata)['nfl_private_inputs']['payload']
    assert saved_packet['original_packet']==original
    assert saved_packet['publisher_available_at'] is None and saved_packet['publisher_availability']=='UNKNOWN'
    frames=[per_game_board(result['card'],result['captured'],family=f,novig_only=True) for f in ('overall','sides','totals')]
    assert any(json.loads(v)['raw_probability']==baseline.iloc[side].ml_probability for v in frames[1].nfl_private_research_display)
    assert 'bytes_base64' not in encode(saved['package']) and 'OWNER_REVIEWED' not in encode(saved['package'])
    assert all(r['status']=='PASS' and r['win_estimate'] is None and r['odds'] is None for f in saved['package']['games'].values() for r in f)
    assert all(c['production_bet_amount']==0 for c in result['card'].wager_contract)
    from streamlit.testing.v1 import AppTest
    at=AppTest.from_function(private_app);at.session_state['private_candidates']=retained;at.run()
    assert not at.exception and any(m.value==f'{baseline.iloc[side].ml_probability:.2%}' for m in at.metric)
    assert any('OWNER_REVIEWED / PRIVATE_RESEARCH' in m.value for m in at.markdown)


@pytest.mark.parametrize('mutation,code',[
    ('body','NFL_PRIVATE_CUSTODY_CORRUPT'),('rehashed_body','NFL_PRIVATE_SOURCE_PROJECTION_CONFLICT'),
    ('missing_column','NFL_PRIVATE_DEPENDENCY_SCHEMA_MISSING'),
    ('review','NFL_PRIVATE_OWNER_REVIEW_CONFLICT'),('v1_review','NFL_PRIVATE_OWNER_REVIEW_MISSING'),
    ('clock','NFL_PRIVATE_SOURCE_PROJECTION_CONFLICT'),('stale','NFL_PRIVATE_QUOTE_CLOCK_CONFLICT'),
    ('started','NFL_PRIVATE_QUOTE_CLOCK_CONFLICT'),('future','NFL_PRIVATE_QUOTE_CLOCK_CONFLICT'),
    ('price','NFL_PRIVATE_OFFER_IDENTITY_CONFLICT'),('line','NFL_PRIVATE_OFFER_IDENTITY_CONFLICT'),
    ('orientation','NFL_PRIVATE_SOURCE_PROJECTION_CONFLICT'),('feature','NFL_PRIVATE_FEATURE_DERIVATION_CONFLICT'),
    ('feature_clock','NFL_PRIVATE_FEATURE_CLOCK_CONFLICT'),('permission','NFL_PRIVATE_PERMISSION_MISSING'),
    ('projection','NFL_PRIVATE_SOURCE_PROJECTION_CONFLICT'),('historical','NFL_PRIVATE_PACKET_SCHEMA')])
def test_rejections_precede_numerical_inference(monkeypatch,mutation,code):
    from app_core import nfl_observed_research as new, nfl_private_routes as routes
    from app_core.market_probability_model import predict_market_probabilities
    import pandas as pd
    packet,baseline=source_fixture(monkeypatch);p=packet['payload']
    if mutation=='body': p['objects']['scores']['body_sha256']='0'*64
    elif mutation=='rehashed_body':
        raw,_=new.custody.read(p['objects']['scores']);p['objects']['scores']=new.custody.retained(raw.replace(b'25',b'24'),receipt=p['objects']['scores']['receipt'])
    elif mutation=='missing_column':
        obj=p['objects']['scores'];_,rows=new.custody.read(obj)
        for row in rows:row.pop('result')
        body=io.StringIO(newline='');w=csv.DictWriter(body,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        p['objects']['scores']=new.custody.retained(body.getvalue().encode(),receipt=obj['receipt'])
        for name in ('schedule','listing','clock_document'):
            q=p['projections'][name];q['payload']['original_objects']['scores']['body_sha256']=p['objects']['scores']['body_sha256'];q['sha256']=new.digest(q['payload'])
        q=p['projections']['scores'];q['payload']['body_sha256']=p['objects']['scores']['body_sha256'];q['sha256']=new.digest(q['payload'])
        sign_new(p)
    elif mutation=='review': p['owner_review']['subject_sha256']='0'*64
    elif mutation=='v1_review': p['owner_review']['version']='nfl-owner-verification-v1'
    elif mutation=='clock': p['objects']['scores']['receipt']['observed_at']='2026-10-06T20:30:00Z'
    elif mutation=='stale': p['quote']['recorded_at']='2026-10-06T18:00:00Z'
    elif mutation=='future': p['quote']['recorded_at']='2026-10-06T20:00:00Z'
    elif mutation=='started': p['event']['start']='2026-10-06T19:00:00Z'
    elif mutation=='price': p['quote']['price']+=1
    elif mutation=='line': p['quote']['point']+=1
    elif mutation=='orientation': p['event']['home'],p['event']['away']=p['event']['away'],p['event']['home']
    elif mutation=='feature': p['features']['feature_diff_last5']+=1;sign_new(p)
    elif mutation=='feature_clock': p['feature_available_at']='2026-10-06T19:25:00Z';sign_new(p)
    elif mutation=='permission': p['permissions']['scores']['private_research_use']=False
    elif mutation=='projection': p['projections']['scores']['payload']['selected_row_indices']=[0]
    elif mutation=='historical': p['original_inference_time']=INFERENCE
    packet['sha256']=new.digest(p)
    with pytest.raises(ValueError,match=code):new.checked(packet,INFERENCE)
    calls=[]
    expected=deepcopy(p['model'])
    monkeypatch.setattr(new,'model_binding',lambda:expected)
    monkeypatch.setattr('app_core.market_probability_model._normal_cdf',lambda *a,**k:calls.append(a) or pytest.fail('inference forbidden'))
    # Structural/custody staging errors must also make zero mathematical calls.
    try:
        with routes.selected([packet]): out=predict_market_probabilities(pd.DataFrame([baseline.iloc[0].to_dict()]))
    except ValueError: pass
    else: assert out.iloc[0].ml_inference_status=='unavailable'
    assert calls==[]


def test_no_activation_duplicate_packets_or_cross_mode_approval(monkeypatch):
    from app_core import nfl_private_routes as routes, nfl_observed_research as new
    from app_core.market_probability_model import predict_market_probabilities
    import pandas as pd
    packet,baseline=source_fixture(monkeypatch)
    assert not routes.selection_requested()
    with pytest.raises(ValueError,match='NFL_PRIVATE_PACKET_AMBIGUOUS'),routes.selected([packet,packet]):pass
    strict,_=packet_fixture(monkeypatch)
    calls=[];monkeypatch.setattr('app_core.market_probability_model._normal_cdf',lambda x:calls.append(x) or .5)
    with routes.selected([strict,packet]):
        result=predict_market_probabilities(pd.DataFrame([baseline.iloc[0].to_dict()]))
    assert result.iloc[0].ml_unavailable_reason=='NFL_PRIVATE_PACKET_AMBIGUOUS' and calls==[]
    assert not routes.selection_requested()


def test_publisher_unknown_cannot_mask_a_recorded_publication_clock(monkeypatch):
    from app_core import nfl_observed_research as new, nfl_source_projection as projection
    packet,_=source_fixture(monkeypatch);p=packet['payload'];obj=p['objects']['scores']
    _,rows=new.custody.read(obj)
    for r in rows:r['source_available_at']='2099-01-01T00:00:00Z'
    body=io.StringIO(newline='');w=csv.DictWriter(body,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    p['objects']['scores']=new.custody.retained(body.getvalue().encode(),receipt=obj['receipt'])
    decoded=new.custody.objects(p['objects'])
    with pytest.raises(ValueError,match='NFL_PRIVATE_PUBLISHER_AVAILABILITY_CONFLICT'):projection.score_projection(p,decoded)


def test_static_numerical_components_and_strict_reader_are_preserved():
    import ast
    from pathlib import Path
    from app_core import nfl_observed_research as new
    def native_call(source):
        tree=ast.parse(source)
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='predict')
        block=next(n.body for n in fn.body if isinstance(n,ast.Try))
        start=next(i for i,n in enumerate(block) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='native_row' for t in n.targets))
        return ast.dump(ast.Module(body=block[start:start+4],type_ignores=[]),include_attributes=False)
    # The distribution invocation and recursion guard are source-identical, not a new predictor.
    assert native_call(Path(old.__file__).read_text(encoding='utf-8'))==native_call(Path(new.__file__).read_text(encoding='utf-8'))
    assert old.model_binding()==new.model_binding()


def test_documented_full_download_precedes_season_filter(monkeypatch):
    from app_core import feature_processing as fp
    from types import SimpleNamespace
    import pandas as pd
    # Published v0.3.2 loader semantics. No installed-package identity is invented.
    rows=pd.DataFrame([dict(game_id='SYNTHETIC-2025',season=2025,gameday='2025-09-01',home_team='DAL',away_team='BUF',home_score=99,away_score=0,result=99),
        dict(game_id='SYNTHETIC-2026',season=2026,gameday='2026-09-01',home_team='DAL',away_team='BUF',home_score=21,away_score=14,result=7)])
    calls=[]
    def read_csv(url):calls.append(url);return rows.copy()
    monkeypatch.setattr(pd,'read_csv',read_csv)
    def import_schedules(years):
        frame=pd.read_csv('http://www.habitatring.com/games.csv')
        return frame[frame.season.isin(years)]
    monkeypatch.setattr(fp,'nfl',SimpleNamespace(__name__='SYNTHETIC-documented-loader',__version__='SYNTHETIC-v0.3.2-contract',import_schedules=import_schedules))
    stats=fp.fetch_nfl_stats.__wrapped__(2026,as_of_date='2026-10-06')
    assert calls==['http://www.habitatring.com/games.csv']
    assert sorted(s['points_per_game'] for s in stats)==[14,21]
    assert sorted(s['games_played'] for s in stats)==[1,1]


def test_rejected_packet_clock_cannot_drop_independent_coverage(monkeypatch):
    from app_core import nfl_observed_research as new
    from app_core.nfl_bounded_acquisition import admitted_analysis
    packet,baseline=source_fixture(monkeypatch)
    original_event=deepcopy(packet['payload']['event'])
    packet['payload']['event']['start']='not-a-clock'
    packet['sha256']=new.digest(packet['payload'])
    calls=[]
    monkeypatch.setattr('app_core.market_probability_model._normal_cdf',lambda x:calls.append(x) or pytest.fail('inference forbidden'))
    source=baseline.iloc[0].to_dict()
    inventory=dict(version='slate-inventory-v1',league='NFL',source='SYNTHETIC-independent-schedule',selected_date='2026-10-06',
        timezone='America/New_York',observed_at='2026-10-06T19:25:02Z',status='PARTIAL',completeness_basis='SYNTHETIC one target',
        reasons=['SYNTHETIC_TARGET_ONLY'],events=[dict(canonical_event_id='SYNTHETIC-independent-event',home_team=original_event['home'],
        away_team=original_event['away'],original_start=original_event['start'],provider_ids={'odds_api':original_event['provider_event_id']},schedule_status='SCHEDULED')])
    result=admitted_analysis(packet,source,inventories=[inventory])
    assert result.iloc[0].ml_unavailable_reason=='NFL_PRIVATE_EVENT_MAPPING_CONFLICT' and calls==[]
    decisions=result.attrs['slate_coverage']['decisions']
    assert len(decisions)==1 and decisions[0]['coverage_decision_state']=='UNVERIFIED'
    assert decisions[0]['canonical_event_id']=='SYNTHETIC-independent-event'
    assert result.iloc[0].ml_probability != result.iloc[0].ml_probability
    assert result.iloc[0].calibrated_probability != result.iloc[0].calibrated_probability
