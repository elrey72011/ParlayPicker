"""Labelled synthetic prose/custody only. No authentic inference or transport."""
from copy import deepcopy
import json
import pytest
from scripts.benchmark_drive_history_loading import blocked_network
from test_source_contract_pipeline import INFERENCE
from test_nfl_scoring_profile import fixture as scoring_fixture, sign_new, reproject


@pytest.fixture(autouse=True)
def offline():
    with blocked_network():
        yield


def test_before_actual_v3_reader_rejects_prose_interpretation(monkeypatch):
    from app_core import nfl_observed_custody as bodies, nfl_source_projection as projection, nfl_scoring_research as old
    packet, _ = scoring_fixture(monkeypatch, 1024)
    p = packet['payload']
    raw, listing = bodies.read(p['objects']['listing'])
    # Realistic prose, not an asserted provider internal period token.
    excerpt = 'The contract covers the entire game, inclusive of Overtime.'
    listing = listing.replace('period=full_game', 'period=' + excerpt)
    p['objects']['listing'] = bodies.retained(listing.encode(), receipt=p['objects']['listing']['receipt'])
    for key, f in p['projections']['listing']['payload']['facts'].items():
        start = listing.index(key+'=') + len(key) + 1
        end = listing.find('\n', start)
        end = len(listing) if end < 0 else end
        f['text'] = dict(start=start, end=end, exact=listing[start:end])
    packet = reproject(p)
    with pytest.raises(ValueError, match='NFL_PRIVATE_SOURCE_PROJECTION_CONFLICT'):
        old.checked(packet, INFERENCE)
    assert p['projections']['listing']['payload']['values']['period'] == 'full_game'
    assert p['projections']['listing']['payload']['facts']['period']['text']['exact'] == excerpt


def interpreted_fixture(monkeypatch, away=False, reference_away=False):
    from app_core import nfl_source_interpretation as projection, nfl_interpreted_research as reader, nfl_observed_custody as bodies
    from app_core.research_estimate_trace import encode
    packet, baseline = scoring_fixture(monkeypatch, 1024)
    p = packet['payload']; e, q = p['event'], p['quote']
    raw, games = bodies.read(p['objects']['odds']); outcome = games[0]['bookmakers'][0]['markets'][0]['outcomes'][0]
    if away:
        q.update(market_type='spread_away', point=-q['point'], price=baseline.iloc[1].odds_american)
        outcome.update(name='Buffalo Bills', point=q['point'], price=q['price'])
    outcome['sid'] = 'SYNTHETIC-outcome-123'
    p['objects']['odds'] = bodies.retained(encode(games).encode(), receipt=p['objects']['odds']['receipt'])
    ref = 'Buffalo Bills' if reference_away else 'Dallas Cowboys'
    opp = 'Dallas Cowboys' if reference_away else 'Buffalo Bills'
    selected_reference = away == reference_away
    threshold = -q['point'] if selected_reference else q['point']
    specs = {
        'schedule': {
            'canonical_event_id':('literal',e['canonical_event_id']), 'provider_event_id':('literal',e['provider_event_id']),
            'provider_namespace':('literal','odds_api'), 'home':('team','Dallas Cowboys'), 'away':('team','Buffalo Bills'),
            'start':('utc_instant',e['start']), 'season':('integer','2026'), 'neutral_site':('neutral_site','not a neutral-site game')},
        'listing': {
            'listing_id':('literal','SYNTHETIC-NFLSPR-123'), 'product':('literal','Ludlow Exchange doing business as Novig'),
            'jurisdiction':('literal','SYNTHETIC documented contract jurisdiction'),
            'period':('period','entire game'), 'overtime':('overtime','inclusive of Overtime'),
            'payoff':('payoff','Fair Value Settlement'), 'rule_edition':('edition','Rule edition: NFL-001 edition 2026-10-01.'),
            'amendments':('amendments','no applicable amendments'),
            'effective_from':('effective_from','Effective for this listing from 2026-10-01T00:00:00Z'),
            'effective_until':('effective_until','Effective for this listing until 2026-10-08T00:00:00Z'),
            'source_reference':('literal','https://synthetic.invalid/exact-listing'),
            'reference_team':('team',ref), 'opponent':('team',opp), 'threshold':('half_point',str(threshold)),
            'comparison':('comparison','greater than'), 'position':('position','YES' if selected_reference else 'NO'),
            'feed_kind':('feed_kind','Outcome SID'), 'feed_identity':('literal',outcome['sid']),
            'applicability':('applicability',f"Listing SYNTHETIC-NFLSPR-123 governs provider event {e['provider_event_id']} for {ref} versus {opp}.")},
        'clock_document': {
            'provider_quote_field':('quote_field','market-level <code>last_update</code>'),
            'quote_meaning':('quote_meaning','the last time our system saw odds for that market from the bookmaker'),
            'source_reference':('literal','https://synthetic.invalid/provider-guide')},
        'rules': {'rulebook':('literal','NFL-001'), 'ticker':('literal','NFLSPR'),
            'rule_edition':('edition','Rule edition: NFL-001 edition 2026-10-01.'),
            'amendments':('amendments','no applicable amendments'), 'period':('period','entire game'),
            'overtime':('overtime','including overtime'), 'payoff':('payoff','Fair Value Settlement'),
            'reference_definition':('reference_definition',"reference team's final score minus opposing team's final score"),
            'effective_from':('edition_effective_from','Effective for this rule edition from 2026-10-01T00:00:00Z'),
            'effective_until':('edition_effective_until','Effective for this rule edition until 2026-10-08T00:00:00Z')}}
    for name, fields in specs.items():
        receipt = deepcopy(p['objects'].get(name,p['objects']['listing'])['receipt'])
        receipt.update(source_id='SYNTHETIC-original:'+name,endpoint='https://synthetic.invalid/'+name,format='text',provider_clock_meaning='attributed_document_no_publication_clock')
        text = 'SYNTHETIC attributable document; no authentic source assertion.\n' + '\n'.join(k+': '+v[1] for k,v in fields.items())
        p['objects'][name] = bodies.retained(text.encode(),receipt=receipt)
        permission = deepcopy(p['permissions'].get(name,p['permissions']['listing']))
        permission.update(source_id=receipt['source_id'],endpoint=receipt['endpoint'])
        p['permissions'][name] = permission
    identities = {k:dict(body_sha256=o['body_sha256'],receipt_sha256=projection.digest(o['receipt'])) for k,o in p['objects'].items()}
    for name, fields in specs.items():
        _, text = bodies.read(p['objects'][name]); facts = {}; values = {}
        for key,(method,excerpt) in fields.items():
            start = text.index(key+': ')+len(key)+2
            support = dict(object=name,path=None,text=dict(start=start,end=start+len(excerpt),exact=excerpt),operation='identity')
            facts[key] = dict(supports=[support],method=method,status='DOCUMENTED' if method=='literal' else 'OWNER_INTERPRETED')
            values[key] = projection.normalize(method,excerpt)
        v = dict(version=projection.VERSION,method_version=projection.METHOD,original_objects=deepcopy(identities),facts=facts,values=values,
            applicability=dict(event=deepcopy(e),quote=deepcopy(q)),prepared_at='2026-10-06T19:25:02.500Z',review={})
        p['projections'][name] = projection.seal(v)
    p['version'] = reader.VERSION
    resign(p)
    return projection.seal(p), baseline


def resign(p, interpretation=True):
    from app_core import nfl_source_interpretation as projection, nfl_interpreted_research as reader
    if interpretation:
        for name in ('schedule','listing','clock_document','rules'):
            v = p['projections'][name]['payload']
            v['review'] = dict(version=projection.REVIEW,reviewer='Robert Velarde',reviewed_at='2026-10-06T19:25:03Z',
                subject_sha256=projection.subject(v),status='OWNER_REVIEWED',purpose='PRIVATE_RESEARCH',
                findings={k:dict(conclusion='VERIFIED',basis='SYNTHETIC substantive review: exact excerpt, field meaning, applicability and absence of conflicting overrides for '+k) for k in v['values']},
                limitations='SYNTHETIC review only. No real permission, independent acceptance, calibration, qualification or wagering authority.')
            p['projections'][name] = projection.seal(v)
    sign_new(p)
    p['owner_review']['version'] = 'nfl-owner-interpreted-verification-v4'
    p['owner_review']['subject_sha256'] = reader.subject(p)


@pytest.mark.parametrize('away,reference_away',[(False,False),(False,True),(True,False),(True,True)])
def test_prose_actual_reader_caller_capture_export_replay_and_browser(monkeypatch,tmp_path,away,reference_away):
    from app_core import nfl_interpreted_research as reader, nfl_private_routes as routes, research_replay
    from test_nfl_owner_research import actual_pipeline, private_app
    from test_source_contract_pipeline import exported
    from app_core.per_game_boards import per_game_board
    from streamlit.testing.v1 import AppTest
    packet, baseline = interpreted_fixture(monkeypatch,away,reference_away)
    original = deepcopy(packet)
    assert reader.checked(packet,INFERENCE)[0] == packet['payload']['features']
    with routes.selected([packet]):
        analysis = actual_pipeline(monkeypatch)
    i = int(away); row = analysis.iloc[i]
    assert row.ml_probability == baseline.iloc[i].ml_probability
    assert reader.diagnose(row)['status'] == 'COMPLETE'
    result = exported(monkeypatch,tmp_path,analysis)
    receipt = research_replay.retain_export(result['frames'],result['package'],result['card'],result['captured'],path=result['db'])
    saved,sources = research_replay.read_export(receipt['export_id'],path=result['db'])
    retained = research_replay.frame_from_payload(next(iter(sources.values()))['original']['producer'])
    assert routes.diagnose(retained.iloc[i])['status'] == 'COMPLETE'
    private = routes.private_display(retained.iloc[i])
    assert private['raw_probability'] == baseline.iloc[i].ml_probability
    assert private['interpreted_listing']['rule_edition'] == 'NFL-001 edition 2026-10-01'
    assert private['interpreted_listing']['adapter_identity'] != private['interpreted_listing']['rule_edition']
    assert private['operator_update_at'] is None and private['calibrated'] is False
    for family in ('overall','sides','totals'):
        board = per_game_board(result['card'],result['captured'],family=family,novig_only=True)
        assert len(board)==1
        assert all(r['status']=='PASS' and r['win_estimate'] is None and r['ev'] is None for r in saved['package']['games'][family])
    assert 'bytes_base64' not in json.dumps(saved['package']) and 'OWNER_REVIEWED' not in json.dumps(saved['package'])
    assert all(c['production_bet_amount']==0 for c in result['card'].wager_contract)
    at = AppTest.from_function(private_app); at.session_state['private_candidates']=retained;at.run()
    assert not at.exception and any(m.value==f'{row.ml_probability:.2%}' for m in at.metric)
    assert any('OWNER_REVIEWED / PRIVATE_RESEARCH' in m.value for m in at.markdown)
    assert any('external edition: NFL-001 edition 2026-10-01' in c.value for c in at.caption)
    assert packet == original


@pytest.mark.parametrize('attack',[
    'original','excerpt','locator','output','method','unknown','review_subject','empty_basis','late_review','future_preparation',
    'orientation','price','line','position','threshold','sid','applicability','edition','effective_filing_date','effective_gap','override','amendments',
    'contradictory_support','stale','future_quote','started','legacy_review','permission','features','model'])
def test_attacks_reject_before_native_inference(monkeypatch,attack):
    from app_core import nfl_interpreted_research as reader, nfl_private_routes as routes, nfl_source_interpretation as projection
    from app_core.research_estimate_trace import encode
    packet, baseline = interpreted_fixture(monkeypatch);p=deepcopy(packet['payload']);v=p['projections']['listing']['payload']
    field=v['facts']['period'];expected=deepcopy(p['model'])
    if attack=='original':p['objects']['listing']['body_sha256']='0'*64
    if attack=='excerpt':field['supports'][0]['text']['exact']='full game'
    if attack=='locator':field['supports'][0]['text']['start']+=1
    if attack=='output':v['values']['period']='first_half'
    if attack=='method':field['method']='arbitrary_owner_claim'
    if attack=='unknown':field['status']='UNKNOWN'
    if attack=='review_subject':v['review']['subject_sha256']='0'*64
    if attack=='empty_basis':v['review']['findings']['period']['basis']='Owner clicked'
    if attack=='late_review':v['review']['reviewed_at']='2026-10-06T19:26:00Z'
    if attack=='future_preparation':v['prepared_at']='2026-10-06T19:26:00Z'
    if attack=='orientation':p['event']['home'],p['event']['away']=p['event']['away'],p['event']['home']
    if attack=='price':p['quote']['price']+=1
    if attack=='line':p['quote']['point']+=1
    if attack in {'position','threshold','sid','applicability','edition','effective_gap','override','amendments'}:
        key,value={'position':('position','NO'),'threshold':('threshold',999.5),'sid':('feed_identity','unrelated'),
            'applicability':('applicability',{}),'edition':('rule_edition','NFL-001 edition 2026-10-09'),
            'effective_gap':('effective_from','2026-10-07T00:00:00Z'),'override':('period','first_half'),'amendments':('amendments',['unverified'])}[attack]
        v['values'][key]=value
    if attack=='effective_filing_date':v['facts']['effective_from']['method']='utc_instant'
    if attack=='contradictory_support':field['supports'].append(v['facts']['position']['supports'][0])
    if attack=='stale':p['quote']['recorded_at']='2026-10-06T18:00:00Z'
    if attack=='future_quote':p['quote']['recorded_at']='2026-10-06T20:00:00Z'
    if attack=='started':p['event']['start']='2026-10-06T19:00:00Z'
    if attack=='legacy_review':p['owner_review']['version']='nfl-owner-scoring-verification-v3'
    if attack=='permission':p['permissions']['rules']['private_research_use']=False
    if attack=='features':p['features'][p['feature_order'][0]]+=1
    if attack=='model':p['model']['configuration']['score_parameters']['reliability']=.9
    # Rehashing the outer packet/interpretation is not substantive review.
    p['projections']['listing']=projection.seal(v)
    calls=[];monkeypatch.setattr(reader,'model_binding',lambda:expected)
    monkeypatch.setattr('app_core.market_probability_model._normal_cdf',lambda x:calls.append(x) or .5)
    try:
        with routes.selected([projection.seal(p)]):result=reader.predict(baseline.iloc[0])
    except ValueError as exc:
        assert str(exc).startswith('NFL_PRIVATE_')
    else:
        assert result['ml_inference_status']=='unavailable',result
        assert result['ml_unavailable_reason'].startswith('NFL_PRIVATE_')
    assert calls==[]


def test_old_readers_and_cross_mode_review_remain_separate(monkeypatch):
    from app_core import nfl_scoring_research as old, nfl_interpreted_research as new, nfl_source_interpretation as projection
    packet,_=interpreted_fixture(monkeypatch)
    with pytest.raises(ValueError,match='NFL_PRIVATE_PACKET_SCHEMA'):old.checked(packet,INFERENCE)
    old_packet,_=scoring_fixture(monkeypatch,1024)
    assert old.checked(old_packet,INFERENCE)[0]==old_packet['payload']['features']
    with pytest.raises(ValueError,match='NFL_PRIVATE_PACKET_SCHEMA'):new.checked(old_packet,INFERENCE)
    p=deepcopy(packet['payload']);p['owner_review']=old_packet['payload']['owner_review']
    with pytest.raises(ValueError):new.checked(projection.seal(p),INFERENCE)


def replace_supported_text(p,name,key,excerpt):
    from app_core import nfl_observed_custody as bodies, nfl_source_interpretation as projection
    _,text=bodies.read(p['objects'][name]);f=p['projections'][name]['payload']['facts'][key]
    old=f['supports'][0]['text'];text=text[:old['start']]+excerpt+text[old['end']:]
    p['objects'][name]=bodies.retained(text.encode(),receipt=p['objects'][name]['receipt'])
    identities={k:dict(body_sha256=o['body_sha256'],receipt_sha256=projection.digest(o['receipt'])) for k,o in p['objects'].items()}
    for doc in ('schedule','listing','clock_document','rules'):
        v=p['projections'][doc]['payload'];v['original_objects']=deepcopy(identities)
        if doc==name:
            for field,fact in v['facts'].items():
                begin=text.index(field+': ')+len(field)+2;end=text.find('\n',begin)
                end=len(text) if end<0 else end;actual=text[begin:end]
                fact['supports'][0]['text']=dict(start=begin,end=end,exact=actual)
                v['values'][field]=projection.normalize(fact['method'],actual)
        p['projections'][doc]=projection.seal(v)
    resign(p)


@pytest.mark.parametrize('name,key,excerpt,code',[
    ('listing','period','first half','NFL_PRIVATE_PERIOD_RULES_UNSUPPORTED'),
    ('listing','overtime','excluding overtime','NFL_PRIVATE_PERIOD_RULES_UNSUPPORTED'),
    ('listing','rule_edition','Rule edition: NFL-001 edition 2026-10-09.','NFL_PRIVATE_RULE_EDITION_CONFLICT'),
    ('rules','effective_from','Effective for this rule edition from 2026-10-07T00:00:00Z','NFL_PRIVATE_RULE_EDITION_CONFLICT'),
    ('listing','effective_until','Effective for this listing until 2026-10-06T19:25:00Z','NFL_PRIVATE_PERIOD_RULES_UNSUPPORTED'),
    ('listing','threshold','999.5','NFL_PRIVATE_POSITION_MAPPING_CONFLICT'),
    ('listing','position','NO','NFL_PRIVATE_POSITION_MAPPING_CONFLICT'),
    ('listing','feed_identity','SYNTHETIC-unrelated','NFL_PRIVATE_LISTING_APPLICABILITY_MISSING'),
    ('listing','applicability','Listing SYNTHETIC-NFLSPR-123 governs provider event unrelated for Dallas Cowboys versus Buffalo Bills.','NFL_PRIVATE_LISTING_APPLICABILITY_MISSING'),
    ('rules','ticker','UNRELATED','NFL_PRIVATE_RULE_EDITION_CONFLICT'),
    ('schedule','home','Seattle Seahawks','NFL_PRIVATE_EVENT_MAPPING_CONFLICT'),
])
def test_consistently_rehashed_and_reviewed_wrong_facts_still_reject(monkeypatch,name,key,excerpt,code):
    from app_core import nfl_interpreted_research as reader, nfl_source_interpretation as projection
    packet,baseline=interpreted_fixture(monkeypatch);p=deepcopy(packet['payload'])
    replace_supported_text(p,name,key,excerpt)
    calls=[];expected=deepcopy(p['model'])
    monkeypatch.setattr(reader,'model_binding',lambda:expected)
    monkeypatch.setattr('app_core.market_probability_model._normal_cdf',lambda x:calls.append(x) or .5)
    with reader.selected([projection.seal(p)]):result=reader.predict(baseline.iloc[0])
    assert result['ml_unavailable_reason']==code,result
    assert calls==[]


@pytest.mark.parametrize('method,text',[
    ('effective_from','Filed on 2026-10-01T00:00:00Z'),
    ('quote_meaning',"when the operator originally published the price"),
    ('edition','Rule edition: arbitrary new rules.'),
    ('payoff','Automatic refund for all undecided games'),
    ('amendments','Amendment 2026-10-09 may apply'),
    ('applicability','This generic contract template applies to football markets'),
])
def test_unsupported_prose_cannot_be_reviewed_into_supported_facts(method,text):
    from app_core.nfl_source_interpretation import normalize
    with pytest.raises(ValueError):normalize(method,text)
