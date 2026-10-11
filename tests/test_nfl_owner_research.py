"""Labelled SYNTHETIC responses only; actual caller/capture/export/private browser."""
import base64
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import pandas as pd
import pytest
from app_core import nfl_owner_research as owner, nfl_response_custody as custody
from app_core import nfl_inference_evidence as legacy, source_contract, research_replay
from app_core.research_estimate_trace import encode
from scripts.benchmark_drive_history_loading import blocked_network
from test_nfl_native_provenance import native_pipeline, schedule
from test_source_contract_pipeline import exported, INFERENCE, NOW
from app_core import feature_processing as fp
ORIGINAL_FETCH = fp.fetch_nfl_stats


def actual_pipeline(monkeypatch):
    monkeypatch.setattr(fp, 'fetch_nfl_stats', ORIGINAL_FETCH)
    return native_pipeline(monkeypatch)


@pytest.fixture(autouse=True)
def offline():
    with blocked_network():
        yield


def packet_fixture(monkeypatch):
    baseline = actual_pipeline(monkeypatch)
    row = baseline.iloc[0].to_dict()
    from app_core import producer_provenance as producer
    q = deepcopy(producer._matches(row)[0])
    # Only original provider identity fields; listing declarations are separate.
    q = {k:q.get(k) for k in ('book', 'market_type', 'point', 'price', 'recorded_at', 'provider_event_id', 'provider_namespace', 'provider_quote_id')}
    event = dict(canonical_event_id='synthetic-independent-schedule:2026:BUF:DAL', provider_namespace=q['provider_namespace'],
        provider_event_id=q['provider_event_id'], home='Dallas', away='Buffalo', start=row['game_start_utc'], season=2026, neutral_site=False)
    game = dict(id=q['provider_event_id'], sport_key='americanfootball_nfl', home_team='Dallas Cowboys', away_team='Buffalo Bills', commence_time=event['start'],
        bookmakers=[dict(key=q['book'], markets=[dict(key='spreads', last_update=q['recorded_at'], outcomes=[
            dict(name='Dallas Cowboys' if q['market_type']=='spread_home' else 'Buffalo Bills', point=q['point'], price=q['price'])])])])
    listing = dict(event=event, quote=q, listing_id='synthetic-exact-listing', product='synthetic-research-product', jurisdiction='synthetic-jurisdiction',
        period='full_game', overtime=True, payoff='novig_fvs_decided_game', rule_edition=source_contract.RULES,
        effective_from='2026-10-01T00:00:00Z', effective_until='2026-10-08T00:00:00Z', source_reference='SYNTHETIC original offer-specific listing')
    clocks = dict(provider_quote_field='markets.last_update', quote_meaning='provider_market_last_update', publisher_field='source_available_at',
        publisher_meaning='publisher_available_at_per_record', source_reference='SYNTHETIC clock documentation')
    rows = schedule().iloc[:12].to_dict('records') # manufactured complete historical-score response, not a filtered authentic body
    for r in rows:
        r['source_available_at']='2026-10-06T19:19:00Z'
    csv_body=io.StringIO(newline=''); writer=csv.DictWriter(csv_body,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    bodies=dict(schedule=encode([event]).encode(), odds=encode([game]).encode(), scores=csv_body.getvalue().encode(),
        listing=encode(listing).encode(), clock_document=encode(clocks).encode())
    objects={}
    for k, raw in bodies.items():
        receipt=dict(source_id='synthetic-original:'+k, endpoint='https://synthetic.invalid/'+k, format='csv' if k=='scores' else 'json', evidence_label='SYNTHETIC',
            request_started_at='2026-10-06T19:25:00Z', received_at='2026-10-06T19:25:01Z', observed_at='2026-10-06T19:25:02Z',
            observation_meaning='actual_local_first_observation_of_complete_decoded_body',
            provider_clock_meaning='publisher_available_at_per_record' if k=='scores' else 'provider_market_last_update' if k=='odds' else 'official_schedule_effective_at')
        objects[k]=custody.retained(raw, receipt=receipt)
    permissions={k:dict(reviewer='Robert Velarde', reviewed_at='2026-10-06T19:00:00Z', effective_from='2026-10-01T00:00:00Z',
        effective_until='2026-10-08T00:00:00Z', source_id=o['receipt']['source_id'], endpoint=o['receipt']['endpoint'], private_retention=True, private_research_use=True,
        credential_exclusion=True, basis='SYNTHETIC advance permission; no authentic license assertion.') for k,o in objects.items()}
    p=dict(version=owner.VERSION, evidence_label='SYNTHETIC', event=event, quote=q, objects=objects,
        projections=dict(schedule_path=[0], odds_path=[0], listing_path=[], clock_document_path=[]), permissions=permissions,
        owner_review={}, model=owner.model_binding(), features={k:float(json.loads(row['ml_estimate_metadata'])['nfl_inputs']['payload']['features'][k]['value']) for k in legacy.FEATURES},
        feature_order=list(legacy.FEATURES), feature_available_at='2026-10-06T19:25:03Z', original_inference_time=None)
    sign(p)
    monkeypatch.setattr(source_contract,'ACCEPTED_LISTINGS',{})
    return dict(payload=p,sha256=owner.digest(p)), baseline


def sign(p):
    p['owner_review']=dict(version='nfl-owner-verification-v1', reviewer='Robert Velarde', reviewed_at='2026-10-06T19:25:04Z',
        subject_sha256=owner.subject(p), status='OWNER_REVIEWED', purpose='PRIVATE_RESEARCH',
        attestation='I verified the original evidence and applicable facts for this exact subject.', findings={k:dict(conclusion='VERIFIED',
        basis='SYNTHETIC substantive exact original evidence finding for '+k) for k in
        ('event_mapping','offer_and_settlement','permissions','clocks_and_original_custody','feature_provenance','model_binding')})


def rehash(packet, change, *, resign=False):
    p=deepcopy(packet['payload']);change(p)
    if resign:sign(p)
    return dict(payload=p,sha256=owner.digest(p))


def body_change(p, name, change):
    obj=p['objects'][name]; raw, decoded=custody.read(obj);change(decoded)
    raw=encode(decoded).encode()
    p['objects'][name]=custody.retained(raw,receipt=obj['receipt'])


@pytest.mark.parametrize('selected_index',[0,1])
def test_actual_normal_caller_capture_export_replay_and_private_browser(monkeypatch,tmp_path,selected_index):
    packet, baseline=packet_fixture(monkeypatch)
    if selected_index:
        packet=opposite_packet(packet)
    original=deepcopy(packet)
    owner.checked(packet,INFERENCE)
    with owner.selected([packet]):
        analysis=actual_pipeline(monkeypatch)
    # native_pipeline's synthetic independent listing fixture is not borrowed by owner review.
    assert source_contract.ACCEPTED_LISTINGS # synthetic helper fixture only
    assert analysis.iloc[selected_index].ml_probability==baseline.iloc[selected_index].ml_probability
    assessed=owner.diagnose(analysis.iloc[selected_index])
    assert assessed['status']=='COMPLETE', assessed
    assert owner.diagnose(analysis.iloc[1-selected_index])['reason']=='NFL_PRIVATE_EXACT_OFFER_NOT_SELECTED'
    result=exported(monkeypatch,tmp_path,analysis)
    receipt=research_replay.retain_export(result['frames'],result['package'],result['card'],result['captured'],path=result['db'])
    saved,sources=research_replay.read_export(receipt['export_id'],path=result['db'])
    retained=research_replay.frame_from_payload(next(iter(sources.values()))['original']['producer'])
    assert owner.diagnose(retained.iloc[selected_index])['status']=='COMPLETE'
    assert json.loads(retained.iloc[selected_index].ml_estimate_metadata)['nfl_private_inputs']['payload']['original_packet']==packet
    raw=analysis.iloc[selected_index].ml_probability
    from app_core.per_game_boards import per_game_board
    frames=[per_game_board(result['card'],result['captured'],family=f,novig_only=True) for f in ('overall','sides','totals')]
    for frame in frames[:2]:
        displays=[json.loads(v) for v in frame.nfl_private_research_display]
        assert any(d['raw_probability']==raw and d['review_status']=='OWNER_REVIEWED' and d['purpose']=='PRIVATE_RESEARCH' for d in displays), displays
        assert all(d['ev'] is None and d['edge'] is None for d in displays)
    from app_core.public_board import build_package
    assert build_package(*frames)==saved['package']
    public=encode(saved['package'])
    assert 'bytes_base64' not in public and 'synthetic-original' not in public and 'OWNER_REVIEWED' not in public
    for family in ('overall','sides','totals'):
        assert all(r['status']=='PASS' and r['win_estimate'] is None and r['ev'] is None
            and r['research_display']['probability'] is None for r in saved['package']['games'][family])
        if family in {'overall','sides'}:
            assert all(r['pick']=='' and r['odds'] is None for r in saved['package']['games'][family])
        else:
            assert all(r['odds'] is None and r['pick'] != result['frames'][0].iloc[0]['pick'] for r in saved['package']['games'][family])
    assert not result['captured'].production_eligible.fillna(False).any()
    assert all(c['production_bet_amount']==0 for c in result['card'].wager_contract)
    from streamlit.testing.v1 import AppTest
    at=AppTest.from_function(private_app)
    at.session_state['private_candidates']=retained
    at.run()
    assert not at.exception and any(m.value==f'{raw:.2%}' for m in at.metric)
    assert any('OWNER_REVIEWED / PRIVATE_RESEARCH' in m.value for m in at.markdown)
    assert packet==original
    changed=result['captured'].copy(); changed['export_run_id']='synthetic-conflicting-run'
    with pytest.raises(ValueError,match='snapshot/run identity mismatch'):
        research_replay.retain_export(result['frames'],result['package'],result['card'],changed,path=result['db'])


def private_app():
    import streamlit as st
    from app.ui.nfl_private_research import render
    render(st.session_state['private_candidates'])


@pytest.mark.parametrize('mutation,reason',[
    ('missing_custody','NFL_PRIVATE_CUSTODY_MISSING'),('corrupt','NFL_PRIVATE_CUSTODY_CORRUPT'),
    ('price','NFL_PRIVATE_OFFER_IDENTITY_CONFLICT'),('line','NFL_PRIVATE_OFFER_IDENTITY_CONFLICT'),
    ('integer','NFL_PRIVATE_INTEGER_PUSH_UNVALIDATED'),('publisher','NFL_PRIVATE_PUBLISHER_AVAILABILITY_MISSING'),
    ('future_dependency','NFL_PRIVATE_DEPENDENCY_CLOCK_CONFLICT'),('feature','NFL_PRIVATE_FEATURE_DERIVATION_CONFLICT'),
    ('order','NFL_PRIVATE_FEATURE_ORDER_CONFLICT'),('model','NFL_PRIVATE_MODEL_BINDING_CONFLICT'),
    ('review','NFL_PRIVATE_OWNER_REVIEW_CONFLICT'),('late_review','NFL_PRIVATE_OWNER_REVIEW_CLOCK_CONFLICT'),
    ('independent','NFL_PRIVATE_OWNER_REVIEW_MISSING'),('historical','NFL_PRIVATE_PACKET_SCHEMA'),
    ('orientation','NFL_PRIVATE_EVENT_MAPPING_CONFLICT'),('listing','NFL_PRIVATE_PERIOD_RULES_LISTING_MISSING'),
    ('rights','NFL_PRIVATE_PERMISSION_MISSING'),('analysis_rights','NFL_PRIVATE_PERMISSION_MISSING'),('late_permission','NFL_PRIVATE_PERMISSION_CLOCK_CONFLICT'),
    ('clocks','NFL_PRIVATE_CUSTODY_CLOCK_UNKNOWN'),('neutral','NFL_PRIVATE_EVENT_MAPPING_CONFLICT'),
])
def test_rejections_before_numerical_inference(monkeypatch,mutation,reason):
    packet,baseline=packet_fixture(monkeypatch)
    def change(p):
        if mutation=='missing_custody':p['objects'].pop('scores')
        if mutation=='corrupt':p['objects']['scores']['body_sha256']='0'*64
        if mutation=='price':p['quote']['price']+=1
        if mutation=='line':p['quote']['point']+=1
        if mutation=='integer':p['quote']['point']=3
        if mutation=='publisher':
            obj=p['objects']['scores'];raw,_=custody.read(obj);raw=raw.replace(b'source_available_at',b'unrecorded_clock')
            p['objects']['scores']=custody.retained(raw,receipt=obj['receipt'])
        if mutation=='future_dependency':
            obj=p['objects']['scores'];raw,_=custody.read(obj);raw=raw.replace(b'19:19:00',b'23:19:00')
            p['objects']['scores']=custody.retained(raw,receipt=obj['receipt'])
        if mutation=='feature':p['features'][legacy.FEATURES[0]]+=1
        if mutation=='order':p['feature_order'].reverse()
        if mutation=='model':p['model']['configuration']['score_parameters']['reliability']=.9
        if mutation=='review':p['owner_review']['findings']['feature_provenance']['conclusion']='UNKNOWN'
        if mutation=='late_review':p['owner_review']['reviewed_at']='2026-10-06T19:26:00Z'
        if mutation=='independent':p['owner_review']['status']='INDEPENDENT_ACCEPTANCE'
        if mutation=='historical':p['original_inference_time']=INFERENCE
        if mutation=='orientation':p['event']['home'],p['event']['away']=p['event']['away'],p['event']['home']
        if mutation=='listing':body_change(p,'listing',lambda x:x.pop('product'))
        if mutation=='rights':p['permissions']['scores']['private_retention']=False
        if mutation=='analysis_rights':p['permissions']['odds']['private_research_use']=False
        if mutation=='late_permission':p['permissions']['scores']['reviewed_at']='2026-10-06T19:25:03Z'
        if mutation=='clocks':p['objects']['scores']['receipt']['provider_clock_meaning']='UNKNOWN'
        if mutation=='neutral':p['event']['neutral_site']=True
    broken=rehash(packet,change,resign=mutation not in {'review','late_review','independent'})
    calls=[]
    monkeypatch.setattr('app_core.market_probability_model._normal_cdf',lambda x:calls.append(x) or .5)
    # Maintain valid consumed model identity for this spy; otherwise reader correctly stops earlier.
    expected=deepcopy(packet['payload']['model'])
    monkeypatch.setattr(owner,'model_binding',lambda:expected)
    if mutation in {'corrupt','clocks','historical'}:
        with pytest.raises(ValueError,match=reason),owner.selected([broken]):pass
    else:
        with owner.selected([broken]):result=owner.predict(baseline.iloc[0])
        assert result['ml_unavailable_reason']==reason, result
    assert calls==[]


@pytest.mark.parametrize('at',['2026-10-06T19:23:00Z','2026-10-06T20:30:00Z','2026-10-06T23:00:00Z'])
def test_future_stale_started_quotes(monkeypatch,at):
    p,baseline=packet_fixture(monkeypatch)
    monkeypatch.setattr('app_core.research_estimate_trace.generated_time',lambda:at)
    with owner.selected([p]):result=owner.predict(baseline.iloc[0])
    assert result['ml_unavailable_reason']=='NFL_PRIVATE_QUOTE_CLOCK_CONFLICT'


def test_no_default_activation_duplicates_and_legacy_reader(monkeypatch):
    packet,baseline=packet_fixture(monkeypatch)
    assert owner.selection_requested() is False
    assert legacy.diagnose(baseline.iloc[0])['status']=='INCOMPLETE' # independent catalog deliberately empty
    with pytest.raises(ValueError,match='NFL_PRIVATE_PACKET_AMBIGUOUS'),owner.selected([packet,packet]):pass
    with owner.selected([packet]):analysis=actual_pipeline(monkeypatch)
    assert legacy.diagnose(analysis.iloc[0])['status']=='UNKNOWN' # separate route, not accepted legacy packet
    assert owner.selection_requested() is False
    assert owner.load(encode(packet).encode())==packet
    with pytest.raises(ValueError):
        owner.load(encode(packet).encode(),owner_upload=True)


def test_limits_secret_echo_and_static_probability_tampering(monkeypatch):
    packet,baseline=packet_fixture(monkeypatch)
    receipt=packet['payload']['objects']['listing']['receipt']
    with pytest.raises(ValueError,match='NFL_PRIVATE_RESPONSE_SIZE'):custody.retained(b'x'*(custody.MAX_BODY+1),receipt=receipt)
    with pytest.raises(ValueError,match='NFL_PRIVATE_CREDENTIAL_FORBIDDEN'):custody.retained(b'{"api_key":"synthetic-secret"}',receipt=receipt)
    with owner.selected([packet]):analysis=actual_pipeline(monkeypatch)
    row=analysis.iloc[0].to_dict(); row['ml_probability']+=.01
    assert owner.diagnose(row)['reason']=='NFL_PRIVATE_PROBABILITY_CONFLICT'


def test_private_ui_refresh_keeps_raw_and_original_stages(monkeypatch):
    packet,_=packet_fixture(monkeypatch)
    with owner.selected([packet]):analysis=actual_pipeline(monkeypatch)
    before=deepcopy(analysis)
    after=deepcopy(analysis)
    after.loc[0,'calibrated_probability']=.53
    after.loc[0,'expected_value']=.02
    monkeypatch.setattr(legacy,'ui_reblend_time',lambda:'2026-10-06T19:25:51Z')
    owner.retain_ui_refresh(before,after,{'p_ml':analysis.ml_probability})
    first=json.loads(before.iloc[0].ml_estimate_metadata)['nfl_private_inputs']['payload']
    saved=json.loads(after.iloc[0].ml_estimate_metadata)['nfl_private_inputs']['payload']
    assert saved['raw_probability']==first['raw_probability']
    assert saved['original_blend']==first['original_blend']
    assert len(saved['ui_refresh'])==1 and saved['ui_refresh'][0]['payload']['probability']['value']==.53
    assert owner.diagnose(after.iloc[0])['status']=='COMPLETE'
    after.loc[0,'calibrated_probability']=.54
    assert owner.diagnose(after.iloc[0])['reason']=='NFL_PRIVATE_PROBABILITY_CONFLICT'


def test_actual_private_browser_token_gate_and_no_staging_inference(monkeypatch,tmp_path):
    packet,_=packet_fixture(monkeypatch)
    with owner.selected([packet]):analysis=actual_pipeline(monkeypatch)
    result=exported(monkeypatch,tmp_path,analysis)
    monkeypatch.setenv('PARLAYPICKER_PUBLISH_TOKEN','synthetic-owner-token-0000')
    monkeypatch.setenv('PARLAYPICKER_NETLIFY_SITE_ID','')
    monkeypatch.setenv('PARLAYPICKER_EVIDENCE_DIR',str(tmp_path))
    monkeypatch.setattr('app.ui.public_results.render_history',lambda *a,**k:[])
    from streamlit.testing.v1 import AppTest
    at=AppTest.from_function(gated_app)
    at.session_state['private_games']=result['card'];at.session_state['private_candidates']=result['captured']
    at.run()
    assert not at.exception and not at.metric
    at.text_input(key='publication_token').set_value('wrong').run()
    assert not at.exception and not at.metric
    at.text_input(key='publication_token').set_value('synthetic-owner-token-0000').run()
    assert not at.exception and any(m.label=='Raw NFL research probability' for m in at.metric)
    assert at.session_state['nfl_private_packets']==[]


def gated_app():
    import streamlit as st
    from app.ui.publish_panel import render_publish_panel
    render_publish_panel(st.session_state['private_games'],st.session_state['private_candidates'])


@pytest.mark.parametrize('side',['home','away'])
def test_both_signed_selected_sides_unchanged_and_no_cross_mode_acceptance(monkeypatch,side):
    p,baseline=packet_fixture(monkeypatch)
    if side=='away':
        p=opposite_packet(p)
    row=baseline.iloc[0 if side=='home' else 1]
    from app_core.market_probability_model import predict_market_probabilities
    with owner.selected([p]):result=predict_market_probabilities(pd.DataFrame([row])).iloc[0]
    assert result.ml_inference_status=='success' and result.ml_probability==row.ml_probability
    assert legacy.diagnose(result)['status']=='UNKNOWN'
    assert source_contract.ACCEPTED_LISTINGS=={}


def opposite_packet(p):
    q=deepcopy(p['payload']['quote']);q.update(market_type='spread_away',point=-q['point'],price=-104)
    def change(x):
        x['quote']=q
        body_change(x,'odds',lambda games:games[0]['bookmakers'][0]['markets'][0]['outcomes'][0].update(name='Buffalo Bills',point=q['point'],price=q['price']))
        body_change(x,'listing',lambda listing:listing.update(quote=q))
    return rehash(p,change,resign=True)


@pytest.mark.parametrize('case,reason',[
    ('ambiguous','NFL_PRIVATE_EVENT_MAPPING_CONFLICT'),('different_event','NFL_PRIVATE_EVENT_MAPPING_CONFLICT'),
    ('no_history','NFL_PRIVATE_HISTORY_MISSING'),('rule','NFL_PRIVATE_PERIOD_RULES_UNSUPPORTED'),
    ('future_observation','NFL_PRIVATE_CUSTODY_CLOCK_CONFLICT'),('early_review','NFL_PRIVATE_OWNER_REVIEW_CLOCK_CONFLICT'),
])
def test_rehashed_identity_history_and_chronology_attacks(monkeypatch,case,reason):
    p,baseline=packet_fixture(monkeypatch)
    def change(x):
        if case=='ambiguous':x['event']['home']='New York'
        if case=='different_event':body_change(x,'schedule',lambda rows:rows[0].update(provider_event_id='unrelated'))
        if case=='no_history':
            raw,_=custody.read(x['objects']['scores']);raw=raw.replace(b'DAL',b'KC')
            x['objects']['scores']=custody.retained(raw,receipt=x['objects']['scores']['receipt'])
        if case=='rule':body_change(x,'listing',lambda v:v.update(rule_edition='unverified-successor'))
        if case=='future_observation':x['objects']['scores']['receipt']['observed_at']='2026-10-06T19:26:00Z'
    broken=rehash(p,change,resign=True)
    if case=='early_review':broken=rehash(broken,lambda v:v['owner_review'].update(reviewed_at='2026-10-06T19:25:02Z'))
    with owner.selected([broken]):result=owner.predict(baseline.iloc[0])
    assert result['ml_unavailable_reason']==reason
    assert not pd.notna(result['ml_probability'])
    assert json.loads(result['ml_estimate_metadata'])['nfl_private_inputs']['payload']['review_status']=='UNVERIFIED'


def test_unprofiled_synthetic_clock_isolation_has_exact_scope():
    from scripts.nfl_private_synthetic_clock import applies,TARGET
    assert applies(TARGET+'[spread_home]')
    assert not applies('tests/test_nfl_owner_research.py::test_future_stale_started_quotes[x]')
    assert not applies('tests/test_ncaaf_pilot.py::test_stale_quote')
    assert not applies('tests/test_ncaaf_pilot.py::test_started_game')


def test_private_rejection_preserves_independent_slate_and_public_placeholders(monkeypatch):
    from app_core.slate_coverage import build_coverage, publication_rows
    from app_core.per_game_boards import per_game_board
    from app_core.public_board import build_package
    packet, baseline = packet_fixture(monkeypatch)
    broken = rehash(packet, lambda p: p['objects'].pop('scores'), resign=True)
    with owner.selected([broken]):
        rejected = actual_pipeline(monkeypatch)
    e = packet['payload']['event']
    # Independent labelled schedule denominator includes an entirely unquoted game.
    events = [dict(canonical_event_id=e['canonical_event_id'], home_team=e['home'], away_team=e['away'],
                   original_start=e['start'], schedule_status='SCHEDULED'),
              dict(canonical_event_id='SYNTHETIC-independent-no-odds', home_team='Atlanta', away_team='Seattle',
                   original_start=e['start'], schedule_status='SCHEDULED')]
    inventory = dict(version='slate-inventory-v1', league='NFL', source='SYNTHETIC independent schedule',
                     evidence_label='SYNTHETIC', selected_date='2026-10-06', timezone='America/New_York',
                     observed_at=INFERENCE, status='COMPLETE', completeness_basis='SYNTHETIC complete index',
                     reasons=[], events=events)
    coverage = build_coverage([inventory], selected_date='2026-10-06', as_of=INFERENCE,
                              run_id='SYNTHETIC-independent-run', candidates=rejected)
    decision = next(d for d in coverage['decisions'] if d['canonical_event_id'] == e['canonical_event_id'])
    assert decision['coverage_decision_state'] == 'UNVERIFIED'
    assert 'NFL_PRIVATE_CUSTODY_MISSING' in decision['blocker_codes']
    board, missing = publication_rows(pd.DataFrame(), rejected, coverage)
    assert len(board) == len(events)
    frames = [per_game_board(board, rejected, family=f) for f in ('overall', 'sides', 'totals')]
    package = build_package(*frames)
    for family in ('overall', 'sides', 'totals'):
        rows = package['games'][family]
        assert len(rows) == len(events)
        assert {r['coverage_decision']['canonical_event_id'] for r in rows} == {e['canonical_event_id'] for e in events}
        assert all(r['status'] == 'PASS' and r['win_estimate'] is None and r['odds'] is None and r['ev'] is None for r in rows)
    assert 'bytes_base64' not in encode(package)


def test_full_original_body_retains_unscored_future_game_without_invented_clock(monkeypatch):
    packet, baseline = packet_fixture(monkeypatch)
    raw, rows = custody.read(packet['payload']['objects']['scores'])
    pending = dict(rows[0], game_id='SYNTHETIC-future-unscored', gameday='2026-10-08',
                   home_score='', away_score='', result='', source_available_at='')
    full = io.StringIO(newline=''); writer = csv.DictWriter(full, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows([*rows, pending])
    body = full.getvalue().encode()
    updated = rehash(packet, lambda p:p['objects'].update(scores=custody.retained(body, receipt=p['objects']['scores']['receipt'])), resign=True)
    with owner.selected([updated]):
        analysis = actual_pipeline(monkeypatch)
    assert analysis.iloc[0].ml_probability == baseline.iloc[0].ml_probability
    saved = json.loads(analysis.iloc[0].ml_estimate_metadata)['nfl_private_inputs']['payload']['original_packet']
    original, decoded = custody.read(saved['payload']['objects']['scores'])
    assert original == body and decoded[-1] == pending and decoded[-1]['source_available_at'] == ''
    assert owner.diagnose(analysis.iloc[0])['status'] == 'COMPLETE'


@pytest.mark.parametrize('field',['original_raw_probability','previous_probability','consumed_blend'])
def test_rehashed_ui_receipt_cannot_borrow_probability_or_blend_binding(monkeypatch, field):
    packet,_ = packet_fixture(monkeypatch)
    with owner.selected([packet]):
        analysis = actual_pipeline(monkeypatch)
    after = deepcopy(analysis)
    monkeypatch.setattr(legacy,'ui_reblend_time',lambda:'2026-10-06T19:25:51Z')
    owner.retain_ui_refresh(analysis,after,{'p_ml':analysis.ml_probability})
    row = after.iloc[0].to_dict()
    item = json.loads(row['ml_estimate_metadata']); saved=item['nfl_private_inputs']; receipt=saved['payload']['ui_refresh'][0]
    receipt['payload'][field] = {'SYNTHETIC':'unrelated binding'} if field=='consumed_blend' else owner.fact(.01)
    receipt['sha256']=owner.digest(receipt['payload']);saved['sha256']=owner.digest(saved['payload'])
    row['ml_estimate_metadata']=encode(item)
    assert owner.diagnose(row)['reason']=='NFL_PRIVATE_PROBABILITY_CONFLICT'


def test_ui_refresh_chain_cannot_move_backwards_with_valid_hashes(monkeypatch):
    packet,_=packet_fixture(monkeypatch)
    with owner.selected([packet]):
        analysis=actual_pipeline(monkeypatch)
    first=deepcopy(analysis); second=deepcopy(analysis)
    monkeypatch.setattr(legacy,'ui_reblend_time',lambda:'2026-10-06T19:25:51Z')
    owner.retain_ui_refresh(analysis,first,{'p_ml':analysis.ml_probability})
    second=deepcopy(first)
    monkeypatch.setattr(legacy,'ui_reblend_time',lambda:'2026-10-06T19:25:50Z')
    owner.retain_ui_refresh(first,second,{'p_ml':first.ml_probability})
    assert owner.diagnose(second.iloc[0])['reason']=='NFL_PRIVATE_QUOTE_CLOCK_CONFLICT'


@pytest.mark.parametrize('field',['listing_id','product','jurisdiction','source_reference'])
def test_explicit_unknown_listing_fact_cannot_be_promoted_by_owner_review(monkeypatch,field):
    packet,baseline=packet_fixture(monkeypatch)
    changed=rehash(packet,lambda p:body_change(p,'listing',lambda listing:listing.update({field:'UNKNOWN'})),resign=True)
    with owner.selected([changed]):
        result=owner.predict(baseline.iloc[0])
    assert result['ml_unavailable_reason']=='NFL_PRIVATE_PERIOD_RULES_LISTING_MISSING'
    assert not pd.notna(result['ml_probability'])


@pytest.mark.parametrize('field,reason',[('history_id','NFL_PRIVATE_DEPENDENCY_IDENTITY_CONFLICT'),('quote_id','NFL_PRIVATE_OFFER_IDENTITY_CONFLICT')])
def test_unknown_evidence_identifiers_reject_before_numerical_call(monkeypatch,field,reason):
    packet,baseline=packet_fixture(monkeypatch)
    def change(p):
        if field=='history_id':
            obj=p['objects']['scores'];raw,rows=custody.read(obj)
            raw=raw.replace(rows[0]['game_id'].encode(),b'UNKNOWN',1)
            p['objects']['scores']=custody.retained(raw,receipt=obj['receipt'])
        else:
            p['quote']['provider_quote_id']='UNKNOWN'
            body_change(p,'odds',lambda games:games[0]['bookmakers'][0]['markets'][0]['outcomes'][0].update(quote_id='UNKNOWN'))
            body_change(p,'listing',lambda listing:listing['quote'].update(provider_quote_id='UNKNOWN'))
    broken=rehash(packet,change,resign=True)
    with owner.selected([broken]):result=owner.predict(baseline.iloc[0])
    assert result['ml_unavailable_reason']==reason
    assert not pd.notna(result['ml_probability'])
