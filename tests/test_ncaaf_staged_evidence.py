"""SYNTHETIC original capture -> joined review -> native caller/private replay."""
import base64
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import pytest
from app_core import ncaaf_staged_acquisition as stage, ncaaf_staged_evidence as evidence
from app_core import ncaaf_owner_staged_research as owner, ncaaf_owner_event_crosswalk as aliases
from app_core import ncaaf_model_compatibility as model, ncaaf_response_custody as custody
from app_core import ncaaf_pipeline_evidence as adapter, ncaaf_prospective_chronology as chronology
from app_core.ncaaf_compatible_observation import mapping_binding
from scripts.benchmark_drive_history_loading import blocked_network
from test_ncaaf_model_compatibility import synthetic
from test_ncaaf_staged_acquisition import prepared, plan, capture, odds, bodies, Clock, TARGET

NOW = datetime(2026, 10, 16, 18, 30, 5, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def offline():
    with blocked_network(): yield


def alias_review(pref):
    p = pref['payload']; raw = odds(-7)
    v = dict(version=aliases.VERSION, discovery=dict(body_b64=base64.b64encode(raw).decode(), body_sha256=hashlib.sha256(raw).hexdigest(), received_at='2026-10-16T17:00:01Z'),
        prior_qualified_review=stage.seal(dict(version='SYNTHETIC-qualified-answer',finding='SYNTHETIC comparison basis; no personal technical audit, acceptance or selected team.')),
        source_references=[dict(source='SYNTHETIC official school comparison', sha256='a'*64, locator='SYNTHETIC named schedule record', finding='SYNTHETIC documented school identity and orientation comparison')],
        event=p['event'], schedule=p['schedule'], crosswalk=[dict(provider_name=provider, schedule_name=school, team_id=tid) for provider, school, tid in
            [('Tulane Green Wave', 'Tulane', 2655), ('Memphis Tigers', 'Memphis', 235)]], owner_review={})
    v['owner_review'] = dict(reviewer=owner.OWNER, reviewed_at='2026-10-16T18:02:00Z', subject_sha256=aliases.subject(v),
        finding='SYNTHETIC owner verified the exact aliases against original source and orientation evidence.',
        qualifications='SYNTHETIC comparison basis; no independent personal technical audit; no offer acceptance.', prior_review_sha256=v['prior_qualified_review']['sha256'])
    return v


def resign(packet):
    p = packet['payload']; c = p['custody_packet']; n = c['payload']['native_packet']; o = n['payload']['observation']['payload']; r = o['source_review']; q = o['quote']
    r['quote_observation']['quote_sha256'] = model.digest(q)
    r['offer_verification'].update(quote_sha256=model.digest(q), observation_sha256=model.digest(r['quote_observation']), terms_sha256=model.digest(r['terms_review']), mapping_sha256=model.digest(o['mapping_review']))
    r['owner_verification'].update(subject_version=owner.ReviewPolicy.OBSERVATION_VERSION, subject_sha256=chronology.subject_hash(o))
    n['payload']['observation']['sha256'] = model.digest(o); n['sha256'] = model.digest(n['payload'])
    d = p['dependency_review']; v = d['dependency_verification']; v.update(subject_sha256=chronology.dependency_subject_hash(n), dependency_hashes=[x['sha256'] for x in n['payload']['dependency_objects']], permissions_sha256=model.digest(d['permissions_review']))
    d['owner_verification'].update(subject_sha256=v['subject_sha256'], verification_sha256=model.digest(v))
    admission = c['payload']['custody_admission']; v = admission['verification']; v.update(subject_sha256=custody.subject_hash(c), permissions_sha256=model.digest(d['permissions_review']), terms_sha256=model.digest(r['terms_review']), native_verification_sha256=model.digest(d['dependency_verification']))
    admission['owner_verification'].update(subject_sha256=v['subject_sha256'], verification_sha256=model.digest(v))
    c['sha256'] = model.digest(c['payload'])
    p['owner_review'] = dict(version=owner.REVIEW_VERSION, review_id='SYNTHETIC-staged-owner-review', reviewer=owner.OWNER, reviewed_at='2026-10-16T18:30:04Z',
        review_status=owner.STATUS, use_scope=owner.SCOPE, subject_version=owner.SUBJECT_VERSION, subject_sha256=owner.subject_hash(packet), attestation=owner.ATTESTATION,
        conclusions={k:dict(conclusion='VERIFIED', evidence_sha256=v, finding='SYNTHETIC owner inspected the exact staged '+k+' evidence, not an actual human review.') for k,v in owner.evidence_subjects(packet).items()})
    packet['sha256'] = model.digest(p)


def fixture(synthetic, monkeypatch, root, kind='spread_home', line=-3.5):
    from test_ncaaf_owner_terms import fixture as template
    packet, row = template(synthetic, monkeypatch, kind, line)
    def migrate(v):
        if isinstance(v, dict): return {k:migrate(x) for k,x in v.items()}
        if isinstance(v, list): return [migrate(x) for x in v]
        if isinstance(v, str) and v.startswith('ncaaf-owner-') and v.endswith('-v2') and 'applicable-terms' not in v: return v[:-1]+'3'
        if isinstance(v, str) and v.startswith('2026-10-09T12:00:'): return v.replace('2026-10-09T12:00:', '2026-10-16T18:30:')
        return v
    packet = migrate(packet); p = packet['payload']; c = p['custody_packet']; n = c['payload']['native_packet']; o = n['payload']['observation']['payload']
    h, pref = prepared(synthetic, monkeypatch, root)
    clock = Clock(); clock.value = '2026-10-16T18:30:01Z'
    qbundle, _, _ = capture(monkeypatch, root, plan(stage.QUOTE_PLAN, h, pref), [odds(line if kind == 'spread_home' else -line)], clock, h, pref)
    p.update(version=owner.VERSION, staged_binding=evidence.join(h, pref, qbundle), alias_review=alias_review(pref))
    o.update(as_of='2026-10-16T18:30:04Z', event=pref['payload']['event'], schedule=pref['payload']['schedule'], crosswalk=p['alias_review']['crosswalk'], model=pref['payload']['model'],
        features=dict(order=list(evidence.research.FEATURES), values=pref['payload']['ordered_features'], available_at=pref['payload']['prepared_at']))
    o['mapping_review'].update(binding=mapping_binding(o['event'], o['schedule'], o['crosswalk']), reviewed_at='2026-10-16T18:30:02Z')
    q = o['quote']; q.update(provider_event_id=TARGET['provider_event_id'], event_home_team=TARGET['home_team'], event_away_team=TARGET['away_team'],
        event_start_utc=TARGET['start_utc'], book='draftkings', operator='draftkings', recorded_at='2026-10-16T18:30:00Z', price=-110)
    terms = o['source_review']['terms_review']; terms['operator'] = 'draftkings'
    batches = evidence.batches(h)
    n['payload']['dependency_objects'] = [dict(sha256=hashlib.sha256(model.encode(b)).hexdigest(), bytes_b64=base64.b64encode(model.encode(b)).decode()) for b in batches]
    deps = []
    for bi, b in enumerate(batches):
        for g in batches[0]['records']:
            if not g['completed']: continue
            for tid in (2655,235):
                if tid not in (g['homeId'],g['awayId']) or (bi and g['week'] != b['request']['week']): continue
                deps.append(dict(game_id=g['id'], team_id=tid, season=2026, start_utc=g['startDate'], available_at=b['retrieved_at'], kind='scoring' if bi == 0 else 'yardage', source_sha256=n['payload']['dependency_objects'][bi]['sha256']))
    o['feature_dependencies'] = deps
    d = p['dependency_review']; d['dependency_verification']['dependency_hashes'] = [x['sha256'] for x in n['payload']['dependency_objects']]
    objects = []
    with owner.projection.selected(packet):
        for i,obj in enumerate(h['payload']['objects']+qbundle['payload']['objects']):
            raw = base64.b64decode(obj['body_b64'])
            objects.append(custody.capture(raw, obj['metadata'], native_batch=batches[i]) if i < 4 else custody.capture(raw, obj['metadata'], quote=q, terms=terms, owner_reviewed=True))
    c['payload']['response_objects'] = objects
    resign(packet)
    from app_core.prediction_evidence import provider_quotes
    row.update(home_team=TARGET['home_team'], away_team=TARGET['away_team'], provider_event_id=TARGET['provider_event_id'], game_start_utc=TARGET['start_utc'],
               provider_quotes=provider_quotes(json.loads(odds(line if kind == 'spread_home' else -line))[0]), odds_american=-110,
               matchup_id='TULANE|MEMPHIS|2026-10-16', best_pick=(TARGET['home_team'] if kind=='spread_home' else TARGET['away_team'])+f' {line:+.1f}')
    # One selected quote in the actual caller, preserving the full two-outcome raw original.
    quotes=json.loads(row['provider_quotes']); row['provider_quotes']=json.dumps([x for x in quotes if x['market_type']==kind])
    monkeypatch.setattr(adapter, 'generated_time', lambda:NOW.isoformat())
    return packet, row


@pytest.mark.parametrize('kind,line,expected', [('spread_home',-3.5,.636830651175619), ('spread_away',3.5,.363169348824381)])
def test_staged_actual_caller_and_static_result(synthetic, monkeypatch, tmp_path, kind, line, expected):
    packet,row = fixture(synthetic,monkeypatch,tmp_path,kind,line); before=deepcopy(packet)
    owner.load(packet)
    with adapter.selected([packet],private_research=True): result=adapter.predict(row)
    assert result['ml_inference_status']=='success',result['ml_unavailable_reason']
    assert result['ml_probability']==pytest.approx(expected)
    assert adapter.diagnose(dict(row,**result))==dict(status='COMPLETE',reason='AVAILABLE')
    assert packet==before and not adapter.ACCEPTED_PACKETS and owner.projection.SELECTED.get() is None


@pytest.mark.parametrize('attack',['original','stage_event','stage_parent','stage_budget','feature','model','quote_clock','started','expired','review','review_subject','late_review','alias','side','line','price','old_version'])
def test_join_conflicts_never_reach_math(synthetic,monkeypatch,tmp_path,attack):
    packet,row=fixture(synthetic,monkeypatch,tmp_path);p=packet['payload'];o=p['custody_packet']['payload']['native_packet']['payload']['observation']['payload']
    if attack=='original':p['custody_packet']['payload']['response_objects'][0]['payload']['body_b64']='W10='
    elif attack=='stage_event':p['staged_binding']['payload']['quote']['payload']['plan']['payload']['target']['away_team']='Other'
    elif attack=='stage_parent':p['staged_binding']['payload']['prequote']['payload']['history_bundle_sha256']='a'*64
    elif attack=='stage_budget':p['staged_binding']['payload']['quote']['payload']['prior_active_seconds']=0
    elif attack=='feature':o['features']['values'][0]+=1
    elif attack=='model':o['model']['original_record_b64']='QQ=='
    elif attack=='quote_clock':o['quote']['recorded_at']='2026-10-16T17:00:00Z'
    elif attack=='started':monkeypatch.setattr(adapter,'generated_time',lambda:'2026-10-16T23:30:00Z')
    elif attack=='expired':monkeypatch.setattr(adapter,'generated_time',lambda:'2026-10-17T18:00:02Z')
    elif attack=='review':p['owner_review']['conclusions']['staged_evidence']['conclusion']='UNKNOWN'
    elif attack=='review_subject':p['owner_review']['subject_sha256']='a'*64
    elif attack=='late_review':p['owner_review']['reviewed_at']='2026-10-16T18:30:06Z'
    elif attack=='alias':p['alias_review']['owner_review']['qualifications']=''
    elif attack=='side':row['market_type']='spread_away'
    elif attack=='line':row['spread_line']=-7
    elif attack=='price':row['odds_american']=-120
    else:p['version']='ncaaf-owner-reviewed-private-inputs-v2'
    packet['sha256']=model.digest(p)
    monkeypatch.setattr(evidence.research,'centers',lambda *a,**k:pytest.fail('REJECTED STAGE REACHED NUMERICAL INFERENCE'))
    try:
        with adapter.selected([packet],private_research=True):r=adapter.predict(row)
    except (ValueError,KeyError):pass
    else:assert r['ml_inference_status']=='unavailable'


@pytest.mark.parametrize('change',['scoring','yards','cutoff','provisional','duplicate','names'])
def test_prequote_original_histories_not_assumptions(synthetic,monkeypatch,tmp_path,change):
    from test_ncaaf_owner_terms import fixture as old
    p,_=old(synthetic,monkeypatch);m=p['payload']['custody_packet']['payload']['native_packet']['payload']['observation']['payload']['model']
    raw=bodies();games=json.loads(raw[0])
    if change=='scoring':games=games[:5]
    elif change=='yards':raw[3]=model.encode([])
    elif change=='cutoff':
        for g in games[1:]:g['startDate']='2026-10-09T23:30:00Z'
    elif change=='provisional':games[0]['homeId']=999
    elif change=='duplicate':games.append(deepcopy(games[1]))
    else:games[0]['homeTeam']='Different school'
    raw[0]=model.encode(games);h,_,_=capture(monkeypatch,tmp_path,raw=raw)
    with pytest.raises(ValueError):evidence.prepare(h,model_packet=m,prepared_at='2026-10-16T18:01:00Z')


def test_actual_pipeline_capture_export_static_replay_private_display(synthetic,monkeypatch,tmp_path):
    from datetime import timedelta
    from app_core import research_display,research_replay
    packet,row=fixture(synthetic,monkeypatch,tmp_path);before=deepcopy(packet)
    analysis=owner.admitted_analysis(packet,row)
    selected=analysis.loc[analysis.market_type.eq('spread_home')].iloc[0]
    assert selected.ml_inference_status=='success',selected.ml_unavailable_reason
    assert selected.ml_probability==pytest.approx(.636830651175619) and selected.ml_probability!=.99
    from test_source_contract_pipeline import exported
    import test_source_contract_pipeline as ef
    monkeypatch.setattr(ef,'NOW',NOW);monkeypatch.setattr(ef,'CAPTURE',(NOW+timedelta(seconds=1)).isoformat())
    # This complete pipeline uses the same explicitly synthetic evaluation clock.
    monkeypatch.setattr('app_core.candidate_chronology.now_utc',lambda:NOW)
    saved=exported(monkeypatch,tmp_path/'export',analysis)
    from app_core.per_game_boards import per_game_board
    frames=[per_game_board(saved['card'],saved['captured'],family=f,novig_only=True,college_fallback=True,private_research=True) for f in ('overall','sides','totals')]
    shown=json.loads(frames[1].iloc[0].research_display)
    assert shown['probability']==selected.ml_probability and 'OWNER_REVIEWED' in shown['basis'] and shown['ev'] is None and shown['edge'] is None
    receipt=research_replay.retain_export(frames,saved['package'],saved['card'],saved['captured'],path=saved['db'])
    _,sources=research_replay.read_export(receipt['export_id'],path=saved['db'])
    retained=research_replay.frame_from_payload(next(iter(sources.values()))['original']['producer'])
    row=retained.loc[retained.market_type.eq('spread_home')].iloc[0]
    assert json.loads(row.ml_estimate_metadata)['ncaaf_inputs']['payload']['original_packet']==before
    assert research_display.from_export(row)['probability'] is None and all(x['production_bet_amount']==0 for x in saved['card'].wager_contract)
    assert 'body_b64' not in json.dumps(saved['package']) and not saved['captured'].production_eligible.fillna(False).any()
    assert adapter.diagnose(row)==dict(status='COMPLETE',reason='AVAILABLE')
    from streamlit.testing.v1 import AppTest
    script=tmp_path/'SYNTHETIC-private-display.py'
    script.write_text('import streamlit as st\nfrom app.ui.ncaaf_pipeline_research import render\nrender(st.session_state["games"],st.session_state["candidates"])',encoding='utf-8')
    app=AppTest.from_file(str(script));app.session_state['games']=saved['card'];app.session_state['candidates']=saved['captured'];app.session_state['ncaaf_private_research']=True;app.run()
    assert not app.exception and any(m.value==f'{selected.ml_probability:.2%}' for m in app.metric)
    assert packet==before and not adapter.ACCEPTED_PACKETS


@pytest.mark.parametrize('change',['values','runtime','prepared_clock','availability','model'])
def test_consistently_rehashed_preparation_cannot_invent_values_or_refresh_clocks(synthetic,monkeypatch,tmp_path,change):
    h,p=prepared(synthetic,monkeypatch,tmp_path);v=p['payload']
    if change=='values':v['ordered_features'][0]+=10
    elif change=='runtime':v['runtime']['python']='0.0.0'
    elif change=='prepared_clock':v['prepared_at']='2026-10-17T18:00:02Z'
    elif change=='availability':v['publisher_availability']='VERIFIED_AT_PREPARATION'
    else:v['model']['compatibility']['family']='total'
    p=stage.seal(v)
    with pytest.raises(ValueError):evidence.validate_prequote(p,h,at='2026-10-16T18:30:01Z')


def test_integer_discovery_is_preserved_without_integer_admission(synthetic,monkeypatch,tmp_path,capsys):
    from scripts import ncaaf_staged_acquisition as cli
    h,p=prepared(synthetic,monkeypatch,tmp_path);clock=Clock();clock.value='2026-10-16T18:30:01Z'
    q,_,_=capture(monkeypatch,tmp_path,plan(stage.QUOTE_PLAN,h,p),[odds(-7)],clock,h,p)
    assert q['payload']['status']=='CAPTURED_UNREVIEWED' and not q['payload']['inference']
    for name,value in [('history.json',h),('prequote.json',p),('quote.json',q)]: (tmp_path/name).write_bytes(model.encode(value))
    assert cli.main(['join',str(tmp_path/'quote.json'),'--history',str(tmp_path/'history.json'),'--prequote',str(tmp_path/'prequote.json'),'--output',str(tmp_path/'joined.json')])==0
    assert json.loads(capsys.readouterr().out)['status']=='BOUND_UNREVIEWED'
    assert json.loads((tmp_path/'joined.json').read_bytes())==evidence.join(h,p,q)
    from app_core import ncaaf_prospective_chronology as chronology
    # The historical caller's explicit zero-push guard remains in its reader.
    import inspect
    assert 'NCAAF_INTEGER_PUSH_MODEL_UNVALIDATED' in inspect.getsource(chronology.read_observation)


def test_provider_observation_expiry_is_not_refreshed_by_later_inference(synthetic,monkeypatch,tmp_path):
    packet,row=fixture(synthetic,monkeypatch,tmp_path)
    monkeypatch.setattr(adapter,'generated_time',lambda:'2026-10-16T18:45:01Z') # 901 seconds, still hours before kickoff.
    monkeypatch.setattr(evidence.research,'centers',lambda *a,**k:pytest.fail('STALE ORIGINAL OBSERVATION REACHED INFERENCE'))
    with adapter.selected([packet],private_research=True):result=adapter.predict(row)
    assert result['ml_inference_status']=='unavailable' and result['ml_unavailable_reason']=='NCAAF_COMPAT_QUOTE_CLOCK_CONFLICT'


@pytest.mark.parametrize('change',['missing_review','review_subject','missing_configuration','mismatched_configuration'])
def test_review_and_configuration_omissions_cannot_enter_caller_or_private_replay(synthetic,monkeypatch,tmp_path,change):
    from app_core import research_display
    packet,row=fixture(synthetic,monkeypatch,tmp_path)
    analysis=owner.admitted_analysis(packet,row)
    retained=analysis.loc[analysis.market_type.eq('spread_home')].iloc[0].copy()
    bad=deepcopy(packet);joined=bad['payload']['staged_binding'];j=joined['payload']
    if change.startswith('missing_review') or change=='review_subject':
        p=j['prequote']['payload']
        if change=='missing_review':p['history_review']=None
        else:p['history_review']['subject_sha256']='f'*64
        j['prequote']=stage.seal(p)
    else:
        p=j['quote']['payload']
        if change=='missing_configuration':p.pop('control_receipt')
        else:p['control_receipt']=deepcopy(j['history']['payload']['control_receipt'])
        j['quote']=stage.seal(p)
    bad['payload']['staged_binding']=stage.seal(j);bad=stage.seal(bad['payload'])
    monkeypatch.setattr(evidence.research,'centers',lambda *a,**k:pytest.fail('INVALID REVIEW OR RECEIPT REACHED NUMERICAL EXECUTION'))
    with pytest.raises((ValueError,KeyError)):owner.load(bad)
    with pytest.raises((ValueError,KeyError)):
        with adapter.selected([bad],private_research=True):adapter.predict(row)
    metadata=json.loads(retained.ml_estimate_metadata)
    metadata['ncaaf_inputs']['payload']['original_packet']=bad
    metadata['ncaaf_inputs']['sha256']=model.digest(metadata['ncaaf_inputs']['payload'])
    retained['ml_estimate_metadata']=json.dumps(metadata)
    assert adapter.diagnose(retained)['status']!='COMPLETE'
    display=research_display.from_export(retained,private_research=True)
    assert display['probability'] is None and display['availability_reason']!='AVAILABLE'
