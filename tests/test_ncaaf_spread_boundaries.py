"""Synthetic complete-boundary regressions; no authentic inputs or transport."""
import base64
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import hashlib
import json
import pytest
from app_core import ncaaf_owner_spread as owner,ncaaf_spread_pair as pair,ncaaf_spread_mass as mass
from app_core import ncaaf_spread_execution as execution,ncaaf_future_staged as evidence,ncaaf_future_target as target
from app_core import ncaaf_staged_acquisition as stage,ncaaf_model_compatibility as model,ncaaf_pipeline_evidence as adapter
from test_ncaaf_model_compatibility import synthetic
from ncaaf_future_pair_fixtures import fixture,pair_fixture,inference_authorization,NOW,resign,bodies,capture,plan,Clock,odds,stage_configuration,review_prequote,identity


@pytest.mark.parametrize('line',[-7,0,7,-.5,.5])
def test_discrete_boundary_orientation_uses_unchanged_native_primitive(line):
    from app_core import ncaaf_research as native
    # The native Gaussian margin remains unchanged, including half-unit bins.
    values=native.probabilities(2.,9.,-line,total=False)
    home=mass.from_native(values,'spread_home');away=mass.from_native(values,'spread_away')
    assert home['cover']==values['over'] and home['loss']==values['under']
    assert away['cover']==values['under'] and away['loss']==values['over']
    assert mass.compare(home,away)['shared_push']==values['push']
    assert (home['push']>0) if float(line).is_integer() else home['push']==0
    assert mass.target_contract('spread_home',line,integer=float(line).is_integer())['signed_line']==line


@pytest.mark.parametrize('handicap',[-7,0,7])
@pytest.mark.parametrize('delta',[-1,0,1])
def test_integer_scores_immediately_below_at_above_signed_threshold(handicap,delta):
    from app_core import ncaaf_research as native
    margin=-handicap+delta
    values=native.probabilities(margin,.01,-handicap,total=False)
    home=mass.from_native(values,'spread_home');away=mass.from_native(values,'spread_away')
    assert home[('loss','push','cover')[delta+1]]==pytest.approx(1.)
    assert away[('cover','push','loss')[delta+1]]==pytest.approx(1.)
    assert mass.compare(home,away)['shared_push']==(1. if delta==0 else 0.)


@pytest.mark.parametrize('attack',['body','feature','model','event','orientation','kickoff','review','review_subject','review_late','prequote_review','prequote_subject','history_config','quote_config','discovery_config','original_alias','alias_review','price','line','borrow_old'])
def test_changed_input_never_reaches_math_or_replay(synthetic,monkeypatch,tmp_path,attack):
    packet,row=fixture(synthetic,monkeypatch,tmp_path);p=packet['payload'];o=pair._observation(packet)
    if attack=='body':p['custody_packet']['payload']['response_objects'][0]['payload']['body_b64']='QQ=='
    elif attack=='feature':o['features']['values'][0]+=1
    elif attack=='model':o['model']['compatibility']['family']='total'
    elif attack=='event':o['event']['provider_event_id']='OTHER'
    elif attack=='orientation':o['event']['home_team'],o['event']['away_team']=o['event']['away_team'],o['event']['home_team']
    elif attack=='kickoff':o['event']['start_utc']='2026-10-18T04:00:00Z'
    elif attack=='review':p['owner_review']=None
    elif attack=='review_subject':p['owner_review']['subject_sha256']='0'*64
    elif attack=='review_late':p['owner_review']['reviewed_at']='2026-10-17T18:30:06Z'
    elif attack.startswith('prequote'):
        j=p['staged_binding']['payload'];v=j['prequote']['payload']
        if attack=='prequote_review':v['history_review']=None
        else:v['history_review']['subject_sha256']='0'*64
        j['prequote']=stage.seal(v);p['staged_binding']=stage.seal(j)
    elif attack in ('history_config','quote_config'):
        j=p['staged_binding']['payload'];k=attack.split('_')[0];v=j[k]['payload']
        v['control_receipt']=deepcopy(j['quote' if k=='history' else 'history']['payload']['control_receipt'])
        j[k]=stage.seal(v);p['staged_binding']=stage.seal(j)
    elif attack=='discovery_config':
        c=p['alias_review']['discovery']['configuration_receipt'];data=json.loads(base64.b64decode(c['file_b64']))
        data['payload']['subject']['plan_sha256']='0'*64;data=stage.seal(data['payload']);raw=model.encode(data)
        c.update(file_b64=base64.b64encode(raw).decode(),file_sha256=hashlib.sha256(raw).hexdigest())
    elif attack=='original_alias':p['alias_review']['crosswalk'][0]['provider_name']='Different school'
    elif attack=='alias_review':p['alias_review']['owner_review']['subject_sha256']='0'*64
    elif attack=='price':row['odds_american']=-120
    elif attack=='line':row['spread_line']=-8
    else:p['version']='ncaaf-owner-reviewed-private-inputs-v2'
    packet=stage.seal(p)
    monkeypatch.setattr(evidence.research,'centers',lambda *a,**k:pytest.fail('INVALID INPUT REACHED MATH'))
    try:
        with adapter.selected([packet],private_research=True):result=adapter.predict(row)
    except (ValueError,TypeError,KeyError):pass
    else:assert result['ml_inference_status']=='unavailable'
    assert not list(tmp_path.glob('*.inference-attempts.jsonl'))


@pytest.mark.parametrize('attack',['features','event','period','terms','line','side','clock','history','model'])
def test_rehashed_comparison_cannot_join_different_subjects(synthetic,monkeypatch,tmp_path,attack):
    from ncaaf_future_pair_fixtures import other_side
    h,hr=fixture(synthetic,monkeypatch,tmp_path);a,ar=other_side(h,hr);o=pair._observation(a)
    if attack=='features':o['features']['values'][0]+=1
    elif attack=='event':o['event']['provider_event_id']='OTHER'
    elif attack=='period':o['quote']['period']='first_half'
    elif attack=='terms':o['source_review']['terms_review']['product']='UNKNOWN'
    elif attack=='line':o['quote']['point']=8
    elif attack=='side':o['quote']['market_type']='spread_home'
    elif attack=='clock':o['quote']['recorded_at']='2026-10-17T18:29:59Z'
    elif attack=='model':o['model']['compatibility']['family']='total'
    else:a['payload']['staged_binding']['payload']['history']['payload']['objects'][0]['body_b64']='QQ=='
    # This is an adversarial rehash, not a substantive owner answer.
    resign(a)
    with pytest.raises((ValueError,TypeError,KeyError)):pair._same(dict(home=h,away=a))


@pytest.mark.parametrize('attack',['unused_pool','alter_pool','review','child_subject','future_prepared','wrong_child_review'])
def test_pair_envelope_review_and_exact_shared_bytes(synthetic,monkeypatch,tmp_path,attack):
    p,_=pair_fixture(synthetic,monkeypatch,tmp_path);v=p['payload']
    if attack=='unused_pool':v['shared_bytes'][model.digest('QQ==')]='QQ=='
    elif attack=='alter_pool':v['shared_bytes'][next(iter(v['shared_bytes']))]='QQ=='
    elif attack=='review':v['owner_review']=None
    elif attack=='child_subject':v['child_subjects']['home']='0'*64
    elif attack=='future_prepared':v['prepared_at']='2026-10-17T18:30:06Z'
    else:v['owner_review']['child_reviews']['home']=v['owner_review']['child_reviews']['away']
    with pytest.raises((ValueError,TypeError,KeyError)):pair.load(stage.seal(v))


def test_shared_carriers_cannot_expand_beyond_child_limit_before_encoding():
    value=base64.b64encode(b'SYNTHETIC'*150000).decode();sha=model.digest(value)
    ref={'body_b64':{pair.REFERENCE:sha}}
    assert pair._unpack([ref],{sha:value},set())==[{'body_b64':value}]
    with pytest.raises(ValueError,match='NCAAF_PAIR_LIMIT'):
        pair._unpack([ref]*5,{sha:value},set())


@pytest.mark.parametrize('change',['missing_scoring','missing_yards','cutoff','duplicate','orientation','unconfirmed_id'])
def test_original_history_is_required_before_prequote(synthetic,monkeypatch,tmp_path,change):
    (tmp_path/'first').mkdir();(tmp_path/'second').mkdir()
    packet,_=fixture(synthetic,monkeypatch,tmp_path/'first');m=pair._observation(packet)['model']
    raw=bodies();games=json.loads(raw[0])
    if change=='missing_scoring':games=games[:5]
    elif change=='missing_yards':raw[3]=model.encode([])
    elif change=='cutoff':
        for g in games[1:]:g['startDate']='2026-10-11T03:00:00Z'
    elif change=='duplicate':games.append(deepcopy(games[1]))
    elif change=='orientation':games[0]['homeTeam']='Nebraska'
    else:games[0]['homeId']=999
    raw[0]=model.encode(games);h,_,_=capture(monkeypatch,tmp_path/'second',raw=raw)
    with pytest.raises(ValueError):evidence.prepare(h,model_packet=m,prepared_at='2026-10-17T18:01:00Z')


@pytest.mark.parametrize('clock',['2026-10-17T18:30:03Z','2026-10-17T18:45:01Z','2026-10-18T03:00:00Z','2026-10-18T18:30:05Z'])
def test_actual_evaluation_time_cannot_be_refreshed(synthetic,monkeypatch,tmp_path,clock):
    p,row=fixture(synthetic,monkeypatch,tmp_path)
    monkeypatch.setattr(adapter,'generated_time',lambda:clock)
    monkeypatch.setattr(evidence.research,'centers',lambda *a,**k:pytest.fail('INVALID CLOCK REACHED MATH'))
    with adapter.selected([p],private_research=True):r=adapter.predict(row)
    assert r['ml_inference_status']=='unavailable'


def test_pregame_gate_independently_rejects_a_quote_still_within_900_seconds(synthetic,monkeypatch,tmp_path):
    from app_core import ncaaf_prospective_chronology as chronology
    p,_=fixture(synthetic,monkeypatch,tmp_path);o=pair._observation(p);r=o['source_review']
    # Isolate the unchanged chronology validator. This is not a new original
    # capture/admission: the complete original-binding tests are separate.
    o['quote']['recorded_at']='2026-10-18T02:59:00Z'
    r['quote_observation']['observed_at']='2026-10-18T02:59:01Z'
    r['offer_verification']['verified_at']='2026-10-18T02:59:02Z'
    r['owner_verification']['reviewed_at']='2026-10-18T02:59:03Z'
    o['as_of']='2026-10-18T02:59:04Z';resign(p)
    policy=owner.ReviewPolicy(p)
    chronology.check_chronology(o,'2026-10-18T02:59:05Z',review_policy=policy)
    with pytest.raises(ValueError,match='NCAAF_COMPAT_QUOTE_CLOCK_CONFLICT'):
        chronology.check_chronology(o,'2026-10-18T03:00:00Z',review_policy=policy)


def test_new_history_age_cannot_be_refreshed_by_a_new_prequote_review(synthetic,monkeypatch,tmp_path):
    from ncaaf_future_pair_fixtures import target_review
    p=plan();v=p['payload']
    def previous_day(value):
        if isinstance(value,dict):return {k:previous_day(x) for k,x in value.items()}
        if isinstance(value,list):return [previous_day(x) for x in value]
        return value.replace('2026-10-17T','2026-10-16T') if isinstance(value,str) else value
    for key in ('target_evidence','execution_window','advance_permissions'):v[key]=previous_day(v[key])
    te=v['target_evidence']
    for ref in te['references']:ref['sha256']=model.digest({k:x for k,x in ref.items() if k!='sha256'})
    te['owner_review']['subject_sha256']=target.subject(te)
    te['owner_review']['conclusions']['source_basis']['evidence_sha256']=model.digest(te['references'])
    p=stage.seal(v);a,raw=stage_configuration(p,monkeypatch,tmp_path)
    c=previous_day(json.loads(raw)['payload']);a=stage.seal(previous_day(a['payload']));c['authorization']=a
    c['review']['subject_sha256']=model.digest({k:x for k,x in c.items() if k!='review'});raw=model.encode(stage.seal(c))
    monkeypatch.setattr(stage,'AUTHORIZED_COLLECTIONS',{a['payload']['reference']:a['sha256']})
    monkeypatch.setattr(stage,'AUTHORIZED_CONFIGURATIONS',{p['sha256']:dict(file_sha256=hashlib.sha256(raw).hexdigest(),file_b64=base64.b64encode(raw).decode())})
    monkeypatch.setattr(stage,'AUTHORIZED_ACCOUNT_EVIDENCE',{a['payload']['reference']:c['account']})
    from test_ncaaf_pilot import Stream
    clock=Clock('2026-10-16T18:00:01Z');responses=bodies();sent=[]
    def transport(*args):sent.append(args);return Stream(responses[len(sent)-1])
    h=stage.acquire(p,a,root=tmp_path,transport=transport,wall=clock.wall,monotonic=clock.mono)
    from test_ncaaf_owner_terms import fixture as old
    original,_=old(synthetic,monkeypatch);m=original['payload']['custody_packet']['payload']['native_packet']['payload']['observation']['payload']['model']
    pref=evidence.prepare(h,model_packet=m,prepared_at='2026-10-16T18:01:00Z');ident=identity(monkeypatch,tmp_path,pref)
    pref=review_prequote(pref,h) # New review on October17 cannot renew October16 originals.
    qp=plan(target.QUOTE_PLAN,h,pref,ident);qp['payload']['target_evidence']=deepcopy(te);qp=stage.seal(qp['payload'])
    qa,_=stage_configuration(qp,monkeypatch,tmp_path);before={x.name:x.read_bytes() for x in tmp_path.iterdir()}
    clock=Clock('2026-10-17T18:30:01Z')
    with pytest.raises(ValueError,match='INPUT_CLOCK'):
        stage.acquire(qp,qa,root=tmp_path,history_bundle=h,prequote=pref,transport=lambda *a:pytest.fail('EXPIRED INPUT DISPATCH'),wall=clock.wall,monotonic=clock.mono)
    assert {x.name:x.read_bytes() for x in tmp_path.iterdir()}==before


@pytest.mark.parametrize('failure_type',[RuntimeError,KeyboardInterrupt])
def test_separate_inference_permission_is_required_and_spent_on_interruption(synthetic,monkeypatch,tmp_path,failure_type):
    p,rows=pair_fixture(synthetic,monkeypatch,tmp_path);a=inference_authorization(p,monkeypatch,paired=True)
    # Fail only the second real caller. First computation remains attributable.
    actual=owner.admitted_analysis;calls=[]
    def interrupted(*args,**kwargs):
        calls.append(1)
        if len(calls)==2:raise failure_type('SYNTHETIC sensitive exception is not retained')
        return actual(*args,**kwargs)
    monkeypatch.setattr(owner,'admitted_analysis',interrupted)
    with pytest.raises(failure_type):pair.admitted_analysis(p,rows,authorization=a,custody_root=tmp_path)
    raw=(tmp_path/(a['sha256']+'.inference-attempts.jsonl')).read_bytes()
    events=[json.loads(x) for x in raw.splitlines()]
    assert events[-1]['status']=='INCOMPLETE' and events[-1]['completed']==1
    failure=json.loads((tmp_path/(a['sha256']+'.comparison-incomplete.json')).read_bytes())['payload']
    assert failure['status']=='INCOMPLETE' and len(failure['completed_children'])==1
    files=list(tmp_path.glob('*.computation.json'));assert len(files)==1
    retained=json.loads(files[0].read_bytes())
    assert retained==failure['completed_children']['home']['payload']['custody_computation']
    assert b'sensitive exception' not in model.encode(failure) and not list(tmp_path.glob('*.comparison.json'))
    with pytest.raises(ValueError,match='ALREADY_SPENT'):pair.admitted_analysis(p,rows,authorization=a,custody_root=tmp_path)


def test_late_completed_computation_is_retained_unavailable_not_relabelled(synthetic,monkeypatch,tmp_path):
    p,row=fixture(synthetic,monkeypatch,tmp_path);a=inference_authorization(p,monkeypatch)
    calls=[]
    def clock():
        calls.append(1);return NOW.isoformat() if len(calls)<3 else '2026-10-17T18:45:01Z'
    monkeypatch.setattr(adapter,'generated_time',clock)
    try:r=owner.admitted_analysis(p,row,authorization=a,custody_root=tmp_path)
    except (ValueError,KeyError):pass
    else:assert r.ml_inference_status.eq('unavailable').all()
    files=list(tmp_path.glob('*.unavailable.json'))
    assert len(files)==1
    receipt=json.loads(files[0].read_bytes())['payload']
    assert receipt['status']=='UNAVAILABLE' and receipt['computation']['sha256'] and receipt['live_stake']==0
    events=[json.loads(x) for x in (tmp_path/(a['sha256']+'.inference-attempts.jsonl')).read_bytes().splitlines()]
    assert events[-1]['status']=='INCOMPLETE' and events[-1]['attempted']==1 and events[-1]['completed']==0


@pytest.mark.parametrize('attack',['missing','subject','input','count','expired','trust'])
def test_inference_authority_cannot_be_supplied_by_packet_or_upload(synthetic,monkeypatch,tmp_path,attack):
    p,row=fixture(synthetic,monkeypatch,tmp_path);a=inference_authorization(p,monkeypatch)
    if attack=='subject':a['payload']['subject_sha256']='0'*64
    elif attack=='input':a['payload']['input_hashes']=['0'*64]
    elif attack=='count':a['payload']['max_computations']=3
    elif attack=='expired':a['payload']['expires_at']=NOW.isoformat()
    elif attack=='trust':monkeypatch.setattr(execution,'AUTHORIZED_INFERENCES',{})
    if attack=='missing':a=None
    elif attack not in ('trust',):
        a=stage.seal(a['payload'])
        monkeypatch.setattr(execution,'AUTHORIZED_INFERENCES',{a['payload']['reference']:a['sha256']})
    monkeypatch.setattr(evidence.research,'centers',lambda *a,**k:pytest.fail('UNAUTHORIZED INFERENCE'))
    with pytest.raises((ValueError,TypeError,KeyError)):owner.admitted_analysis(p,row,authorization=a,custody_root=tmp_path)
    assert not list(tmp_path.glob('*.inference-attempts.jsonl'))


@pytest.mark.parametrize('attack',['cover','push','conditional','clock','journal','authorization'])
def test_rehashed_result_and_execution_receipt_changes_reject_statically(synthetic,monkeypatch,tmp_path,attack):
    p,row=fixture(synthetic,monkeypatch,tmp_path);a=inference_authorization(p,monkeypatch)
    result=owner.admitted_analysis(p,row,authorization=a,custody_root=tmp_path)
    saved=json.loads(result.iloc[0].ml_estimate_metadata)['ncaaf_inputs']['payload']['computation']
    v=saved['payload']
    if attack=='cover':v['outcome_probabilities']['cover']+=.01;v['outcome_probabilities']['loss']-=.01
    elif attack=='push':v['outcome_probabilities']['push']+=.01;v['outcome_probabilities']['loss']-=.01
    elif attack=='conditional':v['conditional_no_push']['value']=.99
    elif attack=='clock':v['execution_receipt']['completed_at']='2026-10-17T18:45:01Z'
    elif attack=='journal':
        proof=v['execution_receipt'];events=[json.loads(x) for x in base64.b64decode(proof['journal_prefix_b64']).splitlines()]
        events[1]['input_sha256']='0'*64;raw=b''.join(model.encode(x)+b'\n' for x in events)
        proof.update(journal_prefix_b64=base64.b64encode(raw).decode(),journal_prefix_sha256=hashlib.sha256(raw).hexdigest())
    else:
        proof=v['execution_receipt'];au=proof['authorization']['payload'];au['input_hashes']=['0'*64];proof['authorization']=stage.seal(au)
    bad=stage.seal(v)
    monkeypatch.setattr(evidence.research,'centers',lambda *a,**k:pytest.fail('STATIC REPLAY RAN INFERENCE'))
    with pytest.raises(ValueError):owner.inspect_result(p,bad,NOW.isoformat())
    assert not adapter.ACCEPTED_PACKETS


@pytest.mark.parametrize('line',[-7,-3.5])
def test_complete_pair_capture_export_static_replay_and_private_display(synthetic,monkeypatch,tmp_path,line):
    from app_core import research_display,research_replay
    from app_core.per_game_boards import per_game_board
    from test_source_contract_pipeline import exported
    import test_source_contract_pipeline as ef
    p,rows=pair_fixture(synthetic,monkeypatch,tmp_path,line);before=deepcopy(p)
    assert rows['home']['spread_line']==line and rows['away']['spread_line']==-line
    a=inference_authorization(p,monkeypatch,paired=True)
    analysis=pair.admitted_analysis(p,rows,authorization=a,custody_root=tmp_path)
    monkeypatch.setattr(ef,'NOW',NOW);monkeypatch.setattr(ef,'CAPTURE',(NOW+timedelta(seconds=1)).isoformat())
    monkeypatch.setattr('app_core.candidate_chronology.now_utc',lambda:NOW)
    monkeypatch.setattr('app_core.prediction_evidence.now_utc',lambda:(NOW+timedelta(seconds=1)).isoformat())
    monkeypatch.setattr('app_core.public_board.datetime',ef.FrozenDateTime)
    exports=pair.capture_exports(analysis,custody_root=tmp_path,database=tmp_path/'private.sqlite3')
    saved=exports[0];saved['db']=tmp_path/'private.sqlite3'
    frames=[per_game_board(saved['card'],saved['captured'],family=f,novig_only=True,college_fallback=True,private_research=True) for f in ('overall','sides','totals')]
    assert not frames[1].empty
    shown=json.loads(frames[1].iloc[0].ncaaf_spread_private_display)
    assert shown['comparison']['shared_push']>=0 and shown['wager_action']=='PASS' and shown['live_stake']==0
    receipt=research_replay.retain_export(frames,saved['package'],saved['card'],saved['captured'],path=saved['db'])
    _,sources=research_replay.read_export(receipt['export_id'],path=saved['db'])
    retained=research_replay.frame_from_payload(next(iter(sources.values()))['original']['producer'])
    for row in retained.to_dict('records'):
        assert adapter.diagnose(row)==dict(status='COMPLETE',reason='AVAILABLE')
        assert research_display.from_export(row)['probability'] is None
        assert pair.private_display(row)['outcome_probabilities']['cover']==row['ml_probability']
    assert p==before and 'body_b64' not in json.dumps(saved['package'])
    assert all(game['research_display']['probability'] is None for games in saved['package']['games'].values() for game in games)
    assert not saved['captured'].production_eligible.fillna(False).any()
    assert all(x['production_bet_amount']==0 for x in saved['card'].wager_contract)
    # Replay is static: disable all numerical execution, then load/display again.
    monkeypatch.setattr(evidence.research,'centers',lambda *a,**k:pytest.fail('REPLAY RAN MATH'))
    for row in retained.to_dict('records'):assert pair.private_display(row)['ev'] is None
    from streamlit.testing.v1 import AppTest
    script=tmp_path/'SYNTHETIC-pair-display.py'
    script.write_text('import streamlit as st\nfrom app.ui.ncaaf_pipeline_research import render\nrender(st.session_state["games"],st.session_state["candidates"])',encoding='utf-8')
    app=AppTest.from_file(str(script),default_timeout=40);app.session_state['games']=saved['card'];app.session_state['candidates']=saved['captured'];app.session_state['ncaaf_private_research']=True;app.run()
    assert not app.exception and any(m.value==f"{analysis.iloc[0].ml_probability:.2%}" for m in app.metric)
    assert not adapter.ACCEPTED_PACKETS and execution.ACTIVE.get() is None
