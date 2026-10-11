"""Labelled synthetic fixtures only. All external transports are blocked."""
import base64
from copy import deepcopy
from datetime import timedelta
import json
from unittest.mock import Mock

import pytest
from app_core import ncaaf_owner_research as owner, ncaaf_pipeline_evidence as adapter
from app_core import ncaaf_response_custody as custody, ncaaf_compatible_pipeline as native
from app_core import ncaaf_prospective_chronology as chronology, ncaaf_compatible_observation as original
from app_core import ncaaf_model_compatibility as model, ncaaf_research as research
from scripts.benchmark_drive_history_loading import blocked_network
from test_ncaaf_model_compatibility import synthetic
import test_ncaaf_response_custody as previous
import test_ncaaf_compatible_pipeline as pipeline_fixture

NOW = previous.NOW


@pytest.fixture(autouse=True)
def offline():
    with blocked_network():
        yield


def verify_owner(packet):
    """Manufactured human conclusions ONLY inside a labelled synthetic fixture."""
    p = packet['payload']
    p['owner_review'] = dict(version=owner.REVIEW_VERSION, review_id='SYNTHETIC-owner-review',
        reviewer=owner.OWNER, reviewed_at='2026-10-09T12:00:03Z', review_status=owner.STATUS,
        use_scope=owner.SCOPE, subject_version=owner.SUBJECT_VERSION,
        subject_sha256=owner.subject_hash(packet), attestation=owner.ATTESTATION,
        conclusions={k:dict(conclusion='VERIFIED', evidence_sha256=v,
            finding='SYNTHETIC owner inspected the exact '+k+' evidence; fixture only.')
            for k,v in owner.evidence_subjects(packet).items()})
    packet['sha256'] = model.digest(p)


def fixture(synthetic, monkeypatch, kind='spread_home', line=-3.5):
    c, row, approval = previous.fixture(synthetic, monkeypatch, kind, line)
    c = deepcopy(c); n = c['payload']['native_packet']; o = n['payload']['observation']['payload']
    o['version'] = owner.ReviewPolicy.OBSERVATION_VERSION
    o['mapping_review']['reviewer'] = owner.OWNER
    r = o['source_review']; r['version'] = owner.ReviewPolicy.OFFER_VERSION
    t = r['terms_review']; t['reviewer'] = owner.OWNER; t['permitted_uses'][-1] = 'private_derived_output'
    r['offer_verification'].update(verifier=owner.OWNER, terms_sha256=model.digest(t), mapping_sha256=model.digest(o['mapping_review']))
    a = r.pop('acceptance'); a['reviewed_at'] = a.pop('accepted_at')
    a.update(reviewer=owner.OWNER, subject_version=o['version'])
    r['owner_verification'] = a; a['subject_sha256'] = chronology.subject_hash(o)
    n['payload']['version'] = owner.ReviewPolicy.NATIVE_VERSION
    n['payload']['observation']['sha256'] = model.digest(o); n['sha256'] = model.digest(n['payload'])
    d = deepcopy(approval['dependency_source_review']); d['version'] = owner.ReviewPolicy.DEPENDENCY_VERSION
    perm = d['permissions_review']; perm['reviewer'] = owner.OWNER; perm['permitted_uses'][-1] = 'private_derived_output'
    v = d['dependency_verification']; v.update(verifier=owner.OWNER, permissions_sha256=model.digest(perm), subject_sha256=chronology.dependency_subject_hash(n))
    a = d.pop('acceptance'); a['reviewed_at'] = a.pop('accepted_at')
    a.update(reviewer=owner.OWNER, subject_sha256=v['subject_sha256'], verification_sha256=model.digest(v))
    d['owner_verification'] = a
    for obj in c['payload']['response_objects']:
        b = obj['payload']
        if b['metadata']['provider'] == 'odds_api':
            rows = json.loads(base64.b64decode(b['body_b64']))
            b['projection'] = custody._project_quote(rows, b['metadata'], o['quote'], t, owner_reviewed=True)
        obj['sha256'] = model.digest(b)
    c['payload']['version'] = owner.ReviewPolicy.VERSION
    r = c['payload']['custody_admission']; r['version'] = owner.ReviewPolicy.CUSTODY_VERSION
    v = r['verification']; v.update(verifier=owner.OWNER, subject_sha256=custody.subject_hash(c),
        permissions_sha256=model.digest(perm), terms_sha256=model.digest(t), native_verification_sha256=model.digest(d['dependency_verification']))
    a = r.pop('acceptance'); a['reviewed_at'] = a.pop('accepted_at')
    a.update(reviewer=owner.OWNER, subject_sha256=v['subject_sha256'], verification_sha256=model.digest(v))
    r['owner_verification'] = a; c['sha256'] = model.digest(c['payload'])
    packet = dict(payload=dict(version=owner.VERSION, evidence_label='SYNTHETIC', custody_packet=c,
        dependency_review=d, owner_review={}), sha256='')
    verify_owner(packet)
    # No independent trust catalog is used or populated by private processing.
    for module, names in ((adapter,['ACCEPTED_PACKETS']), (custody,['ACCEPTED_ADMISSIONS']),
        (original,['ACCEPTED_EVENT_MAPPINGS','ACCEPTED_SOURCE_REVIEWS']),
        (chronology,['ACCEPTED_TERMS_REVIEWS','ACCEPTED_ADMISSIONS','ACCEPTED_DEPENDENCY_PERMISSIONS','ACCEPTED_DEPENDENCY_ADMISSIONS'])):
        for name in names: monkeypatch.setattr(module, name, {})
    return packet, row


def actual(monkeypatch, packet):
    view = native.view
    monkeypatch.setattr(native, 'view', lambda p:owner.view(p) if p['payload']['version']==owner.VERSION else view(p))
    selected = adapter.selected
    # Explicit owner selection in the test's actual application entrypoint.
    monkeypatch.setattr(adapter, 'selected', lambda packets=():selected(packets, private_research=True))
    return pipeline_fixture.actual(monkeypatch, packet)


@pytest.mark.parametrize('kind,line,probability', [('spread_home',-3.5,.636830651175619),
    ('spread_away',3.5,.363169348824381), ('total_over',50.5,.4800611941616275),
    ('total_under',50.5,.5199388058383725)])
def test_owner_caller_same_math_and_strict_modes_cannot_borrow(synthetic, monkeypatch, kind, line, probability):
    packet,row = fixture(synthetic,monkeypatch,kind,line); before=deepcopy(packet)
    with pytest.raises(ValueError, match='NCAAF_PRIVATE_RESEARCH_NOT_SELECTED'):
        with adapter.selected([packet]): pass
    with pytest.raises(ValueError): custody.load(packet['payload']['custody_packet'])
    with pytest.raises(ValueError): native.load(packet['payload']['custody_packet']['payload']['native_packet'])
    with pytest.raises(ValueError): chronology.read_observation(packet['payload']['custody_packet']['payload']['native_packet']['payload']['observation'])
    with adapter.selected([packet], private_research=True): result=adapter.predict(row)
    assert result['ml_inference_status']=='success',result['ml_unavailable_reason']
    assert result['ml_probability']==pytest.approx(probability)
    assert adapter.diagnose(dict(row,**result))==dict(status='COMPLETE',reason='AVAILABLE')
    saved=json.loads(result['ml_estimate_metadata'])['ncaaf_inputs']['payload']['computation']['payload']
    assert (saved['review_status'],saved['use_scope'])==('OWNER_REVIEWED','PRIVATE_RESEARCH')
    assert saved['wager_action']=='PASS' and saved['live_stake']==0
    assert not any(saved[k] for k in ('source_acceptance','scientific_acceptance','probability_calibration','wagering_authority'))
    assert not adapter.ACCEPTED_PACKETS and not custody.ACCEPTED_ADMISSIONS and not chronology.ACCEPTED_ADMISSIONS
    assert packet==before


@pytest.mark.parametrize('change,reason', [
    ('subject','NCAAF_OWNER_REVIEW_SUBJECT_CONFLICT'), ('conclusion','NCAAF_OWNER_REVIEW_INCOMPLETE'),
    ('reviewer','NCAAF_OWNER_REVIEW_SCHEMA'), ('late_review','NCAAF_OWNER_REVIEW_CLOCK_CONFLICT'),
    ('borrowed_independent','NCAAF_CUSTODY_SCHEMA'), ('corrupt_body','NCAAF_CUSTODY_INTEGRITY'),
    ('stale','NCAAF_OWNER_REVIEW_CLOCK_CONFLICT'), ('missing_rules','NCAAF_OWNER_REVIEW_SUBJECT_CONFLICT'),
    ('future_dependency','NCAAF_DEPENDENCY_SUBJECT_FUTURE_FACT')])
def test_rejects_before_numerical_inference(synthetic,monkeypatch,change,reason):
    packet,row=fixture(synthetic,monkeypatch); p=packet['payload']; c=p['custody_packet']; n=c['payload']['native_packet'];o=n['payload']['observation']['payload']
    if change=='subject':p['owner_review']['subject_sha256']='e'*64
    elif change=='conclusion':p['owner_review']['conclusions']['event_mapping']['conclusion']='UNKNOWN'
    elif change=='reviewer':p['owner_review']['reviewer']='Another owner'
    elif change=='late_review':p['owner_review']['reviewed_at']='2026-10-09T12:00:05Z'
    elif change=='borrowed_independent':c['payload']['version']=custody.VERSION
    elif change=='corrupt_body':c['payload']['response_objects'][0]['payload']['body_b64']=base64.b64encode(b'[]').decode()
    elif change=='stale':p['owner_review']['reviewed_at']='2026-10-10T12:00:03Z'
    elif change=='missing_rules':o['quote']['rules']=''
    elif change=='future_dependency':
        d=p['dependency_review'];d['dependency_verification']['verified_at']='2026-10-09T12:00:01.500000Z'
        d['owner_verification']['verification_sha256']=model.digest(d['dependency_verification'])
        c['payload']['custody_admission']['verification']['native_verification_sha256']=model.digest(d['dependency_verification'])
        c['payload']['custody_admission']['owner_verification']['verification_sha256']=model.digest(c['payload']['custody_admission']['verification'])
    # Consistent hash rewrites cannot replace semantic clock/custody checks.
    n['payload']['observation']['sha256']=model.digest(o); n['sha256']=model.digest(n['payload']);c['sha256']=model.digest(c['payload'])
    if change in {'corrupt_body','future_dependency'}:verify_owner(packet)
    else:packet['sha256']=model.digest(p)
    monkeypatch.setattr(research,'centers',Mock(side_effect=AssertionError('Rejected before numerical inference')))
    try:
        with adapter.selected([packet],private_research=True): result=adapter.predict(row)
    except ValueError as exc: assert str(exc)==reason
    else: assert result['ml_unavailable_reason']==reason
    research.centers.assert_not_called()


def test_synthetic_clock_isolation_does_not_renew_quotes_or_started_games(synthetic,monkeypatch):
    from app_core import candidate_chronology
    packet,row=fixture(synthetic,monkeypatch)
    old=deepcopy(packet)
    numeric=Mock(side_effect=AssertionError('Expired evidence cannot infer'))
    monkeypatch.setattr(research,'centers',numeric)
    with adapter.selected([packet],private_research=True):
        with pytest.raises(ValueError) as rejected:
            owner.infer(packet,(NOW+timedelta(days=2)).isoformat())
    assert str(rejected.value) != 'NCAAF_PRIVATE_RESEARCH_NOT_SELECTED'
    assert packet==old
    classified=candidate_chronology.classify(row,as_of=row['game_start_utc'])
    assert classified['candidate_context']!='CURRENT_PREGAME'
    numeric.assert_not_called()


def test_actual_capture_export_replay_private_display_and_public_exclusion(synthetic,monkeypatch,tmp_path):
    from app_core import research_replay
    from app_core.per_game_boards import per_game_board
    from app_core.public_board import build_package,validate_package
    from scripts.publish_board import render
    import test_source_contract_pipeline as ef
    packet,_=fixture(synthetic,monkeypatch); original_packet=deepcopy(packet)
    analysis,diagnostics=actual(monkeypatch,packet)
    row=analysis.loc[analysis.market_type.eq('spread_home')].iloc[0]
    assert row.ml_inference_status=='success',row.ml_unavailable_reason
    monkeypatch.setattr(ef,'NOW',NOW);monkeypatch.setattr(ef,'CAPTURE',(NOW+timedelta(seconds=1)).isoformat())
    captured=pipeline_fixture.exported(monkeypatch,tmp_path/'export',analysis)
    frames=[per_game_board(captured['card'],captured['captured'],family=f,novig_only=True,college_fallback=True,private_research=True) for f in ('overall','sides','totals')]
    display=json.loads(frames[1].iloc[0].research_display)
    assert display['probability']==row.ml_probability and display['availability_reason']=='AVAILABLE'
    assert 'OWNER_REVIEWED' in display['basis'] and 'PRIVATE_RESEARCH' in display['basis']
    assert display['ev'] is None and display['edge'] is None
    binding=json.loads(row.ml_estimate_metadata)['ncaaf_inputs']['payload']['consumed_reader']
    components=binding['owner_reader']['custody_reader']['custody_reader']
    assert components['version']==owner.ReviewPolicy.VERSION
    assert set(components['native_reader']['private_components'])=={'app_core.ncaaf_prospective_chronology','app_core.ncaaf_owner_mapping'}
    # Even an old positive final ticket cannot lend authority to this mode.
    claimed=captured['card'].copy()
    claimed['Bettable']=True;claimed['Play_Stake']=25.;claimed['Status']='APPROVED'
    private=per_game_board(claimed,captured['captured'],family='sides',novig_only=True,college_fallback=True,private_research=True)
    assert all(private.status=='PASS') and all(private.Play_Stake==0) and all(private.Trial_Stake==0)
    package=build_package(*frames);validate_package(package)
    assert package['games']['sides'][0]['research_display']['probability'] is None
    assert package['games']['sides'][0]['research_display']['availability_reason']=='NCAAF_PRIVATE_RESEARCH_ONLY'
    assert package['games']['sides'][0]['win_estimate'] is None and package['games']['sides'][0]['ev'] is None
    assert all(c['production_bet_amount']==0 for c in captured['card'].wager_contract)
    assert all(r['status']=='PASS' for r in package['games']['sides'])
    exported=research_replay.retain_export(frames,package,captured['card'],captured['captured'],path=captured['db'])
    replay,sources=research_replay.read_export(exported['export_id'],path=captured['db'])
    retained=research_replay.frame_from_payload(next(iter(sources.values()))['original']['producer'])
    recorded=retained.loc[retained.market_type.eq('spread_home')].iloc[0]
    saved=json.loads(recorded.ml_estimate_metadata)['ncaaf_inputs']['payload']
    assert saved['original_packet']==original_packet and packet==original_packet
    assert json.loads(frames[1].iloc[0].research_display)['probability']==row.ml_probability
    public=json.dumps(package)+render(package)
    assert all(k not in public for k in ('body_b64','conclusions','owner_review','SYNTHETIC_PRIVATE_BODY_CANARY'))


def test_independent_packet_cannot_borrow_private_route(synthetic,monkeypatch):
    strict,row,_=previous.fixture(synthetic,monkeypatch)
    packet,_=fixture(synthetic,monkeypatch)
    packet['payload']['custody_packet']=strict
    verify_owner(packet)
    with pytest.raises(ValueError,match='NCAAF_CUSTODY_SCHEMA'):owner.load(packet)


@pytest.mark.parametrize('minutes', [16, 1441, -1])
def test_actual_quote_freshness_and_future_clocks_remain_rejected(synthetic,monkeypatch,minutes):
    packet,row=fixture(synthetic,monkeypatch)
    monkeypatch.setattr(adapter,'generated_time',lambda:(NOW+timedelta(minutes=minutes)).isoformat())
    monkeypatch.setattr(research,'centers',Mock(side_effect=AssertionError('No stale/future inference')))
    with adapter.selected([packet],private_research=True): result=adapter.predict(row)
    assert result['ml_inference_status']=='unavailable'
    assert result['ml_unavailable_reason']=={16:'NCAAF_COMPAT_QUOTE_CLOCK_CONFLICT',1441:'NCAAF_FUTURE_OR_STALE_DEPENDENCY',-1:'NCAAF_OWNER_REVIEW_CLOCK_CONFLICT'}[minutes]
    research.centers.assert_not_called()


def test_rejected_private_packet_keeps_independent_slate_coverage(synthetic,monkeypatch):
    packet,_=fixture(synthetic,monkeypatch)
    packet['payload']['custody_packet']['payload']['response_objects'].pop()
    c=packet['payload']['custody_packet'];c['sha256']=model.digest(c['payload'])
    verify_owner(packet)
    monkeypatch.setattr(research,'centers',Mock(side_effect=AssertionError('No missing-response inference')))
    analysis,diagnostics=actual(monkeypatch,packet)
    from app_core.slate_coverage import build_coverage,native_ncaaf
    run='20261009T120004.000000Z'
    coverage=build_coverage([native_ncaaf(diagnostics['ncaaf_schedule'],'2026-10-10')],selected_date='2026-10-10',
        as_of=NOW.isoformat(),run_id=run,candidates=analysis.assign(export_run_id=run).to_dict('records'),
        provider_health={'sports':{'americanfootball_ncaaf':{'outcome':'SUCCESS','processing':'SUCCESS'}}})
    assert len(coverage['decisions'])==2
    assert all(d['coverage_decision_state']=='UNVERIFIED' for d in coverage['decisions'])
    assert 'NCAAF_ORIGINAL_RESPONSE_UNAVAILABLE' in json.dumps(coverage['decisions'])
    research.centers.assert_not_called()


def test_private_panel_renders_recorded_probability_without_running_analysis(synthetic,monkeypatch,tmp_path):
    from streamlit.testing.v1 import AppTest
    import test_source_contract_pipeline as ef
    packet,row=fixture(synthetic,monkeypatch)
    analysis,_=actual(monkeypatch,packet)
    monkeypatch.setattr(ef,'NOW',NOW);monkeypatch.setattr(ef,'CAPTURE',(NOW+timedelta(seconds=1)).isoformat())
    captured=pipeline_fixture.exported(monkeypatch,tmp_path/'export',analysis)
    script=tmp_path/'SYNTHETIC_private_panel.py'
    script.write_text('import streamlit as st\nfrom app.ui.ncaaf_pipeline_research import render\nrender(st.session_state["SYNTHETIC_games"],st.session_state["SYNTHETIC_candidates"])',encoding='utf-8')
    at=AppTest.from_file(str(script))
    at.session_state['SYNTHETIC_games']=captured['card']
    at.session_state['SYNTHETIC_candidates']=captured['captured']
    at.session_state['ncaaf_private_research']=True
    monkeypatch.setattr(research,'centers',Mock(side_effect=AssertionError('Display must not infer')))
    at.run(timeout=20)
    assert not at.exception
    assert at.metric[0].label=='OWNER_REVIEWED / PRIVATE_RESEARCH probability'
    assert at.metric[0].value=='63.68%'
    assert any('not independent acceptance' in c.value for c in at.caption)
    research.centers.assert_not_called()
