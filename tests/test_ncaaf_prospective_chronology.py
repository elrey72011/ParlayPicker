"""SYNTHETIC clocks, artifacts and bytes only; no historical reconstruction."""
from copy import deepcopy
from datetime import timedelta
import hashlib
import json
from unittest.mock import Mock

import pytest
import pandas as pd
from app_core import ncaaf_prospective_chronology as chronology
from app_core import ncaaf_compatible_pipeline as caller, ncaaf_pipeline_evidence as adapter
from app_core import ncaaf_compatible_observation as v2, ncaaf_model_compatibility as model
from app_core import ncaaf_research as research, research_replay
from app_core.research_estimate_trace import encode
from scripts.benchmark_drive_history_loading import blocked_network
import test_ncaaf_compatible_pipeline as legacy
from test_ncaaf_model_compatibility import synthetic, AT

NOW = legacy.NOW + timedelta(seconds=2)


def advance_dependency_permission():
    """Independent permission prepared before any future synthetic bytes exist."""
    return dict(review_id='SYNTHETIC-CFBD-advance-permission', reviewer='SYNTHETIC-permission-reviewer',
        reviewed_at='2026-10-09T11:00:00Z', provider='cfbd', endpoints=['games','games/teams'], season=2026,
        permitted_uses=['collection','private_retention','prospective_research_features','public_derived_output'],
        rights_edition='SYNTHETIC-CFBD-edition-1', rights_document_sha256='c'*64,
        effective_from='2026-10-01T00:00:00Z', effective_until='2026-11-01T00:00:00Z',
        capture_clock_field='retrieved_at', capture_clock_meaning='local_native_batch_capture')


def dependency_admission(packet, permission):
    verified = dict(verified_at='2026-10-09T12:00:02Z', verifier='SYNTHETIC-dependency-verifier',
        subject_version=chronology.DEPENDENCY_SUBJECT_VERSION, subject_sha256=chronology.dependency_subject_hash(packet),
        permissions_sha256=model.digest(permission),
        dependency_hashes=[v['sha256'] for v in packet['payload']['dependency_objects']])
    accepted = dict(review_id='SYNTHETIC-CFBD-independent-admission', reviewer='SYNTHETIC-independent-reviewer',
        accepted_at='2026-10-09T12:00:03Z', subject_version=chronology.DEPENDENCY_SUBJECT_VERSION,
        subject_sha256=verified['subject_sha256'], verification_sha256=model.digest(verified))
    return dict(version=chronology.DEPENDENCY_REVIEW_VERSION, permissions_review=deepcopy(permission),
        dependency_verification=verified, acceptance=accepted)


def trust_dependencies(review, monkeypatch, *, accept=True):
    permission, admission = review['permissions_review'], review['acceptance']
    monkeypatch.setattr(chronology, 'ACCEPTED_DEPENDENCY_PERMISSIONS',
        {permission['review_id']:model.digest(permission)} if accept else {})
    monkeypatch.setattr(chronology, 'ACCEPTED_DEPENDENCY_ADMISSIONS',
        {admission['review_id']:model.digest(admission)} if accept else {})


@pytest.fixture(autouse=True)
def offline():
    with blocked_network():
        yield


def seal(packet, monkeypatch, *, accept=True, permission=None):
    """Only labelled tests may manufacture trusted synthetic acceptance."""
    prior = adapter.ACCEPTED_PACKETS.get(packet['sha256'], {}).get('dependency_source_review', {})
    permission = deepcopy(permission or prior.get('permissions_review') or advance_dependency_permission())
    p = packet['payload']['observation']['payload']
    review = p['source_review']
    review['acceptance']['subject_sha256'] = chronology.subject_hash(p)
    packet['payload']['observation']['sha256'] = model.digest(p)
    packet['sha256'] = model.digest(packet['payload'])
    monkeypatch.setattr(chronology, 'ACCEPTED_TERMS_REVIEWS', {review['terms_review']['review_id']:model.digest(review['terms_review'])} if accept else {})
    monkeypatch.setattr(chronology, 'ACCEPTED_ADMISSIONS', {review['acceptance']['review_id']:model.digest(review['acceptance'])} if accept else {})
    dependency_review = dependency_admission(packet, permission)
    trust_dependencies(dependency_review, monkeypatch, accept=accept)
    monkeypatch.setattr(adapter, 'ACCEPTED_PACKETS', {packet['sha256']:dict(source_review_sha256=model.digest(review),
        public_derived_output='permitted', dependency_source_review=dependency_review)} if accept else {})
    return packet


def fixture(synthetic, monkeypatch, kind='spread_home', line=-3.5, *, permission=None):
    permission = deepcopy(permission or advance_dependency_permission())
    monkeypatch.setattr(chronology, 'ACCEPTED_DEPENDENCY_PERMISSIONS',
        {permission['review_id']:model.digest(permission)})
    packet, row = legacy.fixture(synthetic, monkeypatch, kind, line)
    p = packet['payload']['observation']['payload'];q = p['quote']
    p['version'] = chronology.VERSION
    p['as_of'] = '2026-10-09T12:00:03Z'
    terms = dict(review_id='SYNTHETIC-advance-terms',reviewer='SYNTHETIC-source-reviewer',reviewed_at='2026-10-09T11:00:00Z',
        effective_from='2026-10-01T00:00:00Z',effective_until='2026-11-01T00:00:00Z',
        operator=q['operator'],product=q['product'],listing_id=q['listing_id'],period=q['period'],settlement=q['rules'],
        rule_edition='SYNTHETIC-edition-1',rules_document_sha256='a'*64,rights_document_sha256='b'*64,
        permitted_uses=['collection','private_retention','research','public_derived_output'],provider_clock_field='recorded_at',provider_clock_meaning='provider_market_last_update')
    observed = dict(observed_at='2026-10-09T12:00:01Z',quote_sha256=model.digest(q),provider_clock_field='recorded_at',provider_clock_meaning='provider_market_last_update')
    verified = dict(verified_at='2026-10-09T12:00:02Z',verifier='SYNTHETIC-offer-verifier',quote_sha256=model.digest(q),
        observation_sha256=model.digest(observed),terms_sha256=model.digest(terms),mapping_sha256=model.digest(p['mapping_review']),rule_edition=terms['rule_edition'])
    accepted = dict(review_id='SYNTHETIC-independent-admission',reviewer='SYNTHETIC-independent-reviewer',accepted_at='2026-10-09T12:00:03Z',subject_version=chronology.VERSION,subject_sha256='')
    p['source_review'] = dict(version=chronology.REVIEW_VERSION,terms_review=terms,quote_observation=observed,offer_verification=verified,acceptance=accepted)
    packet['payload']['version'] = caller.SUCCESSOR_VERSION
    seal(packet,monkeypatch,permission=permission)
    monkeypatch.setattr(adapter,'generated_time',lambda:NOW.isoformat())
    return packet,row


def refreshed_hashes(packet):
    p=packet['payload']['observation']['payload']
    packet['payload']['observation']['sha256']=model.digest(p)
    packet['sha256']=model.digest(packet['payload'])


def test_v2_rejection_and_explicit_successor_static(synthetic,monkeypatch):
    old,_=legacy.fixture(synthetic,monkeypatch)
    p=old['payload']['observation']['payload'];p['as_of']=NOW.isoformat();p['source_review']['reviewed_at']='2026-10-09T12:00:02Z'
    legacy.bind(old,monkeypatch)
    immutable=deepcopy(old)
    with pytest.raises(ValueError,match='^NCAAF_COMPAT_SOURCE_REVIEW_CLOCK_CONFLICT$'):v2.read_observation(old['payload']['observation'])
    assert old==immutable
    packet,_=fixture(synthetic,monkeypatch)
    checked=chronology.read_observation(packet['payload']['observation'])
    assert checked['probability'] is None and not checked['feature_derivation_verified'] and not checked['dependency_objects_verified']
    with pytest.raises(ValueError,match='^NCAAF_COMPAT_OBSERVATION_SCHEMA$'):v2.read_observation(packet['payload']['observation'])


@pytest.mark.parametrize('change,reason',[
    ('unknown_meaning','NCAAF_QUOTE_CLOCK_MEANING_UNAVAILABLE'),('missing_observation','NCAAF_QUOTE_OBSERVATION_MISSING_OR_CONFLICT'),
    ('before_observation','NCAAF_EXACT_OFFER_CLOCK_CONFLICT'),('late_verification','NCAAF_ACCEPTANCE_CLOCK_CONFLICT'),
    ('late_acceptance','NCAAF_ACCEPTANCE_CLOCK_CONFLICT'),('late_advance_review','NCAAF_ADVANCE_REVIEW_CLOCK_CONFLICT'),
    ('expired','NCAAF_TERMS_NOT_EFFECTIVE'),('not_effective','NCAAF_TERMS_NOT_EFFECTIVE'),
    ('unknown_provider_clock','NCAAF_QUOTE_CLOCK_MEANING_UNAVAILABLE'),('wrong_price','NCAAF_QUOTE_OBSERVATION_MISSING_OR_CONFLICT'),
    ('wrong_line','NCAAF_QUOTE_OBSERVATION_MISSING_OR_CONFLICT'),('wrong_listing','NCAAF_ADVANCE_TERMS_IDENTITY_CONFLICT'),
    ('wrong_product','NCAAF_ADVANCE_TERMS_IDENTITY_CONFLICT'),('wrong_orientation','NCAAF_COMPAT_QUOTE_IDENTITY_CONFLICT'),
    ('wrong_rule_edition','NCAAF_EXACT_OFFER_VERIFICATION_CONFLICT'),('self_acceptance','NCAAF_INDEPENDENT_ACCEPTANCE_CONFLICT'),
    ('source_not_accepted','NCAAF_ADVANCE_TERMS_NOT_ACCEPTED'),('admission_not_trusted','NCAAF_INDEPENDENT_ACCEPTANCE_NOT_TRUSTED'),
    ('changed_subject','NCAAF_ADMISSION_SUBJECT_CONFLICT'),('changed_receipt','NCAAF_EXACT_OFFER_VERIFICATION_CONFLICT'),
    ('borrowed_v2','NCAAF_COMPAT_OBSERVATION_SCHEMA'),('permissions','NCAAF_ADVANCE_PERMISSIONS_UNAVAILABLE'),
])
def test_rehashed_negative_before_inference(synthetic,monkeypatch,change,reason):
    packet,row=fixture(synthetic,monkeypatch);p=packet['payload']['observation']['payload'];r=p['source_review']
    terms,observed,verified,accepted=[r[k] for k in ('terms_review','quote_observation','offer_verification','acceptance')]
    if change=='unknown_meaning':terms['provider_clock_meaning']=observed['provider_clock_meaning']='UNKNOWN'
    elif change=='missing_observation':observed['observed_at']=None
    elif change=='before_observation':verified['verified_at']='2026-10-09T12:00:00Z'
    elif change=='late_verification':verified['verified_at']='2026-10-09T12:00:05Z'
    elif change=='late_acceptance':accepted['accepted_at']='2026-10-09T12:00:05Z'
    elif change=='late_advance_review':terms['reviewed_at']='2026-10-09T12:00:01Z'
    elif change=='expired':terms['effective_until']='2026-10-09T12:00:03Z'
    elif change=='not_effective':terms['effective_from']='2026-10-09T12:00:01Z'
    elif change=='unknown_provider_clock':observed['provider_clock_field']='observed_at'
    elif change=='wrong_price':p['quote']['price']=-120
    elif change=='wrong_line':p['quote']['point']=-4.5
    elif change=='wrong_listing':p['quote']['listing_id']='other-listing'
    elif change=='wrong_product':p['quote']['product']='other-product'
    elif change=='wrong_orientation':p['quote']['event_home_team']='Georgia'
    elif change=='wrong_rule_edition':verified['rule_edition']='other-edition'
    elif change=='self_acceptance':accepted['reviewer']=verified['verifier']
    elif change=='permissions':terms['permitted_uses']=[]
    elif change=='borrowed_v2':p['version']=v2.VERSION
    # Rehashing and even independently accepting a contradictory subject may
    # not bypass mechanical chronology/identity restrictions.
    if change not in {'changed_receipt','wrong_rule_edition'}:
        verified['terms_sha256']=model.digest(terms);verified['observation_sha256']=model.digest(observed)
    if change=='changed_receipt':observed['observed_at']='2026-10-09T12:00:00Z'
    seal(packet,monkeypatch)
    if change=='source_not_accepted':monkeypatch.setattr(chronology,'ACCEPTED_TERMS_REVIEWS',{})
    elif change=='admission_not_trusted':monkeypatch.setattr(chronology,'ACCEPTED_ADMISSIONS',{})
    elif change=='changed_subject':p['features']['values'][0]+=1;refreshed_hashes(packet)
    monkeypatch.setattr(research,'centers',Mock(side_effect=AssertionError('Rejected inputs must not infer')))
    if change=='borrowed_v2':
        with pytest.raises(ValueError,match='^'+reason+'$'):caller.load(packet)
    else:
        with pytest.raises(ValueError,match='^'+reason+'$'):caller.infer(packet,NOW.isoformat())
    research.centers.assert_not_called()


@pytest.mark.parametrize('at',[NOW-timedelta(seconds=2),NOW+timedelta(minutes=16),NOW+timedelta(days=2)])
def test_actual_inference_clock_cannot_borrow_admission(synthetic,monkeypatch,at):
    packet,_=fixture(synthetic,monkeypatch)
    with pytest.raises(ValueError,match='NCAAF_COMPAT_NEW_INFERENCE_CLOCK_CONFLICT'):caller.infer(packet,at.isoformat())


def test_future_quote_and_missing_provider_clock(synthetic,monkeypatch):
    packet,_=fixture(synthetic,monkeypatch)
    for clock in (None,(NOW+timedelta(seconds=1)).isoformat()):
        p=deepcopy(packet);p['payload']['observation']['payload']['quote']['recorded_at']=clock
        refreshed_hashes(p)
        with pytest.raises(ValueError,match='NCAAF_COMPAT_NEW_INFERENCE_CLOCK_CONFLICT'):caller.infer(p,NOW.isoformat())


def test_dependency_permission_must_precede_capture(synthetic,monkeypatch):
    packet,_=fixture(synthetic,monkeypatch)
    approval=adapter.ACCEPTED_PACKETS[packet['sha256']]
    approval['dependency_source_review']['permissions_review']['reviewed_at']='2026-10-09T12:00:01Z'
    trust_dependencies(approval['dependency_source_review'],monkeypatch)
    approval['dependency_source_review']['dependency_verification']['permissions_sha256']=model.digest(approval['dependency_source_review']['permissions_review'])
    approval['dependency_source_review']['acceptance']['verification_sha256']=model.digest(approval['dependency_source_review']['dependency_verification'])
    trust_dependencies(approval['dependency_source_review'],monkeypatch)
    with pytest.raises(ValueError,match='NCAAF_DEPENDENCY_ADVANCE_PERMISSION_CLOCK_CONFLICT'):
        caller.accepted_dependencies(packet,approval,NOW.isoformat())


def test_immutable_advance_permission_then_exact_bytes_through_actual_caller(synthetic,monkeypatch):
    permission = advance_dependency_permission()
    original = deepcopy(permission)
    original_hash = model.digest(permission)
    assert not any(k in permission for k in ('dependency_hashes','input_sha256','verification_sha256'))
    # Permission is independently trusted before the fixture produces previously
    # unknown synthetic native bytes. It is never amended with those hashes.
    monkeypatch.setattr(chronology,'ACCEPTED_DEPENDENCY_PERMISSIONS',{permission['review_id']:original_hash})
    packet,_ = fixture(synthetic,monkeypatch,permission=permission)
    approval = adapter.ACCEPTED_PACKETS[packet['sha256']]
    review = deepcopy(approval['dependency_source_review'])
    assert review['permissions_review'] == permission == original
    assert review['dependency_verification']['permissions_sha256'] == original_hash
    assert review['dependency_verification']['verified_at'] == '2026-10-09T12:00:02Z'
    assert review['acceptance']['accepted_at'] == '2026-10-09T12:00:03Z'
    monkeypatch.setattr(legacy,'NOW',NOW)
    analysis,_ = legacy.actual(monkeypatch,packet)
    row = analysis.loc[analysis.market_type.eq('spread_home')].iloc[0]
    assert row.ml_inference_status == 'success', row.ml_unavailable_reason
    retained = json.loads(row.ml_estimate_metadata)['ncaaf_inputs']['payload']['consumed_dependency_review']
    assert retained == review and retained['permissions_review'] == original
    assert permission == original and model.digest(permission) == original_hash
    assert chronology.ACCEPTED_DEPENDENCY_PERMISSIONS[permission['review_id']] == original_hash
    assert approval['dependency_source_review'] == review
    assert adapter.diagnose(row.to_dict()) == dict(status='COMPLETE',reason='AVAILABLE')


@pytest.mark.parametrize('change,reason',[
    ('legacy_receipt','NCAAF_DEPENDENCY_ADMISSION_SCHEMA'),
    ('future_hashes_in_advance','NCAAF_DEPENDENCY_PERMISSION_SCOPE_CONFLICT'),
    ('wrong_season','NCAAF_DEPENDENCY_PERMISSION_SCOPE_CONFLICT'),
    ('wrong_endpoint','NCAAF_DEPENDENCY_PERMISSION_SCOPE_CONFLICT'),
    ('missing_permission','NCAAF_DEPENDENCY_PERMISSION_UNAVAILABLE'),
    ('unknown_clock','NCAAF_DEPENDENCY_CLOCK_MEANING_UNAVAILABLE'),
    ('untrusted_permission','NCAAF_DEPENDENCY_PERMISSIONS_NOT_TRUSTED'),
    ('late_permission','NCAAF_DEPENDENCY_ADVANCE_PERMISSION_CLOCK_CONFLICT'),
    ('not_effective','NCAAF_DEPENDENCY_TERMS_NOT_EFFECTIVE'),
    ('expired','NCAAF_DEPENDENCY_TERMS_NOT_EFFECTIVE'),
    ('verification_before_capture','NCAAF_DEPENDENCY_VERIFICATION_CLOCK_CONFLICT'),
    ('verification_at_capture','NCAAF_DEPENDENCY_VERIFICATION_CLOCK_CONFLICT'),
    ('wrong_permission_link','NCAAF_DEPENDENCY_VERIFICATION_CONFLICT'),
    ('wrong_byte_hashes','NCAAF_DEPENDENCY_VERIFICATION_CONFLICT'),
    ('wrong_input','NCAAF_DEPENDENCY_VERIFICATION_CONFLICT'),
    ('acceptance_before_verification','NCAAF_DEPENDENCY_ACCEPTANCE_CLOCK_CONFLICT'),
    ('acceptance_after_inference','NCAAF_DEPENDENCY_ACCEPTANCE_CLOCK_CONFLICT'),
    ('acceptance_after_full_admission','NCAAF_DEPENDENCY_ACCEPTANCE_CLOCK_CONFLICT'),
    ('self_acceptance','NCAAF_DEPENDENCY_ACCEPTANCE_CONFLICT'),
    ('untrusted_acceptance','NCAAF_DEPENDENCY_ACCEPTANCE_NOT_TRUSTED'),
    ('rehash_without_trust','NCAAF_DEPENDENCY_ACCEPTANCE_NOT_TRUSTED'),
])
def test_dependency_successor_rejects_before_actual_inference(synthetic,monkeypatch,change,reason):
    packet,_ = fixture(synthetic,monkeypatch)
    review = adapter.ACCEPTED_PACKETS[packet['sha256']]['dependency_source_review']
    permission,verified,accepted = [review[k] for k in ('permissions_review','dependency_verification','acceptance')]
    if change=='future_hashes_in_advance':permission['dependency_hashes']=verified['dependency_hashes']
    elif change=='wrong_season':permission['season']=2025
    elif change=='wrong_endpoint':permission['endpoints']=['games']
    elif change=='missing_permission':permission['permitted_uses']=[]
    elif change=='unknown_clock':permission['capture_clock_meaning']='UNKNOWN'
    elif change=='late_permission':permission['reviewed_at']='2026-10-09T12:00:01Z'
    elif change=='not_effective':permission['effective_from']='2026-10-09T12:00:01Z'
    elif change=='expired':permission['effective_until']=NOW.isoformat()
    elif change=='verification_before_capture':verified['verified_at']='2026-10-09T11:59:59Z'
    elif change=='verification_at_capture':verified['verified_at']='2026-10-09T12:00:00Z'
    elif change=='wrong_permission_link':verified['permissions_sha256']='d'*64
    elif change=='wrong_byte_hashes':verified['dependency_hashes']=['d'*64]
    elif change=='wrong_input':verified['subject_sha256']='d'*64
    elif change=='acceptance_before_verification':accepted['accepted_at']='2026-10-09T12:00:01Z'
    elif change=='acceptance_after_inference':accepted['accepted_at']='2026-10-09T12:00:05Z'
    elif change=='acceptance_after_full_admission':accepted['accepted_at']='2026-10-09T12:00:03.500000Z'
    elif change=='self_acceptance':accepted['reviewer']=verified['verifier']
    elif change=='rehash_without_trust':verified['verifier']='SYNTHETIC-other-verifier'
    if change!='wrong_permission_link':verified['permissions_sha256']=model.digest(permission)
    accepted['verification_sha256']=model.digest(verified)
    if change!='rehash_without_trust':trust_dependencies(review,monkeypatch)
    if change=='untrusted_permission':monkeypatch.setattr(chronology,'ACCEPTED_DEPENDENCY_PERMISSIONS',{})
    elif change=='untrusted_acceptance':monkeypatch.setattr(chronology,'ACCEPTED_DEPENDENCY_ADMISSIONS',{})
    elif change=='legacy_receipt':adapter.ACCEPTED_PACKETS[packet['sha256']]['dependency_source_review']=dict(
        provider='cfbd',endpoints=['games','games/teams'],dependency_hashes=verified['dependency_hashes'],
        permitted_use='prospective_research_features',public_derived_output='permitted',
        rights_document='SYNTHETIC old receipt',reviewed_at='2026-10-09T11:00:00Z',effective_until='2026-11-01T00:00:00Z')
    monkeypatch.setattr(legacy,'NOW',NOW)
    monkeypatch.setattr(research,'centers',Mock(side_effect=AssertionError('Rejected dependencies cannot infer')))
    analysis,_ = legacy.actual(monkeypatch,packet)
    row = analysis.loc[analysis.market_type.eq('spread_home')].iloc[0]
    assert row.ml_unavailable_reason == reason
    assert row.ml_inference_status == 'unavailable'
    research.centers.assert_not_called()


def test_original_v2_dependency_review_behavior_is_unchanged(synthetic,monkeypatch):
    packet,_ = legacy.fixture(synthetic,monkeypatch)
    approval = adapter.ACCEPTED_PACKETS[packet['sha256']]
    approval['dependency_source_review']['reviewed_at']='2026-10-09T12:00:01Z'
    original = deepcopy(approval['dependency_source_review'])
    assert caller.accepted_dependencies(packet,approval,NOW.isoformat()) == original
    successor,_ = fixture(synthetic,monkeypatch)
    successor_review = adapter.ACCEPTED_PACKETS[successor['sha256']]['dependency_source_review']
    with pytest.raises(ValueError,match='^NCAAF_COMPAT_DEPENDENCY_SOURCE_REVIEW_MISSING_OR_CONFLICT$'):
        caller.accepted_dependencies(packet,dict(dependency_source_review=successor_review),NOW.isoformat())
    assert approval['dependency_source_review'] == original


def test_production_dependency_acceptance_catalogs_remain_empty():
    assert chronology.ACCEPTED_DEPENDENCY_PERMISSIONS == {}
    assert chronology.ACCEPTED_DEPENDENCY_ADMISSIONS == {}


def test_dependency_subject_exists_before_later_acceptance_and_checkpoint(synthetic,monkeypatch):
    packet,_=fixture(synthetic,monkeypatch)
    original=deepcopy(packet)
    prior=deepcopy(packet)
    prior['payload']['observation']['payload']['source_review'].pop('acceptance')
    prior['payload']['observation']['payload'].pop('as_of')
    subject=chronology.dependency_subject_hash(prior)
    review=adapter.ACCEPTED_PACKETS[packet['sha256']]['dependency_source_review']
    assert review['dependency_verification']['subject_sha256']==subject
    assert chronology.dependency_subject_hash(packet)==subject
    assert packet==original
    for fact in ('quote','features','model'):
        changed=deepcopy(packet)
        changed['payload']['observation']['payload'][fact]['SYNTHETIC_altered_fact']=True
        assert chronology.dependency_subject_hash(changed)!=subject
    altered=deepcopy(packet)
    altered['payload']['dependency_objects'][0]['bytes_b64']='SYNTHETIC altered bytes'
    assert chronology.dependency_subject_hash(altered)!=subject


def test_later_checkpoint_still_requires_exact_full_admission(synthetic,monkeypatch):
    packet,_=fixture(synthetic,monkeypatch)
    subject=chronology.dependency_subject_hash(packet)
    packet['payload']['observation']['payload']['as_of']='2026-10-09T12:00:03.500000Z'
    refreshed_hashes(packet)
    assert chronology.dependency_subject_hash(packet)==subject
    monkeypatch.setattr(research,'centers',Mock(side_effect=AssertionError('Changed admission cannot infer')))
    with pytest.raises(ValueError,match='^NCAAF_ADMISSION_SUBJECT_CONFLICT$'):
        caller.infer(packet,NOW.isoformat())
    research.centers.assert_not_called()


def test_reported_future_dependency_subject_rejects_before_actual_inference(synthetic,monkeypatch):
    packet,_=fixture(synthetic,monkeypatch)
    p=packet['payload']['observation']['payload']
    p['features']['available_at']='2026-10-09T12:00:01.500000Z'
    seal(packet,monkeypatch)
    review=adapter.ACCEPTED_PACKETS[packet['sha256']]['dependency_source_review']
    review['dependency_verification']['verified_at']='2026-10-09T12:00:00.500000Z'
    review['acceptance']['verification_sha256']=model.digest(review['dependency_verification'])
    trust_dependencies(review,monkeypatch)
    original,original_review=deepcopy(packet),deepcopy(review)
    monkeypatch.setattr(legacy,'NOW',NOW)
    monkeypatch.setattr(research,'centers',Mock(side_effect=AssertionError('Future subject facts cannot infer')))
    analysis,diagnostics=legacy.actual(monkeypatch,packet)
    row=analysis.loc[analysis.market_type.eq('spread_home')].iloc[0]
    assert row.ml_unavailable_reason=='NCAAF_DEPENDENCY_SUBJECT_FUTURE_FACT'
    assert row.ml_inference_status=='unavailable'
    research.centers.assert_not_called()
    assert packet==original and review==original_review
    assert pd.isna(row.ml_probability)
    from app_core.slate_coverage import build_coverage,native_ncaaf
    run='20261009T120004.000000Z'
    report=build_coverage([native_ncaaf(diagnostics['ncaaf_schedule'],'2026-10-10')],selected_date='2026-10-10',
        as_of=NOW.isoformat(),run_id=run,candidates=analysis.assign(export_run_id=run).to_dict('records'),
        provider_health={'sports':{'americanfootball_ncaaf':{'outcome':'SUCCESS','processing':'SUCCESS'}}})
    first=next(r for r in report['decisions'] if 'Alabama' in r['home_team'])
    assert len(report['decisions'])==2 and first['coverage_decision_state']=='UNVERIFIED'
    assert 'NCAAF_DEPENDENCY_SUBJECT_FUTURE_FACT' in first['blocker_codes']


@pytest.mark.parametrize('fact',['observation','features','offer_verification','mapping','terms','quote','dependency','model'])
@pytest.mark.parametrize('clock',['2026-10-09T12:00:02.500000Z',None])
def test_included_subject_fact_must_exist_by_dependency_verification(synthetic,monkeypatch,fact,clock):
    packet,_=fixture(synthetic,monkeypatch)
    p=packet['payload']['observation']['payload'];review=p['source_review']
    if fact=='observation':review['quote_observation']['observed_at']=clock
    elif fact=='features':p['features']['available_at']=clock
    elif fact=='offer_verification':review['offer_verification']['verified_at']=clock
    elif fact=='mapping':p['mapping_review']['reviewed_at']=clock
    elif fact=='terms':review['terms_review']['reviewed_at']=clock
    elif fact=='quote':p['quote']['recorded_at']=clock
    elif fact=='dependency':p['feature_dependencies'][0]['available_at']=clock
    else:
        # Manufacture a separate SYNTHETIC record and binding, never alter the
        # authentic reviewed model. Missing model clocks already fail lineage.
        import base64
        original=json.loads(base64.b64decode(p['model']['original_record_b64']))
        original['created_at']=clock
        raw=model.encode(original);binding=deepcopy(model._binding())
        binding['original_record_sha256']=hashlib.sha256(raw).hexdigest()
        monkeypatch.setattr(model,'_binding',lambda:deepcopy(binding))
        p['model']['original_record_b64']=base64.b64encode(raw).decode()
        p['model']['compatibility']=model.envelope(family='spread')
    # Even independently trusted, consistently rehashed test receipts cannot
    # authorize facts that do not exist at dependency verification (12:00:02Z).
    review['quote_observation']['quote_sha256']=model.digest(p['quote'])
    review['offer_verification'].update(quote_sha256=model.digest(p['quote']),
        observation_sha256=model.digest(review['quote_observation']),
        terms_sha256=model.digest(review['terms_review']),mapping_sha256=model.digest(p['mapping_review']))
    seal(packet,monkeypatch)
    original=deepcopy(packet)
    monkeypatch.setattr(legacy,'NOW',NOW)
    monkeypatch.setattr(research,'centers',Mock(side_effect=AssertionError('Unavailable subject cannot infer')))
    analysis,_=legacy.actual(monkeypatch,packet)
    row=analysis.loc[analysis.market_type.eq('spread_home')].iloc[0]
    expected='NCAAF_COMPAT_LINEAGE_CLOCK_CONFLICT' if fact=='model' and clock is None else 'NCAAF_DEPENDENCY_SUBJECT_FUTURE_FACT'
    assert row.ml_inference_status=='unavailable' and row.ml_unavailable_reason==expected
    assert pd.isna(row.ml_probability) and packet==original
    research.centers.assert_not_called()


@pytest.mark.parametrize('kind,line',[('spread_home',-3.5),('spread_away',3.5),('total_over',51.5),('total_under',51.5)])
def test_actual_caller_capture_export_display(synthetic,monkeypatch,tmp_path,kind,line):
    packet,_=fixture(synthetic,monkeypatch,kind,line);original=deepcopy(packet)
    monkeypatch.setattr(legacy,'NOW',NOW)
    analysis,diagnostics=legacy.actual(monkeypatch,packet)
    row=analysis.loc[analysis.market_type.eq(kind)].iloc[0].to_dict()
    assert row['ml_inference_status']=='success',row['ml_unavailable_reason']
    assert adapter.diagnose(row)==dict(status='COMPLETE',reason='AVAILABLE')
    checked=chronology.read_observation(packet['payload']['observation'])
    features=dict(zip(research.FEATURES,checked['ordered_features']))
    family='total' if kind.startswith('total') else 'margin'
    expected=research.probabilities(float(research.centers(checked['fit'],[features],family)[0]),checked['fit']['sigma'],
        -line if kind=='spread_home' else line,total=family=='total')['over' if kind in {'spread_home','total_over'} else 'under']
    assert row['ml_probability']==expected and row['ml_probability']!=.99
    import test_source_contract_pipeline as ef
    monkeypatch.setattr(ef,'NOW',NOW);monkeypatch.setattr(ef,'CAPTURE',(NOW+timedelta(seconds=1)).isoformat())
    result=legacy.exported(monkeypatch,tmp_path/'export',analysis)
    from app_core.per_game_boards import per_game_board
    from app_core.public_board import build_package,validate_package
    from scripts.publish_board import render,assets_from_html
    result['frames']=[per_game_board(result['card'],result['captured'],family=f,novig_only=True,college_fallback=True) for f in ('overall','sides','totals')]
    package=json.loads(assets_from_html(render(build_package(*result['frames'])))['board-data.json']);validate_package(package)
    shown=package['games']['sides' if family=='margin' else 'totals'][0]['research_display']
    assert shown['availability_reason']=='AVAILABLE' and shown['probability']==expected and shown['ev'] is None and shown['edge'] is None
    assert 'SYNTHETIC' in shown['basis'] and 'uncalibrated' in shown['basis']
    receipt=research_replay.retain_export(result['frames'],package,result['card'],result['captured'],path=result['db'])
    _,sources=research_replay.read_export(receipt['export_id'],path=result['db'])
    producer=research_replay.frame_from_payload(next(iter(sources.values()))['original']['producer'])
    retained=producer.loc[producer.market_type.eq(kind)].iloc[0]
    evidence=json.loads(retained.ml_estimate_metadata)['ncaaf_inputs']['payload']
    assert evidence['version']==caller.SUCCESSOR_RESULT_VERSION and evidence['original_packet']==original
    assert evidence['consumed_dependency_review']==adapter.ACCEPTED_PACKETS[packet['sha256']]['dependency_source_review']
    assert evidence['consumed_dependency_review']['permissions_review']['reviewed_at']=='2026-10-09T11:00:00Z'
    assert evidence['consumed_dependency_review']['dependency_verification']['verified_at']=='2026-10-09T12:00:02Z'
    assert evidence['consumed_dependency_review']['acceptance']['accepted_at']=='2026-10-09T12:00:03Z'
    assert evidence['computation']['payload']['admission_receipts']==original['payload']['observation']['payload']['source_review']
    assert evidence['computation']['payload']['inference_time']==NOW.isoformat() and evidence['ui_refresh'] is None
    assert evidence['original_blend']['probability']['value']==retained.calibrated_probability
    assert evidence['consumed_reader']['chronology_reader']['version']==chronology.VERSION
    assert packet==original and all(c['production_bet_amount']==0 for c in result['card'].wager_contract)
    assert not result['captured'].production_eligible.fillna(False).any()
    assert all(k not in json.dumps(package) for k in ('admission_receipts','dependency_objects','bytes_b64','ncaaf_inputs','SYNTHETIC_UNUSED_SECRET_CANARY'))
    browser=legacy.inspect_browser(package,tmp_path/'browser',NOW)
    assert browser['initial']['current']==browser['initial']['top']==0 and browser['initial']['shown'][0]['probability']==expected


def test_rejected_chronology_remains_coverage_unverified(synthetic,monkeypatch):
    packet,_=fixture(synthetic,monkeypatch)
    monkeypatch.setattr(legacy,'NOW',NOW);monkeypatch.setattr(chronology,'ACCEPTED_ADMISSIONS',{})
    analysis,diagnostics=legacy.actual(monkeypatch,packet)
    row=analysis.loc[analysis.market_type.eq('spread_home')].iloc[0]
    assert row.ml_unavailable_reason=='NCAAF_INDEPENDENT_ACCEPTANCE_NOT_TRUSTED'
    from app_core.slate_coverage import build_coverage, native_ncaaf
    run='20261009T120004.000000Z'
    report=build_coverage([native_ncaaf(diagnostics['ncaaf_schedule'],'2026-10-10')],selected_date='2026-10-10',as_of=NOW.isoformat(),
        run_id=run,candidates=analysis.assign(export_run_id=run).to_dict('records'),
        provider_health={'sports':{'americanfootball_ncaaf':{'outcome':'SUCCESS','processing':'SUCCESS'}}})
    assert len(report['decisions'])==2 and all(r['coverage_decision_state']=='UNVERIFIED' for r in report['decisions'])
    first=next(r for r in report['decisions'] if 'Alabama' in r['home_team'])
    assert 'NCAAF_INDEPENDENT_ACCEPTANCE_NOT_TRUSTED' in first['blocker_codes']


def test_static_result_receipts_do_not_repeat_inference(synthetic,monkeypatch):
    packet,_=fixture(synthetic,monkeypatch)
    checked,center,probability,result=caller.infer(packet,NOW.isoformat())
    monkeypatch.setattr(research,'centers',Mock(side_effect=AssertionError('Static inspection cannot infer')))
    assert caller.inspect_result(packet,result,NOW.isoformat())[1]['raw_probability']==probability
    corrupted=deepcopy(result);corrupted['payload']['admission_receipts']['acceptance']['accepted_at']=NOW.isoformat()
    corrupted['sha256']=model.digest(corrupted['payload'])
    with pytest.raises(ValueError,match='NCAAF_COMPAT_COMPUTATION_RECEIPT_CONFLICT'):caller.inspect_result(packet,corrupted,NOW.isoformat())
    research.centers.assert_not_called()


def test_advance_review_can_precede_effective_terms(synthetic,monkeypatch):
    packet,_=fixture(synthetic,monkeypatch)
    p=packet['payload']['observation']['payload'];r=p['source_review']
    r['terms_review']['effective_from']='2026-10-09T11:30:00Z'
    r['offer_verification']['terms_sha256']=model.digest(r['terms_review'])
    seal(packet,monkeypatch)
    assert caller.infer(packet,NOW.isoformat())[2] > 0


def test_acceptance_must_strictly_precede_actual_inference(synthetic,monkeypatch):
    packet,_=fixture(synthetic,monkeypatch)
    with pytest.raises(ValueError,match='NCAAF_ACCEPTANCE_CLOCK_CONFLICT'):caller.infer(packet,'2026-10-09T12:00:03Z')


def test_acceptance_cannot_cover_not_yet_available_inputs(synthetic,monkeypatch):
    packet,_=fixture(synthetic,monkeypatch)
    p=packet['payload']['observation']['payload']
    p['as_of']=NOW.isoformat();p['features']['available_at']=NOW.isoformat()
    seal(packet,monkeypatch)
    with pytest.raises(ValueError,match='NCAAF_ADMISSION_SUBJECT_FUTURE_INPUT'):caller.infer(packet,(NOW+timedelta(seconds=1)).isoformat())


def test_consistently_rehashed_offer_cannot_create_trust(synthetic,monkeypatch):
    packet,_=fixture(synthetic,monkeypatch);p=packet['payload']['observation']['payload'];r=p['source_review']
    p['quote']['price']=-120
    r['quote_observation']['quote_sha256']=r['offer_verification']['quote_sha256']=model.digest(p['quote'])
    r['offer_verification']['observation_sha256']=model.digest(r['quote_observation'])
    r['acceptance']['subject_sha256']=chronology.subject_hash(p)
    refreshed_hashes(packet)
    with pytest.raises(ValueError,match='NCAAF_INDEPENDENT_ACCEPTANCE_NOT_TRUSTED'):caller.infer(packet,NOW.isoformat())


def test_missing_packet_selection_does_not_infer(synthetic,monkeypatch):
    packet,row=fixture(synthetic,monkeypatch)
    monkeypatch.setattr(research,'centers',Mock(side_effect=AssertionError('No selection must not infer')))
    with adapter.selected([]):result=adapter.predict(row)
    assert result['ml_unavailable_reason']=='NCAAF_EXACT_OFFER_NOT_SELECTED'
    research.centers.assert_not_called()


@pytest.mark.parametrize('attack',['dependency_missing','dependency_corrupt','integer','novig','feature_order'])
def test_successor_preserves_original_pipeline_barriers(synthetic,monkeypatch,attack):
    packet,_=fixture(synthetic,monkeypatch);p=packet['payload']['observation']['payload']
    if attack=='dependency_missing':packet['payload']['dependency_objects'].pop()
    elif attack=='dependency_corrupt':packet['payload']['dependency_objects'][0]['bytes_b64']='broken'
    elif attack=='integer':p['quote']['point']=-3
    elif attack=='novig':p['quote'].update(book='novig',operator='novig')
    else:p['features']['order'].reverse()
    seal(packet,monkeypatch)
    monkeypatch.setattr(research,'centers',Mock(side_effect=AssertionError('Rejected facts cannot infer')))
    with pytest.raises(ValueError):caller.infer(packet,NOW.isoformat())
    research.centers.assert_not_called()
