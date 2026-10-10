"""Static SYNTHETIC row/coverage reports; never reconstruct probabilities."""
from copy import deepcopy
import json
import pytest
from app_core.availability_inventory import inventory
from scripts.benchmark_drive_history_loading import blocked_network


def test_preserves_recorded_rejection_clocks_and_separate_availability():
    c=dict(league='NFL',market_type='spread_home',provider_event_id='SYNTHETIC-E',matchup_id='UNORDERED-LEGACY',ml_estimate_metadata=json.dumps(dict(
        inference_status='success',probability=dict(state='VALUE',value=.516),generated_at='2026-10-05T22:19:18Z',
        producer_contract=dict(event=dict(home='Home',away='Away'),offer=dict(line=-1.5,source_time='2026-10-05T22:19:05Z')))),
        research_estimate_trace=dict(first_rejection=dict(code='SOURCE_LISTING_BINDING_NOT_VERIFIED',stage='source_review')))
    before=deepcopy(c)
    with blocked_network():report=inventory([c],reference='SYNTHETIC-original.json')
    r=report['rows'][0]
    assert r['quote_clock']=='2026-10-05T22:19:05Z' and r['observation_clock'] is None and r['inference_clock']=='2026-10-05T22:19:18Z'
    assert r['first_actual_rejection_code']=='SOURCE_LISTING_BINDING_NOT_VERIFIED' and r['first_actual_rejection_stage']=='source_review'
    assert r['research_probability']['value']==.516 and r['wagering']['new_stake']==0 and not r['wagering']['new_authority']
    assert r['canonical_identity_status']=='UNKNOWN' and r['current_hosted_status']=='UNKNOWN'
    assert report['independent_schedule_denominator'] is None and c==before


def test_legacy_message_is_not_fabricated_code_or_schedule_completeness():
    with blocked_network():report=inventory([dict(league='NHL',ml_estimate_metadata=json.dumps(dict(inference_status='unavailable',reason='No compatible cover inputs retained')))],reference='SYNTHETIC')
    r=report['rows'][0]
    assert r['first_actual_rejection_code'] is None and r['recorded_rejection_message']=='No compatible cover inputs retained'
    assert report['inventory_status']=='UNKNOWN' and r['research_probability']['value'] is None


def test_conflicting_trace_identity_rejects():
    with pytest.raises(ValueError,match='AVAILABILITY_CONFLICTING_TRACE'):
        inventory([],traces=[dict(candidate_id='SYNTHETIC',rejection_code='A'),dict(candidate_id='SYNTHETIC',rejection_code='B')],reference='SYNTHETIC')


def test_null_trace_source_preserves_original_source_code():
    r=inventory([dict(league='MLB',research_estimate_trace=dict(source=None,first_rejection_stage='quote.source_contract',
        origin=dict(source_contract_diagnostics=['SOURCE_MARKET_LISTING_BINDING_NOT_VERIFIED'])))],reference='SYNTHETIC')['rows'][0]
    assert r['first_actual_rejection_code']=='SOURCE_MARKET_LISTING_BINDING_NOT_VERIFIED'
    assert r['required_evidence']=='independent_source_evidence' and r['research_probability']['value'] is None


def test_independent_schedule_only_rows_never_disappear():
    from app_core.slate_coverage import build_coverage,native_ncaaf
    from app_core.ncaaf_schedule import inventory_from_events
    from test_ncaaf_schedule_coverage import event
    inv=inventory_from_events([('FBS',[event('1','2026-10-10T16:00:00Z','Alabama','Georgia')]),
        ('FCS',[event('2','2026-10-10T16:00:00Z','McNeese Cowboys','Other FCS')])],'2026-10-10','2026-10-10',complete=True,observed_at='2026-10-09T12:00:00Z')
    coverage=build_coverage([native_ncaaf(inv,'2026-10-10')],selected_date='2026-10-10',as_of='2026-10-09T12:00:00Z',run_id='SYNTHETIC',candidates=[],leagues=['NCAAF'])
    report=inventory([],coverage=coverage,reference='SYNTHETIC')
    assert report['independent_schedule_denominator']==2 and len(report['rows'])==2
    assert all(r['quote_clock'] is None and r['inference_clock'] is None for r in report['rows'])
