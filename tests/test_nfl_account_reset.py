"""Synthetic account observations only; external transports remain blocked."""
from datetime import datetime, timedelta, timezone

import pytest

from app_core.nfl_private_workflow import account_reset_evidence
from nfl_workflow_fixtures import make_workflow, SyntheticClock, SyntheticProvider


@pytest.mark.parametrize('zone,expected,basis',[
    ('UNKNOWN','2026-11-30T00:00:00+00:00','date_start_minus_24h_unknown_timezone_v1'),
    ('UTC','2026-12-01T00:00:00+00:00','utc_date_start_v1'),
])
def test_date_only_keeps_original_precision_separate_from_bound(zone,expected,basis):
    reported=dict(precision='date',date='2026-12-01',timezone=zone)
    evidence,bound=account_reset_evidence(reset_observation=reported)
    assert evidence['reset_observation']==reported and evidence['reset_observation'] is not reported
    assert evidence['reset_time']=='UNKNOWN' and 'reset_at' not in evidence
    assert evidence['reset_not_before']==bound==expected and evidence['reset_bound_basis']==basis
    if zone=='UNKNOWN':assert evidence['supported_utc_offset_hours']==[-24,24]


def test_exact_time_authority_preserves_original_offset_and_string():
    value='2026-12-01T02:34:56-05:00'
    assert account_reset_evidence(reset_at=value)==({'reset_at':value},value)


@pytest.mark.parametrize('offset_minutes',[-1440,-720,-330,0,330,840,1440])
@pytest.mark.parametrize('seconds_into_day',[0,86399])
def test_unknown_bound_precedes_every_supported_offset_and_time(offset_minutes,seconds_into_day):
    _,bound=account_reset_evidence(reset_observation=dict(precision='date',date='2026-12-01',timezone='UNKNOWN'))
    possible_reset=datetime(2026,12,1,tzinfo=timezone.utc)+timedelta(seconds=seconds_into_day,minutes=-offset_minutes)
    assert datetime.fromisoformat(bound)<=possible_reset


@pytest.mark.parametrize('reported',[
    {},dict(precision='instant',date='2026-12-01',timezone='UNKNOWN'),
    dict(precision='date',date=None,timezone='UNKNOWN'),
    dict(precision='date',date='2026-12-1',timezone='UNKNOWN'),
    dict(precision='date',date='2026-02-30',timezone='UNKNOWN'),
    dict(precision='date',date='2026-12-01T00:00:00Z',timezone='UNKNOWN'),
    dict(precision='date',date='2026-12-01',timezone='EST'),
    dict(precision='date',date='2026-12-01',timezone=[]),
    dict(precision='date',date='0001-01-01',timezone='UNKNOWN'),
    dict(precision='date',date='2026-12-01',timezone='UNKNOWN',reset_at='invented'),
])
def test_invalid_or_ambiguous_date_only_evidence_rejects(reported):
    with pytest.raises(ValueError,match='ACCOUNT_RESET_INVALID'):
        account_reset_evidence(reset_observation=reported)


@pytest.mark.parametrize('kwargs',[{},dict(reset_at='2026-12-01T00:00:00Z',
    reset_observation=dict(precision='date',date='2026-12-01',timezone='UTC'))])
def test_missing_or_conflicting_precision_rejects(kwargs):
    with pytest.raises(ValueError,match='ACCOUNT_RESET_INVALID'):account_reset_evidence(**kwargs)


def prepared(tmp_path,anchor=datetime(2026,11,1,16,tzinfo=timezone.utc)):
    # Fixed UTC clocks; no host-date dependency in boundary/chronology checks.
    clock=SyntheticClock(anchor)
    provider=SyntheticProvider(anchor=anchor,clock=clock)
    service,_=make_workflow(tmp_path/'SYNTHETIC-account.sqlite3',provider=provider,clock=lambda:clock().isoformat())
    operation=service.source_operation()
    service.capture_source(operation['id'],private_retention=True,private_research=True,
                           permission_end=(anchor+timedelta(days=20)).isoformat())
    ids=[row['event']['canonical_event_id'] for row in service.games()['games']]
    operations=service.prepare(ids)
    return service,provider,[r['operation']['id'] for r in operations]


def test_date_only_capture_retains_actual_report_and_duplicate_stays_spent(tmp_path,monkeypatch):
    service,provider,ids=prepared(tmp_path)
    day=(datetime.fromisoformat(service.clock())+timedelta(days=10)).date().isoformat()
    reported=dict(precision='date',date=day,timezone='UNKNOWN')
    def forbidden(*a,**k):pytest.fail('Account qualification must not predict')
    monkeypatch.setattr('app_core.nfl_scenario_research.compute',forbidden)
    service.capture_quotes(ids,remaining_credits=2,reset_observation=reported,private_use=True,no_overage=True)
    assert sum(kind=='market' for kind,_ in provider.calls)==2
    for identity in ids:
        # The durable operation claim retains the account evidence before HTTP.
        claim=service.store.claim_record('operation:'+identity)['payload']
        review=claim['approval']
        assert review['reset_observation']==reported and review['reset_time']=='UNKNOWN'
        assert 'reset_at' not in review and review['remaining_included_credit_floor']==2
    with pytest.raises(ValueError,match='ALREADY_SPENT'):
        service.capture_quotes(ids,remaining_credits=2,reset_observation=reported,private_use=True,no_overage=True)
    assert len(provider.calls)==3 and service.saved()==[]


@pytest.mark.parametrize('zone,day',[('UNKNOWN','2026-12-01'),('UTC','2026-11-30')])
@pytest.mark.parametrize('adjust_seconds,accepted',[(-1,True),(0,False),(1,False)])
def test_date_only_bound_strictly_covers_complete_window(tmp_path,zone,day,adjust_seconds,accepted):
    # The genuine 20-minute synthetic operation end straddles the conservative
    # bound. No signed operation or clock-comparison function is patched.
    anchor=datetime(2026,11,29,23,40,tzinfo=timezone.utc)+timedelta(seconds=adjust_seconds)
    service,provider,ids=prepared(tmp_path,anchor)
    reported=dict(precision='date',date=day,timezone=zone)
    if accepted:
        service.capture_quotes(ids[:1],remaining_credits=1,reset_observation=reported,private_use=True,no_overage=True)
    else:
        with pytest.raises(ValueError,match='ACCOUNT_EXPIRED'):
            service.capture_quotes(ids[:1],remaining_credits=1,reset_observation=reported,private_use=True,no_overage=True)
        assert not service.store.spent('operation:'+ids[0])
    assert sum(kind=='market' for kind,_ in provider.calls)==int(accepted)


@pytest.mark.parametrize('age,private,no_overage,credits,code',[
    (-1,True,True,2,'ACCOUNT_EXPIRED'),(0,True,True,2,'ACCOUNT_EXPIRED'),
    (10,False,True,2,'ACCOUNT_REVIEW_REQUIRED'),(10,True,False,2,'ACCOUNT_REVIEW_REQUIRED'),
    (10,True,True,1,'ACCOUNT_REVIEW_REQUIRED'),
])
def test_date_only_overlap_and_missing_account_findings_stop_before_claim(tmp_path,age,private,no_overage,credits,code):
    service,provider,ids=prepared(tmp_path)
    reported=dict(precision='date',date=(datetime.fromisoformat(service.clock())+timedelta(days=age)).date().isoformat(),timezone='UNKNOWN')
    with pytest.raises(ValueError,match=code):
        service.capture_quotes(ids,remaining_credits=credits,reset_observation=reported,private_use=private,no_overage=no_overage)
    assert len(provider.calls)==1 and not any(service.store.spent('operation:'+key) for key in ids)
