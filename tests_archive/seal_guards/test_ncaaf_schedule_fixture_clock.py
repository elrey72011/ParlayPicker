"""SYNTHETIC clock boundaries; original pipeline assertions remain unchanged."""
from datetime import datetime,timedelta
import hashlib
from pathlib import Path

import pytest
from app_core import ncaaf_schedule as schedule,ncaaf_pipeline_evidence as adapter
from scripts.benchmark_drive_history_loading import blocked_network
from scripts import ncaaf_filtered_history_scope as history
from tests.nfl_profile_generation_inputs import record_bytes
from test_ncaaf_normal_pipeline import AT,actual,fixture_packet
from test_ncaaf_schedule_coverage import event

ROOT=Path(__file__).resolve().parents[1]
PREDECESSORS={
    'tests/test_ncaaf_normal_pipeline.py':(19025,'a5cd6db15867bb8aebbf93148931ab492bf0865feeabbdd3a295df70ae246ca0'),
    'scripts/nfl_tls_trust_scope.py':(8547,'ae400437a7a4729bc9cef6a89162839f60fb481ca66926a3ef43ab35ecc40641'),
    'tests/nfl_profile_generation_inputs.py':(5198,'aac94ff3b7b012e39737d8e70872b2f86a52b6225acf8ff951658b126e7df841')}


@pytest.fixture(autouse=True)
def offline():
    with blocked_network():
        yield


@pytest.mark.parametrize('offset',[-1,0,1],ids=['before','equal','after'])
def test_synthetic_success_is_independent_of_live_schedule_clock(tmp_path,monkeypatch,offset):
    wall=AT+timedelta(seconds=offset)
    class Clock(datetime):
        @classmethod
        def now(cls,tz=None):
            return wall.astimezone(tz) if tz else wall.replace(tzinfo=None)
    monkeypatch.setattr(schedule,'datetime',Clock)
    original=schedule.inventory_from_events
    observed=[]
    def inventory(*a,**kw):
        value=original(*a,**kw);observed.append(value['observed_at']);return value
    monkeypatch.setattr(schedule,'inventory_from_events',inventory)
    packet,_=fixture_packet(tmp_path,monkeypatch)
    frame,_=actual(monkeypatch,packet)
    row=frame.loc[frame.market_type.eq('spread_home')].iloc[0].to_dict()
    assert observed and set(observed)=={AT.isoformat()}
    assert row['ml_inference_status']=='success'
    assert adapter.diagnose(row)==dict(status='COMPLETE',reason='AVAILABLE')


@pytest.mark.parametrize('offset',[-1,0,1],ids=['before','equal','after'])
def test_original_production_receipt_chronology_boundary(tmp_path,monkeypatch,offset):
    packet,_=fixture_packet(tmp_path,monkeypatch)
    quote=packet['payload']['original_quote']
    observed=(AT+timedelta(seconds=offset)).isoformat()
    inventory=schedule.inventory_from_events([('FBS',[event('9',quote['event_start_utc'],
        quote['event_home_team'],quote['event_away_team'])])],
        '2026-10-30','2026-10-30',complete=True,observed_at=observed)
    source={'matchup_id':'espn:college-football:9'}
    if offset>0:
        with pytest.raises(ValueError,match='^NCAAF_SCHEDULE_RECEIPT_CONFLICT$'):
            adapter.board_schedule(source,packet,inventory,AT.isoformat())
    else:
        receipt=adapter.board_schedule(source,packet,inventory,AT.isoformat())
        assert receipt['payload']['observed_at']==observed


def clock_metadata():
    binding=history.authenticate(ROOT)
    return binding['qualification_release_repair']['build_preflight']['receipt_successor']['first_driver_successor']['capacity_context_successor']['tls_trust_successor']['clock_fixture_successor']


@pytest.mark.parametrize('path',list(PREDECESSORS))
def test_pre_clock_bytes_match_independently_verified_failed_head(path):
    value=clock_metadata()
    assert value['base']=='e4a490dd2830d1a293412f6a1a99cf529eb30084'
    assert set(value['historical_sources'])==set(PREDECESSORS)
    raw=record_bytes(value['historical_sources'][path])
    size,digest=PREDECESSORS[path]
    assert len(raw)==size and hashlib.sha256(raw).hexdigest()==digest


def test_exact_fixture_argument_is_the_only_original_file_change():
    raw=record_bytes(clock_metadata()['historical_sources']['tests/test_ncaaf_normal_pipeline.py'])
    before=b'"2026-10-30","2026-10-30",complete=True)'
    assert raw.count(before)==1
    assert (ROOT/'tests/test_ncaaf_normal_pipeline.py').read_bytes().replace(b'\r\n',b'\n')==raw.replace(
        before,b'"2026-10-30","2026-10-30",complete=True,observed_at=AT.isoformat())')


def test_tls_implementation_and_production_schedule_are_preserved():
    value=clock_metadata()
    expected={
        'app_core/nfl_tls_trust.py':'9161ab15d0a0f990e5a34954d55c9a4bdfc5aa3ac5eb594ff14994ebfb7f88f2',
        'app_core/nfl_build_preflight.py':'6c0c97f20cbb4dddcedb4689088ca87e80d39952fcb31c6085560cfcefc23c0f',
        'app_core/ncaaf_schedule.py':'37297b7844df42e643a38a8aaa4ffbef592531b3aa37791791c37aa4dbeb13fd',
        'app_core/ncaaf_pipeline_evidence.py':'24feb58730709a3cc207aab33038f42aa52a58abf590fe79bfbade9108e66709'}
    for path,digest in expected.items():
        assert hashlib.sha256((ROOT/path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==digest
    assert value['fixture_change']=='explicit synthetic schedule observed_at; original assertions unchanged'
