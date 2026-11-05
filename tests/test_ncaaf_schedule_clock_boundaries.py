"""SYNTHETIC NCAAF schedule clock boundaries.

Application tests copied verbatim from tests/test_ncaaf_schedule_fixture_clock.py
(archived 2026-11-04 with its historical byte-pinning seal checks).
"""
from datetime import datetime,timedelta

import pytest
from app_core import ncaaf_schedule as schedule,ncaaf_pipeline_evidence as adapter
from scripts.benchmark_drive_history_loading import blocked_network
from test_ncaaf_normal_pipeline import AT,actual,fixture_packet
from test_ncaaf_schedule_coverage import event


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
