from copy import deepcopy
from datetime import datetime, timezone
import pytest
from app_core.public_parlays import build_parlays
from app_core.public_board import validate_package

NOW = datetime(2026,9,9,16,tzinfo=timezone.utc)
def row(i, **extra):
    return dict(sport='MLB',game=f'A{i} at B{i}',pick=f'A{i} +1.5',player='',market='spread_away',odds=-110,win_estimate=.65,ev=.1,status='APPROVED',start='2026-09-09T20:00:00+00:00',as_of='2026-09-09T15:59:00+00:00',**extra)

def test_three_pairs_never_repeat_teams_even_across_games():
    rows=[row(i) for i in range(8)]
    rows[1]['game']='A0 at C'
    result=build_parlays(rows,NOW)
    assert len(result)==3
    teams=[t for p in result for leg in p['legs'] for t in leg['game'].split(' at ')]
    assert len(teams)==len(set(teams))
    assert all(p['status']=='RESEARCH ONLY' for p in result)
    assert result[0]['win_estimate']==pytest.approx(.65**2)

def test_started_stale_unknown_and_invalid_are_excluded():
    rows=[row(i) for i in range(6)]
    rows[0]['start']=NOW.isoformat()
    rows[1]['as_of']='2026-09-09T15:00:00+00:00'
    rows[2]['start']=None
    rows[3]['odds']=None
    assert len(build_parlays(rows,NOW))==1
    assert build_parlays(rows[:4],NOW)==[]

def test_approved_prioritized_but_pass_never_promoted():
    rows=[row(i) for i in range(4)]
    rows[0].update(status='PASS',win_estimate=.99)
    result=build_parlays(rows,NOW)
    assert result[0]['approved_legs']
    assert not result[1]['approved_legs']
    assert all(p['status']=='RESEARCH ONLY' for p in result)

def test_saved_pairs_validated_and_legacy_supported():
    rows=[row(i) for i in range(6)]
    package=dict(schema_version=2,built_at=NOW.isoformat(),stale_after_minutes=15,games={k:deepcopy(rows) for k in ('overall','sides','totals')},props=[],dfs=[],parlays=build_parlays(rows,NOW))
    validate_package(package)
    package['parlays'][0]['status']='APPROVED'
    with pytest.raises(ValueError):validate_package(package)
    package.pop('parlays');package['schema_version']=1
    validate_package(package)
