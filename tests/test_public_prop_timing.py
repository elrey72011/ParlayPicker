from copy import deepcopy
import pandas as pd
from app_core.public_prop_timing import with_game_starts
from app_core.public_prop_history import selections
from app_core.public_board import build_package
from test_public_prop_history import publication
from test_public_board import boards


def test_archive_start_recovery_preserves_evidence_and_original_identity():
    p=publication();before=selections([p]);p['package']['props'][0]['start']=None;original=deepcopy(p)
    after=selections([p])
    assert after==before and p==original


def test_new_publication_includes_same_run_start():
    frames=boards()
    props=pd.DataFrame([{'league':'MLB','player':'Test Batter','matchup':'A @ B','best_pick':'Test Batter Over 0.5 Hits','market_type':'batter_hits_over','odds_american':-110,'export_run_id':frames[0].iloc[0]['export_run_id']}])
    package=build_package(*frames,props=props)
    assert package['props'][0]['start']==package['games']['overall'][0]['start']


def test_missing_ambiguous_different_league_or_run_never_guessed():
    p=publication();leg=p['package']['props'][0];leg['start']=None;game=p['package']['games']['overall'][0]
    for games in [[],[game,game],[{**game,'sport':'NFL'}],[{**game,'as_of':'2026-09-08T19:55:00+00:00'}],[{**game,'game':'Boston at Seattle'}],[{**game,'start':None}]]:
        assert with_game_starts([leg],games)[0]['start'] is None


def test_recovered_start_still_enforces_pregame_and_freshness():
    p=publication();p['package']['props'][0]['start']=None
    p['confirmed_at']='2026-09-09T20:01:00+00:00'
    assert selections([p])==[]
    p['confirmed_at']='2026-09-09T19:59:00+00:00'
    p['package']['props'][0]['as_of']='2026-09-09T19:00:00+00:00'
    p['package']['games']['overall'][0]['as_of']='2026-09-09T19:00:00+00:00'
    assert selections([p])==[]


def test_explicit_start_unchanged_and_scoped_aliases():
    p=publication();leg=p['package']['props'][0];game=p['package']['games']['overall'][0]
    assert with_game_starts([leg],[{**game,'start':'2026-09-09T23:00:00+00:00'}])[0]==leg
    leg={**leg,'start':None,'game':'Minnesota Twins @ Detroit Tigers'}
    game={**game,'game':'Minnesota at Detroit'}
    assert with_game_starts([leg],[game])[0]['start']==game['start']
