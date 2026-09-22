import json
from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest
import requests

from core.nfl_teams import NFL_TEAMS, nfl_stats_identity
from core.team_mapper import normalize_team_name
from app_core import feature_processing as fp
from app_core.nfl_novig import recover_nfl_novig
from app_core.prediction_evidence import provider_quotes
from app_core.per_game_boards import public_quote


@pytest.mark.parametrize('code,full', NFL_TEAMS.items())
def test_nfl_stats_names_and_codes_match(code, full):
    assert fp.normalize_team_for_stats(code, 'NFL') == full.upper()
    assert fp.normalize_team_for_stats(full, 'NFL') == full.upper()
    city = full.rsplit(' ', 1)[0]
    if city not in {'New York', 'Los Angeles'}:
        assert fp.normalize_team_for_stats(city, 'NFL') == full.upper()


@pytest.mark.parametrize('name', ['New York Jets','New York Giants','Los Angeles Rams','Los Angeles Chargers'])
def test_nfl_franchises_survive_repeated_pipeline_normalization(name):
    assert normalize_team_name(normalize_team_name(name)) == name


def test_nfl_ambiguous_city_never_fuzzy_matches():
    index = {x.upper().lower(): x for x in NFL_TEAMS.values()}
    for city in ['New York', 'Los Angeles', 'LA']:
        assert fp.resolve_stats_team_match(city, 'NFL', index, {}, {})[0] is None
    assert nfl_stats_identity('LA', schedule_code=True) == 'LOS ANGELES RAMS'


def test_nfl_actual_schedule_codes_enrich_games(monkeypatch):
    schedule = pd.DataFrame([dict(home_team='IND',away_team='BAL',home_score=20,away_score=24,result=-4),
                             dict(home_team='NYJ',away_team='NYG',home_score=31,away_score=17,result=14),
                             dict(home_team='LA',away_team='LAC',home_score=27,away_score=28,result=-1)])
    monkeypatch.setattr(fp, 'nfl', SimpleNamespace(import_schedules=lambda years: schedule))
    stats = pd.DataFrame(fp.fetch_nfl_stats.__wrapped__(2026))
    monkeypatch.setattr(fp, 'fetch_team_stats', lambda *a, **kw: stats)
    games = pd.DataFrame([dict(Home='Indianapolis',Away='Baltimore',sport_title='NFL'),
                          dict(Home='New York Jets',Away='New York Giants',sport_title='NFL'),
                          dict(Home='Los Angeles Rams',Away='Los Angeles Chargers',sport_title='NFL')])
    enriched = fp.enrich_with_model_features(games, {}, season_year=2026)
    assert enriched.stats_resolution_status.eq('resolved').all()
    assert not enriched.feature_stats_fallback.any()
    assert enriched.feature_home_ppg.tolist() == [20,31,27]
    assert enriched.feature_home_games_played.tolist() == [1,1,1]
    assert enriched.feature_home_last_game_summary.tolist() == [
        'L 20-24 vs BAL', 'W 31-17 vs NYG', 'L 27-28 vs LAC'
    ]


def test_nfl_stats_are_point_in_time_and_retain_prior_result(monkeypatch):
    schedule = pd.DataFrame([
        dict(gameday='2026-09-13', home_team='NYG', away_team='DAL', home_score=28, away_score=20, result=8),
        dict(gameday='2026-09-13', home_team='PHI', away_team='LA', home_score=27, away_score=7, result=20),
        # This later result must not leak into a September 21 slate.
        dict(gameday='2026-09-27', home_team='NYG', away_team='LA', home_score=10, away_score=31, result=-21),
    ])
    monkeypatch.setattr(fp, 'nfl', SimpleNamespace(import_schedules=lambda years: schedule))

    stats = pd.DataFrame(fp.fetch_nfl_stats.__wrapped__(2026, as_of_date='2026-09-21'))
    giants = stats.loc[stats.team_norm.eq('NEW YORK GIANTS')].iloc[0]
    rams = stats.loc[stats.team_norm.eq('LOS ANGELES RAMS')].iloc[0]

    assert giants.games_played == 1 and rams.games_played == 1
    assert giants.last_game_summary == 'W 28-20 vs DAL (2026-09-13)'
    assert rams.last_game_summary == 'L 7-27 at PHI (2026-09-13)'
    assert giants.recent_point_margin == 8
    assert rams.recent_point_margin == -20


def game():
    return dict(id='nfl1',home_team='Indianapolis Colts',away_team='Baltimore Ravens',
                commence_time='2026-09-13T17:00:00Z',bookmakers=[])


def novig():
    return dict(key='novig',last_update='2026-09-13T15:53:00Z',markets=[
        dict(key='totals',outcomes=[dict(name='Over',point=48.5,price=-110),dict(name='Under',point=48.5,price=-110)]),
        dict(key='spreads',outcomes=[dict(name='Indianapolis Colts',point=-2.5,price=-110),dict(name='Baltimore Ravens',point=2.5,price=-110)])])


NOW=datetime(2026,9,13,16,tzinfo=timezone.utc)


def test_nfl_bulk_recovery_binds_exact_lock_quote_without_mutating_original():
    original=[game()];response=[dict(game(),bookmakers=[novig()])];calls=[]
    def get(*a,**kw):
        calls.append(kw)
        return SimpleNamespace(raise_for_status=lambda:None,json=lambda:response)
    recovered=recover_nfl_novig(original,'test',get=get,now=NOW)
    assert original[0]['bookmakers']==[]
    assert len(calls)==1 and calls[0]['timeout']==5
    row=dict(league='NFL',market_type='total_over',total_line=48.5,odds_american=-110,
             export_run_id=NOW.isoformat(),provider_quotes=provider_quotes(recovered[0]))
    assert public_quote(row)==('Novig','2026-09-13T15:53:00+00:00')
    row['total_line']=49.5
    assert public_quote(row) is None


@pytest.mark.parametrize('failure', ['timeout','wrong_team','duplicate','started','existing'])
def test_nfl_recovery_fails_closed_and_preserves_existing_quotes(failure):
    original=[game()];offered=dict(game(),bookmakers=[novig()]);calls=[]
    if failure=='wrong_team':offered['home_team']='New York Jets'
    if failure=='started':original[0]['commence_time']='2026-09-13T15:00:00Z'
    if failure=='existing':original[0]['bookmakers']=[novig()]
    def get(*a,**kw):
        calls.append(1)
        if failure=='timeout':raise requests.ReadTimeout()
        return SimpleNamespace(raise_for_status=lambda:None,json=lambda:[offered,offered] if failure=='duplicate' else [offered])
    before=deepcopy(original)
    assert recover_nfl_novig(original,'test',get=get,now=NOW)==before
    assert len(calls)==(0 if failure in {'started','existing'} else 1)


def test_wager_summary_reports_saved_blockers():
    from app.ui.daily_dashboard import wager_rejection_summary
    board=pd.DataFrame([
        dict(league='NFL',Bettable=False,feature_stats_fallback=True),
        dict(league='MLB',Bettable=False,stats_source='live',stats_resolution_status='resolved',Production_Gate_Reason='model EV is not positive'),
        dict(league='MLB',Bettable=True)])
    result=wager_rejection_summary(board)
    assert result.Games.sum()==2
    assert set(result.Reason)=={'Incomplete team statistics; research only','model EV is not positive'}


def test_default_odds_sport_list_includes_nfl(monkeypatch):
    import core.streamlit_pipeline as sp
    import app_core.odds_api as odds
    import app_core.espn_ncaaf_odds as espn
    calls=[]
    class Client:
        def __init__(self, **kwargs):pass
        def get_odds(self, sport_key, date=None):
            calls.append(sport_key)
            return []
    monkeypatch.setattr(sp,'_get_odds_api_key',lambda:'test')
    monkeypatch.setattr(odds,'TheOddsAPIClient',Client)
    monkeypatch.setattr(espn,'fetch_espn_ncaaf_fcs_odds',lambda *a:[])
    sp.fetch_live_odds_dataframe(None,date='2026-09-13')
    assert calls.count('americanfootball_nfl')==1


def test_nfl_fallback_is_opt_in_prefers_novig_and_locks_named_book():
    from test_per_game_boards import final, quoted_candidate
    from app_core.per_game_boards import per_game_board
    from app_core.public_board import build_package, validate_package
    from app_core.locked_picks import lock_candidates, locked_selections
    from app_core.public_history import eligible
    board=pd.DataFrame([final(league='NFL',export_run_id='20260911T200000.000000Z',game_time_est='2026-09-11 7:00 PM ET')])
    dk=quoted_candidate('draftkings',league='NFL',best_available_rank=1)
    nv=quoted_candidate(league='NFL',best_available_rank=2)
    assert per_game_board(board,pd.DataFrame([dk]),novig_only=True,college_fallback=True).iloc[0].quote_source=='Unavailable'
    assert per_game_board(board,pd.DataFrame([dk,nv]),novig_only=True,nfl_fallback=True).iloc[0].quote_source=='Novig'
    views=[per_game_board(board,pd.DataFrame([dk]),f,novig_only=True,nfl_fallback=True) for f in ('overall','sides','totals')]
    assert not views[0].Bettable.any() and views[0].Play_Stake.sum()==0
    package=build_package(*views);validate_package(package)
    leg=package['games']['overall'][0]
    at='2026-09-11T20:01:00+00:00'
    assert leg['quote_source']=='DraftKings' and leg['status']=='PASS'
    locks=lock_candidates(package,at)
    assert len(locks)==1 and locked_selections(locks)[0]['legs'][0]['quote_source']=='DraftKings'
    assert not eligible(dict(leg,quote_time='2026-09-11T19:00:00Z'),datetime.fromisoformat(at))
    assert not eligible(dict(leg,quote_time_basis='espn_observed'),datetime.fromisoformat(at))
    for bad in [dict(dk,total_line=9.5),dict(dk,odds_american=-120),dict(dk,market_type='total_over'),dict(dk,export_run_id='20260911T210000.000000Z')]:
        assert per_game_board(board,pd.DataFrame([bad]),novig_only=True,nfl_fallback=True).iloc[0].quote_source=='Unavailable'
