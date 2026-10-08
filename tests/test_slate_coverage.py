"""Labelled synthetic independent-slate regressions; no real inference or calls."""
from datetime import datetime, timezone
from copy import deepcopy
import json
import pandas as pd
import pytest

from app_core.game_coverage import publication_games
from test_provider_caller_health import caller
from test_prediction_evidence import frozen

RUN = '20261007T160000.000000Z'
AT = '2026-10-07T16:00:00+00:00'
DAY = '2026-10-07'


def inventory(events=None, status='COMPLETE'):
    return dict(version='slate-inventory-v1', league='NFL', source='synthetic_schedule',
        evidence_label='SYNTHETIC', selected_date=DAY, timezone='America/New_York',
        observed_at=AT, status=status, completeness_basis='synthetic complete index',
        reasons=[] if status == 'COMPLETE' else ['SYNTHETIC_INVENTORY_LIMIT'],
        events=events if events is not None else [dict(canonical_event_id='nfl:one',
            home_team='Home', away_team='Away', home_team_id='h', away_team_id='a',
            original_start='2026-10-07T23:00:00+00:00', schedule_status='SCHEDULED')])


def report(inv=None, **kwargs):
    from app_core.slate_coverage import build_coverage
    return build_coverage([inv or inventory()], selected_date=DAY, as_of=AT,
                          run_id=RUN, **kwargs)


def test_schedule_without_candidates_reaches_publication_path():
    coverage = report()
    games, missing = publication_games(pd.DataFrame(), pd.DataFrame(), slate_report=coverage)
    assert len(games) == len(missing) == 1
    assert games.coverage_decision_state.tolist() == ['UNVERIFIED']
    assert games.Play_Stake.tolist() == [0]
    assert 'odds_american' not in games or games.odds_american.isna().all()


def test_no_inventory_does_not_claim_empty_reconciliation():
    from app_core.slate_coverage import build_coverage
    coverage = build_coverage([], selected_date=DAY, as_of=AT, run_id=RUN)
    assert coverage['inventory_status'] == 'UNAVAILABLE'
    assert coverage['fully_reconciled'] is False


def test_orphan_never_enters_scheduled_denominator():
    coverage = report(provider_events=[dict(league='NFL', id='alien', home_team='Other',
        away_team='Away', commence_time='2026-10-07T23:00:00Z', export_run_id=RUN)])
    assert coverage['counts']['scheduled_events'] == 1
    assert len(coverage['orphan_events']) == 1


def offered(**changes):
    return dict(dict(league='NFL', canonical_event_id='nfl:one', matchup_id='one',
        home_team='Home', away_team='Away', game_start_utc='2026-10-07T23:00:00Z',
        export_run_id=RUN, candidate_id='SYNTHETIC-one-home', market_type='spread_home',
        best_pick='Home -2.5', spread_line=-2.5, odds_american=-110,
        quote_bookmaker='DraftKings', odds_recorded_at=AT,
        ml_inference_status='success'), **changes)


def health(outcome='SUCCESS', processing='SUCCESS'):
    return {'sports': {'americanfootball_nfl': {'outcome': outcome, 'processing': processing}}}


def rejection(c, code='negative_conservative_ev'):
    # Labelled synthetic *observation* receipt; actual finalizer tested below.
    return dict(candidate_id=c['candidate_id'], sport=c['league'], game_id=c['matchup_id'],
        market_type=c['market_type'], selection=c['best_pick'], odds=c['odds_american'],
        line=c.get('spread_line'), start=c['game_start_utc'], sportsbook='DraftKings', coverage_run_id=RUN,
        quote_timestamp=c.get('odds_recorded_at'), coverage_evaluated_at=AT,
        coverage_gate_trace=[dict(gate='candidate_contract', status='FAIL', code=code)])


@pytest.mark.parametrize('value,code', [(None,'QUOTE_TIMESTAMP_MISSING'),
    ('wrong','QUOTE_TIMESTAMP_INVALID'), ('2026-10-07T15:59:00','QUOTE_TIMESTAMP_INVALID'),
    ('2026-10-07T16:01:00Z','QUOTE_TIMESTAMP_FUTURE'),
    ('2026-10-07T15:00:00Z','QUOTE_TIMESTAMP_STALE')])
def test_original_quote_failures_remain_distinct(value, code):
    c = offered(odds_recorded_at=value)
    d = report(candidates=[c], gate_audit=[rejection(c)], provider_health=health())['decisions'][0]
    assert d['coverage_decision_state'] == 'UNVERIFIED'
    assert code in d['blocker_codes']
    assert d['market_results'][0]['candidates'][0]['gate_results'][0]['code'] == code


@pytest.mark.parametrize('outcome,code', [('TIMEOUT','PROVIDER_FAILURE'),
    ('SUCCESS_EMPTY','PROVIDER_SUCCESS_EMPTY'), ('SUCCESS','NO_MATCHING_ODDS_EVENT')])
def test_empty_and_failed_provider_requests_are_not_equivalent(outcome, code):
    d = report(provider_health=health(outcome))['decisions'][0]
    assert code in d['blocker_codes'] and d['coverage_decision_state'] == 'UNVERIFIED'


def test_partial_response_does_not_drop_other_scheduled_game():
    inv = inventory(); second = deepcopy(inv['events'][0]); second.update(canonical_event_id='nfl:two', home_team='Other')
    inv['events'].append(second)
    r = report(inv, candidates=[offered()], provider_health=health())
    assert len(r['decisions']) == 2
    assert 'NO_MATCHING_ODDS_EVENT' in r['decisions'][1]['blocker_codes']


def test_started_and_missing_start_are_visible():
    for value, code in [(AT,'GAME_STARTED'), (None,'START_UNAVAILABLE')]:
        inv = inventory(); inv['events'][0]['original_start'] = value
        d = report(inv)['decisions'][0]
        assert code in d['blocker_codes']
        assert d['market_results'][0]['gate_results'][0]['status'] == 'PASS'
        assert d['first_observed_failure']['gate'] == 'pregame'


def test_repeated_matchup_and_ambiguous_same_clock_keep_two_identities():
    inv = inventory(); second = deepcopy(inv['events'][0]); second['canonical_event_id'] = 'nfl:two'
    inv['events'].append(second)
    unbound = offered(); unbound.pop('canonical_event_id')
    r = report(inv, candidates=[unbound])
    assert len(r['decisions']) == 2 and r['orphan_events'][0]['status'] == 'IDENTITY_AMBIGUOUS'
    assert all('IDENTITY_AMBIGUOUS' in d['blocker_codes'] for d in r['decisions'])
    bound = offered(canonical_event_id='nfl:two')
    r = report(inv, candidates=[bound])
    assert len(r['decisions'][0]['market_results'][0]['candidates']) == 0
    assert len(r['decisions'][1]['market_results'][0]['candidates']) == 1
    inv['events'][1]['original_start'] = '2026-10-08T01:00:00Z'
    r = report(inv, candidates=[unbound])
    assert len(r['decisions'][0]['market_results'][0]['candidates']) == 1
    assert not r['decisions'][1]['market_results'][0]['candidates']


def test_resolved_rejection_and_mixed_scope():
    c = offered(); receipt = rejection(c)
    r = report(candidates=[c], gate_audit=[receipt], provider_health=health(), required_markets=['spread_home'])
    assert r['decisions'][0]['coverage_decision_state'] == 'PASS'
    assert r['decisions'][0]['market_results'][0]['candidates'][0]['decision_state'] == 'PASS'
    mixed = report(candidates=[c], gate_audit=[receipt], provider_health=health())['decisions'][0]
    assert mixed['coverage_decision_state'] == 'UNVERIFIED'
    assert mixed['market_results'][0]['state'] == 'PASS'
    assert mixed['market_results'][1]['state'] == 'UNVERIFIED'
    for field in ('quote_timestamp', 'line', 'game_id', 'odds', 'selection'):
        altered = dict(receipt, **{field:'alien'})
        assert report(candidates=[c], gate_audit=[altered], provider_health=health(), required_markets=['spread_home'])['decisions'][0]['coverage_decision_state'] == 'UNVERIFIED'


def test_declared_policy_exclusion_has_no_invented_gate_results():
    exclusion = dict(canonical_event_id='nfl:one', policy_id='SYNTHETIC-policy', reason_code='FCS_OUTSIDE_STAGE1',
        scope=['spread_home', 'spread_away', 'total_over', 'total_under'])
    d = report(policy_exclusions=[exclusion])['decisions'][0]
    assert d['coverage_decision_state'] == 'PASS' and d['blocker_codes'][0] == 'FCS_OUTSIDE_STAGE1'
    assert all(g['status'] == 'NOT_EVALUATED' for m in d['market_results'] for g in m['gate_results'][:-1])
    assert 'FBS_ONLY' in d['stage1_cohort']


@pytest.mark.parametrize('scope,resolved', [('stage1', []), (['spread_home'], ['spread_home'])])
def test_policy_exclusion_scope_cannot_resolve_other_required_markets(scope, resolved):
    exclusion = dict(canonical_event_id='nfl:one', policy_id='SYNTHETIC-policy',
        reason_code='FCS_OUTSIDE_STAGE1', scope=scope)
    d = report(policy_exclusions=[exclusion])['decisions'][0]
    assert d['coverage_decision_state'] == 'UNVERIFIED'
    assert [m['market'] for m in d['market_results'] if m['state'] == 'PASS'] == resolved
    assert d['policy_exclusion'] == exclusion


@pytest.mark.parametrize('status,complete', [('COMPLETE',True), ('PARTIAL',False), ('UNAVAILABLE',False)])
def test_empty_inventory_requires_external_completeness_basis(status, complete):
    r = report(inventory(events=[], status=status))
    assert r['fully_reconciled'] is complete and r['counts']['scheduled_events'] == 0
    assert r['reconciliation'] == dict(missing=[], duplicates=[], extras=[])


@pytest.mark.parametrize('day,start', [('2026-03-08','2026-03-09T03:59:59Z'),
    ('2026-11-01','2026-11-01T05:30:00Z'), ('2026-11-01','2026-11-01T06:30:00Z'),
    ('2026-10-07','2026-10-08T03:59:59Z')])
def test_eastern_calendar_and_dst(day,start):
    from app_core.slate_coverage import build_coverage
    inv = inventory(); inv['selected_date'] = day; inv['observed_at'] = start
    inv['events'][0]['original_start'] = start
    assert len(build_coverage([inv], selected_date=day, as_of=start, run_id=RUN)['decisions']) == 1
    inv['selected_date'] = '2026-01-01'
    with pytest.raises(ValueError, match='OUTSIDE_EASTERN_DATE'):
        build_coverage([inv], selected_date='2026-01-01', as_of=start, run_id=RUN)


def test_conflicting_run_reconciliation_and_model_blockers():
    from app_core.slate_coverage import reconcile
    with pytest.raises(ValueError, match='CONFLICTING_RUN'):
        report(candidates=[offered(export_run_id='other')])
    for code in ('NCAAF_MODEL_SCHEMA','NCAAF_FROZEN_RUNTIME_MISMATCH'):
        c=offered(ml_unavailable_reason=code,ml_inference_status='unavailable')
        d=report(candidates=[c])['decisions'][0]
        assert 'MODEL_INCOMPATIBLE' in d['blocker_codes']
        assert d['market_results'][0]['candidates'][0]['original_model_blocker'] == code
    r=report(); r['decisions'].append(deepcopy(r['decisions'][0]))
    with pytest.raises(ValueError, match='RECONCILIATION_MISMATCH'): reconcile(r)
    r=report(); r['decisions'].clear()
    with pytest.raises(ValueError, match='RECONCILIATION_MISMATCH'): reconcile(r)


def test_actual_boards_public_schema_and_no_borrowed_aliases():
    from app_core.per_game_boards import per_game_board
    from app_core.public_board import build_package, validate_package
    c = offered(calibrated_probability=.99, expected_value=100, Play_Stake=100)
    r = report(candidates=[c])
    games,_ = publication_games(pd.DataFrame(),pd.DataFrame([c]),slate_report=r)
    boards={f:per_game_board(games,pd.DataFrame([c]),f,novig_only=True) for f in ('overall','sides','totals')}
    package=build_package(**boards); validate_package(package)
    for family in package['games'].values():
        row=family[0]
        assert row['record_kind']=='coverage_only_no_inference'
        assert row['win_estimate'] is None and row['ev'] is None and row['odds'] is None
        assert row['as_of'] is None and row['status']=='PASS'
        assert row['coverage_decision']['coverage_decision_state']=='UNVERIFIED'
    tampered=deepcopy(package); tampered['games']['overall'][0]['coverage_decision']['raw_dependencies']={'secret':'private'}
    with pytest.raises(ValueError):validate_package(tampered)
    tampered=deepcopy(package); tampered['slate_coverage']['decisions'][0]['coverage_decision_state']='APPROVED'
    with pytest.raises(ValueError):validate_package(tampered)
    r=report(inventory(events=[]))
    games,_=publication_games(pd.DataFrame(),pd.DataFrame(),slate_report=r)
    empty=build_package(**{f:per_game_board(games,family=f) for f in boards});validate_package(empty)
    assert empty['slate_coverage']['inventory_status']=='COMPLETE'


def test_actual_finalizer_observations_do_not_create_approval(tmp_path, frozen, monkeypatch):
    from activation_fixture import setup
    from test_wager_integrity_audit import NOW
    from core.live_wager_contract import finalize_live_wagers
    base,p,config=setup(tmp_path/'ledger.db')
    base.update(candidate_id='SYNTHETIC-finalizer',best_pick='Home -2.5',
        league='NFL',home_team='Home',away_team='Away',game_start_utc=base['start'],
        export_run_id='20260914T150000.000000Z',odds_recorded_at=base['quote_time'],
        spread_line=-2.5,quote_bookmaker='DraftKings',ml_inference_status='success')
    out,audit=finalize_live_wagers(pd.DataFrame([base]),pd.DataFrame([base]),1000,now=NOW,
        policies={'NFL':p},config=config)
    assert audit[0]['coverage_gate_trace'] and audit[0]['coverage_evaluated_at']==NOW.isoformat()
    inv=inventory(); inv['selected_date']='2026-09-14';inv['observed_at']=NOW.isoformat()
    inv['events'][0]['original_start']=base['start']
    from app_core.slate_coverage import build_coverage
    r=build_coverage([inv],selected_date='2026-09-14',as_of=NOW.isoformat(),run_id=base['export_run_id'],
        candidates=[base],final=out,gate_audit=audit,provider_health=health())
    assert out.iloc[0]['Bettable'] and out.iloc[0]['Play_Stake']>0
    assert r['decisions'][0]['coverage_decision_state']=='APPROVED'
    saved_only=build_coverage([inv],selected_date='2026-09-14',as_of=NOW.isoformat(),run_id=base['export_run_id'],
        candidates=[base],final=out,gate_audit=[],provider_health=health())
    assert saved_only['decisions'][0]['coverage_decision_state']=='UNVERIFIED'
    # Protected actual capture assigns a separate export clock. Retain both.
    from app_core import prediction_evidence as evidence
    context,db,_=frozen
    base['finalization_run_id']=base['export_run_id']
    base['best_available_selected']=True
    # Use the original finalized packet's candidate ID without recomputation.
    monkeypatch.setattr(evidence,'now_utc',lambda:'2026-09-14T15:00:01Z')
    saved,card=evidence.capture_run(context,pd.DataFrame([base]),out,pd.DataFrame([base]),path=db)
    assert saved.iloc[0]['finalization_run_id']==base['export_run_id']
    assert saved.iloc[0]['export_run_id']!=base['export_run_id']
    captured=build_coverage([inv],selected_date='2026-09-14',as_of='2026-09-14T15:00:02Z',run_id=saved.iloc[0]['export_run_id'],
        candidates=saved,final=card,gate_audit=audit,provider_health=health())
    assert captured['decisions'][0]['coverage_decision_state']=='APPROVED'
    legacy=dict(base,Bettable=True,Play_Stake=100,wager_approved=True)
    r=build_coverage([inv],selected_date='2026-09-14',as_of=NOW.isoformat(),run_id=base['export_run_id'],
        candidates=[base],final=[legacy],gate_audit=[],provider_health=health())
    assert r['decisions'][0]['coverage_decision_state']=='UNVERIFIED'


def test_actual_browser_retains_schedule_only_games(tmp_path):
    from app_core.per_game_boards import per_game_board
    from app_core.public_board import build_package,validate_package
    from test_research_probability_browser import inspect_browser
    r=report(); games,_=publication_games(pd.DataFrame(),pd.DataFrame(),slate_report=r)
    package=build_package(**{f:per_game_board(games,family=f) for f in ('overall','sides','totals')})
    validate_package(package)
    result=inspect_browser(package,tmp_path/'coverage',datetime.fromisoformat(AT))
    assert result['initial']['selected']==1 and result['initial']['current']==0
    assert all(result[f]==1 for f in ('overall','sides','totals'))
    assert 'UNVERIFIED' in result['initial']['cards'][0]
    assert 'No matching odds event' in result['initial']['cards'][0]


def test_cli_and_readiness_use_inventory_even_when_candidate_audit_empty(tmp_path):
    from scripts.reconcile_slate import main
    from core.run_readiness import build_readiness,game_table
    inv=tmp_path/'inv.json'; inv.write_text(json.dumps(inventory()),encoding='utf-8')
    target=tmp_path/'report.json'
    assert main(['--inventory',str(inv),'--date',DAY,'--as-of',AT,'--run-id',RUN,'--output',str(target)])==0
    r=json.loads(target.read_text())
    result=build_readiness(pd.DataFrame(),pd.DataFrame(),diagnostics={'slate_coverage':r})
    assert game_table(result).shape[0]==1
    assert result['slate_coverage']['counts']['scheduled_events']==1


def test_existing_identity_caller_retains_bounded_inventory_without_new_requests(monkeypatch):
    from app_core import football_identity_capture as capture
    from app_core.slate_coverage import native_football
    from test_football_identity_capture import fixture
    from types import SimpleNamespace
    import requests
    game,event=fixture(); calls=[]
    def fake(url,**kwargs):
        calls.append(url)
        return SimpleNamespace(raise_for_status=lambda:None,json=lambda:{'events':[event]})
    monkeypatch.setattr(capture.requests,'get',fake)
    monkeypatch.setattr(capture,'_scoreboard_urls',lambda sport,day:[day])
    games=[dict(game,commence_time=f'2026-09-{21+i:02}T00:20:00Z') for i in range(5)]
    result=capture.collect(games,'NFL')
    assert len(calls)==3 and len(result)==5 and games[0]==dict(game)
    observation=result.retained_schedule_observation
    assert observation['status']=='PARTIAL' and 'DATE_LIMIT_REACHED' in observation['reasons']
    inv=native_football(observation,'2026-09-20')
    assert inv['status']=='PARTIAL' and len(inv['events'])==1
    # A canonical event with unavailable team facts stays an explicit blocker.
    observation['events']=[{'id':'bad','date':game['commence_time']}]
    assert len(native_football(observation,'2026-09-20')['events'])==1
    def outage(*a,**k):raise requests.Timeout('SYNTHETIC')
    monkeypatch.setattr(capture.requests,'get',outage)
    result=capture.collect(games,'NFL')
    assert 'SCHEDULE_REQUEST_FAILED' in result.retained_schedule_observation['reasons']
    assert result.retained_schedule_observation['events']==[]


def coverage_ui_app():
    import pandas as pd
    import streamlit as st
    from app.ui.daily_dashboard import render_daily_dashboard
    from app_core.game_coverage import publication_games
    from test_slate_coverage import report
    frame,_=publication_games(pd.DataFrame(),pd.DataFrame(),slate_report=report())
    today,details=st.tabs(['Today','Pick Details'])
    render_daily_dashboard(today.empty(),details.empty(),frame)


def test_today_details_and_downloads_keep_same_schedule_decision():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_function(coverage_ui_app).run()
    assert not app.exception
    assert sum(b.proto.label=='Download slate coverage JSON' for b in app.get('download_button'))==2
    assert sum(b.proto.label=='Download slate coverage CSV' for b in app.get('download_button'))==2
    assert any('UNVERIFIED' in str(d.value) for d in app.dataframe)


def test_actual_odds_caller_keeps_provider_receipts_before_candidate_loss(caller, monkeypatch):
    sp,scenario,calls,_=caller
    from app_core import football_identity_capture as capture
    from test_provider_caller_health import game
    row=game('americanfootball_nfl','one');row.update(home_team='Home',away_team='Away',commence_time='2026-10-07T23:00:00Z')
    row['bookmakers'][0]['markets'][0]['outcomes'][0]['name']='Home'
    row['bookmakers'][0]['markets'][0]['outcomes'][1]['name']='Away'
    scenario['americanfootball_nfl']=[row]
    observation=dict(sport='NFL',observed_at=AT,status='PARTIAL',events=[],reasons=['SYNTHETIC_LIMIT'],requested_utc_dates=['20261007'])
    monkeypatch.setattr(capture,'collect',lambda games,sport:capture.ObservedGames(games,observation))
    frame=sp.fetch_live_odds_dataframe(['NFL'],date=DAY)
    assert calls==[('americanfootball_nfl_preseason',DAY),('americanfootball_nfl',DAY)]
    assert frame.attrs['retained_football_schedules']==[observation]
    retained=frame.attrs['coverage_provider_events']
    assert len(retained)==1 and retained[0]['id']=='one'
    r=report(provider_events=retained,provider_health=frame.attrs['provider_health'])
    assert len(r['decisions'])==1 and 'CANDIDATE_GENERATION_LOSS' in r['decisions'][0]['blocker_codes']
