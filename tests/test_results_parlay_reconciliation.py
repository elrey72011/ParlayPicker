from copy import deepcopy
from datetime import datetime, date, timezone
from types import SimpleNamespace
import pandas as pd
import pytest
import requests
from app_core.public_history import grade_leg, fetch_scores, report
from app_core.result_reconciliation import match_result, revision_signature
from app_core.public_parlays import build_parlays, parlay_funnel
from app_core.gemini_bet_gate import apply_gemini_bet_gate, gemini_gate_mask
from test_public_parlays import row, NOW


def mlb(away='Seattle Mariners',home='Athletics',a=7,h=8,**extra):
    return dict(sport='MLB',event_id='one',result_source='ESPN',away=away,home=home,away_score=a,home_score=h,start='2026-09-13T23:00:00Z',**extra)


def leg(game='Seattle at Athletics',pick='Athletics +1.5',market='spread_home',**extra):
    return dict(sport='MLB',game=game,pick=pick,market=market,start='2026-09-13T17:00:00Z',**extra)


@pytest.mark.parametrize('game,pick,market,score,outcome',[
 ('Seattle at Athletics','Athletics +1.5','spread_home',mlb(),'WIN'),
 ('Seattle at Athletics','Under 7.5','total_under',mlb(),'LOSS'),
 ('Kansas City at Boston','Kansas City +1.5','spread_away',mlb('Kansas City Royals','Boston Red Sox',1,4),'LOSS'),
 ('San Diego at San Francisco','Under 7.5','total_under',mlb('San Diego Padres','San Francisco Giants',6,4),'LOSS'),
])
def test_september_13_delayed_finals(game,pick,market,score,outcome):
    original=leg(game,pick,market);before=deepcopy(original)
    assert grade_leg(original,[score])[0]==outcome
    assert original==before


def test_doubleheader_disambiguation_and_unfinished_game():
    scores=[mlb(),dict(mlb(),event_id='two',start='2026-09-14T00:00:00Z',completed=False,away_score=None,home_score=None)]
    assert match_result(leg(),scores)[1]=='DOUBLEHEADER_AMBIGUOUS'
    assert grade_leg(leg(espn_event_id='one'),scores)[0]=='WIN'
    assert match_result(leg(espn_event_id='missing'),scores)[1]=='DOUBLEHEADER_AMBIGUOUS'
    assert grade_leg(dict(leg(),start='2026-09-13T23:00:00Z'),scores)[0]=='PENDING'
    assert grade_leg(leg(game_number=2),[dict(scores[0],game_number=1),dict(scores[1],game_number=2)])[0]=='PENDING'


def test_conflicting_provider_and_invalid_scores_never_grade():
    assert match_result(leg(),[mlb(),dict(mlb(),result_source='MLB',event_id='other',home_score=3)])[1]=='PROVIDER_SCORE_CONFLICT'
    assert match_result(leg(),[dict(mlb(),home_score=float('nan'))])[1]=='FINAL_SCORE_INVALID'
    assert match_result(leg(),[dict(mlb(),start='2026-09-15T00:00:00Z')])[1]=='DATE_MISMATCH'


def test_official_mlb_fallback_after_espn_failure(monkeypatch):
    def get(url,**kw):
        if 'espn' in url: raise requests.Timeout('test')
        return SimpleNamespace(raise_for_status=lambda:None,json=lambda:{'dates':[{'games':[dict(gamePk=42,gameDate='2026-09-13T23:00:00Z',gameNumber=1,status={'abstractGameState':'Final'},teams={'away':{'team':{'name':'Seattle Mariners'},'score':7},'home':{'team':{'name':'Athletics'},'score':8}})]}]})
    monkeypatch.setattr(requests,'get',get)
    revision=fetch_scores(date(2026,9,13),{'MLB'})
    assert grade_leg(leg(),revision['scores'])[0]=='WIN'
    assert revision['scores'][0]['result_source']=='MLB'
    assert revision['scores'][0]['provider_event_id']=='42'
    repeat=deepcopy(revision);repeat['recorded_at']='later'
    for item in repeat['scores']:item['provider_recorded_at']='later'
    assert revision_signature(revision)==revision_signature(repeat)


@pytest.mark.parametrize('book',['Novig','DraftKings','FanDuel','BetMGM'])
def test_same_supported_book_qualified_pair(book):
    rows=[dict(row(i),sport='NFL',quote_source=book,quote_time=NOW.isoformat(),maturity='STANDARD') for i in range(2)]
    assert len(build_parlays(rows,NOW,qualified_only=True))==1
    assert build_parlays(rows,NOW,qualified_only=True)[0]['status']=='RESEARCH ONLY'


@pytest.mark.parametrize('change,reason',[
 ({'market':'moneyline_home'},'moneyline_or_invalid_market'),
 ({'ev':-.1},'nonpositive_ev'),({'quote_time':'2026-09-09T14:00:00Z'},'stale_or_unresolved_quote'),
 ({'maturity':'PROVISIONAL'},'provisional_straight_only'),({'gemini_review_status':'HOLD'},'gemini_hard_veto'),
 ({'quote_source':'FanDuel'},'no_same_book_partner'),
])
def test_funnel_explains_rejected_pair(change,reason):
    rows=[dict(row(i),sport='NFL',quote_source='DraftKings',quote_time=NOW.isoformat(),maturity='PREMIUM') for i in range(2)]
    rows[1].update(change)
    assert not build_parlays(rows,NOW,qualified_only=True)
    assert parlay_funnel(rows,NOW)['exclusions'][reason]>=1


def outage(**extra):
    return dict(dict(production_eligible=True,expected_value=.1,gemini_pick='No Gemini pick',gemini_error='SERVICE_TIMEOUT_OR_5XX',gemini_reviewed=False,bankroll=1000,Kelly_Bet_Size=20),**extra)


@pytest.mark.parametrize('extra,allowed',[({},True),({'production_eligible':False},False),({'expected_value':-.1},False),({'gemini_flags':'stale_data'},False),({'gemini_error':''},False)])
def test_explicit_outage_never_approves_or_promotes(extra,allowed):
    frame=apply_gemini_bet_gate(pd.DataFrame([outage(**extra)]),enabled=True,product='best_pick',outage_policy={'mode':'capped','fraction':.001,'multiplier':.5})
    assert not frame.iloc[0].gemini_approved
    assert bool(gemini_gate_mask(frame).iloc[0])==allowed
    assert frame.iloc[0].Kelly_Bet_Size==(1 if allowed else 0)
    if allowed: assert frame.iloc[0].maturity=='PROVISIONAL'


def test_outage_requires_policy():
    frame=apply_gemini_bet_gate(pd.DataFrame([outage()]),enabled=True,product='best_pick',outage_policy={'mode':'hold'})
    assert not gemini_gate_mask(frame).iloc[0]


def test_nfl_denver_kansas_city_schedule_and_no_completed_games(monkeypatch):
    from app_core import feature_processing as fp
    schedule=pd.DataFrame([dict(home_team='KC',away_team='DEN',home_score=24,away_score=20,result=4)])
    monkeypatch.setattr(fp,'nfl',SimpleNamespace(import_schedules=lambda years:schedule))
    stats=fp.fetch_nfl_stats.__wrapped__(2026)
    assert {r['team_norm'] for r in stats}=={'DENVER BRONCOS','KANSAS CITY CHIEFS'}
    for r in stats: assert fp.normalize_team_for_stats(r['team_norm'],'NFL')==r['team_norm']
    schedule.loc[0,['home_score','away_score','result']]=float('nan')
    assert fp.fetch_nfl_stats.__wrapped__(2026)==[]


def test_explicit_update_is_idempotent_and_recomputes_ledger(monkeypatch):
    from app.ui import public_results
    from test_public_history import pub, scores, Memory, History
    monkeypatch.setattr('app_core.public_record.START_DATE','2026-09-09')
    store=History('site-1234','folder',Memory())
    pubs=[pub()];original=deepcopy(pubs)
    saved={'publications':pubs,'revisions':[],'imports':[],'locks':[],'rows':report(pubs,[])}
    monkeypatch.setattr(public_results,'history',lambda setting:store)
    monkeypatch.setattr(public_results,'fetch_scores',lambda *a:{'recorded_at':'2026-09-10T00:00:00Z','scores':scores()})
    public_results.update_pending_results(lambda key:'',saved)
    assert saved['reconciliation_summary']['pending_after']==0
    assert saved['reconciliation_summary']['resolved']==3
    public_results.update_pending_results(lambda key:'',saved)
    assert len(store.all('scores'))==1 and pubs==original


def test_batch_timeout_transport_is_preserved(monkeypatch):
    from app_core import llm_assistant as llm
    from integrations.gemini_client import _attach_gemini_results
    def fail(**kw):raise RuntimeError('504 DEADLINE_EXCEEDED')
    monkeypatch.setattr(llm,'_GEMINI_AVAILABLE',True)
    monkeypatch.setattr(llm,'initialize_gemini',lambda:(SimpleNamespace(models=SimpleNamespace(generate_content=fail)),None))
    monkeypatch.setattr(llm,'genai',SimpleNamespace(types=SimpleNamespace(GenerateContentConfig=lambda **kw:kw)))
    payload=llm.generate_batch_confidence_explanation([{'game_id':'one','side_a':{'best_pick':'Home +1.5'}}],_retry_incomplete=False)
    assert payload['one']['error']=='SERVICE_TIMEOUT_OR_5XX'
    frame=_attach_gemini_results(pd.DataFrame([dict(outage(),best_pick='Home +1.5')]),['one'],payload)
    gated=apply_gemini_bet_gate(frame,enabled=True,product='best_pick')
    assert gated.iloc[0].gemini_review_status=='OUTAGE_CAPPED'
    assert not gated.iloc[0].gemini_approved


def test_dallas_actual_pending_selection():
    original=dict(sport='NFL',game='Dallas at New York Giants',pick='Over 47.5',market='total_over',start='2026-09-14T00:20:00Z')
    score=dict(sport='NFL',event_id='401872930',start=original['start'],away='Dallas Cowboys',home='New York Giants',away_score=20,home_score=28)
    assert grade_leg(original,[score])[0]=='WIN'


def test_outage_cap_survives_final_portfolio_allocation():
    from test_gemini_bet_gate import _portfolio_row
    from core.streamlit_pipeline import optimize_portfolio_allocation
    base=_portfolio_row(production_eligible=True,gemini_pick='No Gemini pick',gemini_error='SERVICE_TIMEOUT_OR_5XX',gemini_reviewed=False)
    frame=apply_gemini_bet_gate(pd.DataFrame([base]),enabled=True,product='best_pick',outage_policy={'mode':'capped','fraction':.001,'multiplier':.5})
    result=optimize_portfolio_allocation(frame,bankroll=1000).iloc[0]
    assert 0 < result.production_bet_amount <= 1
    assert not result.gemini_approved
    assert result.maturity=='PROVISIONAL'
