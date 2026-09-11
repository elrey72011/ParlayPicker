from copy import deepcopy
from datetime import date
import pytest
from app_core import public_prop_history as props
from test_public_history import pub


def publication():
    p=deepcopy(pub());p['package']['props']=[{'sport':'MLB','game':'Seattle Mariners @ Boston Red Sox','player':'Test Batter','pick':'Test Batter Over 1.5 Hits','market':'batter_hits_over','odds':-110,'win_estimate':.6,'ev':.1,'status':'APPROVED','as_of':'2026-09-09T19:55:00+00:00','start':'2026-09-09T20:00:00+00:00'}]
    return p


def test_first_publication_timing_and_line_identity():
    p=publication();later=deepcopy(p);later['confirmed_at']='2026-09-09T19:58:00+00:00';later['package']['props'][0]['pick']='Test Batter Under 2.5 Hits'
    later['package']['props'][0]['market']='batter_hits_under'
    entries=props.selections([later,p]);assert len(entries)==1 and entries[0]['leg']['pick']=='Test Batter Over 1.5 Hits'
    stale=deepcopy(p);stale['confirmed_at']='2026-09-09T20:00:00+00:00'
    assert props.selections([stale])==[]
    stale['confirmed_at']='2026-09-09T19:59:00+00:00';stale['package']['props'][0]['as_of']='2026-09-09T19:00:00+00:00'
    assert props.selections([stale])==[]


def test_report_settlement_and_import_separation():
    p=publication();leg=p['package']['props'][0];batch={'id':'original','as_of':leg['as_of'],'props':[leg]}
    entries=props.selections([p],[batch]);assert len(entries)==2
    revision={'recorded_at':'2026-09-10T01:00:00+00:00','actuals':[{'id':e['id'],'value':2} for e in entries]}
    rows=props.report([p],[revision],[batch]);assert {r['group'] for r in rows}=={'Approved','Imported research'}
    assert all(r['outcome']=='WIN' for r in rows)
    assert len(props.selections([], [batch,batch]))==1
    corrected={'recorded_at':'2026-09-10T02:00:00+00:00','actuals':[{'id':next(e['id'] for e in entries if e['group']=='Approved'),'value':1}]}
    assert props.report([p],[revision,corrected])[0]['outcome']=='LOSS'


def fixtures():
    game={'gamePk':1,'gameDate':'2026-09-09T20:00:00Z','status':{'detailedState':'Final'},'teams':{'away':{'team':{'name':'Seattle Mariners'}},'home':{'team':{'name':'Boston Red Sox'}}}}
    player={'person':{'id':123,'fullName':'Test Batter'},'stats':{'batting':{'plateAppearances':4,'hits':2}}}
    return {'dates':[{'games':[game]}]},{'teams':{'away':{'players':{'ID123':player}}}}


def fetcher(schedule,box,calls):
    def get(url,**kw):
        calls.append(url)
        class Response:
            def raise_for_status(self):pass
            def json(self):return schedule if url.endswith('/schedule') else box
        return Response()
    return get


def test_final_boxscore_is_game_and_player_specific_and_cached():
    schedule,box=fixtures();entries=props.selections([publication()]);calls=[]
    revision=props.fetch_actuals(date(2026,9,9),entries+entries,http_get=fetcher(schedule,box,calls))
    assert len(calls)==2 and revision['actuals'][0]['value']==2
    assert revision['actuals'][0]['player_id']=='123' and revision['actuals'][0]['game_id']=='1'


@pytest.mark.parametrize('failure',['live','duplicate_game','wrong_day','duplicate_player','no_stat','dnp'])
def test_uncertain_stats_remain_pending(failure):
    schedule,box=fixtures();game=schedule['dates'][0]['games'][0];player=box['teams']['away']['players']['ID123']
    if failure=='live':game['status']['detailedState']='In Progress'
    if failure=='duplicate_game':schedule['dates'][0]['games'].append({**game,'gamePk':2})
    if failure=='wrong_day':game['gameDate']='2026-09-10T23:00:00Z'
    if failure=='duplicate_player':box['teams']['away']['players']['ID456']=deepcopy(player)
    if failure=='no_stat':del player['stats']['batting']['hits']
    if failure=='dnp':player['stats']['batting']['plateAppearances']=0
    r=props.fetch_actuals(date(2026,9,9),props.selections([publication()]),http_get=fetcher(schedule,box,[]))
    assert r['actuals']==[]


def test_import_requires_original_times_and_ignores_supplied_grades():
    import pandas as pd
    leg=publication()['package']['props'][0]
    row={'league':'MLB','player':leg['player'],'best_pick':leg['pick'],'market_type':leg['market'],'matchup':leg['game'],'odds_american':-110,'export_run_id':leg['as_of'],'game_start_utc':leg['start'],'result':'WIN','actual_value':99,'Bettable':True,'Kelly_Bet_Size':10}
    batch=props.import_export(pd.DataFrame([row]));assert props.report([],[],[batch])[0]['outcome']=='PENDING'
    assert props.report([],[],[batch])[0]['group']=='Imported research'
    del row['game_start_utc']
    with pytest.raises(ValueError):props.import_export(pd.DataFrame([row]))


def test_package_v4_and_legacy_compatibility():
    from app_core.public_board import validate_package
    p=publication()['package'];p['schema_version']=4;p['parlays']=[];p['results']=props.report([publication()])
    assert validate_package(p)==p
    p['results'][0]['secret']='no'
    with pytest.raises(ValueError):validate_package(p)


def test_boxscore_limit_and_provider_failure():
    schedule,box=fixtures();calls=[]
    result=props.fetch_actuals(date(2026,9,9),props.selections([publication()]),http_get=fetcher(schedule,box,calls),max_games=0)
    assert len(calls)==1 and not result['actuals']
    def fail(*a,**kw):raise RuntimeError('provider unavailable')
    with pytest.raises(RuntimeError):props.fetch_actuals(date(2026,9,9),props.selections([publication()]),http_get=fail)

def history_app():
    import streamlit as st
    from app.ui.public_results import render_history
    st.session_state['returned_rows']=render_history(lambda key:'site-1234' if key=='PARLAYPICKER_NETLIFY_SITE_ID' else 'folder')


def test_owner_restore_and_explicit_grade_persist_to_drive(monkeypatch):
    # Exercise historical grading independently of the owner-selected public epoch.
    monkeypatch.setattr("app_core.public_record.START_DATE", "2026-09-09")
    from streamlit.testing.v1 import AppTest
    from app.ui import public_results
    from app_core.public_history import History
    from test_public_history import Memory
    store=History('site-1234','folder',Memory());p=publication();key=store.archive(p['package']);store.confirm('deployment123',key,p['confirmed_at'])
    monkeypatch.setattr(public_results,'history',lambda _:store)
    calls=[]
    def fetch(day,entries):
        calls.append(day)
        return {'recorded_at':'2026-09-10T01:00:00+00:00','actuals':[{'id':e['id'],'value':2} for e in entries]}
    monkeypatch.setattr(props,'fetch_actuals',fetch)
    at=AppTest.from_function(history_app).run()
    at.button(key='public_history_restore').click().run()
    assert not at.exception and not calls
    assert any(r['category']=='props' and r['outcome']=='PENDING' for r in at.session_state['returned_rows'])
    at.date_input(key='public_results_day').set_value(date(2026,9,9)).run()
    at.button(key='public_props_grade').click().run()
    assert not at.exception and len(calls)==1 and len(store.all('prop_stats'))==1
    assert any(r['category']=='props' and r['outcome']=='WIN' for r in at.session_state['returned_rows'])
    at.button(key='public_history_restore').click().run()
    assert not at.exception and len(calls)==1
    assert any(r['category']=='props' and r['outcome']=='WIN' for r in at.session_state['returned_rows'])


def test_corrected_time_requires_unique_same_day_and_pregame_publication():
    schedule,box=fixtures();entry=props.selections([publication()])[0]
    entry['leg']['start']='2026-09-09T23:00:00+00:00'
    revision=props.fetch_actuals(date(2026,9,9),[entry],http_get=fetcher(schedule,box,[]))
    assert revision['actuals'][0]['match_method']=='unique_matchup_date'
    assert revision['actuals'][0]['value']==2
    entry['published_at']='2026-09-09T20:01:00+00:00'
    blocked=props.fetch_actuals(date(2026,9,9),[entry],http_get=fetcher(schedule,box,[]))
    assert not blocked['actuals'] and 'timing conflicts' in blocked['unresolved'][0]['reason']
    entry['published_at']='2026-09-09T19:56:00+00:00'
    second=deepcopy(schedule['dates'][0]['games'][0]);second['gamePk']=2;second['gameDate']='2026-09-09T17:00:00Z'
    schedule['dates'][0]['games'].append(second)
    blocked=props.fetch_actuals(date(2026,9,9),[entry],http_get=fetcher(schedule,box,[]))
    assert not blocked['actuals'] and 'Ambiguous' in blocked['unresolved'][0]['reason']


def test_pending_reason_survives_revision_and_is_replaced_by_real_stat():
    p=publication();entry=props.selections([p])[0];schedule,box=fixtures()
    box['teams']['away']['players']['ID123']['stats']['batting']={}
    revision=props.fetch_actuals(date(2026,9,9),[entry],http_get=fetcher(schedule,box,[]))
    row=props.report([p],[revision])[0]
    assert row['outcome']=='NEEDS_REVIEW' and 'No recorded appearance' in row['final_score']
    revision2={'recorded_at':'2099-01-01T00:00:00+00:00','actuals':[{'id':entry['id'],'value':2}]}
    row=props.report([p],[revision,revision2])[0]
    assert row['outcome']=='WIN' and row['final_score']=='2 hits'


@pytest.mark.parametrize('reason',sorted(props.REVIEW_REASONS))
def test_final_evidence_issues_are_needs_review(reason):
    p=publication();entry=props.selections([p])[0]
    revision={'recorded_at':'2026-09-10T01:00:00+00:00','actuals':[],'unresolved':[{'id':entry['id'],'reason':reason}]}
    row=props.report([p],[revision])[0]
    assert row['outcome']=='NEEDS_REVIEW' and row['final_score']==reason
    from app_core.public_board import validate_package
    package=deepcopy(p['package']);package.update(schema_version=5,parlays=[],results=[row])
    validate_package(package)
    package['schema_version']=4
    with pytest.raises(ValueError):validate_package(package)
    package['schema_version']=5;row['category']='overall';row.pop('sport');row.pop('market')
    with pytest.raises(ValueError):validate_package(package)


@pytest.mark.parametrize('reason',['Game not final','Batch limit reached; run the next grading batch'])
def test_temporary_states_remain_pending(reason):
    p=publication();entry=props.selections([p])[0]
    revision={'recorded_at':'2026-09-10T01:00:00+00:00','actuals':[],'unresolved':[{'id':entry['id'],'reason':reason}]}
    assert props.report([p],[revision])[0]['outcome']=='PENDING'


def test_review_recheck_requires_explicit_selection(monkeypatch):
    # Exercise historical grading independently of the owner-selected public epoch.
    monkeypatch.setattr("app_core.public_record.START_DATE", "2026-09-09")
    from streamlit.testing.v1 import AppTest
    from app.ui import public_results
    from app_core.public_history import History
    from test_public_history import Memory
    store=History('site-1234','folder',Memory());p=publication();key=store.archive(p['package']);store.confirm('deployment123',key,p['confirmed_at'])
    entry=props.selections([p])[0]
    revision={'recorded_at':'2026-09-10T01:00:00+00:00','actuals':[],'unresolved':[{'id':entry['id'],'reason':next(iter(props.REVIEW_REASONS))}]}
    store.put('prop_stats/review.json',revision)
    monkeypatch.setattr(public_results,'history',lambda _:store)
    calls=[]
    monkeypatch.setattr(props,'fetch_actuals',lambda day,entries:calls.append(entries) or {'recorded_at':'2026-09-10T02:00:00+00:00','actuals':[{'id':entry['id'],'value':2}]})
    at=AppTest.from_function(history_app).run();at.button(key='public_history_restore').click().run()
    at.date_input(key='public_results_day').set_value(date(2026,9,9)).run()
    at.button(key='public_props_grade').click().run()
    assert not at.exception and not calls
    at.checkbox(key='public_props_recheck').check().run();at.button(key='public_props_grade').click().run()
    assert not calls
    at.checkbox(key='public_props_review').check().run();at.button(key='public_props_grade').click().run()
    assert not at.exception and len(calls)==1
    assert any(r['outcome']=='WIN' and r['category']=='props' for r in at.session_state['returned_rows'])


def test_original_projection_survives_later_analysis():
    p=publication();p['package']['props'][0]['expected_stat']=1.8
    later=deepcopy(p);later['confirmed_at']='2026-09-09T19:58:00+00:00'
    later['package']['props'][0]['expected_stat']=2.4
    assert props.report([later,p])[0]['expected_stat']==1.8
    assert 'expected_stat' not in props.report([publication()])[0]
