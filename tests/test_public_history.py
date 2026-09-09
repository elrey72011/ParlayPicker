from copy import deepcopy
from datetime import datetime, timezone
from io import BytesIO
import pytest
from app_core.public_history import History, digest, report, selections, grade_leg


def leg(pick='Boston +1.5', market='spread_home'):
    return dict(sport='MLB',game='Seattle at Boston',pick=pick,market=market,odds=-110,status='APPROVED',start='2026-09-09T20:00:00+00:00',as_of='2026-09-09T19:55:00+00:00',player='',win_estimate=.6,ev=.1)


def pub():
    package=dict(schema_version=1,built_at='2026-09-09T19:55:00+00:00',stale_after_minutes=15,games={'overall':[leg()],'sides':[leg()],'totals':[leg('Under 8.5','total_under')]},props=[],dfs=[])
    return dict(package=package,package_hash=digest(package),confirmed_at='2026-09-09T19:56:00+00:00')


def scores():
    from app_core.result_team_names import normalize_result_team as norm
    return [{'sport':'MLB','away':norm('Seattle'),'home':norm('Boston'),'event_id':'1','start':'2026-09-09T20:00:00+00:00','away_score':4,'home_score':3}]


def test_first_publication_dedup_and_late_exclusion():
    first=pub();later=deepcopy(first);later['confirmed_at']='2026-09-09T19:57:00+00:00';later['package']['games']['overall'][0]['pick']='Boston -1.5'
    assert len(selections([later,first]))==3
    assert selections([later,first])[0]['legs'][0]['pick']=='Boston +1.5'
    first['confirmed_at']='2026-09-09T20:00:00+00:00'
    assert selections([first])==[]


def test_outcomes_and_complete_ticket_push_pending():
    p=pub();r=report([p],[{'recorded_at':'2026-09-10T00:00:00Z','scores':scores()}])
    assert len(r)==3 and all(x['outcome']=='WIN' for x in r)
    assert grade_leg(leg('Boston -1.5'),scores())[0]=='LOSS'
    assert grade_leg(leg('Under 7','total_under'),scores())[0]=='PUSH'
    assert grade_leg(leg(),[scores()[0],{**scores()[0],'event_id':'2'}])[0]=='PENDING'
    p['package']['parlays']=[{'legs':[leg(),leg('Under 7','total_under')]}]
    assert report([p],[{'recorded_at':'2026-09-10T00:00:00Z','scores':scores()}])[-1]['outcome']=='PUSH'
    assert report([p],[])[-1]['outcome']=='PENDING'
    assert report([p],[])[-1]['group']=='Research'


class Exists(Exception):
    response={'Error':{'Code':'PreconditionFailed'}}
class Memory:
    def __init__(self):self.data={}
    def get_object(self,Key):return {'Body':BytesIO(self.data[Key])}
    def put_object(self,Key,Body,**kw):
        if Key in self.data:raise Exists()
        self.data[Key]=Body
    def get_paginator(self,*a):return self
    def paginate(self,Prefix):return [{'Contents':[{'Key':k} for k in self.data if k.startswith(Prefix)]}]


def test_archive_restore_idempotent_confirmation_and_site_isolation():
    client=Memory();store=History('site-1234','folder',client)
    package=pub()['package'];key=store.archive(package)
    assert store.publications()==[] # draft not counted
    store.confirm('deploy-123',key,'2026-09-09T19:56:00+00:00')
    store.confirm('deploy-123',key,'2026-09-10T19:56:00+00:00')
    restored=History('site-1234','folder',client).publications()
    assert restored[0]['confirmed_at']=='2026-09-09T19:56:00+00:00'
    assert len(selections(restored))==3
    assert History('other-1234','folder',client).publications()==[]


def test_score_correction_revises_outcome_not_selection():
    p=pub();s=scores();corrected=[{**s[0],'away_score':8}]
    rows=report([p],[{'recorded_at':'2026-09-10T00:00:00Z','scores':s},{'recorded_at':'2026-09-10T01:00:00Z','scores':corrected}])
    assert rows[0]['outcome']=='LOSS' and 'Boston +1.5' in rows[0]['picks']

def test_stale_and_unknown_markets_are_not_counted():
    p=pub();p['confirmed_at']='2026-09-09T20:11:00+00:00'
    assert selections([p])==[]
    p=pub()
    for rows in p['package']['games'].values():rows[0]['market']='unknown'
    assert selections([p])==[]


def test_results_schema_rejects_private_fields():
    from app_core.public_board import validate_package
    p=pub()['package'];p.update(schema_version=3,parlays=[],results=report([pub()],[]))
    validate_package(p)
    p['results'][0]['token']='private'
    with pytest.raises(ValueError):validate_package(p)


def test_explicit_restore_and_grade_no_rerun_requests(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from app.ui import public_results as ui
    client=Memory();store=History('site-1234','folder',client)
    key=store.archive(pub()['package']);store.confirm('deploy-123',key,pub()['confirmed_at'])
    calls=[]
    monkeypatch.setattr(ui,'history',lambda setting:store)
    monkeypatch.setattr(ui,'fetch_scores',lambda day,sports: calls.append((day,sports)) or {'recorded_at':'2026-09-10T00:00:00Z','scores':scores()})
    at=AppTest.from_string("from app.ui.public_results import render_history\nrender_history(lambda key: 'site-1234' if key=='PARLAYPICKER_NETLIFY_SITE_ID' else 'folder')").run()
    assert not at.exception and not calls
    at.button(key='public_history_restore').click().run()
    from datetime import date
    at.date_input(key='public_results_day').set_value(date(2026,9,9)).run()
    assert not calls
    at.button(key='public_history_grade').click().run()
    assert not at.exception and len(calls)==1
    at.run()
    assert len(calls)==1 and len(store.all('scores'))==1


def test_provider_only_accepts_completed_final_and_keeps_event_identity(monkeypatch):
    import requests
    from types import SimpleNamespace
    from datetime import date
    from app_core.public_history import fetch_scores
    game={'date':'2026-09-09T20:00:00Z','status':{'type':{'completed':True,'state':'post','name':'STATUS_FINAL'}},'competitors':[{'homeAway':'away','score':'4','team':{'displayName':'Seattle'}},{'homeAway':'home','score':'0','team':{'displayName':'Boston'}}]}
    cancelled=deepcopy(game);cancelled['status']['type']['name']='STATUS_POSTPONED'
    monkeypatch.setattr(requests,'get',lambda *a,**k:SimpleNamespace(raise_for_status=lambda:None,json=lambda:{'events':[{'id':'1','competitions':[game]},{'id':'2','competitions':[cancelled]}]}))
    result=fetch_scores(date(2026,9,9),{'MLB'})
    assert len(result['scores'])==1 and result['scores'][0]['home_score']==0
