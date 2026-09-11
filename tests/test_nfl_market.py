from copy import deepcopy
from datetime import datetime, timezone, timedelta
from io import BytesIO
import sqlite3
import pytest
from app_core import nfl_market as n, nfl_market_store as store, research_scheduler as scheduler
from app_core.research_api_budget import Budget, BudgetLimit

NOW = datetime.now(timezone.utc)


def event(gid="game", start=None):
    return {"id": gid, "sport_key": "americanfootball_nfl", "home_team": "Detroit Lions", "away_team": "New Orleans Saints",
            "commence_time": (start or NOW + timedelta(hours=1)).isoformat(),
            "bookmakers": [{"key": "book", "markets": [
                {"key": "h2h", "last_update": NOW.isoformat(), "outcomes": [{"name": "Detroit Lions", "price": -150}, {"name": "New Orleans Saints", "price": 130}]},
                {"key": "spreads", "last_update": NOW.isoformat(), "outcomes": [{"name": "Detroit Lions", "price": -110, "point": -3}, {"name": "New Orleans Saints", "price": -110, "point": 3}]},
                {"key": "totals", "last_update": NOW.isoformat(), "outcomes": [{"name": "Over", "price": -110, "point": 43}, {"name": "Under", "price": -110, "point": 43}]}]}]}


class Response:
    status_code = 200
    headers = {}
    def __init__(self, data): self.data = data
    def json(self): return self.data


class Cloud:
    def __init__(self): self.objects = {}
    def get_paginator(self, name): return self
    def paginate(self, **kw): return [{"Contents": [{"Key": k} for k in self.objects if k.startswith(kw["Prefix"])]}]
    def put_object(self, **kw): self.objects[kw["Key"]] = kw["Body"]
    def get_object(self, **kw): return {"Body": BytesIO(self.objects[kw["Key"]])}


def test_fresh_complete_pairs_only():
    e = event(); q, rejected = n.quotes(e, NOW)
    assert len(q) == 6 and not rejected
    bad = deepcopy(e)
    bad['bookmakers'][0]['markets'][0]['last_update'] = (NOW - timedelta(minutes=16)).isoformat()
    bad['bookmakers'][0]['markets'][1]['outcomes'][1]['point'] = 4
    bad['bookmakers'][0]['markets'][2]['outcomes'][0]['price'] = float('nan')
    q, rejected = n.quotes(bad, NOW)
    assert not q and sum(rejected.values()) == 3
    e['bookmakers'][0]['markets'][0].pop('last_update')
    e['bookmakers'][0]['last_update'] = NOW.isoformat()
    assert len(n.quotes(e, NOW)[0]) == 4  # book-level timestamp is insufficient


def test_capture_once_and_no_paid_odds_outside_window(tmp_path, monkeypatch):
    monkeypatch.setattr(n, 'utcnow', lambda: NOW)
    path = tmp_path / 'nfl.sqlite3'; calls = []; backups = []
    def get(url, **kwargs):
        calls.append(url.rsplit('/', 1)[-1])
        return Response([event(), event('later', NOW+timedelta(hours=3))])
    r = n.run(path, 'key', lambda: backups.append(1), get)
    assert r['captured'] == 1 and calls == ['events', 'odds'] and backups
    assert n.report(path)['quote_rows'] == 6
    assert n.report(path)['production_eligible'] is False
    calls.clear()
    r = n.run(path, 'key', lambda: None, get)
    assert r['captured'] == 0 and calls == ['events']
    assert n.report(path)['captured_games'] == 1


def saved_capture(path, start):
    e = event(start=start); ident = n.identity(e)
    q, _ = n.quotes(e, NOW)
    store.insert({'schema':1, 'kind':'capture', 'created_at':(start-timedelta(hours=1)).isoformat(),
                  'data':{'sport':'NFL','protocol':n.PROTOCOL,'events':[{**ident,'quotes':q}]}}, path)
    return e, {**ident, 'captured_at': (start-timedelta(hours=1)).isoformat(), 'quotes': q}


def test_completed_scores_grade_exact_identity_and_push(tmp_path, monkeypatch):
    monkeypatch.setattr(n, 'utcnow', lambda: NOW)
    path = tmp_path/'nfl.sqlite3'
    e, captured = saved_capture(path, NOW-timedelta(hours=4))
    e.update(completed=True, last_update=NOW.isoformat(), scores=[{'name':e['home_team'],'score':'23'}, {'name':e['away_team'],'score':'20'}])
    calls = []
    def get(url, **kw):
        calls.append(url.rsplit('/',1)[-1]); return Response([e] if url.endswith('scores') else [])
    r = n.run(path, 'key', lambda: None, get)
    assert r['graded'] == 1 and calls == ['scores', 'events']
    report = n.report(path)
    assert report['graded_games'] == 1
    assert [q['result'] for q in report['quotes']] == ['win','loss','push','push','push','push']
    calls.clear(); n.run(path, 'key', lambda: None, get)
    assert calls == ['events']
    for modification in ({'id':'other'}, {'away_team':'Wrong'}, {'commence_time':(NOW-timedelta(hours=3)).isoformat()}):
        with pytest.raises(ValueError): n.final_score({**e, **modification}, captured, NOW)
    assert n.final_score({**e,'completed':False}, captured, NOW) is None
    with pytest.raises(ValueError): n.final_score({**e,'scores':[{'name':e['home_team'],'score':True}, {'name':e['away_team'],'score':20}]},captured,NOW)
    with pytest.raises(ValueError): n.final_score({**e,'last_update':(NOW+timedelta(minutes=1)).isoformat()},captured,NOW)
    tied = {**e,'scores':[{'name':e['home_team'],'score':'20'}, {'name':e['away_team'],'score':'20'}]}
    score = n.final_score(tied,captured,NOW)
    assert n.comparison(captured['quotes'][0],captured,score) == 'tie'


def test_stale_pending_is_reported_without_paid_score_request(tmp_path, monkeypatch):
    monkeypatch.setattr(n,'utcnow',lambda:NOW)
    path=tmp_path/'nfl.sqlite3';saved_capture(path,NOW-timedelta(days=4));calls=[]
    def get(url,**kw):calls.append(url);return Response([])
    r=n.run(path,'key',lambda:None,get)
    assert r['unresolved_past_score_window']==1 and len(calls)==1 and calls[0].endswith('events')


def test_remote_restore_integrity_and_append_only(tmp_path):
    path=tmp_path/'a.sqlite3';saved_capture(path,NOW-timedelta(hours=4));c=Cloud()
    store.sync(path,client=c,folder='folder')
    restored=tmp_path/'b.sqlite3';store.sync(restored,client=c,folder='folder')
    assert store.records(path)==store.records(restored)
    with sqlite3.connect(restored) as db:
        with pytest.raises(sqlite3.IntegrityError): db.execute('DELETE FROM records')
    key=next(iter(c.objects));c.objects[key]=b'{}'
    with pytest.raises(ValueError,match='integrity'):store.sync(restored,client=c,folder='folder')
    with pytest.raises(ValueError,match='NFL'):store.insert({'schema':1,'kind':'scores','data':{'sport':'MLB'}},path)


def test_scheduler_nfl_no_model_or_cfbd_required(tmp_path,monkeypatch):
    from app_core import research_api_budget as budgets
    monkeypatch.setattr(budgets,'is_open',lambda at:True)
    monkeypatch.setattr(n,'utcnow',lambda:NOW)
    c=Cloud();calls=[]
    def get(url,**kwargs):calls.append(url);return Response([event()])
    monkeypatch.setattr(budgets.requests,'get',get)
    r=scheduler.run(['NFL'],tmp_path,c,'folder',odds_key='secret')
    assert not r['errors'] and r['sports']['NFL']['captured']==1
    assert r['api_budget']['usage']['ODDS']['daily']==3
    assert r['api_budget']['usage']['CFBD']['daily']==0
    assert any(k.startswith(store.PREFIX) for k in c.objects)
    assert 'secret' not in str(c.objects)
    calls.clear()
    r=scheduler.run(['NFL'],tmp_path/'restored',c,'folder',odds_key='secret')
    assert not r['errors'] and len(calls)==1 and calls[0].endswith('events')


def test_restore_failure_prevents_nfl_requests(tmp_path,monkeypatch):
    def fail(*a,**kw):raise RuntimeError('private-key')
    monkeypatch.setattr(store,'sync',fail)
    monkeypatch.setattr(n,'run',lambda *a:pytest.fail('No work before restore'))
    r=scheduler.run(['NFL'],tmp_path,Cloud(),'folder',odds_key='key')
    assert r['errors']==['NFL:RuntimeError']


def test_budget_pause_and_provider_error_safe(tmp_path,monkeypatch):
    monkeypatch.setattr(n,'utcnow',lambda:NOW)
    def paused(*a,**kw):raise BudgetLimit('api_budget:ODDS:daily')
    r=n.run(tmp_path/'a','key',lambda:None,paused)
    assert r['budget_paused'] and not r['errors']
    def failed(*a,**kw):raise RuntimeError('apiKey=SECRET')
    r=n.run(tmp_path/'b','key',lambda:None,failed)
    assert r['errors']==['nfl_capture_failed'] and 'SECRET' not in str(r)


def test_duplicate_events_and_post_kickoff_odds_rejected(tmp_path,monkeypatch):
    e=event()
    with pytest.raises(ValueError,match='duplicate'): n.fetch('events','key',lambda *a,**kw:Response([e,{**e,'home_team':'Other'}]))
    monkeypatch.setattr(n,'utcnow',lambda:NOW)
    def get(url,**kw):return Response([e] if url.endswith('events') else [event(start=NOW-timedelta(seconds=1))])
    r=n.run(tmp_path/'x','key',lambda:None,get)
    assert r['captured']==0 and r['excluded_markets']['identity_or_pregame_window']==1


def test_minor_kickoff_corrections_preserve_original_capture(tmp_path):
    e,captured=saved_capture(tmp_path/'timing',NOW-timedelta(hours=4))
    e.update(completed=True,last_update=NOW.isoformat(),scores=[{'name':e['home_team'],'score':'23'},{'name':e['away_team'],'score':'20'}])
    for seconds in (202,145,-202):
        changed={**e,'commence_time':(n.timestamp(captured['start'])+timedelta(seconds=seconds)).isoformat()}
        score=n.final_score(changed,captured,NOW)
        assert score['start']==captured['start']
        assert score['reported_start']==changed['commence_time']
        assert score['home_score']==23
    with pytest.raises(ValueError):
        n.final_score({**changed,'home_team':'Different team'},captured,NOW)
    with pytest.raises(ValueError):
        n.final_score({**e,'commence_time':(n.timestamp(captured['start'])+timedelta(minutes=16)).isoformat()},captured,NOW)
    late={**captured,'captured_at':(n.timestamp(captured['start'])-timedelta(seconds=30)).isoformat()}
    with pytest.raises(ValueError,match='timing'):
        n.final_score({**e,'commence_time':(n.timestamp(captured['start'])-timedelta(minutes=2)).isoformat()},late,NOW)
