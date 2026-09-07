from datetime import datetime, timezone
from unittest.mock import Mock, patch
from copy import deepcopy
import pytest
from app_core import ncaaf_closing as c
from app_core import ncaaf_prospective_store as store

AT=datetime(2030,9,1,11,45,tzinfo=timezone.utc)
START='2030-09-01T12:00:00+00:00'


def event():
    return dict(id='event',home_team='Alabama',away_team='Georgia',commence_time=START,
                bookmakers=[dict(key='book',last_update=AT.isoformat(),markets=[dict(key='spreads',outcomes=[dict(name='Alabama',point=-3.5,price=-120)])])])


def seed(path):
    mid=store.insert(dict(schema=1,kind='model',created_at='2030-08-01T00:00:00Z',data={}),path)
    q=dict(book='book',market_type='spread_home',point=-3.5,price=100,recorded_at='2030-09-01T10:00:00Z')
    e=dict(event_id='event',cfbd_id=1,start=START,home='Alabama',away='Georgia',models={'ridge':{'selected':q}})
    store.insert(dict(schema=1,kind='capture',created_at='2030-09-01T10:00:01Z',data=dict(model_id=mid,captured_at='2030-09-01T10:00:00Z',events=[e])),path)


def save_close(path,e=None):
    with patch.object(c.prospective,'utcnow',return_value=AT):
        return c.capture('secret',get=Mock(return_value=Mock(status_code=200,json=Mock(return_value=[e or event()]))),path=path)


def test_capture_without_predictions_and_positive_clv(tmp_path):
    path=tmp_path/'x.db'
    assert save_close(path)['saved_events']==1
    assert c.report(path)['comparable_markets']==0
    seed(path)
    r=c.report(path)
    assert r['comparable_markets']==1
    assert r['rows'][0]['price_clv']==pytest.approx(2/(1+100/120)-1)


@pytest.mark.parametrize('case',['started','far','stale','future','missing_time'])
def test_capture_timing_gates(tmp_path,case):
    e=event()
    if case=='started':e['commence_time']='2030-09-01T11:00:00Z'
    if case=='far':e['commence_time']='2030-09-01T13:00:00Z'
    if case=='stale':e['bookmakers'][0]['last_update']='2030-09-01T11:00:00Z'
    if case=='future':e['bookmakers'][0]['last_update']='2030-09-01T11:50:00Z'
    if case=='missing_time':e['bookmakers'][0].pop('last_update')
    assert save_close(tmp_path/'x.db',e)['saved_events']==0


@pytest.mark.parametrize('case',['line','book','event','teams','start','duplicate'])
def test_noncomparable_quotes(tmp_path,case):
    path=tmp_path/'x.db';seed(path);e=event()
    if case=='line':e['bookmakers'][0]['markets'][0]['outcomes'][0]['point']=-4.5
    if case=='book':e['bookmakers'][0]['key']='other'
    if case=='event':e['id']='other'
    if case=='teams':e['away_team']='Other'
    if case=='start':e['commence_time']='2030-09-01T12:01:00Z'
    if case=='duplicate':e['bookmakers'][0]['markets'][0]['outcomes']*=2
    save_close(path,e)
    assert c.report(path)['comparable_markets']==0


def test_latest_changed_line_does_not_cherry_pick(tmp_path):
    path=tmp_path/'x.db';seed(path);save_close(path)
    r=next(x for x in store.records(path) if x['kind']=='closing')
    data=deepcopy(r['data']);data['observed_at']='2030-09-01T11:50:00Z'
    data['events'][0]['quotes'][0]['point']=-4.5
    store.insert(dict(schema=1,kind='closing',created_at='2030-09-01T11:50:00Z',data=data),path)
    assert c.report(path)['rows'][0]['status']=='line_changed_no_price_clv'
