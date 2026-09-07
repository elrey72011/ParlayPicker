from copy import deepcopy
from datetime import datetime, timezone, timedelta
from io import BytesIO
from unittest.mock import Mock, patch
import json
import sqlite3
import pytest
import requests
from app_core import ncaaf_prospective as p
from app_core import ncaaf_prospective_store as store
from app_core.ncaaf_history import build_dataset

NOW = datetime(2026,10,29,12,tzinfo=timezone.utc)


def inputs():
    games = []
    batches = []
    for i, day in enumerate((1,9,17),1):
        g = dict(id=i, season=2026, week=i, seasonType="regular", startDate=f"2026-10-{day:02d}T12:00:00Z",
                 startTimeTBD=False, completed=True, neutralSite=False,
                 homeId=1, awayId=2, homeTeam="Alabama", awayTeam="Georgia", homePoints=30, awayPoints=20)
        games.append(g)
        batches.append(dict(request=dict(kind="stats", year=2026, week=i, season_type="regular"), retrieved_at=NOW.isoformat(),
                            records=[dict(id=i, teams=[dict(teamId=t, points=score, stats=[dict(category="totalYards", stat="350")]) for t,score in ((1,30),(2,20))])]))
    future = dict(games[0], id=9, week=5, startDate="2026-10-30T12:00:00Z", completed=False, homePoints=None, awayPoints=None)
    games.append(future)
    return dict(schema=1, years=[2026], batches=[dict(request=dict(kind="games",year=2026),retrieved_at=NOW.isoformat(),records=games)]+batches)


def model():
    value = {"id":"frozen", "created_at":"2026-10-28T12:00:00Z", "data":{"runtime_hash":p.runtime_hash(),
            "artifact":{"models":{"constant":{"margin":{"kind":"constant","intercept":10.,"bias":0.,"sigma":12.},
                                                "total":{"kind":"constant","intercept":50.,"bias":0.,"sigma":12.}}}}}}
    value["data"]["artifact_hash"]=p.digest(value["data"]["artifact"])
    return value


def event():
    return dict(id="odds9", home_team="Alabama", away_team="Georgia", commence_time="2026-10-30T12:00:00Z",
                bookmakers=[dict(key="draftkings", last_update=NOW.isoformat(), markets=[dict(key="h2h",outcomes=[
                    dict(name="Alabama",price=-110),dict(name="Georgia",price=100)])])])


def response(value, status=200):
    return Mock(status_code=status,json=Mock(return_value=value))


def test_upcoming_features_without_fabricated_scores():
    state=inputs()
    _,rows,targets=build_dataset(state,feature_targets=[state["batches"][0]["records"][-1]])
    assert rows[0]["home_prior_games"]==3 and rows[0]["home_ppg"]==30
    assert rows[0]["scoring_features_available"]
    assert 9 not in [r["game_id"] for r in targets]


def test_capture_and_grade_exact_id_then_first_selection(tmp_path):
    path=tmp_path/'p.sqlite3'
    frozen=model()
    frozen['id']=store.insert({'schema':1,'kind':'model','created_at':frozen['created_at'],'data':frozen['data']},path)
    with patch.object(p,'utcnow',return_value=NOW):
        _, info=p.capture(inputs(),frozen,'secret',get=Mock(return_value=response([event()])),path=path)
        assert info['saved_games']==1
        _,info=p.capture(inputs(),frozen,'secret',get=Mock(return_value=response([event()])),path=path)
    assert len(store.records(path))==3
    captured=next(r['data'] for r in store.records(path) if r['kind']=='capture')
    q=captured['events'][0]['models']['constant']['selected']
    assert q['live_stake']==0 and q['no_vig_probability'] is not None
    assert 'secret' not in json.dumps(captured)
    final=dict(inputs()['batches'][0]['records'][-1],completed=True,homePoints=35,awayPoints=20)
    with patch.object(p,'utcnow',return_value=NOW+timedelta(days=2)):
        info=p.grade('key',path=path,get=Mock(return_value=response([final])))
    assert info['graded']==1
    report=p.report(path)
    assert report['captured_games_by_cohort']==1 and report['graded_selections']==1
    assert report['summary'][0]['paper_hit_rate']==1
    with store.connect(path) as db:
        with pytest.raises(sqlite3.IntegrityError):
            db.execute('DELETE FROM records')


@pytest.mark.parametrize('change', ['started','stale','future_quote','ambiguous','no_stats'])
def test_capture_fails_closed(tmp_path,change):
    state,e=inputs(),event()
    if change=='started': e['commence_time']='2026-10-28T12:00:00Z'
    if change=='stale': e['bookmakers'][0]['last_update']='2026-10-29T11:00:00Z'
    if change=='future_quote': e['bookmakers'][0]['last_update']='2026-10-29T13:00:00Z'
    if change=='ambiguous': state['batches'][0]['records'].append(deepcopy(state['batches'][0]['records'][-1]))
    if change=='no_stats':
        for b in state['batches'][1:]: b['records']=[]
    with patch.object(p,'utcnow',return_value=NOW):
        _,info=p.capture(state,model(),'key',get=Mock(return_value=response([e])),path=tmp_path/'p.sqlite3')
    assert info['saved_games']==0


def test_old_inputs_and_runtime_change_rejected(tmp_path):
    state=inputs()
    state['batches'][0]['retrieved_at']='2026-10-01T00:00:00Z'
    with patch.object(p,'utcnow',return_value=NOW):
        with pytest.raises(ValueError,match='24 hours'): p.capture(state,model(),'key',path=tmp_path/'p.sqlite3')
        m=model();m['data']['runtime_hash']='old'
        with pytest.raises(ValueError,match='changed'): p.capture(inputs(),m,'key',path=tmp_path/'p.sqlite3')


@pytest.mark.parametrize('value',[requests.Timeout('secret'),ValueError('secret')])
def test_safe_provider_errors(value):
    get=Mock(side_effect=value) if isinstance(value,requests.RequestException) else Mock(return_value=Mock(status_code=200,json=Mock(side_effect=value)))
    with pytest.raises(ValueError) as e: p.fetch('https://api.collegefootballdata.com/games','secret',{},get=get)
    assert 'secret' not in str(e.value)


def test_refresh_bounded_and_never_fetches_future_week():
    state=inputs()
    state['batches']=state['batches'][:1]
    with patch.object(p,'utcnow',return_value=NOW):
        get=Mock(return_value=response([]))
        state,status=p.refresh(state,'key',get=get)
        assert status=='ready' and get.call_count==3
        assert {c.kwargs['params']['week'] for c in get.call_args_list}=={1,2,3}
        assert not p.pending(state)


def test_spread_and_total_orientation_and_push():
    e=event()
    e['bookmakers'][0]['markets']=[dict(key='spreads',outcomes=[dict(name='Alabama',point=-10,price=-110),dict(name='Georgia',point=10,price=-110)]),
                                dict(key='totals',outcomes=[dict(name='Over',point=50,price=-110),dict(name='Under',point=50,price=-110)])]
    feature=build_dataset(inputs(),feature_targets=[inputs()['batches'][0]['records'][-1]])[1][0]
    rows=p.quote_candidates(e,model()['data']['artifact']['models']['constant'],feature,NOW)
    assert len(rows)==4
    assert all(r['win']==pytest.approx(r['loss'],abs=0.0001) and r['push']>0 for r in rows)
    assert all(r['no_vig_probability']==pytest.approx(.5) for r in rows)


def test_drive_roundtrip_and_integrity(tmp_path):
    from app_core.evidence_drive import AlreadyExists
    objects={}
    client=Mock()
    def put(**kw):
        if kw['Key'] in objects: raise AlreadyExists()
        objects[kw['Key']]=kw['Body']
    client.put_object.side_effect=put
    client.get_object.side_effect=lambda **kw:{'Body':BytesIO(objects[kw['Key']])}
    client.get_paginator.return_value.paginate.side_effect=lambda **kw:[{'Contents':[{'Key':k} for k in objects]}]
    a,b=tmp_path/'a.sqlite3',tmp_path/'b.sqlite3'
    store.save('scores',{'scores':[]},a)
    store.sync(a,client=client,folder='folder')
    store.sync(b,client=client,folder='folder')
    assert store.records(a)==store.records(b)
    objects[next(iter(objects))]=b'wrong'
    with pytest.raises(ValueError,match='integrity'): store.sync(b,client=client,folder='folder')


def test_ui_does_not_call_providers_on_rerun():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_string('from app.ui.ncaaf_prospective import render_ncaaf_prospective\nrender_ncaaf_prospective()')
    with patch('app.ui.ncaaf_prospective.store.records',return_value=[]),patch('app.ui.ncaaf_prospective.prospective.report',return_value={'captured_games_by_cohort':0,'graded_selections':0,'summary':[]}),patch('app.ui.ncaaf_prospective.prospective.fetch') as fetch:
        app.run();app.run()
        assert not app.exception
        fetch.assert_not_called()
