import json
from datetime import datetime, timezone, timedelta
from copy import deepcopy
import pytest
from app_core import mlb_prospective as p
from app_core import mlb_prospective_store as store


def fixtures():
    current=datetime.now(timezone.utc)
    finals=[{"gamePk":i,"gameDate":(current-timedelta(days=i+1)).isoformat(),
             "status":{"abstractGameState":"Final"},
             "teams":{"home":{"team":{"id":1},"score":5},"away":{"team":{"id":2},"score":2}}} for i in range(1,11)]
    game={"gamePk":99,"gameDate":(current+timedelta(days=1)).isoformat(),"status":{"abstractGameState":"Preview"},
          "teams":{s:{"team":{"id":i},"probablePitcher":{"id":i+10}} for s,i in [("home",1),("away",2)]}}
    log={"stats":[{"splits":[{"game":{"gamePk":i},"stat":{"inningsPitched":"3.1","earnedRuns":1,"strikeOuts":4,"baseOnBalls":1,"hits":2}} for i in (1,2,3)]}]}
    return game,finals,{"home":log,"away":deepcopy(log)}


def test_observed_final_features_and_innings():
    g,finals,logs=fixtures()
    r=p.live_features(g,finals,logs)
    assert r["home_ppg"]==5 and r["away_ppg"]==2
    assert r["home_starter_era"]==pytest.approx(2.7)
    logs["home"]["stats"][0]["splits"][0]["game"]["gamePk"]=99
    with pytest.raises(ValueError,match="Insufficient pitcher"):
        p.live_features(g,finals,logs)


def test_capture_runtime_pregame_and_persistence(tmp_path,monkeypatch):
    path=tmp_path/"mlb.db"
    model={"runtime_hash":p.runtime_hash(),"models":{name:{"columns":["home_ppg"],"mean":[0],"scale":[1],
           "targets":{t:{"coefficients":[1],"intercept":0} for t in ("margin","total")}} for name in ("team_only","team_and_starter")}}
    store.save("model",model,path)
    g,finals,logs=fixtures()
    monkeypatch.setattr(p,"schedule",lambda:{"dates":[{"games":finals+[g]}]})
    monkeypatch.setattr(p,"fetch",lambda *args,**kwargs:logs["home"])
    finals.append(deepcopy(finals[0]))  # resumed-game duplicate must count once
    p.capture(99,path)
    r=store.records(path)[-1]
    assert r["data"]["events"][0]["probable_pitchers"]=={"home":11,"away":12}
    assert r["data"]["production_eligible"] is False
    g["status"]["abstractGameState"]="Live"
    with pytest.raises(ValueError,match="pregame"):p.capture(99,path)
    g["status"]["abstractGameState"]="Preview"
    del g["teams"]["home"]["probablePitcher"]
    with pytest.raises(ValueError,match="probable"):p.capture(99,path)


def test_first_capture_and_starter_change_retained(tmp_path):
    path=tmp_path/"evidence.db"
    event={"game_id":1,"start":"2030-01-01T00:00:00Z","probable_pitchers":{"home":11,"away":12},
           "forecasts":{k:{"margin":1,"total":8} for k in ("team_only","team_and_starter")}}
    store.save("capture",{"model_id":"a","events":[event]},path)
    second=deepcopy(event);second["forecasts"]["team_only"]["margin"]=99
    store.save("capture",{"model_id":"a","events":[second]},path)
    store.save("scores",{"game_id":1,"started_at":"2030-01-01T00:00:00Z","home_score":5,"away_score":3,
                         "actual_starters":{"home":13,"away":12}},path)
    report=p.report(path)
    assert report["captured_games_by_cohort"]==1
    assert report["summary"][0]["starter_changes"]==1
    assert report["summary"][0]["mae"]["team_only"]["margin"]==1


def test_store_append_only_and_hash_verification(tmp_path):
    import sqlite3
    path=tmp_path/"store.db"
    store.save("model",{"test":True},path)
    with store.connect(path) as db:
        with pytest.raises(sqlite3.IntegrityError):db.execute("DELETE FROM records")
    assert store.PREFIX=="parlaypicker/mlb-prospective-v1/"


def test_grade_identity_and_score_capture(tmp_path,monkeypatch):
    path=tmp_path/"grade.db"
    store.save("capture",{"model_id":"x","events":[{"game_id":5,"start":"2030-01-01T00:00:00Z","home_id":1,"away_id":2}]},path)
    monkeypatch.setattr(p,"fetch",lambda *a,**kw:{"gameData":{"status":{"abstractGameState":"Final"}}})
    monkeypatch.setattr(p,"normalize_game",lambda feed:{"game_id":5,"home_id":1,"away_id":2,"home_score":5,"away_score":3})
    monkeypatch.setattr(p,"parse_boxscore",lambda *a:{"starters":{"home":11,"away":12}})
    assert p.grade(path)==1
    assert p.grade(path)==0


def test_empty_ui(tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("PARLAYPICKER_EVIDENCE_DIR",str(tmp_path))
    app=AppTest.from_string("from app.ui.mlb_prospective import render_mlb_prospective\nrender_mlb_prospective()")
    app.run(timeout=20)
    assert not app.exception
    assert any(b.label=="Freeze MLB comparison models" and b.disabled for b in app.button)
