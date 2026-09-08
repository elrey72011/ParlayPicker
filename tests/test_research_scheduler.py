from datetime import datetime, timezone, timedelta
from io import BytesIO
import pytest
from app_core import research_scheduler as s


class Cloud:
    def __init__(self):self.objects={}
    def get_paginator(self,name):return self
    def paginate(self,**kw):return [{"Contents":[{"Key":k} for k in self.objects if k.startswith(kw["Prefix"])]}]
    def put_object(self,**kw):self.objects[kw["Key"]]=kw["Body"]
    def get_object(self,**kw):return {"Body":BytesIO(self.objects[kw["Key"]])}


def test_state_roundtrip_and_tamper():
    c=Cloud();assert s.checkpoint(c,"folder")=={}
    s.checkpoint(c,"folder",{"attempt":1})
    assert s.checkpoint(c,"folder")=={"attempt":1}
    c.objects[next(iter(c.objects))]=b"{}"
    with pytest.raises(ValueError,match="integrity"):s.checkpoint(c,"folder")


def test_capture_window():
    now=datetime(2026,9,8,tzinfo=timezone.utc)
    assert not s.due(now.isoformat(),now)
    assert s.due((now+timedelta(hours=2)).isoformat(),now)
    assert not s.due((now+timedelta(hours=2,seconds=1)).isoformat(),now)


def test_restore_failure_prevents_work_and_hides_secrets(tmp_path,monkeypatch):
    c=Cloud();calls=[]
    def bad(*a,**kw):raise RuntimeError("secret-api-key")
    monkeypatch.setattr(s.ms,"sync",bad)
    monkeypatch.setattr(s,"run_mlb",lambda *a:calls.append(1))
    result=s.run(["MLB"],tmp_path,c,"folder")
    assert not calls and result["errors"]==["MLB:RuntimeError"]
    assert "secret-api-key" not in str(c.objects)


def test_backup_after_stage_failure_and_other_sport_continues(tmp_path,monkeypatch):
    c=Cloud();calls=[]
    monkeypatch.setattr(s.ms,"sync",lambda *a,**kw:calls.append("mlb_backup"))
    monkeypatch.setattr(s.ns,"sync",lambda *a,**kw:calls.append("ncaaf_backup"))
    def fail(*a):raise ValueError("bad")
    monkeypatch.setattr(s,"run_mlb",fail)
    monkeypatch.setattr(s,"run_ncaaf",lambda *a:{"errors":[],"captured":1})
    r=s.run(["MLB","NCAAF"],tmp_path,c,"folder")
    assert calls.count("mlb_backup")==2
    assert r["sports"]["NCAAF"]["captured"]==1


def test_mlb_skips_existing_and_outside_window(tmp_path,monkeypatch):
    now=datetime.now(timezone.utc);saved=[]
    records=[{"kind":"model","id":"m","data":{"runtime_hash":"h"}},
             {"kind":"capture","data":{"model_id":"m","events":[{"game_id":1,"start":(now+timedelta(hours=1)).isoformat()}]}}]
    monkeypatch.setattr(s.ms,"records",lambda *a:records)
    monkeypatch.setattr(s.mlb,"runtime_hash",lambda:"h")
    monkeypatch.setattr(s.mlb,"upcoming",lambda:[{"gamePk":1,"gameDate":(now+timedelta(hours=1)).isoformat()},
        {"gamePk":2,"gameDate":(now+timedelta(hours=3)).isoformat()}])
    monkeypatch.setattr(s.mlb,"capture",lambda *a:saved.append(a))
    r=s.run_mlb(tmp_path/"x",{},lambda:None)
    assert not saved and r["captured"]==0


def test_ncaaf_refresh_continues_without_odds_until_ready(tmp_path,monkeypatch):
    monkeypatch.setattr(s.ns,"records",lambda *a:[{"kind":"model","id":"m","data":{"runtime_hash":"h"}}])
    monkeypatch.setattr(s.ncaaf,"runtime_hash",lambda:"h")
    monkeypatch.setattr(s.ncaaf,"refresh",lambda *a:({"years":[2026],"batches":[]},"continue"))
    monkeypatch.setattr(s.ncaaf,"capture",lambda *a,**kw:pytest.fail("No odds before inputs ready"))
    monkeypatch.setattr(s.ncaaf,"grade",lambda *a,**kw:{"graded":0,"error":None})
    state={};r=s.run_ncaaf(tmp_path/"x",state,"cfbd","odds",lambda:None)
    assert r["input_status"]=="continue" and "ncaaf_inputs" in state


def test_ncaaf_filters_seen_and_limits_odds_to_due_games(tmp_path,monkeypatch):
    import requests
    now=datetime.now(timezone.utc)
    games=[{"id":i,"startDate":(now+timedelta(hours=1 if i<4 else 3)).isoformat()} for i in (1,2,3,4)]
    inputs={"years":[now.year],"batches":[{"retrieved_at":now.isoformat(),"records":games}]}
    records=[{"kind":"model","id":"m","data":{"runtime_hash":"h"}},
        {"kind":"capture","data":{"model_id":"m","events":[{"cfbd_id":1}]}}]
    monkeypatch.setattr(s.ns,"records",lambda *a:records)
    monkeypatch.setattr(s.ncaaf,"runtime_hash",lambda:"h")
    monkeypatch.setattr(s.ncaaf,"refresh",lambda *a:(inputs,"ready"))
    monkeypatch.setattr(s.ncaaf,"build_dataset",lambda *a,**kw:({},[{}],[]))
    monkeypatch.setattr(s.ncaaf,"_eligible",lambda f:True)
    monkeypatch.setattr(s.ncaaf,"_match",lambda e,g:next(x for x in g if x["id"]==e["id"]))
    class Response:
        status_code=200
        def json(self):return [{"id":i} for i in (1,2,3,4)]
    monkeypatch.setattr(requests,"get",lambda *a,**kw:Response())
    def capture(*a,**kw):
        assert kw["get"]("https://example.invalid").json()==[{"id":2},{"id":3}]
        return "capture",{"saved_games":2,"skipped_games":0}
    monkeypatch.setattr(s.ncaaf,"capture",capture)
    monkeypatch.setattr(s.ncaaf,"grade",lambda *a,**kw:{"graded":0,"error":None})
    assert s.run_ncaaf(tmp_path/"x",{},"c","o",lambda:None)["captured"]==2
    monkeypatch.setattr(s.ncaaf,"_eligible",lambda f:False)
    result=s.run_ncaaf(tmp_path/"x",{},"c","o",lambda:None)
    assert result["captured"]==0 and result["insufficient_features"]==2
    monkeypatch.setattr(s.ncaaf,"_eligible",lambda f:True)
    def failed_capture(*a,**kw):raise RuntimeError("provider secret")
    monkeypatch.setattr(s.ncaaf,"capture",failed_capture)
    monkeypatch.setattr(s.ncaaf,"grade",lambda *a,**kw:{"graded":3,"error":None})
    result=s.run_ncaaf(tmp_path/"x",{},"c","o",lambda:None)
    assert result["graded"]==3 and result["errors"]==["ncaaf_capture_failed"]


def test_mlb_missing_probables_are_not_failure(tmp_path,monkeypatch):
    now=datetime.now(timezone.utc)
    records=[{"kind":"model","id":"m","data":{"runtime_hash":"h"}}]
    monkeypatch.setattr(s.ms,"records",lambda *a:records)
    monkeypatch.setattr(s.mlb,"runtime_hash",lambda:"h")
    monkeypatch.setattr(s.mlb,"upcoming",lambda:[{"gamePk":i,"gameDate":(now+timedelta(hours=1)).isoformat()} for i in range(10)])
    calls=[]
    def capture(gid,path):
        calls.append(gid)
        raise ValueError("Both probable pitchers required")
    monkeypatch.setattr(s.mlb,"capture",capture)
    state={}
    r=s.run_mlb(tmp_path/"x",state,lambda:None)
    assert len(calls)==6 and not r["errors"]
    calls.clear();s.run_mlb(tmp_path/"x",state,lambda:None)
    assert calls[:4]==[6,7,8,9]
