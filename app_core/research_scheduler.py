"""Bounded orchestration around frozen research modules; never fits models."""
import hashlib
import json
from datetime import datetime, timezone, timedelta
from app_core import mlb_prospective as mlb, mlb_prospective_store as ms
from app_core import ncaaf_prospective as ncaaf, ncaaf_prospective_store as ns
from app_core import nfl_market as nfl, nfl_market_store as fs
from app_core.mlb_history import timestamp
from app_core.research_api_budget import Budget, BudgetLimit
from app_core.research_schedule import is_open

PREFIX = "parlaypicker/research-scheduler-v1/"
_last_checkpoint_time = None


def utcnow():
    return datetime.now(timezone.utc)


def due(start, now):
    return now < timestamp(start) <= now + timedelta(hours=2)


def latest_model(records):
    models=[r for r in records if r["kind"]=="model"]
    if not models:raise ValueError("missing_frozen_model")
    return models[-1]


def checkpoint(client, folder, state=None):
    global _last_checkpoint_time
    prefix=PREFIX+"state/"
    if state is None:
        keys=[o["Key"] for page in client.get_paginator("list_objects_v2").paginate(Bucket=folder,Prefix=prefix) for o in page.get("Contents",[])]
        if not keys:return {}
        key=max(keys)
        with client.get_object(Bucket=folder,Key=key)["Body"] as b:raw=b.read(40_000_001)
        if len(raw)>40_000_000 or not key.endswith(hashlib.sha256(raw).hexdigest()+".json"):
            raise ValueError("scheduler_state_integrity")
        written=datetime.strptime(key[len(prefix):].split("-")[0],"%Y%m%dT%H%M%S%f").replace(tzinfo=timezone.utc)
        _last_checkpoint_time=max(_last_checkpoint_time or written,written)
        return json.loads(raw)
    raw=json.dumps(state,sort_keys=True,allow_nan=False).encode()
    written=utcnow()
    if _last_checkpoint_time is not None:
        written=max(written,_last_checkpoint_time+timedelta(microseconds=1))
    _last_checkpoint_time=written
    key=prefix+written.strftime("%Y%m%dT%H%M%S%f")+"-"+hashlib.sha256(raw).hexdigest()+".json"
    client.put_object(Bucket=folder,Key=key,Body=raw,ContentType="application/json",IfNoneMatch="*")
    with client.get_object(Bucket=folder,Key=key)["Body"] as b:
        if b.read()!=raw:raise ValueError("scheduler_state_readback")


def retry_schedule(fetch):
    """Retry only the read-only MLB schedule, without changing frozen model code."""
    import requests
    import time
    for attempt in range(3):
        try:
            return fetch()
        except (requests.Timeout, requests.ConnectionError):
            if attempt==2:
                raise
            time.sleep(attempt+1)


def run_mlb(path, state, backup):
    records=ms.records(path);model=latest_model(records)
    if model["data"]["runtime_hash"]!=mlb.runtime_hash():raise ValueError("stale_frozen_model")
    seen={e["game_id"] for r in records if r["kind"]=="capture" and r["data"]["model_id"]==model["id"] for e in r["data"]["events"]}
    counts={"captured":0,"graded":0,"blocked_captures":0,"errors":[]}
    games=sorted(retry_schedule(mlb.upcoming),key=lambda g:(g["gameDate"],g["gamePk"]))
    candidates=[g for g in games if g["gamePk"] not in seen and due(g["gameDate"],utcnow())]
    capture_attempts=state.setdefault("mlb_capture_attempts",{})
    candidates.sort(key=lambda g:(capture_attempts.get(str(g["gamePk"]),""),g["gameDate"],g["gamePk"]))
    for g in candidates[:6]:
        if not is_open():break
        capture_attempts[str(g["gamePk"])]=utcnow().isoformat()
        try:
            mlb.capture(g["gamePk"],path)
            # Storage may remove a game that starts during computation.
            saved=ms.records(path)[-1]["data"].get("events",[])
            counts["captured"]+=len(saved)
        except ValueError as exc:
            counts["blocked_captures"]+=1
            expected=("Both probable pitchers required","Insufficient team history","Insufficient pitcher history",
                      "Game is not pregame","Capture expired or game started")
            if str(exc) not in expected:counts["errors"].append("mlb_capture_validation")
        except Exception:
            counts["errors"].append("mlb_capture_provider")
        backup()
    records=ms.records(path)
    completed={r["data"]["game_id"] for r in records if r["kind"]=="scores"}
    events={e["game_id"]:e for r in records if r["kind"]=="capture" for e in r["data"]["events"] if timestamp(e["start"])<utcnow() and e["game_id"] not in completed}
    attempts=state.setdefault("mlb_grade_attempts",{})
    for gid,e in sorted(events.items(),key=lambda item:(attempts.get(str(item[0]),""),item[0]))[:6]:
        if not is_open():break
        attempts[str(gid)]=utcnow().isoformat()
        try:
            feed=mlb.fetch(f"api/v1.1/game/{gid}/feed/live")
            if feed["gameData"]["status"]["abstractGameState"]!="Final":continue
            game=mlb.normalize_game(feed)
            if game["game_id"]!=gid or any(game[s+"_id"]!=e[s+"_id"] for s in ("home","away")):
                raise ValueError("score_identity")
            starters=mlb.parse_boxscore(mlb.fetch(f"api/v1/game/{gid}/boxscore"),game)["starters"]
            ms.save("scores",{**game,"actual_starters":starters},path);counts["graded"]+=1
            backup()
        except Exception:
            counts["errors"].append("mlb_grading_failed")
    return counts


def run_ncaaf(path, state, cfbd_key, odds_key, backup, request_get=None):
    records=ns.records(path);model=latest_model(records)
    if model["data"]["runtime_hash"]!=ncaaf.runtime_hash():raise ValueError("stale_frozen_model")
    if not cfbd_key or not odds_key:raise ValueError("missing_provider_keys")
    inputs=state.get("ncaaf_inputs")
    if inputs and (inputs["years"]!=[utcnow().year] or any(not 0 <= (utcnow()-timestamp(b["retrieved_at"])).total_seconds()<23*3600 for b in inputs["batches"])):
        inputs=None
    provider_args={"get":request_get} if request_get is not None else {}
    inputs,status=ncaaf.refresh(inputs,cfbd_key,**provider_args)
    state["ncaaf_inputs"]=inputs
    counts={"captured":0,"graded":0,"input_status":status,"errors":[]}
    if status not in ("ready","continue") and not status.startswith("api_budget:"):
        counts["errors"].append("ncaaf_refresh_failed")
    if status=="ready":
        seen={e["cfbd_id"] for r in records if r["kind"]=="capture" and r["data"]["model_id"]==model["id"] for e in r["data"]["events"]}
        targets=[g for g in inputs["batches"][0]["records"] if g.get("id") not in seen and g.get("startDate") and due(g["startDate"],utcnow())]
        eligible=[]
        for game in targets:
            _,features,_=ncaaf.build_dataset(inputs,feature_targets=[game])
            if features and ncaaf._eligible(features[0]):eligible.append(game)
        counts["insufficient_features"]=len(targets)-len(eligible)
        targets=eligible
        attempts=state.setdefault("ncaaf_capture_attempts",{})
        ordered=sorted(targets,key=lambda g:(attempts.get(str(g["id"]),""),g["startDate"],g["id"]))[:24]
        allowed={g["id"] for g in ordered}
        for gid in allowed:attempts[str(gid)]=utcnow().isoformat()
        if allowed:
            import requests
            def filtered_get(url,**kwargs):
                response=(request_get or requests.get)(url,**kwargs)
                if response.status_code!=200:return response
                values=response.json()
                if not isinstance(values,list):return response
                filtered=[]
                for event in values:
                    g=ncaaf._match(event,inputs["batches"][0]["records"])
                    if g and g["id"] in allowed:filtered.append(event)
                class Filtered:
                    status_code=200
                    def json(self):return filtered
                return Filtered()
            try:
                _,result=ncaaf.capture(inputs,model,odds_key,get=filtered_get,path=path)
                counts["captured"]=result["saved_games"]
                counts["excluded"]=result["skipped_games"]
            except BudgetLimit:
                counts["budget_paused"]=True
            except Exception:
                counts["errors"].append("ncaaf_capture_failed")
            backup()
    result=ncaaf.grade(cfbd_key,path=path,**provider_args)
    counts["graded"]=result["graded"]
    if result.get("error") and not result["error"].startswith("api_budget:"):counts["errors"].append("ncaaf_grading_failed")
    backup()
    return counts


def run(sports, root, client, folder, cfbd_key=None, odds_key=None):
    if not sports or any(s not in ("MLB", "NCAAF", "NFL") for s in sports) or len(set(sports)) != len(sports):
        raise ValueError("Invalid sports")
    root.mkdir(parents=True,exist_ok=True)
    state=checkpoint(client,folder)
    budget=Budget(state,lambda:checkpoint(client,folder,state))
    report={"started_at":utcnow().isoformat(),"sports":{},"errors":[],"production_eligible":False}
    for sport in sports:
        store={"MLB": ms, "NCAAF": ns, "NFL": fs}[sport]
        path=root/("nfl-market.sqlite3" if sport=="NFL" else sport.lower()+"-prospective.sqlite3")
        def backup():return store.sync(path,client=client,folder=folder)
        try:
            backup()  # Restore must succeed before any capture or grading.
            try:
                if sport=="NFL":
                    result=nfl.run(path,odds_key,backup,budget.request)
                else:
                    result=run_mlb(path,state,backup) if sport=="MLB" else run_ncaaf(path,state,cfbd_key,odds_key,backup,budget.request)
                report["sports"][sport]=result
                report["errors"].extend(result["errors"])
            finally:
                backup()
        except Exception as exc:
            # Never include provider exception text: URLs may contain API keys.
            known={"missing_frozen_model","stale_frozen_model","missing_provider_keys"}
            code=str(exc) if str(exc) in known else type(exc).__name__
            report["errors"].append(sport+":"+code)
    report["finished_at"]=utcnow().isoformat()
    report["api_budget"]=budget.report()
    state["last_run"]=report
    checkpoint(client,folder,state)
    return report
