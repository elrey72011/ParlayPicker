"""Frozen paired score forecasts with observed pregame probable pitchers."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import requests
from app_core import mlb_prospective_store as store
from app_core.mlb_history import build_dataset, timestamp, normalize_game
from app_core.mlb_pitcher_history import enrich, PITCHER_FEATURES, parse_boxscore
from app_core.mlb_research import FEATURES


def now():
    return datetime.now(timezone.utc)


def runtime_hash():
    return hashlib.sha256(b"".join(Path(__file__).with_name(n).read_bytes() for n in
        ("mlb_prospective.py", "mlb_pitcher_history.py", "mlb_history.py", "mlb_research.py"))).hexdigest()


def freeze(history_raw, pitcher_raw, path=None):
    history, pitchers = json.loads(history_raw), json.loads(pitcher_raw)
    if pitchers["source_hash"] != hashlib.sha256(history_raw).hexdigest():
        raise ValueError("History checkpoint mismatch")
    scheduled={str(g["gamePk"]) for d in history["schedules"]["2023"]["payload"]["dates"] for g in d["games"]}
    if not scheduled or scheduled-history["games"].keys()-history["excluded"].keys():
        raise ValueError("Complete 2023 history first")
    games = {int(k):v["record"] for k,v in history["games"].items() if v["record"]["season"] == 2023}
    if set(map(str,games)) - pitchers["boxes"].keys() - pitchers["excluded"].keys():
        raise ValueError("Complete 2023 pitcher collection first")
    boxes = {k:v["data"] for k,v in pitchers["boxes"].items() if int(k) in games}
    rows, _ = enrich(build_dataset(list(games.values()))["features"], games, boxes)
    if len(rows) < 100:
        raise ValueError("Insufficient matched training history")
    artifact = {"runtime_hash":runtime_hash(), "train_year":2023, "ridge_alpha":10,
                "production_eligible":False, "history_hash":hashlib.sha256(history_raw).hexdigest(),
                "pitcher_hash":hashlib.sha256(pitcher_raw).hexdigest(), "train_rows":len(rows), "models":{}}
    for name, columns in (("team_only",FEATURES),("team_and_starter",FEATURES+PITCHER_FEATURES)):
        x=np.array([[r[k] for k in columns] for r in rows]); mean,scale=x.mean(0),x.std(0)
        scale[scale==0]=1; x=(x-mean)/scale
        model={"columns":columns,"mean":mean.tolist(),"scale":scale.tolist(),"targets":{}}
        for target in ("margin","total"):
            y=np.array([games[r["game_id"]]["home_score"]+(1 if target=="total" else -1)*games[r["game_id"]]["away_score"] for r in rows])
            coef=np.linalg.solve(x.T@x+10*np.eye(len(columns)),x.T@(y-y.mean()))
            model["targets"][target]={"intercept":float(y.mean()),"coefficients":coef.tolist()}
        artifact["models"][name]=model
    for r in store.records(path):
        if r["kind"]=="model" and r["data"]==artifact:return r["id"]
    return store.save("model",artifact,path)


def fetch(endpoint, params=None):
    response=requests.get("https://statsapi.mlb.com/"+endpoint,params=params,timeout=30)
    response.raise_for_status()
    return response.json()


def schedule():
    return fetch("api/v1/schedule",{"sportId":1,"season":now().year,"gameType":"R","hydrate":"probablePitcher"})


def games_from(payload):
    return [g for d in payload["dates"] for g in d["games"]]


def upcoming():
    return [g for g in games_from(schedule()) if g["status"]["abstractGameState"]=="Preview"
            and timestamp(g["gameDate"])>now()]


def live_features(game, final_games, logs):
    row={}
    for side in ("home","away"):
        team=game["teams"][side]["team"]["id"]
        prior=[g for g in final_games if team in (g["teams"]["home"]["team"]["id"],g["teams"]["away"]["team"]["id"])]
        if len(prior)<10:raise ValueError("Insufficient team history")
        scored=[];allowed=[]
        for g in prior:
            own="home" if g["teams"]["home"]["team"]["id"]==team else "away"
            other="away" if own=="home" else "home"
            a,b=g["teams"][own]["score"],g["teams"][other]["score"]
            if not isinstance(a,int) or not isinstance(b,int) or a==b:raise ValueError("Invalid final score")
            scored.append(a);allowed.append(b)
        row[side+"_ppg"]=sum(scored)/len(prior)
        row[side+"_oppg"]=sum(allowed)/len(prior)
        row[side+"_win_pct"]=sum(a>b for a,b in zip(scored,allowed))/len(prior)
        row[side+"_source_game_ids"]=[g["gamePk"] for g in prior]
        final_ids={g["gamePk"] for g in final_games}
        splits=[s for stats in logs[side].get("stats",[]) for s in stats.get("splits",[]) if s["game"]["gamePk"] in final_ids]
        if len({s["game"]["gamePk"] for s in splits})!=len(splits):raise ValueError("Duplicate pitching appearances")
        totals={k:0 for k in ("outs","earnedRuns","strikeOuts","baseOnBalls","hits")}
        for s in splits:
            stat=s["stat"]
            # Game-log innings are baseball outs notation, never decimal innings.
            whole,_,part=str(stat["inningsPitched"]).partition(".")
            if part not in ("","0","1","2") or not whole.isdigit():raise ValueError("Invalid innings")
            values={"outs":int(whole)*3+int(part or 0),**{k:stat[k] for k in totals if k!="outs"}}
            if any(isinstance(v,bool) or not isinstance(v,int) or v<0 for v in values.values()):raise ValueError("Invalid pitching stats")
            for k,v in values.items():totals[k]+=v
        if len(splits)<3 or totals["outs"]<27:raise ValueError("Insufficient pitcher history")
        prefix=side+"_starter_"
        for key,stat in (("era","earnedRuns"),("k9","strikeOuts"),("bb9","baseOnBalls")):
            row[prefix+key]=totals[stat]*27/totals["outs"]
        row[prefix+"whip"]=(totals["hits"]+totals["baseOnBalls"])*3/totals["outs"]
        row[prefix+"source_game_ids"]=[s["game"]["gamePk"] for s in splits]
    return row


def capture(game_id, path=None):
    models=[r for r in store.records(path) if r["kind"]=="model"]
    if not models:raise ValueError("Freeze or restore models first")
    model=models[-1]
    if model["data"]["runtime_hash"]!=runtime_hash():raise ValueError("Model runtime changed; freeze again")
    started=now();payload=schedule();all_games=games_from(payload)
    matches=[g for g in all_games if g["gamePk"]==int(game_id)]
    if len(matches)!=1:raise ValueError("Game identity missing or ambiguous")
    game=matches[0];start=timestamp(game["gameDate"])
    if game["status"]["abstractGameState"]!="Preview" or start<=now():raise ValueError("Game is not pregame")
    probable={s:game["teams"][s].get("probablePitcher",{}).get("id") for s in ("home","away")}
    if not all(probable.values()):raise ValueError("Both probable pitchers required")
    logs={s:fetch(f"api/v1/people/{pid}/stats",{"stats":"gameLog","group":"pitching","season":started.year}) for s,pid in probable.items()}
    # Only results already marked Final in the earlier observed schedule can contribute.
    finals=[g for g in all_games if g["gamePk"]!=int(game_id) and g["status"]["abstractGameState"]=="Final"
            and timestamp(g["gameDate"])<started and all("score" in g["teams"][s] for s in ("home","away"))]
    unique={}
    for g in finals:
        prior=unique.get(g["gamePk"])
        if prior and any((prior["teams"][s]["team"]["id"],prior["teams"][s]["score"]) !=
                         (g["teams"][s]["team"]["id"],g["teams"][s]["score"]) for s in ("home","away")):
            raise ValueError("Conflicting final game records")
        unique[g["gamePk"]]=g
    finals=list(unique.values())
    features=live_features(game,finals,logs)
    forecasts={}
    for name,m in model["data"]["models"].items():
        x=(np.array([features[k] for k in m["columns"]])-m["mean"])/m["scale"]
        forecasts[name]={t:float(x@np.array(v["coefficients"])+v["intercept"]) for t,v in m["targets"].items()}
        forecasts[name]["total"]=max(0,forecasts[name]["total"])
    finished=now()
    if finished>=start or (finished-started).total_seconds()>300:raise ValueError("Capture expired or game started")
    event={"game_id":int(game_id),"start":start.isoformat(),"probable_pitchers":probable,
           "home_id":game["teams"]["home"]["team"]["id"],"away_id":game["teams"]["away"]["team"]["id"],
           "features":features,"forecasts":forecasts}
    return store.save("capture",{"model_id":model["id"],"observed_at":finished.isoformat(),
        "request_started_at":started.isoformat(),"starter_status":"provider-listed probable, not confirmed",
        "events":[event],"schedule":payload,"pitcher_logs":logs,"production_eligible":False},path)


def grade(path=None):
    records=store.records(path)
    done={r["data"]["game_id"] for r in records if r["kind"]=="scores"}
    events={e["game_id"]:e for r in records if r["kind"]=="capture" for e in r["data"]["events"]}
    saved=0
    for gid,e in list((k,v) for k,v in events.items() if k not in done)[:6]:
        feed=fetch(f"api/v1.1/game/{gid}/feed/live")
        if feed["gameData"]["status"]["abstractGameState"]!="Final":continue
        game=normalize_game(feed)
        if game["game_id"]!=gid or any(game[s+"_id"]!=e[s+"_id"] for s in ("home","away")):raise ValueError("Score identity mismatch")
        box=fetch(f"api/v1/game/{gid}/boxscore")
        actual=parse_boxscore(box,game)["starters"]
        store.save("scores",{**game,"actual_starters":actual},path);saved+=1
    return saved


def report(path=None):
    records=store.records(path);scores={r["data"]["game_id"]:r["data"] for r in records if r["kind"]=="scores"}
    selected={}
    for r in records:
        if r["kind"]=="capture":
            for e in r["data"]["events"]:selected.setdefault((r["data"]["model_id"],e["game_id"]),(r,e))
    results=[]
    for (cohort,gid),(r,e) in selected.items():
        score=scores.get(gid)
        if not score:continue
        valid=timestamp(r["created_at"])<timestamp(score["started_at"])
        changed=e["probable_pitchers"]!=score["actual_starters"]
        results.append({"cohort":cohort,"game_id":gid,"pregame_verified":valid,"starter_changed":changed,
            "errors":{name:{t:abs(pred[t]-(score["home_score"]+(1 if t=="total" else -1)*score["away_score"])) for t in ("margin","total")} for name,pred in e["forecasts"].items()}})
    summary=[]
    for cohort in sorted({r["cohort"] for r in results}):
        rows=[r for r in results if r["cohort"]==cohort and r["pregame_verified"]]
        if rows:summary.append({"cohort":cohort,"games":len(rows),"starter_changes":sum(r["starter_changed"] for r in rows),
            "mae":{name:{t:sum(r["errors"][name][t] for r in rows)/len(rows) for t in ("margin","total")} for name in ("team_only","team_and_starter")}})
    return {"captured_games_by_cohort":len(selected),"graded_games_by_cohort":len(results),"summary":summary,"results":results,
        "production_eligible":False,"limitations":["Probable pitchers observed before start are not confirmed starters.",
        "First capture per game/cohort is retained. Starter changes remain in the primary comparison.",
        "No betting returns or win probabilities. Invalid capture timing is excluded from summary."]}
