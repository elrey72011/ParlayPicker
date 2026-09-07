"""Prospective paper evaluation: immutable pregame data, fixed policy, no stakes."""
from copy import deepcopy
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
from pathlib import Path
import requests
from app_core.ncaaf_history import _clean, build_dataset, timestamp, integer
from app_core.ncaaf_research import centers, probabilities, digest, _eligible
from app_core import ncaaf_prospective_store as store


def utcnow():
    return datetime.now(timezone.utc)


def runtime_hash():
    root = Path(__file__).resolve().parents[1]
    names = ("app_core/ncaaf_prospective.py", "app_core/ncaaf_research.py", "app_core/ncaaf_history.py", "core/team_mapper.py")
    return digest({n: hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names})


def freeze(result, path=None):
    artifact = result["artifact"]
    if digest(artifact) != result["report"]["artifact_hash"] or artifact["protocol"]["production_eligible"] is not False:
        raise ValueError("Research artifact integrity failure")
    data = {"artifact": artifact, "artifact_hash": digest(artifact), "runtime_hash": runtime_hash(),
            "policy": "First eligible capture per game and frozen model; highest EV quote, paper unit only if EV > 0; no live stake."}
    for r in store.records(path):
        if r["kind"] == "model" and r["data"] == data:
            return r["id"]
    return store.save("model", data, path)


def fetch(url, token, params, *, cfbd=True, get=None):
    token = str(token or "").strip()
    if cfbd and token.lower().startswith("bearer "):
        token = token[7:].strip()
    if not token or any(c.isspace() for c in token):
        raise ValueError("Missing or invalid API key")
    params = dict(params)
    headers = {"Authorization": "Bearer " + token} if cfbd else {}
    if not cfbd:
        params["apiKey"] = token
    try:
        response = (get or requests.get)(url, params=params, headers=headers, timeout=6, allow_redirects=False)
    except requests.RequestException:
        raise ValueError("Provider connection failed") from None
    if response.status_code != 200:
        raise ValueError("Provider HTTP " + str(int(response.status_code)))
    try:
        value = response.json()
        if not isinstance(value, list) or any(not isinstance(r, dict) for r in value):
            raise ValueError()
        json.dumps(value, allow_nan=False)
        return value
    except (ValueError, TypeError):
        raise ValueError("Invalid provider response") from None


def pending(state):
    if not state["batches"]:
        return [{"kind": "games", "year": state["years"][0]}]
    queue = []
    cutoff = utcnow() - timedelta(days=7)
    for g in state["batches"][0]["records"]:
        start = timestamp(g.get("startDate"))
        if (g.get("completed") is True and start and start < cutoff
            and g.get("season") == state["years"][0] and integer(g.get("week"))
            and 0 <= g["week"] <= 30 and g.get("seasonType") in ("regular", "postseason")):
            req = {"kind": "stats", "year": g["season"], "week": g["week"], "season_type": g["seasonType"]}
            if req not in queue:
                queue.append(req)
    done = [b["request"] for b in state["batches"]]
    return [r for r in queue if r not in done]


def refresh(state, token, *, get=None):
    state = deepcopy(state) if state else {"schema": 1, "years": [utcnow().year], "batches": []}
    for _ in range(6):
        queue = pending(state)
        if not queue:
            break
        req = queue[0]
        params = {"year": req["year"], "seasonType": "both", "classification": "fbs"}
        endpoint = "games"
        if req["kind"] == "stats":
            endpoint = "games/teams"
            params = {"year": req["year"], "week": req["week"], "seasonType": req["season_type"]}
        try:
            value = _clean(req["kind"], fetch("https://api.collegefootballdata.com/"+endpoint, token, params, get=get))
        except ValueError as exc:
            return state, str(exc)
        state["batches"].append({"request": req, "retrieved_at": utcnow().isoformat(), "records": value})
    return state, "ready" if not pending(state) else "continue"


def _match(event, games):
    from core.team_mapper import normalize_team_name
    def name(x):
        return normalize_team_name(str(x)).casefold()
    start = timestamp(event.get("commence_time"))
    if not start:
        return None
    matches = [g for g in games if name(g.get("homeTeam")) == name(event.get("home_team"))
               and name(g.get("awayTeam")) == name(event.get("away_team"))
               and timestamp(g.get("startDate")) and abs((timestamp(g["startDate"])-start).total_seconds()) <= 60
               and g.get("startTimeTBD") is False and g.get("completed") is False
               and integer(g.get("homeId")) and integer(g.get("awayId")) and integer(g.get("id"))
               and g["homeId"] != g["awayId"]]
    return matches[0] if len(matches) == 1 else None


def quote_candidates(event, model, feature, captured):
    from app_core.prediction_evidence import provider_quotes
    try:
        quotes = json.loads(provider_quotes(event))
    except (TypeError, KeyError, AttributeError, ValueError):
        return []
    result = []
    margin = float(centers(model["margin"], [feature], "margin")[0])
    total = float(centers(model["total"], [feature], "total")[0])
    for q in quotes:
        t = timestamp(q.get("recorded_at"))
        price, line, kind = q.get("price"), q.get("point"), q["market_type"]
        if not t or not 0 <= (captured-t).total_seconds() <= 900:
            continue
        if not isinstance(price, (int, float)) or isinstance(price, bool) or not math.isfinite(price) or abs(price) < 100:
            continue
        if not kind.startswith("moneyline") and (not isinstance(line, (int, float)) or isinstance(line, bool) or not math.isfinite(line)):
            continue
        decimal = 1 + (price/100 if price > 0 else 100/abs(price))
        if kind.startswith("total"):
            p = probabilities(total, model["total"]["sigma"], float(line), total=True)
            win = p["over"] if kind.endswith("over") else p["under"]
        else:
            threshold = 0.0 if kind.startswith("moneyline") else (-float(line) if kind.endswith("home") else float(line))
            p = probabilities(margin, model["margin"]["sigma"], threshold)
            win = p["over"] if kind.endswith("home") else p["under"]
        loss = max(0., 1-win-p["push"])
        result.append({**q, "decimal_odds": decimal, "win": win, "push": p["push"], "loss": loss,
                       "ev": win*(decimal-1)-loss, "implied_probability": 1/decimal, "live_stake": 0})
    opposite = {"moneyline_home": "moneyline_away", "moneyline_away": "moneyline_home",
                "spread_home": "spread_away", "spread_away": "spread_home",
                "total_over": "total_under", "total_under": "total_over"}
    for q in result:
        matches = [other for other in result if other["book"] == q["book"] and other["market_type"] == opposite[q["market_type"]]
                   and (q["market_type"].startswith("moneyline") or other["point"] == (-q["point"] if q["market_type"].startswith("spread") else q["point"]))]
        q["no_vig_probability"] = q["implied_probability"]/(q["implied_probability"]+matches[0]["implied_probability"]) if len(matches)==1 else None
    return sorted(result, key=lambda q: (-q["ev"], digest(q)))


def capture(state, model_record, odds_key, *, get=None, path=None):
    if digest(model_record["data"]["artifact"]) != model_record["data"]["artifact_hash"]:
        raise ValueError("Frozen model integrity failure")
    if model_record["data"]["runtime_hash"] != runtime_hash():
        raise ValueError("Implementation changed; freeze a new research cohort")
    if not state or not state["batches"] or pending(state):
        raise ValueError("Complete current-season inputs first")
    before = utcnow()
    if timestamp(model_record["created_at"]) > before:
        raise ValueError("Model freeze time is invalid")
    if any(not timestamp(b["retrieved_at"]) or not 0 <= (before-timestamp(b["retrieved_at"])).total_seconds() <= 86400 for b in state["batches"]):
        raise ValueError("Refresh current-season inputs; maximum age is 24 hours")
    events = fetch("https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds", odds_key,
                   {"regions": "us", "markets": "h2h,spreads,totals", "oddsFormat": "american"}, cfbd=False, get=get)
    captured = utcnow()
    saved, skipped = [], []
    games = state["batches"][0]["records"]
    for event in events:
        g = _match(event, games)
        start = timestamp(event.get("commence_time"))
        if not g or not start or start <= captured or start > captured+timedelta(days=7):
            skipped.append({"event_id": event.get("id"), "reason": "unmatched_or_not_upcoming"})
            continue
        _, features, _ = build_dataset(state, feature_targets=[g])
        feature = features[0]
        if not _eligible(feature):
            skipped.append({"event_id": event.get("id"), "reason": "insufficient_prior_features"})
            continue
        models = {}
        for name, model in model_record["data"]["artifact"]["models"].items():
            candidates = quote_candidates(event, model, feature, captured)
            if candidates:
                models[name] = {"candidates": candidates, "selected": candidates[0], "paper_unit": int(candidates[0]["ev"] > 0)}
        if not models:
            skipped.append({"event_id": event.get("id"), "reason": "no_fresh_exact_quotes"})
            continue
        saved.append({"event_id": event["id"], "cfbd_id": g["id"], "season": g["season"],
                      "home_id": g["homeId"], "away_id": g["awayId"], "home": g["homeTeam"], "away": g["awayTeam"],
                      "start": start.isoformat(), "features": feature, "models": models})
    # Final pregame check after computation; no in-play rows can enter the ledger.
    finished = utcnow()
    saved = [e for e in saved if timestamp(e["start"]) > finished and all(
        0 <= (finished-timestamp(m["selected"]["recorded_at"])).total_seconds() <= 900 for m in e["models"].values())]
    data = {"model_id": model_record["id"], "captured_at": finished.isoformat(), "inputs": state,
            "events": saved, "skipped": skipped, "production_eligible": False}
    return store.save("capture", data, path), {"saved_games": len(saved), "skipped_games": len(skipped)}


def grade(token, *, path=None, get=None):
    records = store.records(path)
    done = {s["cfbd_id"] for r in records if r["kind"] == "scores" for s in r["data"]["scores"]}
    games = {}
    for r in records:
        if r["kind"] == "capture":
            for e in r["data"]["events"]:
                if e["cfbd_id"] not in done and timestamp(e["start"]) < utcnow():
                    games[e["cfbd_id"]] = e
    attempted_at = {gid: r["created_at"] for r in records if r["kind"] == "scores" for gid in r["data"].get("attempted_ids", [])}
    scores, error, attempted = [], None, []
    for gid, e in sorted(games.items(), key=lambda item: (attempted_at.get(item[0], ""), item[0]))[:6]:
        attempted.append(gid)
        try:
            rows = fetch("https://api.collegefootballdata.com/games", token, {"id": gid}, get=get)
        except ValueError as exc:
            error = str(exc)
            break
        matches = [g for g in rows if g.get("id") == gid and g.get("homeId") == e["home_id"] and g.get("awayId") == e["away_id"]
                   and g.get("completed") is True and all(integer(g.get(k)) and g[k] >= 0 for k in ("homePoints", "awayPoints"))]
        if len(matches) == 1:
            g = matches[0]
            scores.append({"cfbd_id": gid, "home_id": e["home_id"], "away_id": e["away_id"], "home_score": g["homePoints"], "away_score": g["awayPoints"]})
    if attempted:
        store.save("scores", {"scores": scores, "attempted_ids": attempted}, path)
    return {"graded": len(scores), "pending_before_request": len(games), "error": error}


def report(path=None):
    records = store.records(path)
    scores = {s["cfbd_id"]: s for r in records if r["kind"] == "scores" for s in r["data"]["scores"]}
    frozen = {r["id"]: r for r in records if r["kind"] == "model"}
    first, results = {}, []
    for r in records:
        if r["kind"] != "capture":
            continue
        cohort = frozen.get(r["data"]["model_id"])
        if not cohort or timestamp(cohort["created_at"]) > timestamp(r["data"]["captured_at"]):
            continue
        for e in r["data"]["events"]:
            key = (r["data"]["model_id"], e["cfbd_id"])
            if timestamp(r["data"]["captured_at"]) >= timestamp(e["start"]):
                continue
            first.setdefault(key, (r, e))
    for (model_id, gid), (r, e) in first.items():
        s = scores.get(gid)
        if not s or (s["home_id"],s["away_id"]) != (e["home_id"],e["away_id"]):
            continue
        margin, total = s["home_score"]-s["away_score"], s["home_score"]+s["away_score"]
        for name, m in e["models"].items():
            q = m["selected"]
            kind, line = q["market_type"], q["point"]
            if kind.startswith("total"):
                value = total-line if kind.endswith("over") else line-total
            elif kind.startswith("spread"):
                value = (margin if kind.endswith("home") else -margin)+line
            else:
                value = margin if kind.endswith("home") else -margin
            outcome = "win" if value > 0 else "loss" if value < 0 else "push"
            profit = (q["decimal_odds"]-1 if value>0 else -1 if value<0 else 0)*m["paper_unit"]
            results.append({"model_id": model_id, "model": name, "game_id": gid, "market": kind,
                            "outcome": outcome, "paper_unit": m["paper_unit"], "paper_profit": profit,
                            "win_probability": q["win"], "log_loss": -math.log(max(q[outcome], 1e-12)),
                            "brier_three_way": sum((q[k] - int(outcome == k))**2 for k in ("win", "loss", "push"))})
    summary = []
    for key in sorted({(r["model_id"],r["model"]) for r in results}):
        rows = [r for r in results if (r["model_id"],r["model"]) == key]
        bets = [r for r in rows if r["paper_unit"]]
        decisions = [r for r in bets if r["outcome"] != "push"]
        summary.append({"model_id": key[0], "model": key[1], "graded_games": len(rows), "paper_bets": len(bets),
                        "paper_hit_rate": sum(r["outcome"]=="win" for r in decisions)/len(decisions) if decisions else None,
                        "paper_profit_units": sum(r["paper_profit"] for r in bets),
                        "paper_roi": sum(r["paper_profit"] for r in bets)/len(bets) if bets else None,
                        "brier_three_way": sum(r["brier_three_way"] for r in rows)/len(rows),
                        "log_loss": sum(r["log_loss"] for r in rows)/len(rows),
                        "calibration": [{"bin": i, "n": len(bucket), "mean_win_probability": sum(r["win_probability"] for r in bucket)/len(bucket),
                                         "observed_win_rate": sum(r["outcome"]=="win" for r in bucket)/len(bucket)}
                                        for i in range(10) if (bucket := [r for r in rows if min(9,int(r["win_probability"]*10)) == i])]})
    return {"captured_games_by_cohort": len(first), "graded_selections": len(results), "summary": summary, "results": results,
            "limitations": ["Paper evaluation only; quotes are observed offers, not confirmed executions.",
                            "No closing-line value or automatic schedule is implemented.",
                            "Models have separate frozen cohorts; repeated captures do not replace the first selection."]}
