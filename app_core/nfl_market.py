"""NFL market snapshots and score comparisons, not model predictions or wagers."""
from collections import Counter
from datetime import datetime, timezone, timedelta
import math
from app_core import nfl_market_store as store
from app_core.research_api_budget import BudgetLimit
from app_core.ncaaf_history import timestamp

BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/"
PROTOCOL = "nfl-market-v1"


def utcnow():
    return datetime.now(timezone.utc)


def identity(event):
    if not isinstance(event, dict) or event.get("sport_key") != "americanfootball_nfl":
        raise ValueError("nfl_event_identity")
    fields = [event.get(k) for k in ("id", "home_team", "away_team")]
    if any(not isinstance(v, str) or not v.strip() for v in fields) or fields[1] == fields[2]:
        raise ValueError("nfl_event_identity")
    start = timestamp(event.get("commence_time"))
    if start is None:
        raise ValueError("nfl_event_time")
    return {"event_id": fields[0], "home": fields[1], "away": fields[2], "start": start.isoformat()}


def fetch(endpoint, key, request_get, **params):
    response = request_get(BASE + endpoint, params={"apiKey": key, "dateFormat": "iso", **params},
                           timeout=15, allow_redirects=False)
    if response.status_code != 200:
        raise ValueError("nfl_provider_response")
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError("nfl_provider_schema")
    events = {}
    for event in payload:
        ident = identity(event)
        gid = ident["event_id"]
        if gid in events and events[gid] != event:
            raise ValueError("nfl_duplicate_event")
        events[gid] = event
    return events


def number(value):
    if isinstance(value, bool):
        raise ValueError("nfl_invalid_number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("nfl_invalid_number")
    return result


def quotes(event, observed):
    """Accept complete, same-book pairs with fresh exact-market timestamps."""
    accepted = []
    rejected = Counter()
    books_seen = set()
    for book in event.get("bookmakers", []):
        book_key = book.get("key")
        if not isinstance(book_key, str) or not book_key or book_key in books_seen:
            raise ValueError("nfl_duplicate_book")
        books_seen.add(book_key)
        markets_seen = set()
        for market in book.get("markets", []):
            kind = market.get("key")
            if kind not in ("h2h", "spreads", "totals"):
                continue
            if kind in markets_seen:
                raise ValueError("nfl_duplicate_market")
            markets_seen.add(kind)
            at = timestamp(market.get("last_update"))
            if at is None or not 0 <= (observed - at).total_seconds() <= 900:
                rejected["missing_or_stale_market_timestamp"] += 1
                continue
            try:
                outcomes = market["outcomes"]
                names = [o["name"] for o in outcomes]
                expected = {"Over", "Under"} if kind == "totals" else {event["home_team"], event["away_team"]}
                if len(outcomes) != 2 or set(names) != expected:
                    raise ValueError("incomplete_pair")
                prices = [number(o["price"]) for o in outcomes]
                if any(abs(p) < 100 or p != int(p) for p in prices):
                    raise ValueError("invalid_american_odds")
                points = [None, None] if kind == "h2h" else [number(o["point"]) for o in outcomes]
                if kind != "h2h" and any(p * 2 != int(p * 2) for p in points):
                    raise ValueError("invalid_line")
                if kind == "spreads" and abs(sum(points)) > 1e-9:
                    raise ValueError("unpaired_spread")
                if kind == "totals" and (points[0] != points[1] or points[0] <= 0):
                    raise ValueError("unpaired_total")
                accepted.extend({"book": book_key, "market": kind, "selection": name,
                                 "point": point, "odds_american": price, "recorded_at": at.isoformat()}
                                for name, point, price in zip(names, points, prices))
            except (KeyError, TypeError, ValueError, OverflowError):
                rejected["invalid_market_pair"] += 1
    return accepted, rejected


def captures(records):
    # First saved entry per event; repeated manual/remote records cannot inflate counts.
    result = {}
    for record in records:
        if record["kind"] == "capture" and record["data"].get("protocol") == PROTOCOL:
            for event in record["data"]["events"]:
                result.setdefault(event["event_id"], {**event, "captured_at": record["created_at"]})
    return result


def final_score(event, captured, observed):
    ident = identity(event)
    if any(ident[k] != captured[k] for k in ("event_id", "home", "away", "start")):
        raise ValueError("nfl_score_identity_or_schedule_changed")
    if event.get("completed") is not True:
        return None
    if not timestamp(captured["captured_at"]) < timestamp(ident["start"]) <= observed:
        raise ValueError("nfl_score_timing")
    updated = timestamp(event.get("last_update"))
    if updated is None or not timestamp(ident["start"]) <= updated <= observed:
        raise ValueError("nfl_score_timestamp")
    values = event.get("scores")
    if not isinstance(values, list) or len(values) != 2 or {s.get("name") for s in values} != {ident["home"], ident["away"]}:
        raise ValueError("nfl_score_pair")
    scores = {s["name"]: number(s["score"]) for s in values}
    if any(v < 0 or v != int(v) for v in scores.values()):
        raise ValueError("nfl_invalid_score")
    return {**ident, "home_score": int(scores[ident["home"]]), "away_score": int(scores[ident["away"]]),
            "provider_updated_at": updated.isoformat()}


def comparison(quote, game, score):
    home, away = score["home_score"], score["away_score"]
    if quote["market"] == "totals":
        delta = home + away - quote["point"]
        if quote["selection"] == "Under":
            delta = -delta
    else:
        delta = home - away if quote["selection"] == game["home"] else away - home
        if quote["market"] == "spreads":
            delta += quote["point"]
    if delta == 0:
        return "tie" if quote["market"] == "h2h" else "push"
    return "win" if delta > 0 else "loss"


def report(path=None, *, now=None):
    now = now or utcnow()
    records = store.records(path)
    games = captures(records)
    scores = {r["data"]["event_id"]: r["data"] for r in records if r["kind"] == "scores"}
    rows = []
    unresolved = []
    for gid, game in games.items():
        score = scores.get(gid)
        if score is None and now - timestamp(game["start"]) > timedelta(days=3):
            unresolved.append(gid)
        for quote in game["quotes"]:
            rows.append({"event_id": gid, "matchup": game["away"] + " at " + game["home"],
                         "captured_at": game["captured_at"], **quote,
                         "result": comparison(quote, game, score) if score else "pending"})
    return {"protocol": PROTOCOL, "mode": "market_tracking_only", "production_eligible": False,
            "captured_games": len(games), "graded_games": len(set(games) & set(scores)),
            "quote_rows": len(rows), "unresolved_past_score_window": unresolved,
            "notes": ["All quotes are observations, not selected bets or independent predictions.",
                      "Win/loss/push compares the quoted line with reported final scores; bookmaker settlement rules are not verified.",
                      "Pregame timing uses the provider's scheduled start, not verified actual kickoff.",
                      "No ROI, model accuracy, win probabilities or closing-line value is claimed."],
            "quotes": rows}


def run(path, odds_key, backup, request_get):
    if not odds_key:
        raise ValueError("missing_provider_keys")
    result = {"mode": "market_tracking_only", "captured": 0, "graded": 0, "errors": [],
              "excluded_markets": {}, "budget_paused": False}
    records = store.records(path)
    seen = captures(records)
    completed = {r["data"]["event_id"] for r in records if r["kind"] == "scores"}
    now = utcnow()
    pending = {gid: e for gid, e in seen.items() if gid not in completed and
               timedelta(hours=3) <= now - timestamp(e["start"]) <= timedelta(days=3)}
    # Batch final-score lookup first so new captures cannot starve grading.
    if pending:
        try:
            response = fetch("scores", odds_key, request_get, daysFrom=3, eventIds=",".join(sorted(pending)))
            for gid, event in response.items():
                if gid not in pending:
                    continue
                try:
                    score = final_score(event, pending[gid], utcnow())
                    if score:
                        store.save("scores", {"sport": "NFL", **score}, path)
                        result["graded"] += 1
                except (ValueError, TypeError, KeyError, OverflowError) as exc:
                    allowed={"nfl_score_identity_or_schedule_changed","nfl_score_timing","nfl_score_timestamp","nfl_score_pair","nfl_invalid_score","nfl_invalid_number"}
                    reason=str(exc) if str(exc) in allowed else "nfl_score_validation"
                    result["errors"].append(reason)
                    result.setdefault("score_rejections",[]).append({"event_id":gid,"reason":reason,
                        "captured_start":pending[gid].get("start"),"reported_start":event.get("commence_time"),
                        "captured_at":pending[gid].get("captured_at"),"reported_update":event.get("last_update")})
            backup()
        except BudgetLimit:
            result["budget_paused"] = True
        except Exception:
            result["errors"].append("nfl_scores_failed")
    try:
        events = fetch("events", odds_key, request_get)
        now = utcnow()
        due = {gid: e for gid, e in events.items() if gid not in seen and
               now < timestamp(e["commence_time"]) <= now + timedelta(hours=2)}
        result["upcoming_games"] = len(events)
        result["due_games"] = len(due)
        if due:
            payload = fetch("odds", odds_key, request_get, regions="us", markets="h2h,spreads,totals",
                            oddsFormat="american", eventIds=",".join(sorted(due)))
            captured = []
            excluded = Counter()
            for gid, event in payload.items():
                if gid not in due:
                    continue
                ident = identity(event)
                observed = utcnow()
                if ident != identity(due[gid]) or not observed < timestamp(ident["start"]) <= observed + timedelta(hours=2):
                    excluded["identity_or_pregame_window"] += 1
                    continue
                valid, rejected = quotes(event, observed)
                excluded.update(rejected)
                if valid:
                    captured.append({**ident, "quotes": valid, "response_received_at": observed.isoformat()})
            if captured:
                key = store.save("capture", {"sport": "NFL", "protocol": PROTOCOL, "events": captured}, path)
                saved = next(r for r in store.records(path) if r["id"] == key)
                result["captured"] = len(saved["data"]["events"])
                backup()
            result["excluded_markets"] = dict(excluded)
    except BudgetLimit:
        result["budget_paused"] = True
    except Exception:
        result["errors"].append("nfl_capture_failed")
    result["unresolved_past_score_window"] = len(report(path)["unresolved_past_score_window"])
    return result
