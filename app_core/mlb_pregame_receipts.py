"""Live MLB observations and append-only research receipts. No wager authority.

Only live response boundaries assign observation times. Historical exports are
not an input. Each event/target retains its first accepted pregame receipt.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

import requests

from app_core.mlb_history import normalize_game, timestamp
from app_core.mlb_spread_total_model import SCHEMA, TARGETS, canonical, digest, identity, label, receipt_features
from app_core.prediction_evidence import database_path, provider_quotes
from app_core.public_quote_policy import canonical_book_label

BASE = "https://statsapi.mlb.com/"
BOOKS = ("Novig", "DraftKings", "FanDuel", "BetMGM")


def now():
    return datetime.now(timezone.utc)


class Rejected(ValueError):
    pass


def stable_id(value):
    if isinstance(value, bool) or value is None or not str(value).isdigit() or int(value) <= 0:
        raise Rejected("missing_provider_event_id")
    return str(int(value))


def team_id(value):
    try:
        return "mlb:" + stable_id(value)
    except Rejected:
        raise Rejected("missing_team_ids") from None


def connect(path=None):
    path = Path(path or database_path().with_name("mlb-pregame-receipts.sqlite3"))
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=5)
    for table in ("observations", "receipts", "outcomes"):
        db.execute(f"CREATE TABLE IF NOT EXISTS {table} (id TEXT PRIMARY KEY, sha256 TEXT NOT NULL, payload TEXT NOT NULL)")
        db.execute(f"CREATE TRIGGER IF NOT EXISTS {table}_replace BEFORE INSERT ON {table} WHEN EXISTS (SELECT 1 FROM {table} WHERE id=NEW.id) BEGIN SELECT RAISE(ABORT, 'append-only'); END")
        for operation in ("UPDATE", "DELETE"):
            db.execute(f"CREATE TRIGGER IF NOT EXISTS {table}_{operation} BEFORE {operation} ON {table} BEGIN SELECT RAISE(ABORT, 'append-only'); END")
    return db


def read(table, path=None):
    if table not in {"observations", "receipts", "outcomes"}:
        raise ValueError("invalid table")
    with closing(connect(path)) as db:
        records = db.execute(f"SELECT id,sha256,payload FROM {table} ORDER BY rowid").fetchall()
    result = {}
    for key, expected, raw in records:
        payload = json.loads(raw)
        if digest(payload) != expected:
            raise Rejected("stored_hash_mismatch")
        result[key] = payload
    return result


def append(table, key, payload, path=None, *, pregame_start=None):
    if table not in {"observations", "receipts", "outcomes"}:
        raise ValueError("invalid table")
    raw = canonical(payload).decode()
    with closing(connect(path)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        existing = db.execute(f"SELECT sha256,payload FROM {table} WHERE id=?", (key,)).fetchone()
        if existing:
            if existing != (digest(payload), raw):
                raise Rejected("duplicate_conflict")
            return False
        # Recheck at the actual write, not just before slow network calls.
        if pregame_start is not None and now() >= timestamp(pregame_start):
            raise Rejected("invalid_capture_time")
        db.execute(f"INSERT INTO {table} VALUES (?,?,?)", (key, digest(payload), raw))
    return True


def event_key(payload):
    identity(payload)
    return digest({"provider_namespace": "mlb", "provider_event_id": payload["provider_event_id"]})


def receipt_key(payload):
    return digest({"event": event_key(payload), "market_type": payload["quote"]["market_type"]})


def save_receipt(snapshot, path=None):
    p, _ = receipt_features(snapshot)
    if timestamp(p["captured_at"]) > now():
        raise Rejected("invalid_capture_time")
    return append("receipts", receipt_key(p), snapshot, path, pregame_start=p["game_start_utc"])


def save_outcome(outcome, path=None):
    event = event_key(outcome)
    receipts = [r for r in read("receipts", path).values() if event_key(r["payload"]) == event]
    if not receipts or any(identity(r["payload"]) != identity(outcome) for r in receipts):
        raise Rejected("outcome_identity_mismatch")
    if not timestamp(outcome["game_start_utc"]) <= timestamp(outcome["available_at"]) <= now():
        raise Rejected("invalid_outcome_time")
    label("total_over", 8.5, outcome.get("home_score"), outcome.get("away_score"), outcome["status"])
    return append("outcomes", event, outcome, path)


def export_records(path=None, *, settled_only=False):
    outcomes = read("outcomes", path)
    result = []
    for snapshot in read("receipts", path).values():
        outcome = outcomes.get(event_key(snapshot["payload"]))
        if not settled_only or outcome is not None:
            result.append({"snapshot": snapshot, "outcome": outcome})
    return result


def observe(endpoint, params=None):
    response = requests.get(BASE + endpoint, params=params, timeout=(3, 4))
    response.raise_for_status()
    payload = response.json()
    return {"source": "mlb_statsapi", "endpoint": endpoint, "params": params or {},
            "observed_at": now().isoformat(), "payload": payload}


def persist_observation(observation, path=None):
    if timestamp(observation["observed_at"]) > now():
        raise Rejected("future_observation")
    key = digest(observation)
    append("observations", key, observation, path)
    return key


class ScheduleGames(list):
    """Unique events plus quarantined variants; never choose a conflict winner."""
    def __init__(self, games, conflicts):
        super().__init__(games)
        self.conflicts = conflicts


def schedule_games(observation, reasons=None):
    grouped = {}
    duplicates = 0
    for day in observation["payload"]["dates"]:
        for game in day["games"]:
            key = stable_id(game.get("gamePk"))
            variants = grouped.setdefault(key, [])
            if game in variants:
                duplicates += 1
            else:
                variants.append(game)
    conflicts = [g for variants in grouped.values() if len(variants) > 1 for g in variants]
    if reasons is not None:
        if duplicates:
            reasons["identical_schedule_duplicates_collapsed"] += duplicates
        conflict_count = sum(len(v) > 1 for v in grouped.values())
        if conflict_count:
            reasons["conflicting_schedule_events_quarantined"] += conflict_count
    return ScheduleGames([v[0] for v in grouped.values() if len(v) == 1], conflicts)


def resolve_event(odds_game, games):
    """Reuse established MLB aliases; never equate two providers' numeric IDs."""
    from app_core.public_history import grading_team_name
    from app_core.mlb_event_matcher import scheduled_eastern_date
    if not odds_game.get("id"):
        raise Rejected("missing_provider_event_id")
    pair = tuple(grading_team_name(odds_game.get(s+"_team", ""), "MLB") for s in ("home", "away"))
    if not all(pair):
        raise Rejected("missing_matchup")
    day = scheduled_eastern_date(odds_game)
    for variant in getattr(games, "conflicts", []):
        variant_pair = tuple(grading_team_name(variant["teams"][s]["team"].get("name", ""), "MLB") for s in ("home", "away"))
        claimed = odds_game.get("provider_ids", {}).get("mlb")
        if (claimed is not None and str(claimed) == str(variant["gamePk"])) or (
                variant_pair == pair and scheduled_eastern_date({"start": variant["gameDate"]}) == day):
            raise Rejected("conflicting_schedule_event")
    candidates = [g for g in games if g.get("gameType") == "R" and
        tuple(grading_team_name(g["teams"][s]["team"].get("name", ""), "MLB") for s in ("home", "away")) == pair
        and scheduled_eastern_date({"start": g["gameDate"]}) == day]
    # Do not choose a doubleheader game by nearest start or by ignoring a final.
    if len(candidates) != 1:
        raise Rejected("event_identity_ambiguous" if candidates else "event_identity_missing")
    game = candidates[0]
    stable_id(game.get("gamePk"))
    for side in ("home", "away"):
        team_id(game["teams"][side]["team"].get("id"))
    if abs((timestamp(game["gameDate"]) - timestamp(odds_game["commence_time"])).total_seconds()) > 600:
        raise Rejected("event_start_mismatch")
    if game["status"]["abstractGameState"] != "Preview" or min(timestamp(game["gameDate"]), timestamp(odds_game["commence_time"])) <= now():
        raise Rejected("invalid_capture_time")
    claimed = odds_game.get("provider_ids", {}).get("mlb")
    if claimed is not None and str(claimed) != str(game["gamePk"]):
        raise Rejected("provider_id_conflict")
    return game


def exact_quotes(game):
    from core.wager_decisions import decimal_price
    observed = timestamp(game.get("live_receipt_observed_at"))
    if game.get("live_receipt_price_format") != "american" or game.get("odds_feed_source", "the_odds_api") != "the_odds_api":
        raise Rejected("unverified_live_quote_source")
    if observed > now() or (now()-observed).total_seconds() > 1800:
        raise Rejected("invalid_quote_time")
    grouped = {}
    for quote in json.loads(provider_quotes(game)):
        kind = quote["market_type"]
        book = canonical_book_label(quote["book"])
        if kind not in TARGETS or book not in BOOKS:
            continue
        if not quote.get("provider_event_id") or quote.get("provider_namespace") != "odds_api":
            raise Rejected("missing_provider_event_id")
        if quote.get("recorded_at") and timestamp(quote["recorded_at"]) > observed:
            raise Rejected("invalid_quote_time")
        price = decimal_price(quote["price"])
        if price is None or price <= 1 or quote.get("point") is None:
            continue
        q = {"market_type": kind, "line": quote["point"], "decimal_odds": price,
             "sportsbook": book, "observed_at": observed.isoformat(),
             "provider_event_id": quote["provider_event_id"], "provider_namespace": "odds_api",
             "provider_updated_at": quote.get("recorded_at"),
             "source_game_start_utc": timestamp(game["commence_time"]).isoformat()}
        key = (kind, book)
        if key in grouped and grouped[key] != q:
            raise Rejected("ambiguous_quote")
        grouped[key] = q
    # Fixed source order, not an outcome-dependent price/line choice.
    return {kind: next(grouped[(kind, b)] for b in BOOKS if (kind, b) in grouped)
            for kind in TARGETS if any((kind, b) in grouped for b in BOOKS)}


def prior_from_observation(observation):
    record = normalize_game(observation["payload"])
    available = timestamp(observation["observed_at"])
    if timestamp(record["completed_at"]) > available:
        raise Rejected("invalid_prior_game")
    return {"provider_namespace": "mlb", "game_id": stable_id(record["game_id"]),
            "season": record["season"], "home_id": team_id(record["home_id"]), "away_id": team_id(record["away_id"]),
            "home_score": record["home_score"], "away_score": record["away_score"], "status": "FINAL",
            "completed_at": record["completed_at"], "available_at": available.isoformat(),
            "observation_hash": digest(observation)}


def build_receipt(game, quote, prior_games, *, captured_at, source_refs):
    gid = stable_id(game.get("gamePk"))
    if quote.get("provider_namespace") != "odds_api" or not quote.get("provider_event_id"):
        raise Rejected("missing_provider_event_id")
    if not timestamp(quote["observed_at"]) <= timestamp(captured_at) or (timestamp(captured_at)-timestamp(quote["observed_at"])).total_seconds() > 1800:
        raise Rejected("invalid_quote_time")
    p = {"schema_version": SCHEMA, "provider_namespace": "mlb", "provider_event_id": gid,
         "home_team_id": team_id(game["teams"]["home"]["team"].get("id")),
         "away_team_id": team_id(game["teams"]["away"]["team"].get("id")),
         "season": int(game["season"]), "game_start_utc": timestamp(game["gameDate"]).isoformat(),
         "captured_at": captured_at, "prediction_cutoff": captured_at,
         "quote": deepcopy(quote), "prior_games": deepcopy(prior_games),
         "provider_ids": {"mlb": gid, "odds_api": quote["provider_event_id"]},
         "source_observations": source_refs}
    if game["status"]["abstractGameState"] != "Preview":
        raise Rejected("invalid_capture_time")
    if quote.get("source_game_start_utc") and timestamp(captured_at) >= timestamp(quote["source_game_start_utc"]):
        raise Rejected("invalid_capture_time")
    result = {"payload": p, "sha256": digest(p)}
    receipt_features(result)
    return result


def choose_prior_ids(game, games):
    selected = set()
    for side in ("home", "away"):
        team = team_id(game["teams"][side]["team"].get("id"))
        # Do not silently substitute older history for an unresolved prior event.
        for variant in getattr(games, "conflicts", []):
            if (str(variant.get("season")) == str(game["season"])
                    and team in [team_id(variant["teams"][s]["team"].get("id")) for s in ("home", "away")]
                    and timestamp(variant["gameDate"]) < timestamp(game["gameDate"])):
                raise Rejected("conflicting_prior_schedule_event")
        candidates = [g for g in games if g.get("gameType") == "R" and str(g.get("season")) == str(game["season"])
            and g["status"]["abstractGameState"] == "Final" and str(g["gamePk"]) != str(game["gamePk"])
            and team in [team_id(g["teams"][s]["team"].get("id")) for s in ("home", "away")]
            and timestamp(g["gameDate"]) < timestamp(game["gameDate"])]
        candidates.sort(key=lambda g: (timestamp(g["gameDate"]), int(g["gamePk"])), reverse=True)
        if len(candidates) < 10:
            raise Rejected("insufficient_prior_games")
        selected.update(stable_id(g["gamePk"]) for g in candidates[:10])
    return selected


def capture_live_games(odds_games, *, path=None, max_feeds=20, fetch=observe):
    """One fresh schedule plus a bounded batch of missing final-game feeds.

    Returns copies; ordinary odds, probabilities and wager controls are untouched.
    Calls without a real live Odds API observation marker make no network requests.
    """
    if not 0 <= max_feeds <= 100:
        raise ValueError("max_feeds must be 0..100")
    output = deepcopy(odds_games)
    for game in output:
        game.pop("mlb_pregame_receipts", None)  # Never replay a prior run as a fresh observation.
    report = {"receipts_created": 0, "receipts_skipped": 0, "reasons": {}, "prior_feeds_requested": 0, "prior_feeds_remaining": 0}
    reasons = Counter()
    eligible = []
    for game in output:
        try:
            if game.get("sport_key") != "baseball_mlb":
                raise Rejected("unsupported_sport")
            if not game.get("live_receipt_observed_at"):
                raise Rejected("invalid_quote_time")
            quotes = exact_quotes(game)
            if not quotes:
                raise Rejected("missing_quote")
            eligible.append((game, quotes))
        except (ValueError, KeyError, TypeError) as exc:
            reasons[str(exc) if isinstance(exc, Rejected) else "invalid_quote"] += 1
            report["receipts_skipped"] += 4
    if not eligible:
        report["reasons"] = dict(reasons)
        return output, report
    try:
        season = now().year
        observation = fetch("api/v1/schedule", {"sportId": 1, "season": season, "gameType": "R"})
        if (now()-timestamp(observation["observed_at"])).total_seconds() > 60:
            raise Rejected("stale_schedule_observation")
        source_ref = persist_observation(observation, path)
        games = schedule_games(observation, reasons)
        cached = {}
        for obs in read("observations", path).values():
            if obs.get("source") == "mlb_statsapi" and "/feed/live" in obs.get("endpoint", ""):
                try:
                    prior = prior_from_observation(obs)
                    cached[prior["game_id"]] = prior
                except (ValueError, KeyError, TypeError):
                    continue
        existing = read("receipts", path)
        resolved, missing = [], []
        for raw, quotes in eligible:
            try:
                game = resolve_event(raw, games)
                gid = stable_id(game["gamePk"])
                fields = {"home_team_id": team_id(game["teams"]["home"]["team"]["id"]),
                          "away_team_id": team_id(game["teams"]["away"]["team"]["id"]),
                          "mlb_provider_event_id": gid, "provider_ids": {"mlb": gid, "odds_api": raw["id"]}}
                fields["team_ids"] = [fields["home_team_id"], fields["away_team_id"]]
                for key in ("home_team_id", "away_team_id", "team_ids"):
                    if raw.get(key) is not None and raw[key] != fields[key]:
                        raise Rejected("stable_team_id_conflict")
                raw.update(fields)
                needed = choose_prior_ids(game, games)
                # Reject cache versions inconsistent with today's observed Final scores.
                current = {stable_id(g["gamePk"]): g for g in games}
                for key in needed:
                    prior = cached.get(key)
                    g = current[key]
                    if prior and any(prior[s+"_score"] != g["teams"][s].get("score") or prior[s+"_id"] != team_id(g["teams"][s]["team"].get("id")) for s in ("home", "away")):
                        cached.pop(key)
                    if key not in cached and key not in missing:
                        missing.append(key)
                resolved.append((raw, game, quotes, needed))
            except (ValueError, KeyError, TypeError) as exc:
                reasons[str(exc) if isinstance(exc, Rejected) else "invalid_event"] += 1
                report["receipts_skipped"] += 4
        batch = missing[:max_feeds]
        report["prior_feeds_requested"] = len(batch)
        report["prior_feeds_remaining"] = max(0, len(missing)-len(batch))
        def fetch_prior(gid):
            try:
                obs = fetch(f"api/v1.1/game/{gid}/feed/live")
                prior = prior_from_observation(obs)
                if prior["game_id"] != gid:
                    raise Rejected("invalid_prior_game")
                return obs, prior
            except (requests.RequestException, ValueError, KeyError, TypeError):
                return None
        with ThreadPoolExecutor(max_workers=4) as executor:
            for fetched in executor.map(fetch_prior, batch):
                if fetched is None:
                    reasons["invalid_prior_game"] += 1
                    continue
                obs, prior = fetched
                persist_observation(obs, path)
                cached[prior["game_id"]] = prior
        for raw, game, quotes, needed in resolved:
            raw_obs = {"source": "odds_api", "observed_at": raw["live_receipt_observed_at"],
                       "payload": {k:raw.get(k) for k in ("id", "sport_key", "home_team", "away_team", "commence_time", "bookmakers")}}
            quote_ref = persist_observation(raw_obs, path)
            for market in TARGETS:
                try:
                    if market not in quotes:
                        raise Rejected("missing_quote")
                    if needed - cached.keys():
                        raise Rejected("insufficient_prior_games")
                    receipt = build_receipt(game, quotes[market], [cached[k] for k in sorted(needed)],
                        captured_at=now().isoformat(), source_refs={"schedule": source_ref, "quotes": quote_ref})
                    # Pass this run's exact observed quote to inference, even when
                    # first-receipt training storage already contains this target.
                    raw.setdefault("mlb_pregame_receipts", {})[market] = receipt
                    key = receipt_key(receipt["payload"])
                    if key in existing:
                        reasons["duplicate_receipt"] += 1
                        report["receipts_skipped"] += 1
                        continue
                    save_receipt(receipt, path)
                    existing[key] = receipt
                    report["receipts_created"] += 1
                except (ValueError, KeyError, TypeError) as exc:
                    reasons[str(exc) if isinstance(exc, Rejected) else "invalid_receipt"] += 1
                    report["receipts_skipped"] += 1
    except (requests.RequestException, OSError, sqlite3.Error, ValueError, KeyError, TypeError) as exc:
        reasons[str(exc) if isinstance(exc, Rejected) else "receipt_source_or_storage_unavailable"] += 1
    report["receipts_skipped"] = len(output) * len(TARGETS) - report["receipts_created"]
    report["reasons"] = dict(reasons)
    return output, report


def reconcile(path=None, *, max_games=20, fetch=observe):
    """Append verified finals. Cancellation/postponement never implies a bet VOID."""
    if not 1 <= max_games <= 100:
        raise ValueError("max_games must be 1..100")
    receipts = read("receipts", path)
    done = read("outcomes", path)
    pending = {event_key(r["payload"]): r["payload"] for r in receipts.values() if event_key(r["payload"]) not in done}
    reasons, created = Counter(), 0
    for key, p in list(pending.items())[:max_games]:
        try:
            obs = fetch(f"api/v1.1/game/{stable_id(p['provider_event_id'])}/feed/live")
            game = normalize_game(obs["payload"])
            scheduled = timestamp(obs["payload"]["gameData"]["datetime"]["dateTime"])
            if (str(game["game_id"]) != p["provider_event_id"] or game["season"] != p["season"]
                or scheduled != timestamp(p["game_start_utc"])
                or any(team_id(game[s+"_id"]) != p[s+"_team_id"] for s in ("home", "away"))
                or timestamp(p["captured_at"]) >= timestamp(game["started_at"])):
                raise Rejected("outcome_identity_or_start_mismatch")
            if timestamp(game["completed_at"]) > timestamp(obs["observed_at"]):
                raise Rejected("invalid_outcome_time")
            ref = persist_observation(obs, path)
            outcome = {k:p[k] for k in ("provider_namespace", "provider_event_id", "home_team_id", "away_team_id", "season", "game_start_utc")}
            outcome.update(status="FINAL", home_score=game["home_score"], away_score=game["away_score"],
                           available_at=obs["observed_at"], observation_hash=ref)
            created += int(save_outcome(outcome, path))
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            reasons[str(exc) if isinstance(exc, Rejected) else "result_not_verified_final"] += 1
    return {"outcomes_created": created, "events_pending": len(pending)-created, "reasons": dict(reasons)}


def health(path=None):
    receipts, outcomes = read("receipts", path), read("outcomes", path)
    return {"receipts": len(receipts), "events": len({event_key(r["payload"]) for r in receipts.values()}),
            "outcomes": len(outcomes), "research_only": True, "automatic_training": False,
            "persistence_note": "Dedicated local append-only research store; retain/back up the database on persistent storage."}
