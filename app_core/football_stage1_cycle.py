"""Authenticated Stage 1 football capture; intentionally has no model or wager path."""
from __future__ import annotations

from collections import Counter
from contextlib import closing
from datetime import datetime, time, timedelta, timezone
import json
from pathlib import Path
import uuid

import requests

from app_core import football_stage1 as foundation
from app_core import prospective_evidence as evidence
from app_core import prospective_remote

ESPN = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
CFBD = "https://api.collegefootballdata.com"
ODDS = "https://api.the-odds-api.com/v4/sports"


class ProviderFailure(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _get_json(url, *, params=None, headers=None, get=None, ledger=None, provider="UNKNOWN"):
    get = get or requests.get
    ledger[provider]["attempts"] += 1
    try:
        response = get(url, params=params, headers=headers, timeout=15, allow_redirects=False)
        code = int(response.status_code)
        if provider == "ODDS":
            response_headers = getattr(response, "headers", {}) or {}
            for header, field in (("x-requests-remaining", "quota_remaining"),
                                  ("x-requests-used", "quota_used")):
                raw_value = response_headers.get(header)
                if raw_value is not None:
                    try:
                        ledger[provider][field] = max(0, int(raw_value))
                    except (ValueError, TypeError):
                        pass
        if code == 429:
            ledger[provider]["rate_limits"] += 1
            ledger[provider]["failures"] += 1
            raise ProviderFailure(provider + "_RATE_LIMIT")
        if code != 200:
            ledger[provider]["failures"] += 1
            raise ProviderFailure(provider + "_HTTP_" + str(code))
        payload = response.json()
    except ProviderFailure:
        raise
    except (requests.RequestException, ValueError, TypeError):
        ledger[provider]["failures"] += 1
        raise ProviderFailure(provider + "_API_FAILURE") from None
    ledger[provider]["successes"] += 1
    return payload


def _nfl_schedule(now, *, get, ledger):
    # Independent daily scoreboard calls include all games on each UTC/ET edge.
    # The 15-day window supplies prior results and the entire 7-day early horizon.
    events, errors = {}, []
    first = (now - timedelta(days=7)).date()
    for offset in range(15):
        day = first + timedelta(days=offset)
        try:
            payload = _get_json(ESPN, params={"dates": day.strftime("%Y%m%d"), "limit": 100},
                                get=get, ledger=ledger, provider="ESPN")
            if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
                raise ProviderFailure("NFL_SCHEDULE_SCHEMA_FAILURE")
            for item in payload["events"]:
                if isinstance(item, dict) and item.get("id"):
                    ident = str(item["id"])
                    if ident in events and events[ident] != item:
                        # The current response wins for this run; immutable revisions
                        # preserve prior observations from prior runs.
                        ledger["ESPN"]["duplicate_revisions"] += 1
                    events[ident] = item
        except ProviderFailure as exc:
            errors.append({"date": day.isoformat(), "reason": exc.code})
    return list(events.values()), errors


def _ncaaf_schedule(now, token, *, get, ledger):
    headers = {"Authorization": "Bearer " + token.removeprefix("Bearer ")}
    games = _get_json(CFBD + "/games", params={"year": now.year, "seasonType": "regular"},
                      headers=headers, get=get, ledger=ledger, provider="CFBD")
    teams = _get_json(CFBD + "/teams/fbs", params={"year": now.year},
                      headers=headers, get=get, ledger=ledger, provider="CFBD")
    if not isinstance(games, list) or not isinstance(teams, list):
        raise ProviderFailure("NCAAF_CFBD_SCHEMA_FAILURE")
    if not games:
        raise ProviderFailure("NCAAF_CFBD_DISCOVERY_EMPTY")
    team_catalog = {str(x["id"]): x for x in teams if isinstance(x, dict) and x.get("id") is not None}
    fbs_ids = set(team_catalog)
    if not fbs_ids:
        raise ProviderFailure("NCAAF_FBS_POPULATION_EMPTY")
    alias_owners = {}
    for team_id, team in team_catalog.items():
        names = [team.get("school"), team.get("alt_name")]
        if team.get("school") and team.get("mascot"):
            names.append(str(team["school"]) + " " + str(team["mascot"]))
        for name in names:
            if isinstance(name, str) and name.strip():
                alias_owners.setdefault(foundation._name("NCAAF", name), set()).add(team_id)
    aliases = {team_id: {name for name, owners in alias_owners.items() if owners == {team_id}}
               for team_id in fbs_ids}
    first, last = now - timedelta(days=7), now + timedelta(days=8)
    selected, invalid = [], 0
    for game in games:
        if not isinstance(game, dict):
            invalid += 1
            continue
        try:
            when = foundation.at(game["startDate"])
        except (KeyError, ValueError, TypeError):
            invalid += 1
            selected.append(game)  # Keep malformed provider rows visible in coverage.
            continue
        if first <= when < last:
            selected.append(game)
    def policy(schedule):
        home_fbs = schedule["home_team_id"] in fbs_ids
        away_fbs = schedule["away_team_id"] in fbs_ids
        if not home_fbs and not away_fbs:
            return "NON_FBS_GAME_EXCLUDED"
        if not home_fbs or not away_fbs:
            return "FBS_VS_NON_FBS_OR_UNKNOWN_EXCLUDED"
        return None
    return selected, policy, team_catalog, aliases, {"cfbd_regular_games": len(games), "fbs_team_ids": len(fbs_ids),
                              "window_games": len(selected), "invalid_schedule_rows": invalid,
                              "population_policy": "FBS vs FBS regular season; neutral and nonconference included; FBS vs FCS, bowls, CFP, postseason excluded"}


def _odds(sport, key, *, get, ledger):
    payload = _get_json(f"{ODDS}/{foundation.SPORT_KEYS[sport]}/odds",
                        params={"apiKey": key, "regions": "us", "markets": "spreads,totals",
                                "oddsFormat": "american", "dateFormat": "iso"},
                        get=get, ledger=ledger, provider="ODDS")
    if not isinstance(payload, list):
        raise ProviderFailure(sport + "_ODDS_SCHEMA_FAILURE")
    return payload


def _completed_result(sport, source):
    try:
        if sport == "NCAAF":
            if source.get("completed") is not True:
                return None
            return {"provider_event_id": str(source["id"]), "home_team_id": str(source["homeId"]),
                    "away_team_id": str(source["awayId"]), "home_score": source["homePoints"],
                    "away_score": source["awayPoints"], "status": "FINAL", "provider_response": source}
        game = source["competitions"][0]
        if source.get("status", {}).get("type", {}).get("completed") is not True:
            return None
        teams = {x["homeAway"]: x for x in game["competitors"]}
        return {"provider_event_id": str(source["id"]),
                "home_team_id": str(teams["home"]["team"]["id"]),
                "away_team_id": str(teams["away"]["team"]["id"]),
                "home_score": int(teams["home"]["score"]),
                "away_score": int(teams["away"]["score"]),
                "status": "FINAL", "provider_response": source}
    except (KeyError, IndexError, ValueError, TypeError):
        raise ProviderFailure(sport + "_RESULT_SCHEMA_FAILURE") from None


def _readiness(path, denominator, now, current_sources=None):
    games = []
    completed, result_count, settled_games = 0, 0, 0
    with closing(evidence.connect(path)) as db:
        selected_manifest = {(row["game_id"], row["market_family"]): row["manifest_id"]
                             for row in db.execute(
                                 "SELECT game_id,market_family,manifest_id FROM prospective_football_training_manifest WHERE sport=?",
                                 (denominator["sport"],))}
        for game in denominator["games"]:
            if not game.get("regular_season_target"):
                continue
            game_id = game["game_id"]
            event_row = foundation._read(db, "prospective_football_event", "version_id", game["event_version_id"])
            identity_verified = bool(event_row["home_team_id"] and event_row["away_team_id"] and
                                     event_row["home_team_id"] != event_row["away_team_id"])
            source = (current_sources or {}).get(game["provider_event_id"])
            if source is None:
                source = json.loads(event_row["raw_source"])
            is_completed = (source.get("completed") is True if denominator["sport"] == "NCAAF" else
                            source.get("status", {}).get("type", {}).get("completed") is True)
            completed += int(is_completed)
            has_result = db.execute("SELECT 1 FROM prospective_football_result WHERE game_id=? LIMIT 1",
                                    (game_id,)).fetchone() is not None
            result_count += int(is_completed and has_result)
            has_settlement = db.execute("SELECT 1 FROM prospective_football_settlement WHERE game_id=? LIMIT 1",
                                        (game_id,)).fetchone() is not None
            settled_games += int(has_settlement)
            market_rows = []
            for market in ("SPREAD", "TOTAL"):
                quotes = db.execute("SELECT quote_id FROM prospective_football_quote WHERE game_id=? AND market_family=?",
                                    (game_id, market)).fetchall()
                pregame_quotes = db.execute("""
                    SELECT count(*) FROM prospective_football_quote q
                    JOIN prospective_football_event e ON e.version_id=q.event_version_id
                    WHERE q.game_id=? AND q.market_family=? AND q.quote_verified=1
                      AND q.observed_at<e.scheduled_start
                """, (game_id, market)).fetchone()[0]
                result = db.execute("SELECT result_id FROM prospective_football_result WHERE game_id=?",
                                    (game_id,)).fetchone()
                result_versions = db.execute("SELECT DISTINCT home_score,away_score FROM prospective_football_result WHERE game_id=?",
                                             (game_id,)).fetchall()
                ready = db.execute("SELECT training_row_id FROM prospective_football_active_training_row WHERE game_id=? AND market_family=? LIMIT 1",
                                   (game_id, market)).fetchone()
                selected = selected_manifest.get((game_id, market))
                conflict = len(result_versions) > 1
                blockers = (["RESULT_CORRECTION_CONFLICT"] if conflict else [] if ready else
                            ["NO_VERIFIED_PREGAME_PRICE"] if not quotes else
                            ["RESULT_PENDING"] if foundation.at(game["kickoff"]) > foundation.at(now) else
                            ["RESULT_UNVERIFIED"] if not result else ["TRAINING_ELIGIBILITY_BLOCKED"])
                if ready and not selected and not conflict:
                    blockers = ["MANIFEST_INTEGRITY_FILTER"]
                market_rows.append({"market_family": market, "training_row_status":
                                    "TRAINING_READY" if selected and not conflict else "TRAINING_BLOCKED", "blockers": blockers,
                                    "verified_quotes": len(quotes), "pregame_verified_quotes": pregame_quotes,
                                    "result_present": bool(result), "manifest_id": selected})
            games.append({"game_id": game_id, "identity_verified": identity_verified,
                          "completed": is_completed, "markets": market_rows})
    total = len(games)
    rates = {}
    for market, field in (("SPREAD", "spread_price_available"), ("TOTAL", "total_price_available")):
        rates[market.lower() + "_price_coverage"] = (sum(g.get(field, False) for g in denominator["games"]
                                                      if g.get("regular_season_target")) / total if total else None)
        rates[market.lower() + "_training_ready_rate"] = (sum(any(m["market_family"] == market and m["training_row_status"] == "TRAINING_READY"
                                                               for m in g["markets"]) for g in games) / total if total else None)
    rates["schedule_identity_coverage"] = (sum(g["identity_verified"] for g in games) / total if total else None)
    rates["result_coverage_for_completed_games"] = result_count / completed if completed else None
    rates["settlement_coverage"] = settled_games / total if total else None
    rates["training_ready_rate"] = (sum(m["training_row_status"] == "TRAINING_READY" for g in games
                                         for m in g["markets"]) / (2 * total) if total else None)
    with closing(evidence.connect(path)) as db:
        result_rows = db.execute("SELECT count(*) FROM prospective_football_result WHERE sport=?",
                                 (denominator["sport"],)).fetchone()[0]
        settlement_rows = db.execute("SELECT count(*) FROM prospective_football_settlement WHERE sport=?",
                                     (denominator["sport"],)).fetchone()[0]
        training_ready_rows = db.execute("SELECT count(*) FROM prospective_football_active_training_row WHERE sport=?",
                                         (denominator["sport"],)).fetchone()[0]
        market_summary = {}
        for market in ("SPREAD", "TOTAL"):
            selected = sum(key[1] == market for key in selected_manifest)
            settled_rows = db.execute("SELECT count(*) FROM prospective_football_settlement WHERE sport=? AND market_family=?",
                                      (denominator["sport"], market)).fetchone()[0]
            ready_rows = db.execute("SELECT count(*) FROM prospective_football_active_training_row WHERE sport=? AND market_family=?",
                                    (denominator["sport"], market)).fetchone()[0]
            blocked_rows = db.execute("SELECT count(*) FROM prospective_football_training_row WHERE sport=? AND market_family=? AND training_row_status='TRAINING_BLOCKED'",
                                      (denominator["sport"], market)).fetchone()[0]
            statuses = [m for g in games for m in g["markets"] if m["market_family"] == market]
            market_summary[market] = {
                "target_games": total,
                "priced_pregame_games": sum(m["pregame_verified_quotes"] > 0 for m in statuses),
                "completed_games": completed,
                "settled_rows": settled_rows,
                "training_ready_rows": ready_rows,
                "training_blocked_rows": blocked_rows,
                "training_blocked_games": sum(m["training_row_status"] == "TRAINING_BLOCKED" for m in statuses),
                "independent_manifest_events": selected,
                "blocker_counts": dict(Counter(b for m in statuses for b in m["blockers"]))}
    return {"sport": denominator["sport"], "target_games": total,
            "completed_games_in_window": completed, "completed_games_with_result": result_count,
            "games_with_settlement": settled_games, "result_rows": result_rows,
            "settlement_rows": settlement_rows, "training_ready_rows": training_ready_rows,
            "rates": rates, "market_summary": market_summary, "games": games}


def _lifecycle(path, denominator, readiness):
    """Prove a real game's source-to-manifest chain, or expose its blocker."""
    sport = denominator["sport"]
    target = [g for g in denominator["games"] if g.get("regular_season_target")]
    by_id = {g["game_id"]: g for g in readiness["games"]}
    if sport == "NFL":
        selected = next((g for g in target if
                         {foundation._name("NFL", g["home"]), foundation._name("NFL", g["away"])} ==
                         {foundation._name("NFL", "Atlanta Falcons"), foundation._name("NFL", "Green Bay Packers")}), None)
        case = "ATLANTA_GREEN_BAY"
    else:
        priced_completed = [g for g in target if by_id[g["game_id"]]["completed"] and
                            any(m["pregame_verified_quotes"] for m in by_id[g["game_id"]]["markets"])]
        selected = min(priced_completed, key=lambda g: (g["kickoff"], g["game_id"])) if priced_completed else None
        case = "EARLIEST_COMPLETED_PRICED_NCAAF"
    if selected is None:
        return {"case": case, "status": "BLOCKED",
                "blocker": ("TARGET_GAME_OUTSIDE_CURRENT_SCHEDULE_WINDOW" if sport == "NFL" else
                            "NO_COMPLETED_TARGET_WITH_VERIFIED_PREGAME_PRICE"),
                "completed_target_games": sum(x["completed"] for x in readiness["games"]),
                "priced_pregame_target_games": sum(any(m["pregame_verified_quotes"] for m in x["markets"])
                                                   for x in readiness["games"])}
    game_id = selected["game_id"]
    with closing(evidence.connect(path)) as db:
        event = foundation._read(db, "prospective_football_event", "version_id", selected["event_version_id"])
        quotes = [foundation._read(db, "prospective_football_quote", "quote_id", x[0])
                  for x in db.execute("SELECT quote_id FROM prospective_football_quote WHERE game_id=? ORDER BY observed_at,quote_id",
                                      (game_id,))]
        results = [foundation._read(db, "prospective_football_result", "result_id", x[0])
                   for x in db.execute("SELECT result_id FROM prospective_football_result WHERE game_id=? ORDER BY observed_at,result_id",
                                       (game_id,))]
        settlements = [dict(x) for x in db.execute(
            "SELECT settlement_id,quote_id,result_id,market_family,selection,line,outcome,settled_at "
            "FROM prospective_football_settlement WHERE game_id=? ORDER BY market_family,settlement_id", (game_id,))]
        manifest = [dict(x) for x in db.execute(
            "SELECT * FROM prospective_football_training_manifest WHERE game_id=? ORDER BY market_family", (game_id,))]
    market_status = {m["market_family"]: m for m in by_id[game_id]["markets"]}
    blockers = {market: item["blockers"] for market, item in market_status.items()}
    return {"case": case, "status": "PROVED" if len(manifest) == 2 else "BLOCKED",
            "game_id": game_id, "home": selected["home"], "away": selected["away"],
            "kickoff": selected["kickoff"], "event_version_id": event["version_id"],
            "event_source_hash": event["source_hash"], "identity_mapping_version": event["identity_mapping_version"],
            "quotes": [{k: q[k] for k in ("quote_id", "market_family", "selection", "line",
                                           "american_odds", "sportsbook", "observed_at", "provider_last_update",
                                           "capture_horizon", "source_hash", "quote_verified")}
                       for q in quotes],
            "results": [{k: r[k] for k in ("result_id", "home_score", "away_score", "observed_at",
                                           "available_at", "result_status", "source_hash")}
                        for r in results],
            "settlements": settlements, "manifest": manifest, "market_blockers": blockers,
            "blocker": None if len(manifest) == 2 else
            ("RESULT_NOT_FINAL" if not results else "MARKET_TRAINING_READINESS_BLOCKED")}


def run_cycle(path, folder, client, odds_key, cfbd_key, *, now=None, get=None, theover_files=()):
    now = foundation.at(now or datetime.now(timezone.utc))
    run_id = "football-stage1-" + uuid.uuid4().hex
    ledger = {p: Counter() for p in ("ESPN", "CFBD", "ODDS")}
    report = {"schema": "football-stage1-cycle-v1", "run_id": run_id,
              "started_at": foundation.iso(now), "execution_state": "RUNNING",
              "requested_slate_success": False, "sports": {}, "remote": {}, "provider_requests": {},
              "policy": {"NFL": "regular season; preseason and postseason excluded",
                         "NCAAF": "FBS vs FBS regular season, all conferences and neutral sites; FBS vs FCS, bowls and CFP excluded"},
              "no_model_calibration_prediction_activation_stake_or_wager": True}
    if not odds_key or not cfbd_key:
        raise ProviderFailure("MISSING_FOOTBALL_PROVIDER_CREDENTIALS")
    remote_session = {}
    report["remote"]["restore"] = prospective_remote.sync(path, client, folder, session=remote_session)
    report["remote"]["pre_mutation_verified"] = True
    all_schedules = []
    for sport in ("NFL", "NCAAF"):
        sport_report = {"requested_slate_success": False, "errors": []}
        report["sports"][sport] = sport_report
        try:
            if sport == "NFL":
                raw_schedules, errors = _nfl_schedule(now, get=get, ledger=ledger)
                sport_report["errors"].extend(errors)
                policy, team_catalog, aliases = None, None, None
                discovery = {"schedule_window_days": 15, "schedule_source": "ESPN daily scoreboard"}
            else:
                raw_schedules, policy, team_catalog, aliases, discovery = _ncaaf_schedule(
                    now, cfbd_key, get=get, ledger=ledger)
            sport_report["discovery"] = discovery
            odds_events = _odds(sport, odds_key, get=get, ledger=ledger)
            observed = datetime.now(timezone.utc) if get is None else now
            current_sources = {str(x["id"]): x for x in raw_schedules
                               if isinstance(x, dict) and x.get("id") is not None}
            denominator = foundation.coverage(path, raw_schedules, odds_events, sport=sport,
                                              observed=observed, run_id=run_id, target_policy=policy,
                                              team_catalog=team_catalog, aliases=aliases)
            sport_report["denominator"] = denominator
            schedules = []
            with closing(evidence.connect(path)) as db:
                for item in denominator["games"]:
                    if item.get("event_version_id"):
                        schedules.append(foundation._read(db, "prospective_football_event", "version_id", item["event_version_id"]))
            all_schedules.extend(schedules)
            schedule_window_start = (datetime.combine((now-timedelta(days=7)).date(), time.min,
                                                      tzinfo=timezone.utc) if sport == "NFL"
                                     else now-timedelta(days=7))
            schedule_window_end = schedule_window_start + timedelta(days=15)
            diagnostic = foundation.provider_diagnostic(schedules, odds_events, sport=sport,
                                                       observed=observed, aliases=aliases,
                                                       target_policy=policy,
                                                       schedule_window_start=schedule_window_start,
                                                       schedule_window_end=schedule_window_end)
            persisted = {x["game_id"] for x in denominator["games"] if x.get("game_id") and
                         (x.get("spread_price_available") or x.get("total_price_available"))}
            for item in diagnostic:
                item["persisted"] = item.get("canonical_match") in persisted
            sport_report["provider_events"] = diagnostic
            classes = Counter(x["classification"] for x in diagnostic)
            revised = [x for x in diagnostic if x["classification"] == "KICKOFF_TIME_REVISION"]
            sport_report["reconciliation"] = {
                "classification_counts": dict(classes),
                "unexplained_provider_events": sum(not x.get("classification") or
                                                    not x.get("classification_reason") for x in diagnostic),
                "unexplained_target_games": sum(not x.get("status") or
                                                x.get("status") == "PENDING_ODDS_LOOKUP"
                                                for x in denominator["games"] if x.get("regular_season_target")),
                "strict_kickoff_tolerance_seconds": 60,
                "matching_policy": "STRICT_60_SECONDS_UNCHANGED",
                "kickoff_revision_candidates": [
                    {k: x.get(k) for k in ("provider_event_id", "home", "away", "commence_time",
                                            "candidate_game_ids", "kickoff_delta_seconds")}
                    for x in revised],
                "matched_kickoff_delta_seconds": [x["kickoff_delta_seconds"] for x in diagnostic
                                                  if x["classification"] == "MATCHED_TARGET"]}
            sport_report["provider_event_aggregate"] = {
                "response_events": len(diagnostic), "unique_provider_ids": len({x["provider_event_id"] for x in diagnostic}),
                "status_counts": dict(Counter(x["status"] for x in diagnostic)),
                "canonical_matches": sum(bool(x.get("canonical_match")) for x in diagnostic),
                "spread_market_events": sum(bool(x.get("spread_present")) for x in diagnostic),
                "spread_price_events": sum(bool(x.get("spread_price_available")) for x in diagnostic),
                "total_market_events": sum(bool(x.get("total_present")) for x in diagnostic),
                "total_price_events": sum(bool(x.get("total_price_available")) for x in diagnostic),
                "pregame_valid_events": sum(bool(x.get("pregame_valid")) for x in diagnostic),
                "persisted_events": sum(bool(x.get("persisted")) for x in diagnostic)}
            for schedule in schedules:
                if schedule["season_type"].casefold() != "regular" or (policy and policy(schedule)):
                    continue
                source = current_sources.get(schedule["provider_event_id"])
                if source is None:
                    source = json.loads(schedule["raw_source"])
                try:
                    raw_result = _completed_result(sport, source)
                    if raw_result is None:
                        continue
                    result, created = foundation.append_result(path, schedule, raw_result, observed, source=schedule["provider_namespace"])
                    foundation.settle_game(path, schedule, result, observed)
                except (ProviderFailure, ValueError) as exc:
                    sport_report["errors"].append({"game_id": schedule["game_id"],
                                                   "reason": exc.code if isinstance(exc, ProviderFailure) else str(exc)})
            sport_report["readiness"] = _readiness(path, denominator, observed, current_sources)
            sport_report["lifecycle"] = _lifecycle(path, denominator, sport_report["readiness"])
            with closing(evidence.connect(path)) as db:
                target_ids_for_horizons = [g["game_id"] for g in denominator["games"]
                                           if g.get("regular_season_target")]
                if target_ids_for_horizons:
                    placeholders = ",".join("?" for _ in target_ids_for_horizons)
                    sport_report["capture_horizons"] = dict(Counter(
                        row[0] for row in db.execute(
                            "SELECT capture_horizon FROM prospective_football_quote WHERE game_id IN (" +
                            placeholders + ") AND quote_verified=1", target_ids_for_horizons)))
                else:
                    sport_report["capture_horizons"] = {}
            target_ids = {g["game_id"] for g in denominator["games"] if g.get("regular_season_target")}
            matched_ids = {x.get("canonical_match") for x in diagnostic if x.get("canonical_match") in target_ids}
            sport_report["readiness"]["rates"]["provider_event_match_rate"] = (
                len(matched_ids) / len(target_ids) if target_ids else None)
            for market, raw_key in (("SPREAD", "spread_present"), ("TOTAL", "total_present")):
                market_ids = {x.get("canonical_match") for x in diagnostic
                              if x.get("canonical_match") in target_ids and x.get(raw_key)}
                sport_report["readiness"]["rates"][market.lower() + "_market_coverage"] = (
                    len(market_ids) / len(target_ids) if target_ids else None)
            sport_report["readiness"]["rates"]["pregame_valid_quote_rate"] = (
                sum(bool(x.get("pregame_valid")) for x in diagnostic if x.get("canonical_match") in target_ids)
                / len(target_ids) if target_ids else None)
            quote_due = [g for g in denominator["games"] if g.get("regular_season_target") and
                         foundation.horizon(g["kickoff"], observed) != "SNAPSHOT_WINDOW_MISSED"]
            sport_report["requested_slate_success"] = (not sport_report["errors"] and
                                                         denominator["requested_slate_success"])
            if sport == "NCAAF" and denominator["scheduled_target_games"] == 0:
                sport_report["discovery"]["zero_discovery_reason"] = (
                    "NCAAF_QUERY_WINDOW_EMPTY" if not raw_schedules else "NCAAF_FILTER_REMOVED_ALL")
            elif sport == "NCAAF" and quote_due and not odds_events:
                sport_report["discovery"]["zero_discovery_reason"] = "NCAAF_ODDS_DISCOVERY_EMPTY"
                sport_report["requested_slate_success"] = False
            elif sport == "NCAAF" and quote_due and all(x.get("sport_key") != foundation.SPORT_KEYS["NCAAF"]
                                           for x in odds_events if isinstance(x, dict)):
                sport_report["discovery"]["zero_discovery_reason"] = "NCAAF_WRONG_SPORT_KEY"
                sport_report["requested_slate_success"] = False
            elif sport == "NCAAF" and quote_due and not matched_ids:
                sport_report["discovery"]["zero_discovery_reason"] = "NCAAF_IDENTITY_PROJECTION_FAILED"
                sport_report["requested_slate_success"] = False
            elif sport == "NCAAF" and target_ids and all(
                    foundation.horizon(g["kickoff"], observed) == "SNAPSHOT_WINDOW_MISSED"
                    for g in denominator["games"] if g.get("regular_season_target")):
                sport_report["discovery"]["window_status"] = "NCAAF_SCHEDULER_WINDOW_MISSED"
            due_ids = {g["game_id"] for g in quote_due}
            priced_due_ids = {d.get("canonical_match") for d in diagnostic
                              if d.get("canonical_match") in due_ids and
                              d.get("spread_price_available") and d.get("total_price_available") and
                              d.get("pregame_valid")}
            missing_due_ids = sorted(due_ids - priced_due_ids)
            if missing_due_ids:
                sport_report["errors"].append({"reason": sport + "_NO_VERIFIED_PREGAME_PRICE",
                                               "quote_due_games": len(due_ids),
                                               "missing_game_ids": missing_due_ids})
                sport_report["requested_slate_success"] = False
        except ProviderFailure as exc:
            sport_report["errors"].append({"reason": exc.code})
            if sport == "NCAAF":
                sport_report["discovery"] = {"zero_discovery_reason": (
                    "NCAAF_RATE_LIMIT" if exc.code.endswith("RATE_LIMIT") else
                    exc.code if exc.code.startswith("NCAAF_") else "NCAAF_API_FAILURE"),
                    "provider_blocker": exc.code}
        except Exception as exc:
            sport_report["errors"].append({"reason": sport + "_PIPELINE_FAILURE", "error_type": type(exc).__name__})
    for file_path in theover_files:
        report.setdefault("theover", []).append(foundation.ingest_theover(path, file_path, all_schedules, now))
    try:
        report["remote"]["backup"] = prospective_remote.sync(path, client, folder, session=remote_session)
        report["remote"]["backup_readback_verified"] = True
    except Exception as exc:
        report["remote"]["backup_readback_verified"] = False
        report["remote"]["backup_error_type"] = type(exc).__name__
        report["execution_state"] = "BACKUP_VERIFICATION_FAILED"
    report["provider_requests"] = {p: dict(counts) for p, counts in ledger.items()}
    report["requested_slate_success"] = (all(x.get("requested_slate_success") for x in report["sports"].values())
                                         and report["remote"].get("backup_readback_verified", False))
    if report["execution_state"] != "BACKUP_VERIFICATION_FAILED":
        report["execution_state"] = "COMPLETE" if report["requested_slate_success"] else "PARTIAL_FAILURE"
    return report


def write_artifacts(report, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "football-stage1-cycle-audit.json": report,
        "football-stage1-schedule-denominator.json": {s: v.get("denominator") for s, v in report["sports"].items()},
        "football-stage1-nfl-32-event-diagnostic.json": report["sports"].get("NFL", {}).get("provider_events", []),
        "football-stage1-ncaaf-discovery-diagnostic.json": report["sports"].get("NCAAF", {}).get("discovery", {}),
        "football-stage1-odds-coverage.json": {s: v.get("provider_event_aggregate") for s, v in report["sports"].items()},
        "football-stage1-training-readiness.json": {s: v.get("readiness") for s, v in report["sports"].items()},
        "football-stage1-provider-reconciliation.json": {
            s: {"summary": v.get("reconciliation"), "events": v.get("provider_events")}
            for s, v in report["sports"].items()},
        "football-stage1-lifecycle-audit.json": {
            s: v.get("lifecycle") for s, v in report["sports"].items()},
        "football-stage1-research-manifest-readiness.json": {
            s: {"market_summary": v.get("readiness", {}).get("market_summary"),
                "capture_horizons": v.get("capture_horizons")}
            for s, v in report["sports"].items()},
    }
    if "theover" in report:
        artifacts["football-stage1-theover-reconciliation.json"] = report["theover"]
    for name, data in artifacts.items():
        (out / name).write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n")
