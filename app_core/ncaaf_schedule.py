"""Independent ESPN FBS/FCS schedule inventory. No prices or wager authority."""
from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
import re
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import requests

from app_core.ncaaf_identity import normalize_ncaaf_team
from app_core.provider_health import failure, sanitized_health

ET = ZoneInfo("America/New_York")
SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard"
INDEX = "https://sports.core.api.espn.com/v2/sports/football/leagues/college-football/events"
# Replaces the existing NCAAF identity capture's maximum six requests per refresh.
# Primary odds/fallback limits are independent and remain unchanged.
MAX_REQUESTS = 6
PAGE_SIZE = 300


def timestamp(value):
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else None
    except (ValueError, TypeError):
        return None


def window(start, end):
    first, last = date.fromisoformat(str(start)[:10]), date.fromisoformat(str(end)[:10])
    if last < first or (last - first).days >= 31:
        raise ValueError("DATE_RANGE_INVALID")
    return first, last


def _event(event, division, first, last):
    """Keep unknown kickoffs/statuses visible rather than inventing a date."""
    if not isinstance(event, dict) or not str(event.get("id", "")).isdigit():
        raise ValueError("SCHEDULE_EVENT_INVALID")
    competitions = event.get("competitions")
    incomplete = (not isinstance(competitions, list) or len(competitions) != 1
                  or not isinstance(competitions[0], dict))
    c = competitions[0] if not incomplete and isinstance(competitions[0], dict) else {}
    competitors = c.get("competitors")
    if not isinstance(competitors, list):
        competitors = []
        incomplete = True
    home = [t.get("team", {}) for t in competitors if isinstance(t, dict) and t.get("homeAway") == "home"]
    away = [t.get("team", {}) for t in competitors if isinstance(t, dict) and t.get("homeAway") == "away"]
    if len(home) != 1 or len(away) != 1:
        incomplete = True
        home, away = [{}], [{}]
    if not isinstance(home[0], dict) or not isinstance(away[0], dict):
        incomplete = True
        home, away = [{}], [{}]
    if not all(str(t.get("id", "")).isdigit() for t in (home[0], away[0])):
        incomplete = True
    kickoff = timestamp(c.get("date") or event.get("date"))
    if kickoff and not first <= kickoff.astimezone(ET).date() <= last:
        return None
    def names(team):
        return sorted({normalize_ncaaf_team(team[k]) for k in ("displayName", "shortDisplayName", "name", "abbreviation") if team.get(k)})
    eid = str(event["id"])
    status = c.get("status") or event.get("status") or {}
    status = status.get("type", {}) if isinstance(status, dict) else {}
    status = status if isinstance(status, dict) else {}
    name = str(status.get("name", "")).upper()
    state = ("CANCELLED" if "CANCEL" in name else "POSTPONED" if "POSTPON" in name else
             "STARTED" if status.get("state") in {"in", "post"} else
             "SCHEDULED" if status.get("state") == "pre" else "UNKNOWN")
    return {"schedule_event_id": "espn:college-football:" + eid,
            "schedule_namespace": "espn:college-football", "schedule_provider_id": eid,
            "competition_ids": [str(c.get("id") or eid)], "divisions": [division],
            "home_team": home[0].get("displayName") or home[0].get("name") or "",
            "away_team": away[0].get("displayName") or away[0].get("name") or "",
            "home_team_id": "espn:ncaaf:" + str(home[0]["id"]) if home[0].get("id") else None,
            "away_team_id": "espn:ncaaf:" + str(away[0]["id"]) if away[0].get("id") else None,
            "home_aliases": names(home[0]), "away_aliases": names(away[0]),
            "kickoff": kickoff.isoformat() if kickoff else None,
            "kickoff_revisions": [kickoff.isoformat()] if kickoff else [],
            "schedule_status": state, "identity_conflict": incomplete}


def inventory_from_events(batches, start, end, *, complete=False, reasons=(), observed_at=None):
    first, last = window(start, end)
    rows, raw_events, issues = {}, [], list(reasons)
    for division, events in batches:
        for event in events:
            try:
                row = _event(event, division, first, last)
            except (ValueError, TypeError, AttributeError):
                issues.append("SCHEDULE_RECORD_INVALID")
                continue
            if row is None:
                continue
            # Identity attachment needs schedule facts, never incidental prices.
            raw_events.append({"id": str(event["id"]), "date": event.get("date"), "competitions": []})
            competitions = event.get("competitions")
            for c in competitions if isinstance(competitions, list) else []:
                if not isinstance(c, dict):
                    continue
                source_teams = c.get("competitors")
                competitors = [{"homeAway": t.get("homeAway"), "team": {k: t.get("team", {}).get(k) for k in
                                ("id", "displayName", "shortDisplayName", "name")}}
                               for t in (source_teams if isinstance(source_teams, list) else []) if isinstance(t, dict) and isinstance(t.get("team"), dict)]
                raw_events[-1]["competitions"].append({"id": c.get("id"), "date": c.get("date"), "competitors": competitors})
            key = row["schedule_event_id"]
            if key in rows:
                old = rows[key]
                for field in ("divisions", "competition_ids", "kickoff_revisions"):
                    old[field] = sorted(set(old[field] + row[field]))
                if (old["home_team_id"], old["away_team_id"]) != (row["home_team_id"], row["away_team_id"]) or len(old["kickoff_revisions"]) > 1 or old["schedule_status"] != row["schedule_status"]:
                    old["identity_conflict"] = True
                    issues.append("SCHEDULE_REVISION_CONFLICT")
            else:
                rows[key] = row
            if row["kickoff"] is None or row["schedule_status"] == "UNKNOWN" or row["identity_conflict"]:
                issues.append("SCHEDULE_FACTS_INCOMPLETE")
    return {"schema_version": 1, "source": "espn_schedule", "start_date": first.isoformat(),
            "end_date": last.isoformat(), "observed_at": observed_at or datetime.now(timezone.utc).isoformat(),
            "complete": bool(complete and not issues), "status": "COMPLETE" if complete and not issues else "PARTIAL",
            "reasons": sorted(set(issues)), "events": list(rows.values()), "identity_events": raw_events}


def fetch_schedule(start, end, *, max_requests=MAX_REQUESTS):
    """Bounded read-only requests; independent index equality proves feed coverage.

    Query both nominal Eastern dates and the following UTC calendar day. All
    returned events are filtered by their actual Eastern kickoff. Pagination
    stalls, missing indexes, record errors and budget exhaustion stay PARTIAL.
    """
    first, last = window(start, end)
    dates = first.strftime("%Y%m%d") + "-" + (last + timedelta(days=1)).strftime("%Y%m%d")
    batches, receipts, reasons = [], [], []
    budget = min(MAX_REQUESTS, max(0, int(max_requests)))
    def get(url, params):
        nonlocal budget
        if budget <= 0:
            reasons.append("SCHEDULE_REQUEST_BUDGET_EXHAUSTED")
            return None
        budget -= 1
        try:
            response = requests.get(url, params=params, timeout=(3, 5))
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("INVALID_RESPONSE")
            receipts.append({"endpoint": "index" if url == INDEX else "scoreboard", "group": params["groups"],
                             "page": params.get("page", 1), "outcome": "SUCCESS",
                             "sha256": hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()})
            return data
        except Exception as exc:
            detail = failure(exc)
            if isinstance(exc, ValueError):
                detail = {"outcome": "INVALID_RESPONSE", "http_status": None}
            receipts.append({"endpoint": "index" if url == INDEX else "scoreboard", "group": params["groups"], **detail})
            reasons.append("SCHEDULE_" + detail["outcome"])
            return None
    # College scoreboard accepts one date, not a dates range. Its returned
    # season calendar supplies authoritative week IDs; fetch whole weeks when
    # available, then filter by Eastern kickoff. Never invent a week number.
    seed = get(SCOREBOARD, {"dates": first.strftime("%Y%m%d"), "groups": "80", "limit": PAGE_SIZE})
    weeks = []
    try:
        league = seed["leagues"][0]
        year = league["season"]["year"]
        begin = datetime.combine(first, time.min, ET).astimezone(timezone.utc)
        finish = datetime.combine(last + timedelta(days=1), time.min, ET).astimezone(timezone.utc)
        for season in league["calendar"]:
            for entry in season.get("entries", []):
                lo, hi = timestamp(entry.get("startDate")), timestamp(entry.get("endDate"))
                if lo and hi and lo < finish and hi >= begin:
                    weeks.append({"year": int(year), "seasontype": int(season["value"]), "week": int(entry["value"])})
    except (KeyError, TypeError, ValueError, IndexError):
        weeks = []
    page_queries = {"80": [], "81": []}
    boards = {"80": {"events": list(seed.get("events", []))} if seed and isinstance(seed.get("events"), list) else {"events": []},
              "81": {"events": []}}
    if weeks:
        for week in weeks:
            for group in ("80", "81"):
                board = get(SCOREBOARD, {**week, "groups": group, "limit": PAGE_SIZE})
                if board and isinstance(board.get("events"), list):
                    boards[group]["events"].extend(board["events"])
                    if len(board["events"]) >= PAGE_SIZE:
                        page_queries[group].append({**week, "groups": group, "limit": PAGE_SIZE})
                else:
                    reasons.append("SCHEDULE_EVENTS_MISSING")
    else:
        # Missing calendars do not authorize guessed weeks. Bounded daily
        # fallback includes the following UTC day; longer windows stay PARTIAL.
        day = first
        while day <= last + timedelta(days=1):
            for group in ("80", "81"):
                if day == first and group == "80":
                    continue
                board = get(SCOREBOARD, {"dates": day.strftime("%Y%m%d"), "groups": group, "limit": PAGE_SIZE})
                if board and isinstance(board.get("events"), list):
                    boards[group]["events"].extend(board["events"])
                    if len(board["events"]) >= PAGE_SIZE:
                        page_queries[group].append({"dates": day.strftime("%Y%m%d"), "groups": group, "limit": PAGE_SIZE})
                else:
                    reasons.append("SCHEDULE_EVENTS_MISSING")
            day += timedelta(days=1)
    for group, division in (("80", "FBS"), ("81", "FCS")):
        board = boards[group]
        events = board.get("events") if board else None
        if not isinstance(events, list):
            reasons.append("SCHEDULE_EVENTS_MISSING")
            events = []
        index_ids, count, pages = set(), None, None
        page = 1
        while True:
            data = get(INDEX, {"dates": dates, "groups": group, "limit": PAGE_SIZE, "page": page})
            if data is None:
                break
            try:
                n, p, size, total = (data[k] for k in ("count", "pageIndex", "pageSize", "pageCount"))
                if not all(type(x) is int and x >= 0 for x in (n, p, size, total)) or p != page or size < 1 or (n > 0 and total < 1):
                    raise ValueError()
                if count is not None and (n, total) != (count, pages):
                    raise ValueError()
                count, pages = n, total
                items = data["items"]
                if not isinstance(items, list):
                    raise ValueError()
                ids = set()
                for item in items:
                    ref = urlparse(item["$ref"])
                    match = re.fullmatch(r"/v2/sports/football/leagues/college-football/events/(\d+)", ref.path)
                    if ref.hostname != "sports.core.api.espn.com" or not match:
                        raise ValueError()
                    ids.add(match[1])
                if len(ids) != len(items) or (page > 1 and ids & index_ids):
                    raise ValueError()
                index_ids.update(ids)
            except (KeyError, ValueError, TypeError):
                reasons.append("SCHEDULE_PAGINATION_INVALID")
                break
            if page >= pages:
                break
            page += 1
        # A capped scoreboard needs its own pages; repeated pages are a failure.
        board_ids = {str(e.get("id")) for e in events if isinstance(e, dict)}
        for query in page_queries[group]:
            offset = PAGE_SIZE
            while count is not None and not index_ids.issubset(board_ids):
                extra = get(SCOREBOARD, {**query, "page": offset // PAGE_SIZE + 1, "offset": offset})
                more = extra.get("events") if extra else None
                if not isinstance(more, list) or not more or not any(str(e.get("id")) not in board_ids for e in more):
                    reasons.append("SCHEDULE_PAGINATION_STALLED")
                    break
                events.extend(more)
                board_ids.update(str(e.get("id")) for e in more)
                offset += len(more)
                if len(more) < PAGE_SIZE:
                    break
        if count is None or len(index_ids) != count or not index_ids.issubset(board_ids):
            reasons.append("SCHEDULE_INDEX_MISMATCH")
        batches.append((division, events))
    result = inventory_from_events(batches, start, end, complete=not reasons, reasons=reasons)
    result["requests"] = receipts
    return result


def _known_espn_id(game):
    ids = game.get("provider_ids")
    if isinstance(ids, dict) and str(ids.get("espn", "")).isdigit():
        return str(ids["espn"])
    value = str(game.get("id") or game.get("game_id") or "")
    if re.fullmatch(r"espn-\d+", value):
        return value[5:]
    quote_ids = {str(q.get("provider_event_id"))[5:] for q in _quotes(game)
                 if q.get("provider_namespace") == "espn_ncaaf_fcs_scoreboard"
                 and re.fullmatch(r"espn-\d+", str(q.get("provider_event_id")))}
    if len(quote_ids) == 1:
        return next(iter(quote_ids))
    return None


def match_event(game, inventory):
    """No fuzzy matching; explicit IDs still need compatible teams/kickoff."""
    if game.get("schedule_match_status") == "AMBIGUOUS_PROVIDER_ID":
        return None, "AMBIGUOUS"
    start = timestamp(game.get("commence_time") or game.get("game_start_utc") or game.get("kickoff"))
    hid = _known_espn_id(game)
    home, away = (normalize_ncaaf_team(game.get(k)) for k in ("home_team", "away_team"))
    matches = []
    for event in inventory.get("events", []):
        if hid and event["schedule_provider_id"] != hid:
            continue
        agrees = home in event["home_aliases"] and away in event["away_aliases"]
        if agrees and (hid or (start and start.isoformat() in event["kickoff_revisions"])):
            matches.append(event)
    if len(matches) != 1:
        return None, "AMBIGUOUS" if len(matches) > 1 else "UNRESOLVED"
    event = matches[0]
    if event["identity_conflict"] or not start or start.isoformat() not in event["kickoff_revisions"]:
        return event, "KICKOFF_OR_IDENTITY_CONFLICT"
    return event, "MATCHED"


def merge_schedule_odds(primary, fallback, inventory):
    """Primary quotes win only for one unambiguous canonical event."""
    result = deepcopy(list(primary or []))
    fallback = deepcopy(list(fallback or []))
    identities = {}
    for game in result + fallback:
        event, state = match_event(game, inventory)
        namespace = str(game.get("odds_feed_source") or
                        ("espn_ncaaf_fcs_scoreboard" if _known_espn_id(game) else "odds_api"))
        if state == "MATCHED":
            identities.setdefault((namespace, event["schedule_event_id"]), []).append(game)
    for games in identities.values():
        if len({str(g.get("id") or g.get("game_id")) for g in games}) > 1:
            for game in games:
                game["schedule_match_status"] = "AMBIGUOUS_PROVIDER_ID"
    seen = {e["schedule_event_id"] for g in result if (pair := match_event(g, inventory))[1] == "MATCHED" for e in [pair[0]]}
    for game in fallback or []:
        event, state = match_event(game, inventory)
        if state != "MATCHED" or event["schedule_event_id"] not in seen:
            result.append(deepcopy(game))
            if state == "MATCHED":
                seen.add(event["schedule_event_id"])
    for game in result:
        event, state = match_event(game, inventory)
        game["schedule_match_status"] = ("AMBIGUOUS_PROVIDER_ID" if game.get("schedule_match_status") == "AMBIGUOUS_PROVIDER_ID" else state)
        if event:
            game["schedule_event_id"] = event["schedule_event_id"]
        # Preserve historical matchup and provider IDs. Unresolved events must
        # not collide just because teams happen to share a calendar day.
        game["historical_matchup_id"] = game.get("historical_matchup_id") or game.get("matchup_id")
        game["matchup_id"] = (event["schedule_event_id"] if state == "MATCHED" else
                              "ncaaf:unresolved:" + str(game.get("odds_feed_source") or "odds_api") + ":" + str(game.get("id")))
        game["schedule_inventory_key"] = game["matchup_id"]
    return result


def _records(value):
    return value.to_dict("records") if hasattr(value, "to_dict") else list(value or [])


def _quotes(row):
    value = row.get("provider_quotes")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            value = []
    return [q for q in value if isinstance(q, dict)] if isinstance(value, list) else []


def _qualified(row):
    """Report the existing strict contract, never infer authority from a pick."""
    contract = row.get("wager_contract")
    if row.get("production_eligible") is not True or not isinstance(contract, dict):
        return False
    from core.live_wager_contract import validate_snapshot
    try:
        validate_snapshot(contract)
    except (ValueError, TypeError):
        return False
    return contract.get("production_eligible") is True and contract.get("quote_fresh") is True


def coverage(inventory, games=(), candidates=(), selections=(), provider_health=None, *, now=None):
    """One row per schedule event; projection never manufactures a selection."""
    now = timestamp(now) or datetime.now(timezone.utc)
    stages, unresolved = {}, []
    for name, records in (("games", games), ("candidates", candidates), ("selections", selections)):
        groups = {}
        for row in _records(records):
            if str(row.get("league", "NCAAF")).upper() != "NCAAF":
                continue
            event, state = match_event(row, inventory)
            if state == "MATCHED":
                groups.setdefault(event["schedule_event_id"], []).append(row)
            else:
                possible = [e["schedule_event_id"] for e in inventory.get("events", [])
                            if normalize_ncaaf_team(row.get("home_team")) in e["home_aliases"] and
                            normalize_ncaaf_team(row.get("away_team")) in e["away_aliases"]]
                unresolved.append({"stage": name, "matchup_id": str(row.get("matchup_id") or ""), "status": state,
                                   "possible_schedule_event_ids": possible,
                                   "provider_kickoff": str(row.get("commence_time") or row.get("game_start_utc") or "")})
        stages[name] = groups
    health = sanitized_health(provider_health)
    provider = health.get("sports", {}).get("americanfootball_ncaaf", {})
    failed = bool(provider and (provider.get("outcome") not in {"SUCCESS", "SUCCESS_EMPTY"} or provider.get("fallback_errors") or provider.get("processing") == "FAILED"))
    rows = []
    for event in inventory.get("events", []):
        eid = event["schedule_event_id"]
        matched, pool, selected = (stages[s].get(eid, []) for s in ("games", "candidates", "selections"))
        quotes = [q for r in matched + pool for q in _quotes(r)]
        valid_quotes = [q for q in quotes if timestamp(q.get("recorded_at")) and
                        timestamp(q.get("recorded_at")) <= now and timestamp(event["kickoff"]) and
                        timestamp(q.get("recorded_at")) < timestamp(event["kickoff"])]
        # Existing authority must already approve the exact selected row.
        qualified = [r for r in selected if _qualified(r) and
                     not event["identity_conflict"] and event["schedule_status"] == "SCHEDULED" and
                     timestamp(event["kickoff"]) and timestamp(event["kickoff"]) > now and valid_quotes]
        reasons = []
        if event["identity_conflict"] or any(eid in r["possible_schedule_event_ids"] for r in unresolved):
            reasons.append("IDENTITY_UNRESOLVED")
        if event["schedule_status"] in {"POSTPONED", "CANCELLED"}:
            reasons.append(event["schedule_status"])
        elif event["schedule_status"] == "STARTED" or (timestamp(event["kickoff"]) and timestamp(event["kickoff"]) <= now):
            reasons.append("STARTED")
        if not matched:
            reasons.append("PROVIDER_FAILURE" if failed else "QUOTE_UNAVAILABLE")
        elif not quotes:
            reasons.append("QUOTE_UNAVAILABLE")
        elif not valid_quotes:
            reasons.append("QUOTE_PROVENANCE_INCOMPLETE")
        if not qualified:
            reasons.append("QUALIFIED_SELECTION_UNAVAILABLE")
        row = {k: event[k] for k in ("schedule_event_id", "schedule_namespace", "schedule_provider_id", "home_team", "away_team", "home_team_id", "away_team_id", "kickoff", "schedule_status")}
        row.update(divisions="|".join(event["divisions"]), kickoff_revisions=json.dumps(event["kickoff_revisions"]),
                   provider_match=json.dumps(sorted({str(r.get("game_id") or r.get("id") or r.get("matchup_id")) for r in matched})),
                   historical_matchup_ids=json.dumps(sorted({str(r.get("historical_matchup_id") or r.get("matchup_id")) for r in matched + pool})),
                   possible_unresolved_matchup_ids=json.dumps(sorted({r["matchup_id"] for r in unresolved if eid in r["possible_schedule_event_ids"]})),
                   possible_provider_kickoffs=json.dumps(sorted({r["provider_kickoff"] for r in unresolved if eid in r["possible_schedule_event_ids"]})),
                   provider_identities=json.dumps(sorted({json.dumps({"namespace": q.get("provider_namespace"), "event_id": q.get("provider_event_id")}, sort_keys=True) for q in quotes})),
                   provider_failure=failed, provider_outcome=provider.get("outcome", "NOT_RECORDED"),
                   quote_coverage="TIMESTAMPED_EVIDENCE" if valid_quotes else "PROVENANCE_INCOMPLETE" if quotes else "UNAVAILABLE",
                   quote_count=len({json.dumps(q, sort_keys=True, default=str) for q in quotes}),
                   candidate_count=len(pool), ranked_count=len(selected), qualified_count=len(qualified),
                   selection_status="QUALIFIED_SELECTION_AVAILABLE" if qualified else "PASS",
                   research_state="RESEARCH_CANDIDATE_AVAILABLE" if pool else "NO_RESEARCH_CANDIDATE",
                   exclusion_reasons="|".join(reasons), inventory_status=inventory["status"],
                   schedule_observed_at=inventory["observed_at"])
        rows.append(row)
    counts = {"scheduled": len(rows), "matched": sum(r["provider_match"] != "[]" for r in rows),
              "quoted": sum(r["quote_count"] > 0 for r in rows), "timestamped": sum(r["quote_coverage"] == "TIMESTAMPED_EVIDENCE" for r in rows),
              "ranked": sum(r["ranked_count"] > 0 for r in rows), "qualified": sum(r["qualified_count"] > 0 for r in rows)}
    return {"rows": rows, "counts": counts, "inventory_status": inventory["status"],
            "inventory_reasons": inventory["reasons"], "unresolved": unresolved}


def refresh_coverage(diagnostics, candidates=(), selections=()):
    inventory = diagnostics.get("ncaaf_schedule")
    if isinstance(inventory, dict):
        diagnostics["ncaaf_coverage"] = coverage(inventory, diagnostics.get("ncaaf_provider_games", []),
                                               candidates, selections, diagnostics.get("provider_health"))
