"""Schedule-first, append-only football research evidence. No prediction authority."""
from __future__ import annotations

from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path

from app_core import prospective_evidence as evidence
from app_core.nfl_identity import nfl_result_name
from app_core.ncaaf_identity import ALIASES as NCAAF_ALIASES, _key as ncaaf_key

SPORT_KEYS = {"NFL": "americanfootball_nfl", "NCAAF": "americanfootball_ncaaf"}
MARKETS = {"spreads": "SPREAD", "totals": "TOTAL"}
IDENTITY_VERSION = "football-exact-provider-pair-v1"
HORIZON_VERSION = "football-early-7d-mid-24h-final-2h-v2"
TRAINING_VERSION = "football-score-settlement-v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    raw = value if isinstance(value, bytes) else (value if isinstance(value, str) else canonical(value)).encode()
    return hashlib.sha256(raw).hexdigest()


def at(value):
    if not isinstance(value, (str, datetime)):
        raise ValueError("INVALID_TIMESTAMP")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    except ValueError:
        raise ValueError("INVALID_TIMESTAMP") from None
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("INVALID_TIMESTAMP")
    return result.astimezone(timezone.utc)


def iso(value):
    return at(value).isoformat()


def _name(sport, value):
    if not isinstance(value, str) or not value.strip():
        return None
    if sport == "NFL":
        return nfl_result_name(value)
    key = ncaaf_key(value)
    # Only exact normalized spelling or a curated provider alias is allowed.
    return NCAAF_ALIASES.get(key, key)


def _same_identity(schedule, offer, aliases=None):
    try:
        if offer.get("sport_key") != SPORT_KEYS[schedule["sport"]]:
            return False
        if abs((at(offer["commence_time"]) - at(schedule["scheduled_start"])).total_seconds()) > 60:
            return False
        for side in ("home", "away"):
            candidate = _name(schedule["sport"], offer[f"{side}_team"])
            if aliases and schedule["sport"] == "NCAAF":
                allowed = aliases.get(schedule[f"{side}_team_id"], set())
                if candidate not in allowed:
                    return False
            elif candidate != _name(schedule["sport"], schedule[f"{side}_team"]):
                return False
        return True
    except (KeyError, ValueError, TypeError):
        return False


def _matching_sides(schedule, offer, aliases=None):
    """Return verified team-side matches without asserting an event match."""
    matched = []
    for side in ("home", "away"):
        candidate = _name(schedule["sport"], offer.get(f"{side}_team"))
        if aliases and schedule["sport"] == "NCAAF" and schedule[f"{side}_team_id"] in aliases:
            verified = candidate in aliases[schedule[f"{side}_team_id"]]
        else:
            verified = candidate is not None and candidate == _name(schedule["sport"], schedule[f"{side}_team"])
        matched.append(verified)
    return tuple(matched)


def _provider_classification(event, schedules, *, sport, aliases, target_policy,
                             schedule_window_start, schedule_window_end, duplicate):
    """Classify one provider event; candidate rows never become forced matches."""
    if not isinstance(event, dict) or not event.get("id") or event.get("sport_key") != SPORT_KEYS[sport]:
        return "PROVIDER_DATA_INVALID", "MISSING_ID_OR_WRONG_SPORT_KEY", [], None
    if duplicate:
        return "DUPLICATE_PROVIDER_EVENT", "REPEATED_PROVIDER_EVENT_ID", [], None
    try:
        start = at(event["commence_time"])
        if not all(isinstance(event.get(f"{side}_team"), str) and event[f"{side}_team"].strip()
                   for side in ("home", "away")):
            raise ValueError("MISSING_TEAM")
    except (KeyError, ValueError, TypeError):
        return "PROVIDER_DATA_INVALID", "INVALID_START_OR_TEAM", [], None
    candidates = [(s, _matching_sides(s, event, aliases)) for s in schedules if s.get("sport") == sport]
    pair = [s for s, sides in candidates if sides == (True, True)]
    exact = [s for s in pair if abs((start-at(s["scheduled_start"])).total_seconds()) <= 60]
    if len(exact) > 1:
        return "AMBIGUOUS_MATCH", "MULTIPLE_EXACT_SCHEDULE_CANDIDATES", [s["game_id"] for s in exact], None
    if len(exact) == 1:
        schedule = exact[0]
        if schedule["season_type"].casefold() != "regular":
            return "NON_TARGET_SEASON_TYPE", schedule["season_type"], [schedule["game_id"]], 0
        excluded = target_policy(schedule) if target_policy else None
        if excluded:
            return "NON_TARGET_POPULATION", excluded, [schedule["game_id"]], 0
        delta = round((start-at(schedule["scheduled_start"])).total_seconds())
        return "MATCHED_TARGET", "STRICT_TEAM_AND_KICKOFF_MATCH", [schedule["game_id"]], delta
    if len(pair) > 1:
        return "AMBIGUOUS_MATCH", "MULTIPLE_TEAM_PAIR_CANDIDATES", [s["game_id"] for s in pair], None
    if len(pair) == 1:
        schedule = pair[0]
        delta = round((start-at(schedule["scheduled_start"])).total_seconds())
        if abs(delta) <= 24 * 3600:
            return "KICKOFF_TIME_REVISION", "STRICT_60_SECOND_TOLERANCE_EXCEEDED", [schedule["game_id"]], delta
        if schedule_window_start and schedule_window_end and not at(schedule_window_start) <= start < at(schedule_window_end):
            return "OUTSIDE_TARGET_WINDOW", "PROVIDER_START_OUTSIDE_SCHEDULE_QUERY", [], None
        return "SCHEDULE_EVENT_MISSING", "TEAM_PAIR_OUTSIDE_24_HOUR_REVISION_AUDIT_BOUND", [schedule["game_id"]], delta
    outside = bool(schedule_window_start and schedule_window_end and
                   not at(schedule_window_start) <= start < at(schedule_window_end))
    if outside:
        return "OUTSIDE_TARGET_WINDOW", "PROVIDER_START_OUTSIDE_SCHEDULE_QUERY", [], None
    # A unique nearby one-sided candidate can explain a non-target game
    # without treating the other, unverified team as a canonical match.
    nearby = [s for s, sides in candidates if any(sides) and
              abs((start-at(s["scheduled_start"])).total_seconds()) <= 60]
    if len(nearby) == 1:
        schedule = nearby[0]
        excluded = target_policy(schedule) if target_policy else None
        if excluded:
            return "NON_TARGET_POPULATION", excluded + "; OTHER_TEAM_UNVERIFIED", [schedule["game_id"]], None
        return "TEAM_IDENTITY_MISMATCH", "ONE_SCHEDULE_TEAM_VERIFIED", [schedule["game_id"]], None
    if len(nearby) > 1:
        return "AMBIGUOUS_MATCH", "MULTIPLE_ONE_SIDED_CANDIDATES", [s["game_id"] for s in nearby], None
    return "SCHEDULE_EVENT_MISSING", "NO_VERIFIED_SCHEDULE_CANDIDATE", [], None


def _read(db, table, key, value):
    row = db.execute(f"SELECT * FROM {table} WHERE {key}=?", (value,)).fetchone()
    if row is None:
        return None
    item = dict(row)
    if digest(item["payload"]) != item["payload_hash"] or (
            "raw_source" in item and digest(item["raw_source"]) != item["source_hash"]):
        raise ValueError("FOOTBALL_STORED_HASH_MISMATCH")
    return item


def _append(db, table, key, row, raw=None):
    row = dict(row)
    if raw is not None:
        raw = canonical(raw).encode()
        row["raw_source"] = raw
        row["source_hash"] = digest(raw)
    payload = {k: v for k, v in row.items() if k != "raw_source"}
    row["payload"] = canonical(payload)
    row["payload_hash"] = digest(row["payload"])
    old = _read(db, table, key, row[key])
    if old is not None:
        if any(old[k] != v for k, v in row.items()):
            raise ValueError("FOOTBALL_IMMUTABLE_CONFLICT")
        return False
    columns = list(row)
    db.execute(f"INSERT INTO {table} ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
               [row[k] for k in columns])
    return True


def _event_fields(sport, source):
    if sport == "NFL":
        competitions = source.get("competitions") or []
        if len(competitions) != 1:
            raise ValueError("EVENT_IDENTITY_AMBIGUOUS")
        game = competitions[0]
        competitors = game.get("competitors") or []
        home = [x.get("team", {}) for x in competitors if x.get("homeAway") == "home"]
        away = [x.get("team", {}) for x in competitors if x.get("homeAway") == "away"]
        if len(home) != 1 or len(away) != 1:
            raise ValueError("EVENT_IDENTITY_AMBIGUOUS")
        season = source.get("season") or {}
        kind = {1: "preseason", 2: "regular", 3: "postseason"}.get(season.get("type"), "unknown")
        start = game.get("date") or source.get("date")
        return dict(provider_namespace="ESPN", provider_event_id=str(source.get("id") or ""),
                    season=season.get("year"), week=(source.get("week") or {}).get("number"),
                    season_type=kind, home_team=home[0].get("displayName"),
                    away_team=away[0].get("displayName"),
                    home_team_id=str(home[0].get("id") or ""),
                    away_team_id=str(away[0].get("id") or ""),
                    scheduled_start=iso(start), neutral_site=int(bool(game.get("neutralSite"))),
                    venue=(game.get("venue") or {}).get("fullName"))
    if sport == "NCAAF":
        return dict(provider_namespace="CFBD", provider_event_id=str(source.get("id") or ""),
                    season=source.get("season"), week=source.get("week"),
                    season_type=source.get("seasonType") or "unknown",
                    home_team=source.get("homeTeam"), away_team=source.get("awayTeam"),
                    home_team_id=str(source.get("homeId") or ""),
                    away_team_id=str(source.get("awayId") or ""),
                    scheduled_start=iso(source.get("startDate")),
                    neutral_site=int(bool(source.get("neutralSite"))),
                    venue=source.get("venue"))
    raise ValueError("UNSUPPORTED_SPORT")


def append_schedule(path, sport, source, observed, team_catalog=None):
    """Keep a schedule revision even if its identity or market is blocked."""
    fields = _event_fields(sport, source)
    if not fields["provider_event_id"]:
        raise ValueError("NO_PROVIDER_EVENT")
    game_id = f"{sport.lower()}:{fields['provider_namespace'].lower()}:{fields['provider_event_id']}"
    # ESPN refreshes presentation/status metadata on every read. Version the
    # schedule facts themselves; result revisions live in the result table.
    # The first provider response for each schedule version remains immutable.
    version_id = digest(["football-event-v2", game_id, fields])
    with closing(evidence.connect(path)) as db, db:
        old = _read(db, "prospective_football_event", "version_id", version_id)
        if old is not None:
            if team_catalog:
                _append_team_identities(db, fields, sport, team_catalog, observed)
            return old, False
        row = dict(version_id=version_id, game_id=game_id, sport=sport,
                   season=fields["season"] if type(fields["season"]) is int else 0,
                   week=fields["week"] if type(fields["week"]) is int else None,
                   season_type=fields["season_type"],
                   provider_namespace=fields["provider_namespace"],
                   provider_event_id=fields["provider_event_id"],
                   home_team=fields["home_team"] or "UNKNOWN",
                   away_team=fields["away_team"] or "UNKNOWN",
                   home_team_id=fields["home_team_id"] or None,
                   away_team_id=fields["away_team_id"] or None,
                   scheduled_start=fields["scheduled_start"],
                   neutral_site=fields["neutral_site"], venue=fields["venue"],
                   discovered_at=iso(observed), identity_mapping_version=IDENTITY_VERSION)
        _append(db, "prospective_football_event", "version_id", row, source)
        _append_team_identities(db, fields, sport, team_catalog or {}, observed)
        return _read(db, "prospective_football_event", "version_id", version_id), True


def _append_team_identities(db, fields, sport, team_catalog, observed):
    for role in ("home", "away"):
        team_id, team_name = fields[f"{role}_team_id"], fields[f"{role}_team"]
        if not team_id or not team_name:
            continue
        source = team_catalog.get(team_id) or {"id": team_id, "name": team_name}
        aliases = sorted({str(x) for x in (source.get("school"), source.get("alt_name"),
                                            (str(source.get("school")) + " " + str(source.get("mascot")))
                                            if source.get("school") and source.get("mascot") else None)
                          if isinstance(x, str) and x.strip()})
        identity_id = digest([sport, fields["provider_namespace"], team_id, team_name, source])
        if not _read(db, "prospective_football_team_identity", "identity_id", identity_id):
            _append(db, "prospective_football_team_identity", "identity_id", dict(
                identity_id=identity_id, sport=sport,
                canonical_team_id=f"{sport.lower()}:{fields['provider_namespace'].lower()}:{team_id}",
                provider_namespace=fields["provider_namespace"], provider_team_id=team_id,
                canonical_name=team_name, provider_name=team_name, aliases=canonical(aliases),
                mapping_version=IDENTITY_VERSION, verified_at=iso(observed),
                mapping_source="CFBD_FBS_TEAM_CATALOG" if team_id in team_catalog else "schedule_provider"), source)


def horizon(start, observed):
    """Classify immutable pregame research snapshots without leaving a 2h-24h gap."""
    minutes = (at(start) - at(observed)).total_seconds() / 60
    if 24 * 60 <= minutes <= 7 * 24 * 60:
        return "EARLY_RESEARCH"
    if 120 < minutes < 24 * 60:
        return "MID_PREGAME"
    if 5 <= minutes <= 120:
        return "FINAL_LEGAL_PREGAME"
    return "SNAPSHOT_WINDOW_MISSED"


def _valid_price(price):
    return type(price) is int and (price >= 100 or price <= -100)


def _quotes(event, observed):
    """Yield exact two-sided offers; reject malformed/late markets as a unit."""
    start = at(event["commence_time"])
    if at(observed) >= start:
        return []
    offers = []
    books = set()
    for book in event.get("bookmakers", []):
        if not isinstance(book, dict) or not isinstance(book.get("key"), str) or book["key"] in books:
            continue
        books.add(book["key"])
        markets = set()
        for market in book.get("markets", []):
            if not isinstance(market, dict) or market.get("key") not in MARKETS or market["key"] in markets:
                continue
            markets.add(market["key"])
            try:
                updated = at(market["last_update"])
                if not updated <= at(observed) < start or (at(observed)-updated).total_seconds() > 900:
                    continue
                outcomes = market["outcomes"]
                expected = ({event["home_team"], event["away_team"]} if market["key"] == "spreads"
                            else {"Over", "Under"})
                if len(outcomes) != 2 or {x["name"] for x in outcomes} != expected:
                    continue
                if any(isinstance(x.get("point"), bool) for x in outcomes):
                    continue
                lines = [float(x["point"]) for x in outcomes]
                prices = [x["price"] for x in outcomes]
                if (not all(math.isfinite(x) and x * 2 == int(x * 2) for x in lines) or
                        not all(_valid_price(x) for x in prices)):
                    continue
                if market["key"] == "spreads" and abs(sum(lines)) > 1e-9:
                    continue
                if market["key"] == "totals" and (lines[0] != lines[1] or lines[0] <= 0):
                    continue
                for outcome, line, price in zip(outcomes, lines, prices):
                    offers.append((book["key"], market, outcome["name"], line, price, updated))
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
    return offers


def append_offers(path, schedule, event, observed, run_id, aliases=None):
    if not schedule.get("home_team_id") or not schedule.get("away_team_id") or (
            schedule["home_team_id"] == schedule["away_team_id"]):
        return 0, "EVENT_IDENTITY_AMBIGUOUS"
    if not _same_identity(schedule, event, aliases):
        return 0, "EVENT_IDENTITY_AMBIGUOUS"
    capture_horizon = horizon(schedule["scheduled_start"], observed)
    if capture_horizon == "SNAPSHOT_WINDOW_MISSED":
        return 0, "SNAPSHOT_WINDOW_MISSED"
    offers = _quotes(event, observed)
    count = 0
    with closing(evidence.connect(path)) as db, db:
        existing_markets = {x[0] for x in db.execute(
            "SELECT DISTINCT market_family FROM prospective_football_quote WHERE game_id=? AND capture_horizon=? AND quote_verified=1",
            (schedule["game_id"], capture_horizon))}
        if existing_markets == set(MARKETS.values()):
            return 0, "HORIZON_ALREADY_CAPTURED"
        for book, market, selection, line, price, updated in offers:
            if MARKETS[market["key"]] in existing_markets:
                continue
            raw = canonical(event).encode()
            raw_hash = digest(raw)
            # An unchanged offer may be observed on both sides of a capture
            # boundary. Keep one immutable snapshot per horizon, while the
            # existing-market guard prevents duplicate snapshots within it.
            quote_id = digest(["football-quote-v2", HORIZON_VERSION, capture_horizon,
                               schedule["game_id"], event["id"], book, market["key"],
                               selection, line, price, iso(updated), raw_hash])
            old = _read(db, "prospective_football_quote", "quote_id", quote_id)
            if old is not None:
                continue
            row = dict(quote_id=quote_id, game_id=schedule["game_id"],
                       event_version_id=schedule["version_id"], sport=schedule["sport"],
                       market_family=MARKETS[market["key"]], selection=selection, line=line,
                       american_odds=price,
                       decimal_odds=1 + (price / 100 if price > 0 else 100 / abs(price)),
                       sportsbook=book, provider="THE_ODDS_API", provider_event_id=event["id"],
                       odds_event_id=event["id"], observed_at=iso(observed),
                       provider_last_update=iso(updated), capture_run_id=run_id,
                       identity_mapping_hash=digest({k: sorted((aliases or {}).get(k, set())) for k in
                                                     (schedule["home_team_id"], schedule["away_team_id"])}),
                       capture_horizon=capture_horizon,
                       minutes_to_start=(at(schedule["scheduled_start"])-at(observed)).total_seconds()/60,
                       quote_verified=1)
            count += _append(db, "prospective_football_quote", "quote_id", row, event)
    return count, None if count else "NO_NEW_VERIFIED_MARKET" if existing_markets else "NO_SPREAD_OR_TOTAL_PRICE"


def append_result(path, schedule, raw, observed, *, source):
    """A correction is a new source hash/result ID; original evidence stays frozen."""
    start = at(schedule["scheduled_start"])
    when = at(observed)
    if when <= start:
        raise ValueError("RESULT_BEFORE_KICKOFF")
    if (source != schedule["provider_namespace"] or
            raw.get("provider_event_id") != schedule["provider_event_id"] or
            str(raw.get("home_team_id")) != schedule["home_team_id"] or
            str(raw.get("away_team_id")) != schedule["away_team_id"] or
            raw.get("status") != "FINAL" or
            not isinstance(raw.get("provider_response"), dict)):
        raise ValueError("RESULT_IDENTITY_OR_AVAILABILITY_UNVERIFIED")
    home, away = raw.get("home_score"), raw.get("away_score")
    if any(type(x) is not int or x < 0 for x in (home, away)):
        raise ValueError("RESULT_SCORE_UNVERIFIED")
    provider = raw["provider_response"]
    try:
        if source == "CFBD":
            provider_values = (str(provider["id"]), str(provider["homeId"]),
                               str(provider["awayId"]), provider["homePoints"], provider["awayPoints"])
            complete = provider["completed"] is True
        else:
            competitors = {x["homeAway"]: x for x in provider["competitions"][0]["competitors"]}
            provider_values = (str(provider["id"]), str(competitors["home"]["team"]["id"]),
                               str(competitors["away"]["team"]["id"]),
                               int(competitors["home"]["score"]), int(competitors["away"]["score"]))
            complete = provider["status"]["type"]["completed"] is True
    except (KeyError, IndexError, TypeError, ValueError):
        raise ValueError("RESULT_PROVIDER_SOURCE_INVALID") from None
    if not complete or provider_values != (raw["provider_event_id"], raw["home_team_id"],
                                          raw["away_team_id"], home, away):
        raise ValueError("RESULT_PROVIDER_SOURCE_MISMATCH")
    # Score/status are the result facts. ESPN may revise presentation metadata
    # on every fetch without changing those facts, so reuse the first verified
    # immutable source for the same score. A score correction remains a new row.
    result_id = digest(["football-result-v2", schedule["game_id"], source, home, away])
    with closing(evidence.connect(path)) as db, db:
        same_score = db.execute("""
            SELECT result_id FROM prospective_football_result
            WHERE game_id=? AND home_score=? AND away_score=?
            ORDER BY observed_at, result_id LIMIT 1
        """, (schedule["game_id"], home, away)).fetchone()
        if same_score:
            return _read(db, "prospective_football_result", "result_id", same_score[0]), False
        old = _read(db, "prospective_football_result", "result_id", result_id)
        if old is not None:
            return old, False
        row = dict(result_id=result_id, game_id=schedule["game_id"],
                   event_version_id=schedule["version_id"], sport=schedule["sport"],
                   result_source=source, result_source_event_id=schedule["provider_event_id"],
                   home_score=home, away_score=away, observed_at=iso(observed),
                   available_at=iso(observed), result_status="FINAL")
        _append(db, "prospective_football_result", "result_id", row, raw)
        return _read(db, "prospective_football_result", "result_id", result_id), True


def _spread_side(quote):
    # append_offers verified the exact provider home/away pair when the
    # immutable raw event was captured. A mascot suffix need not appear in
    # the canonical CFBD school name, so the stored pair is authoritative.
    provider_home = quote.get("provider_home_team")
    provider_away = quote.get("provider_away_team")
    if provider_home and provider_away and provider_home != provider_away:
        if quote["selection"] == provider_home:
            return "home"
        if quote["selection"] == provider_away:
            return "away"
        return None
    selected = _name(quote["sport"], quote["selection"])
    home = _name(quote["sport"], quote["home_team"])
    away = _name(quote["sport"], quote["away_team"])
    if selected == home and home != away:
        return "home"
    if selected == away and home != away:
        return "away"
    return None


def outcome(quote, result):
    if result["result_status"] != "FINAL":
        return "NEEDS_REVIEW"
    home, away, line = result["home_score"], result["away_score"], quote["line"]
    if quote["market_family"] == "SPREAD":
        side = _spread_side(quote)
        if side is None:
            return "NEEDS_REVIEW"
        is_home = side == "home"
        margin = (home - away if is_home else away - home) + line
        if margin == 0:
            return "PUSH"
        return ("HOME_COVER" if is_home else "AWAY_COVER") if margin > 0 else (
            "AWAY_COVER" if is_home else "HOME_COVER")
    if quote["market_family"] == "TOTAL":
        delta = home + away - line
        return "PUSH" if delta == 0 else ("OVER" if delta > 0 else "UNDER")
    return "NEEDS_REVIEW"


def settle_game(path, schedule, result, observed):
    created = 0
    with closing(evidence.connect(path)) as db, db:
        quotes = [_read(db, "prospective_football_quote", "quote_id", x[0])
                  for x in db.execute(
                      "SELECT quote_id FROM prospective_football_quote WHERE game_id=? AND quote_verified=1",
                      (schedule["game_id"],))]
        for quote in quotes:
            event_version = _read(db, "prospective_football_event", "version_id", quote["event_version_id"])
            quote["home_team"], quote["away_team"] = event_version["home_team"], event_version["away_team"]
            raw_event = json.loads(quote["raw_source"])
            quote["provider_home_team"], quote["provider_away_team"] = (
                raw_event.get("home_team"), raw_event.get("away_team"))
            result_kind = outcome(quote, result)
            if result_kind == "NEEDS_REVIEW":
                continue
            # Provider metadata can change without changing the final score.
            # Retain the new result evidence, but keep one settlement/label per
            # quote and score. A changed score is quarantined by the active
            # training view until the correction can be reviewed.
            same_score = db.execute("""
                SELECT 1 FROM prospective_football_settlement s
                JOIN prospective_football_result r ON r.result_id=s.result_id
                WHERE s.quote_id=? AND r.home_score=? AND r.away_score=? LIMIT 1
            """, (quote["quote_id"], result["home_score"], result["away_score"])).fetchone()
            if same_score:
                continue
            settlement_id = digest(["football-settlement-v1", quote["quote_id"], result["result_id"],
                                    result_kind])
            if _read(db, "prospective_football_settlement", "settlement_id", settlement_id):
                continue
            row = dict(settlement_id=settlement_id, quote_id=quote["quote_id"],
                       result_id=result["result_id"], game_id=schedule["game_id"],
                       sport=schedule["sport"], market_family=quote["market_family"],
                       selection=quote["selection"], line=quote["line"], outcome=result_kind,
                       settled_at=iso(observed), settlement_version=1)
            created += _append(db, "prospective_football_settlement", "settlement_id", row)
            training_id = digest(["football-training-v1", settlement_id])
            side = _spread_side(quote) if quote["market_family"] == "SPREAD" else None
            label = ("PUSH" if result_kind == "PUSH" else
                     "WIN" if (side == "home" and result_kind == "HOME_COVER") or
                              (side == "away" and result_kind == "AWAY_COVER") or
                              (quote["selection"] == "Over" and result_kind == "OVER") or
                              (quote["selection"] == "Under" and result_kind == "UNDER") else "LOSS")
            blockers = []
            if at(result["available_at"]) <= at(quote["observed_at"]):
                blockers.append("RESULT_NOT_AFTER_QUOTE")
            if at(quote["observed_at"]) >= at(event_version["scheduled_start"]):
                blockers.append("QUOTE_NOT_PREGAME")
            if quote["quote_verified"] != 1:
                blockers.append("QUOTE_UNVERIFIED")
            manifest = digest([event_version["source_hash"], quote["source_hash"],
                               result["source_hash"], settlement_id])
            _append(db, "prospective_football_training_row", "training_row_id",
                    dict(training_row_id=training_id, quote_id=quote["quote_id"],
                         result_id=result["result_id"], settlement_id=settlement_id,
                         game_id=schedule["game_id"], sport=schedule["sport"],
                         market_family=quote["market_family"], label=label,
                         training_row_status="TRAINING_BLOCKED" if blockers else "TRAINING_READY",
                         blockers=canonical(blockers),
                         available_for_training_at=result["available_at"],
                         source_manifest_hash=manifest))
    return created


def coverage(path, schedules, odds_events, *, sport, observed, run_id, target_policy=None,
             team_catalog=None, aliases=None):
    """One denominator row per scheduled game; odds cannot shrink this list."""
    if sport not in SPORT_KEYS:
        raise ValueError("UNSUPPORTED_SPORT")
    saved = []
    raw_by_row = []
    failures = Counter()
    seen = set()
    duplicates = 0
    pending = []
    for source in schedules:
        try:
            schedule, inserted = append_schedule(path, sport, source, observed, team_catalog)
        except (KeyError, TypeError, ValueError) as exc:
            reason = str(exc) if str(exc) in {"NO_PROVIDER_EVENT", "EVENT_IDENTITY_AMBIGUOUS"} else "SCHEDULE_INVALID"
            failures[reason] += 1
            saved.append({"provider_event_id": str(source.get("id", "")) if isinstance(source, dict) else None,
                          "status": reason, "regular_season_target": False, "source_hash": digest(source)})
            raw_by_row.append(source)
            continue
        if schedule["game_id"] in seen:
            duplicates += 1
            continue
        seen.add(schedule["game_id"])
        exclusion = ("NON_TARGET_SEASON" if schedule["season_type"].casefold() != "regular"
                     else target_policy(schedule) if target_policy else None)
        saved.append({"game_id": schedule["game_id"], "event_version_id": schedule["version_id"],
                      "provider_event_id": schedule["provider_event_id"],
                      "home": schedule["home_team"], "away": schedule["away_team"],
                      "kickoff": schedule["scheduled_start"], "season_type": schedule["season_type"],
                      "regular_season_target": not bool(exclusion),
                      "status": exclusion or "PENDING_ODDS_LOOKUP", "new_schedule": inserted})
        raw_by_row.append(source)
        pending.append((len(saved) - 1, schedule, exclusion))
    # The complete schedule population is now in the canonical store.
    for index, schedule, exclusion in pending:
        if exclusion:
            status = exclusion
        else:
            matches = list({(str(x.get("id")), digest(x)): x for x in odds_events
                            if isinstance(x, dict) and _same_identity(schedule, x, aliases)}.values())
            if len(matches) != 1:
                status = "NO_ODDS_EVENT" if not matches else "EVENT_IDENTITY_AMBIGUOUS"
            else:
                _, status = append_offers(path, schedule, matches[0], observed, run_id, aliases)
        with closing(evidence.connect(path)) as db:
            markets = {x[0] for x in db.execute(
                "SELECT DISTINCT market_family FROM prospective_football_quote WHERE game_id=? AND quote_verified=1",
                (schedule["game_id"],))}
        saved[index].update(spread_price_available="SPREAD" in markets,
                            total_price_available="TOTAL" in markets,
                            status=status or "QUOTE_CAPTURED")
    target = [x for x in saved if x.get("regular_season_target")]
    cycle_source = {"source_hashes": [digest(x) for x in raw_by_row],
                    "games": saved, "duplicate_schedule_events": duplicates,
                    "unparseable_schedule_events": dict(failures)}
    coverage_id = digest(["football-cycle-coverage-v1", run_id, sport])
    with closing(evidence.connect(path)) as db, db:
        _append(db, "prospective_football_cycle_coverage", "coverage_id", dict(
            coverage_id=coverage_id, capture_run_id=run_id, sport=sport,
            observed_at=iso(observed), target_games=len(target),
            requested_slate_success=int(not failures)), cycle_source)
    return {"sport": sport, "schedule_source_events": len(schedules),
            "scheduled_target_games": len(target), "persisted_events": len(saved),
            "duplicate_schedule_events": duplicates,
            "unparseable_schedule_events": dict(failures), "games": saved,
            "odds_response_events": len(odds_events),
            "requested_slate_success": not failures}


def provider_diagnostic(schedules, odds_events, *, sport, observed, aliases=None,
                        target_policy=None, schedule_window_start=None, schedule_window_end=None):
    """Explain every Odds API event against the independent schedule population."""
    rows, seen = [], set()
    for event in odds_events:
        event = event if isinstance(event, dict) else {}
        event_id = str(event.get("id") or "")
        classification, reason, candidates, delta = _provider_classification(
            event, schedules, sport=sport, aliases=aliases, target_policy=target_policy,
            schedule_window_start=schedule_window_start, schedule_window_end=schedule_window_end,
            duplicate=event_id in seen)
        try:
            start = iso(event["commence_time"])
            matches = [s for s in schedules if _same_identity(s, event, aliases)]
            target = [s for s in matches if s["season_type"].casefold() == "regular"]
            outside_window = bool(not matches and schedule_window_start and schedule_window_end and
                                  not at(schedule_window_start) <= at(start) < at(schedule_window_end))
            status = ("WRONG_SPORT_KEY" if event.get("sport_key") != SPORT_KEYS[sport] else
                      "DUPLICATE_PROVIDER_EVENT" if event_id in seen else
                      "OUTSIDE_SCHEDULE_WINDOW" if outside_window else
                      "NO_CANONICAL_MATCH" if not matches else
                      "AMBIGUOUS_CANONICAL_MATCH" if len(matches) != 1 else
                      "NON_TARGET_SEASON" if not target else
                      "SNAPSHOT_WINDOW_MISSED" if horizon(start, observed) == "SNAPSHOT_WINDOW_MISSED" else
                      "MATCHED")
            valid = _quotes(event, observed) if status == "MATCHED" else []
            markets = {MARKETS[x[1]["key"]] for x in valid}
            raw_markets = {m.get("key") for b in event.get("bookmakers", []) if isinstance(b, dict)
                           for m in b.get("markets", []) if isinstance(m, dict)}
            if status == "MATCHED" and not valid:
                status = "NO_VERIFIED_PRICE"
            rows.append({"provider_event_id": event_id, "provider_sport_key": event.get("sport_key"),
                         "home": event.get("home_team"), "away": event.get("away_team"),
                         "commence_time": start, "status": status,
                         "target_date_utc": at(start).date().isoformat(),
                         "target_week": matches[0]["week"] if len(matches) == 1 else None,
                         "regular_season_target": bool(target),
                         "canonical_match": matches[0]["game_id"] if len(matches) == 1 else None,
                         "spread_present": "spreads" in raw_markets,
                         "total_present": "totals" in raw_markets,
                         "spread_price_available": "SPREAD" in markets,
                         "total_price_available": "TOTAL" in markets,
                         "book_count": len({x[0] for x in valid}),
                         "pregame_valid": bool(valid), "persisted": False,
                         "exclusion_reason": None if status == "MATCHED" else status,
                         "classification": classification, "classification_reason": reason,
                         "candidate_game_ids": candidates, "kickoff_delta_seconds": delta})
        except (KeyError, TypeError, ValueError):
            rows.append({"provider_event_id": event_id, "provider_sport_key": event.get("sport_key"),
                         "status": "INVALID_PROVIDER_EVENT", "exclusion_reason": "INVALID_PROVIDER_EVENT",
                         "persisted": False, "classification": classification,
                         "classification_reason": reason, "candidate_game_ids": candidates,
                         "kickoff_delta_seconds": delta})
        seen.add(event_id)
    return rows


def ingest_theover(path, csv_path, schedules, observed):
    """Optional research rows; never creates a price or resolves an odds identity."""
    file_path = Path(csv_path)
    file_hash = digest(file_path.read_bytes())
    output = []
    with file_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        for number, row in enumerate(reader, 2):
            sport = str(row.get("League") or "").upper()
            home, away = row.get("HomeTeam"), row.get("AwayTeam")
            matches = [s for s in schedules if s["sport"] == sport and
                       _name(sport, home) == _name(sport, s["home_team"]) and
                       _name(sport, away) == _name(sport, s["away_team"])] if sport in SPORT_KEYS else []
            game_id = matches[0]["game_id"] if len(matches) == 1 else None
            status = "MATCHED_RESEARCH_ONLY" if game_id else "NO_UNIQUE_SCHEDULE_MATCH"
            def optional_number(value):
                try:
                    parsed = float(value)
                    return parsed if math.isfinite(parsed) else None
                except (ValueError, TypeError):
                    return None
            line = optional_number(row.get("Line"))
            market = "SPREAD" if row.get("Market", "").casefold() == "spread" else "TOTAL"
            if game_id and line is not None:
                with closing(evidence.connect(path)) as db:
                    lines = {abs(x[0]) if market == "SPREAD" else x[0] for x in db.execute(
                        "SELECT line FROM prospective_football_quote WHERE game_id=? AND market_family=?",
                        (game_id, market))}
                if lines and abs(line) not in lines:
                    status = "LINE_DISAGREEMENT"
            row_id = digest([file_hash, number, row])
            with closing(evidence.connect(path)) as db, db:
                if not _read(db, "prospective_football_theover", "research_row_id", row_id):
                    _append(db, "prospective_football_theover", "research_row_id", dict(
                        research_row_id=row_id, source_filename=file_path.name,
                        source_file_hash=file_hash, source_row_number=number,
                        ingested_at=iso(observed), sport=sport, matchup=row.get("Matchup"),
                        selection=row.get("Pick"), line=line,
                        win_probability=optional_number(row.get("WinProbability")),
                        model_hit_rate=optional_number(row.get("ModelHitRate")),
                        matched_game_id=game_id, match_status=status), row)
            output.append({"row_id": row_id, "source_row_number": number,
                           "matched_game_id": game_id, "status": status,
                           "win_probability": optional_number(row.get("WinProbability")),
                           "model_hit_rate": optional_number(row.get("ModelHitRate"))})
    return {"source_filename": file_path.name, "source_file_hash": file_hash,
            "rows": output}
