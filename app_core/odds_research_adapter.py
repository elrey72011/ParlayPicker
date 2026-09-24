"""Provider-bound NBA/NCAAB/NHL market research, with no wager authority.

The Odds API supplies event IDs, bookmaker lines and participant IDs.  Raw
responses are retained in sport-isolated immutable records.  No model,
calibration, prediction or verified sportsbook close is inferred here.
"""

from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import time

import requests

from app_core.ncaaf_history import timestamp
from app_core.odds_market_store import for_sport
from app_core.quote_freshness import QUOTE_MAX_AGE_SECONDS
from app_core.research_api_budget import BudgetLimit


SPORT_KEYS = {"NBA": "basketball_nba", "NCAAB": "basketball_ncaab", "NHL": "icehockey_nhl"}
SIDE_MARKETS = {"NBA": "SPREAD", "NCAAB": "SPREAD", "NHL": "PUCK_LINE"}
BASE = "https://api.the-odds-api.com/v4/sports/"
PROTOCOL = "odds-sport-research-v1"


def utcnow():
    return datetime.now(timezone.utc)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _number(value):
    if isinstance(value, bool):
        raise ValueError("invalid_provider_number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("invalid_provider_number")
    return number


def _american(value):
    number = _number(value)
    if number != int(number) or abs(number) < 100:
        raise ValueError("invalid_american_odds")
    return int(number)


def _decimal(american):
    return 1.0 + (american / 100.0 if american > 0 else 100.0 / abs(american))


def _text(value):
    return value if isinstance(value, str) and value.strip() and value == value.strip() else None


class ProviderError(ValueError):
    """Static, sanitized provider status; never carries response bodies or URLs."""


def fetch(sport, endpoint, key, request_get, *, params=None):
    if sport not in SPORT_KEYS or endpoint not in {"events", "participants", "odds", "scores"}:
        raise ValueError("unsupported_provider_endpoint")
    if not key:
        raise ProviderError("missing_provider_key")
    url = BASE + SPORT_KEYS[sport] + "/" + endpoint
    arguments = {"apiKey": key, **({"dateFormat": "iso"} if endpoint != "participants" else {}), **(params or {})}
    for attempt in range(3):
        try:
            response = request_get(url, params=arguments, timeout=15, allow_redirects=False)
        except (requests.Timeout, requests.ConnectionError):
            if attempt == 2:
                raise ProviderError("provider_transient_exhausted") from None
            time.sleep(attempt + 1)
            continue
        if response.status_code in (401, 403):
            raise ProviderError("provider_auth_blocked")
        if response.status_code in (400, 404, 422):
            raise ProviderError("provider_endpoint_or_request_blocked")
        if response.status_code == 429 or 500 <= response.status_code <= 599:
            if attempt == 2:
                raise ProviderError("provider_transient_exhausted")
            time.sleep(attempt + 1)
            continue
        if response.status_code != 200:
            raise ProviderError("provider_response_blocked")
        try:
            payload = response.json()
        except (ValueError, TypeError):
            raise ProviderError("provider_invalid_json") from None
        if not isinstance(payload, list):
            raise ProviderError("provider_invalid_schema")
        return payload
    raise ProviderError("provider_transient_exhausted")


def identity(sport, event):
    if not isinstance(event, dict) or event.get("sport_key") != SPORT_KEYS[sport]:
        raise ValueError("provider_event_identity")
    event_id, home, away = (event.get(k) for k in ("id", "home_team", "away_team"))
    if not all(_text(v) for v in (event_id, home, away)) or home == away:
        raise ValueError("provider_event_identity")
    start = timestamp(event.get("commence_time"))
    if start is None:
        raise ValueError("provider_event_time")
    return {"event_id": event_id, "home": home, "away": away, "start": start.isoformat()}


def participant_ids(rows):
    """Exact provider whitelist. Ambiguous names are absent, never fuzzy mapped."""
    names = {}
    conflicted = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("provider_participants_schema")
        name, ident = row.get("full_name"), row.get("id")
        if not _text(name) or not _text(ident):
            raise ValueError("provider_participants_schema")
        if name in names and names[name] != ident:
            conflicted.add(name)
        names[name] = ident
    for name in conflicted:
        names.pop(name)
    if len(set(names.values())) != len(names):
        raise ValueError("provider_participants_identity_conflict")
    return names


def quotes(sport, event, observed):
    """Only complete, fresh, exact-book two-sided spread/total pairs."""
    accepted, rejected = [], Counter()
    books = set()
    for book in event.get("bookmakers", []):
        if not isinstance(book, dict) or not _text(book.get("key")) or book["key"] in books:
            raise ValueError("provider_duplicate_book")
        books.add(book["key"])
        markets = set()
        for market in book.get("markets", []):
            if not isinstance(market, dict) or market.get("key") not in ("spreads", "totals"):
                continue
            kind = market["key"]
            if kind in markets:
                raise ValueError("provider_duplicate_market")
            markets.add(kind)
            at = timestamp(market.get("last_update"))
            if at is None or not 0 <= (observed - at).total_seconds() <= QUOTE_MAX_AGE_SECONDS or at >= timestamp(event["commence_time"]):
                rejected["missing_future_or_stale_market_timestamp"] += 1
                continue
            try:
                outcomes = market["outcomes"]
                if not isinstance(outcomes, list) or len(outcomes) != 2:
                    raise ValueError("incomplete_pair")
                names = [o["name"] for o in outcomes]
                expected = {event["home_team"], event["away_team"]} if kind == "spreads" else {"Over", "Under"}
                if set(names) != expected:
                    raise ValueError("wrong_selection_pair")
                lines = [_number(o["point"]) for o in outcomes]
                prices = [_american(o["price"]) for o in outcomes]
                if kind == "spreads" and abs(sum(lines)) > 1e-9:
                    raise ValueError("unpaired_spread")
                if kind == "totals" and (lines[0] != lines[1] or lines[0] <= 0):
                    raise ValueError("unpaired_total")
                for name, line, price in zip(names, lines, prices):
                    accepted.append({"market_family": SIDE_MARKETS[sport] if kind == "spreads" else "TOTAL",
                                     "selection": name, "line": line, "american_odds": price,
                                     "decimal_odds": _decimal(price), "sportsbook": book["key"],
                                     "quote_timestamp": at.isoformat(), "quote_source": "THE_ODDS_API",
                                     "quote_verified": True,
                                     "source_id": f"{event['id']}:{book['key']}:{kind}:{name}:{at.isoformat()}",
                                     "source_hash": digest(market), "raw_source": market})
            except (KeyError, TypeError, ValueError, OverflowError):
                rejected["invalid_market_pair"] += 1
    return accepted, dict(rejected)


def captures(records):
    result = {}
    for record in records:
        if record["kind"] == "capture" and record["data"].get("protocol") == PROTOCOL:
            for event in record["data"].get("events", []):
                result.setdefault(event["event_id"], event)
    return result


def scores(records):
    result = {}
    for record in records:
        if record["kind"] == "scores" and record["data"].get("protocol") == PROTOCOL:
            event = record["data"]["event"]
            result[event["event_id"]] = {**event, "record_id": record["id"]}
    return result


class OddsResearchAdapter:
    supported_markets = ()

    def __init__(self, sport):
        if sport not in SPORT_KEYS:
            raise ValueError("unsupported_research_sport")
        self.sport = sport
        self.supported_markets = (SIDE_MARKETS[sport], "TOTAL")
        self.store = for_sport(sport)
        self.path_name = sport.lower() + "-market.sqlite3"

    def restore(self, path, client, folder, session):
        return self.store.sync(path, client=client, folder=folder, session=session)

    def backup(self, path, client, folder, session):
        return self.store.sync(path, client=client, folder=folder, session=session)

    def upcoming_events(self, odds_key, request_get):
        rows = fetch(self.sport, "events", odds_key, request_get)
        result = {}
        for row in rows:
            ident = identity(self.sport, row)
            if ident["event_id"] in result and result[ident["event_id"]] != row:
                raise ValueError("provider_duplicate_event")
            result[ident["event_id"]] = row
        return result

    def capture_pregame(self, path, state, odds_key, request_get, backup, *, upcoming=None):
        records = self.store.records(path)
        existing = captures(records)
        upcoming = self.upcoming_events(odds_key, request_get) if upcoming is None else upcoming
        now = utcnow()
        due = {gid: event for gid, event in upcoming.items() if gid not in existing and
               now < timestamp(event["commence_time"]) <= now + timedelta(hours=2)}
        result = {"discovered": len(upcoming), "due": len(due), "captured": 0,
                  "captured_quote_rows": 0, "identity_blocked": 0,
                  "price_unavailable_events": 0, "deferred_due_events": max(0, len(due) - 24),
                  "excluded_markets": {}}
        if not due:
            return result
        participant_response = fetch(self.sport, "participants", odds_key, request_get)
        participant_observed = utcnow().isoformat()
        participants = participant_ids(participant_response)
        attempts = state.setdefault(self.sport.lower() + "_capture_attempts", {})
        if not isinstance(attempts, dict):
            raise ValueError("invalid_capture_attempt_state")
        allowed = sorted(due, key=lambda gid: (attempts.get(gid, ""),
                                                due[gid]["commence_time"], gid))[:24]
        for gid in allowed:
            attempts[gid] = utcnow().isoformat()
        payload = fetch(self.sport, "odds", odds_key, request_get,
                        params={"regions": "us", "markets": "spreads,totals", "oddsFormat": "american",
                                "eventIds": ",".join(allowed)})
        accepted, rejected = [], Counter()
        seen = set()
        for row in payload:
            ident = identity(self.sport, row)
            gid = ident["event_id"]
            if gid not in due:
                continue
            if gid in seen:
                raise ValueError("provider_duplicate_event")
            seen.add(gid)
            if ident != identity(self.sport, due[gid]) or not utcnow() < timestamp(ident["start"]):
                rejected["event_identity_or_start_changed"] += 1
                continue
            observed = utcnow()
            valid, exclusions = quotes(self.sport, row, observed)
            rejected.update(exclusions)
            home_id, away_id = participants.get(ident["home"]), participants.get(ident["away"])
            if not home_id or not away_id or home_id == away_id:
                result["identity_blocked"] += 1
                valid = [{**quote, "quote_verified": False} for quote in valid]
            if not valid:
                continue
            accepted.append({**ident, "provider_namespace": "THE_ODDS_API",
                             "provider_event_id": gid, "home_team_id": home_id,
                             "away_team_id": away_id, "quotes": valid,
                             "discovery_source": {"source_id": gid,
                                                  "source_hash": digest(due[gid]),
                                                  "observed_at": now.isoformat(),
                                                  "raw_source": due[gid]},
                             "response_received_at": observed.isoformat(),
                             "source_id": gid, "source_hash": digest(row), "raw_source": row,
                             "model_id": None, "calibration_id": None,
                             "prediction_timestamp": None, "feature_snapshot_id": None,
                             "production_eligible": False})
        if accepted:
            self.store.save("capture", {"sport": self.sport, "protocol": PROTOCOL,
                                        "participants_source": {
                                            "source_id": SPORT_KEYS[self.sport] + ":participants",
                                            "source_hash": digest(participant_response),
                                            "observed_at": participant_observed,
                                            "raw_source": participant_response},
                                        "events": accepted}, path)
            result["captured"] = len(accepted)
            result["captured_quote_rows"] = sum(len(event["quotes"]) for event in accepted)
            backup()
        result["price_unavailable_events"] = len(allowed) - len(accepted)
        result["excluded_markets"] = dict(rejected)
        return result

    def capture_closes(self, path, odds_key, request_get, backup):
        records = self.store.records(path)
        closed = {event["event_id"] for record in records if record["kind"] == "pregame_close_candidate"
                  for event in record["data"].get("events", [])}
        now = utcnow()
        near = {gid: event for gid, event in captures(records).items() if gid not in closed and
                now < timestamp(event["start"]) <= now + timedelta(minutes=40)}
        result = {"close_candidates": 0, "verified_closes": 0,
                  "close_blocker": "NO_VERIFIED_CLOSE_QUOTES",
                  "deferred_close_events": max(0, len(near) - 24),
                  "close_identity_rejections": 0,
                  "close_price_unavailable_events": 0}
        if not near:
            return result
        payload = fetch(self.sport, "odds", odds_key, request_get,
                        params={"regions": "us", "markets": "spreads,totals", "oddsFormat": "american",
                                "eventIds": ",".join(sorted(near)[:24])})
        accepted = []
        seen = set()
        for row in payload:
            ident = identity(self.sport, row)
            gid = ident["event_id"]
            if gid not in near:
                continue
            if gid in seen:
                raise ValueError("provider_duplicate_event")
            seen.add(gid)
            if ident != {k: near[gid][k] for k in ("event_id", "home", "away", "start")}:
                result["close_identity_rejections"] += 1
                continue
            observed = utcnow()
            if observed >= timestamp(ident["start"]):
                continue
            valid, _ = quotes(self.sport, row, observed)
            if valid:
                accepted.append({**ident, "quotes": valid, "response_received_at": observed.isoformat(),
                                 "source_id": gid, "source_hash": digest(row), "raw_source": row,
                                 "verified_close": False})
        if accepted:
            self.store.save("pregame_close_candidate", {"sport": self.sport, "protocol": PROTOCOL,
                                                         "events": accepted}, path)
            result["close_candidates"] = len(accepted)
            backup()
        result["close_price_unavailable_events"] = len(sorted(near)[:24]) - len(accepted)
        return result

    def grade(self, path, odds_key, request_get, backup, state=None):
        records = self.store.records(path)
        captured, completed = captures(records), scores(records)
        state = {} if state is None else state
        attempts = state.setdefault(self.sport.lower() + "_score_check_attempts", {})
        if not isinstance(attempts, dict):
            raise ValueError("invalid_score_attempt_state")
        now = utcnow()
        pending = {gid: event for gid, event in captured.items() if
                   timedelta(hours=3) <= now - timestamp(event["start"]) <= timedelta(days=3) and
                   (timestamp(attempts.get(gid)) is None or
                    now - timestamp(attempts[gid]) >=
                    (timedelta(hours=6) if gid in completed else timedelta(minutes=30)))}
        result = {"graded": 0, "corrected": 0,
                  "pending_grade": sum(gid not in completed for gid in pending),
                  "correction_monitoring": sum(gid in completed for gid in pending),
                  "deferred_grade_events": max(0, len(pending) - 24),
                  "score_rejections": {}}
        if not pending:
            return result
        selected = sorted(pending, key=lambda gid: (attempts.get(gid, ""), gid))[:24]
        for gid in selected:
            attempts[gid] = now.isoformat()
        payload = fetch(self.sport, "scores", odds_key, request_get,
                        params={"daysFrom": 3, "eventIds": ",".join(selected)})
        seen = set()
        for row in payload:
            ident = identity(self.sport, row)
            gid = ident["event_id"]
            if gid not in selected:
                continue
            if gid in seen:
                raise ValueError("provider_duplicate_score")
            seen.add(gid)
            if row.get("completed") is not True:
                continue
            captured_event = pending[gid]
            original_start = timestamp(captured_event["start"])
            reported_start = timestamp(ident["start"])
            if any(ident[k] != captured_event[k] for k in ("event_id", "home", "away")) or abs(reported_start-original_start) > timedelta(minutes=15):
                result["score_rejections"][gid] = "score_event_identity_changed"
                continue
            updated = timestamp(row.get("last_update"))
            observed = utcnow()
            values = row.get("scores")
            if (updated is None or not original_start <= updated <= observed or
                    not isinstance(values, list) or len(values) != 2 or
                    {v.get("name") for v in values} != {ident["home"], ident["away"]}):
                result["score_rejections"][gid] = "score_source_invalid"
                continue
            try:
                points = {v["name"]: _number(v["score"]) for v in values}
                if any(v < 0 or v != int(v) for v in points.values()):
                    raise ValueError("invalid_score")
            except (KeyError, TypeError, ValueError, OverflowError):
                result["score_rejections"][gid] = "score_source_invalid"
                continue
            previous = completed.get(gid)
            if previous is not None:
                if (int(points[ident["home"]]), int(points[ident["away"]])) == (
                        previous["home_score"], previous["away_score"]):
                    continue
                if (updated <= timestamp(previous["provider_updated_at"]) or
                        observed <= timestamp(previous["available_at"])):
                    result["score_rejections"][gid] = "score_correction_timestamp_not_advanced"
                    continue
            revision = (previous.get("grading_version") if previous and
                        type(previous.get("grading_version")) is int else 1)
            self.store.save("scores", {"sport": self.sport, "protocol": PROTOCOL,
                                       "event": {**ident, "start": captured_event["start"],
                                                 "reported_start": ident["start"],
                                                 "home_score": int(points[ident["home"]]),
                                                 "away_score": int(points[ident["away"]]),
                                                 "result_source": "THE_ODDS_API",
                                                 "result_source_id": gid,
                                                 "provider_updated_at": updated.isoformat(),
                                                 "observed_at": observed.isoformat(),
                                                 "available_at": observed.isoformat(),
                                                 "grading_version": revision + 1 if previous else 1,
                                                 "revises_score_record_id": previous["record_id"] if previous else None,
                                                 "source_hash": digest(row), "raw_source": row}}, path)
            result["corrected" if previous else "graded"] += 1
            backup()
        return result

    def audit(self, path):
        records = self.store.records(path)
        captured, graded = captures(records), scores(records)
        by_market = Counter(quote["market_family"] for event in captured.values() for quote in event["quotes"])
        return {"captured_events": len(captured), "settled_events": len(set(captured) & set(graded)),
                "quote_rows_by_market": dict(by_market), "production_eligible": False,
                "unresolved_past_score_window": sum(
                    1 for gid, event in captured.items() if gid not in graded and
                    utcnow() - timestamp(event["start"]) > timedelta(days=3)),
                "model_status": "MISSING_SPORT_MARKET_MODEL",
                "calibration_status": "MISSING_CALIBRATION",
                "close_status": "NO_VERIFIED_CLOSE_QUOTES"}

    def health(self, path):
        return self.audit(path)

    def run_cycle(self, path, state, cfbd_key, odds_key, backup, budget,
                  after_capture=None):
        if not odds_key:
            raise ValueError("missing_provider_keys")
        result = {"mode": "market_tracking_only", "captured": 0, "graded": 0,
                  "errors": [], "blockers": ["MISSING_SPORT_MARKET_MODEL", "MISSING_CALIBRATION",
                                            "NO_VERIFIED_CLOSE_QUOTES"], "budget_paused": False}
        try:
            graded = self.grade(path, odds_key, budget.request, backup, state)
            result.update(graded)
            result["grade_status"] = ("partial" if graded["score_rejections"] or
                                      graded["deferred_grade_events"] else "success")
            if graded["score_rejections"]:
                result["blockers"].append("SCORE_SOURCE_IDENTITY_OR_TIMING_REJECTED")
            if graded["deferred_grade_events"]:
                result["blockers"].append("DUE_SCORE_BATCH_LIMIT")
        except BudgetLimit:
            result["budget_paused"] = True
            result["grade_status"] = "budget_paused"
        except ProviderError as exc:
            result["errors"].append(str(exc))
            result["grade_status"] = "failed"
        except Exception:
            result["errors"].append("grading_failed")
            result["grade_status"] = "failed"
        try:
            captured = self.capture_pregame(path, state, odds_key, budget.request, backup)
            result.update(captured)
            result["capture_status"] = ("partial" if any(captured[k] for k in
                ("identity_blocked", "price_unavailable_events", "deferred_due_events")) else "success")
            if captured["identity_blocked"]:
                result["blockers"].append("MISSING_VERIFIED_TEAM_IDS")
            if captured["price_unavailable_events"]:
                result["blockers"].append("MISSING_PREGAME_PRICES")
            if captured["deferred_due_events"]:
                result["blockers"].append("DUE_EVENT_BATCH_LIMIT")
        except BudgetLimit:
            result["budget_paused"] = True
            result["capture_status"] = "budget_paused"
        except ProviderError as exc:
            result["errors"].append(str(exc))
            result["capture_status"] = "failed"
        except Exception:
            result["errors"].append("capture_failed")
            result["capture_status"] = "failed"
        # Canonical quote insertion has a strict live freshness window. Freeze
        # research evidence immediately after capture, before close polling.
        if after_capture is not None:
            try:
                result["model_cycle"] = after_capture()
                result["predictions"] = result["model_cycle"]["predictions"]
            except Exception:
                result["errors"].append("research_model_cycle_failed")
        try:
            result.update(self.capture_closes(path, odds_key, budget.request, backup))
            result["close_capture_status"] = "partial" if any(result[k] for k in
                ("deferred_close_events", "close_identity_rejections",
                 "close_price_unavailable_events")) else "success"
            if result["deferred_close_events"]:
                result["blockers"].append("CLOSE_CANDIDATE_BATCH_LIMIT")
            if result["close_identity_rejections"]:
                result["blockers"].append("CLOSE_EVENT_IDENTITY_CHANGED")
            if result["close_price_unavailable_events"]:
                result["blockers"].append("CLOSE_PRICE_UNAVAILABLE")
        except BudgetLimit:
            result["budget_paused"] = True
            result["close_capture_status"] = "budget_paused"
        except ProviderError as exc:
            result["errors"].append(str(exc))
            result["close_capture_status"] = "failed"
        except Exception:
            result["errors"].append("close_capture_failed")
            result["close_capture_status"] = "failed"
        result["audit"] = self.audit(path)
        if result["audit"]["unresolved_past_score_window"]:
            result["blockers"].append("SCORE_UNAVAILABLE_AFTER_PROVIDER_WINDOW")
            if result["grade_status"] == "success":
                result["grade_status"] = "partial"
        return result
