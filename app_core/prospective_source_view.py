"""Read-only six-sport projection of immutable research source stores.

A source observation is descriptive. It cannot be inserted as a production
prediction merely because its provider quote and score are replayable.
"""
from __future__ import annotations

import math
from pathlib import Path

from app_core.prospective_legacy_view import _records, _row, _time, legacy_evidence
from app_core.odds_research_adapter import (
    PROTOCOL, digest, identity, participant_ids, quotes,
)

NEW_SPORTS = frozenset({"NBA", "NCAAB", "NHL"})


def _score(value):
    if isinstance(value, bool):
        raise ValueError("invalid provider score")
    number = float(value)
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        raise ValueError("invalid provider score")
    return int(number)


def _source_bound(source, *, expected=None):
    if not isinstance(source, dict) or source.get("source_hash") != digest(source.get("raw_source")):
        return False
    observed = _time(source.get("observed_at"))
    if observed is None:
        return False
    if expected is not None:
        try:
            if identity(expected["sport"], source["raw_source"]) != expected["identity"]:
                return False
        except (KeyError, TypeError, ValueError):
            return False
    return True


def _new_sport_evidence(sport, path):
    records = _records(path, "records")
    for _, record in records:
        if record.get("data", {}).get("sport") != sport:
            raise ValueError("research source sport mismatch")
    score_versions = {}
    close_rows = {}
    for key, record in records:
        data = record.get("data", {})
        if data.get("protocol") != PROTOCOL:
            continue
        if record.get("kind") == "scores":
            event = data.get("event", {})
            event_id = str(event.get("event_id"))
            score_versions.setdefault(event_id, []).append((key, event))
        elif record.get("kind") == "pregame_close_candidate":
            for event in data.get("events", []):
                close_rows.setdefault(str(event.get("event_id")), []).append((key, event))
    score_rows = {}
    for event_id, revisions in score_versions.items():
        # Remote object stores can restore immutable hashes in any order.
        # Follow the declared revision chain, not SQLite insertion order.
        for key, event in sorted(revisions, key=lambda item: (
                item[1].get("grading_version") if type(item[1].get("grading_version")) is int else -1,
                item[0])):
            previous = score_rows.get(event_id)
            version = event.get("grading_version")
            if (type(version) is not int or version < 1 or
                    (previous is None and (version != 1 or
                                           event.get("revises_score_record_id") is not None)) or
                    (previous is not None and (
                        version != previous[1]["grading_version"] + 1 or
                        event.get("revises_score_record_id") != previous[0] or
                        _time(event.get("available_at")) is None or
                        _time(previous[1].get("available_at")) is None or
                        _time(event["available_at"]) <=
                        _time(previous[1]["available_at"])))):
                raise ValueError("sport research score revision chain invalid")
            score_rows[event_id] = (key, event)
    result = []
    for source_id, record in records:
        data = record.get("data", {})
        if record.get("kind") != "capture" or data.get("protocol") != PROTOCOL:
            continue
        participants = data.get("participants_source", {})
        participants_bound = _source_bound(participants)
        try:
            ids = participant_ids(participants["raw_source"]) if participants_bound else {}
        except (KeyError, TypeError, ValueError):
            participants_bound, ids = False, {}
        for event in data.get("events", []):
            event_id = str(event.get("event_id"))
            selected = {key: event.get(key) for key in ("event_id", "home", "away", "start")}
            expected = {"sport": sport, "identity": selected}
            discovery_bound = _source_bound(event.get("discovery_source"), expected=expected)
            odds_bound = (event.get("source_hash") == digest(event.get("raw_source")))
            try:
                odds_bound = odds_bound and identity(sport, event["raw_source"]) == selected
                observed = _time(event.get("response_received_at"))
                odds_bound = odds_bound and observed is not None and observed < _time(event.get("start"))
                accepted, _ = quotes(sport, event["raw_source"], observed) if odds_bound else ([], {})
            except (KeyError, TypeError, ValueError, OverflowError):
                odds_bound, accepted = False, []
            identity_bound = (participants_bound and discovery_bound and odds_bound
                              and bool(event.get("home_team_id")) and bool(event.get("away_team_id"))
                              and ids.get(event.get("home")) == event.get("home_team_id")
                              and ids.get(event.get("away")) == event.get("away_team_id")
                              and _time(participants.get("observed_at")) < _time(event.get("start"))
                              and _time(event["discovery_source"].get("observed_at")) < _time(event.get("start"))
                              and event.get("home_team_id") != event.get("away_team_id"))
            score_record = score_rows.get(event_id)
            score = score_record[1] if score_record else None
            score_bound = False
            if score is not None:
                try:
                    score_bound = (score.get("source_hash") == digest(score.get("raw_source"))
                                   and score.get("event_id") == event.get("event_id")
                                   and score.get("home") == event.get("home")
                                   and score.get("away") == event.get("away")
                                   and identity(sport, score["raw_source"])["event_id"] == event.get("event_id")
                                   and identity(sport, score["raw_source"])["home"] == event.get("home")
                                   and identity(sport, score["raw_source"])["away"] == event.get("away")
                                   and abs((_time(identity(sport, score["raw_source"])["start"])
                                            - _time(event.get("start"))).total_seconds()) <= 900
                                   and len(score["raw_source"]["scores"]) == 2
                                   and {v.get("name"): _score(v.get("score")) for v in score["raw_source"]["scores"]}
                                       == {event.get("home"): score.get("home_score"),
                                           event.get("away"): score.get("away_score")}
                                   and _time(score.get("available_at")) >= _time(event.get("start")))
                except (KeyError, TypeError, ValueError, OverflowError):
                    score_bound = False
            for quote in event.get("quotes", []):
                if quote.get("market_family") not in (("PUCK_LINE", "TOTAL") if sport == "NHL" else ("SPREAD", "TOTAL")):
                    continue
                same = ("market_family", "selection", "line", "american_odds",
                        "decimal_odds", "sportsbook", "quote_timestamp", "source_hash")
                quote_bound = odds_bound and any(all(candidate.get(k) == quote.get(k) for k in same)
                                                 for candidate in accepted)
                row = _row(sport, quote["market_family"], source_id, quote.get("selection"),
                    game_id=event.get("event_id"), provider_namespace="THE_ODDS_API",
                    provider_event_id=event.get("provider_event_id"),
                    home_team=event.get("home"), away_team=event.get("away"),
                    home_team_id=event.get("home_team_id") if identity_bound else None,
                    away_team_id=event.get("away_team_id") if identity_bound else None,
                    scheduled_start=event.get("start"), line=quote.get("line"),
                    american_odds=quote.get("american_odds"),
                    decimal_odds=quote.get("decimal_odds"),
                    sportsbook=quote.get("sportsbook"),
                    quote_timestamp=quote.get("quote_timestamp"),
                    quote_source=quote.get("quote_source"),
                    quote_verified=bool(quote_bound and quote.get("quote_verified") is True),
                    evidence_snapshot_id=source_id, evidence_hash=event.get("source_hash"),
                    result_outcome="NEEDS_REVIEW" if score_bound else None,
                    result_available_at=score.get("available_at") if score_bound else None,
                    result_source=score.get("result_source") if score_bound else None,
                    result_home_score=score.get("home_score") if score_bound else None,
                    result_away_score=score.get("away_score") if score_bound else None)
                row["source_store"] = "native_odds_research"
                row["identity_verified"] = bool(identity_bound)
                row["discovery_source_verified"] = bool(discovery_bound)
                row["close_candidate_count"] = len(close_rows.get(event_id, []))
                row["close_status"] = "UNVERIFIED_CLOSE_CANDIDATE" if row["close_candidate_count"] else "NO_VALID_CLOSE_QUOTES"
                row["blockers"] = tuple(dict.fromkeys((*row["blockers"],
                    *((["MISSING_REPLAYABLE_TEAM_IDENTITY"] if not identity_bound else [])),
                    "MISSING_VERIFIED_CLOSE", "MISSING_MARKET_SETTLEMENT_RULES")))
                result.append(row)
    return result


def source_evidence(sport, path):
    """Return a canonical, research-only view without modifying any source DB."""
    if sport in {"MLB", "NCAAF", "NFL"}:
        return legacy_evidence(sport, path)
    if sport in NEW_SPORTS:
        return _new_sport_evidence(sport, path)
    raise ValueError("unsupported prospective sport")


SOURCE_FILENAMES = {
    "NFL": "nfl-market.sqlite3",
    "NCAAF": "ncaaf-prospective.sqlite3",
    "NBA": "nba-market.sqlite3",
    "NCAAB": "ncaab-market.sqlite3",
    "MLB": "mlb-pregame-receipts.sqlite3",
    "NHL": "nhl-market.sqlite3",
}


def all_source_readiness(directory=None):
    """Read descriptive local capture counts separately from validation authority.

    Sources are never created or copied into the canonical evidence database.
    A missing local file does not claim its remote backup is empty.
    """
    from app_core.prediction_evidence import database_path
    from app_core.prospective_evidence import SPORT_MARKETS

    root = Path(directory) if directory is not None else database_path().parent
    report = []
    for sport, markets in SPORT_MARKETS.items():
        path = root / SOURCE_FILENAMES[sport]
        source_present = path.is_file()
        records = source_evidence(sport, path)
        for market in markets:
            rows = [row for row in records if row["market_family"] == market]
            events = {str(row["game_id"]) for row in rows if row.get("game_id") is not None}
            scored = {str(row["game_id"]) for row in rows
                      if row.get("game_id") is not None and row.get("result_outcome") is not None}
            settled = {str(row["game_id"]) for row in rows
                       if row.get("game_id") is not None and
                       row.get("result_outcome") in {"WIN", "LOSS", "PUSH", "VOID"}}
            report.append({
                "sport": sport, "market_family": market,
                "source_store": SOURCE_FILENAMES[sport],
                "local_source_status": "PRESENT" if source_present else "ABSENT",
                "research_quote_rows": len(rows),
                "research_captured_events": len(events),
                "research_score_observed_events": len(scored),
                "research_settled_events": len(settled),
                "replay_verified_quote_rows": sum(row.get("quote_verified") is True for row in rows),
                "research_only": True, "production_eligible": False,
                "recommended_stake": 0.0,
            })
    return report
