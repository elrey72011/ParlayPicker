"""Versioned research-only provider facts; no qualification or authority reader."""
from __future__ import annotations
import hashlib
import json
import re

VERSION = "research-producer-v2"
QUOTE_VERSION = "provider-offer-facts-v1"
DERIVED_NAMESPACE = "parlaypicker:offer:v1"
TRANSFER_FIELDS = ("quote_id", "market_period", "settlement_rules", "prediction_generated_at",
                   "game_start_utc", "provider_namespace", "provider_event_id")


def text(value):
    return value.strip() if isinstance(value, str) else ""


def number(value):
    from app_core.research_display import _number
    return _number(value)


def clock(value):
    from app_core.research_display import _time
    return _time(value)


def team(value, league):
    from core.team_mapper import normalize_team_name
    value = text(value)
    compact = re.sub(r"[^a-z]", "", value.lower())
    if not compact or (league == "MLB" and compact in {"la", "losangeles", "newyork", "ny", "chicago"}):
        return ""
    return normalize_team_name(value)


def quote_facts(game, book, market, outcome):
    """Copy only actual transport facts, with their source paths. No rule defaults.

    A provider event ID is never used as a provider quote ID. Standard market
    names alone do not establish period or a bookmaker's settlement rules.
    """
    return dict(provenance_version=QUOTE_VERSION, event_home_team=game.get("home_team"),
        event_away_team=game.get("away_team"), event_start_utc=game.get("commence_time"),
        provider_quote_id=outcome.get("quote_id"),
        period=market.get("period"), period_source="market.period" if text(market.get("period")) else "",
        rules=market.get("settlement_rules"),
        rules_source="market.settlement_rules" if text(market.get("settlement_rules")) else "")


def _quotes(source):
    try:
        quotes = json.loads(source.get("provider_quotes") or "[]")
        return quotes if isinstance(quotes, list) else []
    except (ValueError, TypeError):
        return []


def _line(source):
    return number(source.get("total_line" if text(source.get("market_type")).startswith("total") else "spread_line"))


def _matches(source):
    from app_core.prediction_evidence import matching_quotes
    return [q for q in matching_quotes(source) if q.get("provenance_version") == QUOTE_VERSION]


def _offer(source, generated_at):
    league = text(source.get("league") or source.get("League")).upper()
    matches = _matches(source)
    q = matches[0] if len(matches) == 1 else {}
    kind = text(source.get("market_type"))
    side = kind.rsplit("_", 1)[-1]
    event = dict(provider_namespace=text(q.get("provider_namespace")),
        provider_event_id=text(q.get("provider_event_id")), home=team(q.get("event_home_team"), league),
        away=team(q.get("event_away_team"), league), start=clock(q.get("event_start_utc")), sport=league)
    offer = dict(book=text(q.get("book")), market=kind, side=side, line=_line(source),
        price=number(source.get("odds_american")), source_time=clock(q.get("recorded_at")),
        period=text(q.get("period")), rules=text(q.get("rules")),
        period_source=text(q.get("period_source")), rules_source=text(q.get("rules_source")))
    provider_id = text(q.get("provider_quote_id"))
    identified = all(event.values()) and offer["book"] and offer["source_time"] and offer["line"] is not None and offer["price"] is not None
    raw = json.dumps(dict(event=event, offer=offer), sort_keys=True, separators=(",", ":"), allow_nan=False)
    offer.update(quote_id=provider_id if provider_id else (
        DERIVED_NAMESPACE + ":" + hashlib.sha256(raw.encode()).hexdigest() if identified else ""),
        quote_namespace=event["provider_namespace"] if provider_id else DERIVED_NAMESPACE,
        quote_kind="provider_issued" if provider_id else "locally_derived")
    return dict(version=VERSION, event=event, offer=offer, inference_time=generated_at, target_period="full_game",
        matchup_key_semantics="unordered_team_pair_et_day", matched_offer_count=len(matches))


def record(source, result, metadata, generated_at):
    """Used at the inference boundary, only for explicitly versioned transport.

    Legacy inputs retain their original metadata/interpretation. Supplied aliases
    remain sticky, so a conflicting row cannot be repaired by copying the quote.
    """
    if not any(isinstance(q, dict) and q.get("provenance_version") == QUOTE_VERSION for q in _quotes(source)):
        return metadata, {}
    item = json.loads(metadata)
    contract = _offer(source, generated_at)
    event, offer = contract["event"], contract["offer"]
    supplied = dict(quote_id=offer["quote_id"], market_period=offer["period"], settlement_rules=offer["rules"],
        prediction_generated_at=generated_at, game_start_utc=event["start"],
        provider_namespace=event["provider_namespace"], provider_event_id=event["provider_event_id"])
    fields = {k: source[k] if text(source.get(k)) and k != "prediction_generated_at" else v for k, v in supplied.items()}
    from app_core.research_estimate_trace import fact
    for field in fields:
        if field in item["identity"]:
            item["identity"][field] = fact(fields[field])
    item.update(version=2, producer_contract=contract)
    return json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False), fields


def diagnose(source, item):
    """Exact owner diagnostics. An unordered key never supplies orientation."""
    missing, conflicts = [], []
    contract = item.get("producer_contract")
    try:
        if (not isinstance(contract, dict) or set(contract) != {"version", "event", "offer", "inference_time", "target_period", "matchup_key_semantics", "matched_offer_count"}
            or contract["version"] != VERSION or contract["matchup_key_semantics"] != "unordered_team_pair_et_day"):
            raise ValueError("producer_contract.schema")
        event, offer = contract["event"], contract["offer"]
        if contract["target_period"] != "full_game" or (offer["period"] and offer["period"] != contract["target_period"]):
            conflicts.append("offer.period_target")
        if set(event) != {"provider_namespace", "provider_event_id", "home", "away", "start", "sport"} or set(offer) != {
            "book", "market", "side", "line", "price", "source_time", "period", "rules", "period_source", "rules_source", "quote_id", "quote_namespace", "quote_kind"}:
            raise ValueError("producer_contract.schema")
        if _offer(source, contract["inference_time"]) != contract:
            conflicts.append("provider_quotes.exact_offer")
        if type(contract["matched_offer_count"]) is not int or contract["matched_offer_count"] != 1:
            conflicts.append("producer_contract.matched_offer_count")
        for field, value in {**{"event."+k:v for k,v in event.items()}, **{"offer."+k:v for k,v in offer.items()},
                             "inference_time":contract["inference_time"]}.items():
            if value is None or value == "": missing.append(field)
        if clock(contract["inference_time"]) != clock(item["generated_at"]):
            conflicts.append("prediction_generated_at")
        if not text(source.get("prediction_generated_at")):
            missing.append("prediction_generated_at")
        elif clock(source.get("prediction_generated_at")) != clock(contract["inference_time"]):
            conflicts.append("prediction_generated_at")
        league = text(source.get("league")).upper()
        for side in ("home", "away"):
            if team(source.get(side+"_team"), league) != event[side] or not event[side]:
                conflicts.append(side+"_team")
        if event["home"] == event["away"]: conflicts.append("event.orientation")
        kind = text(source.get("market_type"))
        if (event["sport"] != league or offer["market"] != kind or offer["side"] != kind.rsplit("_", 1)[-1]
            or offer["line"] != _line(source) or offer["price"] != number(source.get("odds_american"))):
            conflicts.append("offer.selection_line_price")
        pick = text(source.get("best_pick"))
        if pick:
            match = re.fullmatch(r"(.+?)\s+([+-]?\d+(?:\.\d+)?)", pick)
            selected = event.get(offer["side"], offer["side"])
            if (not match or number(match[2]) != offer["line"] or
                (team(match[1], league) if offer["side"] in {"home", "away"} else match[1].lower()) != selected):
                conflicts.append("best_pick")
        aliases = dict(provider_namespace=event["provider_namespace"], provider_event_id=event["provider_event_id"],
            quote_id=offer["quote_id"], market_period=offer["period"], period=offer["period"], settlement_rules=offer["rules"])
        for field, expected in aliases.items():
            current = text(source.get(field))
            if current and current != expected: conflicts.append(field)
            elif field != "period" and not current: missing.append(field)
        for field, expected in {"game_start_utc":event["start"], "odds_recorded_at":offer["source_time"],
                                "quote_time":offer["source_time"], "selected_quote_recorded_at":offer["source_time"]}.items():
            if expected and text(source.get(field)) and clock(source[field]) != expected: conflicts.append(field)
        for field in ("quote_bookmaker", "opposing_odds_source", "sportsbook"):
            if text(source.get(field)) and text(source[field]).lower() != offer["book"].lower(): conflicts.append(field)
        if not clock(event["start"]) or not clock(offer["source_time"]) or not clock(contract["inference_time"]):
            missing.append("valid_original_clocks")
        if offer["quote_kind"] not in {"provider_issued", "locally_derived"}:
            conflicts.append("offer.quote_kind")
        if offer["quote_kind"] == "locally_derived":
            basis = {k:v for k,v in offer.items() if k not in {"quote_id", "quote_namespace", "quote_kind"}}
            raw = json.dumps(dict(event=event, offer=basis), sort_keys=True, separators=(",", ":"), allow_nan=False)
            if offer["quote_namespace"] != DERIVED_NAMESPACE or (offer["quote_id"] and offer["quote_id"] != DERIVED_NAMESPACE+":"+hashlib.sha256(raw.encode()).hexdigest()):
                conflicts.append("offer.derived_identity")
        elif offer["quote_namespace"] != event["provider_namespace"] or offer["quote_id"].startswith(DERIVED_NAMESPACE+":"):
            conflicts.append("offer.provider_namespace")
        # Validate an event label only as a pair/day, never as ordered home/away.
        key = text(source.get("matchup_id"))
        parts = key.split("|")
        if len(parts) != 3:
            conflicts.append("matchup_id")
        else:
            day = next((p for p in parts if re.fullmatch(r"\d{4}-\d{2}-\d{2}", p)), "")
            names = [team(p, league) for p in parts if p != day]
            from pandas import Timestamp
            expected_day = Timestamp(event["start"]).tz_convert("America/New_York").date().isoformat() if event["start"] else ""
            if len(names) != 2 or set(names) != {event["home"], event["away"]} or (expected_day and day != expected_day):
                conflicts.append("matchup_id")
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        conflicts.append(str(exc) if str(exc).startswith("producer_contract") else "producer_contract.schema")
    reason = ("TARGET_MISMATCH" if conflicts == ["offer.period_target"] else "ESTIMATE_IDENTITY_MISMATCH") if conflicts else "ESTIMATE_PROVENANCE_NOT_RECORDED" if missing else None
    return dict(stage="producer_contract", reason=reason, missing_fields=sorted(set(missing)), conflicting_fields=sorted(set(conflicts)))
