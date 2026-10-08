"""Versioned static prospective-input reader for the exact compatibility route.

Does not recalculate features/probabilities, collect, restore, register or infer.
It returns a verified fit only after explicit event mapping and source evidence.
Original labels, bytes and clocks remain in a private export. Neither a mapping
review nor compatibility populates the independently accepted source catalog.
"""
from copy import deepcopy
from datetime import timedelta
import math

from app_core import ncaaf_model_compatibility as model
from app_core import ncaaf_history as history, ncaaf_research as research
from app_core.ncaaf_research_contract import target_contract

VERSION = "ncaaf-compatible-prospective-inputs-v2"
# Separate reviewed exact-event receipts; no owner upload writes either catalog.
ACCEPTED_EVENT_MAPPINGS = {}
ACCEPTED_SOURCE_REVIEWS = {}


def _name(value):
    from app_core.ncaaf_identity import _key
    return _key(value)


def _canonical(value):
    key = _name(value)
    reviewed = model._binding()["reviewed_aliases"]
    # Explicit reviewed alias groups or exact source spelling; no fuzzy or
    # dynamic-alias fallback, including shared-city names from another sport.
    return reviewed.get(key, key)


def mapping_binding(event, schedule, crosswalk):
    return dict(event_sha256=model.digest(event), schedule_sha256=model.digest(schedule),
                crosswalk_sha256=model.digest(crosswalk),
                compatibility_binding_sha256=model.digest(model._binding()))


def check_mapping(event, schedule, crosswalk, review, *, as_of):
    require = model.require
    require(isinstance(event, dict) and isinstance(schedule, list) and schedule,
            "NCAAF_COMPAT_EVENT_MISSING")
    require(isinstance(crosswalk, list) and bool(crosswalk), "NCAAF_COMPAT_CROSSWALK_MISSING")
    names, ids = {}, {}
    for entry in crosswalk:
        require(set(entry) == {"provider_name", "schedule_name", "team_id"}
                and type(entry["team_id"]) is int and entry["team_id"] > 0,
                "NCAAF_COMPAT_CROSSWALK_CONFLICT")
        key = _canonical(entry["provider_name"])
        require(key and key == _canonical(entry["schedule_name"]), "NCAAF_COMPAT_ALIAS_UNREVIEWED")
        require(key not in ids or ids[key] == entry["team_id"], "NCAAF_COMPAT_ALIAS_COLLISION")
        require(_name(entry["provider_name"]) not in names, "NCAAF_COMPAT_ALIAS_COLLISION")
        ids[key] = entry["team_id"]
        names[_name(entry["provider_name"])] = entry
    require(event.get("sport") == "NCAAF" and event.get("provider_namespace") == "odds_api"
            and bool(event.get("provider_event_id")) and bool(event.get("canonical_event_id")),
            "NCAAF_COMPAT_EVENT_IDENTITY_MISSING")
    home, away = [names.get(_name(event.get(k))) for k in ("home_team", "away_team")]
    require(home is not None and away is not None and home["team_id"] != away["team_id"],
            "NCAAF_COMPAT_TEAM_MAPPING_UNRESOLVED")
    kickoff = history.timestamp(event.get("start_utc"))
    require(kickoff is not None, "NCAAF_COMPAT_EVENT_CLOCK_MISSING")
    matches = [g for g in schedule if g.get("homeId") == home["team_id"]
        and g.get("awayId") == away["team_id"] and g.get("homeTeam") == home["schedule_name"]
        and g.get("awayTeam") == away["schedule_name"]
        and history.timestamp(g.get("startDate")) == kickoff]
    require(len(matches) == 1, "NCAAF_COMPAT_EVENT_AMBIGUOUS_OR_ORIENTATION_CONFLICT")
    game = matches[0]
    require(type(game.get("id")) is int and type(event.get("schedule_game_id")) is int
            and game.get("id") == event.get("schedule_game_id")
            and event["canonical_event_id"] == "cfbd:" + str(game["id"])
            and game.get("completed") is False and game.get("startTimeTBD") is False
            and type(game.get("neutralSite")) is bool and type(event.get("neutral_site")) is bool
            and game["neutralSite"] == event["neutral_site"], "NCAAF_COMPAT_EVENT_FACT_CONFLICT")
    binding = mapping_binding(event, schedule, crosswalk)
    require(isinstance(review, dict) and review.get("binding") == binding,
            "NCAAF_COMPAT_EVENT_REVIEW_MISSING")
    reviewed = history.timestamp(review.get("reviewed_at"))
    require(reviewed is not None and reviewed <= as_of and bool(review.get("review_id"))
            and ACCEPTED_EVENT_MAPPINGS.get(review["review_id"]) == model.digest(review),
            "NCAAF_COMPAT_EVENT_REVIEW_NOT_ACCEPTED")
    return deepcopy(game)


def read_observation(packet):
    """Static checks only; original data/clock missingness is never repaired."""
    require = model.require
    require(isinstance(packet, dict) and set(packet) == {"payload", "sha256"}
            and model.digest(packet["payload"]) == packet["sha256"], "NCAAF_COMPAT_PACKET_INTEGRITY")
    p = packet["payload"]
    require(p.get("version") == VERSION and p.get("evidence_label") in {"RETAINED", "SYNTHETIC"},
            "NCAAF_COMPAT_OBSERVATION_SCHEMA")
    require(p.get("source_acceptance") is False and p.get("scientific_qualification") is False
            and p.get("probability_calibration") is False and p.get("wagering_authority") is False
            and p.get("wager_action") == "PASS" and p.get("live_stake") == 0,
            "NCAAF_COMPAT_AUTHORITY_FORBIDDEN")
    checked = model.load_model(p["model"])
    as_of = history.timestamp(p.get("as_of"))
    require(as_of is not None, "NCAAF_COMPAT_AS_OF_MISSING")
    game = check_mapping(p.get("event"), p.get("schedule"), p.get("crosswalk"),
        p.get("mapping_review"), as_of=as_of)
    require(type(game.get("season")) is int and game["season"] > research.PROTOCOL["evaluate"],
            "NCAAF_EVALUATED_HOLDOUT_CONTAMINATION")
    require(history.timestamp(checked["original_record"]["created_at"]) <= as_of,
            "NCAAF_FUTURE_MODEL")
    event, quote = p["event"], p.get("quote")
    require(isinstance(quote, dict), "NCAAF_COMPAT_QUOTE_MISSING")
    kind, line = quote.get("market_type"), quote.get("point")
    target = target_contract(kind, line)
    require(abs(line % 1) == .5, "NCAAF_INTEGER_PUSH_MODEL_UNVALIDATED")
    require(target["family"] == checked["compatibility"]["family"], "NCAAF_COMPAT_TARGET_CONFLICT")
    require(all(quote.get(k) == event.get(v) for k, v in (
        ("provider_event_id", "provider_event_id"), ("provider_namespace", "provider_namespace"),
        ("event_home_team", "home_team"), ("event_away_team", "away_team"), ("event_start_utc", "start_utc")))
        and type(quote.get("price")) in (int, float) and math.isfinite(quote["price"])
        and abs(quote["price"]) >= 100 and bool(quote.get("book")), "NCAAF_COMPAT_QUOTE_IDENTITY_CONFLICT")
    qt, start = history.timestamp(quote.get("recorded_at")), history.timestamp(event["start_utc"])
    require(qt is not None and 0 <= (as_of-qt).total_seconds() <= 900
            and as_of < start <= as_of + timedelta(days=7), "NCAAF_COMPAT_QUOTE_CLOCK_CONFLICT")
    require(quote.get("period") == "full_game" and quote.get("period_source")
            and quote.get("rules") and quote.get("rules_source"), "NCAAF_PERIOD_OR_SETTLEMENT_MISSING")
    features, receipts = p.get("features"), p.get("feature_dependencies")
    require(isinstance(features, dict) and set(features) == {"order", "values", "available_at"}
            and features["order"] == list(research.FEATURES)
            and len(features["values"]) == len(research.FEATURES)
            and all(type(v) in (int, float) and math.isfinite(v) for v in features["values"])
            and features["values"][-1] == int(event["neutral_site"]), "NCAAF_COMPAT_FEATURE_CONFLICT")
    available = history.timestamp(features.get("available_at"))
    require(available is not None and available <= as_of, "NCAAF_COMPAT_FEATURE_CLOCK_MISSING_OR_FUTURE")
    require(isinstance(receipts, list) and bool(receipts), "NCAAF_COMPAT_DEPENDENCIES_MISSING")
    ids = set()
    counts = {side: {"scoring": set(), "yardage": set()} for side in ("home", "away")}
    for r in receipts:
        require(set(r) == {"game_id", "team_id", "season", "start_utc", "available_at", "kind", "source_sha256"}
                and r["game_id"] not in {game["id"]} and r["season"] == game["season"]
                and r["kind"] in {"scoring", "yardage"}
                and r["team_id"] in {game["homeId"], game["awayId"]}, "NCAAF_COMPAT_DEPENDENCY_CONFLICT")
        key = (r["game_id"], r["team_id"], r["kind"])
        require(key not in ids, "NCAAF_COMPAT_DUPLICATE_DEPENDENCY")
        ids.add(key)
        t, a = history.timestamp(r["start_utc"]), history.timestamp(r["available_at"])
        require(t is not None and a is not None and t < start-timedelta(days=7)
                and t <= a <= available and 0 <= (as_of-a).total_seconds() <= 86400,
                "NCAAF_COMPAT_DEPENDENCY_CLOCK_CONFLICT")
        require(isinstance(r["source_sha256"], str) and len(r["source_sha256"]) == 64
                and all(c in "0123456789abcdef" for c in r["source_sha256"]), "NCAAF_COMPAT_DEPENDENCY_HASH_MISSING")
        counts["home" if r["team_id"] == game["homeId"] else "away"][r["kind"]].add(r["game_id"])
    require(all(len(games) >= research.PROTOCOL["minimum_prior_games"]
                for side in counts.values() for games in side.values()), "NCAAF_COMPAT_MINIMUM_HISTORY_MISSING")
    review = p.get("source_review")
    require(isinstance(review, dict) and review.get("quote_sha256") == model.digest(quote)
            and all(isinstance(review.get(k), str) and review[k].strip()
                    for k in ("review_id", "operator", "product", "listing_id", "rights_document"))
            and all(review[k] == quote.get(k) for k in ("operator", "product", "listing_id"))
            and review["operator"] == quote["book"]
            and review.get("settlement") == "full_game_including_overtime_binary_win_push_loss"
            and not quote["book"].lower().startswith("novig"), "NCAAF_COMPAT_SOURCE_REVIEW_MISSING_OR_CONFLICT")
    since, until, reviewed = [history.timestamp(review.get(k)) for k in
        ("effective_from", "effective_until", "reviewed_at")]
    require(all((since, until, reviewed)) and since <= qt and reviewed <= qt and as_of < until,
            "NCAAF_COMPAT_SOURCE_REVIEW_CLOCK_CONFLICT")
    require(ACCEPTED_SOURCE_REVIEWS.get(review["review_id"]) == model.digest(review),
            "NCAAF_SOURCE_REVIEW_NOT_ACCEPTED")
    return dict(status="COMPATIBLE_INPUT_READER", model=checked, target=target,
        fit=deepcopy(checked["artifact"]["models"]["ridge"]["margin" if target["family"] == "spread" else "total"]),
        ordered_features=deepcopy(features["values"]), original_packet=deepcopy(packet),
        probability=None, original_inference_time=p.get("original_inference_time"),
        feature_derivation_verified=False, dependency_objects_verified=False,
        source_acceptance=False, scientific_qualification=False, probability_calibration=False,
        wagering_authority=False, wager_action="PASS", live_stake=0)
