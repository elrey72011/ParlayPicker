"""Explicit prospective chronology v3. No acceptance is created by processing.

The v2 reader and original packets remain unchanged. Separate trusted catalogs
bind advance terms and an independently accepted, exact admission subject.
Static inspection never derives features, infers, collects or changes storage.
"""
from copy import deepcopy
from datetime import timedelta
import math
from app_core import ncaaf_compatible_observation as observation
from app_core import ncaaf_model_compatibility as model
from app_core import ncaaf_history as history, ncaaf_research as research
from app_core.ncaaf_research_contract import target_contract

VERSION = "ncaaf-compatible-prospective-inputs-v3"
REVIEW_VERSION = "ncaaf-prospective-admission-v1"
ACCEPTED_TERMS_REVIEWS = {}
ACCEPTED_ADMISSIONS = {}
DEPENDENCY_REVIEW_VERSION = "ncaaf-prospective-dependency-admission-v1"
DEPENDENCY_SUBJECT_VERSION = "ncaaf-prospective-dependency-subject-v1"
ACCEPTED_DEPENDENCY_PERMISSIONS = {}
ACCEPTED_DEPENDENCY_ADMISSIONS = {}
REASONS = frozenset("""NCAAF_CHRONOLOGY_SCHEMA NCAAF_QUOTE_CLOCK_MEANING_UNAVAILABLE
NCAAF_QUOTE_OBSERVATION_MISSING_OR_CONFLICT NCAAF_ADVANCE_TERMS_NOT_ACCEPTED
NCAAF_ADVANCE_TERMS_IDENTITY_CONFLICT NCAAF_ADVANCE_PERMISSIONS_UNAVAILABLE
NCAAF_TERMS_NOT_EFFECTIVE NCAAF_ADVANCE_REVIEW_CLOCK_CONFLICT
NCAAF_EXACT_OFFER_VERIFICATION_CONFLICT NCAAF_EXACT_OFFER_CLOCK_CONFLICT
NCAAF_INDEPENDENT_ACCEPTANCE_CONFLICT NCAAF_INDEPENDENT_ACCEPTANCE_NOT_TRUSTED
NCAAF_ACCEPTANCE_CLOCK_CONFLICT NCAAF_ADMISSION_SUBJECT_CONFLICT
NCAAF_DEPENDENCY_ADVANCE_PERMISSION_CLOCK_CONFLICT NCAAF_ADMISSION_SUBJECT_FUTURE_INPUT
NCAAF_DEPENDENCY_ADMISSION_SCHEMA NCAAF_DEPENDENCY_PERMISSIONS_NOT_TRUSTED
NCAAF_DEPENDENCY_PERMISSION_SCOPE_CONFLICT NCAAF_DEPENDENCY_PERMISSION_UNAVAILABLE
NCAAF_DEPENDENCY_CLOCK_MEANING_UNAVAILABLE NCAAF_DEPENDENCY_TERMS_NOT_EFFECTIVE
NCAAF_DEPENDENCY_VERIFICATION_CONFLICT NCAAF_DEPENDENCY_VERIFICATION_CLOCK_CONFLICT
NCAAF_DEPENDENCY_SUBJECT_FUTURE_FACT
NCAAF_DEPENDENCY_ACCEPTANCE_CONFLICT NCAAF_DEPENDENCY_ACCEPTANCE_CLOCK_CONFLICT
NCAAF_DEPENDENCY_ACCEPTANCE_NOT_TRUSTED""".split())
require = model.require


def dependency_subject_hash(packet):
    """Hash only facts available before admission; never rewrite the packet.

    The later offer-acceptance receipt and actual admission checkpoint cannot
    be known by the earlier dependency verifier. Final packet/catalog binding
    separately covers those clocks after admission. All original event, offer,
    model, feature and exact native-byte facts remain in this subject.
    """
    p = deepcopy(packet["payload"])
    original = p["observation"]["payload"]
    original["source_review"].pop("acceptance", None)
    original["source_review"].pop("owner_verification", None)
    original.pop("as_of", None)
    return model.digest(dict(version=DEPENDENCY_SUBJECT_VERSION,
        input_version=p["version"], evidence_label=p["evidence_label"],
        observation=original, dependency_objects=p["dependency_objects"]))


def check_dependency_admission(packet, review, index, inference_at, *, review_policy=None):
    """Separate immutable advance permission from later exact native-byte review.

    Only the explicit successor uses this contract. The permission contains no
    future byte hashes; later verification binds it to the exact input packet
    and captured objects through a pre-admission subject. Independent catalogs are read-only and empty by
    default. Processing never creates a review or acceptance.
    """
    require(isinstance(review, dict) and set(review) == {
        "version", "permissions_review", "dependency_verification",
        "owner_verification" if review_policy is not None else "acceptance"}
        and review["version"] == (review_policy.DEPENDENCY_VERSION if review_policy is not None else DEPENDENCY_REVIEW_VERSION), "NCAAF_DEPENDENCY_ADMISSION_SCHEMA")
    permission, verified, accepted = [review[k] for k in
        ("permissions_review", "dependency_verification", "owner_verification" if review_policy is not None else "acceptance")]
    p = packet["payload"]["observation"]["payload"]
    require(isinstance(permission, dict) and set(permission) == {
        "review_id", "reviewer", "reviewed_at", "provider", "endpoints", "season",
        "permitted_uses", "rights_edition", "rights_document_sha256", "effective_from",
        "effective_until", "capture_clock_field", "capture_clock_meaning"},
        "NCAAF_DEPENDENCY_PERMISSION_SCOPE_CONFLICT")
    require(permission["provider"] == "cfbd" and permission["endpoints"] == ["games", "games/teams"]
        and type(permission["season"]) is int and permission["season"] == p["schedule"][0]["season"]
        and all(isinstance(permission[k], str) and bool(permission[k].strip())
                for k in ("review_id", "reviewer", "rights_edition", "rights_document_sha256"))
        and len(permission["rights_document_sha256"]) == 64
        and all(c in "0123456789abcdef" for c in permission["rights_document_sha256"]),
        "NCAAF_DEPENDENCY_PERMISSION_SCOPE_CONFLICT")
    require(permission["permitted_uses"] == ["collection", "private_retention",
        "prospective_research_features", "private_derived_output" if review_policy is not None else "public_derived_output"], "NCAAF_DEPENDENCY_PERMISSION_UNAVAILABLE")
    require(permission["capture_clock_field"] == "retrieved_at"
        and permission["capture_clock_meaning"] == "local_native_batch_capture",
        "NCAAF_DEPENDENCY_CLOCK_MEANING_UNAVAILABLE")
    require((review_policy.permission(permission) if review_policy is not None else
        ACCEPTED_DEPENDENCY_PERMISSIONS.get(permission["review_id"]) == model.digest(permission)),
        "NCAAF_DEPENDENCY_PERMISSIONS_NOT_TRUSTED")
    require(isinstance(verified, dict) and set(verified) == {
        "verified_at", "verifier", "subject_version", "subject_sha256", "permissions_sha256", "dependency_hashes"}
        and isinstance(verified["verifier"], str) and bool(verified["verifier"].strip())
        and verified["subject_version"] == DEPENDENCY_SUBJECT_VERSION
        and verified["subject_sha256"] == dependency_subject_hash(packet)
        and verified["permissions_sha256"] == model.digest(permission)
        and verified["dependency_hashes"] == [v["sha256"] for v in packet["payload"]["dependency_objects"]]
        and set(verified["dependency_hashes"]) == set(index), "NCAAF_DEPENDENCY_VERIFICATION_CONFLICT")
    require(isinstance(accepted, dict) and set(accepted) == {
        "review_id", "reviewer", "reviewed_at" if review_policy is not None else "accepted_at", "subject_version", "subject_sha256", "verification_sha256"}
        and all(isinstance(v, str) and bool(v.strip()) for v in accepted.values())
        and (accepted["reviewer"] == verified["verifier"] == review_policy.OWNER if review_policy is not None else accepted["reviewer"] != verified["verifier"])
        and accepted["subject_version"] == DEPENDENCY_SUBJECT_VERSION
        and accepted["subject_sha256"] == verified["subject_sha256"]
        and accepted["verification_sha256"] == model.digest(verified), "NCAAF_DEPENDENCY_ACCEPTANCE_CONFLICT")
    require((review_policy.dependency(accepted) if review_policy is not None else
        ACCEPTED_DEPENDENCY_ADMISSIONS.get(accepted["review_id"]) == model.digest(accepted)),
        "NCAAF_DEPENDENCY_ACCEPTANCE_NOT_TRUSTED")
    advance, start, end, vt, accepted_at, quote_accepted_at, inference = [history.timestamp(v) for v in
        (permission["reviewed_at"], permission["effective_from"], permission["effective_until"],
         verified["verified_at"], accepted["reviewed_at" if review_policy is not None else "accepted_at"],
         p["source_review"]["owner_verification" if review_policy is not None else "acceptance"]["reviewed_at" if review_policy is not None else "accepted_at"], inference_at)]
    captured = [history.timestamp(batch["retrieved_at"]) for batch in index.values()]
    require(advance is not None and bool(captured) and all(captured)
        and all(advance < clock for clock in captured), "NCAAF_DEPENDENCY_ADVANCE_PERMISSION_CLOCK_CONFLICT")
    require(all((start, end, inference)) and all(start <= clock < end for clock in captured)
        and advance < end and inference < end, "NCAAF_DEPENDENCY_TERMS_NOT_EFFECTIVE")
    require(vt is not None and all(clock < vt for clock in captured), "NCAAF_DEPENDENCY_VERIFICATION_CLOCK_CONFLICT")
    # This subject also hashes observation, features and the exact-offer review.
    # Captured bytes alone cannot make those later facts exist at verification.
    # Acceptance/checkpoint are deliberately excluded from the subject; target
    # kickoff and terms' effective end are future applicability, not availability.
    checked_model = model.load_model(p["model"])
    subject_clocks = [history.timestamp(v) for v in (
        p["quote"].get("recorded_at"), p["mapping_review"].get("reviewed_at"),
        p["features"].get("available_at"), p["source_review"]["terms_review"].get("reviewed_at"),
        p["source_review"]["quote_observation"].get("observed_at"),
        p["source_review"]["offer_verification"].get("verified_at"),
        checked_model["original_record"]["created_at"], checked_model["predecessor_record"]["created_at"])]
    subject_clocks.extend(history.timestamp(r.get("available_at")) for r in p["feature_dependencies"])
    require(all(subject_clocks) and all(clock <= vt for clock in subject_clocks),
        "NCAAF_DEPENDENCY_SUBJECT_FUTURE_FACT")
    require(all((accepted_at, quote_accepted_at, inference))
        and vt <= accepted_at <= quote_accepted_at < inference, "NCAAF_DEPENDENCY_ACCEPTANCE_CLOCK_CONFLICT")
    return deepcopy(review)


def subject_hash(payload):
    """No circular hash: all original admission facts except the acceptance.

    Version, immutable model, mapping, ordered features, dependency references
    and all earlier receipts are covered. A v2 packet cannot borrow admission.
    """
    subject = deepcopy(payload)
    subject["source_review"].pop("acceptance", None)
    subject["source_review"].pop("owner_verification", None)
    return model.digest(subject)


def check_chronology(p, inference_at, *, review_policy=None):
    q, review = p["quote"], p.get("source_review")
    require(isinstance(review, dict) and set(review) == {"version", "terms_review", "quote_observation", "offer_verification", "owner_verification" if review_policy is not None else "acceptance"}
            and review["version"] == (review_policy.OFFER_VERSION if review_policy is not None else REVIEW_VERSION), "NCAAF_CHRONOLOGY_SCHEMA")
    terms, observed, verified, accepted = [review[k] for k in
        ("terms_review", "quote_observation", "offer_verification", "owner_verification" if review_policy is not None else "acceptance")]
    require(isinstance(terms, dict) and set(terms) == set("review_id reviewer reviewed_at effective_from effective_until operator product listing_id period settlement rule_edition rules_document_sha256 rights_document_sha256 permitted_uses provider_clock_field provider_clock_meaning".split()),
            "NCAAF_CHRONOLOGY_SCHEMA")
    require(all(isinstance(terms[k], str) and terms[k].strip() for k in terms if k != "permitted_uses"), "NCAAF_CHRONOLOGY_SCHEMA")
    require(all(len(terms[k]) == 64 and all(c in "0123456789abcdef" for c in terms[k])
        for k in ("rules_document_sha256", "rights_document_sha256")), "NCAAF_CHRONOLOGY_SCHEMA")
    require((review_policy.terms(terms) if review_policy is not None else
        ACCEPTED_TERMS_REVIEWS.get(terms["review_id"]) == model.digest(terms)), "NCAAF_ADVANCE_TERMS_NOT_ACCEPTED")
    require(all(terms[k] == q.get(k) for k in ("operator", "product", "listing_id", "period"))
        and terms["operator"] == q["book"] and not q["book"].lower().startswith("novig")
        and terms["period"] == "full_game" and q["rules"] == terms["settlement"]
        and terms["settlement"] == "full_game_including_overtime_binary_win_push_loss",
        "NCAAF_ADVANCE_TERMS_IDENTITY_CONFLICT")
    require(terms["permitted_uses"] == ["collection", "private_retention", "research", "private_derived_output" if review_policy is not None else "public_derived_output"],
            "NCAAF_ADVANCE_PERMISSIONS_UNAVAILABLE")
    require(terms["provider_clock_field"] == "recorded_at"
        and terms["provider_clock_meaning"] == "provider_market_last_update", "NCAAF_QUOTE_CLOCK_MEANING_UNAVAILABLE")
    require(isinstance(observed, dict) and set(observed) == {"observed_at", "quote_sha256", "provider_clock_field", "provider_clock_meaning"}
        and observed["quote_sha256"] == model.digest(q), "NCAAF_QUOTE_OBSERVATION_MISSING_OR_CONFLICT")
    require(all(observed[k] == terms[k] for k in ("provider_clock_field", "provider_clock_meaning")),
            "NCAAF_QUOTE_CLOCK_MEANING_UNAVAILABLE")
    require(isinstance(verified, dict) and set(verified) == {"verified_at", "verifier", "quote_sha256", "observation_sha256", "terms_sha256", "mapping_sha256", "rule_edition"}
        and isinstance(verified["verifier"], str) and bool(verified["verifier"].strip())
        and verified["quote_sha256"] == model.digest(q) and verified["observation_sha256"] == model.digest(observed)
        and verified["terms_sha256"] == model.digest(terms) and verified["mapping_sha256"] == model.digest(p["mapping_review"])
        and verified["rule_edition"] == terms["rule_edition"], "NCAAF_EXACT_OFFER_VERIFICATION_CONFLICT")
    require(isinstance(accepted, dict) and set(accepted) == {"review_id", "reviewer", "reviewed_at" if review_policy is not None else "accepted_at", "subject_version", "subject_sha256"}
        and all(isinstance(v, str) and bool(v.strip()) for v in accepted.values())
        and (accepted["reviewer"] == verified["verifier"] == review_policy.OWNER if review_policy is not None else accepted["reviewer"] != verified["verifier"]), "NCAAF_INDEPENDENT_ACCEPTANCE_CONFLICT")
    require(accepted["subject_version"] == (review_policy.OBSERVATION_VERSION if review_policy is not None else VERSION) and accepted["subject_sha256"] == subject_hash(p), "NCAAF_ADMISSION_SUBJECT_CONFLICT")
    require((review_policy.offer(accepted) if review_policy is not None else
        ACCEPTED_ADMISSIONS.get(accepted["review_id"]) == model.digest(accepted)), "NCAAF_INDEPENDENT_ACCEPTANCE_NOT_TRUSTED")
    qt, ot, vt, at, rt, start, end, checkpoint, inference, mapped = [history.timestamp(v) for v in
        (q.get("recorded_at"), observed["observed_at"], verified["verified_at"], accepted["reviewed_at" if review_policy is not None else "accepted_at"], terms["reviewed_at"],
         terms["effective_from"], terms["effective_until"], p["as_of"], inference_at, p["mapping_review"].get("reviewed_at"))]
    require(ot is not None and qt is not None and qt <= ot, "NCAAF_QUOTE_OBSERVATION_MISSING_OR_CONFLICT")
    require(rt is not None and rt < ot, "NCAAF_ADVANCE_REVIEW_CLOCK_CONFLICT")
    require(vt is not None and ot < vt, "NCAAF_EXACT_OFFER_CLOCK_CONFLICT")
    require(all((at, checkpoint, inference, mapped)) and vt <= at <= checkpoint <= inference and mapped <= at,
            "NCAAF_ACCEPTANCE_CLOCK_CONFLICT")
    input_clocks = [history.timestamp(p["features"]["available_at"])] + [history.timestamp(r["available_at"]) for r in p["feature_dependencies"]]
    require(all(input_clocks) and all(clock <= at for clock in input_clocks), "NCAAF_ADMISSION_SUBJECT_FUTURE_INPUT")
    require(start is not None and end is not None and start <= qt and rt < end and inference < end,
            "NCAAF_TERMS_NOT_EFFECTIVE")
    # Exact offer freshness and pregame checks are ALSO repeated by the caller
    # at its actual inference clock; no receipt is substituted for that clock.
    require(0 <= (inference-qt).total_seconds() <= 900 and inference < history.timestamp(p["event"]["start_utc"]),
            "NCAAF_COMPAT_QUOTE_CLOCK_CONFLICT")
    return deepcopy(review)


def read_observation(packet, *, review_policy=None):
    """Static checks only; original data/clock missingness is never repaired."""
    require = model.require
    require(isinstance(packet, dict) and set(packet) == {"payload", "sha256"}
            and model.digest(packet["payload"]) == packet["sha256"], "NCAAF_COMPAT_PACKET_INTEGRITY")
    p = packet["payload"]
    require(p.get("version") == (review_policy.OBSERVATION_VERSION if review_policy is not None else VERSION) and p.get("evidence_label") in {"RETAINED", "SYNTHETIC"},
            "NCAAF_COMPAT_OBSERVATION_SCHEMA")
    require(p.get("source_acceptance") is False and p.get("scientific_qualification") is False
            and p.get("probability_calibration") is False and p.get("wagering_authority") is False
            and p.get("wager_action") == "PASS" and p.get("live_stake") == 0,
            "NCAAF_COMPAT_AUTHORITY_FORBIDDEN")
    checked = model.load_model(p["model"])
    as_of = history.timestamp(p.get("as_of"))
    require(as_of is not None, "NCAAF_COMPAT_AS_OF_MISSING")
    game = (observation.check_mapping if review_policy is None else review_policy.check_mapping)(p.get("event"), p.get("schedule"), p.get("crosswalk"),
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
    check_chronology(p, p["as_of"], review_policy=review_policy)
    return dict(status="COMPATIBLE_INPUT_READER", model=checked, target=target,
        fit=deepcopy(checked["artifact"]["models"]["ridge"]["margin" if target["family"] == "spread" else "total"]),
        ordered_features=deepcopy(features["values"]), original_packet=deepcopy(packet),
        probability=None, original_inference_time=p.get("original_inference_time"),
        feature_derivation_verified=False, dependency_objects_verified=False,
        source_acceptance=False, scientific_qualification=False, probability_calibration=False,
        wagering_authority=False, wager_action="PASS", live_stake=0)
