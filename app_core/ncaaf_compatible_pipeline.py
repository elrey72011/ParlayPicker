"""Explicit NEW prospective inference for the reviewed recovered-model route.

No connector, persistence initializer, fitting, registration or authority. The
immutable v2 observation is nested unchanged; original native dependency bytes
are retained. Static display never derives features or repeats inference.
"""
import base64
from copy import deepcopy
from datetime import timedelta
import hashlib
import json
import math

from app_core import ncaaf_compatible_observation as observation
from app_core import ncaaf_prospective_chronology as chronology
from app_core import ncaaf_model_compatibility as model
from app_core import ncaaf_research_contract as contract
from app_core import ncaaf_history as history, ncaaf_research as research

VERSION = "ncaaf-compatible-normal-inputs-v1"
RESULT_VERSION = "ncaaf-compatible-normal-result-v1"
SUCCESSOR_VERSION = "ncaaf-compatible-normal-inputs-v2"
SUCCESSOR_RESULT_VERSION = "ncaaf-compatible-normal-result-v2"
INPUT_VERSIONS = {VERSION, SUCCESSOR_VERSION}
RESULT_VERSIONS = {RESULT_VERSION, SUCCESSOR_RESULT_VERSION}
MAX_OBJECTS = 64
MAX_OBJECT_BYTES = 512 * 1024
REASONS = frozenset("""NCAAF_COMPAT_DEPENDENCY_BYTES_MISSING NCAAF_COMPAT_DEPENDENCY_BYTES_CORRUPT
NCAAF_COMPAT_DEPENDENCY_SCHEMA NCAAF_COMPAT_DEPENDENCY_IDENTITY_CONFLICT
NCAAF_COMPAT_DEPENDENCY_REFERENCE_CONFLICT NCAAF_COMPAT_FEATURE_DERIVATION_CONFLICT
NCAAF_COMPAT_DEPENDENCY_SOURCE_REVIEW_MISSING_OR_CONFLICT NCAAF_COMPAT_DEPENDENCY_RIGHTS_UNAVAILABLE
NCAAF_COMPAT_NEW_INFERENCE_CLOCK_CONFLICT NCAAF_COMPAT_RECORDED_INFERENCE_FORBIDDEN
NCAAF_COMPAT_COMPUTATION_RECEIPT_CONFLICT NCAAF_COMPAT_OBSERVATION_SCHEMA
NCAAF_COMPAT_PACKET_INTEGRITY NCAAF_COMPAT_EVENT_MISSING NCAAF_COMPAT_CROSSWALK_MISSING
NCAAF_COMPAT_CROSSWALK_CONFLICT NCAAF_COMPAT_ALIAS_UNREVIEWED NCAAF_COMPAT_ALIAS_COLLISION
NCAAF_COMPAT_EVENT_IDENTITY_MISSING NCAAF_COMPAT_TEAM_MAPPING_UNRESOLVED
NCAAF_COMPAT_EVENT_CLOCK_MISSING NCAAF_COMPAT_EVENT_AMBIGUOUS_OR_ORIENTATION_CONFLICT
NCAAF_COMPAT_EVENT_FACT_CONFLICT NCAAF_COMPAT_EVENT_REVIEW_MISSING NCAAF_COMPAT_EVENT_REVIEW_NOT_ACCEPTED
NCAAF_COMPAT_AS_OF_MISSING NCAAF_FUTURE_MODEL NCAAF_COMPAT_QUOTE_MISSING
NCAAF_COMPAT_TARGET_CONFLICT NCAAF_COMPAT_QUOTE_IDENTITY_CONFLICT NCAAF_COMPAT_QUOTE_CLOCK_CONFLICT
NCAAF_COMPAT_FEATURE_CONFLICT NCAAF_COMPAT_FEATURE_CLOCK_MISSING_OR_FUTURE
NCAAF_COMPAT_DEPENDENCIES_MISSING NCAAF_COMPAT_DEPENDENCY_CONFLICT NCAAF_COMPAT_DUPLICATE_DEPENDENCY
NCAAF_COMPAT_DEPENDENCY_CLOCK_CONFLICT NCAAF_COMPAT_DEPENDENCY_HASH_MISSING
NCAAF_COMPAT_MINIMUM_HISTORY_MISSING NCAAF_COMPAT_SOURCE_REVIEW_MISSING_OR_CONFLICT
NCAAF_COMPAT_SOURCE_REVIEW_CLOCK_CONFLICT NCAAF_COMPAT_AUTHORITY_FORBIDDEN
NCAAF_COMPAT_RECORD_HASH NCAAF_COMPAT_ENVELOPE_CONFLICT NCAAF_COMPAT_RUNTIME_COMPONENT_CHANGED
NCAAF_COMPAT_RUNTIME_BINDING_CONFLICT NCAAF_COMPAT_DYNAMIC_ALIASES_UNREVIEWED
NCAAF_COMPAT_ALIAS_TABLE_CHANGED NCAAF_UNSUPPORTED_TARGET NCAAF_INVALID_LINE
NCAAF_INPUT_SCHEMA NCAAF_DUPLICATE_DEPENDENCY NCAAF_FUTURE_OR_STALE_DEPENDENCY NCAAF_INPUT_NOT_ALLOWLISTED
NCAAF_ORIGINAL_INPUTS_MISSING NCAAF_COMPAT_RECORD_SIZE NCAAF_COMPAT_RECORD_SCHEMA
NCAAF_COMPAT_SHARED_CITY_CONFLICT NCAAF_COMPAT_TARGET_UNSUPPORTED NCAAF_COMPAT_LINEAGE_CLOCK_CONFLICT
NCAAF_COMPAT_HISTORICAL_COMPONENT_CONFLICT NCAAF_COMPAT_RECOVERY_SCHEMA NCAAF_COMPAT_RECOVERY_CONFLICT
NCAAF_COMPAT_LINEAGE_CONFLICT NCAAF_COMPAT_ARTIFACT_LINEAGE_CONFLICT NCAAF_COMPAT_ARTIFACT_CONFLICT
NCAAF_COMPAT_MATH_CHANGED NCAAF_COMPAT_PACKET_SCHEMA""".split()) | chronology.REASONS
require = model.require


def view(packet):
    """A transport view only; neither the nested original nor its hash changes."""
    p = packet["payload"]["observation"]["payload"]
    q = p["quote"]
    return dict(original_quote=q, selection=dict(kind=q["market_type"], line=q["point"],
        price=q["price"], event_id=q["provider_event_id"], model_name="ridge"), source_review=p["source_review"])


def load(packet):
    require(set(packet) == {"payload", "sha256"} and model.digest(packet["payload"]) == packet["sha256"],
            "NCAAF_COMPAT_PACKET_INTEGRITY")
    p = packet["payload"]
    require(set(p) == {"version", "evidence_label", "observation", "dependency_objects"}
        and p["version"] in INPUT_VERSIONS and p["evidence_label"] in {"RETAINED", "SYNTHETIC"},
        "NCAAF_COMPAT_OBSERVATION_SCHEMA")
    original = p["observation"]
    require(set(original) == {"payload", "sha256"} and model.digest(original["payload"]) == original["sha256"],
            "NCAAF_COMPAT_PACKET_INTEGRITY")
    o = original["payload"]
    require(set(o) == set("version evidence_label model event schedule crosswalk mapping_review quote as_of features feature_dependencies source_review original_inference_time source_acceptance scientific_qualification probability_calibration wagering_authority wager_action live_stake".split())
        and o["version"] == (chronology.VERSION if p["version"] == SUCCESSOR_VERSION else observation.VERSION) and o["evidence_label"] == p["evidence_label"],
        "NCAAF_COMPAT_OBSERVATION_SCHEMA")
    require(isinstance(p["dependency_objects"], list) and 0 < len(p["dependency_objects"]) <= MAX_OBJECTS,
            "NCAAF_COMPAT_DEPENDENCY_BYTES_MISSING")
    return packet


def result_version(packet):
    return SUCCESSOR_RESULT_VERSION if packet["payload"]["version"] == SUCCESSOR_VERSION else RESULT_VERSION


def checked_observation(packet, at):
    original = packet["payload"]["observation"]
    if packet["payload"]["version"] == SUCCESSOR_VERSION:
        checked = chronology.read_observation(original)
        chronology.check_chronology(original["payload"], at)
        require(history.timestamp(original["payload"]["source_review"]["acceptance"]["accepted_at"]) < history.timestamp(at),
                "NCAAF_ACCEPTANCE_CLOCK_CONFLICT")
        return checked
    return observation.read_observation(original)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "NCAAF_COMPAT_DEPENDENCY_SCHEMA")
        result[key] = value
    return result


def objects(packet, at):
    """Verify original bytes and allowlisted native shapes, without mathematics."""
    p = load(packet)["payload"]
    season = p["observation"]["payload"]["schedule"][0]["season"]
    batches, index = [], {}
    for item in p["dependency_objects"]:
        require(isinstance(item, dict) and set(item) == {"sha256", "bytes_b64"}, "NCAAF_COMPAT_DEPENDENCY_SCHEMA")
        try:
            raw = base64.b64decode(item["bytes_b64"], validate=True)
        except (ValueError, TypeError):
            raise ValueError("NCAAF_COMPAT_DEPENDENCY_BYTES_CORRUPT") from None
        require(0 < len(raw) <= MAX_OBJECT_BYTES and hashlib.sha256(raw).hexdigest() == item["sha256"],
                "NCAAF_COMPAT_DEPENDENCY_BYTES_CORRUPT")
        require(item["sha256"] not in index, "NCAAF_COMPAT_DUPLICATE_DEPENDENCY")
        try:
            batch = json.loads(raw, object_pairs_hook=_unique)
        except (ValueError, TypeError, UnicodeError):
            raise ValueError("NCAAF_COMPAT_DEPENDENCY_SCHEMA") from None
        batches.append(batch)
        index[item["sha256"]] = batch
    state = dict(schema=1, years=[season], batches=batches)
    contract._state(state, at)
    return state, index


def fresh(packet, at):
    p = packet["payload"]["observation"]["payload"]
    now, observed, quote, start = [history.timestamp(v) for v in
        (at, p["as_of"], p["quote"].get("recorded_at"), p["event"].get("start_utc"))]
    require(all((now, observed, quote, start)) and observed <= now
        and 0 <= (now-quote).total_seconds() <= 900 and now < start <= now+timedelta(days=7),
        "NCAAF_COMPAT_NEW_INFERENCE_CLOCK_CONFLICT")
    require(p["original_inference_time"] is None, "NCAAF_COMPAT_RECORDED_INFERENCE_FORBIDDEN")
    return now


def accepted_dependencies(packet, approval, at):
    """Exact trusted packet review must cover the native feature sources too.

    This consumes the existing independent packet catalog; no upload creates a
    review. A quote's permissions do not imply feature-provider permissions.
    """
    review = approval.get("dependency_source_review")
    if packet["payload"]["version"] == SUCCESSOR_VERSION:
        now = history.timestamp(at)
        require(now is not None, "NCAAF_DEPENDENCY_ACCEPTANCE_CLOCK_CONFLICT")
        _, index = objects(packet, now)
        return chronology.check_dependency_admission(packet, review, index, at)
    require(isinstance(review, dict) and set(review) == {"provider", "endpoints", "dependency_hashes",
        "permitted_use", "public_derived_output", "rights_document", "reviewed_at", "effective_until"}
        and review["provider"] == "cfbd" and review["endpoints"] == ["games", "games/teams"]
        and review["dependency_hashes"] == [v["sha256"] for v in packet["payload"]["dependency_objects"]]
        and isinstance(review["rights_document"], str) and bool(review["rights_document"].strip()),
        "NCAAF_COMPAT_DEPENDENCY_SOURCE_REVIEW_MISSING_OR_CONFLICT")
    require(review["permitted_use"] == "prospective_research_features" and review["public_derived_output"] == "permitted",
            "NCAAF_COMPAT_DEPENDENCY_RIGHTS_UNAVAILABLE")
    now, reviewed, until = [history.timestamp(v) for v in (at, review["reviewed_at"], review["effective_until"])]
    require(all((now, reviewed, until)) and reviewed <= now < until, "NCAAF_COMPAT_SOURCE_REVIEW_CLOCK_CONFLICT")
    return deepcopy(review)


def derive(packet, checked, at):
    """NEW caller only: verify exact native dependencies, then unchanged math."""
    p = packet["payload"]["observation"]["payload"]
    state, index = objects(packet, at)
    game = observation.check_mapping(p["event"], p["schedule"], p["crosswalk"], p["mapping_review"], as_of=at)
    cutoff = history.timestamp(game["startDate"])-timedelta(days=7)
    historical = state["batches"][0]
    require(historical["request"]["kind"] == "games", "NCAAF_COMPAT_DEPENDENCY_SCHEMA")
    by_id = {}
    for g in historical["records"]:
        require(g["id"] not in by_id, "NCAAF_COMPAT_DEPENDENCY_IDENTITY_CONFLICT")
        by_id[g["id"]] = g
    needed = set()
    feature_clock = history.timestamp(p["features"]["available_at"])
    for r in p["feature_dependencies"]:
        require(all(type(r[k]) is int for k in ("game_id", "team_id", "season")),
                "NCAAF_COMPAT_DEPENDENCY_IDENTITY_CONFLICT")
        batch, g = index.get(r["source_sha256"]), by_id.get(r["game_id"])
        require(batch is not None and g is not None and history._valid_game(g, game["season"]),
                "NCAAF_COMPAT_DEPENDENCY_IDENTITY_CONFLICT")
        require(history.timestamp(g["startDate"]) == history.timestamp(r["start_utc"])
            and history.timestamp(batch["retrieved_at"]) == history.timestamp(r["available_at"])
            and history.timestamp(batch["retrieved_at"]) <= feature_clock
            and r["team_id"] in (g["homeId"], g["awayId"]), "NCAAF_COMPAT_DEPENDENCY_IDENTITY_CONFLICT")
        # Named identities must agree, even when all attacker-controlled hashes
        # were consistently rewritten. No fuzzy or city-only identity fallback.
        for field, name in (("homeId", "homeTeam"), ("awayId", "awayTeam")):
            if g[field] in (game["homeId"], game["awayId"]):
                expected = game["homeTeam" if g[field] == game["homeId"] else "awayTeam"]
                require(observation._canonical(g[name]) == observation._canonical(expected),
                        "NCAAF_COMPAT_DEPENDENCY_IDENTITY_CONFLICT")
        if r["kind"] == "scoring":
            require(batch is historical, "NCAAF_COMPAT_DEPENDENCY_REFERENCE_CONFLICT")
        else:
            require(batch["request"] == dict(kind="stats", year=g["season"], week=g["week"], season_type=g["seasonType"]),
                    "NCAAF_COMPAT_DEPENDENCY_REFERENCE_CONFLICT")
            rows = [v for v in batch["records"] if v["id"] == g["id"]]
            require(len(rows) == 1 and len(rows[0]["teams"]) == 2
                and {v["teamId"] for v in rows[0]["teams"]} == {g["homeId"], g["awayId"]},
                "NCAAF_COMPAT_DEPENDENCY_IDENTITY_CONFLICT")
            for team in rows[0]["teams"]:
                points = g["homePoints" if team["teamId"] == g["homeId"] else "awayPoints"]
                require(team["points"] == points and history._yards(team) is not None,
                        "NCAAF_COMPAT_DEPENDENCY_IDENTITY_CONFLICT")
        needed.add(r["source_sha256"])
    require(needed == set(index), "NCAAF_COMPAT_DEPENDENCY_REFERENCE_CONFLICT")
    _, rows, _ = history.build_dataset(state, feature_targets=[game])
    require(len(rows) == 1 and research._eligible(rows[0]), "NCAAF_COMPAT_MINIMUM_HISTORY_MISSING")
    row = rows[0]
    expected = set()
    for side in ("home", "away"):
        tid = game[side+"Id"]
        for gid in map(int, row[side+"_prior_game_ids"].split(";")):
            expected.add((gid, tid, "scoring"))
            matches = [t for b in state["batches"][1:] for g in b["records"] if g["id"] == gid
                for t in g["teams"] if t["teamId"] == tid and history._yards(t) is not None]
            require(len(matches) <= 1, "NCAAF_COMPAT_DEPENDENCY_IDENTITY_CONFLICT")
            if matches:
                expected.add((gid, tid, "yardage"))
        require(sum(k == "yardage" and team == tid for _, team, k in expected) == row[side+"_yards_games"],
                "NCAAF_COMPAT_DEPENDENCY_REFERENCE_CONFLICT")
    actual = {(r["game_id"], r["team_id"], r["kind"]) for r in p["feature_dependencies"]}
    require(actual == expected and all(history.timestamp(by_id[gid]["startDate"]) < cutoff for gid, _, _ in actual),
            "NCAAF_COMPAT_DEPENDENCY_REFERENCE_CONFLICT")
    values = [row[name] for name in research.FEATURES]
    require(values == checked["ordered_features"], "NCAAF_COMPAT_FEATURE_DERIVATION_CONFLICT")
    return dict(zip(research.FEATURES, values))


def verified_model_bytes(packet):
    """Prospective byte/schema diagnostic; immutable compatibility checks follow."""
    value = packet["payload"]["observation"]["payload"]["model"]
    require(isinstance(value, dict) and set(value) == {"version", "original_record_b64",
        "predecessor_record_b64", "compatibility"} and value["version"] == model.VERSION,
        "NCAAF_COMPAT_PACKET_SCHEMA")
    for key in ("original_record_b64", "predecessor_record_b64"):
        require(isinstance(value[key], str) and 0 < len(value[key]) <= model.MAX_RECORD_BYTES*2,
                "NCAAF_COMPAT_RECORD_SIZE")
        try:
            base64.b64decode(value[key], validate=True)
        except (ValueError, TypeError):
            raise ValueError("NCAAF_COMPAT_RECORD_SCHEMA") from None


def infer(packet, at):
    now = fresh(load(packet), at)  # Historical/stale inputs rejected before computation.
    verified_model_bytes(packet)
    checked = checked_observation(packet, at)
    p = packet["payload"]["observation"]["payload"]
    if packet["payload"]["version"] == VERSION:
        require(p["quote"]["rules"] == p["source_review"]["settlement"], "NCAAF_COMPAT_SOURCE_REVIEW_MISSING_OR_CONFLICT")
        require(now < history.timestamp(packet["payload"]["observation"]["payload"]["source_review"]["effective_until"]),
                "NCAAF_COMPAT_SOURCE_REVIEW_CLOCK_CONFLICT")
    features = derive(packet, checked, now)
    target = checked["target"]
    kind = packet["payload"]["observation"]["payload"]["quote"]["market_type"]
    line, family = target["signed_line"], target["family"]
    fitted = "total" if family == "total" else "margin"
    center = float(research.centers(checked["fit"], [features], fitted)[0])
    threshold = -line if kind == "spread_home" else line
    mass = research.probabilities(center, checked["fit"]["sigma"], threshold, total=family == "total")
    win = mass["over" if kind in {"spread_home", "total_over"} else "under"]
    require(mass["push"] == 0 and math.isfinite(win) and 0 <= win <= 1, "NCAAF_PROBABILITY_BINDING_CONFLICT")
    receipt = dict(version=result_version(packet), input_sha256=packet["sha256"],
        observation_sha256=packet["payload"]["observation"]["sha256"], inference_time=at,
        ordered_features=list(features.values()), feature_order=list(features), target=target,
        dependency_hashes=[v["sha256"] for v in packet["payload"]["dependency_objects"]],
        feature_derivation_verified=True, dependency_objects_verified=True,
        raw_probability=win, push_probability=0., projection=center, sigma=checked["fit"]["sigma"],
        model_record_sha256=hashlib.sha256(checked["model"]["original_bytes"]).hexdigest(),
        compatibility=deepcopy(checked["model"]["compatibility"]),
        scientific_acceptance=False, probability_calibration=False, wagering_authority=False, live_stake=0)
    if packet["payload"]["version"] == SUCCESSOR_VERSION:
        receipt["admission_receipts"] = chronology.check_chronology(p, at)
    return checked, center, win, dict(payload=receipt, sha256=model.digest(receipt))


def inspect_result(packet, saved, at):
    """Static authentic packet inspection: no feature derivation or inference."""
    now = fresh(load(packet), at)
    verified_model_bytes(packet)
    checked = checked_observation(packet, at)
    objects(packet, now)
    r = saved["payload"]
    require(set(saved) == {"payload", "sha256"} and model.digest(r) == saved["sha256"]
        and r["version"] == result_version(packet) and r["input_sha256"] == packet["sha256"]
        and r["observation_sha256"] == packet["payload"]["observation"]["sha256"]
        and r["inference_time"] == at and r["ordered_features"] == checked["ordered_features"]
        and r["feature_order"] == list(research.FEATURES) and r["target"] == checked["target"]
        and r["dependency_hashes"] == [v["sha256"] for v in packet["payload"]["dependency_objects"]]
        and r["model_record_sha256"] == hashlib.sha256(checked["model"]["original_bytes"]).hexdigest()
        and r["compatibility"] == checked["model"]["compatibility"]
        and r["feature_derivation_verified"] is True and r["dependency_objects_verified"] is True
        and r["push_probability"] == 0 and r["sigma"] == checked["fit"]["sigma"]
        and r["scientific_acceptance"] is False and r["probability_calibration"] is False
        and r["wagering_authority"] is False and r["live_stake"] == 0,
        "NCAAF_COMPAT_COMPUTATION_RECEIPT_CONFLICT")
    if packet["payload"]["version"] == SUCCESSOR_VERSION:
        require(r.get("admission_receipts") == chronology.check_chronology(packet["payload"]["observation"]["payload"], at),
                "NCAAF_COMPAT_COMPUTATION_RECEIPT_CONFLICT")
    return checked, r
