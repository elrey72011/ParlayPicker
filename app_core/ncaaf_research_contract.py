"""Private contracts for the existing NCAAF prospective capture/export path.

No default inference, collection, fitting, restore or authority reader. An
original capture and a separately supplied exact source review are required.
Historical publication clocks remain unknown under research protocol v1.
"""
from copy import deepcopy
from datetime import timedelta
import hashlib
import json
import math
from pathlib import Path
import platform
import sqlite3
from contextlib import closing

from app_core import ncaaf_history as history, ncaaf_research as research
from app_core import ncaaf_prospective as prospective, ncaaf_prospective_store as store

VERSION = "ncaaf-private-target-replay-v1"
TARGETS = {
    "spread": "ncaaf-selected-full-game-discrete-spread-v1",
    "total": "ncaaf-full-game-discrete-total-v1",
}
KINDS = frozenset({"spread_home", "spread_away", "total_over", "total_under"})


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _hash(value):
    return hashlib.sha256(store.encode(value)).hexdigest()


def target_contract(kind, line):
    _require(kind in KINDS, "NCAAF_UNSUPPORTED_TARGET")
    _require(type(line) in (int, float) and math.isfinite(line)
             and float(line * 2).is_integer(), "NCAAF_INVALID_LINE")
    family = "spread" if kind.startswith("spread") else "total"
    _require(family != "total" or line >= 0, "NCAAF_INVALID_LINE")
    return dict(version=TARGETS[family], sport="NCAAF", family=family,
        selection=kind.rsplit("_", 1)[-1], signed_line=line,
        period="full_game", overtime="included", orientation="named_home_away",
        outcome=("selected_margin_plus_line" if family == "spread" else
                 "total_minus_line" if kind == "total_over" else "line_minus_total"),
        probabilities="unconditional_win_push_loss",
        push="discretized_score_mass_at_integer_threshold" if float(line).is_integer()
             else "structural_zero_at_half_point",
        distribution=research.PROTOCOL["distribution"])


def read_records(path):
    """Read a supplied coherent snapshot, never a live store with pending WAL."""
    target = Path(path).resolve()
    _require(target.is_file(), "NCAAF_EXISTING_STORE_INACCESSIBLE")
    wal = Path(str(target)+"-wal")
    _require(not wal.exists() or wal.stat().st_size == 0, "NCAAF_STORE_SNAPSHOT_REQUIRED")
    with closing(sqlite3.connect(target.as_uri()+"?mode=ro&immutable=1", uri=True)) as db:
        records = []
        for key, raw in db.execute("SELECT id,payload FROM records ORDER BY rowid"):
            _require(hashlib.sha256(raw.encode()).hexdigest() == key,
                     "NCAAF_RECORD_INTEGRITY")
            records.append(dict(id=key, **json.loads(raw)))
    return records


def _records(records):
    index = {}
    for record in records:
        _require(isinstance(record, dict) and set(record) ==
                 {"id", "schema", "kind", "created_at", "data"}, "NCAAF_RECORD_SCHEMA")
        raw = {k: v for k, v in record.items() if k != "id"}
        _require(record["schema"] == 1 and _hash(raw) == record["id"], "NCAAF_RECORD_INTEGRITY")
        _require(record["id"] not in index, "NCAAF_DUPLICATE_RECORD")
        index[record["id"]] = record
    return index


def _state(value, when):
    _require(isinstance(value, dict) and set(value) == {"schema", "years", "batches"}
             and value["schema"] == 1 and len(value["years"]) == 1
             and history.integer(value["years"][0]), "NCAAF_INPUT_SCHEMA")
    year = value["years"][0]
    seen = set()
    for i, batch in enumerate(value["batches"]):
        _require(set(batch) == {"request", "retrieved_at", "records"}, "NCAAF_INPUT_SCHEMA")
        req = batch["request"]
        expected = {"kind", "year"} if req.get("kind") == "games" else {
            "kind", "year", "week", "season_type"}
        _require(set(req) == expected and req["kind"] in {"games", "stats"}
                 and req["year"] == year and (i == 0) == (req["kind"] == "games"),
                 "NCAAF_INPUT_SCHEMA")
        if req["kind"] == "stats":
            _require(history.integer(req["week"]) and 0 <= req["week"] <= 30
                     and req["season_type"] in {"regular", "postseason"}, "NCAAF_INPUT_SCHEMA")
        key = _hash(req)
        _require(key not in seen, "NCAAF_DUPLICATE_DEPENDENCY")
        seen.add(key)
        at = history.timestamp(batch["retrieved_at"])
        _require(at is not None and 0 <= (when-at).total_seconds() <= 86400,
                 "NCAAF_FUTURE_OR_STALE_DEPENDENCY")
        _require(history._clean(req["kind"], batch["records"]) == batch["records"],
                 "NCAAF_INPUT_NOT_ALLOWLISTED")
    _require(bool(value["batches"]), "NCAAF_ORIGINAL_INPUTS_MISSING")
    return value


def _artifact(model, when):
    at = history.timestamp(model["created_at"])
    _require(at is not None and at <= when, "NCAAF_FUTURE_MODEL")
    data = model["data"]
    _require(set(data) == {"artifact", "artifact_hash", "runtime_hash", "policy"},
             "NCAAF_MODEL_SCHEMA")
    artifact = data["artifact"]
    _require(_hash(artifact) == data["artifact_hash"], "NCAAF_ARTIFACT_INTEGRITY")
    _require(data["runtime_hash"] == prospective.runtime_hash(), "NCAAF_FROZEN_RUNTIME_MISMATCH")
    _require(set(artifact) == {"protocol", "source_hash", "train_hash", "calibration_hash", "models"}
             and artifact["protocol"] == research.PROTOCOL, "NCAAF_ARTIFACT_TARGET_CONFLICT")
    _require(artifact["source_hash"] == hashlib.sha256(Path(research.__file__).read_bytes()).hexdigest(),
             "NCAAF_ARTIFACT_READER_MISMATCH")
    for field in ("train_hash", "calibration_hash"):
        h = artifact[field]
        _require(isinstance(h, str) and len(h) == 64
                 and all(c in "0123456789abcdef" for c in h), "NCAAF_TRAINING_LINEAGE_MISSING")
    _require(bool(artifact["models"]) and set(artifact["models"]) <= {"ridge", "constant", "scoring_blend"},
             "NCAAF_ARTIFACT_TARGET_CONFLICT")
    for kind, targets in artifact["models"].items():
        _require(set(targets) == {"margin", "total"}, "NCAAF_ARTIFACT_TARGET_CONFLICT")
        for fit in targets.values():
            required = {"kind", "intercept", "bias", "sigma"}
            if kind == "ridge":
                required |= {"x_mean", "x_scale", "coefficients"}
            _require(set(fit) == required and fit["kind"] == kind
                     and all(type(fit[k]) in (int, float) and math.isfinite(fit[k])
                             for k in ("intercept", "bias", "sigma")) and fit["sigma"] >= 1,
                     "NCAAF_ARTIFACT_PARAMETER_CONFLICT")
            if kind == "ridge":
                _require(all(len(fit[k]) == len(research.FEATURES)
                    and all(type(v) in (int, float) and math.isfinite(v) for v in fit[k])
                    for k in ("x_mean", "x_scale", "coefficients"))
                    and all(v > 0 for v in fit["x_scale"]), "NCAAF_ORDERED_FEATURE_WIDTH_CONFLICT")
    return artifact


def _observation(index, capture_id, event_id, model_name, kind, line, book, price, quote_time):
    capture = index[capture_id]
    _require(capture["kind"] == "capture", "NCAAF_CAPTURE_MISSING")
    data = capture["data"]
    _require(set(data) == {"model_id", "captured_at", "inputs", "events", "skipped", "production_eligible"}
             and data["production_eligible"] is False, "NCAAF_CAPTURE_SCHEMA")
    when, created = history.timestamp(data["captured_at"]), history.timestamp(capture["created_at"])
    _require(when is not None and created is not None and when <= created, "NCAAF_CAPTURE_CLOCK_CONFLICT")
    state = _state(data["inputs"], when)
    models = [r for r in index.values() if r["kind"] == "model" and r["id"] == data["model_id"]]
    _require(len(models) == 1, "NCAAF_MODEL_RECORD_MISSING")
    model = models[0]
    artifact = _artifact(model, when)
    events = [e for e in data["events"] if e["event_id"] == event_id]
    _require(len(events) == 1 and len({e["cfbd_id"] for e in data["events"]}) == len(data["events"]),
             "NCAAF_EVENT_AMBIGUOUS")
    event = events[0]
    _require(model_name in artifact["models"] and model_name in event["models"], "NCAAF_MODEL_TARGET_MISSING")
    candidates = event["models"][model_name]["candidates"]
    quotes = [q for q in candidates if q["market_type"] == kind and q["point"] == line
              and q["book"] == book and q["price"] == price and q["recorded_at"] == quote_time]
    _require(len(quotes) == 1, "NCAAF_EXACT_OFFER_AMBIGUOUS_OR_MISSING")
    quote = quotes[0]
    _require(quote.get("provenance_version") == "provider-offer-facts-v1"
             and quote.get("provider_namespace") == "odds_api"
             and quote.get("provider_event_id") == event_id, "NCAAF_PROVIDER_BINDING_MISSING")
    _require(quote.get("period") == "full_game" and quote.get("period_source")
             and quote.get("rules") and quote.get("rules_source"), "NCAAF_PERIOD_OR_SETTLEMENT_MISSING")
    start, qt = history.timestamp(event["start"]), history.timestamp(quote_time)
    _require(start is not None and qt is not None and 0 <= (when-qt).total_seconds() <= 900
             and created < start <= when+timedelta(days=7), "NCAAF_QUOTE_CAPTURE_START_CLOCK_CONFLICT")
    provider_event = dict(id=event_id, home_team=quote["event_home_team"],
        away_team=quote["event_away_team"], commence_time=quote["event_start_utc"])
    game = prospective._match(provider_event, state["batches"][0]["records"])
    _require(game is not None and game["id"] == event["cfbd_id"]
             and game["homeId"] == event["home_id"] and game["awayId"] == event["away_id"]
             and game["season"] == event["season"]
             and game["homeTeam"] == event["home"] and game["awayTeam"] == event["away"]
             and history.timestamp(provider_event["commence_time"]) == start
             and game["homePoints"] is None and game["awayPoints"] is None,
             "NCAAF_CANONICAL_EVENT_CONFLICT")
    _require(event["season"] > research.PROTOCOL["evaluate"], "NCAAF_EVALUATED_HOLDOUT_CONTAMINATION")
    _, features, _ = history.build_dataset(state, feature_targets=[game])
    _require(len(features) == 1 and research._eligible(features[0]), "NCAAF_ORIGINAL_FEATURES_UNAVAILABLE")
    _require(features[0] == event["features"], "NCAAF_ALTERED_ORDERED_FEATURES")
    contract = target_contract(kind, line)
    return capture, model, event, quote, features[0], contract


def review_binding(records, **selection):
    """Describe exact review scope; this creates no approval or source registration."""
    c, m, e, q, f, target = _observation(_records(records), **selection)
    return dict(capture_hash=c["id"], model_record_hash=m["id"],
        artifact_hash=m["data"]["artifact_hash"], inputs_hash=_hash(c["data"]["inputs"]),
        event_hash=_hash(e), quote_hash=_hash(q), features_hash=_hash(f), target_contract=target)


def export_observation(records, *, source_review, evidence_label="RETAINED", **selection):
    """Export one existing exact observation; neither acquire nor mutate evidence."""
    _require(evidence_label in {"RETAINED", "SYNTHETIC"}, "NCAAF_EVIDENCE_LABEL_INVALID")
    index = _records(records)
    c, m, e, q, f, target = _observation(index, **selection)
    binding = review_binding(records, **selection)
    _require(isinstance(source_review, dict) and source_review.get("binding") == binding,
             "NCAAF_EXACT_SOURCE_REVIEW_MISSING_OR_CONFLICT")
    review = source_review
    _require(review.get("rights") == "ACCEPTED_FOR_PRIVATE_RESEARCH"
             and all(isinstance(review.get(k), str) and review[k].strip()
                     for k in ("review_id", "operator", "product", "listing_id"))
             and review.get("settlement") == "full_game_including_overtime_binary_win_push_loss"
             and not q["book"].lower().startswith("novig"), "NCAAF_SOURCE_PAYOFF_OR_RIGHTS_UNSUPPORTED")
    since, until, reviewed = [history.timestamp(review.get(k)) for k in
                              ("effective_from", "effective_until", "reviewed_at")]
    qt, when = history.timestamp(q["recorded_at"]), history.timestamp(c["data"]["captured_at"])
    _require(all((since, until, reviewed)) and since <= qt and reviewed <= qt and when < until,
             "NCAAF_SOURCE_REVIEW_CLOCK_CONFLICT")
    payload = dict(version=VERSION, evidence_label=evidence_label, selection=selection,
        records=[deepcopy(m), deepcopy(c)], source_review=deepcopy(review), target_contract=target,
        feature_order=list(research.FEATURES), ordered_features=[f[k] for k in research.FEATURES],
        dependency_receipts=[dict(request=b["request"], observed_at=b["retrieved_at"],
            sha256=_hash(b)) for b in c["data"]["inputs"]["batches"]],
        probabilities={k: q[v] for k, v in (("win", "win"), ("push", "push"), ("loss", "loss"))},
        original_quote=q, original_ev=q["ev"], predictor_binding=binding,
        probability_stage="frozen_ncaaf_research_v1_discrete_distribution",
        reader_binding=dict(prospective_runtime=m["data"]["runtime_hash"],
            research_sha256=m["data"]["artifact"]["source_hash"],
            contract_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
            python=platform.python_version(), numpy=research.np.__version__),
        original_blend=None, ui_refresh=None,
        original_capture_finished_at=c["data"]["captured_at"], original_inference_time=None,
        clock_missingness="NCAAF_PROSPECTIVE_V1_PER_MODEL_INFERENCE_CLOCK_NOT_RETAINED",
        availability_basis="prospective_retrieval_before_capture_finish_seven_day_history_lag",
        historical_publication_time_verified=False, scientific_acceptance=False,
        wagering_authority=False, live_stake=0)
    return dict(payload=payload, sha256=_hash(payload))


def replay(packet):
    """Explicit private execution with the installed score-distribution reader.

    Callers must have execution authorization. Static inspection of historical
    packets is a separate operation and must not call this function.
    """
    p = packet["payload"]
    _require(set(packet) == {"payload", "sha256"} and _hash(p) == packet["sha256"], "NCAAF_PACKET_INTEGRITY")
    rebuilt = export_observation(p["records"], source_review=p["source_review"],
                                 evidence_label=p["evidence_label"], **p["selection"])
    _require(rebuilt == packet, "NCAAF_EXPORT_CONTRACT_CONFLICT")
    c, m, e, q, f, target = _observation(_records(p["records"]), **p["selection"])
    fit = m["data"]["artifact"]["models"][p["selection"]["model_name"]]
    family = target["family"]
    center = float(research.centers(fit["margin" if family == "spread" else "total"], [f],
                                  "margin" if family == "spread" else "total")[0])
    kind, line = q["market_type"], q["point"]
    threshold = (-line if kind == "spread_home" else line) if family == "spread" else line
    probabilities = research.probabilities(center, fit["margin" if family == "spread" else "total"]["sigma"],
                                          threshold, total=family == "total")
    win = probabilities["over" if kind in {"spread_home", "total_over"} else "under"]
    result = dict(win=win, push=probabilities["push"], loss=max(0., 1-win-probabilities["push"]))
    _require(result == p["probabilities"] and all(type(v) in (int, float) and 0 <= v <= 1 for v in result.values())
             and math.isclose(sum(result.values()), 1., abs_tol=1e-12), "NCAAF_REPLAY_PROBABILITY_CONFLICT")
    price = q["price"]
    _require(type(price) in (int, float) and math.isfinite(price) and abs(price) >= 100,
             "NCAAF_INVALID_PRICE")
    decimal = 1 + (price/100 if price > 0 else 100/abs(price))
    _require(q["decimal_odds"] == decimal and result["win"]*(decimal-1)-result["loss"] == q["ev"]
             and q["live_stake"] == 0, "NCAAF_REPLAY_PAYOFF_CONFLICT")
    return dict(probabilities=result, conditional_win_no_push=(win/(1-result["push"])
        if result["push"] < 1 else None), scientific_acceptance=False, wagering_authority=False,
        live_stake=0, evidence_label=p["evidence_label"])
