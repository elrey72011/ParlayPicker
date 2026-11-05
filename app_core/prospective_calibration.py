"""Settlement, calibration and plan-bound predictions for research score models (roadmap C2).

The research sports (NBA, NCAAB; NHL once its shootout rule is confirmed) train score
models in ``prospective_research_models``, but three links to the frozen validation
plans were missing:

1. **Settlement.** Their results are final scores without a market outcome, while
   plan evaluation needs a WIN/LOSS/PUSH result per market and selection.
   ``settle_scope`` settles each event's earliest predicted quote from its replay-verified
   score (``prospective_evidence.insert_score_settlement``).
2. **Calibration.** Plans require a registered calibration for the exact model.
   ``fit_calibration`` fits Platt scaling (logistic on the decided-win log-odds) only on
   that model's own pregame predictions of games settled *after* it became available,
   so no outcome it trained on is reused.
3. **A pinned candidate.** The research cycle retrains on every new game, so its newest
   model changes nearly every cycle and never collects a calibration sample. The
   *candidate* is the earliest model trained on at least ``CANDIDATE_MIN_TRAINING_EVENTS``
   games. That choice never changes, and the candidate predicts every upcoming event,
   uncalibrated, until its calibration is fitted.
4. **Plan-bound predictions.** Plan evaluation counts only predictions carrying the
   plan's exact model and calibration IDs, but the research cycle predicts with the
   newest model and no calibration. ``predict_bound`` predicts every upcoming event
   with the latest frozen plan's bound model and calibration, with full provenance.

This module lives outside ``prospective_research_models`` on purpose. That module's
models are bound to a hash of its own source (``runtime_hash``), and changes here
must not invalidate a bound model. Nothing here grants wager authority: a plan still
needs validation, review and the owner's activation.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math

from app_core import prospective_evidence as evidence
from app_core import prospective_research_models as research

CALIBRATION_VERSION = "platt-logit-v1"
MIN_CALIBRATION_DECIDED = 40
CANDIDATE_MIN_TRAINING_EVENTS = 120
POLICY_VERSION = "plan-bound-research-v1"
CANDIDATE_POLICY_VERSION = "calibration-candidate-v1"


def _time(value):
    return research._time(value)


def _logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _sigmoid(x):
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def _latest_scores(scores):
    best = {}
    for row in scores:
        current = best.get(row["event_id"])
        if current is None or row["grading_version"] > current["grading_version"]:
            best[row["event_id"]] = row
    return best


def _earliest_predictions(path, sport, market, model_id=None, calibration_id=...):
    out = {}
    for row in evidence.read_records(path, "prospective_prediction", sport=sport, market_family=market):
        if model_id is not None and row["model_id"] != model_id:
            continue
        if calibration_id is not ... and row["calibration_id"] != calibration_id:
            continue
        if not row["quote_id"]:
            continue
        key = row["event_id"]
        if key not in out or (row["prediction_timestamp"], row["observation_id"]) < (
                out[key]["prediction_timestamp"], out[key]["observation_id"]):
            out[key] = row
    return out


def _market_result(path, sport, market):
    results = {}
    for row in evidence.read_records(path, "prospective_result", sport=sport, market_family=market):
        if row["outcome"] in {"WIN", "LOSS", "PUSH", "VOID"}:
            current = results.get((row["event_id"], row["selection"]))
            if current is None or row["grading_version"] > current["grading_version"]:
                results[(row["event_id"], row["selection"])] = row
    return results


def settle_scope(path, sport, market):
    """Settle each event's earliest predicted quote once a verified final score exists."""
    report = {"settled": 0, "already_settled": 0, "awaiting_score": 0, "errors": []}
    if sport not in evidence.SCORE_SETTLEMENT_SPORTS:
        report["errors"].append("SCORE_SETTLEMENT_NOT_ENABLED")
        return report
    _, _, scores = research._read_scope(path, sport, market)
    latest = _latest_scores(scores)
    settled = _market_result(path, sport, market)
    for event_id, prediction in sorted(_earliest_predictions(path, sport, market).items()):
        if (event_id, prediction["selection"]) in settled:
            report["already_settled"] += 1
            continue
        score = latest.get(event_id)
        if score is None:
            report["awaiting_score"] += 1
            continue
        try:
            evidence.insert_score_settlement(path, score_result_id=score["result_id"], quote_id=prediction["quote_id"])
            report["settled"] += 1
        except (ValueError, evidence.EvidenceConflict) as exc:
            report["errors"].append(f"{event_id}:{str(exc)[:80]}")
    return report


def platt_fit(pairs, *, l2=1e-3, iterations=50):
    """(a, b) minimising log loss of sigmoid(a + b * logit(p)) on (p, y) pairs (Newton, small ridge)."""
    a, b = 0.0, 1.0
    xs = [_logit(p) for p, _ in pairs]
    ys = [y for _, y in pairs]
    for _ in range(iterations):
        ga = gb = haa = hab = hbb = 0.0
        for x, y in zip(xs, ys):
            q = _sigmoid(a + b * x)
            w = q * (1 - q)
            ga += q - y
            gb += (q - y) * x
            haa += w
            hab += w * x
            hbb += w * x * x
        ga += l2 * a
        gb += l2 * (b - 1)
        haa += l2
        hbb += l2
        det = haa * hbb - hab * hab
        if abs(det) < 1e-12:
            break
        da = (hbb * ga - hab * gb) / det
        db = (haa * gb - hab * ga) / det
        a, b = a - da, b - db
        if abs(da) < 1e-10 and abs(db) < 1e-10:
            break
    return a, b


def calibrated(probabilities, artifact):
    """Apply a Platt artifact to {win, push, loss}; the push probability is unchanged."""
    push = probabilities["push"]
    decided = probabilities["win"] + probabilities["loss"]
    p = probabilities["win"] / decided if decided > 0 else 0.5
    q = _sigmoid(artifact["a"] + artifact["b"] * _logit(p))
    return {"win": q * (1 - push), "push": push, "loss": (1 - q) * (1 - push)}


def fit_calibration(path, sport, market, model_id, *, now=None, min_decided=MIN_CALIBRATION_DECIDED):
    """Register a Platt calibration for ``model_id`` from its own post-availability settled predictions."""
    now = now or datetime.now(timezone.utc)
    model = evidence.load_record(path, "prospective_model", model_id)
    if model is None or (model["sport"], model["market_family"]) != (sport, market):
        return {"status": "UNKNOWN_MODEL", "calibration_id": None}
    existing = [row for row in evidence.read_records(path, "prospective_calibration", sport=sport, market_family=market)
                if row["model_id"] == model_id]
    if existing:
        return {"status": "PRESENT", "calibration_id": max(existing, key=lambda r: r["available_at"])["calibration_id"]}
    fit_start = _time(model["available_at"])
    results = _market_result(path, sport, market)
    pairs, result_ids = [], []
    for event_id, prediction in sorted(_earliest_predictions(path, sport, market, model_id, None).items()):
        if _time(prediction["prediction_timestamp"]) < fit_start:
            continue
        result = results.get((event_id, prediction["selection"]))
        if result is None or not fit_start <= _time(result["available_at"]) <= now:
            continue
        result_ids.append(result["result_id"])
        if result["outcome"] in ("WIN", "LOSS") and prediction["mean_probability"] is not None:
            win, loss = prediction["mean_probability"], prediction["loss_probability"]
            if win + loss > 0:
                pairs.append((win / (win + loss), int(result["outcome"] == "WIN")))
    if len(pairs) < min_decided:
        return {"status": "INSUFFICIENT_EVIDENCE", "calibration_id": None, "decided": len(pairs),
                "required_decided": min_decided}
    a, b = platt_fit(pairs)
    fit_end = max(_time(results_row["available_at"]) for results_row in
                  (evidence.load_record(path, "prospective_result", rid) for rid in result_ids))
    artifact = {"method": CALIBRATION_VERSION, "a": a, "b": b, "decided": len(pairs),
                "model_id": model_id, "sport": sport, "market_family": market,
                "fit_start": fit_start.isoformat(), "fit_end": fit_end.isoformat(),
                "fit_result_ids": result_ids, "applies_to": "decided_win_probability"}
    artifact_hash = research._digest(artifact)
    calibration_id = f"platt-calibration:{sport}:{market}:{artifact_hash[:24]}"
    created = max(now, fit_end).isoformat()
    evidence.insert_calibration(path, {
        "calibration_id": calibration_id, "sport": sport, "market_family": market,
        "calibration_version": CALIBRATION_VERSION, "model_id": model_id,
        "fit_start": fit_start.isoformat(), "fit_end": fit_end.isoformat(), "method": CALIBRATION_VERSION,
        "created_at": created, "available_at": created, "artifact_hash": artifact_hash,
        "calibration_artifact": artifact, "fit_result_ids": result_ids})
    return {"status": "FITTED", "calibration_id": calibration_id, "decided": len(pairs), "a": a, "b": b}


def bound_plan(path, sport, market):
    """The latest frozen plan version for this market, when it binds a model and calibration."""
    plans = evidence.read_records(path, "prospective_validation_plan", sport=sport, market_family=market)
    if not plans:
        return None
    latest = max(plans, key=lambda row: row["version"])
    return latest if latest["model_id"] and latest["calibration_id"] else None


def candidate_model(path, sport, market, *, min_events=None):
    """The earliest research model trained on at least ``min_events`` games: stable once chosen."""
    min_events = CANDIDATE_MIN_TRAINING_EVENTS if min_events is None else min_events
    models = [row for row in evidence.read_records(path, "prospective_model", sport=sport, market_family=market)
              if row["independent_event_count"] >= min_events]
    return min(models, key=lambda r: (r["available_at"], r["model_id"])) if models else None


def _predict(path, sport, market, model, calibration, *, now, policy_version, tag):
    artifact = json.loads(model["payload"]).get("model_artifact")
    if not isinstance(artifact, dict) or research._digest(artifact) != model["artifact_hash"]:
        return {"predictions": 0, "blocker": f"{tag}_MODEL_ARTIFACT_INTEGRITY"}
    if artifact.get("runtime_hash") != research.runtime_hash():
        # Visible, never silent: the research module changed after this model was trained.
        return {"predictions": 0, "blocker": f"{tag}_MODEL_RUNTIME_CHANGED"}
    cal_artifact = None
    if calibration is not None:
        cal_artifact = json.loads(calibration["payload"]).get("calibration_artifact")
        if not isinstance(cal_artifact, dict) or research._digest(cal_artifact) != calibration["artifact_hash"]:
            return {"predictions": 0, "blocker": f"{tag}_CALIBRATION_ARTIFACT_INTEGRITY"}
    calibration_id = calibration["calibration_id"] if calibration else None
    events, quotes, scores = research._read_scope(path, sport, market)
    done = {(row["event_id"], row["quote_id"]) for row in evidence.read_records(
        path, "prospective_prediction", sport=sport, market_family=market)
        if row["model_id"] == model["model_id"] and row["calibration_id"] == calibration_id}
    created, no_history = 0, 0
    commit = research.source_commit()
    for event_id, quote in sorted(research._reference_quotes(events, quotes, now).items()):
        event = events[event_id]
        if now >= _time(event["scheduled_start"]) or (event_id, quote["quote_id"]) in done:
            continue
        feature_result = research._features(event, now, events, scores, market)
        if feature_result is None:
            no_history += 1
            continue
        features, lineage = feature_result
        fit = artifact["model"]
        raw = research.score_probabilities(research._expected(fit, features), fit["residual_sd"], market, quote["line"])
        probabilities = calibrated(raw, cal_artifact) if cal_artifact else raw
        n = max(1, cal_artifact["decided"] if cal_artifact else artifact["training_independent_events"])
        conservative = max(0.0, probabilities["win"] - 1.645 * math.sqrt(
            probabilities["win"] * (1 - probabilities["win"]) / n))
        snapshot = {"protocol": research.FEATURE_VERSION, "sport": sport, "market_family": market,
                    "event_id": event_id, "quote_id": quote["quote_id"], "as_of": now.isoformat(),
                    "features": features, "source_results": lineage, "exact_line": quote["line"],
                    "american_odds": quote["american_odds"], "decimal_odds": quote["decimal_odds"],
                    "sportsbook": quote["sportsbook"], "provider_event_hash": event["source_hash"],
                    "provider_quote_hash": quote["source_hash"], "raw_probabilities": raw,
                    "calibration_id": calibration_id, "policy_version": policy_version}
        snapshot_id = evidence._sha(evidence._json(snapshot))
        observation_id = f"{tag.lower()}-prediction:" + research._digest([model["model_id"], calibration_id, snapshot_id])[:32]
        record = {
            "observation_id": observation_id, "event_id": event_id, "sport": sport, "market_family": market,
            "selection": quote["selection"], "quote_id": quote["quote_id"], "model_id": model["model_id"],
            "model_version": model["model_version"], "feature_version": research.FEATURE_VERSION,
            "feature_snapshot_id": snapshot_id, "feature_snapshot": snapshot, "feature_frozen_at": now.isoformat(),
            "prediction_timestamp": now.isoformat(), "mean_probability": probabilities["win"],
            "conservative_probability": conservative, "push_probability": probabilities["push"],
            "loss_probability": probabilities["loss"], "probability_semantics": "WIN_PUSH_LOSS",
            "policy_version": policy_version, "evidence_snapshot_id": snapshot_id,
            "evidence_hash": research._digest([event["source_hash"], quote["source_hash"], lineage]),
            "runtime_hash": research.runtime_hash(), "source_commit": commit,
            "uncertainty": {"method": "one_sided_normal_approximation_v1", "calibration_id": calibration_id,
                            "sample": n, "research_only": True}}
        if calibration is not None:
            record.update(calibration_id=calibration_id, calibration_version=calibration["calibration_version"])
        evidence.insert_prediction(path, record)
        created += 1
    return {"predictions": created, "skipped_insufficient_team_history": no_history, "blocker": None,
            "model_id": model["model_id"], "calibration_id": calibration_id}


def predict_candidate(path, sport, market, *, now=None):
    """Uncalibrated pregame predictions from the pinned candidate, to build its calibration sample."""
    now = now or datetime.now(timezone.utc)
    model = candidate_model(path, sport, market)
    if model is None:
        return {"predictions": 0, "blocker": "NO_CANDIDATE_MODEL"}
    return _predict(path, sport, market, model, None, now=now, policy_version=CANDIDATE_POLICY_VERSION,
                    tag="CANDIDATE")


def predict_bound(path, sport, market, *, now=None):
    """Freeze pregame predictions with the bound plan's exact model and calibration."""
    now = now or datetime.now(timezone.utc)
    plan = bound_plan(path, sport, market)
    if plan is None:
        return {"predictions": 0, "blocker": "NO_BOUND_PLAN"}
    model = evidence.load_record(path, "prospective_model", plan["model_id"])
    calibration = evidence.load_record(path, "prospective_calibration", plan["calibration_id"])
    result = _predict(path, sport, market, model, calibration, now=now, policy_version=POLICY_VERSION, tag="BOUND")
    return dict(result, validation_plan_id=plan["validation_plan_id"])


def draft_bound_plan(path, sport, market, *, validation_start, validation_days, holdout_days,
                     model_id=None, calibration_id=None):
    """A proposed plan version binding the candidate model and its calibration. Nothing is frozen.

    Thresholds, sample minimums and policies are the frozen v1 template's
    (``prospective_validation_plans._template``). Only the version chain, the binding
    and the window dates change. The windows are the owner's decision: v1 validates
    for a year and holds out for another year. Freezing is a separate, explicit call
    to ``prospective_evidence.freeze_validation_plan`` after review.
    """
    from datetime import timedelta
    from app_core.prospective_validation_plans import _template
    plans = evidence.read_records(path, "prospective_validation_plan", sport=sport, market_family=market)
    if not plans:
        raise ValueError("no frozen plan to supersede")
    previous = max(plans, key=lambda row: row["version"])
    if model_id is None:
        candidate = candidate_model(path, sport, market)
        if candidate is None:
            raise ValueError("no candidate model")
        model_id = candidate["model_id"]
    model = evidence.load_record(path, "prospective_model", model_id)
    if calibration_id is None:
        found = [row for row in evidence.read_records(path, "prospective_calibration", sport=sport, market_family=market)
                 if row["model_id"] == model_id]
        if not found:
            raise ValueError("candidate model has no calibration yet")
        calibration_id = max(found, key=lambda r: r["available_at"])["calibration_id"]
    calibration = evidence.load_record(path, "prospective_calibration", calibration_id)
    start = _time(validation_start)
    if start < _time(calibration["available_at"]):
        raise ValueError("validation must start after the calibration is available")
    validation_end = start + timedelta(days=validation_days)
    holdout_end = validation_end + timedelta(days=holdout_days)
    version = previous["version"] + 1
    spec = _template(sport, market)
    spec.update(
        validation_plan_id=f"prospective-{sport.lower()}-{market.lower()}-bound-v{version}",
        version=version, supersedes_plan_id=previous["validation_plan_id"],
        model_id=model_id, calibration_id=calibration_id,
        training_cutoff=max(_time(model["training_cutoff"]), _time(calibration["fit_end"])).isoformat(),
        validation_start=start.isoformat(), validation_end=validation_end.isoformat(),
        holdout_start=validation_end.isoformat(), holdout_end=holdout_end.isoformat())
    spec["model_scope"] = dict(spec["model_scope"], initial_binding="BOUND_EXACT_MODEL_AND_CALIBRATION")
    return spec


def run_cycle(path, sport, *, now=None):
    """Post-research-cycle hook: settle, calibrate the latest model, and predict for bound plans."""
    now = now or datetime.now(timezone.utc)
    report = {"sport": sport, "markets": {}}
    if sport not in evidence.SCORE_SETTLEMENT_SPORTS:
        report["skipped"] = "SCORE_SETTLEMENT_NOT_ENABLED"
        return report
    for market in research.SCOPES[sport]:
        entry = {"settlement": settle_scope(path, sport, market)}
        candidate = candidate_model(path, sport, market)
        if candidate is not None:
            entry["calibration"] = fit_calibration(path, sport, market, candidate["model_id"], now=now)
            if entry["calibration"]["status"] != "PRESENT":
                entry["candidate_predictions"] = predict_candidate(path, sport, market, now=now)
        entry["bound_predictions"] = predict_bound(path, sport, market, now=now)
        report["markets"][market] = entry
    return report
