"""Research-only score-distribution models for NBA, NCAAB and NHL markets.

An authenticated final score is a training target for a score distribution,
not proof that a sportsbook settled a particular line.  This module never
creates market settlement rows, calibration by assertion, or wager authority.
Every fitted model is exact sport/market scoped and carries its coefficients,
chronological validation diagnostics, and immutable source lineage.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess

from app_core import prospective_evidence as evidence
from app_core.odds_market_store import for_sport
from app_core.odds_research_adapter import (PROTOCOL, digest as native_digest,
                                             identity, participant_ids)


SCOPES = {"NBA": ("SPREAD", "TOTAL"), "NCAAB": ("SPREAD", "TOTAL"),
          "NHL": ("PUCK_LINE", "TOTAL")}
MODEL_VERSION = "pregame-score-ridge-normal-v1"
FEATURE_VERSION = "provider-final-score-form-v1"
TARGET_KIND = "FINAL_SCORE_DISTRIBUTION_RESEARCH_V1"
MIN_TRAIN_EVENTS = 60
MIN_VALIDATION_EVENTS = 18
MIN_TEAM_HISTORY = 2
MAX_TEAM_HISTORY = 8
HISTORY_DAYS = 240
RIDGE_STRENGTHS = (0.1, 1.0, 10.0)
FEATURE_LIMITATIONS = {
    "NBA": ("NO_AUTHENTIC_INJURY_LINEUP_FEATURE",),
    "NCAAB": ("NO_TIMESTAMPED_RANKING_EFFICIENCY_FEATURE",),
    "NHL": ("NO_AUTHENTIC_GOALIE_STATUS_FEATURE",),
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _time(value: str | datetime) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("timezone-aware timestamp required")
    return result.astimezone(timezone.utc)


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: object) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def runtime_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def source_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                            check=False, capture_output=True, text=True, timeout=3)
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and len(commit) == 40 else "UNAVAILABLE"


def _read_scope(path, sport, market):
    if sport not in SCOPES or market not in SCOPES[sport]:
        raise ValueError("unsupported research score model scope")
    events = {row["event_id"]: row for row in evidence.read_records(path, "prospective_event", sport=sport)}
    quotes = evidence.read_records(path, "prospective_quote", sport=sport, market_family=market)
    scores = evidence.read_records(path, "prospective_result", sport=sport)
    scores = [row for row in scores if row["market_family"] is None and row["event_id"] in events
              and evidence.verify_provider_score(events[row["event_id"]], row)]
    return events, quotes, scores


def _score_as_of(scores, event_id, when):
    eligible = [row for row in scores if row["event_id"] == event_id and
                _time(row["available_at"]) <= when]
    return max(eligible, key=lambda row: (row["grading_version"], row["available_at"],
                                          row["result_id"])) if eligible else None


def _reference_quotes(events, quotes, cutoff):
    """One replayable home/Over quote per event avoids repeated slate units."""
    selected = {}
    for quote in quotes:
        event = events.get(quote["event_id"])
        if event is None or quote["quote_verified"] != 1 or quote["line"] is None:
            continue
        canonical_selection = event["home_team"] if quote["market_family"] != "TOTAL" else "Over"
        if quote["selection"] != canonical_selection or _time(quote["quote_timestamp"]) > cutoff:
            continue
        if not evidence.verify_provider_offer(event, quote):
            continue
        line = quote["line"]
        if abs(2 * line - round(2 * line)) > 1e-8:
            continue  # quarter-lines need a separate split-stake settlement contract
        key = event["event_id"]
        previous = selected.get(key)
        if previous is None or (quote["quote_timestamp"], quote["quote_id"]) < (
                previous["quote_timestamp"], previous["quote_id"]):
            selected[key] = quote
    return selected


def _team_form(event, team_id, when, events, scores):
    rows = []
    target_start = _time(event["scheduled_start"])
    for past in events.values():
        if past["event_id"] == event["event_id"] or past["sport"] != event["sport"]:
            continue
        start = _time(past["scheduled_start"])
        if not (target_start - timedelta(days=HISTORY_DAYS) <= start < target_start and start < when):
            continue
        home = past["home_team_id"] == team_id
        away = past["away_team_id"] == team_id
        if not home and not away:
            continue
        score = _score_as_of(scores, past["event_id"], when)
        if score is None:
            continue
        points_for = score["home_score"] if home else score["away_score"]
        points_against = score["away_score"] if home else score["home_score"]
        rows.append((start, points_for, points_against, score))
    rows.sort(key=lambda item: (item[0], item[3]["result_id"]), reverse=True)
    rows = rows[:MAX_TEAM_HISTORY]
    if len(rows) < MIN_TEAM_HISTORY:
        return None
    count = len(rows)
    rest_days = min(7.0, max(0.0, (target_start - rows[0][0]).total_seconds() / 86400))
    return {"points_for": sum(row[1] for row in rows) / count,
            "points_against": sum(row[2] for row in rows) / count,
            "margin": sum(row[1] - row[2] for row in rows) / count,
            "rest_days": rest_days,
            "source_results": sorted({(row[3]["result_id"], row[3]["source_hash"])
                                      for row in rows})}


def _features(event, when, events, scores, market):
    if (event["provider_namespace"] != "THE_ODDS_API" or not event["home_team_id"] or
            not event["away_team_id"] or event["home_team_id"] == event["away_team_id"] or
            when >= _time(event["scheduled_start"])):
        return None
    home = _team_form(event, event["home_team_id"], when, events, scores)
    away = _team_form(event, event["away_team_id"], when, events, scores)
    if home is None or away is None:
        return None
    if market == "TOTAL":
        vector = [1.0, home["points_for"], home["points_against"],
                  away["points_for"], away["points_against"],
                  home["rest_days"] / 7.0, away["rest_days"] / 7.0]
    else:
        vector = [1.0, home["margin"], away["margin"],
                  home["points_for"] - away["points_against"],
                  away["points_for"] - home["points_against"],
                  home["rest_days"] / 7.0, away["rest_days"] / 7.0]
    lineage = sorted(set(tuple(row) for row in home["source_results"] + away["source_results"]))
    return vector, lineage


def _research_outcome(score, quote, market):
    """Score-line research label, deliberately separate from sportsbook grade."""
    value = (score["home_score"] + score["away_score"] if market == "TOTAL" else
             score["home_score"] - score["away_score"])
    difference = value - quote["line"] if market == "TOTAL" else value + quote["line"]
    if abs(difference) < 1e-9:
        return "PUSH"
    return "WIN" if difference > 0 else "LOSS"


def _examples(events, quotes, scores, market, cutoff):
    reference = _reference_quotes(events, quotes, cutoff)
    examples = []
    for event_id, quote in reference.items():
        event = events[event_id]
        score = _score_as_of(scores, event_id, cutoff)
        if score is None or _time(score["available_at"]) <= _time(event["observed_at"]):
            continue
        observed = _time(event["observed_at"])
        features = _features(event, observed, events, scores, market)
        if features is None:
            continue
        vector, lineage = features
        target = (score["home_score"] + score["away_score"] if market == "TOTAL" else
                  score["home_score"] - score["away_score"])
        examples.append({"event_id": event_id, "quote_id": quote["quote_id"],
                         "result_id": score["result_id"], "result_hash": score["source_hash"],
                         "event_hash": event["source_hash"], "quote_hash": quote["source_hash"],
                         "available_at": score["available_at"], "start": event["scheduled_start"],
                         "observed_at": event["observed_at"],
                         "line": quote["line"], "features": vector,
                         "feature_lineage": lineage, "target": target,
                         "research_outcome": _research_outcome(score, quote, market)})
    return sorted(examples, key=lambda row: (row["start"], row["event_id"]))


def _solve(matrix, values):
    n = len(values)
    rows = [list(matrix[i]) + [values[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(rows[r][col]))
        if abs(rows[pivot][col]) < 1e-12:
            raise ValueError("singular research fit")
        rows[col], rows[pivot] = rows[pivot], rows[col]
        divisor = rows[col][col]
        rows[col] = [x / divisor for x in rows[col]]
        for row in range(n):
            if row == col:
                continue
            factor = rows[row][col]
            rows[row] = [a - factor * b for a, b in zip(rows[row], rows[col])]
    return [row[-1] for row in rows]


def _fit(rows, alpha):
    width = len(rows[0]["features"])
    means = [0.0] + [sum(row["features"][j] for row in rows) / len(rows)
                     for j in range(1, width)]
    scales = [1.0] + [max(1e-9, math.sqrt(sum((row["features"][j] - means[j]) ** 2
                                      for row in rows) / len(rows))) for j in range(1, width)]
    matrix = [[0.0] * width for _ in range(width)]
    values = [0.0] * width
    for row in rows:
        x = [1.0] + [(row["features"][j] - means[j]) / scales[j] for j in range(1, width)]
        for i in range(width):
            values[i] += x[i] * row["target"]
            for j in range(width):
                matrix[i][j] += x[i] * x[j]
    for j in range(1, width):
        matrix[j][j] += alpha
    coefficients = _solve(matrix, values)
    fit = {"means": means, "scales": scales, "coefficients": coefficients,
           "ridge_strength": alpha}
    residuals = [row["target"] - _expected(fit, row["features"]) for row in rows]
    fit["residual_sd"] = max(0.75, math.sqrt(sum(x * x for x in residuals) / len(residuals)))
    return fit


def _expected(fit, features):
    return fit["coefficients"][0] + sum(
        fit["coefficients"][j] * (features[j] - fit["means"][j]) / fit["scales"][j]
        for j in range(1, len(features)))


def _cdf(value):
    return 0.5 * (1 + math.erf(value / math.sqrt(2)))


def score_probabilities(mean_score, residual_sd, market, line, *, selection="canonical"):
    """Discrete integer-score win/push/loss probabilities for an exact line."""
    if not all(math.isfinite(float(x)) for x in (mean_score, residual_sd, line)) or residual_sd <= 0:
        raise ValueError("invalid score distribution")
    if abs(2 * line - round(2 * line)) > 1e-8:
        raise ValueError("quarter or unsupported line increment")
    threshold = line if market == "TOTAL" else -line
    nearest = round(threshold)
    if abs(threshold - nearest) < 1e-8:
        lower = _cdf((nearest - 0.5 - mean_score) / residual_sd)
        upper = _cdf((nearest + 0.5 - mean_score) / residual_sd)
        push = max(0.0, upper - lower)
        win = max(0.0, 1 - upper)
    else:
        boundary = math.floor(threshold) + 0.5
        push = 0.0
        win = max(0.0, 1 - _cdf((boundary - mean_score) / residual_sd))
    loss = max(0.0, 1 - win - push)
    if selection == "opposite":
        win, loss = loss, win
    elif selection != "canonical":
        raise ValueError("unknown market selection orientation")
    total = win + push + loss
    return {"win": win / total, "push": push / total, "loss": loss / total}


def _diagnostics(fit, rows, market):
    probabilities = [score_probabilities(_expected(fit, row["features"]),
                                         fit["residual_sd"], market, row["line"])
                     for row in rows]
    true = [row["research_outcome"].lower() for row in rows]
    brier = sum(sum((prob[key] - int(label == key)) ** 2 for key in ("win", "push", "loss"))
                for prob, label in zip(probabilities, true)) / len(rows)
    log_loss = -sum(math.log(max(prob[label], 1e-15))
                    for prob, label in zip(probabilities, true)) / len(rows)
    decided = [(prob["win"] / max(1e-15, prob["win"] + prob["loss"]), int(label == "win"))
               for prob, label in zip(probabilities, true) if label != "push"]
    ece = 0.0
    for bucket in range(10):
        members = [(p, y) for p, y in decided if min(int(p * 10), 9) == bucket]
        if members:
            ece += len(members) / len(decided) * abs(
                sum(p for p, _ in members) / len(members) - sum(y for _, y in members) / len(members))
    positives = [p for p, y in decided if y]
    negatives = [p for p, y in decided if not y]
    roc_auc = (sum(1.0 if p > n else 0.5 if p == n else 0.0
                   for p in positives for n in negatives) / (len(positives) * len(negatives))
               if positives and negatives else None)
    ranked = sorted(decided, key=lambda row: row[0], reverse=True)
    average_precision = (sum(sum(y for _, y in ranked[:i]) / i for i, (_, y) in
                             enumerate(ranked, 1) if y) / len(positives) if positives and negatives else None)
    return {"independent_events": len(rows), "brier_ternary": brier, "log_loss_ternary": log_loss,
            "conditional_win_calibration_error": ece if decided else None,
            "coverage": len(probabilities) / len(rows), "roc_auc_decided": roc_auc,
            "average_precision_decided": average_precision,
            "research_labels_only": True, "market_settlement_certified": False}


def train_scope(path, sport, market, *, as_of=None):
    """Fit/register only after enough independent, replayable final-score events."""
    cutoff = _time(as_of) if as_of is not None else _utcnow()
    events, quotes, scores = _read_scope(path, sport, market)
    examples = _examples(events, quotes, scores, market, cutoff)
    if len(examples) < MIN_TRAIN_EVENTS:
        return {"sport": sport, "market_family": market, "status": "INSUFFICIENT_EVIDENCE",
                "independent_events": len(examples), "required_independent_events": MIN_TRAIN_EVENTS,
                "model_id": None}
    preferred_split = max(MIN_TRAIN_EVENTS - MIN_VALIDATION_EVENTS, int(len(examples) * 0.7))
    development = validation = None
    for split in range(preferred_split, len(examples) - MIN_VALIDATION_EVENTS + 1):
        candidate_validation = examples[split:]
        first_validation_observation = _time(candidate_validation[0]["observed_at"])
        candidate_development = [row for row in examples[:split] if
                                 _time(row["available_at"]) < first_validation_observation]
        if len(candidate_development) >= MIN_TRAIN_EVENTS - MIN_VALIDATION_EVENTS:
            development, validation = candidate_development, candidate_validation
            break
    if development is None:
        return {"sport": sport, "market_family": market,
                "status": "INSUFFICIENT_CHRONOLOGICAL_VALIDATION",
                "independent_events": len(examples), "model_id": None}
    candidates = []
    for alpha in RIDGE_STRENGTHS:
        fitted = _fit(development, alpha)
        diagnostics = _diagnostics(fitted, validation, market)
        candidates.append((diagnostics["log_loss_ternary"], diagnostics["brier_ternary"],
                           alpha, diagnostics))
    _, _, strength, diagnostics = min(candidates, key=lambda item: item[:3])
    fit = _fit(examples, strength)
    code_hash = runtime_hash()
    commit = source_commit()
    if commit == "UNAVAILABLE":
        return {"sport": sport, "market_family": market, "status": "SOURCE_COMMIT_UNAVAILABLE",
                "independent_events": len(examples), "model_id": None}
    artifact = {"protocol": MODEL_VERSION, "sport": sport, "market_family": market,
                "feature_version": FEATURE_VERSION, "training_target_kind": TARGET_KIND,
                "target": "final_score_total" if market == "TOTAL" else "final_score_home_margin",
                "model": fit, "runtime_hash": code_hash, "source_commit": commit,
                "training_independent_events": len(examples),
                "development_independent_events": len(development),
                "validation_independent_events": len(validation),
                "validation_cutoff": max(row["available_at"] for row in development),
                "validation_first_observed_at": validation[0]["observed_at"],
                "validation_diagnostics": diagnostics,
                "training_sources": [{key: row[key] for key in (
                    "event_id", "quote_id", "result_id", "result_hash", "event_hash", "quote_hash",
                    "available_at", "start", "observed_at", "line", "features",
                    "feature_lineage", "target")}
                    for row in examples],
                "production_eligible": False, "recommended_stake": 0.0,
                "market_settlement_certified": False,
                "feature_limitations": FEATURE_LIMITATIONS[sport]}
    artifact_hash = _digest(artifact)
    model_id = f"score-research:{sport}:{market}:{artifact_hash[:24]}"
    current = evidence.load_record(path, "prospective_model", model_id)
    new_model = current is None
    if current is None:
        first = min(_time(row["available_at"]) for row in examples)
        training_cutoff = max(_time(row["available_at"]) for row in examples)
        created = _utcnow()
        if training_cutoff > created:
            raise ValueError("training result is not yet available")
        evidence.insert_model(path, {"model_id": model_id, "sport": sport,
            "market_family": market, "model_version": MODEL_VERSION,
            "training_start": (first - timedelta(microseconds=1)).isoformat(),
            "training_cutoff": training_cutoff.isoformat(),
            "training_observation_count": len(examples),
            "independent_event_count": len(examples), "feature_version": FEATURE_VERSION,
            "training_code_commit": commit, "created_at": created.isoformat(),
            "available_at": created.isoformat(), "artifact_hash": artifact_hash,
            "model_artifact": artifact, "training_target_kind": TARGET_KIND,
            "training_result_ids": [row["result_id"] for row in examples]})
    calibrations = [row for row in evidence.read_records(path, "prospective_calibration",
                   sport=sport, market_family=market) if row["model_id"] == model_id]
    return {"sport": sport, "market_family": market, "status": "FITTED_RESEARCH_ONLY",
            "model_id": model_id, "artifact_hash": artifact_hash,
            "training_cutoff": max(row["available_at"] for row in examples),
            "independent_events": len(examples), "validation": diagnostics,
            "new_model": new_model,
            "calibration_status": "PRESENT" if calibrations else "INSUFFICIENT_EVIDENCE",
            "production_eligible": False,
            "recommended_stake": 0.0}


def _latest_model(path, sport, market, when):
    models = evidence.read_records(path, "prospective_model", sport=sport, market_family=market)
    eligible = [model for model in models if _time(model["available_at"]) <= when and
                json.loads(model["payload"]).get("training_target_kind") == TARGET_KIND]
    if not eligible:
        return None, None
    model = max(eligible, key=lambda row: (row["training_cutoff"], row["available_at"], row["model_id"]))
    artifact = json.loads(model["payload"]).get("model_artifact")
    if not isinstance(artifact, dict) or _digest(artifact) != model["artifact_hash"]:
        raise ValueError("model artifact integrity failure")
    if artifact.get("runtime_hash") != runtime_hash():
        raise ValueError("stale research model runtime")
    return model, artifact


def predict_scope(path, sport, market, *, as_of=None):
    """Freeze live pregame predictions; absent models produce an explicit blocker."""
    now = _time(as_of) if as_of is not None else _utcnow()
    events, quotes, scores = _read_scope(path, sport, market)
    try:
        model, artifact = _latest_model(path, sport, market, now)
    except ValueError as exc:
        if str(exc) == "stale research model runtime":
            return {"sport": sport, "market_family": market, "predictions": 0,
                    "blocker": "STALE_SPORT_MARKET_MODEL"}
        raise
    if model is None:
        return {"sport": sport, "market_family": market, "predictions": 0,
                "blocker": "MISSING_SPORT_MARKET_MODEL"}
    reference = _reference_quotes(events, quotes, now)
    existing = evidence.read_records(path, "prospective_prediction", sport=sport,
                                     market_family=market)
    already = {(row["event_id"], row["quote_id"], row["model_id"]) for row in existing}
    created = 0
    no_history = 0
    for event_id, quote in sorted(reference.items()):
        event = events[event_id]
        if now >= _time(event["scheduled_start"]) or (event_id, quote["quote_id"], model["model_id"]) in already:
            continue
        feature_result = _features(event, now, events, scores, market)
        if feature_result is None:
            no_history += 1
            continue
        features, lineage = feature_result
        fit = artifact["model"]
        probabilities = score_probabilities(_expected(fit, features), fit["residual_sd"],
                                            market, quote["line"])
        standard_error = math.sqrt(probabilities["win"] * (1 - probabilities["win"]) /
                                   max(1, artifact["training_independent_events"]))
        conservative = max(0.0, probabilities["win"] - 1.645 * standard_error)
        snapshot = {"protocol": FEATURE_VERSION, "sport": sport, "market_family": market,
                    "event_id": event_id, "quote_id": quote["quote_id"],
                    "as_of": now.isoformat(), "features": features,
                    "source_results": lineage, "exact_line": quote["line"],
                    "american_odds": quote["american_odds"],
                    "decimal_odds": quote["decimal_odds"],
                    "sportsbook": quote["sportsbook"],
                    "provider_event_hash": event["source_hash"],
                    "provider_quote_hash": quote["source_hash"]}
        snapshot_hash = _digest(snapshot)
        observation_id = f"score-prediction:{sport}:{market}:{_digest([model['model_id'], snapshot_hash])[:24]}"
        evidence.insert_prediction(path, {"observation_id": observation_id,
            "event_id": event_id, "sport": sport, "market_family": market,
            "selection": quote["selection"], "quote_id": quote["quote_id"],
            "model_id": model["model_id"], "model_version": model["model_version"],
            "feature_version": FEATURE_VERSION, "feature_snapshot_id": snapshot_hash,
            "feature_snapshot": snapshot, "feature_frozen_at": now.isoformat(),
            "prediction_timestamp": now.isoformat(), "mean_probability": probabilities["win"],
            "conservative_probability": conservative,
            "push_probability": probabilities["push"],
            "loss_probability": probabilities["loss"],
            "probability_semantics": "WIN_PUSH_LOSS", "policy_version": MODEL_VERSION,
            "evidence_snapshot_id": snapshot_hash, "evidence_hash": _digest([
                event["source_hash"], quote["source_hash"], lineage]),
            "runtime_hash": runtime_hash(), "source_commit": source_commit(),
            "uncertainty": {"method": "one_sided_normal_approximation_v1",
                            "training_independent_events": artifact["training_independent_events"],
                            "standard_error": standard_error,
                            "feature_limitations": FEATURE_LIMITATIONS[sport],
                            "research_only": True, "market_settlement_certified": False}})
        created += 1
    return {"sport": sport, "market_family": market, "predictions": created,
            "skipped_insufficient_team_history": no_history,
            "model_id": model["model_id"], "blocker": None,
            "production_eligible": False, "recommended_stake": 0.0}


def _native_bound(capture, event, sport, now):
    """Require the retained discovery, participant and odds responses to agree."""
    try:
        data = capture["data"]
        if data.get("protocol") != PROTOCOL or data.get("sport") != sport:
            return False
        participant_source = data["participants_source"]
        if native_digest(participant_source["raw_source"]) != participant_source["source_hash"]:
            return False
        participants = participant_ids(participant_source["raw_source"])
        source = event["discovery_source"]
        selected = {key: event[key] for key in ("event_id", "home", "away", "start")}
        if (native_digest(source["raw_source"]) != source["source_hash"] or
                identity(sport, source["raw_source"]) != selected or
                native_digest(event["raw_source"]) != event["source_hash"] or
                identity(sport, event["raw_source"]) != selected or
                event["provider_namespace"] != "THE_ODDS_API" or
                event["provider_event_id"] != event["event_id"] or
                participants.get(event["home"]) != event["home_team_id"] or
                participants.get(event["away"]) != event["away_team_id"] or
                event["home_team_id"] == event["away_team_id"]):
            return False
        observed = _time(event["response_received_at"])
        return (all(_time(item) <= observed for item in (
                    source["observed_at"], participant_source["observed_at"])) and
                observed <= now < _time(event["start"]))
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def ingest_native(path, canonical_path, sport):
    """Append current replayable native captures and score-only revisions."""
    if sport not in SCOPES:
        raise ValueError("unsupported research sport")
    records = for_sport(sport).records(path)
    now = _utcnow()
    counts = {"canonical_events": 0, "canonical_quotes": 0, "canonical_results": 0,
              "stale_or_started_captures": 0, "identity_blocked": 0,
              "quote_replay_blocked": 0, "score_replay_blocked": 0}
    events = {row["provider_event_id"]: row for row in evidence.read_records(
        canonical_path, "prospective_event", sport=sport) if row["provider_namespace"] == "THE_ODDS_API"}
    quote_ids = {row["quote_id"] for market in SCOPES[sport] for row in evidence.read_records(
        canonical_path, "prospective_quote", sport=sport, market_family=market)}
    for record in records:
        if record["kind"] != "capture":
            continue
        for native in record["data"].get("events", []):
            provider_id = native.get("event_id")
            if provider_id in events:
                event_id = events[provider_id]["event_id"]
            elif not _native_bound(record, native, sport, now):
                try:
                    started = _time(native["start"]) <= now
                except (KeyError, TypeError, ValueError):
                    started = False
                counts["stale_or_started_captures" if started else "identity_blocked"] += 1
                continue
            else:
                event_id = f"odds-event:{sport}:{_digest(provider_id)[:24]}"
                evidence.insert_event(canonical_path, {"event_id": event_id, "sport": sport,
                    "game_id": provider_id, "provider_namespace": "THE_ODDS_API",
                    "provider_event_id": provider_id, "home_team": native["home"],
                    "away_team": native["away"], "home_team_id": native["home_team_id"],
                    "away_team_id": native["away_team_id"], "scheduled_start": native["start"],
                    "observed_at": native["response_received_at"],
                    "source_id": f"native-capture:{record['id']}:{provider_id}",
                    "raw_source": native["raw_source"]})
                counts["canonical_events"] += 1
                events[provider_id] = evidence.load_record(canonical_path, "prospective_event", event_id)
            if _time(native["start"]) <= now:
                continue
            for quote in native.get("quotes", []):
                if quote.get("market_family") not in SCOPES[sport] or quote.get("quote_verified") is not True:
                    continue
                quote_id = f"odds-quote:{sport}:{_digest([provider_id, quote.get('source_id')])[:24]}"
                if quote_id in quote_ids:
                    continue
                try:
                    evidence.insert_quote(canonical_path, {"quote_id": quote_id,
                        "event_id": event_id, "sport": sport,
                        "market_family": quote["market_family"], "selection": quote["selection"],
                        "line": quote["line"], "american_odds": quote["american_odds"],
                        "decimal_odds": quote["decimal_odds"], "sportsbook": quote["sportsbook"],
                        "quote_timestamp": quote["quote_timestamp"],
                        "quote_source": quote["quote_source"], "quote_verified": True,
                        "source_id": quote["source_id"], "raw_source": quote["raw_source"]})
                except evidence.EvidenceConflict:
                    raise
                except ValueError:
                    counts["quote_replay_blocked"] += 1
                    continue
                counts["canonical_quotes"] += 1
                quote_ids.add(quote_id)
    canonical_events = {row["provider_event_id"]: row for row in evidence.read_records(
        canonical_path, "prospective_event", sport=sport) if row["provider_namespace"] == "THE_ODDS_API"}
    existing_results = defaultdict(list)
    for row in evidence.read_records(canonical_path, "prospective_result", sport=sport):
        if row["market_family"] is None:
            existing_results[row["event_id"]].append(row)
    score_records = [record for record in records if record["kind"] == "scores" and
                     record["data"].get("protocol") == PROTOCOL]
    score_records.sort(key=lambda row: (row["data"].get("event", {}).get("event_id", ""),
                                        row["data"].get("event", {}).get("grading_version", 0),
                                        row["id"]))
    for record in score_records:
        native = record["data"].get("event", {})
        event = canonical_events.get(native.get("event_id"))
        if event is None:
            continue  # no backdated event creation from an old native capture
        result_id = f"odds-score:{sport}:{record['id'][:24]}"
        if any(row["result_id"] == result_id for row in existing_results[event["event_id"]]):
            continue
        previous = max(existing_results[event["event_id"]], key=lambda row: row["grading_version"],
                       default=None)
        if previous is not None and previous["source_hash"] == native.get("source_hash"):
            continue
        candidate = {"result_id": result_id, "event_id": event["event_id"],
                     "sport": sport, "market_family": None, "selection": None,
                     "result_source": "THE_ODDS_API", "result_source_id": event["provider_event_id"],
                     "observed_at": native.get("observed_at"),
                     "available_at": native.get("available_at"),
                     "home_score": native.get("home_score"),
                     "away_score": native.get("away_score"), "outcome": None,
                     "grading_version": previous["grading_version"] + 1 if previous else 1,
                     "revises_result_id": previous["result_id"] if previous else None,
                     "raw_source": native.get("raw_source")}
        try:
            source_bound = native_digest(native.get("raw_source")) == native.get("source_hash")
        except (TypeError, ValueError):
            source_bound = False
        if not source_bound or not evidence.verify_provider_score(event, candidate):
            counts["score_replay_blocked"] += 1
            continue
        evidence.insert_result(canonical_path, candidate)
        counts["canonical_results"] += 1
        existing_results[event["event_id"]].append(
            evidence.load_record(canonical_path, "prospective_result", result_id))
    return counts


def run_sport_model_cycle(native_path, canonical_path, sport):
    """Post-capture hook: reconcile, fit if legal, then freeze live predictions."""
    report = {"sport": sport, "production_eligible": False, "recommended_stake": 0.0,
              "models_fitted": 0, "models_available": 0, "predictions": 0,
              "markets": {}, "blockers": []}
    report.update(ingest_native(native_path, canonical_path, sport))
    for market in SCOPES[sport]:
        trained = train_scope(canonical_path, sport, market)
        if trained["status"] == "FITTED_RESEARCH_ONLY":
            report["models_available"] += 1
            report["models_fitted"] += int(trained["new_model"])
        else:
            report["blockers"].append(f"{market}_{trained['status']}")
        predicted = predict_scope(canonical_path, sport, market)
        report["predictions"] += predicted["predictions"]
        report["markets"][market] = {
            "model_status": trained["status"], "model_id": trained.get("model_id"),
            "model_artifact_hash": trained.get("artifact_hash"),
            "training_cutoff": trained.get("training_cutoff"),
            "independent_events": trained.get("independent_events", 0),
            "calibration_status": trained.get("calibration_status", "INSUFFICIENT_EVIDENCE"),
            "predictions": predicted["predictions"],
            "validation_diagnostics": trained.get("validation"),
            "production_eligible": False, "recommended_stake": 0.0}
        if predicted["blocker"]:
            report["blockers"].append(f"{market}_{predicted['blocker']}")
        market_results = evidence.read_records(canonical_path, "prospective_result", sport=sport,
                                               market_family=market)
        if not any(row["outcome"] in {"WIN", "LOSS", "PUSH", "VOID"} for row in market_results):
            report["blockers"].append(f"{market}_MISSING_AUTHENTIC_MARKET_SETTLEMENT")
        if not evidence.read_records(canonical_path, "prospective_calibration", sport=sport,
                                     market_family=market):
            report["blockers"].append(f"{market}_MISSING_SPORT_MARKET_CALIBRATION")
        if not any(row["close_verified"] == 1 for row in evidence.read_records(
                canonical_path, "prospective_close", sport=sport, market_family=market)):
            report["blockers"].append(f"{market}_NO_VERIFIED_CLOSE_QUOTES")
    report["blockers"] = sorted(set(report["blockers"]))
    report["feature_limitations"] = list(FEATURE_LIMITATIONS[sport])
    return report
