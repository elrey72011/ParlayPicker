"""MLB point-in-time fitted challenger. No production promotion or calibration.

Inputs are immutable, contemporaneously captured JSON receipts, not backfilled
aggregate CSVs. Hashes establish integrity, not source authenticity: receipt
collection/storage remains a trusted boundary and must be audited separately.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from app_core.mlb_history import timestamp
from app_core.mlb_research import FEATURES

SCHEMA = "mlb-pregame-receipt-v1"
FEATURE_SCHEMA = "mlb-prior-scoring-line-v1"
TARGETS = ("spread_home", "spread_away", "total_over", "total_under")
SEMANTICS = "win_conditional_on_decision"
FEATURE_COLUMNS = FEATURES + ["reference_line"]
CONFIG = {"ridge": 1.0, "iterations": 100, "tolerance": 1e-10,
          "minimum_prior_games": 10, "minimum_rows_per_family_split": 20}
BASIS = "max outcome available_at among fitted decided training rows; no validation/holdout fitting or tuning"


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def utcnow():
    return datetime.now(timezone.utc)


def finite(value):
    if isinstance(value, bool):
        raise ValueError("boolean numeric input")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("nonfinite input")
    return result


def identity(payload):
    if not isinstance(payload, dict):
        raise ValueError("identity object required")
    keys = ("provider_namespace", "provider_event_id", "home_team_id", "away_team_id", "season", "game_start_utc")
    if payload.get("provider_namespace") != "mlb" or any(not str(payload.get(k, "")).strip() for k in keys):
        raise ValueError("MLB provider-scoped identity required")
    if payload["home_team_id"] == payload["away_team_id"]:
        raise ValueError("same team")
    return tuple(str(payload[k]) for k in keys)


def receipt_features(receipt):
    """Recompute every feature from earlier observed completed games only."""
    if not isinstance(receipt, dict) or not isinstance(receipt.get("payload"), dict):
        raise ValueError("receipt object required")
    p = receipt["payload"]
    if receipt.get("sha256") != digest(p) or p.get("schema_version") != SCHEMA:
        raise ValueError("receipt integrity/schema mismatch")
    identity(p)
    cutoff, observed, start = (timestamp(p[k]) for k in ("prediction_cutoff", "captured_at", "game_start_utc"))
    if not observed <= cutoff < start or int(p["season"]) != start.year:
        raise ValueError("non-pregame receipt")
    quote = p["quote"]
    market = quote["market_type"]
    if market not in TARGETS or not quote.get("sportsbook"):
        raise ValueError("unsupported target or missing book")
    if not timestamp(quote["observed_at"]) <= observed:
        raise ValueError("later quote")
    line, price = finite(quote["line"]), finite(quote["decimal_odds"])
    if price <= 1 or line * 2 != int(line * 2) or (market.startswith("total") and line <= 0):
        raise ValueError("invalid exact line/price")
    games = p["prior_games"]
    seen = set()
    for g in games:
        gid = str(g["game_id"])
        if gid in seen or gid == str(p["provider_event_id"]) or g.get("provider_namespace") != "mlb":
            raise ValueError("duplicate/target/unscoped feature game")
        seen.add(gid)
        if not timestamp(g["completed_at"]) <= timestamp(g["available_at"]) <= observed or timestamp(g["completed_at"]) >= cutoff:
            raise ValueError("future/unavailable feature game")
        if (int(g["season"]) != int(p["season"]) or g["home_id"] == g["away_id"]
            or g.get("status") != "FINAL" or g["home_score"] == g["away_score"]):
            raise ValueError("invalid feature identity")
        for k in ("home_score", "away_score"):
            v = finite(g[k])
            if v < 0 or v != int(v):
                raise ValueError("invalid feature score")
    values = {}
    for side in ("home", "away"):
        team = p[side + "_team_id"]
        prior = [g for g in games if team in (g["home_id"], g["away_id"])]
        if len(prior) < CONFIG["minimum_prior_games"]:
            raise ValueError("insufficient observed prior games")
        scores, allowed = [], []
        for g in prior:
            own = "home" if g["home_id"] == team else "away"
            other = "away" if own == "home" else "home"
            scores.append(float(g[own + "_score"]))
            allowed.append(float(g[other + "_score"]))
        values[side + "_ppg"] = float(np.mean(scores))
        values[side + "_oppg"] = float(np.mean(allowed))
        values[side + "_win_pct"] = float(np.mean(np.array(scores) > allowed))
    values["reference_line"] = -line if market == "spread_away" else line
    return p, [values[k] for k in FEATURE_COLUMNS]


def label(market, line, home, away, status="FINAL"):
    if market not in TARGETS:
        raise ValueError("Moneyline/unknown target")
    if status == "VOID":
        return "VOID"
    if status != "FINAL":
        raise ValueError("unsettled target")
    h, a, line = finite(home), finite(away), finite(line)
    if min(h, a) < 0 or h != int(h) or a != int(a) or h == a:
        raise ValueError("invalid final score")
    delta = {"spread_home": h-a+line, "spread_away": a-h+line,
             "total_over": h+a-line, "total_under": line-h-a}[market]
    return "WIN" if delta > 0 else "LOSS" if delta < 0 else "PUSH"


def prepare_rows(records):
    rows, seen, events = [], set(), {}
    for record in records:
        p, x = receipt_features(record["snapshot"])
        o = record["outcome"]
        event = identity(p)
        if identity(o) != event or timestamp(o["available_at"]) < timestamp(p["game_start_utc"]):
            raise ValueError("outcome identity/availability mismatch")
        event_key = event[:2]
        if event_key in events and events[event_key] != (event, o):
            raise ValueError("conflicting event identity/outcome")
        events[event_key] = (event, o)
        q = p["quote"]
        # One snapshot per event/target: later alternate lines cannot overweight fitting.
        key = event_key + (q["market_type"],)
        if key in seen:
            raise ValueError("duplicate training event/target")
        seen.add(key)
        outcome = label(q["market_type"], q["line"], o.get("home_score"), o.get("away_score"), o["status"])
        start = timestamp(p["game_start_utc"])
        baselines = {"market_implied": 1 / finite(q["decimal_odds"])}
        for name in ("deterministic", "configured_blend"):
            b = p.get("baselines", {}).get(name)
            if b is not None:
                if b.get("probability_semantics") != SEMANTICS or timestamp(b["generated_at"]) > timestamp(p["captured_at"]):
                    raise ValueError("baseline scope/time mismatch")
                value = finite(b["probability"])
                if not 0 < value < 1 or not b.get("source_version"):
                    raise ValueError("invalid baseline provenance")
                baselines[name] = value
        rows.append({"event": event_key, "market": q["market_type"], "family": q["market_type"].split("_")[0],
                     "start": start.isoformat(), "slate": str(start.astimezone(ZoneInfo("America/New_York")).date()),
                     "season": int(p["season"]), "outcome_at": timestamp(o["available_at"]).isoformat(),
                     "cutoff": p["prediction_cutoff"], "x": x, "outcome": outcome,
                     "price": finite(q["decimal_odds"]), "baselines": baselines,
                     "reference_outcome": ({"WIN": "LOSS", "LOSS": "WIN"}.get(outcome, outcome)
                         if q["market_type"] in {"spread_away", "total_under"} else outcome),
                     "receipt_hash": record["snapshot"]["sha256"]})
    return sorted(rows, key=lambda r: (r["start"], r["event"], r["market"]))


def split_rows(rows, train_through, validation_through):
    # Explicit whole Eastern-date boundaries, no random splitting or tuning.
    from datetime import date
    train_day, valid_day = date.fromisoformat(train_through), date.fromisoformat(validation_through)
    if train_day >= valid_day:
        raise ValueError("invalid chronological boundaries")
    splits = {"train": [], "validation": [], "holdout": []}
    for r in rows:
        key = "train" if r["slate"] <= train_through else "validation" if r["slate"] <= validation_through else "holdout"
        splits[key].append(r)
    if any(not v for v in splits.values()):
        raise ValueError("three chronological periods required")
    for before, after in (("train", "validation"), ("validation", "holdout")):
        if max(timestamp(r["start"]) for r in splits[before]) >= min(timestamp(r["start"]) for r in splits[after]):
            raise ValueError("overlapping event times")
        if max(timestamp(r["outcome_at"]) for r in splits[before]) >= min(timestamp(r["cutoff"]) for r in splits[after]):
            raise ValueError("labels unavailable before future evaluation")
    return splits


def fit_estimator(rows):
    """Fixed ridge-logistic IRLS. Scaling and fitting read development rows only."""
    x = np.array([r["x"] for r in rows], dtype=float)
    y = np.array([r["reference_outcome"] == "WIN" for r in rows], dtype=float)
    if len(set(y)) != 2:
        raise ValueError("both decided outcomes required")
    mean, scale = x.mean(0), x.std(0)
    scale[scale == 0] = 1
    z = np.column_stack([np.ones(len(x)), (x-mean)/scale])
    beta = np.zeros(z.shape[1])
    penalty = np.diag([0.] + [CONFIG["ridge"]] * x.shape[1])
    for _ in range(CONFIG["iterations"]):
        p = 1/(1+np.exp(-np.clip(z@beta, -30, 30)))
        w = np.maximum(p*(1-p), 1e-10)
        step = np.linalg.solve(z.T@(w[:, None]*z)+penalty, z.T@(y-p)-penalty@beta)
        beta += step
        if np.max(np.abs(step)) < CONFIG["tolerance"]:
            break
    else:
        raise ValueError("fitting did not converge")
    return {"mean": mean.tolist(), "scale": scale.tolist(), "coefficients": beta.tolist()}


def probabilities(estimator, rows):
    x = np.array([r["x"] for r in rows], dtype=float)
    z = np.column_stack([np.ones(len(x)), (x-estimator["mean"])/estimator["scale"]])
    reference = 1/(1+np.exp(-np.clip(z@estimator["coefficients"], -30, 30)))
    return [float(1-p if r["market"] in {"spread_away", "total_under"} else p) for r, p in zip(rows, reference)]


def metrics(rows, ps):
    y = np.array([r["outcome"] == "WIN" for r in rows], dtype=float)
    p = np.clip(np.array(ps), 1e-12, 1-1e-12)
    return {"sample_count": len(rows), "unique_events": len({tuple(r["event"]) for r in rows}), "brier": float(np.mean((p-y)**2)),
            "log_loss": float(np.mean(-y*np.log(p)-(1-y)*np.log(1-p))),
            "accuracy": float(np.mean((p >= .5) == y)),
            "roi_decided_selections_unit_stake": float(np.mean([r["price"]-1 if r["outcome"] == "WIN" else -1 for r in rows])),
            "clv": None, "clv_reason": "no verified closing observation in this schema"}


def evaluate(models, rows):
    result = {}
    for family in ("spread", "total"):
        subset = [r for r in rows if r["family"] == family and r["outcome"] in {"WIN", "LOSS"}]
        if len(subset) < CONFIG["minimum_rows_per_family_split"]:
            raise ValueError("insufficient decided rows per family/period")
        p = probabilities(models[family], subset)
        report = metrics(subset, p)
        report["outcomes"] = dict(Counter(r["outcome"] for r in rows if r["family"] == family))
        report["baselines"] = {}
        for name in ("market_implied", "deterministic", "configured_blend"):
            paired = [(r, q) for r, q in zip(subset, p) if name in r["baselines"]]
            report["baselines"][name] = ({"challenger": metrics([r for r, _ in paired], [q for _, q in paired]),
                "baseline": metrics([r for r, _ in paired], [r["baselines"][name] for r, _ in paired])}
                if paired else {"sample_count": 0, "reason": "no contemporaneous comparable baseline"})
        report["by_season"] = {str(season): metrics([r for r in subset if r["season"] == season],
            [q for r, q in zip(subset, p) if r["season"] == season]) for season in sorted({r["season"] for r in subset})}
        report["by_probability_band"] = {}
        for lower, upper in ((0., .4), (.4, .5), (.5, .6), (.6, .7), (.7, 1.)):
            pairs = [(r, q) for r, q in zip(subset, p) if lower <= q < upper]
            report["by_probability_band"][f"{lower}-{upper}"] = metrics([r for r, _ in pairs], [q for _, q in pairs]) if pairs else {"sample_count": 0}
        result[family] = report
    return result


def model_version(manifest):
    return digest({k: v for k, v in manifest.items() if k != "model_version"})


def train(records, output, *, source, train_through, validation_through):
    started = utcnow()
    if not source.strip():
        raise ValueError("training source required")
    rows = prepare_rows(records)
    if any(timestamp(r["outcome_at"]) > started for r in rows):
        raise ValueError("future training/evaluation data")
    splits = split_rows(rows, train_through, validation_through)
    models, fitted_rows, consumed = {}, {}, []
    for family in ("spread", "total"):
        decided = [r for r in splits["train"] if r["family"] == family and r["outcome"] in {"WIN", "LOSS"}]
        unique = {}
        for r in decided:
            key = tuple(r["event"]) + (r["x"][-1],)
            if key in unique and (unique[key]["x"] != r["x"] or unique[key]["reference_outcome"] != r["reference_outcome"]):
                raise ValueError("conflicting complementary training receipts")
            unique[key] = r
        decided = list(unique.values())
        if len(decided) < CONFIG["minimum_rows_per_family_split"]:
            raise ValueError("insufficient point-in-time training rows")
        models[family] = fit_estimator(decided)
        fitted_rows[family] = len(decided)
        consumed.extend(decided)
    validation = evaluate(models, splits["validation"])
    holdout = evaluate(models, splits["holdout"])
    body = canonical(models)
    periods = {name: {"start": part[0]["start"], "end": part[-1]["start"], "rows": len(part),
                      "slates": len({r["slate"] for r in part})} for name, part in splits.items()}
    manifest = {"schema_version": 1, "sport": "MLB", "market_families": ["spread", "total"],
        "model_family": "separate ridge logistic classifiers for home cover and over at the exact reference line; away/under use complements",
        "feature_schema_version": FEATURE_SCHEMA, "feature_columns": FEATURE_COLUMNS, "targets": list(TARGETS),
        "probability_semantics": SEMANTICS, "training_data_source": source, "dataset_hash": digest(records),
        "training_rows": fitted_rows,
        "training_slates": periods["train"]["slates"], "training_started_at": started.isoformat(),
        "model_trained_through": max(timestamp(r["outcome_at"]) for r in consumed).isoformat(),
        "training_cutoff_basis": BASIS, "config": CONFIG, "periods": periods,
        "validation_period": periods["validation"], "validation_metrics": validation, "holdout_metrics": holdout,
        "artifact_files": ["estimators.json"], "artifact_hashes": {"estimators.json": hashlib.sha256(body).hexdigest()},
        "maturity": "RESEARCH", "production_eligible": False, "recommended_stake": 0,
        "calibration_attached": False, "evaluation_semantics": "historical OOS replay, not prospective model-availability evidence",
        "runtime": {"numpy": np.__version__, "trainer_source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    # Stage exact fitted bytes first; availability is the actual bundle completion
    # receipt, never a historical prediction/capture/mtime substitute.
    import tempfile
    root = Path(output); root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".building-", dir=root) as stage:
        staged = Path(stage)
        (staged/"estimators.json").write_bytes(body)
        manifest["model_available_at"] = utcnow().isoformat()
        manifest["model_version"] = model_version(manifest)
        (staged/"manifest.json").write_bytes(canonical(manifest))
        destination = root/manifest["model_version"]
        destination.mkdir(exist_ok=False)
        # Readers reject an incomplete bundle; existing versions are never overwritten.
        (staged/"estimators.json").replace(destination/"estimators.json")
        (staged/"manifest.json").replace(destination/"manifest.json")
    return destination


def validate_manifest(m):
    if not isinstance(m, dict):
        raise ValueError("manifest object required")
    if (m.get("model_version") != model_version(m)
        or m.get("schema_version") != 1 or m.get("sport") != "MLB"
        or m.get("feature_schema_version") != FEATURE_SCHEMA or m.get("feature_columns") != FEATURE_COLUMNS
        or m.get("targets") != list(TARGETS) or m.get("probability_semantics") != SEMANTICS
        or m.get("maturity") != "RESEARCH" or m.get("production_eligible") is not False
        or m.get("calibration_attached") is not False or m.get("recommended_stake") != 0
        or m.get("training_cutoff_basis") != BASIS or m.get("config") != CONFIG):
        raise ValueError("model manifest integrity/schema/scope mismatch")
    if not timestamp(m["model_trained_through"]) <= timestamp(m["training_started_at"]) <= timestamp(m["model_available_at"]):
        raise ValueError("invalid model chronology")


def load_mlb_spread_total_model(path):
    path = Path(path)
    m = json.loads((path/"manifest.json").read_text(encoding="utf-8"))
    validate_manifest(m)
    if path.name != m["model_version"]:
        raise ValueError("version directory mismatch")
    if m["artifact_files"] != ["estimators.json"]:
        raise ValueError("unexpected artifact")
    raw = (path/"estimators.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != m["artifact_hashes"]["estimators.json"]:
        raise ValueError("model artifact digest mismatch")
    models = json.loads(raw)
    if set(models) != {"spread", "total"}:
        raise ValueError("model family mismatch")
    for model in models.values():
        for key, size in (("mean", len(FEATURE_COLUMNS)), ("scale", len(FEATURE_COLUMNS)), ("coefficients", len(FEATURE_COLUMNS)+1)):
            v = np.asarray(model[key], dtype=float)
            if v.shape != (size,) or not np.isfinite(v).all() or (key == "scale" and (v <= 0).any()):
                raise ValueError("invalid fitted parameters")
    return {"manifest": m, "models": models}


def predict_mlb_spread_total(bundle, receipt, *, prediction_generated_at, calibration=None):
    # Caller must load and verify bundle for inference. Revalidate in-memory hashes too.
    m, models = bundle["manifest"], bundle["models"]
    validate_manifest(m)
    if model_version(m) != m["model_version"] or digest(models) != m["artifact_hashes"]["estimators.json"]:
        raise ValueError("in-memory model changed")
    if calibration is not None:
        raise ValueError("no model/version-scoped OOS calibration attached; blend calibration is incompatible")
    p, x = receipt_features(receipt)
    generated = timestamp(prediction_generated_at)
    if not timestamp(m["model_available_at"]) <= generated < timestamp(p["game_start_utc"]) or timestamp(p["captured_at"]) > generated or timestamp(p["prediction_cutoff"]) > generated:
        raise ValueError("model/feature unavailable or game started")
    family = p["quote"]["market_type"].split("_")[0]
    probability = probabilities(models[family], [{"x": x, "market": p["quote"]["market_type"]}])[0]
    return {"probability": probability, **{k: m[k] for k in ("model_version", "model_trained_through", "model_available_at", "training_cutoff_basis", "probability_semantics")},
            "prediction_generated_at": generated.isoformat(), "receipt_hash": receipt["sha256"],
            "candidate_maturity": "RESEARCH", "production_eligible": False, "production_bet_amount": 0.0}


def attach_challenger(frame, *, model_path=None, now=None):
    """Namespaced diagnostics only: never attach model facts to baseline probabilities."""
    import os
    out = frame.copy()
    path = model_path or os.environ.get("PARLAYPICKER_MLB_CHALLENGER_MODEL")
    out["mlb_challenger_result"] = None
    out["mlb_challenger_status"] = "NOT_CONFIGURED"
    if not path or out.empty:
        return out
    try:
        bundle = load_mlb_spread_total_model(path)
    except (OSError, ValueError, KeyError, TypeError):
        out["mlb_challenger_status"] = "ARTIFACT_UNVERIFIED"
        return out
    for idx, row in out.iterrows():
        if str(row.get("sport", row.get("league", row.get("League", "")))).upper() != "MLB":
            continue
        try:
            receipt = row.get("mlb_pregame_receipt")
            if isinstance(receipt, str):
                receipt = json.loads(receipt)
            p = receipt["payload"]
            # Exact event/selection binding, not team names or positional alignment.
            for key in ("provider_namespace", "provider_event_id", "home_team_id", "away_team_id", "game_start_utc"):
                if str(row.get(key)) != str(p[key]):
                    raise ValueError("challenger event mismatch")
            q = p["quote"]
            if row["market_type"] != q["market_type"]:
                raise ValueError("challenger target mismatch")
            line = row.get("line", row.get("spread_line" if q["market_type"].startswith("spread") else "total_line"))
            if finite(line) != finite(q["line"]):
                raise ValueError("challenger line mismatch")
            result = predict_mlb_spread_total(bundle, receipt, prediction_generated_at=(now or utcnow()).isoformat())
            out.at[idx, "mlb_challenger_result"] = json.dumps(result, sort_keys=True)
            out.at[idx, "mlb_challenger_status"] = "RESEARCH"
        except (ValueError, KeyError, TypeError):
            out.at[idx, "mlb_challenger_status"] = "PREGAME_EVIDENCE_UNAVAILABLE"
    return out
