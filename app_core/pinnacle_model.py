"""Pinnacle's no-vig price as a registered model (C5; D4 option A; D19 option B).

Team form alone does not beat the line (NCAAF D18, NCAAB line screen), and the Pinnacle
value board needs a plan to qualify. This puts Pinnacle's price through the same chain
as any model: register it, calibrate it, bind it to a reviewed plan (``market_skill``
methodology: no worse than the US book's own no-vig price, and positive ROI and CLV on
the bets), then the gate, the owner's activation and Gemini.

* **Model** (``register``): ``prospective_model`` with an artifact of kind
  ``KIND``. It has no training games (it is a price, not a fit), so it is never the
  research cycle's pinned candidate (``prospective_calibration.candidate_model`` needs
  120 training games).
* **Probability** (``fair``): for a US-book quote, Pinnacle's no-vig probability for the
  same side at the **same home-perspective line**, from Pinnacle's two sides in one
  capture within ``PAIR_WINDOW`` of the US quote and no later than the prediction time.
  No line translation: a game without Pinnacle at that line gets no prediction.
* **Calibration** (``fit_calibration``): Platt, fitted from games settled after the model
  was registered, at the quote each game was settled on (the research cycle's first
  quote), using stored quotes only. It records **no predictions**, so it never enters,
  or blocks, a plan bound to another model.
* **Predictions** (``predict``): only through ``prospective_calibration._predict`` when a
  bound plan names this model (``predict_bound``, decision capture), so a market holds
  one plan's predictions at a time.

Research only until a plan binds it and the owner activates the market; no stake.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from app_core import prospective_evidence as evidence

KIND = "pinnacle_no_vig_v1"
FEATURE_VERSION = "pinnacle-no-vig-v1"
REFERENCE_BOOK = "pinnacle"
PAIR_WINDOW = timedelta(minutes=30)
FAMILIES = {"NBA": ("SPREAD", "TOTAL"), "NCAAB": ("SPREAD", "TOTAL")}


def _t(value):
    return evidence._time(value, "timestamp")


def artifact(sport, family):
    return {"kind": KIND, "sport": sport, "market_family": family, "reference_book": REFERENCE_BOOK,
            "line_rule": "SAME_HOME_PERSPECTIVE_LINE", "pair_window_minutes": int(PAIR_WINDOW.total_seconds() // 60),
            "probability": "NO_VIG_TWO_SIDED", "training": "NONE_PRICE_NOT_FIT"}


def is_pinnacle(model):
    import json
    try:
        return (json.loads(model["payload"]).get("model_artifact") or {}).get("kind") == KIND
    except (TypeError, ValueError, KeyError):
        return False


def registered(path, sport, family):
    """The registered Pinnacle model for this market, or None."""
    found = [m for m in evidence.read_records(path, "prospective_model", sport=sport, market_family=family)
             if is_pinnacle(m)]
    return min(found, key=lambda m: (m["available_at"], m["model_id"])) if found else None


def register(path, sport, family, *, now=None, commit="UNAVAILABLE"):
    """Register the Pinnacle model once per market. Returns the model record."""
    if family not in FAMILIES.get(sport, ()):
        raise ValueError("Pinnacle model is for NBA/NCAAB spreads and totals")
    existing = registered(path, sport, family)
    if existing is not None:
        return existing
    now = now or datetime.now(timezone.utc)
    body = artifact(sport, family)
    digest = evidence._sha(evidence._json(body))
    model_id = f"pinnacle-model:{sport}:{family}:{digest[:16]}"
    evidence.insert_model(path, {
        "model_id": model_id, "sport": sport, "market_family": family, "model_version": KIND,
        "training_start": (now - timedelta(seconds=1)).isoformat(), "training_cutoff": now.isoformat(),
        "training_observation_count": 0, "independent_event_count": 0, "feature_version": FEATURE_VERSION,
        "training_code_commit": commit, "created_at": now.isoformat(), "available_at": now.isoformat(),
        "artifact_hash": digest, "model_artifact": body, "training_result_ids": []})
    return evidence.load_record(path, "prospective_model", model_id)


def _home_line(event, quote):
    if quote["market_family"] == "TOTAL" or quote["selection"] == event["home_team"]:
        return round(float(quote["line"]), 2)
    return round(-float(quote["line"]), 2)


def _canonical(event, family):
    return event["home_team"] if family != "TOTAL" else "Over"


def pairs(event, quotes, family):
    """[(quote_timestamp, home_line, p_canonical)] for Pinnacle's two-sided quotes on this event."""
    by_time = {}
    for q in quotes:
        if (q["event_id"] != event["event_id"] or q["sportsbook"] != REFERENCE_BOOK or q["market_family"] != family
                or not q["quote_verified"] or q["line"] is None):
            continue
        by_time.setdefault((q["quote_timestamp"], _home_line(event, q)), {})[q["selection"]] = q
    out = []
    canonical = _canonical(event, family)
    for (stamp, line), sides in by_time.items():
        if len(sides) != 2 or canonical not in sides:
            continue
        mine = 1 / sides[canonical]["decimal_odds"]
        other = 1 / next(q for s, q in sides.items() if s != canonical)["decimal_odds"]
        out.append((stamp, line, mine / (mine + other)))
    return out


def fair(event, quote, quotes, *, now):
    """Pinnacle's no-vig probability for ``quote``'s side at its exact line, or None."""
    line, at = _home_line(event, quote), _t(quote["quote_timestamp"])
    best = None
    for stamp, home_line, p in pairs(event, quotes, quote["market_family"]):
        when = _t(stamp)
        if home_line != line or when > now or abs(when - at) > PAIR_WINDOW:
            continue
        if best is None or abs(when - at) < abs(_t(best[0]) - at):
            best = (stamp, p)
    if best is None:
        return None
    p = best[1]
    return p if quote["selection"] == _canonical(event, quote["market_family"]) else 1 - p


def fit_calibration(path, sport, family, model, *, now=None, min_decided=None):
    """A Platt calibration for the Pinnacle model from settled games, recording no predictions."""
    from app_core import prospective_calibration as calibration
    from app_core import prospective_research_models as research
    now = now or datetime.now(timezone.utc)
    min_decided = calibration.MIN_CALIBRATION_DECIDED if min_decided is None else min_decided
    existing = [c for c in evidence.read_records(path, "prospective_calibration", sport=sport, market_family=family)
                if c["model_id"] == model["model_id"]]
    if existing:
        return {"status": "PRESENT", "calibration_id": max(existing, key=lambda c: c["available_at"])["calibration_id"]}
    fit_start = _t(model["available_at"])
    events = {e["event_id"]: e for e in evidence.read_records(path, "prospective_event", sport=sport)}
    quotes = evidence.read_records(path, "prospective_quote", sport=sport, market_family=family)
    by_id = {q["quote_id"]: q for q in quotes}
    results = calibration._market_result(path, sport, family)
    pairs_, result_ids = [], []
    for event_id, prediction in sorted(calibration._earliest_predictions(path, sport, family).items()):
        result = results.get((event_id, prediction["selection"]))
        quote, event = by_id.get(prediction["quote_id"]), events.get(event_id)
        if result is None or quote is None or event is None or not fit_start <= _t(result["available_at"]) <= now:
            continue
        p = fair(event, quote, quotes, now=_t(event["scheduled_start"]))
        if p is None or result["outcome"] not in ("WIN", "LOSS"):
            continue
        result_ids.append(result["result_id"])
        pairs_.append((p, int(result["outcome"] == "WIN")))
    if len(pairs_) < min_decided:
        return {"status": "INSUFFICIENT_EVIDENCE", "calibration_id": None, "decided": len(pairs_),
                "required_decided": min_decided}
    a, b = calibration.platt_fit(pairs_)
    fit_end = max(_t(evidence.load_record(path, "prospective_result", rid)["available_at"]) for rid in result_ids)
    body = {"method": calibration.CALIBRATION_VERSION, "a": a, "b": b, "decided": len(pairs_),
            "model_id": model["model_id"], "sport": sport, "market_family": family,
            "fit_start": fit_start.isoformat(), "fit_end": fit_end.isoformat(), "fit_result_ids": result_ids,
            "applies_to": "decided_win_probability", "source": "STORED_PINNACLE_QUOTES_AT_SETTLED_QUOTE"}
    digest = research._digest(body)
    calibration_id = f"platt-calibration:{sport}:{family}:{digest[:24]}"
    created = max(now, fit_end).isoformat()
    evidence.insert_calibration(path, {
        "calibration_id": calibration_id, "sport": sport, "market_family": family,
        "calibration_version": calibration.CALIBRATION_VERSION, "model_id": model["model_id"],
        "fit_start": fit_start.isoformat(), "fit_end": fit_end.isoformat(), "method": calibration.CALIBRATION_VERSION,
        "created_at": created, "available_at": created, "artifact_hash": digest,
        "calibration_artifact": body, "fit_result_ids": result_ids})
    return {"status": "FITTED", "calibration_id": calibration_id, "decided": len(pairs_), "a": a, "b": b}


def predict(path, sport, family, model, calibration, *, now, policy_version, tag, select=None):
    """Bound predictions from Pinnacle's price (called by ``prospective_calibration._predict``)."""
    import json
    from app_core import prospective_calibration as cal
    from app_core import prospective_research_models as research
    cal_artifact = json.loads(calibration["payload"]).get("calibration_artifact") if calibration else None
    calibration_id = calibration["calibration_id"] if calibration else None
    events, quotes, _ = research._read_scope(path, sport, family)
    all_quotes = evidence.read_records(path, "prospective_quote", sport=sport, market_family=family)
    done = {(r["event_id"], r["quote_id"]) for r in evidence.read_records(
        path, "prospective_prediction", sport=sport, market_family=family)
        if r["model_id"] == model["model_id"] and r["calibration_id"] == calibration_id}
    created, no_reference = 0, 0
    commit = research.source_commit()
    for _, quote in sorted((select or research._reference_quotes)(events, quotes, now).items()):
        event = events[quote["event_id"]]
        if now >= _t(event["scheduled_start"]) or (event["event_id"], quote["quote_id"]) in done:
            continue
        p = fair(event, quote, all_quotes, now=now)
        if p is None:
            no_reference += 1
            continue
        raw = {"win": p, "push": 0.0, "loss": 1 - p}
        probabilities = cal.calibrated(raw, cal_artifact) if cal_artifact else raw
        n = max(1, cal_artifact["decided"] if cal_artifact else 1)
        win = probabilities["win"]
        conservative = max(0.0, win - 1.645 * math.sqrt(win * (1 - win) / n))
        snapshot = {"protocol": FEATURE_VERSION, "sport": sport, "market_family": family,
                    "event_id": event["event_id"], "quote_id": quote["quote_id"], "as_of": now.isoformat(),
                    "pinnacle_probability": p, "exact_line": quote["line"], "american_odds": quote["american_odds"],
                    "decimal_odds": quote["decimal_odds"], "sportsbook": quote["sportsbook"],
                    "provider_event_hash": event["source_hash"], "provider_quote_hash": quote["source_hash"],
                    "raw_probabilities": raw, "calibration_id": calibration_id, "policy_version": policy_version}
        snapshot_id = evidence._sha(evidence._json(snapshot))
        record = {
            "observation_id": f"{tag.lower()}-prediction:" + research._digest([model["model_id"], calibration_id,
                                                                              snapshot_id])[:32],
            "event_id": event["event_id"], "sport": sport, "market_family": family, "selection": quote["selection"],
            "quote_id": quote["quote_id"], "model_id": model["model_id"], "model_version": model["model_version"],
            "feature_version": FEATURE_VERSION, "feature_snapshot_id": snapshot_id, "feature_snapshot": snapshot,
            "feature_frozen_at": now.isoformat(), "prediction_timestamp": now.isoformat(),
            "mean_probability": win, "conservative_probability": conservative,
            "push_probability": probabilities["push"], "loss_probability": probabilities["loss"],
            "probability_semantics": "WIN_PUSH_LOSS", "policy_version": policy_version,
            "evidence_snapshot_id": snapshot_id,
            "evidence_hash": research._digest([event["source_hash"], quote["source_hash"], p]),
            "runtime_hash": evidence._sha(evidence._json(artifact(sport, family))), "source_commit": commit,
            "uncertainty": {"method": "one_sided_normal_approximation_v1", "calibration_id": calibration_id,
                            "sample": n, "research_only": True}}
        if calibration is not None:
            record.update(calibration_id=calibration_id, calibration_version=calibration["calibration_version"])
        evidence.insert_prediction(path, record)
        created += 1
    return {"predictions": created, "skipped_no_pinnacle_at_line": no_reference, "blocker": None}
