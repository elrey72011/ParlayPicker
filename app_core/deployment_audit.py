"""Independent deployment audit for a validated market (roadmap D11 option B, owner-chosen 2026-11-06).

``record_deployment_review`` recomputes the frozen plan's evaluation, so the numbers
themselves cannot be faked. The review exists to catch what those numbers miss. This
audit turns that into concrete, reproducible checks on the plan's validation and holdout
windows, and the owner signs a review only against the exact audit hash it produces:

FAIL (blocks recording):
* ``VALIDATION_PASSED``: the plan's evaluation, recomputed now, is VALIDATION_PASSED.
* ``CHRONOLOGY``: every prediction has its quote, the quote is not after the prediction,
  the prediction is before kickoff, and the model and calibration were available first.
* ``NO_REFERENCE_BOOK_ENTRY``: no entry price is from a reference-only book (Pinnacle).

WARN (the owner decides):
* ``ONE_PREDICTION_PER_EVENT``: duplicates are deduplicated by the evaluation, but listed.
* ``CONCENTRATION``: paper ROI with the 5 largest winning units removed is still >= 0.
* ``HOLDOUT_DRIFT``: holdout Brier is not more than 0.01 worse than validation.
* ``CLOSING_LINE_VALUE``: mean price CLV against certified closes is not negative.

Read-only: it opens the store through ``read_records`` and changes nothing.
"""
from __future__ import annotations

import hashlib
import json

from app_core import prospective_evidence as evidence
from app_core.research_api_budget import REFERENCE_BOOKS

VERSION = "deployment-audit-v1"
CONCENTRATION_TOP = 5
MAX_HOLDOUT_BRIER_DRIFT = 0.01


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _check(name, status, detail):
    return {"check": name, "status": status, "detail": detail}


def _windows(plan):
    return [(plan[f"{n}_start"], plan[f"{n}_end"]) for n in ("validation", "holdout")]


def audit(path, validation_plan_id):
    plan = evidence.load_record(path, "prospective_validation_plan", validation_plan_id)
    if plan is None:
        raise ValueError("unknown validation plan")
    sport, market = plan["sport"], plan["market_family"]
    report = evidence.evaluate_validation_plan(path, validation_plan_id)
    events = {e["event_id"]: e for e in evidence.read_records(path, "prospective_event", sport=sport)}
    quotes = {q["quote_id"]: q for q in evidence.read_records(path, "prospective_quote", sport=sport,
                                                                 market_family=market)}
    closes = [c for c in evidence.read_records(path, "prospective_close", sport=sport, market_family=market)
              if c.get("close_verified")]
    results = {}
    for r in evidence.read_records(path, "prospective_result", sport=sport):
        if r["market_family"] is None and r["event_id"] in events and evidence.verify_provider_score(events[r["event_id"]], r):
            prev = results.get(r["event_id"])
            if prev is None or r["grading_version"] > prev["grading_version"]:
                results[r["event_id"]] = r
    windows = _windows(plan)
    predictions = [p for p in evidence.read_records(path, "prospective_prediction", sport=sport, market_family=market)
                   if p["event_id"] in events and any(a <= events[p["event_id"]]["scheduled_start"] < b
                                                      for a, b in windows)]
    t = evidence._time

    chronology, reference, per_event = [], [], {}
    for p in predictions:
        event, quote = events[p["event_id"]], quotes.get(p["quote_id"])
        per_event[p["event_id"]] = per_event.get(p["event_id"], 0) + 1
        made = t(p["prediction_timestamp"], "prediction_timestamp")
        problems = []
        if quote is None:
            problems.append("missing_quote")
        else:
            if t(quote["quote_timestamp"], "quote_timestamp") > made:
                problems.append("quote_after_prediction")
            if quote["sportsbook"] in REFERENCE_BOOKS:
                reference.append(p["observation_id"])
        if made >= t(event["scheduled_start"], "scheduled_start"):
            problems.append("prediction_not_before_start")
        for field in ("model_available_at", "calibration_available_at"):
            if p.get(field) and t(p[field], field) > made:
                problems.append(field.replace("_available_at", "_not_available_yet"))
        if problems:
            chronology.append({"observation_id": p["observation_id"], "problems": problems})

    units, clv = [], []
    close_by = {(c["event_id"], c["selection"], c["sportsbook"]): c for c in closes}
    for p in predictions:
        quote, result, event = quotes.get(p["quote_id"]), results.get(p["event_id"]), events[p["event_id"]]
        if quote is None:
            continue
        close = close_by.get((p["event_id"], quote["selection"], quote["sportsbook"]))
        if close is not None and close.get("line") == quote["line"] and close.get("decimal_odds"):
            clv.append(quote["decimal_odds"] / close["decimal_odds"] - 1)
        if result is None or result.get("home_score") is None:
            continue
        outcome = evidence.score_settlement_outcome(sport, market, quote["selection"], event["home_team"],
                                                    event["away_team"], result["home_score"],
                                                    result["away_score"], quote["line"])
        units.append(quote["decimal_odds"] - 1 if outcome == "WIN" else (-1.0 if outcome == "LOSS" else 0.0))

    duplicates = {e: n for e, n in per_event.items() if n > 1}
    trimmed = sorted(units)[:-CONCENTRATION_TOP] if len(units) > CONCENTRATION_TOP else []
    trimmed_roi = sum(trimmed) / len(trimmed) if trimmed else None
    v_brier, h_brier = report.get("validation", {}).get("brier"), report.get("holdout", {}).get("brier")
    drift = None if v_brier is None or h_brier is None else h_brier - v_brier
    mean_clv = sum(clv) / len(clv) if clv else None

    checks = [
        _check("VALIDATION_PASSED", "PASS" if report.get("status") == "VALIDATION_PASSED" else "FAIL",
               {"status": report.get("status"), "blockers": report.get("blockers")}),
        _check("CHRONOLOGY", "FAIL" if chronology else "PASS",
               {"predictions": len(predictions), "violations": chronology[:20], "violation_count": len(chronology)}),
        _check("NO_REFERENCE_BOOK_ENTRY", "FAIL" if reference else "PASS", {"observations": reference[:20]}),
        _check("ONE_PREDICTION_PER_EVENT", "WARN" if duplicates else "PASS",
               {"events_with_duplicates": len(duplicates)}),
        _check("CONCENTRATION", "WARN" if trimmed_roi is None or trimmed_roi < 0 else "PASS",
               {"settled": len(units), "roi": sum(units) / len(units) if units else None,
                f"roi_without_top_{CONCENTRATION_TOP}_wins": trimmed_roi}),
        _check("HOLDOUT_DRIFT", "WARN" if drift is None or drift > MAX_HOLDOUT_BRIER_DRIFT else "PASS",
               {"validation_brier": v_brier, "holdout_brier": h_brier, "drift": drift}),
        _check("CLOSING_LINE_VALUE", "WARN" if mean_clv is None or mean_clv < 0 else "PASS",
               {"comparable_closes": len(clv), "mean_price_clv": mean_clv,
                "beat_close_share": sum(1 for c in clv if c > 0) / len(clv) if clv else None}),
    ]
    status = ("FAIL" if any(c["status"] == "FAIL" for c in checks)
              else "WARN" if any(c["status"] == "WARN" for c in checks) else "PASS")
    body = {"version": VERSION, "validation_plan_id": validation_plan_id, "sport": sport, "market_family": market,
            "evaluation_hash": _hash(report), "status": status, "checks": checks}
    return dict(body, audit_hash=_hash(body))


def reviewer_label(reviewer_id, audit_hash):
    """The recorded reviewer identity binds the owner to the exact audit they read."""
    if not isinstance(reviewer_id, str) or not reviewer_id.strip() or ";" in reviewer_id:
        raise ValueError("reviewer_id must be a plain non-empty name")
    return f"{reviewer_id.strip()};{VERSION};sha256={audit_hash}"


def markdown(result):
    lines = [f"## Deployment audit: {result['sport']} {result['market_family']} — **{result['status']}**", "",
             f"- plan: `{result['validation_plan_id']}`", f"- audit hash: `{result['audit_hash']}`", "",
             "| check | status | detail |", "|---|---|---|"]
    for c in result["checks"]:
        lines.append(f"| {c['check']} | {c['status']} | `{json.dumps(c['detail'], default=str)[:300]}` |")
    lines += ["", "FAIL blocks recording. WARN rows are for the owner's judgment. To record the review, run the "
              "workflow with action `record`, this audit hash, and your reviewer name."]
    return "\n".join(lines) + "\n"
