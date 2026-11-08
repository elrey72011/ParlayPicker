"""Per-pick stake fields for bridge rows, derived from stored evidence (go-live Part 2, owner 2026-11-08).

``core.wager_decisions.candidate_decision`` and ``core.candidate_maturity.assign`` need these on a
row before it can reach a stake tier. ``app_core.qualified_bridge.row`` copied none of them, so a
bridged pick of a validated market would still get $0. Each is derived from stored records, never
guessed (``docs/roadmap/GO-LIVE-STAKING-TERMS-2026-11-08.md``, Part 2 and the decisions):

| field | value |
|---|---|
| ``model_validated`` | the market is validated and the prediction's model id and version are the deployment's |
| ``calibration_validated`` | the same for the calibration |
| ``evidence_snapshot_id`` | the prediction's own |
| ``critical_feature_error`` | ``False`` only when the stored feature snapshot hashes to its id, names this quote, (Pinnacle model) holds Pinnacle's probability, and the quote replays from its retained provider bytes |
| ``evidence_effective_sample_size`` | the validation report's validation + holdout scored games |
| ``calibration_uncertainty`` | mean minus the cautious (``uncertainty_quantile``) probability, measured on those games: ``z * sqrt(p (1 - p) / n)`` |
| ``conservative_probability`` | mean minus that uncertainty (the stored one was measured on the calibration's few games) |
| ``current_regime_conflict`` | the drift brake: ``True`` while the last ``window_bets`` live plan bets with a certified close have mean CLV below ``min_mean_clv`` |

Fields grant nothing by themselves: the market gate still requires the deployment, the owner's
activation and the exposure ledger. A market that is not validated gets fail-closed values.
"""
from __future__ import annotations

from contextlib import closing
from datetime import datetime
import json
import math
from statistics import NormalDist

from app_core import prospective_evidence as evidence

VALIDATED = {"PROVISIONAL_VALIDATED", "STANDARD_VALIDATED", "PREMIUM_VALIDATED"}
CLOSED = {"model_validated": False, "calibration_validated": False, "critical_feature_error": True,
          "evidence_effective_sample_size": 0.0, "calibration_uncertainty": None, "current_regime_conflict": None}


def market_context(path, sport, family, *, now: datetime) -> dict:
    """Everything the per-pick fields need for one market, computed once per bridge pass."""
    deployment = evidence.deployment_state(path, sport, family)
    context = {"sport": sport, "family": family, "deployment": deployment,
               "validated": (deployment.get("deployment_state") in VALIDATED
                             and deployment.get("validation_state") == deployment.get("deployment_state"))}
    if not context["validated"]:
        return context
    report = deployment.get("validation_report") or {}
    context["effective_n"] = sum(float((report.get(name) or {}).get("effective_observations") or 0)
                                 for name in ("validation", "holdout"))
    criteria = deployment.get("deployment_criteria") or {}
    quantile = float((criteria.get("validated_policy") or {}).get("uncertainty_quantile") or 0.25)
    context["z"] = -NormalDist().inv_cdf(quantile)
    context["regime"] = regime(path, sport, family, deployment, criteria, now=now)
    return context


def regime(path, sport, family, deployment, criteria, *, now) -> dict:
    """Mean CLV of the market's last ``window_bets`` live plan bets with a certified close.

    Live bets: games starting after the bound plan's holdout ended, the plan's own bet (its
    first prediction whose better side clears ``min_edge``; only the first prediction under the
    earliest-capture rule), by the deployment's exact model and calibration, with a comparable
    certified close. No live bets yet means no conflict.
    """
    brake = criteria.get("regime_brake") or {}
    window, floor = int(brake.get("window_bets") or 50), float(brake.get("min_mean_clv") or 0.0)
    plans = evidence.read_records(path, "prospective_validation_plan", sport=sport, market_family=family)
    plan = max(plans, key=lambda p: p["version"]) if plans else None
    if plan is None:
        return {"bets": 0, "mean_clv": None, "conflict": False}
    rules = (json.loads(plan["payload"]).get("market_skill_methodology") or {})
    min_edge = float(rules.get("min_edge") or 0.02)
    first_edge = rules.get("entry_rule") == "FIRST_EDGE_CAPTURE"
    from app_core.market_skill_validation import opposite
    holdout_end = evidence._time(plan["holdout_end"], "holdout_end")
    clvs = []
    with closing(evidence._reader(path)) as db:
        rows = db.execute("""SELECT p.observation_id, e.scheduled_start FROM prospective_prediction p
            JOIN prospective_event e ON e.event_id = p.event_id
            WHERE p.sport=? AND p.market_family=? AND p.model_id=? AND p.calibration_id=?
            AND e.scheduled_start>=? ORDER BY e.scheduled_start, p.prediction_timestamp, p.observation_id""",
                          (sport, family, deployment["model_id"], deployment["calibration_id"],
                           evidence._iso(holdout_end))).fetchall()
        seen, bets = set(), []
        for row in rows:
            prediction = evidence._stored(db, "prospective_prediction", "observation_id", row[0])
            start = evidence._time(row[1], "scheduled_start")
            if prediction["event_id"] in seen or start > now or not prediction["quote_id"]:
                continue
            quote = evidence._stored(db, "prospective_quote", "quote_id", prediction["quote_id"])
            other = opposite(db, quote) if quote else None
            mean, loss = prediction["mean_probability"], prediction["loss_probability"]
            if other is None or mean is None or loss is None or mean + loss <= 0:
                continue
            p = mean / (mean + loss)
            sides = [(p * quote["decimal_odds"] - 1, quote), ((1 - p) * other["decimal_odds"] - 1, other)]
            ev, side = max(sides, key=lambda s: s[0])
            if ev < min_edge:
                if not first_edge:
                    seen.add(prediction["event_id"])  # earliest rule: only the first prediction counts
                continue
            seen.add(prediction["event_id"])
            bets.append((start, side))
        for _, side in bets[-window:]:
            close = evidence._comparable_close(db, side, now)
            if close is not None:
                clvs.append(side["decimal_odds"] / close["decimal_odds"] - 1)
    mean_clv = sum(clvs) / len(clvs) if clvs else None
    return {"bets": len(clvs), "mean_clv": mean_clv, "conflict": mean_clv is not None and mean_clv < floor}


def _snapshot_ok(prediction, quote, event, model_id) -> bool:
    try:
        payload = json.loads(prediction.get("payload") or "{}")
    except ValueError:
        return False
    snapshot = payload.get("feature_snapshot")
    if not isinstance(snapshot, dict) or evidence._sha(evidence._json(snapshot)) != prediction["feature_snapshot_id"]:
        return False
    if snapshot.get("quote_id") not in (None, quote["quote_id"]):
        return False
    if str(model_id or "").startswith("pinnacle-model:"):
        p = snapshot.get("pinnacle_probability")
        if not isinstance(p, (int, float)) or not 0 < p < 1 or snapshot.get("quote_id") != quote["quote_id"]:
            return False
    try:
        return bool(evidence.verify_provider_offer(event, quote))
    except (KeyError, TypeError, ValueError):
        return False


def fields(context, prediction, quote, event) -> dict:
    """The stake fields for one bridge row (fail-closed unless the market is validated)."""
    out = dict(CLOSED, evidence_snapshot_id=prediction.get("evidence_snapshot_id"))
    deployment = context.get("deployment") or {}
    if not context.get("validated"):
        return out
    out["model_validated"] = (prediction["model_id"] == deployment.get("model_id")
                              and prediction["model_version"] == deployment.get("model_version"))
    out["calibration_validated"] = (prediction["calibration_id"] == deployment.get("calibration_id")
                                    and prediction["calibration_version"] == deployment.get("calibration_version"))
    out["critical_feature_error"] = not _snapshot_ok(prediction, quote, event, prediction["model_id"])
    n = context.get("effective_n") or 0.0
    out["evidence_effective_sample_size"] = n
    mean, loss = prediction["mean_probability"], prediction["loss_probability"]
    if n > 0 and mean is not None and loss is not None and mean + loss > 0:
        p = mean / (mean + loss)
        uncertainty = context["z"] * math.sqrt(p * (1 - p) / n)
        out["calibration_uncertainty"] = round(uncertainty, 6)
        out["conservative_probability"] = max(0.0, mean - uncertainty)
    out["current_regime_conflict"] = bool((context.get("regime") or {}).get("conflict"))
    return out
