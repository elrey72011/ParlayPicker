"""Leak-safe football evidence snapshots for offline validation.

Research implementation. It does not replace a game forecast with a bucket win
rate and is not wired to promote live wagers. Prior weights/ESS/quantile must be
validated in the sport policy before use in production.
"""
from dataclasses import asdict
import hashlib
import json
import math
from core.wager_decisions import aware, finite

VERSION = "football-evidence-research-v1"


def freeze_evidence(records, policy, *, season, week, frozen_at, slate_start):
    if not isinstance(season, int) or isinstance(season, bool) or not isinstance(week, int) or isinstance(week, bool) or week < 1:
        raise ValueError("Season and week must be integers")
    if policy.sport not in {"NFL", "NCAAF"}:
        raise ValueError("Football evidence cannot update another sport")
    freeze, start = aware(frozen_at), aware(slate_start)
    if freeze is None or start is None or freeze > start:
        raise ValueError("Freeze must precede the slate and include timezone")
    accepted, excluded, seen = [], {}, set()
    def reject(reason):
        excluded[reason] = excluded.get(reason, 0) + 1
    for source in records:
        reason = None
        prediction, game = aware(source.get("prediction_at")), aware(source.get("start"))
        outcome_at, recorded = aware(source.get("outcome_at")), aware(source.get("recorded_at"))
        train_end = aware(source.get("training_cutoff"))
        p = finite(source.get("probability"))
        identifier = source.get("id")
        if source.get("sport") != policy.sport:
            reason = "other_sport"
        elif not identifier or identifier in seen:
            reason = "missing_or_duplicate_id"
        elif source.get("pregame_verified") is not True or source.get("outcome") not in {"WIN", "LOSS"}:
            reason = "unverified_or_unsettled"
        elif any(x is None for x in (prediction, game, outcome_at, recorded, train_end)):
            reason = "missing_timestamp"
        elif not train_end < prediction < game <= outcome_at <= recorded <= freeze or outcome_at >= start:
            reason = "future_or_postgame_evidence"
        elif not isinstance(source.get("season"), int) or not isinstance(source.get("week"), int) or source["season"] > season or (source["season"] == season and (source.get("week") is None or source["week"] >= week)):
            reason = "same_or_future_slate"
        elif p is None or not 0 < p < 1 or source.get("market") not in {"spread_home", "spread_away", "total_over", "total_under"}:
            reason = "invalid_probability_or_market"
        elif not all(source.get(key) for key in ("model_version", "calibration_version", "source_hash")):
            reason = "missing_provenance"
        if reason:
            reject(reason)
            continue
        seen.add(identifier)
        age = season - source["season"]
        weight = policy.current_season_weighting if age == 0 else policy.historical_prior_strength * policy.historical_prior_decay ** (age - 1)
        accepted.append({key: source[key] for key in ("id", "sport", "season", "week", "probability", "outcome", "market", "model_version", "calibration_version", "source_hash", "prediction_at", "start", "training_cutoff", "outcome_at", "recorded_at")} | {"weight": weight})
    # Cap historical evidence mass. It cannot overwhelm the current season.
    historical_mass = sum(x["weight"] for x in accepted if x["season"] < season)
    scale = min(1.0, policy.historical_effective_sample_cap / historical_mass) if historical_mass else 1.0
    for row in accepted:
        if row["season"] < season:
            row["weight"] *= scale
    payload = dict(sport=policy.sport, season=season, week=week, frozen_at=freeze.isoformat(), slate_start=start.isoformat(),
                   version=VERSION, sport_policy_version=policy.version, policy_settings=asdict(policy), records=sorted(accepted, key=lambda x: str(x["id"])), excluded=excluded,
                   historical_mass=sum(x["weight"] for x in accepted if x["season"] < season),
                   model_versions=sorted({x["model_version"] for x in accepted}),
                   calibration_versions=sorted({x["calibration_version"] for x in accepted}),
                   source_hashes=sorted({x["source_hash"] for x in accepted}),
                   maximum_outcome_at=max((aware(x["outcome_at"]).isoformat() for x in accepted), default=None),
                   current_season_n=sum(x["season"] == season for x in accepted))
    payload["snapshot_id"] = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return payload


def reliability_distribution(probability, market, snapshot, policy, *, model_version, calibration_version):
    """Residual log-odds correction; prior parent and child rows do not overlap.

    Matching version + sport + family is mandatory. Same-direction rows in the
    candidate's 5-point probability band form the child. Other family rows form
    the parent with capped mass. A Jeffreys prior is research-only, not proof of
    a model's calibration. Missing evidence produces no conservative forecast.
    """
    from scipy.stats import beta
    content = {key: value for key, value in snapshot.items() if key != "snapshot_id"}
    digest = hashlib.sha256(json.dumps(content, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if digest != snapshot.get("snapshot_id") or json.dumps(snapshot.get("policy_settings"), sort_keys=True) != json.dumps(asdict(policy), sort_keys=True):
        raise ValueError("Evidence snapshot has been modified")
    p = finite(probability)
    if p is None or not 0 < p < 1 or snapshot["sport"] != policy.sport or snapshot["sport_policy_version"] != policy.version:
        raise ValueError("Candidate and evidence policy do not match")
    family = market.split("_")[0]
    if family not in {"spread", "total"}:
        raise ValueError("Unsupported candidate market")
    rows = [x for x in snapshot["records"] if x["market"].split("_")[0] == family and x["model_version"] == model_version and x["calibration_version"] == calibration_version and x["weight"] > 0]
    if not rows:
        return {"status": "INSUFFICIENT_EVIDENCE", "conservative_probability": None, "snapshot_id": snapshot["snapshot_id"]}
    child, parent = [], []
    for row in rows:
        (child if row["market"] == market and int(row["probability"] * 20) == int(p * 20) else parent).append(row)
    mass = sum(x["weight"] for x in parent)
    scale = min(1.0, policy.historical_effective_sample_cap / mass) if mass else 1.0
    weighted = [(row, row["weight"]) for row in child] + [(row, row["weight"] * scale) for row in parent]
    total = sum(w for _, w in weighted)
    if total <= 0:
        return {"status": "INSUFFICIENT_EVIDENCE", "conservative_probability": None, "snapshot_id": snapshot["snapshot_id"]}
    alpha = .5 + sum(w for row, w in weighted if row["outcome"] == "WIN")
    b = .5 + sum(w for row, w in weighted if row["outcome"] == "LOSS")
    reference = sum(row["probability"] * w for row, w in weighted) / total
    offset = math.log(p / (1-p)) - math.log(reference / (1-reference))
    def transform(q):
        q = min(1-1e-12, max(1e-12, float(q)))
        return 1 / (1 + math.exp(-(math.log(q / (1-q)) + offset)))
    # Numerical quadrature of the transformed distribution, not beta variance
    # misreported as game-probability variance.
    draws = [transform(beta.ppf((i+.5)/1000, alpha, b)) for i in range(1000)]
    mean = sum(draws)/len(draws)
    return dict(status="RESEARCH_ONLY", mean=mean, sd=math.sqrt(sum((x-mean)**2 for x in draws)/len(draws)),
                **{f"p{q}": transform(beta.ppf(q/100, alpha,b)) for q in (10,25,50,75,90)},
                conservative_probability=min(p, transform(beta.ppf(policy.uncertainty_quantile,alpha,b))),
                raw_model_probability=p, reference_probability=reference, evidence_mass=total,
                effective_sample_size=min(total, total**2/sum(w*w for _,w in weighted)), child_n=len(child), parent_n=len(parent),
                snapshot_id=snapshot["snapshot_id"], evidence_engine_version=VERSION)
