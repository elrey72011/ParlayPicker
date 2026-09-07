"""Fixed chronological research experiment; never consumed by live wagering."""
import hashlib
import json
import math
from statistics import NormalDist
from pathlib import Path
import numpy as np
from app_core.mlb_history import build_dataset, timestamp

FEATURES = ["home_ppg", "away_ppg", "home_oppg", "away_oppg",
            "home_win_pct", "away_win_pct"]
PROTOCOL = {"version": "mlb-research-v1", "train": 2023, "calibrate": 2024, "evaluate": 2025,
            "ridge_alpha": 10.0, "minimum_rows_per_split": 100, "minimum_prior_games": 10,
            "features": FEATURES, "distribution": "discretized_gaussian_totals_truncated_at_zero",
            "production_eligible": False}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def prepare(state):
    if state.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema")
    scheduled = set()
    for year in (2023, 2024, 2025):
        schedule = state["schedules"].get(str(year))
        if not schedule or not schedule["payload"].get("dates"):
            raise ValueError("Missing season schedule")
        for day in schedule["payload"]["dates"]:
            for game in day["games"]:
                scheduled.add(str(game["gamePk"]))
    if scheduled - state["games"].keys() - state["excluded"].keys():
        raise ValueError("Collection has pending games")
    games = []
    for key, entry in state["games"].items():
        game = entry["record"]
        if key not in scheduled or str(game["game_id"]) != key:
            raise ValueError("Unexpected or mismatched game ID")
        if not (timestamp(game["cutoff"]) <= timestamp(game["started_at"]) < timestamp(game["completed_at"])):
            raise ValueError("Invalid game chronology")
        if timestamp(game["cutoff"]).year != game["season"]:
            raise ValueError("Season and cutoff disagree")
        if any(isinstance(game[k], bool) or not isinstance(game[k], int) or game[k] < 0 for k in ("home_score", "away_score")):
            raise ValueError("Invalid score")
        if game["home_score"] == game["away_score"]:
            raise ValueError("Undecided game")
        games.append(game)
    dataset = build_dataset(games, minimum_games=PROTOCOL["minimum_prior_games"])
    lookup = {r["game_id"]: r for r in dataset["targets"]}
    splits, coverage = {}, []
    for year in (2023, 2024, 2025):
        rows = []
        for r in dataset["features"]:
            if r["season"] != year:
                continue
            if not all(math.isfinite(float(r[k])) for k in FEATURES):
                raise ValueError("Nonfinite feature")
            t = lookup[r["game_id"]]
            rows.append({**r, "kickoff": r["cutoff"],
                         "y_margin": t["home_score"] - t["away_score"],
                         "y_total": t["home_score"] + t["away_score"]})
        rows.sort(key=lambda r: (timestamp(r["cutoff"]), r["game_id"]))
        if len(rows) < PROTOCOL["minimum_rows_per_split"]:
            raise ValueError(f"At least 100 eligible games required for {year}")
        splits[year] = rows
        coverage.append({"season": year, "eligible": len(rows),
                         "excluded": sum(g["season"] == year for g in games) - len(rows)})
    if not (timestamp(splits[2023][-1]["cutoff"]) < timestamp(splits[2024][0]["cutoff"]) < timestamp(splits[2025][0]["cutoff"])):
        raise ValueError("Overlapping splits")
    return splits, coverage


def _matrix(rows):
    return np.array([[float(r[k]) for k in FEATURES] for r in rows], dtype=float)


def _base(rows, target):
    home = np.array([(r["home_ppg"] + r["away_oppg"]) / 2 for r in rows])
    away = np.array([(r["away_ppg"] + r["home_oppg"]) / 2 for r in rows])
    return home - away if target == "margin" else home + away


def centers(model, rows, target):
    if model["kind"] == "ridge":
        x = (_matrix(rows) - np.array(model["x_mean"])) / np.array(model["x_scale"])
        values = x @ np.array(model["coefficients"]) + model["intercept"]
    elif model["kind"] == "constant":
        values = np.full(len(rows), model["intercept"])
    else:
        values = _base(rows, target)
    values = values + model.get("bias", 0.0)
    return np.maximum(values, 0) if target == "total" else values


def fit(train, calibration):
    """No holdout argument: scaling, fitting and calibration cannot read 2025."""
    artifact = {"protocol": PROTOCOL, "source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "feature_builder_hash": hashlib.sha256(Path(__file__).with_name("mlb_history.py").read_bytes()).hexdigest(),
                "train_hash": digest(train), "calibration_hash": digest(calibration), "models": {}}
    x = _matrix(train)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1
    x = (x - mean) / scale
    for kind in ("ridge", "constant", "scoring_blend"):
        artifact["models"][kind] = {}
        for target in ("margin", "total"):
            y = np.array([r["y_" + target] for r in train], dtype=float)
            model = {"kind": kind, "intercept": float(y.mean())}
            if kind == "ridge":
                coefficients = np.linalg.solve(x.T @ x + PROTOCOL["ridge_alpha"] * np.eye(len(FEATURES)), x.T @ (y - y.mean()))
                model.update(x_mean=mean.tolist(), x_scale=scale.tolist(), coefficients=coefficients.tolist())
            actual = np.array([r["y_" + target] for r in calibration])
            model["bias"] = float(np.mean(actual - centers(model, calibration, target)))
            model["sigma"] = max(1.0, float(np.sqrt(np.mean((actual - centers(model, calibration, target)) ** 2))))
            artifact["models"][kind][target] = model
    return artifact


def probabilities(mu, sigma, line=0.0, *, total=False):
    """Strictly over/under an integer-valued outcome; explicit push probability."""
    if not all(math.isfinite(v) for v in (mu, sigma, line)) or sigma <= 0:
        raise ValueError("Invalid distribution parameters")
    def cdf(x):
        return 0.5 * math.erfc(-(x - mu) / (sigma * math.sqrt(2)))
    line = float(line)
    lower = cdf(-0.5) if total else 0.0
    denominator = 1 - lower
    if denominator <= 1e-12:
        raise ValueError("Degenerate total distribution")
    def truncated(x):
        return min(1.0, max(0.0, (cdf(x) - lower) / denominator))
    below = truncated(math.ceil(line) - 0.5)
    above = 1 - truncated(math.floor(line) + 0.5)
    push = max(0.0, 1 - below - above) if line.is_integer() else 0.0
    return {"over": above, "under": below, "push": push}


def evaluate(artifact, holdout):
    metrics, predictions = {}, []
    for kind, models in artifact["models"].items():
        metrics[kind] = {}
        for target, model in models.items():
            means = centers(model, holdout, target)
            actual = np.array([r["y_" + target] for r in holdout])
            errors = actual - means
            normal = NormalDist()
            intervals = []
            for mu in means:
                base = normal.cdf((-0.5 - mu) / model["sigma"]) if target == "total" else 0.0
                intervals.append([math.floor(mu + model["sigma"] * normal.inv_cdf(base + q * (1-base)) + .5) for q in (.05, .95)])
            lower, upper = np.array(intervals).T
            metric = {"n": len(holdout), "mae": float(np.mean(abs(errors))),
                      "rmse": float(np.sqrt(np.mean(errors ** 2))),
                      "nominal_90_interval_coverage": float(np.mean((actual >= lower) & (actual <= upper)))}
            if target == "margin":
                pairs = []
                for row, mu in zip(holdout, means):
                    p = probabilities(float(mu), model["sigma"], 0.0)
                    conditional = p["over"] / max(1e-12, 1 - p["push"])
                    if row["y_margin"] != 0:
                        pairs.append((conditional, int(row["y_margin"] > 0)))
                    predictions.append({"game_id": row["game_id"], "model": kind, "kickoff": row["kickoff"], "actual_margin": row["y_margin"], "margin_center": float(mu),
                                        "home_win_probability_no_tie": conditional, "tie_probability": p["push"]})
                if not pairs:
                    raise ValueError("No decided holdout games")
                p, y = np.array(pairs).T
                clipped = np.clip(p, 1e-12, 1 - 1e-12)
                metric.update(decided_games=len(pairs), ties_excluded=len(holdout)-len(pairs),
                              winner_accuracy=float(np.mean((p >= .5) == y)),
                              brier=float(np.mean((p-y)**2)),
                              log_loss=float(-np.mean(y*np.log(clipped)+(1-y)*np.log(1-clipped))))
                metric["reliability"] = []
                for i in range(10):
                    mask = (p >= i/10) & ((p < (i+1)/10) if i < 9 else (p <= 1))
                    if mask.any():
                        metric["reliability"].append({"bin": i, "n": int(mask.sum()),
                            "mean_prediction": float(p[mask].mean()), "observed_home_win_rate": float(y[mask].mean())})
            else:
                for row, mu in zip(holdout, means):
                    predictions.append({"game_id": row["game_id"], "model": kind, "kickoff": row["kickoff"], "actual_total": row["y_total"], "total_center": float(mu)})
            metrics[kind][target] = metric
    return metrics, predictions


def run_research(state):
    splits, coverage = prepare(state)
    artifact = fit(splits[2023], splits[2024])
    frozen_hash = digest(artifact)
    metrics, predictions = evaluate(artifact, splits[2025])
    report = {"protocol": PROTOCOL, "artifact_hash": frozen_hash,
              "checkpoint_hash": digest(state), "coverage": coverage,
              "holdout_metrics": metrics, "production_eligible": False,
              "collection_exclusions": state["excluded"],
              "limitations": [
                  "Historical publication/correction times are unverified; this is retrospective research.",
                  "Excluded source games also leave gaps in affected team averages.",
                  "No historical odds, betting returns or approved-bet hit rate are evaluated.",
                  "2025 is now evaluated; future tuning requires a new evaluation period.",
                  "No live model or wagering approval changes."]}
    return {"report": report, "artifact": artifact, "predictions": predictions}


def markdown_report(result):
    report = result["report"]
    lines = ["# MLB research evaluation", "", "Train: 2023. Calibrate: 2024. Evaluate: 2025. Research only.", "",
             "| Model | Margin MAE | Total MAE | Winner accuracy | Brier | Log loss |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, m in report["holdout_metrics"].items():
        a, b = m["margin"], m["total"]
        lines.append(f"| {name} | {a['mae']:.3f} | {b['mae']:.3f} | {a['winner_accuracy']:.1%} | {a['brier']:.4f} | {a['log_loss']:.4f} |")
    lines += ["", "## Split coverage", "", "| Season | Eligible | Excluded |", "|---|---:|---:|"]
    lines += [f"| {r['season']} | {r['eligible']} | {r['excluded']} |" for r in report["coverage"]]
    lines += ["", "Errors are mean absolute errors in runs. Winner accuracy is across all eligible games, not approved wagers.", "", "Artifact: " + report["artifact_hash"], "", "## Limitations", ""]
    lines += ["- " + x for x in report["limitations"]]
    return "\n".join(lines) + "\n"
