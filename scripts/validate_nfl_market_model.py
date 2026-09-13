"""Offline NFL scoring candidate. Never installs a model or changes wager gates.

Fixed protocol: train 2015-2022, calibrate residual scale on 2023, evaluate
2024-2025 once. The listed historical odds are a benchmark, not evidence of
an executable pregame price. See docs/nfl-model-validation.md.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
import hashlib
import json
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from core.model_validation import compare_candidate_to_market

FEATURES = ["home_ppg", "away_ppg", "home_oppg", "away_oppg", "win_diff", "home_field"]
PROTOCOL = {
    "train_seasons": [2015, 2022], "calibration_seasons": [2023, 2023],
    "evaluation_seasons": [2024, 2025], "prior_games": 16,
    "minimum_prior_games": 4, "ridge_penalty": 10.0,
    "same_day_results_excluded": True, "automatic_installation": False,
}


def pregame_features(games: pd.DataFrame) -> pd.DataFrame:
    """Use only completed regular-season games from strictly earlier dates.

    Team history rolls across seasons, retaining at most 16 games. Franchise
    relocations share history. Same-day results cannot affect another row.
    """
    frame = games.loc[games.game_type.eq("REG")].copy()
    if frame.game_id.isna().any() or frame.game_id.duplicated().any():
        raise ValueError("Schedule must have unique nonempty game IDs")
    frame["gameday"] = pd.to_datetime(frame.gameday, errors="raise")
    if frame.gameday.isna().any():
        raise ValueError("Missing game date")
    for col in ["home_score", "away_score"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    aliases = {"OAK": "LV", "SD": "LAC", "STL": "LAR", "LA": "LAR"}
    history = defaultdict(lambda: deque(maxlen=PROTOCOL["prior_games"]))
    records = []
    for _, day in frame.sort_values(["gameday", "game_id"]).groupby("gameday", sort=True):
        for _, row in day.iterrows():
            home, away = [aliases.get(row[c], row[c]) for c in ["home_team", "away_team"]]
            h, a = history[home], history[away]
            if min(len(h), len(a)) >= PROTOCOL["minimum_prior_games"]:
                hs, aws = np.asarray(h), np.asarray(a)
                record = row.to_dict()
                record.update(home_ppg=hs[:, 0].mean(), home_oppg=hs[:, 1].mean(),
                              away_ppg=aws[:, 0].mean(), away_oppg=aws[:, 1].mean(),
                              win_diff=hs[:, 2].mean()-aws[:, 2].mean(),
                              home_field=float(row["location"] == "Home"))
                records.append(record)
        for _, row in day.iterrows():
            hp, ap = row.home_score, row.away_score
            if not np.isfinite([hp, ap]).all() or min(hp, ap) < 0:
                continue
            home, away = [aliases.get(row[c], row[c]) for c in ["home_team", "away_team"]]
            history[home].append((hp, ap, float(hp > ap) + .5*float(hp == ap)))
            history[away].append((ap, hp, float(ap > hp) + .5*float(ap == hp)))
    return pd.DataFrame(records)


def fit_candidate(train: pd.DataFrame, calibration: pd.DataFrame) -> dict:
    """Fit scoring means on training rows; scale only on the later calibration."""
    if train.empty or calibration.empty:
        raise ValueError("Training and calibration data are required")
    if train.gameday.max() >= calibration.gameday.min():
        raise ValueError("Training must precede calibration")
    x = train[FEATURES].to_numpy(float)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1.0
    design = np.column_stack([np.ones(len(x)), (x-mean)/scale])
    penalty = np.eye(design.shape[1]) * PROTOCOL["ridge_penalty"]
    penalty[0, 0] = 0.0
    model = {"feature_names": FEATURES, "feature_mean": mean.tolist(),
             "feature_scale": scale.tolist(), "targets": {}}
    cx = np.column_stack([np.ones(len(calibration)),
                          (calibration[FEATURES].to_numpy(float)-mean)/scale])
    for target in ["margin", "total"]:
        y = (train.home_score-train.away_score if target == "margin"
             else train.home_score+train.away_score).to_numpy(float)
        beta = np.linalg.solve(design.T@design+penalty, design.T@y)
        cy = (calibration.home_score-calibration.away_score if target == "margin"
              else calibration.home_score+calibration.away_score).to_numpy(float)
        sigma = float(np.sqrt(np.mean((cy-cx@beta)**2)))
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("Invalid calibration residual scale")
        model["targets"][target] = {"coefficients": beta.tolist(), "sigma": sigma}
    return model


def scoring_projection(model: dict, rows: pd.DataFrame, target: str) -> np.ndarray:
    x = (rows[FEATURES].to_numpy(float)-model["feature_mean"])/model["feature_scale"]
    return np.column_stack([np.ones(len(x)), x])@model["targets"][target]["coefficients"]


def probability_above(mean: float, line: float, sigma: float) -> float:
    """Discrete-score normal approximation, conditional on no push.

    For integer thresholds, reserve [line-.5, line+.5] for a push; for a
    fractional threshold separate integer scores at the enclosing half point.
    """
    if not np.isfinite([mean, line, sigma]).all() or sigma <= 0:
        raise ValueError("A finite mean, exact line, and positive scale are required")
    cdf = lambda value: .5*(1+erf((value-mean)/(sigma*sqrt(2))))
    if float(line).is_integer():
        loss, win = cdf(line-.5), 1-cdf(line+.5)
    else:
        loss = cdf(np.floor(line)+.5)
        win = 1-loss
    if win+loss <= 0:
        raise ValueError("Non-push probability is zero")
    return float(np.clip(win/(win+loss), 1e-6, 1-1e-6))


def implied(odds: pd.Series) -> pd.Series:
    odds = pd.to_numeric(odds, errors="coerce")
    valid = odds.abs().ge(100) & np.isfinite(odds)
    result = pd.Series(np.nan, index=odds.index, dtype=float)
    positive, negative = valid & odds.gt(0), valid & odds.lt(0)
    result.loc[positive] = 100/(100+odds.loc[positive])
    result.loc[negative] = -odds.loc[negative]/(100-odds.loc[negative])
    return result


def metrics(outcome: np.ndarray, probability: np.ndarray) -> dict:
    if len(outcome) == 0 or len(np.unique(outcome)) != 2:
        raise ValueError("Benchmark requires both outcome classes")
    p = np.clip(probability, 1e-6, 1-1e-6)
    ranks = pd.Series(p).rank(method="average").to_numpy()
    wins = outcome.sum()
    auc = (ranks[outcome == 1].sum()-wins*(wins+1)/2)/(wins*(len(outcome)-wins))
    return {"ll": float(-np.mean(outcome*np.log(p)+(1-outcome)*np.log(1-p))),
            "brier": float(np.mean((p-outcome)**2)), "auc": float(auc)}


def evaluate(model: dict, rows: pd.DataFrame, target: str) -> dict:
    line_col, win_col, loss_col = (("spread_line", "home_spread_odds", "away_spread_odds")
        if target == "margin" else ("total_line", "over_odds", "under_odds"))
    line = pd.to_numeric(rows[line_col], errors="coerce")
    win, loss = implied(rows[win_col]), implied(rows[loss_col])
    score = rows.home_score-rows.away_score if target == "margin" else rows.home_score+rows.away_score
    usable = line.notna() & np.isfinite(line) & win.notna() & loss.notna()
    push = usable & score.eq(line)
    keep = usable & ~push
    selected = rows.loc[keep]
    if selected.empty:
        raise ValueError("No settled rows with paired historical prices")
    projected = scoring_projection(model, selected, target)
    probabilities = np.array([probability_above(mu, threshold, model["targets"][target]["sigma"])
                              for mu, threshold in zip(projected, line.loc[keep])])
    outcome = score.loc[keep].gt(line.loc[keep]).to_numpy(int)
    market = (win/(win+loss)).loc[keep].to_numpy(float)
    candidate_scores, market_scores = metrics(outcome, probabilities), metrics(outcome, market)
    return {"games": len(rows), "scored": len(selected), "pushes_excluded": int(push.sum()),
            "missing_prices_or_line": int((~usable).sum()), "candidate": candidate_scores,
            "market": market_scores, "decision": compare_candidate_to_market(candidate_scores, market_scores)}


def run(games: pd.DataFrame) -> dict:
    rows = pregame_features(games)
    if rows.empty:
        raise ValueError("No games with sufficient pregame history")
    rows = rows.loc[np.isfinite(rows.home_score) & np.isfinite(rows.away_score)].copy()
    train = rows.loc[rows.season.between(*PROTOCOL["train_seasons"])]
    calibration = rows.loc[rows.season.between(*PROTOCOL["calibration_seasons"])]
    evaluation = rows.loc[rows.season.between(*PROTOCOL["evaluation_seasons"])]
    model = fit_candidate(train, calibration)
    if evaluation.empty or calibration.gameday.max() >= evaluation.gameday.min():
        raise ValueError("Evaluation must follow calibration")
    return {"protocol": PROTOCOL, "status": "research_only_not_installed",
            "training_games": len(train), "calibration_games": len(calibration),
            "evaluation_games": len(evaluation), "model": model,
            "evaluation": {target: evaluate(model, evaluation, target) for target in ["margin", "total"]},
            "by_season": {str(year): {target: evaluate(model, evaluation.loc[evaluation.season.eq(year)], target)
                          for target in ["margin", "total"]} for year in sorted(evaluation.season.unique())}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    raw = args.games.read_bytes()
    report = run(pd.read_csv(args.games))
    report["source_sha256"] = hashlib.sha256(raw).hexdigest()
    report["source_url"] = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["status", "training_games", "calibration_games", "evaluation_games", "evaluation"]}, indent=2))


if __name__ == "__main__":
    main()
