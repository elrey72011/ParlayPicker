"""Market-anchored NCAAF model, screened on archived lines (roadmap C5 after D18).

The frozen research model predicts scores from team stats the market already prices in,
and lost to the line (``docs/evaluation/ncaaf-line-backtest-2026-11-05.md``). This model
starts from the line and predicts only what the market might miss:

* **Spread:** the home cover margin, margin + home spread.
* **Total:** the error of the total line, total points - total line.

Ridge regression on the research features plus the line itself, fit on 2023, centred and
scaled on 2024 (bias and residual sigma), then scored once on 2025 at the archived line.
With no real signal it predicts ~0, so P(cover) stays near 50% and few or no picks clear
a threshold. That is the honest outcome. Research only.
"""
from __future__ import annotations

import math

import numpy as np

from app_core.ncaaf_research import FEATURES

ALPHA = 25.0
THRESHOLDS = (0.52, 0.55, 0.58)
WIN = 100 / 110


def _rows(split, lines, market):
    key = "spread" if market == "SPREAD" else "total"
    out = []
    for row in split:
        line = (lines.get(row["game_id"]) or {}).get(key)
        if line is None:
            continue
        target = row["y_margin"] + line if market == "SPREAD" else row["y_total"] - line
        out.append((row, float(line), float(target)))
    return out


def _matrix(rows):
    return np.array([[float(r[k]) for k in FEATURES] + [line] for r, line, _ in rows], dtype=float)


def fit(train, calibration, lines, market):
    """Ridge on 2023; bias and sigma from 2024. Returns the model dict, or None if too few rows."""
    train_rows, cal_rows = _rows(train, lines, market), _rows(calibration, lines, market)
    if len(train_rows) < 50 or len(cal_rows) < 50:
        return None
    x = _matrix(train_rows)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1
    z = (x - mean) / scale
    y = np.array([t for _, _, t in train_rows])
    coef = np.linalg.solve(z.T @ z + ALPHA * np.eye(z.shape[1]), z.T @ (y - y.mean()))
    model = {"market": market, "mean": mean.tolist(), "scale": scale.tolist(), "coef": coef.tolist(),
             "intercept": float(y.mean()), "train_games": len(train_rows)}
    predicted = predict(model, cal_rows)
    actual = np.array([t for _, _, t in cal_rows])
    model["bias"] = float(np.mean(actual - predicted))
    model["sigma"] = max(1.0, float(np.sqrt(np.mean((actual - predicted - model["bias"]) ** 2))))
    model["calibration_games"] = len(cal_rows)
    return model


def predict(model, rows):
    z = (_matrix(rows) - np.array(model["mean"])) / np.array(model["scale"])
    return z @ np.array(model["coef"]) + model["intercept"] + model.get("bias", 0.0)


def evaluate(model, holdout, lines):
    """Picks on the holdout at the archived line: home/Over when P(target > 0) >= threshold, else away/Under."""
    rows = _rows(holdout, lines, model["market"])
    if not rows:
        return {"games": 0}
    centres = predict(model, rows)
    graded = []
    for (row, line, target), mu in zip(rows, centres):
        p = 0.5 * (1 + math.erf(mu / (model["sigma"] * math.sqrt(2))))  # P(target > 0)
        side_a = p >= 0.5
        confidence = p if side_a else 1 - p
        outcome = "PUSH" if abs(target) < 1e-9 else ("WIN" if (target > 0) == side_a else "LOSS")
        graded.append((confidence, p, target, outcome))
    decided = [(p, int(t > 0)) for _, p, t, o in graded if o != "PUSH"]
    out = {"games": len(graded), "brier": sum((p - y) ** 2 for p, y in decided) / len(decided) if decided else None,
           "mean_abs_centre": float(np.mean(np.abs(centres))), "by_threshold": {}}
    for threshold in THRESHOLDS:
        picks = [g for g in graded if g[0] >= threshold]
        wins = sum(1 for g in picks if g[3] == "WIN")
        losses = sum(1 for g in picks if g[3] == "LOSS")
        units = wins * WIN - losses
        out["by_threshold"][f"{threshold:.2f}"] = {
            "picks": len(picks), "wins": wins, "losses": losses,
            "hit_rate": wins / (wins + losses) if wins + losses else None,
            "roi_per_pick": units / len(picks) if picks else None}
    return out
