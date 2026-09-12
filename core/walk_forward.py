"""Chronological evaluation helpers for probability models.

These helpers are intentionally model-agnostic. They make it difficult to
report an in-sample win rate as evidence of predictive performance.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def chronological_split(
    frame: pd.DataFrame,
    date_col: str,
    *,
    test_fraction: float = 0.20,
    min_train_rows: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return time-ordered train/test frames with no shuffling."""
    if frame is None or frame.empty:
        return frame.copy(), frame.copy()
    if not 0.0 < float(test_fraction) < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    if date_col not in frame.columns:
        raise KeyError(f"missing date column: {date_col}")

    out = frame.copy()
    dates = pd.to_datetime(out[date_col], errors="coerce", utc=True)
    if dates.isna().any():
        raise ValueError("date column contains invalid or missing values")
    out = out.assign(_evaluation_date=dates).sort_values("_evaluation_date", kind="stable")
    # Cut only between UTC calendar slates, never between rows on one day.
    days = out["_evaluation_date"].dt.normalize()
    boundaries = [i for i in range(1, len(out)) if days.iloc[i] != days.iloc[i - 1]
                  and i >= int(min_train_rows)]
    if not boundaries:
        raise ValueError("Need distinct calendar slates and enough training rows for a nonempty holdout")
    target = int(np.floor(len(out) * (1.0 - float(test_fraction))))
    cut = min(boundaries, key=lambda i: (abs(i - target), -i))
    train = out.iloc[:cut].drop(columns="_evaluation_date")
    test = out.iloc[cut:].drop(columns="_evaluation_date")
    return train, test


def probability_metrics(probabilities: Any, outcomes: Any, *, bins: int = 10) -> dict[str, Any]:
    """Calculate proper scoring metrics and a compact calibration table."""
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if p.shape != y.shape:
        raise ValueError("probabilities and outcomes must have the same shape")
    valid = np.isfinite(p) & np.isfinite(y)
    p = np.clip(p[valid], 1e-6, 1.0 - 1e-6)
    y = y[valid]
    if len(p) == 0:
        return {"n": 0, "brier": None, "log_loss": None, "calibration": []}

    brier = float(np.mean((p - y) ** 2))
    log_loss = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    bucket_rows = []
    bucket_ids = np.minimum(np.digitize(p, edges[1:-1], right=False), len(edges) - 2)
    for bucket in range(len(edges) - 1):
        mask = bucket_ids == bucket
        if not np.any(mask):
            continue
        bucket_rows.append({
            "lower": float(edges[bucket]),
            "upper": float(edges[bucket + 1]),
            "n": int(mask.sum()),
            "predicted": float(p[mask].mean()),
            "realized": float(y[mask].mean()),
        })
    return {"n": int(len(p)), "brier": brier, "log_loss": log_loss, "calibration": bucket_rows}


def walk_forward_evaluate(
    frame: pd.DataFrame,
    date_col: str,
    probability_col: str,
    outcome_col: str,
    *,
    min_train_rows: int = 100,
    test_fraction: float = 0.20,
) -> dict[str, Any]:
    """Evaluate the already-produced probabilities only on a future holdout.

    The function does not refit a model. It exists to enforce the reporting
    contract: the evaluated rows must be later than the training window.
    """
    train, test = chronological_split(
        frame,
        date_col,
        test_fraction=test_fraction,
        min_train_rows=min_train_rows,
    )
    if probability_col not in test.columns or outcome_col not in test.columns:
        raise KeyError("holdout is missing probability or outcome column")
    metrics = probability_metrics(test[probability_col], test[outcome_col])
    metrics.update({
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_end": str(pd.to_datetime(train[date_col], utc=True).max()),
        "test_start": str(pd.to_datetime(test[date_col], utc=True).min()),
        "out_of_sample": False,
        "chronological_holdout": True,
        "provenance_note": "Ordering alone does not establish training provenance; use scripts/validate_selector.py for provenance checks.",
    })
    return metrics


def compare_by_league(frame, date_col, probability_col, outcome_col, market_probability_col,
                      *, league_col='league', min_train_rows=100, test_fraction=.20):
    """Compare paired saved forecasts on later slates, separately by league.

    No fitting or weight changes occur here. A date split cannot prove that the
    submitted forecasts were generated out of sample; training provenance must
    be checked separately with validate_selector.py.
    """
    required={date_col,probability_col,outcome_col,market_probability_col,league_col}
    if not required.issubset(frame.columns):
        raise ValueError('Missing evaluation columns: '+', '.join(sorted(required-set(frame.columns))))
    result={}
    for league, rows in frame.groupby(league_col, dropna=False):
        label=str(league)
        rows=rows.copy()
        outcomes=rows[outcome_col].map(lambda x: {'WIN':1,'LOSS':0,'W':1,'L':0}.get(x,x))
        y=pd.to_numeric(outcomes,errors='coerce')
        model=pd.to_numeric(rows[probability_col],errors='coerce')
        market=pd.to_numeric(rows[market_probability_col],errors='coerce')
        # Report coverage: never compare differently filtered model/market sets.
        valid=y.isin([0,1]) & model.between(0,1) & market.between(0,1)
        excluded=int((~valid).sum())
        rows=rows.loc[valid].copy()
        rows[outcome_col]=y.loc[valid]
        rows[probability_col]=model.loc[valid]
        rows[market_probability_col]=market.loc[valid]
        base={'paired_rows':len(rows),'excluded_rows':excluded,'out_of_sample':False,
              'note':'Saved forecasts only; training provenance must be verified separately. No weights changed.'}
        if not len(rows):
            result[label]={**base,'status':'insufficient_history'}
            continue
        try:
            train,test=chronological_split(rows,date_col,min_train_rows=min_train_rows,test_fraction=test_fraction)
        except ValueError as exc:
            result[label]={**base,'status':'insufficient_history','reason':str(exc)}
            continue
        model_metrics=probability_metrics(test[probability_col],test[outcome_col])
        market_metrics=probability_metrics(test[market_probability_col],test[outcome_col])
        result[label]={**base,'status':'evaluated','train_rows':len(train),'test_rows':len(test),
                       'train_end':str(pd.to_datetime(train[date_col],utc=True).max()),
                       'test_start':str(pd.to_datetime(test[date_col],utc=True).min()),
                       'model':model_metrics,'market':market_metrics,
                       'model_minus_market':{k:model_metrics[k]-market_metrics[k] for k in ('brier','log_loss')}}
    return result
