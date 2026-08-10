from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# Default location of the recap-fitted isotonic calibration table written by
# scripts/fit_calibration.py. Callers (e.g. generate_parlays) load this and pass
# it explicitly — nothing in core auto-loads it, so unit tests and callers that
# want raw probabilities are unaffected.
DEFAULT_CALIBRATION_PATH = Path("data/calibration/effective_prob_calibration.json")


def fit_isotonic_calibration(probs: list[float], outcomes: list[int]) -> list[list[float]]:
    """Fit a monotone predicted->realized probability mapping via pooled adjacent
    violators (PAV). Dependency-free on purpose: sklearn is not guaranteed in the
    runtime environments this repo deploys to.

    Returns a list of [predicted, realized] knots (block means), suitable for
    ``apply_calibration`` linear interpolation. Outcomes are 1 for WIN, 0 for LOSS;
    pushes/voids must be excluded by the caller.
    """
    pairs = sorted(
        (float(p), float(o)) for p, o in zip(probs, outcomes)
        if pd.notna(p) and pd.notna(o) and 0.0 < float(p) < 1.0
    )
    if len(pairs) < 2:
        raise ValueError("need at least 2 graded picks to fit calibration")

    # Each block: [sum_x, sum_y, weight]
    blocks: list[list[float]] = []
    for x, y in pairs:
        blocks.append([x, y, 1.0])
        # Merge while the realized mean is non-monotone
        while len(blocks) >= 2 and blocks[-2][1] / blocks[-2][2] > blocks[-1][1] / blocks[-1][2]:
            x2, y2, w2 = blocks.pop()
            blocks[-1][0] += x2
            blocks[-1][1] += y2
            blocks[-1][2] += w2

    return [[sx / w, sy / w] for sx, sy, w in blocks]


def apply_calibration(probs: pd.Series, table: list[list[float]] | None) -> pd.Series:
    """Map predicted probabilities through the fitted knots with linear
    interpolation; values outside the fitted range clamp to the end knots.
    Returns ``probs`` unchanged when no table is available."""
    if not table:
        return probs
    xs = [float(k[0]) for k in table]
    ys = [float(k[1]) for k in table]

    def _interp(p: float) -> float:
        if pd.isna(p):
            return p
        if p <= xs[0]:
            return ys[0]
        if p >= xs[-1]:
            return ys[-1]
        for i in range(1, len(xs)):
            if p <= xs[i]:
                span = xs[i] - xs[i - 1]
                frac = (p - xs[i - 1]) / span if span > 0 else 0.0
                return ys[i - 1] + frac * (ys[i] - ys[i - 1])
        return ys[-1]

    return pd.to_numeric(probs, errors="coerce").map(_interp).clip(0.0, 1.0)


def apply_bucket_calibration(
    probs: pd.Series,
    buckets,
    table: list[list[float]] | None,
    bucket_stats: dict | None,
    shrink_n: int = 50,
) -> pd.Series:
    """Bucket-CONDITIONAL calibration: the global isotonic curve, then a per-bucket tilt.

    The global ``table`` fixes the pooled predicted->realized mapping but averages away the
    fact that accuracy varies a lot by bucket (e.g. MLB under:Agrees ~61% vs over:Neutral
    ~49%). This adds the bucket's realized delta from the overall rate, Laplace-smoothed and
    shrunk by sample size (``n/(n+shrink_n)``), so proven buckets aren't crushed by the pooled
    curve and thin buckets barely move. Identical tilt math to
    ``empirical_tiers.empirical_win_probability`` (pinned by test) — this is just the
    vectorized form so the gates and the lean view share one bucket-aware number.

    Falls back to the plain global calibration when ``bucket_stats`` is missing.
    """
    cal = apply_calibration(probs, table)
    if not bucket_stats or not bucket_stats.get("buckets"):
        return cal
    overall = float(bucket_stats["overall"]["win_rate"])
    bmap = bucket_stats["buckets"]
    cal = pd.to_numeric(cal, errors="coerce")
    bucket_list = list(buckets)

    def _tilt(p, b):
        if pd.isna(p):
            return p
        rec = bmap.get(b)
        if not rec or int(rec.get("n", 0)) <= 0:
            return float(p)
        n = int(rec["n"])
        smoothed = (int(rec["wins"]) + overall * 10.0) / (n + 10.0)
        weight = n / (n + float(shrink_n))
        return float(min(0.95, max(0.05, float(p) + weight * (smoothed - overall))))

    return pd.Series([_tilt(p, b) for p, b in zip(cal, bucket_list)], index=cal.index)


def save_calibration(table: list[list[float]], path: Path | str, meta: dict | None = None) -> None:
    payload = {"knots": table, "meta": meta or {}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_calibration(path: Path | str | None = None) -> list[list[float]] | None:
    """Load a fitted calibration table, failing closed for production.

    The default artifact drives live selection, tiers, recovery, Kelly sizing,
    and parlays, so it is eligible only after the chronological promotion test
    in :mod:`scripts.fit_calibration` records ``validation.promotable=true``.
    Explicit paths remain permissive for tests, research, and legacy imports.
    Missing, unreadable, or unapproved production artifacts return ``None`` so
    callers fall back to the upstream effective probability.
    """
    production_default = path is None
    calibration_path = DEFAULT_CALIBRATION_PATH if path is None else Path(path)
    try:
        payload = json.loads(calibration_path.read_text())
        if production_default:
            validation = (payload.get("meta") or {}).get("validation") or {}
            if validation.get("promotable") is not True:
                return None
        knots = payload.get("knots")
        return knots if knots else None
    except (OSError, ValueError):
        return None


def calibrate_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """Blend model and market probabilities to reduce overconfidence."""
    if df is None or df.empty or "model_probability" not in df.columns:
        return df

    calibrated = df.copy()
    calibrated["calibrated_probability"] = (
        pd.to_numeric(calibrated["model_probability"], errors="coerce").fillna(0.5) * 0.3
        + pd.to_numeric(calibrated.get("market_probability", 0.5), errors="coerce").fillna(0.5) * 0.7
    ).clip(0.0, 1.0)

    return calibrated
