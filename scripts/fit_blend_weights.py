#!/usr/bin/env python3
"""Fit probability-blend weights against historical pick outcomes.

The live blend (``core.streamlit_pipeline.compute_blended_probability``) combines
four signals — Kalshi, Market (Novig de-vig), TheOver (pitcher/pace) and ML
(XGBoost) — with hand-set weights in ``app_core/weights_config.py``. Those weights
are currently chosen by reasoning, not fitted to data. This script fits them by
minimising log-loss against realised WIN/LOSS outcomes, so the weights can be
replaced with empirically-optimal ones once enough graded slates accumulate.

Two modes
---------
1. ``fit`` (default): fit weights to the *oriented* per-signal inputs. Requires the
   ``blend_in_kalshi / blend_in_market / blend_in_ml / blend_in_theover`` columns
   that the pipeline now persists to every best-picks export. Older exports that
   predate those columns cannot be used for fitting — their signal orientation is
   ambiguous after the fact, which is exactly why the pipeline now records the
   oriented inputs explicitly.

2. ``--calibration-only``: skip fitting and just measure how well an existing
   probability column (e.g. ``WinProbability`` / ``calibrated_probability``) is
   calibrated against outcomes (Brier score, log-loss, decile reliability). This
   works on any graded export, including the legacy Strategy-Lab recaps, and is
   useful for quantifying the calibration problem today.

Outcome labels are read from a column (``Pick_Outcome``, ``W/L``, ``outcome`` …);
WIN→1, LOSS→0, PUSH→dropped.

Example
-------
    python scripts/fit_blend_weights.py \
        --exports "app_exports/best_picks_export*.csv" \
        --league MLB --market total --min-samples 150

    python scripts/fit_blend_weights.py --exports "drive_recaps/*.csv" \
        --calibration-only --prob-col WinProbability
"""
from __future__ import annotations

import argparse
import glob
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Keep names, persisted columns, and production weight vectors in one canonical
# order. A previous pair of independent lists silently swapped TheOver and ML.
SIGNAL_SPECS = [
    ("Kalshi", "blend_in_kalshi"),
    ("Market", "blend_in_market"),
    ("TheOver", "blend_in_theover"),
    ("ML", "blend_in_ml"),
]
SIGNAL_NAMES = [name for name, _column in SIGNAL_SPECS]
SIGNAL_COLS = [column for _name, column in SIGNAL_SPECS]

_WIN_TOKENS = {"win", "w", "won", "1", "true"}
_LOSS_TOKENS = {"loss", "l", "lose", "lost", "0", "false"}
_PUSH_TOKENS = {"push", "p", "tie", "void", "tie/push"}
_OUTCOME_CANDIDATES = ["Pick_Outcome", "W/L", "outcome", "result", "Result", "WL"]


# --------------------------------------------------------------------------- #
# Core math — mirrors compute_blended_probability's present-signal renorm.
# --------------------------------------------------------------------------- #
def blended_prob(signals: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Blend per-row, dropping NaN signals and renormalising their weight.

    This reproduces the live blend exactly: a missing signal does not vote 0.5,
    it is removed and the remaining weights are renormalised to sum to 1.

    Parameters
    ----------
    signals : (n, k) array, may contain NaN.
    weights : (k,) non-negative array.
    """
    signals = np.asarray(signals, dtype=float)
    weights = np.asarray(weights, dtype=float)
    present = ~np.isnan(signals)
    w = np.where(present, weights[None, :], 0.0)
    wsum = w.sum(axis=1)
    # Rows with no present signal fall back to 0.5 (matches the live blend guard).
    filled = np.where(present, signals, 0.0)
    num = (filled * w).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(wsum > 0, num / wsum, 0.5)
    return np.clip(p, 1e-6, 1 - 1e-6)


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def fit_weights(signals: np.ndarray, y: np.ndarray, seed_weights: np.ndarray | None = None) -> np.ndarray:
    """Fit non-negative weights summing to 1 that minimise log-loss.

    Falls back to a projected-gradient search if SciPy is unavailable.
    """
    k = signals.shape[1]
    x0 = np.full(k, 1.0 / k) if seed_weights is None else np.asarray(seed_weights, float)

    def objective(w: np.ndarray) -> float:
        return log_loss(y, blended_prob(signals, w))

    try:
        from scipy.optimize import minimize

        res = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * k,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
            options={"maxiter": 500, "ftol": 1e-9},
        )
        w = np.clip(res.x, 0.0, None)
        return w / w.sum()
    except Exception:  # pragma: no cover - scipy is normally present
        return _fit_weights_pgd(signals, y, x0)


def _fit_weights_pgd(signals: np.ndarray, y: np.ndarray, x0: np.ndarray, steps: int = 4000, lr: float = 0.5) -> np.ndarray:
    """Projected gradient descent on the simplex (SciPy-free fallback)."""
    w = x0.copy()
    n = len(y)
    for _ in range(steps):
        p = blended_prob(signals, w)
        present = ~np.isnan(signals)
        wrow = np.where(present, w[None, :], 0.0)
        wsum = wrow.sum(axis=1, keepdims=True)
        filled = np.where(present, signals, 0.0)
        # d p / d w_j (approx, treating renorm denominator)
        dp_dw = np.where(present, (filled - p[:, None]) / np.where(wsum > 0, wsum, 1.0), 0.0)
        grad = ((p - y)[:, None] / (p * (1 - p))[:, None] * dp_dw).sum(axis=0) / n
        w = w - lr * grad
        w = _project_simplex(w)
    return w


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection of v onto the probability simplex."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    rho = np.nonzero(u - css / (np.arange(len(u)) + 1) > 0)[0][-1]
    theta = css[rho] / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def walk_forward_indices(n: int, folds: int, min_train_rows: int | None = None):
    """Yield expanding-window train/test indices without future leakage."""
    if n < 2:
        return []
    min_train = int(min_train_rows or max(1, n // 2))
    min_train = min(max(min_train, 1), n - 1)
    boundaries = np.linspace(min_train, n, min(int(folds), n - min_train) + 1, dtype=int)
    splits = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end > start:
            splits.append((np.arange(0, start), np.arange(start, end)))
    return splits


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _to_outcome(series: pd.Series) -> pd.Series:
    """Map a free-text outcome column to 1 (win) / 0 (loss) / NaN (push/unknown)."""
    s = series.astype(str).str.strip().str.lower()

    def m(v: str):
        if v in _WIN_TOKENS:
            return 1.0
        if v in _LOSS_TOKENS:
            return 0.0
        if v in _PUSH_TOKENS:
            return np.nan
        return np.nan

    return s.map(m)


def _pct_to_float(series: pd.Series) -> pd.Series:
    """Parse probabilities that may be '67.9%' or 0.679."""
    s = series.astype(str).str.strip().str.replace("%", "", regex=False)
    out = pd.to_numeric(s, errors="coerce")
    # Values that look like percentages (>1) are divided by 100.
    return out.where(out <= 1.0, out / 100.0)


def load_exports(pattern: str, outcome_col: str | None) -> pd.DataFrame:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"No files matched: {pattern}")
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:  # noqa: BLE001
            print(f"  ! skipping {p}: {e}", file=sys.stderr)
            continue
        df["__source_file"] = p
        frames.append(df)
    if not frames:
        raise SystemExit("No readable CSVs.")
    df = pd.concat(frames, ignore_index=True)

    date_col = next((c for c in ("game_date", "event_date", "date") if c in df.columns), None)
    dates = (
        pd.to_datetime(df[date_col], errors="coerce", utc=True)
        if date_col
        else pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    )
    if dates.isna().any():
        source_dates = df["__source_file"].astype(str).str.extract(
            r"(20\d{2}-\d{2}-\d{2})", expand=False
        )
        dates = dates.fillna(pd.to_datetime(source_dates, errors="coerce", utc=True))
    df["__evaluation_date"] = dates

    col = outcome_col or next((c for c in _OUTCOME_CANDIDATES if c in df.columns), None)
    if col is None:
        raise SystemExit(
            "No outcome column found. Provide --outcome-col (looked for: "
            + ", ".join(_OUTCOME_CANDIDATES) + ")"
        )
    df["__y"] = _to_outcome(df[col])
    print(f"Loaded {len(df)} rows from {len(paths)} file(s); outcome column = '{col}'")
    return df


def filter_cell(df: pd.DataFrame, league: str | None, market: str | None) -> pd.DataFrame:
    out = df
    if league and "league" in out.columns:
        out = out[out["league"].astype(str).str.upper() == league.upper()]
    if market and "market_type" in out.columns:
        out = out[out["market_type"].astype(str).str.lower().str.contains(market.lower(), na=False)]
    return out


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
@dataclass
class WeightSet:
    name: str
    weights: np.ndarray


def _current_weights(league: str, market: str) -> WeightSet | None:
    """Read the current hand-set weights from weights_config for this cell."""
    try:
        import app_core.weights_config as w
    except Exception:
        return None
    lg, mk = league.upper(), market.lower()
    if lg == "MLB" and "total" in mk:
        vec = [w.MLB_TOTAL_KALSHI_WEIGHT, w.MLB_TOTAL_MARKET_WEIGHT,
               w.MLB_TOTAL_THEOVER_WEIGHT, w.MLB_TOTAL_ML_WEIGHT]
    elif lg == "NBA" and "total" in mk:
        # NBA totals keep base Kalshi/Market with NBA TheOver/ML overrides.
        vec = [w.KALSHI_WEIGHT, w.MARKET_WEIGHT, w.NBA_TOTAL_THEOVER_WEIGHT, w.NBA_TOTAL_ML_WEIGHT]
    else:
        vec = [w.KALSHI_WEIGHT, w.MARKET_WEIGHT, w.THEOVER_WEIGHT, w.ML_MODEL_WEIGHT]
    vec = np.asarray(vec, float)
    return WeightSet("current", vec / vec.sum())


def _print_weight_table(sets: list[WeightSet]) -> None:
    header = "  ".join(f"{n:>8}" for n in SIGNAL_NAMES)
    print(f"\n{'weights':>16} | {header}")
    print("-" * (18 + len(header) + 2))
    for ws in sets:
        row = "  ".join(f"{v:8.3f}" for v in ws.weights)
        print(f"{ws.name:>16} | {row}")


def evaluate(signals: np.ndarray, y: np.ndarray, ws: WeightSet) -> dict:
    p = blended_prob(signals, ws.weights)
    return {"log_loss": log_loss(y, p), "brier": brier(y, p)}


def cross_val(signals: np.ndarray, y: np.ndarray, folds: int, seed_weights: np.ndarray) -> dict:
    splits = walk_forward_indices(len(y), folds)
    if not splits:
        return {}
    ll, br = [], []
    for train, test in splits:
        w = fit_weights(signals[train], y[train], seed_weights=seed_weights)
        p = blended_prob(signals[test], w)
        ll.append(log_loss(y[test], p))
        br.append(brier(y[test], p))
    return {
        "cv_log_loss": float(np.mean(ll)),
        "cv_brier": float(np.mean(br)),
        "cv_folds": len(splits),
        "out_of_sample": True,
    }


def chronological_promotion_gate(
    signals: np.ndarray,
    y: np.ndarray,
    current_weights: np.ndarray,
    *,
    test_fraction: float = 0.20,
) -> dict:
    """Require fitted weights to beat current and simple future baselines."""
    n = len(y)
    if n < 2:
        return {"promotable": False, "reason": "not enough dated rows"}
    cut = min(max(int(np.floor(n * (1.0 - test_fraction))), 1), n - 1)
    train, test = np.arange(cut), np.arange(cut, n)
    fitted = fit_weights(signals[train], y[train], seed_weights=current_weights)
    candidates = {
        "fitted": blended_prob(signals[test], fitted),
        "current": blended_prob(signals[test], current_weights),
    }
    for name, column in (("kalshi", 0), ("market", 1)):
        present = np.isfinite(signals[test, column])
        if present.any():
            candidate = np.full(len(test), float(y[train].mean()))
            candidate[present] = signals[test, column][present]
            candidates[name] = np.clip(candidate, 1e-6, 1 - 1e-6)
    candidates["train_base_rate"] = np.full(len(test), float(y[train].mean()))
    metrics = {
        name: {"log_loss": log_loss(y[test], p), "brier": brier(y[test], p)}
        for name, p in candidates.items()
    }
    benchmarks = [metrics[name] for name in metrics if name != "fitted"]
    promotable = all(
        metrics["fitted"][metric] < min(row[metric] for row in benchmarks)
        for metric in ("log_loss", "brier")
    )
    return {
        "promotable": bool(promotable),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "weights": fitted,
        "metrics": metrics,
        "reason": (
            "beats all holdout baselines"
            if promotable
            else "does not beat every holdout baseline"
        ),
    }


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 5) -> None:
    print(f"\nReliability (predicted vs actual win rate, {bins} bins):")
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if m.sum() == 0:
            continue
        print(f"  [{lo:.2f},{hi:.2f}) n={int(m.sum()):4d}  pred={p[m].mean():.3f}  actual={y[m].mean():.3f}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run_calibration_only(df: pd.DataFrame, prob_col: str) -> int:
    if prob_col not in df.columns:
        raise SystemExit(f"--prob-col '{prob_col}' not in data. Columns: {list(df.columns)[:20]}…")
    work = df[df["__y"].notna()].copy()
    p = _pct_to_float(work[prob_col]).to_numpy()
    y = work["__y"].to_numpy(dtype=float)
    keep = ~np.isnan(p)
    p, y = p[keep], y[keep]
    if len(y) == 0:
        raise SystemExit("No graded rows with a usable probability.")
    print(f"\nCalibration of '{prob_col}' on {len(y)} graded picks:")
    print(f"  win rate     : {y.mean():.3f}")
    print(f"  mean predicted: {p.mean():.3f}")
    print(f"  Brier score  : {brier(y, p):.4f}  (lower is better; 0.25 = coin flip)")
    print(f"  log-loss     : {log_loss(y, p):.4f}  (0.693 = coin flip)")
    reliability(y, p)
    return 0


def run_fit(df: pd.DataFrame, league: str, market: str, min_samples: int, folds: int) -> int:
    missing = [c for c in SIGNAL_COLS if c not in df.columns]
    if missing:
        raise SystemExit(
            "These exports lack the oriented blend-input columns "
            f"{missing}. Only exports produced after the instrumentation change "
            "can be used for fitting. Use --calibration-only on legacy exports."
        )
    cell = filter_cell(df, league, market)
    cell = cell[cell["__y"].notna()].copy()
    dated = "__evaluation_date" in cell.columns and cell["__evaluation_date"].notna().all()
    if dated:
        cell = cell.sort_values("__evaluation_date", kind="stable")
    signals = cell[SIGNAL_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y = cell["__y"].to_numpy(dtype=float)
    # Require at least one present signal per row.
    keep = ~np.all(np.isnan(signals), axis=1)
    signals, y = signals[keep], y[keep]
    n = len(y)

    print(f"\nCell: league={league} market~='{market}'  graded picks n={n}")
    if n == 0:
        raise SystemExit("No graded rows for this cell.")
    print(f"  signal coverage: " + ", ".join(
        f"{name}={int((~np.isnan(signals[:, i])).sum())}" for i, name in enumerate(SIGNAL_NAMES)))

    current = _current_weights(league, market)
    fitted = WeightSet("fitted", fit_weights(signals, y, seed_weights=current.weights if current else None))

    sets = [s for s in (current, fitted) if s is not None]
    _print_weight_table(sets)

    print(f"\n{'':>16} | {'log_loss':>10}  {'brier':>8}")
    print("-" * 42)
    for ws in sets:
        m = evaluate(signals, y, ws)
        print(f"{ws.name + ' (train)':>16} | {m['log_loss']:10.4f}  {m['brier']:8.4f}")
    cv = (
        cross_val(signals, y, folds, seed_weights=current.weights)
        if dated and current is not None
        else {}
    )
    if cv:
        print(f"{'fitted (CV)':>16} | {cv['cv_log_loss']:10.4f}  {cv['cv_brier']:8.4f}")

    reliability(y, blended_prob(signals, fitted.weights))

    promotion = (
        chronological_promotion_gate(signals, y, current.weights)
        if dated and current is not None and n >= min_samples
        else {"promotable": False, "reason": "insufficient dated samples"}
    )
    if promotion.get("metrics"):
        print("\nChronological holdout promotion gate:")
        for name, metrics in promotion["metrics"].items():
            print(
                f"  {name:<15} log_loss={metrics['log_loss']:.4f} "
                f"brier={metrics['brier']:.4f}"
            )
    if n >= min_samples and not promotion["promotable"]:
        print(f"\nNOT PROMOTABLE: {promotion['reason']}. Keep the current production weights.")
        return 0

    if n < min_samples:
        print(
            f"\n⚠  NOT ENOUGH DATA to trust these weights (n={n} < {min_samples}). "
            "Treat the fit as illustrative only; keep accumulating graded slates."
        )
    else:
        print("\nSuggested weights_config values for this cell:")
        for name, v in zip(SIGNAL_NAMES, fitted.weights):
            print(f"  {name:<8} = {v:.2f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exports", required=True, help="Glob for graded best-picks export CSV(s).")
    ap.add_argument("--league", default="MLB")
    ap.add_argument("--market", default="total", help="Substring match on market_type (e.g. total, spread, h2h).")
    ap.add_argument("--outcome-col", default=None, help="Column holding WIN/LOSS (auto-detected if omitted).")
    ap.add_argument("--min-samples", type=int, default=150, help="Min graded picks before recommending weights.")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--calibration-only", action="store_true",
                    help="Skip fitting; report calibration of --prob-col instead.")
    ap.add_argument("--prob-col", default="calibrated_probability",
                    help="Probability column to assess in --calibration-only mode.")
    args = ap.parse_args(argv)

    df = load_exports(args.exports, args.outcome_col)
    if args.calibration_only:
        df = filter_cell(df, args.league, args.market)
        return run_calibration_only(df, args.prob_col)
    return run_fit(df, args.league, args.market, args.min_samples, args.folds)


if __name__ == "__main__":
    raise SystemExit(main())
