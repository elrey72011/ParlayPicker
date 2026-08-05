"""Calibration and conservative decision probabilities for MLB player props.

The prop models produce useful *rank signals*, but their parametric probabilities are
not automatically calibrated.  This module keeps the raw model number visible while
turning a durable graded ledger into the probability used for staking.

Calibration is deliberately directional: ``batter_hits_over`` and
``batter_hits_under`` are different betting markets and must earn trust separately.
Fitted curves are shrink-only until an out-of-sample promotion proves that probability
uplift is durable.  When a directional sample is thin, the model is shrunk toward the
de-vigged market and receives an uncertainty haircut.  Missing history therefore
creates research rows, not false high-confidence production picks.
"""
from __future__ import annotations

from math import sqrt
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pandas as pd

from core.probability_calibration import apply_calibration, fit_isotonic_calibration


DEFAULT_PROP_RESULTS_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "prop_results"
    / "prop_results_log.csv"
)
DEFAULT_PROP_RESULTS_RUNTIME_PATH = (
    DEFAULT_PROP_RESULTS_PATH.parent / "prop_results_runtime.csv"
)
PROP_CALIBRATION_MIN_DIRECTIONAL = 20
PROP_CALIBRATION_MIN_POOLED = 40
PROP_CALIBRATION_PRIOR_WEIGHT = 35.0
PROP_CALIBRATION_Z = 0.75
PROP_UNCALIBRATED_EXTRA_HAIRCUT = 0.025
PROP_CALIBRATION_MAX_ROWS_PER_PROFILE = 250
# In-sample isotonic curves are shrink signals, not permission to move arbitrarily
# far from the de-vigged market. Bound the pre-haircut move until a separately
# validated promotion path exists.
PROP_CALIBRATION_MAX_MARKET_DEVIATION = 0.05


def directional_market_key(market_type: object, pick: object = None) -> str:
    """Return a stable ``market_family_side`` key without trusting player names."""
    market = str(market_type or "").strip().lower()
    if market.endswith(("_over", "_under")):
        return market
    text = f" {str(pick or '').strip().lower()} "
    side = "over" if " over " in text else "under" if " under " in text else ""
    if " tb " in text or " total bases " in text:
        family = "batter_total_bases"
    elif " hits " in text:
        family = "batter_hits"
    elif " outs " in text:
        family = "pitcher_outs"
    elif " bbs " in text or " walks " in text:
        family = "pitcher_walks"
    else:
        family = "pitcher_strikeouts"
    return f"{family}_{side}" if side else family


def _american_decimal(value: object) -> float | None:
    try:
        odds = float(value)
    except (TypeError, ValueError):
        return None
    if odds == 0:
        return None
    return 1.0 + (odds / 100.0 if odds > 0 else 100.0 / -odds)


def _market_probability(raw: pd.Series, edge: pd.Series) -> pd.Series:
    market = raw - edge
    return market.where(market.between(0.01, 0.99)).clip(0.01, 0.99)


def normalize_prop_results(
    log_df: pd.DataFrame | None,
    *,
    as_of_date: str | None = None,
) -> pd.DataFrame:
    """Normalize grader output into one leak-safe calibration table.

    Supported ledgers may call the prediction ``WinProbability``,
    ``raw_probability`` or ``predicted_probability`` and the outcome ``result`` or
    ``Outcome``. Pushes, voids and ungraded rows are excluded.
    """
    if log_df is None or len(log_df) == 0:
        return pd.DataFrame()
    out = log_df.copy()
    result_col = next((c for c in ("result", "Outcome", "outcome") if c in out.columns), None)
    prob_col = next(
        (c for c in ("raw_probability", "RawWinProbability", "WinProbability", "predicted_probability") if c in out.columns),
        None,
    )
    if result_col is None or prob_col is None:
        return pd.DataFrame()
    result = out[result_col].astype(str).str.upper().str.strip()
    out = out[result.isin(["WIN", "LOSS"])].copy()
    if out.empty:
        return out
    out["result"] = result.loc[out.index]
    out["outcome"] = out["result"].eq("WIN").astype(int)
    out["raw_probability"] = pd.to_numeric(out[prob_col], errors="coerce")
    out = out[out["raw_probability"].between(0.01, 0.99)].copy()
    mt = out.get("market_type", pd.Series("", index=out.index))
    pick = out.get("pick", out.get("best_pick", pd.Series("", index=out.index)))
    out["directional_market"] = [
        directional_market_key(m, p) for m, p in zip(mt, pick)
    ]
    out["odds_american"] = pd.to_numeric(
        out.get("odds_american", pd.Series(float("nan"), index=out.index)), errors="coerce"
    )
    date_col = next(
        (
            c
            for c in ("game_date", "date", "graded_at", "settled_at")
            if c in out.columns
        ),
        None,
    )
    if date_col is not None:
        out["result_date"] = pd.to_datetime(out[date_col], errors="coerce", utc=True)
        if as_of_date:
            cutoff = pd.to_datetime(as_of_date, errors="coerce", utc=True)
            if pd.notna(cutoff):
                # A slate may only learn from results known before that slate.
                out = out[out["result_date"].isna() | out["result_date"].lt(cutoff)]
    return out


def _profile(rows: pd.DataFrame, *, min_samples: int) -> dict[str, Any]:
    if "result_date" in rows.columns and rows["result_date"].notna().any():
        rows = rows.sort_values("result_date").tail(PROP_CALIBRATION_MAX_ROWS_PER_PROFILE)
    n = int(len(rows))
    wins = int(rows["outcome"].sum()) if n else 0
    losses = n - wins
    profits = []
    for _, row in rows.iterrows():
        dec = _american_decimal(row.get("odds_american"))
        if dec is None:
            continue
        profits.append(dec - 1.0 if int(row["outcome"]) == 1 else -1.0)
    roi = float(sum(profits) / len(profits)) if profits else float("nan")
    knots = None
    if n >= int(min_samples) and rows["outcome"].nunique() > 1:
        try:
            knots = fit_isotonic_calibration(
                rows["raw_probability"].tolist(), rows["outcome"].tolist()
            )
        except (TypeError, ValueError):
            knots = None
    return {
        "n": n,
        "wins": wins,
        "losses": losses,
        "hit_rate": (wins / n) if n else None,
        "roi": roi,
        "knots": knots,
    }


def fit_prop_calibration(
    log_df: pd.DataFrame | None,
    *,
    as_of_date: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Fit pooled and directional calibration profiles from settled pregame rows."""
    clean = normalize_prop_results(log_df, as_of_date=as_of_date)
    if clean.empty:
        return {}
    profiles = {
        "__pooled__": _profile(clean, min_samples=PROP_CALIBRATION_MIN_POOLED)
    }
    for key, rows in clean.groupby("directional_market"):
        profiles[str(key)] = _profile(
            rows, min_samples=PROP_CALIBRATION_MIN_DIRECTIONAL
        )
    return profiles


def apply_prop_calibration(
    card: pd.DataFrame,
    log_df: pd.DataFrame | None,
    *,
    as_of_date: str | None = None,
) -> pd.DataFrame:
    """Add calibrated fields and make the conservative probability decision-grade.

    ``WinProbability`` is intentionally replaced with the conservative lower estimate
    because every downstream guard historically consumes that column.  The original is
    retained as ``RawWinProbability`` for auditability.
    """
    if card is None or card.empty:
        return card
    out = card.copy()
    raw = pd.to_numeric(out.get("WinProbability"), errors="coerce").clip(0.01, 0.99)
    edge = pd.to_numeric(
        out.get("edge", pd.Series(float("nan"), index=out.index)), errors="coerce"
    )
    market_prob = _market_probability(raw, edge).fillna(0.5)
    profiles = fit_prop_calibration(log_df, as_of_date=as_of_date)
    pooled = profiles.get("__pooled__", {})

    calibrated_values: list[float] = []
    conservative_values: list[float] = []
    sample_sizes: list[int] = []
    profile_sample_sizes: list[int] = []
    directional_sample_sizes: list[int] = []
    sources: list[str] = []
    keys: list[str] = []

    for idx in out.index:
        key = directional_market_key(
            out.at[idx, "market_type"] if "market_type" in out.columns else "",
            out.at[idx, "best_pick"] if "best_pick" in out.columns else "",
        )
        keys.append(key)
        directional_profile = profiles.get(key, {})
        directional_n = int(directional_profile.get("n", 0) or 0)
        profile = directional_profile
        source = "directional"
        if not profile.get("knots"):
            if pooled.get("knots"):
                profile = pooled
                source = "pooled"
            else:
                source = "market_blend_fallback"
        n = int(profile.get("n", 0) or 0)
        p_raw = float(raw.loc[idx])
        p_market = float(market_prob.loc[idx])
        if profile.get("knots"):
            fitted = float(
                apply_calibration(pd.Series([p_raw]), profile["knots"]).iloc[0]
            )
            reliability = n / (n + PROP_CALIBRATION_PRIOR_WEIGHT)
            calibrated = reliability * fitted + (1.0 - reliability) * p_market
        else:
            # With no settled evidence, trust only a quarter of model/market disagreement.
            calibrated = p_market + 0.25 * (p_raw - p_market)
        calibrated = min(
            p_market + PROP_CALIBRATION_MAX_MARKET_DEVIATION,
            max(p_market - PROP_CALIBRATION_MAX_MARKET_DEVIATION, calibrated),
        )
        # In-sample isotonic tails can end in a tiny perfect plateau.  They are useful
        # for identifying overconfidence, but must not turn a raw rank signal into a
        # stronger probability without a separately promoted, out-of-sample curve.
        # Keep both pooled and directional calibration shrink-only in production.
        if source in {"directional", "pooled"}:
            calibrated = min(calibrated, p_raw)
        calibrated = min(0.95, max(0.05, calibrated))
        uncertainty = 0.0
        uncertainty_n = directional_n if source == "pooled" else n
        if uncertainty_n:
            uncertainty = PROP_CALIBRATION_Z * sqrt(
                calibrated * (1.0 - calibrated)
                / (uncertainty_n + PROP_CALIBRATION_PRIOR_WEIGHT)
            )
        if source in {"pooled", "market_blend_fallback"}:
            uncertainty += PROP_UNCALIBRATED_EXTRA_HAIRCUT
        conservative = min(calibrated, max(0.05, calibrated - uncertainty))
        calibrated_values.append(calibrated)
        conservative_values.append(conservative)
        sample_sizes.append(directional_n)
        profile_sample_sizes.append(n)
        directional_sample_sizes.append(directional_n)
        sources.append(source)

    calibrated = pd.Series(calibrated_values, index=out.index)
    conservative = pd.Series(conservative_values, index=out.index)
    decimals = pd.to_numeric(out.get("odds_american"), errors="coerce").map(
        lambda x: _american_decimal(x) if pd.notna(x) else None
    )
    out["directional_market"] = keys
    out["RawWinProbability"] = raw.round(4)
    out["MarketProbability"] = market_prob.round(4)
    out["CalibratedProbability"] = calibrated.round(4)
    out["ConservativeWinProbability"] = conservative.round(4)
    out["CalibrationSampleSize"] = sample_sizes
    out["CalibrationProfileSampleSize"] = profile_sample_sizes
    out["DirectionalCalibrationSampleSize"] = directional_sample_sizes
    out["CalibrationSource"] = sources
    out["CalibrationMarketDeviationCap"] = PROP_CALIBRATION_MAX_MARKET_DEVIATION
    out["WinProbability"] = conservative.round(4)
    out["edge"] = (conservative - market_prob).round(4)
    out["expected_value"] = (conservative * decimals - 1.0).round(4)
    return out


def load_prop_results_log(
    path: Path | str = DEFAULT_PROP_RESULTS_PATH,
    uploaded: Any = None,
    *,
    runtime_path: Path | str | None = None,
) -> pd.DataFrame | None:
    """Load uploaded history or the deploy baseline plus its runtime recovery.

    The tracked baseline makes deployments deterministic.  Ledgers uploaded or
    generated by the app are persisted separately and layered over that baseline
    on a later process/session start.  A custom ``path`` remains isolated unless a
    custom ``runtime_path`` is also supplied, which keeps tests and CLI callers
    from accidentally reading application state.
    """
    if uploaded is not None:
        try:
            from app_core.prop_grading import merge_prop_ledgers

            if isinstance(uploaded, pd.DataFrame):
                return uploaded.copy()
            uploads = (
                list(uploaded)
                if isinstance(uploaded, (list, tuple))
                else [uploaded]
            )
            merged = pd.DataFrame()
            for item in uploads:
                if item is None:
                    continue
                if hasattr(item, "seek"):
                    item.seek(0)
                frame = pd.read_csv(item)
                if hasattr(item, "seek"):
                    item.seek(0)
                merged = merge_prop_ledgers(merged, frame)
            return merged if not merged.empty else None
        except (OSError, ValueError, TypeError):
            return None
    try:
        from app_core.prop_grading import merge_prop_ledgers

        file_path = Path(path)
        baseline = pd.read_csv(file_path) if file_path.exists() else pd.DataFrame()
        recovery_path = runtime_path
        if recovery_path is None:
            try:
                is_default = file_path.resolve() == DEFAULT_PROP_RESULTS_PATH.resolve()
            except OSError:
                is_default = False
            recovery_path = DEFAULT_PROP_RESULTS_RUNTIME_PATH if is_default else None
        recovery = pd.DataFrame()
        if recovery_path is not None:
            candidate = Path(recovery_path)
            if candidate.exists():
                recovery = pd.read_csv(candidate)
        merged = merge_prop_ledgers(baseline, recovery)
        return merged if not merged.empty else None
    except (OSError, ValueError):
        return None


def persist_prop_results_log(
    ledger: pd.DataFrame | None,
    path: Path | str = DEFAULT_PROP_RESULTS_RUNTIME_PATH,
) -> bool:
    """Atomically save cumulative history for recovery after an app restart."""
    if not isinstance(ledger, pd.DataFrame) or ledger.empty:
        return False
    destination = Path(path)
    temporary: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            suffix=".tmp",
            prefix=f".{destination.stem}-",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            ledger.to_csv(handle, index=False)
        temporary.replace(destination)
        return True
    except (OSError, ValueError, TypeError):
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        return False

