"""Target-specific spread and total probability model.

The shipped XGBoost classifier predicts ``home_won``.  Its output is useful for
moneylines, but it is not P(cover) or P(over).  This module turns independently
enriched team scoring statistics into a score distribution and evaluates the
exact live line carried by each candidate.  It deliberately does not consume
the sportsbook price, Kalshi, or TheOver probability, so the downstream blend
does not count those signals twice.
"""

from __future__ import annotations

from math import erf, sqrt
from typing import Any

import numpy as np
import pandas as pd


MODEL_VERSION = "score-distribution-v1"

# Conservative residual scales and home-court/field adjustments.  The output
# is shrunk toward 50% below because this first target-specific model has less
# historical calibration support than the mature market blend.
_LEAGUE_PARAMS: dict[str, dict[str, float]] = {
    "MLB": {
        "home_advantage": 0.15,
        "win_pct_margin_weight": 1.50,
        "recent_margin_weight": 0.50,
        "margin_sigma": 4.40,
        "total_sigma": 4.20,
        "reliability": 0.70,
    },
    "WNBA": {
        "home_advantage": 2.20,
        "win_pct_margin_weight": 8.00,
        "recent_margin_weight": 3.00,
        "margin_sigma": 12.50,
        "total_sigma": 15.00,
        "reliability": 0.65,
    },
}


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").astype("float64")


def _text(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series("", index=frame.index, dtype="string")
    return frame[column].fillna("").astype("string")


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(float(value) / sqrt(2.0)))


def _unscaled_scoring_stat(value: Any, league: str) -> float:
    """Undo the legacy MLB-to-NBA feature scaling when it is still present."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(number):
        return float("nan")
    if league == "MLB" and number > 20.0:
        return number / 25.0
    return number


def predict_market_probabilities(frame: pd.DataFrame) -> pd.DataFrame:
    """Return probabilities aligned to each row's exact spread/total target.

    Unsupported leagues, unresolved team statistics, and missing lines remain
    unavailable.  That is intentional: a neutral constant or market-derived
    fallback must never be presented as an independent model prediction.
    """
    result = pd.DataFrame(index=getattr(frame, "index", pd.RangeIndex(0)))
    result["ml_probability"] = pd.Series(np.nan, index=result.index, dtype="float64")
    result["ml_probability_source"] = pd.Series("", index=result.index, dtype="string")
    result["ml_target"] = pd.Series("", index=result.index, dtype="string")
    result["ml_projection"] = pd.Series(np.nan, index=result.index, dtype="float64")
    result["ml_residual_scale"] = pd.Series(np.nan, index=result.index, dtype="float64")
    result["ml_feature_quality"] = pd.Series("unavailable", index=result.index, dtype="string")

    if frame is None or frame.empty:
        return result

    league = _text(frame, "League").str.upper().str.strip()
    if league.eq("").all():
        league = _text(frame, "league").str.upper().str.strip()
    market_type = _text(frame, "market_type").str.lower().str.strip()
    spread_line = _numeric(frame, "spread_line")
    spread_line = spread_line.where(spread_line.notna(), _numeric(frame, "spread"))
    total_line = _numeric(frame, "total_line")
    total_line = total_line.where(total_line.notna(), _numeric(frame, "total"))

    eligible = pd.Series(True, index=frame.index, dtype=bool)
    if "ml_feature_eligible" in frame.columns:
        eligible &= frame["ml_feature_eligible"].fillna(False).astype(bool)
    if "stats_resolution_status" in frame.columns:
        status = _text(frame, "stats_resolution_status").str.lower()
        eligible &= status.isin({"resolved", "live", "cached"})

    h_win = _numeric(frame, "feature_home_win_pct")
    a_win = _numeric(frame, "feature_away_win_pct")
    recent_diff = _numeric(frame, "feature_diff_last5").fillna(0.0)

    for idx in frame.index:
        lg = str(league.loc[idx])
        mt = str(market_type.loc[idx])
        params = _LEAGUE_PARAMS.get(lg)
        if params is None or not bool(eligible.loc[idx]):
            continue

        h_ppg = _unscaled_scoring_stat(_numeric(frame, "feature_home_ppg").loc[idx], lg)
        a_ppg = _unscaled_scoring_stat(_numeric(frame, "feature_away_ppg").loc[idx], lg)
        h_oppg = _unscaled_scoring_stat(_numeric(frame, "feature_home_oppg").loc[idx], lg)
        a_oppg = _unscaled_scoring_stat(_numeric(frame, "feature_away_oppg").loc[idx], lg)
        scoring = np.asarray([h_ppg, a_ppg, h_oppg, a_oppg], dtype="float64")
        if not np.isfinite(scoring).all() or (scoring <= 0).any():
            continue

        expected_home = 0.5 * (h_ppg + a_oppg)
        expected_away = 0.5 * (a_ppg + h_oppg)
        win_diff = 0.0
        if np.isfinite(h_win.loc[idx]) and np.isfinite(a_win.loc[idx]):
            win_diff = float(h_win.loc[idx] - a_win.loc[idx])
        form_diff = float(recent_diff.loc[idx]) if np.isfinite(recent_diff.loc[idx]) else 0.0
        margin_adjustment = (
            params["home_advantage"]
            + params["win_pct_margin_weight"] * win_diff
            + params["recent_margin_weight"] * form_diff
        )
        expected_margin = (expected_home - expected_away) + margin_adjustment
        expected_total = expected_home + expected_away

        raw_probability: float | None = None
        target = ""
        projection = float("nan")
        residual_scale = float("nan")
        if mt == "spread_home" and np.isfinite(spread_line.loc[idx]):
            residual_scale = params["margin_sigma"]
            raw_probability = _normal_cdf((expected_margin + float(spread_line.loc[idx])) / residual_scale)
            target = "spread_cover"
            projection = expected_margin
        elif mt == "spread_away" and np.isfinite(spread_line.loc[idx]):
            residual_scale = params["margin_sigma"]
            raw_probability = _normal_cdf((float(spread_line.loc[idx]) - expected_margin) / residual_scale)
            target = "spread_cover"
            projection = -expected_margin
        elif mt == "total_over" and np.isfinite(total_line.loc[idx]):
            residual_scale = params["total_sigma"]
            raw_probability = _normal_cdf((expected_total - float(total_line.loc[idx])) / residual_scale)
            target = "total_over"
            projection = expected_total
        elif mt == "total_under" and np.isfinite(total_line.loc[idx]):
            residual_scale = params["total_sigma"]
            raw_probability = _normal_cdf((float(total_line.loc[idx]) - expected_total) / residual_scale)
            target = "total_under"
            projection = expected_total

        if raw_probability is None or not np.isfinite(raw_probability):
            continue

        probability = 0.5 + params["reliability"] * (raw_probability - 0.5)
        probability = float(np.clip(probability, 0.20, 0.80))
        result.at[idx, "ml_probability"] = probability
        result.at[idx, "ml_probability_source"] = f"{MODEL_VERSION}:{lg.lower()}"
        result.at[idx, "ml_target"] = target
        result.at[idx, "ml_projection"] = projection
        result.at[idx, "ml_residual_scale"] = residual_scale
        result.at[idx, "ml_feature_quality"] = "resolved_team_scoring_stats"

    return result

