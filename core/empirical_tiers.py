"""Empirical tier overlay: reassign Actionable / High Variance / Below Threshold
from REALIZED bucket performance + calibrated probability, instead of model-vs-
market EV/edge.

Why (Jun 5-10 graded recaps): the EV/edge promotion gates select for maximum
model-vs-market disagreement, which under a miscalibrated model is adverse
selection — the staked tiers went Actionable 1-4 and HV/Spec 3-11 (~21%) while
Below Threshold went 29-20 (59%). On 10 Jun the slate hit 10-5 and still lost
money because the one big-stake Actionable pick lost while the ten winners were
staked at Below Threshold size. The tiers must express realized accuracy.

The overlay only re-tiers picks already in a viable status (Actionable, HV,
Below Threshold). Safety statuses (No Play, Missing Line — line integrity,
suspicious data, divergence floors) are never overridden: those guards filter
bad DATA; this overlay ranks good data.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from core.probability_calibration import apply_calibration

DEFAULT_BUCKET_STATS_PATH = Path("data/calibration/bucket_stats.json")

VIABLE_STATUSES = {"Actionable", "High Variance/Speculative", "Below Threshold"}

# Shrinkage: a bucket's delta from the overall win rate is weighted by
# n/(n+SHRINK_N), so a 15-pick bucket moves a pick ~23% of its raw delta and a
# 60-pick bucket ~55%. Guards against tiering off small-sample bucket noise.
BUCKET_SHRINK_N = 50

# Tier thresholds on empirical edge = empirical_win_probability - break-even
# at the pick's own odds. Set vs the -110 vig (~2.4% house edge): Actionable
# demands roughly double the vig in real edge; HV demands beating the vig.
ACTIONABLE_MIN_EMPIRICAL_EDGE = 0.04
HIGH_VARIANCE_MIN_EMPIRICAL_EDGE = 0.015

# Actionable additionally requires the pick's bucket to be PROVEN: enough graded
# picks and a smoothed realized rate at/above break-even-plus. A great calibrated
# number in a bucket that has bled (e.g. Neutral-consensus Overs, 48.2% n=56)
# caps at High Variance until the bucket itself earns trust.
# Raised 15 -> 25 on 13 Jun: the MLB:under:Disagrees bucket promoted to Actionable at
# n=16 (60%) and then went 1-3/1-4 on 12 Jun. 16 samples is too thin to trust a 60%
# rate (95% CI roughly +/- 24pts), and it tilted enough Unders into Actionable to form
# the correlated block the concentration cap now also guards. Require a sturdier sample
# before a bucket can carry full Kelly.
ACTIONABLE_MIN_BUCKET_N = 25
ACTIONABLE_MIN_BUCKET_RATE = 0.55

# Actionable ALSO requires market AGREEMENT. Across the graded history the only
# slice that clears the -110 break-even (52.4%) by a real margin is the buckets
# where Kalshi agrees with the model's direction: MLB over:Agrees 61% (n=41) and
# under:Agrees 61% (n=61), versus 51-55% for Neutral/Disagrees. Two independent
# estimates agreeing is a principled reason for higher confidence, not just a
# fitted artifact, so full-Kelly staking is confined to agreement buckets and the
# coin-flip buckets are refused. Honesty note: the Agrees-vs-rest gap is
# suggestive, not statistically airtight (z~1.1 at n=232), which is why agreement
# is REQUIRED to stake but stakes stay fractional (the Kelly fraction is small) and
# non-agreement picks can still surface as High Variance/Speculative — they just
# cannot carry a stake. Empty set disables the requirement.
ACTIONABLE_PROVEN_CONSENSUS = {"Agrees"}


def bucket_key(league: str, market_type: str, consensus: str) -> str:
    """Coarse empirical bucket: (league, over/under/side, consensus).
    'No Kalshi' folds into Neutral — absence of a market is not disagreement."""
    mt = str(market_type or "").strip().lower()
    if "over" in mt:
        family = "over"
    elif "under" in mt:
        family = "under"
    else:
        family = "side"
    cons = str(consensus or "").strip()
    if cons not in ("Agrees", "Neutral", "Disagrees"):
        cons = "Neutral"
    return f"{str(league or '').strip().upper()}:{family}:{cons}"


def load_bucket_stats(path: Path | str = DEFAULT_BUCKET_STATS_PATH) -> dict | None:
    """Load fitted bucket stats; None when absent/unreadable so the pipeline
    degrades gracefully to the existing tier assignment."""
    try:
        payload = json.loads(Path(path).read_text())
        return payload if payload.get("buckets") else None
    except (OSError, ValueError):
        return None


def empirical_win_probability(
    p_calibrated: float, bucket: str, stats: dict, shrink_n: int = BUCKET_SHRINK_N
) -> float:
    """Tilt the calibrated probability by the bucket's realized delta from the
    overall win rate, shrunk by sample size. The calibration table already fixes
    the GLOBAL predicted->realized mapping; the bucket adds the conditional
    (league, direction, consensus) information the global fit averages away."""
    if pd.isna(p_calibrated):
        return p_calibrated
    overall = float(stats["overall"]["win_rate"])
    b = stats["buckets"].get(bucket)
    if not b or int(b["n"]) <= 0:
        return float(p_calibrated)
    n = int(b["n"])
    # Laplace-smoothed bucket rate, then shrink its delta toward overall by n
    smoothed = (int(b["wins"]) + overall * 10.0) / (n + 10.0)
    weight = n / (n + float(shrink_n))
    return float(min(0.95, max(0.05, float(p_calibrated) + weight * (smoothed - overall))))


def smoothed_bucket_rate(bucket: str, stats: dict) -> tuple[float, int]:
    """(Laplace-smoothed realized rate, n) for the Actionable proven-bucket gate."""
    overall = float(stats["overall"]["win_rate"])
    b = stats["buckets"].get(bucket)
    if not b or int(b["n"]) <= 0:
        return overall, 0
    n = int(b["n"])
    return (int(b["wins"]) + overall * 10.0) / (n + 10.0), n


def _decimal_odds(row: pd.Series) -> float:
    dec = pd.to_numeric(row.get("decimal_odds"), errors="coerce")
    if pd.notna(dec) and float(dec) > 1.0:
        return float(dec)
    amer = pd.to_numeric(row.get("odds_american"), errors="coerce")
    if pd.isna(amer) or amer == 0:
        return float("nan")
    return 1 + amer / 100.0 if amer > 0 else 1 + 100.0 / abs(amer)


def assign_empirical_tiers(
    df: pd.DataFrame,
    bucket_stats: dict,
    calibration: list | None,
    prob_col: str = "effective_win_probability",
) -> pd.DataFrame:
    """Return a copy of ``df`` with empirical columns and re-tiered Pick_Status.

    Adds: empirical_win_probability, empirical_edge, empirical_bucket.
    Re-tiers only rows whose current Pick_Status is viable; never promotes or
    demotes safety statuses. Rows missing odds or probability are left as-is.
    """
    out = df.copy()
    if out.empty or prob_col not in out.columns or not bucket_stats:
        return out

    p_cal = pd.to_numeric(out[prob_col], errors="coerce")
    if calibration:
        p_cal = apply_calibration(p_cal, calibration)

    buckets = [
        bucket_key(l, m, c)
        for l, m, c in zip(
            out.get("league", pd.Series("", index=out.index)),
            out.get("market_type", pd.Series("", index=out.index)),
            out.get("consensus_agreement", pd.Series("", index=out.index)),
        )
    ]
    out["empirical_bucket"] = buckets
    out["empirical_win_probability"] = [
        empirical_win_probability(p, b, bucket_stats) for p, b in zip(p_cal, buckets)
    ]

    dec = out.apply(_decimal_odds, axis=1)
    breakeven = 1.0 / dec.where(dec > 1.0)
    out["empirical_edge"] = out["empirical_win_probability"] - breakeven

    viable = out.get("Pick_Status", pd.Series("", index=out.index)).astype(str).isin(VIABLE_STATUSES)
    gradeable = viable & out["empirical_edge"].notna()

    for idx in out.index[gradeable]:
        edge = float(out.at[idx, "empirical_edge"])
        bucket = str(out.at[idx, "empirical_bucket"])
        rate, n = smoothed_bucket_rate(bucket, bucket_stats)
        consensus = bucket.rsplit(":", 1)[-1]
        agreement_ok = (not ACTIONABLE_PROVEN_CONSENSUS) or (consensus in ACTIONABLE_PROVEN_CONSENSUS)
        proven_bucket = (
            n >= ACTIONABLE_MIN_BUCKET_N and rate >= ACTIONABLE_MIN_BUCKET_RATE and agreement_ok
        )
        # The bucket's OWN realized edge (smoothed rate vs break-even at this pick's
        # odds). For a PROVEN bucket this earned edge can carry the pick to Actionable
        # even when the per-pick calibrated edge is dragged below the bar by the
        # over-prob shrink / market debias — as long as the pick is not itself a
        # negative-edge outlier. Without this, proven Agrees-over buckets could never
        # promote, because the suppressed calibrated prob always fell short.
        breakeven = float(out.at[idx, "empirical_win_probability"]) - edge
        proven_edge = rate - breakeven
        calibrated_actionable = edge >= ACTIONABLE_MIN_EMPIRICAL_EDGE
        # Earned path: a proven bucket lifts a pick that already clears the High
        # Variance edge bar (so it is a viable edge on its own, not a coin flip) up
        # to Actionable on the strength of the bucket's realized edge. This fills the
        # gap the MLB over-prob shrink / market debias open up, where a genuinely
        # strong over is suppressed to a calibrated edge between HV and Actionable.
        earned_actionable = (
            proven_edge >= ACTIONABLE_MIN_EMPIRICAL_EDGE
            and edge >= HIGH_VARIANCE_MIN_EMPIRICAL_EDGE
        )
        if proven_bucket and (calibrated_actionable or earned_actionable):
            status = "Actionable"
            basis = (
                f"edge {edge:+.1%} vs break-even at own odds"
                if calibrated_actionable
                else f"proven-bucket realized edge {proven_edge:+.1%} (calibrated {edge:+.1%})"
            )
            reason = (
                f"Actionable (empirical): {basis}; "
                f"bucket {bucket} hit {rate:.0%} (n={n}) with market agreement"
            )
        elif edge >= HIGH_VARIANCE_MIN_EMPIRICAL_EDGE:
            status = "High Variance/Speculative"
            reason = (
                f"High Variance (empirical): edge {edge:+.1%}; bucket "
                f"{out.at[idx, 'empirical_bucket']} {rate:.0%} (n={n})"
            )
        else:
            status = "Below Threshold"
            reason = (
                f"Below Threshold (empirical): edge {edge:+.1%} does not beat the "
                f"vig at these odds"
            )
        if str(out.at[idx, "Pick_Status"]) != status:
            out.at[idx, "Pick_Status"] = status
            out.at[idx, "Status_Reason"] = reason
            if "status_blocker_stage" in out.columns:
                out.at[idx, "status_blocker_stage"] = "empirical_tier_overlay"

    return out
