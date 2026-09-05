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
BUCKET_STATS_MAX_AGE_DAYS = 14

# Short-horizon evidence expires much sooner than the long-horizon empirical
# overlay.  The artifact's ``recent_*`` fields describe a trailing seven-day
# window ending on ``meta.recency_anchor``.  Once that anchor is more than one
# slate day old, the window no longer represents the current regime and must
# not steer finalist selection.  Long-horizon bucket evidence may still remain
# usable under BUCKET_STATS_MAX_AGE_DAYS.
RECENT_REGIME_MAX_AGE_DAYS = 1

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

# Symmetric counterpart to the earned_directional PROMOTION below: a directional
# (over/under) bucket with enough graded picks whose smoothed realized rate sits
# clearly BELOW break-even is a proven LOSER. The bucket tilt in
# empirical_win_probability is a shrunk delta on the calibrated prob, so it is too
# weak to stop a model-overconfident pick in such a bucket from still reporting a
# positive edge (30 Jun refit: MLB:over:Disagrees realized 46% over n=56, yet a
# model-70% pick reads +8% empirical edge). That inflated edge ranks the pick high
# on the card and — for Neutral, which the parlay engine admits in its Agrees+Neutral
# fallback — lets a proven loser leak into parlays. For these buckets, floor the
# empirical probability at the bucket's own smoothed rate (so the reported edge
# reflects the realized loss) and hold the pick at Below Threshold. A sample-size
# floor keeps small-n noise from demoting, and the daily refit moves buckets in and
# out of this set automatically. The floor is LOWER than the promotion floor
# (ACTIONABLE_MIN_BUCKET_N=25) on purpose: promotion risks full Kelly on a bucket so
# it demands a sturdier sample, whereas suppression only DEMOTES (drops a stake or a
# parlay leg), whose false-positive cost is merely a skipped marginal pick — an
# asymmetric cost that warrants an asymmetric bar. At 20 it captures the current
# losers (MLB over:Disagrees n=56, under:Disagrees n=38, under:Neutral n=21) while
# leaving the coin-flip buckets (over:Neutral n=101, over:Agrees n=54, both within
# ~1pt of break-even) untouched.
PROVEN_LOSING_BUCKET_MIN_N = 20
PROVEN_LOSING_BUCKET_EDGE_MARGIN = 0.03

# Finalist selection is a prediction-time ranking decision, so realized bucket
# history is supporting evidence rather than a replacement forecast. Cap its
# influence and require a settled sample before it can move the model estimate.
# This prevents a noisy or stale bucket from reversing the candidate ordering.
EMPIRICAL_SELECTION_MIN_N = 20
EMPIRICAL_SELECTION_SHRINK_N = 100
EMPIRICAL_SELECTION_MAX_WEIGHT = 0.10

# Short-horizon regime guard for finalist ranking.  The fitted artifact carries
# a trailing seven-day record alongside the 21-day-decayed history.  A bucket
# must have at least 20 recent decisions and trail its long-horizon smoothed
# rate by five percentage points before any adjustment is allowed.  The score
# haircut is capped at ten points and never changes the exported probability or
# grants wager authority; it only stops stale confidence from winning a close
# cross-family comparison.
RECENT_REGIME_MIN_N = 20
RECENT_REGIME_MIN_RATE_DROP = 0.05
RECENT_REGIME_MAX_SCORE_PENALTY = 0.10
RECENT_REGIME_PRIOR_N = 10.0

# A league+consensus bucket can remain too small to arm the exact-bucket guard
# even when the same market direction is failing across the whole board.  That
# happened on Aug 22-28: the selected Over family went 5-17, but its losses were
# split across MLB/NFL/WNBA/NCAAF and several consensus labels, so no individual
# bucket reached RECENT_REGIME_MIN_N.  The aggregate fallback needs a smaller
# sample because it only subtracts a bounded finalist score, never grants wager
# authority.  Requiring a larger rate drop than the exact-bucket path keeps an
# ordinary small-sample wobble from moving the card.
RECENT_FAMILY_REGIME_MIN_N = 12
RECENT_FAMILY_REGIME_MIN_RATE_DROP = 0.08
RECENT_FAMILY_REGIME_MAX_SCORE_PENALTY = 0.08

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

# Earned-path promotions (a proven bucket carrying a pick whose per-pick calibrated edge
# fell below the +4% bar) still require a POSITIVE calibrated edge. The bucket's realized
# edge is meant to fill the gap left by over-suppression/down-calibration — not to stake a
# pick the calibrated number itself rates as negative-EV (19 Jun: NYY Over 9.0 promoted to
# Actionable at calibrated -0.3% on the bucket's +8.8%). Set to a negative value to allow
# any calibrated edge (the pre-19-Jun behavior).
ACTIONABLE_EARNED_MIN_CALIBRATED_EDGE = 0.0

# Directional "the OTHER side is the edge here" demotions come from a MORE-specific graded
# bucket than the coarse (league, over/under, consensus) overlay bucket — so the overlay
# must not re-promote a pick they demoted, or it stakes against its own better evidence
# (19 Jun: Miami Over 8.0 re-promoted to Actionable while the mid-line-Over gate flagged
# the Under as the edge side, n=43). Rows carrying one of these blocker stages are left as
# demoted. Empty set disables.
#
# "line_provenance" added 21 Jun: a row whose live line was rejected and recovered to the
# uploaded reference still carries the CORRUPT live odds (and sometimes a wrong-game Kalshi
# match), so its edge/EV is unreliable. The overlay was promoting such a row to Actionable
# S-Tier off a bogus Kalshi 91% (Miami Over 8.0, +265, Kalshi market titled for a different
# game). A line-provenance-flagged row must never be re-promoted, only displayed as demoted.
EDGE_NO_STAKE_BLOCKER_STAGES = frozenset(
    {"mlb_over_mid_line_no_stake", "mlb_total_neutral_no_stake", "line_provenance"}
)


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


def bucket_stats_are_fresh(
    bucket_stats: dict | None,
    *,
    now: pd.Timestamp | None = None,
    max_age_days: int = BUCKET_STATS_MAX_AGE_DAYS,
) -> bool:
    """Return whether dated bucket evidence is recent enough for decisions.

    Undated injected/test statistics remain backward compatible. Production
    artifacts fail closed once their newest graded slate is older than
    ``max_age_days`` so the finalist selector, tier overlay, recovery gate, and
    portfolio calibration cannot silently use different evidence vintages.
    """
    if not bucket_stats:
        return False
    meta = bucket_stats.get("meta") or {}
    # Freshness is about the newest graded event in the evidence, not the day a
    # script happened to rewrite the JSON.  Falling back to fitted_on preserves
    # compatibility with older artifacts and injected tests.
    evidence_date = meta.get("recency_anchor") or meta.get("fitted_on")
    if not evidence_date:
        return True
    evidence_anchor = pd.to_datetime(evidence_date, errors="coerce", utc=True)
    if pd.isna(evidence_anchor):
        return False
    current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if current.tzinfo is None:
        current = current.tz_localize("UTC")
    else:
        current = current.tz_convert("UTC")
    age_days = (current.normalize() - evidence_anchor.normalize()).days
    return -1 <= age_days <= int(max_age_days)


def recent_regime_stats_are_fresh(
    bucket_stats: dict | None,
    *,
    now: pd.Timestamp | None = None,
    max_age_days: int = RECENT_REGIME_MAX_AGE_DAYS,
) -> bool:
    """Whether the artifact's trailing-window fields may steer finalists.

    The long-horizon bucket overlay and the seven-day regression guard have
    different freshness requirements.  Keeping this decision separate lets a
    mature empirical history continue to provide a bounded probability blend
    while an expired short window fails closed to a zero score penalty.
    """

    return bucket_stats_are_fresh(
        bucket_stats,
        now=now,
        max_age_days=max_age_days,
    )


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


def _selection_consensus(df: pd.DataFrame) -> pd.Series:
    """Return the same directional Kalshi label used by the final card.

    Candidate rows are oriented to their own pick, so a Kalshi probability above
    0.52 agrees with that candidate and one below 0.48 disagrees. Recomputing the
    label here prevents stale pre-merge consensus text from routing a finalist
    through the wrong empirical bucket.
    """
    consensus = df.get(
        "consensus_agreement", pd.Series("Neutral", index=df.index)
    ).fillna("Neutral").astype(str)
    valid = consensus.isin(["Agrees", "Neutral", "Disagrees"])
    consensus = consensus.where(valid, "Neutral").copy()
    kalshi = pd.to_numeric(
        df.get("kalshi_probability", pd.Series(float("nan"), index=df.index)),
        errors="coerce",
    )
    available = kalshi.gt(0.0) & kalshi.lt(1.0)
    consensus.loc[available] = "Neutral"
    consensus.loc[available & kalshi.ge(0.52)] = "Agrees"
    consensus.loc[available & kalshi.le(0.48)] = "Disagrees"
    return consensus


def empirical_selection_probabilities(
    df: pd.DataFrame,
    bucket_stats: dict,
    calibration: list | None = None,
    prob_col: str = "calibrated_probability",
) -> pd.Series:
    """Decision probabilities for choosing the one finalist per game.

    The old pipeline selected a winner from raw candidate probabilities and only
    applied empirical bucket evidence afterward. That could demote a proven-losing
    direction but could not replace it with the better candidate. This helper uses
    the same leak-safe global calibration and conditional bucket adjustment before
    selection, including the proven-losing floor used by the final tier overlay.
    """
    base = pd.to_numeric(
        df.get(prob_col, pd.Series(float("nan"), index=df.index)),
        errors="coerce",
    )
    if df.empty or not bucket_stats:
        return base
    calibrated = apply_calibration(base, calibration) if calibration else base.copy()
    consensus = _selection_consensus(df)
    buckets = [
        bucket_key(league, market, agreement)
        for league, market, agreement in zip(
            df.get("league", pd.Series("", index=df.index)),
            df.get("market_type", pd.Series("", index=df.index)),
            consensus,
        )
    ]
    selected: list[float] = []
    for (_, row), probability, bucket in zip(df.iterrows(), calibrated, buckets):
        family = bucket.split(":")[1] if ":" in bucket else "side"
        # Put sides and totals on the same leak-safe calibration path before the
        # cross-family comparison. Previously totals received a realized-bucket
        # adjustment while sides stayed on a different global-only scale.
        eligible_family = family in ("over", "under", "side")
        target = (
            empirical_win_probability(probability, bucket, bucket_stats)
            if eligible_family
            else float(probability)
        )
        rate, n = smoothed_bucket_rate(bucket, bucket_stats)
        decimal = _decimal_odds(row)
        breakeven = 1.0 / decimal if pd.notna(decimal) and decimal > 1.0 else float("nan")
        if (
            eligible_family
            and n >= PROVEN_LOSING_BUCKET_MIN_N
            and pd.notna(breakeven)
            and rate - breakeven <= -PROVEN_LOSING_BUCKET_EDGE_MARGIN
        ):
            target = min(float(target), float(rate))

        if eligible_family and n >= EMPIRICAL_SELECTION_MIN_N:
            selection_weight = min(
                float(EMPIRICAL_SELECTION_MAX_WEIGHT),
                n / (n + float(EMPIRICAL_SELECTION_SHRINK_N)),
            )
            adjusted = float(probability) + selection_weight * (
                float(target) - float(probability)
            )
        else:
            adjusted = float(probability)
        selected.append(float(adjusted) if pd.notna(adjusted) else float("nan"))
    return pd.Series(selected, index=df.index, dtype="float64")


def smoothed_bucket_rate(bucket: str, stats: dict) -> tuple[float, int]:
    """(Laplace-smoothed realized rate, n) for the Actionable proven-bucket gate."""
    overall = float(stats["overall"]["win_rate"])
    b = stats["buckets"].get(bucket)
    if not b or int(b["n"]) <= 0:
        return overall, 0
    n = int(b["n"])
    return (int(b["wins"]) + overall * 10.0) / (n + 10.0), n


def recent_regime_bucket_summary(bucket: str, stats: dict) -> dict[str, object]:
    """Describe a material short-horizon regression for one empirical bucket."""

    long_rate, long_n = smoothed_bucket_rate(bucket, stats)
    payload = (stats.get("buckets") or {}).get(bucket) if stats else None
    recent_n = int((payload or {}).get("recent_n", 0) or 0)
    recent_wins = int((payload or {}).get("recent_wins", 0) or 0)
    raw_recent_rate = (
        float(recent_wins / recent_n) if recent_n > 0 else float("nan")
    )
    summary: dict[str, object] = {
        "bucket": bucket,
        "applied": False,
        "reason": (
            "missing_bucket_history"
            if not payload
            else "insufficient_recent_bucket_history"
        ),
        "penalty": 0.0,
        "long_n": int(long_n),
        "long_rate": float(long_rate),
        "recent_n": recent_n,
        "recent_rate": raw_recent_rate,
        "recent_smoothed_rate": raw_recent_rate,
    }
    if recent_n < RECENT_REGIME_MIN_N:
        return summary

    recent_smoothed = (
        recent_wins + float(long_rate) * RECENT_REGIME_PRIOR_N
    ) / (recent_n + RECENT_REGIME_PRIOR_N)
    rate_drop = float(long_rate - recent_smoothed)
    summary.update({
        "reason": "recent_bucket_not_materially_worse",
        "recent_smoothed_rate": float(recent_smoothed),
        "rate_drop": rate_drop,
    })
    if rate_drop < RECENT_REGIME_MIN_RATE_DROP:
        return summary

    penalty = min(RECENT_REGIME_MAX_SCORE_PENALTY, max(0.0, rate_drop))
    summary.update({
        "applied": penalty > 0.0,
        "reason": "fresh_recent_bucket_regression",
        "penalty": float(penalty),
    })
    return summary


def recent_regime_family_summary(family: str, stats: dict) -> dict[str, object]:
    """Describe a broad directional regression when exact buckets are sparse."""

    payload = (stats.get("families") or {}).get(family) if stats else None
    long_rate = float((payload or {}).get("win_rate", float("nan")))
    long_n = int((payload or {}).get("n", 0) or 0)
    recent_n = int((payload or {}).get("recent_n", 0) or 0)
    recent_wins = int((payload or {}).get("recent_wins", 0) or 0)
    raw_recent_rate = (
        float(recent_wins / recent_n) if recent_n > 0 else float("nan")
    )
    summary: dict[str, object] = {
        "bucket": f"ALL:{family}:ALL",
        "applied": False,
        "reason": (
            "missing_family_history"
            if not payload
            else "insufficient_recent_family_history"
        ),
        "penalty": 0.0,
        "long_n": long_n,
        "long_rate": long_rate,
        "recent_n": recent_n,
        "recent_rate": raw_recent_rate,
        "recent_smoothed_rate": raw_recent_rate,
    }
    if (
        not payload
        or recent_n < RECENT_FAMILY_REGIME_MIN_N
        or pd.isna(long_rate)
    ):
        return summary

    recent_smoothed = (
        recent_wins + long_rate * RECENT_REGIME_PRIOR_N
    ) / (recent_n + RECENT_REGIME_PRIOR_N)
    rate_drop = float(long_rate - recent_smoothed)
    summary.update({
        "reason": "recent_family_not_materially_worse",
        "recent_smoothed_rate": float(recent_smoothed),
        "rate_drop": rate_drop,
    })
    if rate_drop < RECENT_FAMILY_REGIME_MIN_RATE_DROP:
        return summary

    penalty = min(
        RECENT_FAMILY_REGIME_MAX_SCORE_PENALTY,
        max(0.0, rate_drop),
    )
    summary.update({
        "applied": penalty > 0.0,
        "reason": "fresh_recent_family_regression",
        "penalty": float(penalty),
    })
    return summary


def recent_regime_score_adjustments(
    df: pd.DataFrame,
    bucket_stats: dict,
) -> pd.DataFrame:
    """Return auditable finalist-score haircuts aligned to ``df``.

    Consensus is recomputed from the oriented Kalshi probability exactly as it
    is for empirical selection, so the guard cannot be routed through stale
    pre-merge labels.
    """

    columns = [
        "recent_regime_penalty_applied",
        "recent_regime_penalty_value",
        "recent_regime_penalty_reason",
        "recent_regime_bucket",
        "recent_regime_bucket_n",
        "recent_regime_bucket_win_rate",
        "recent_regime_long_win_rate",
    ]
    if df is None or df.empty:
        return pd.DataFrame(index=getattr(df, "index", None), columns=columns)

    consensus = _selection_consensus(df)
    buckets = [
        bucket_key(league, market, agreement)
        for league, market, agreement in zip(
            df.get("league", pd.Series("", index=df.index)),
            df.get("market_type", pd.Series("", index=df.index)),
            consensus,
        )
    ]
    rows = []
    for bucket in buckets:
        summary = recent_regime_bucket_summary(bucket, bucket_stats)
        # Prefer sufficiently sampled bucket evidence.  Fall back to the
        # cross-league market family only when the exact bucket cannot reach the
        # recent sample floor; a healthy, well-sampled bucket must not be
        # overridden by a coarser aggregate.
        if int(summary["recent_n"]) < RECENT_REGIME_MIN_N:
            family = bucket.split(":")[1] if ":" in bucket else "side"
            family_summary = recent_regime_family_summary(family, bucket_stats)
            if bool(family_summary["applied"]):
                summary = family_summary
        rows.append({
            "recent_regime_penalty_applied": bool(summary["applied"]),
            "recent_regime_penalty_value": float(summary["penalty"]),
            "recent_regime_penalty_reason": str(summary["reason"]),
            "recent_regime_bucket": str(summary["bucket"]),
            "recent_regime_bucket_n": int(summary["recent_n"]),
            "recent_regime_bucket_win_rate": summary["recent_rate"],
            "recent_regime_long_win_rate": float(summary["long_rate"]),
        })
    return pd.DataFrame(rows, index=df.index, columns=columns)


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
        # A more-specific directional no-stake demotion ("the other side is the edge")
        # outranks the coarse overlay bucket — never re-promote what it demoted.
        prior_blocker = (
            str(out.at[idx, "status_blocker_stage"])
            if "status_blocker_stage" in out.columns
            else ""
        )
        if prior_blocker in EDGE_NO_STAKE_BLOCKER_STAGES:
            continue
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
        family = bucket.split(":")[1] if ":" in bucket else "side"
        is_directional = family in ("over", "under")

        # Proven-losing-bucket suppression — symmetric to the earned_directional
        # promotion below. When a directional bucket has enough graded picks and its
        # smoothed realized rate is clearly below break-even, the mild probability tilt
        # leaves the pick with a spuriously positive edge. Floor the empirical
        # probability at the bucket's own rate so the edge reflects the realized loss,
        # and hold the pick at Below Threshold (unstaked; excluded from parlays).
        if (
            is_directional
            and n >= PROVEN_LOSING_BUCKET_MIN_N
            and proven_edge <= -PROVEN_LOSING_BUCKET_EDGE_MARGIN
        ):
            honest_p = min(float(out.at[idx, "empirical_win_probability"]), float(rate))
            out.at[idx, "empirical_win_probability"] = honest_p
            out.at[idx, "empirical_edge"] = honest_p - breakeven
            if str(out.at[idx, "Pick_Status"]) != "Below Threshold":
                out.at[idx, "Pick_Status"] = "Below Threshold"
                out.at[idx, "Status_Reason"] = (
                    f"Below Threshold (empirical): proven-losing bucket {bucket} "
                    f"realized {rate:.0%} (n={n}), {proven_edge:+.1%} vs break-even; "
                    f"model edge suppressed to the bucket's own rate"
                )
                if "status_blocker_stage" in out.columns:
                    out.at[idx, "status_blocker_stage"] = "empirical_proven_losing_bucket"
            continue

        # Pre-calibration probability (out[prob_col] is untouched; calibration was only
        # applied to the local p_cal used for the bucket tilt). Used as an outlier guard.
        raw_prob = pd.to_numeric(out.at[idx, prob_col], errors="coerce")
        calibrated_actionable = edge >= ACTIONABLE_MIN_EMPIRICAL_EDGE
        # Earned paths — a PROVEN bucket lifts a pick to Actionable on the strength of
        # the bucket's realized edge, filling the gap the over-prob shrink / market
        # debias / over-calibration open up:
        #   * DIRECTIONAL totals (over AND under): the global calibration maps the
        #     model's predicted prob DOWN regardless of side, so the per-pick calibrated
        #     edge is unreliable for both. Trust the proven bucket's realized edge,
        #     guarded only by the pick's PRE-calibration probability still beating
        #     break-even (so a genuine negative-signal pick is not staked on the bucket's
        #     back). Until the 20 Jun refit this earned path was over-only; the refreshed
        #     global isotonic suppresses proven under:Agrees picks (61%, n=61) the same
        #     way it does overs, so confining the escape hatch to overs was unjustified.
        #     Over volume stays bounded downstream by the MLB total-over concentration cap.
        #   * other families (side): keep per-pick discrimination — the pick must itself
        #     clear the High Variance edge bar, not be a coin flip (tiny graded samples).
        earned_directional = (
            is_directional
            and proven_edge >= ACTIONABLE_MIN_EMPIRICAL_EDGE
            and edge > ACTIONABLE_EARNED_MIN_CALIBRATED_EDGE  # never stake a calibrated-negative pick
            and pd.notna(raw_prob)
            and float(raw_prob) >= breakeven
        )
        earned_other = (
            not is_directional
            and proven_edge >= ACTIONABLE_MIN_EMPIRICAL_EDGE
            and edge > ACTIONABLE_EARNED_MIN_CALIBRATED_EDGE
            and edge >= HIGH_VARIANCE_MIN_EMPIRICAL_EDGE
        )
        if proven_bucket and (calibrated_actionable or earned_directional or earned_other):
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
        # The low-line-over guardrail demotes on LINE-SPECIFIC graded evidence
        # (sub-8.0 Neutral overs ~45%) that the coarse (league, direction,
        # consensus) bucket cannot see. The overlay may demote such a row further
        # (the proven-losing suppression above already can), but must never
        # promote it back above the guardrail's tier on the coarse bucket's
        # strength — more-specific evidence outranks the aggregate.
        if prior_blocker == "low_line_over_guardrail":
            _rank = {"Below Threshold": 0, "High Variance/Speculative": 1, "Actionable": 2}
            if _rank.get(status, 0) > _rank.get(str(out.at[idx, "Pick_Status"]), 0):
                continue

        if str(out.at[idx, "Pick_Status"]) != status:
            out.at[idx, "Pick_Status"] = status
            out.at[idx, "Status_Reason"] = reason
            if "status_blocker_stage" in out.columns:
                out.at[idx, "status_blocker_stage"] = "empirical_tier_overlay"

    return out
