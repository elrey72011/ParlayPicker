"""Slate-level data-quality guards.

Two independent, pure detectors used by the pipeline to catch the failure mode
seen on 13 Jun 2026, where a corrupted TheOver feed (five different games sharing
the identical P(Over)=0.692, plus a column-shift putting a text label in the
numeric probability field) biased direction selection into a 14-of-15 all-Over
card with one staked Actionable pick.

1. ``theover_feed_degraded`` — detects a degenerate per-game TheOver signal
   (too many identical reads). When degraded, the pipeline nulls TheOver for the
   slate so the blend falls back to market + Kalshi + ML (the documented behavior
   for a missing TheOver read) instead of letting a constant amplify one side.

2. ``totals_direction_share`` — measures how one-sided a slate's totals picks
   are. A near-unanimous direction across many games is a near-certain data or
   orientation fault (genuine independent reads are mixed), so the pipeline
   suspends big-Kelly staking and warns rather than shipping the card.

Both are pure and unit-tested; the pipeline wiring is a thin call plus a guarded
DataFrame mutation.
"""
from __future__ import annotations

import pandas as pd

# A TheOver read within this tolerance of 0.50 is "no read" (WinProbSource=
# default_0.5) and is excluded from clustering analysis — it is already dropped
# from the blend upstream.
NO_READ_TOL = 1e-6

# TheOver is flagged degraded when, among real reads, the single most common
# (rounded) value's share meets this threshold and there are at least this many
# real reads. 13 Jun: 5 of 7 real reads were identical 0.692 (share 0.71).
THEOVER_MIN_REAL_READS = 5
THEOVER_MAX_CLUSTER_SHARE = 0.60
THEOVER_CLUSTER_ROUND = 3  # round reads to 3 dp before counting identical values

# A totals slate is flagged one-sided when the dominant direction's share meets
# this threshold across at least this many totals games. 13 Jun: 14/14 = 1.00.
# Set high so genuinely lopsided-but-real slates (e.g. a hot 9/12 over night) do
# not trip; only a near-unanimous card (a data/orientation fault) is caught.
DIRECTION_IMBALANCE_MIN_GAMES = 6
DIRECTION_IMBALANCE_SHARE = 0.90


def theover_feed_degraded(
    over_probs,
    *,
    min_real_reads: int = THEOVER_MIN_REAL_READS,
    max_cluster_share: float = THEOVER_MAX_CLUSTER_SHARE,
) -> tuple[bool, str | None]:
    """True when the per-game TheOver P(Over) reads are degenerately clustered.

    ``over_probs`` is the slate's TheOver P(Over) values, one per totals game
    (pass the ``total_over`` rows' ``theover_probability``). Non-numeric values
    (e.g. a column-shifted text label) coerce to NaN and are ignored; values at
    ~0.50 are "no read" and excluded. Among the remaining real reads, if the most
    common rounded value's share >= ``max_cluster_share`` (with >= ``min_real_reads``
    reads), the feed is treated as degraded.
    """
    s = pd.to_numeric(pd.Series(list(over_probs)), errors="coerce").dropna()
    real = s[(s - 0.5).abs() > NO_READ_TOL]
    n = int(real.shape[0])
    if n < int(min_real_reads):
        return (False, None)
    counts = real.round(THEOVER_CLUSTER_ROUND).value_counts()
    top_val = counts.index[0]
    top_share = float(counts.iloc[0]) / n
    if top_share >= float(max_cluster_share):
        return (
            True,
            f"theover_feed_degraded: {counts.iloc[0]}/{n} real reads share the "
            f"identical value {top_val:.3f} (share {top_share:.0%}) — feed not "
            f"game-specific; dropped from blend for this slate",
        )
    return (False, None)


def totals_direction_share(market_types) -> tuple[str | None, float, int]:
    """(dominant_direction, share, n_totals) over total_over/total_under rows.

    ``market_types`` is the slate's ``market_type`` column (one row per pick).
    Returns the dominant totals direction ('over'/'under'), its share of all
    totals picks, and the number of totals picks. ('', 0.0, 0) when none.
    """
    mt = pd.Series(list(market_types)).astype(str).str.lower()
    overs = int(mt.eq("total_over").sum())
    unders = int(mt.eq("total_under").sum())
    n = overs + unders
    if n == 0:
        return ("", 0.0, 0)
    if overs >= unders:
        return ("over", overs / n, n)
    return ("under", unders / n, n)


def slate_direction_imbalanced(
    market_types,
    *,
    min_games: int = DIRECTION_IMBALANCE_MIN_GAMES,
    share_threshold: float = DIRECTION_IMBALANCE_SHARE,
) -> tuple[bool, str | None]:
    """True when totals picks are near-unanimously one direction across enough
    games to indicate a data/orientation fault rather than a real read."""
    direction, share, n = totals_direction_share(market_types)
    if n >= int(min_games) and share >= float(share_threshold):
        return (
            True,
            f"slate_direction_imbalance: {share:.0%} of {n} totals are "
            f"{direction} — near-unanimous direction indicates a TheOver/"
            f"orientation data fault; big-Kelly staking suspended",
        )
    return (False, None)
