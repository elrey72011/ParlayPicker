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
# CONFIDENCE MAGNITUDE |p - 0.5| meets this share with at least this many real
# reads. Magnitude (not signed P(over)) is the right key because TheOver's
# degeneracy is a constant model hit-rate: when the same ~0.9 hit-rate is applied
# to every game, the over/under flip splits the total_over rows' P(over) into 0.9
# and 0.1, which hides as two signed clusters (13 Jun was caught only because the
# raw read 0.692 was un-flipped; 14 Jun's 0.9/0.1 split slipped through). Folding
# to |p - 0.5| collapses 0.9 and 0.1 to the same 0.4, exposing the cluster.
THEOVER_MIN_REAL_READS = 5
THEOVER_MAX_CLUSTER_SHARE = 0.60
THEOVER_CLUSTER_ROUND = 3  # round magnitudes to 3 dp before counting

# A totals slate is flagged one-sided when the dominant direction's share meets
# this threshold across at least this many totals games. 13 Jun: 14/14 = 1.00.
# Set high so genuinely lopsided-but-real slates (e.g. a hot 9/12 over night) do
# not trip; only a near-unanimous card (a data/orientation fault) is caught.
DIRECTION_IMBALANCE_MIN_GAMES = 6
DIRECTION_IMBALANCE_SHARE = 0.90

# Column names a TheOver upload may use for its win-probability field, after
# lowercasing and collapsing non-alphanumerics to underscores.
_THEOVER_PROB_COLS = ("winprobability", "win_probability", "probability", "win_prob")


def theover_upload_warning(raw_df) -> str | None:
    """User-facing content check for a freshly-uploaded TheOver CSV.

    Runs on the RAW upload (before numeric coercion) so it can see the two
    corruptions behind the 13 Jun all-Over card that the downstream blend guard
    cannot: a column-shift dropping non-numeric text into the probability field
    (coercion would silently turn it into NaN), and the same probability value
    repeated across many games. Returns a warning string to surface at upload
    time, or None when the file looks clean. Never raises — a malformed frame
    just yields None so the upload still proceeds (the blend/card guards remain
    the safety net).
    """
    try:
        if raw_df is None or len(raw_df) == 0:
            return None
        cols = {
            str(c).strip().lower().replace(" ", "_"): c for c in raw_df.columns
        }
        prob_col = next((cols[k] for k in _THEOVER_PROB_COLS if k in cols), None)
        if prob_col is None:
            return None

        raw = raw_df[prob_col].astype("string")
        nonblank = raw.str.strip().fillna("").ne("")
        numeric = pd.to_numeric(raw, errors="coerce")
        contaminated = int((nonblank & numeric.isna()).sum())

        msgs: list[str] = []
        if contaminated >= 1:
            msgs.append(
                f"{contaminated} non-numeric value(s) in the win-probability "
                f"column (likely a column-shift in the export)"
            )
        degraded, reason = theover_feed_degraded(numeric)
        if degraded:
            # reason already describes the identical-value clustering
            msgs.append(reason.split(": ", 1)[-1])
        if not msgs:
            return None
        return "TheOver upload looks corrupt — " + "; ".join(msgs) + (
            ". Proceeding with TheOver dropped from the blend for affected "
            "markets; re-export and re-upload a clean file if possible."
        )
    except Exception:
        return None


def theover_feed_degraded(
    over_probs,
    *,
    min_real_reads: int = THEOVER_MIN_REAL_READS,
    max_cluster_share: float = THEOVER_MAX_CLUSTER_SHARE,
) -> tuple[bool, str | None]:
    """True when the per-game TheOver reads are degenerately clustered.

    ``over_probs`` is the slate's TheOver P(Over) values, one per totals game
    (pass the ``total_over`` rows' ``theover_probability``). Non-numeric values
    (e.g. a column-shifted text label) coerce to NaN and are ignored; values at
    ~0.50 are "no read" and excluded. Clustering is judged on the CONFIDENCE
    MAGNITUDE ``|p - 0.5|`` rather than the signed P(over), so a constant model
    hit-rate is caught regardless of the over/under flip (0.9 and 0.1 both fold to
    0.4). If the most common rounded magnitude's share >= ``max_cluster_share``
    (with >= ``min_real_reads`` real reads), the feed is treated as degraded.
    """
    s = pd.to_numeric(pd.Series(list(over_probs)), errors="coerce").dropna()
    mag = (s - 0.5).abs()
    real_mag = mag[mag > NO_READ_TOL]
    n = int(real_mag.shape[0])
    if n < int(min_real_reads):
        return (False, None)
    counts = real_mag.round(THEOVER_CLUSTER_ROUND).value_counts()
    top_mag = float(counts.index[0])
    top_share = float(counts.iloc[0]) / n
    if top_share >= float(max_cluster_share):
        return (
            True,
            f"theover_feed_degraded: {counts.iloc[0]}/{n} real reads share the "
            f"identical confidence {top_mag:.3f} from 0.5 "
            f"(P(over)≈{0.5 + top_mag:.3f}/{0.5 - top_mag:.3f}, share {top_share:.0%}) "
            f"— feed not game-specific; down-weighted in blend for this slate",
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


# Market-anchored over-bias correction (MLB totals). The de-vig sportsbook line is
# the sharpest, least-biased P(over) estimate; for an efficient market it sits ~0.5.
MLB_DEBIAS_MIN_GAMES = 6
MLB_DEBIAS_MAX_SHIFT = 0.15


def market_anchored_over_bias(
    model_over,
    market_over,
    *,
    min_games: int = MLB_DEBIAS_MIN_GAMES,
    max_shift: float = MLB_DEBIAS_MAX_SHIFT,
) -> float:
    """Slate-level systematic over-bias of the model vs the de-vig market.

    ``model_over`` / ``market_over`` are the slate's per-game P(over): the blended
    model probability and the de-vig sportsbook probability (pass the total_over
    rows, one per game). When the model sits systematically above the market across
    the slate (13 Jun: model mean ~0.56 vs market ~0.48 → a 14/0 all-Over card),
    that gap is bias, not edge — graded MLB overs hit ~52%, no real edge over the
    market. Returns the mean (model − market) gap to SUBTRACT from each game's
    model P(over) (and add to P(under)), clamped to ±``max_shift``.

    Only the slate-MEAN gap is removed, so per-game RELATIVE leans are preserved
    (the games the model likes most still go over), and because the anchor is the
    market — not 0.5 — a genuine market-wide over lean (hot slate) is preserved.
    Returns 0.0 when there are too few games to estimate a stable bias.
    """
    m = pd.to_numeric(pd.Series(list(model_over)), errors="coerce")
    k = pd.to_numeric(pd.Series(list(market_over)), errors="coerce")
    mask = m.notna() & k.notna()
    if int(mask.sum()) < int(min_games):
        return 0.0
    bias = float((m[mask] - k[mask]).mean())
    return max(-float(max_shift), min(float(max_shift), bias))


# Spread orientation guard. The team favored to win outright (the moneyline
# favorite) is the team that lays points (negative spread). When a live feed
# delivers a flipped home/away spread, the favorite's line and price get attached
# to the wrong team — 14 Jun: Texas was shown "Texas -1.5" at +158 (a favorite
# run line) while Texas was actually the +1.5 underdog. The spread/price pairing is
# internally self-consistent, so only an independent reference — the moneyline —
# exposes that the WRONG team is the favorite.
SPREAD_ORIENTATION_MIN_ML_GAP = 0.08


def _american_implied_prob(odds) -> float | None:
    """Implied win probability of an American moneyline price (vig included).
    None when the value is missing or zero."""
    o = pd.to_numeric(pd.Series([odds]), errors="coerce").iloc[0]
    if pd.isna(o) or float(o) == 0.0:
        return None
    o = float(o)
    return (-o) / (-o + 100.0) if o < 0 else 100.0 / (o + 100.0)


def spread_moneyline_orientation_fault(
    market_type,
    spread_line,
    home_ml_price,
    away_ml_price,
    *,
    min_ml_gap: float = SPREAD_ORIENTATION_MIN_ML_GAP,
) -> tuple[bool, str | None]:
    """True when a spread row's favorite contradicts the moneyline favorite.

    ``market_type`` is spread_home/spread_away; ``spread_line`` is the line oriented
    to the pick team (negative = pick is laying points = spread favorite); the
    moneyline prices are the game's American h2h prices for the home and away teams.

    The spread favorite must be the moneyline favorite. A disagreement means the
    home/away spread was delivered flipped. Conservative: returns (False, None)
    unless both moneyline prices are present and clearly separated (implied-prob gap
    >= ``min_ml_gap``), so genuine pick'em games never trip it.
    """
    mt = str(market_type or "").strip().lower()
    if mt not in ("spread_home", "spread_away"):
        return (False, None)
    line = pd.to_numeric(pd.Series([spread_line]), errors="coerce").iloc[0]
    if pd.isna(line) or float(line) == 0.0:
        return (False, None)
    ph = _american_implied_prob(home_ml_price)
    pa = _american_implied_prob(away_ml_price)
    if ph is None or pa is None or abs(ph - pa) < float(min_ml_gap):
        return (False, None)
    pick_is_home = mt == "spread_home"
    pick_ml_prob = ph if pick_is_home else pa
    opp_ml_prob = pa if pick_is_home else ph
    spread_says_favorite = float(line) < 0.0
    ml_says_favorite = pick_ml_prob > opp_ml_prob
    if spread_says_favorite != ml_says_favorite:
        fav = "home" if ph > pa else "away"
        return (
            True,
            f"spread_orientation_fault: {mt} line {float(line):+.1f} makes the pick the "
            f"spread {'favorite' if spread_says_favorite else 'underdog'}, but the moneyline "
            f"favors the {fav} team — flipped home/away spread from the live feed",
        )
    return (False, None)
