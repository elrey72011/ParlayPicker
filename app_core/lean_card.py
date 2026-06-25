"""All-games lean view: the model's read on EVERY game, tiered honestly.

The games card stakes only proven +EV picks, so on an efficient slate it looks empty even
though the model has a directional read on every game. This view re-presents the SAME card
(it adds no staking and changes no guard) so a bettor who wants action across the board can
see, per game: the model's side, its confidence, and an honest tier —

  * BET   — the pick the system would actually stake (Actionable: a real, priced edge).
  * LEAN  — the model has a positive-EV side but below the stake bar, and it is not fading
            Kalshi. A read worth knowing; NOT a proven +EV bet. Bet at your own risk.
  * AVOID — negative EV at the price, or the model is fading consensus (Disagrees). The
            board the math says to stay off.

The tiers are the point: they let the user bet the whole slate WITH the model's read and a
straight risk label, without the tool pretending coin-flips are +EV.
"""
from __future__ import annotations

import pandas as pd


def _first_col(df: pd.DataFrame, *names: str):
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series([None] * len(df), index=df.index)


def classify_lean_tier(status: object, eff_ev: object, consensus: object,
                       *, calibrated_win: object = None, break_even: object = None) -> str:
    """BET / LEAN / AVOID for one row (see module docstring).

    When ``calibrated_win`` and ``break_even`` are supplied, a would-be LEAN is demoted to
    AVOID if its CALIBRATED win probability fails to beat the break-even price — the model's
    raw EV is overconfident in the 0.50-0.55 band (327 graded picks: predicted .53, realized
    .43), so a positive *raw* EV there is usually a negative *calibrated* EV.
    """
    if str(status).strip() == "Actionable":
        return "BET"
    ev = pd.to_numeric(pd.Series([eff_ev]), errors="coerce").iloc[0]
    cons = str(consensus or "").strip()
    if not (pd.notna(ev) and ev > 0 and cons != "Disagrees"):
        return "AVOID"
    if calibrated_win is not None and break_even is not None:
        cw = pd.to_numeric(pd.Series([calibrated_win]), errors="coerce").iloc[0]
        be = pd.to_numeric(pd.Series([break_even]), errors="coerce").iloc[0]
        if pd.notna(cw) and pd.notna(be) and cw <= be:
            return "AVOID"
    return "LEAN"


_TIER_ORDER = {"BET": 0, "LEAN": 1, "AVOID": 2}


def _american_breakeven(odds: object):
    o = pd.to_numeric(pd.Series([odds]), errors="coerce").iloc[0]
    if pd.isna(o):
        return None
    o = float(o)
    return abs(o) / (abs(o) + 100.0) if o < 0 else 100.0 / (o + 100.0)


_UNSET = object()


def build_all_games_lean_card(best_picks_df: pd.DataFrame, *, calibration: object = _UNSET,
                              bucket_stats: object = _UNSET) -> pd.DataFrame:
    """Derive the all-games lean card from the games best-picks frame.

    Pure and side-effect-free: reads the existing per-game pick/probability/EV columns,
    assigns a tier, and returns a compact view RANKED BY EMPIRICAL EDGE (bucket-realized),
    not model EV. Tolerant of the pre- and post-export column names (home_team/Home, etc.).

    The LEAN tier is gated by bucket-conditional calibration: a pick stays LEAN only if its
    calibrated win beats break-even. ``calibration`` / ``bucket_stats`` default to the fitted
    tables on disk; pass ``None`` to disable (raw behavior) or explicit values to inject
    (tests).
    """
    if best_picks_df is None or best_picks_df.empty:
        return pd.DataFrame()

    if bucket_stats is _UNSET:
        try:
            from core.empirical_tiers import load_bucket_stats
            bucket_stats = load_bucket_stats()
        except Exception:
            bucket_stats = None

    if calibration is _UNSET:
        try:
            from core.probability_calibration import load_calibration
            calibration = load_calibration()
        except Exception:
            calibration = None

    df = best_picks_df
    home = _first_col(df, "Home", "home_team").astype(str)
    away = _first_col(df, "Away", "away_team").astype(str)
    status = _first_col(df, "Pick_Status")
    eff_ev = _first_col(df, "effective_expected_value", "expected_value")
    consensus = _first_col(df, "consensus_agreement")
    win = pd.to_numeric(_first_col(df, "effective_win_probability", "WinProbability"), errors="coerce")
    edge = _first_col(df, "effective_edge", "edge")
    odds = _first_col(df, "odds_american")
    kelly = pd.to_numeric(_first_col(df, "Kelly_Bet_Size"), errors="coerce").fillna(0.0)

    # Calibrated win probability + break-even, for the LEAN gate and transparency. Bucket-
    # conditional (global curve + per-bucket realized tilt) so a proven bucket (e.g.
    # under:Agrees ~61%) isn't crushed below break-even by the pooled curve — same number the
    # staking gate uses, so the view and the card agree.
    if calibration:
        try:
            from core.probability_calibration import apply_bucket_calibration
            from core.empirical_tiers import bucket_key
            league = _first_col(df, "league", "League")
            market_type = _first_col(df, "market_type")
            buckets = [bucket_key(l, m, c) for l, m, c in zip(league, market_type, consensus)]
            calib_win = pd.to_numeric(
                apply_bucket_calibration(win, buckets, calibration, bucket_stats), errors="coerce"
            )
        except Exception:
            calib_win = win
    else:
        calib_win = pd.Series([None] * len(df), index=df.index)
    breakeven = pd.Series([_american_breakeven(o) for o in odds], index=df.index)

    tier = [
        classify_lean_tier(s, e, c, calibrated_win=cw, break_even=be)
        for s, e, c, cw, be in zip(status, eff_ev, consensus, calib_win, breakeven)
    ]

    # Empirical edge = bucket-aware calibrated win minus break-even. This — NOT model EV — is
    # the predictive ranking signal: across 63 graded picks the model's EV ranking was
    # inverted (its highest-EV/most-contrarian picks lost), while bucket-realized performance
    # held up. So the card is ordered by Emp_Edge, best first, within each tier.
    calib_num = pd.to_numeric(calib_win, errors="coerce")
    emp_edge = calib_num - pd.to_numeric(breakeven, errors="coerce")

    out = pd.DataFrame({
        "League": _first_col(df, "league", "League"),
        "Matchup": (away + " @ " + home).str.strip(" @"),
        "Pick": _first_col(df, "best_pick"),
        "Win%": win,
        "Calib_Win%": calib_num,
        "Emp_Edge": emp_edge,
        "Edge": pd.to_numeric(edge, errors="coerce"),
        "EV": pd.to_numeric(eff_ev, errors="coerce"),
        "Consensus": consensus,
        "Tier": tier,
        # Stake only on the BET tier (the genuinely actionable picks); LEAN/AVOID are reads,
        # not staked, so the user decides any size themselves.
        "Suggested_Stake": kelly.where(pd.Series(tier, index=df.index) == "BET", 0.0),
    })
    out["_t"] = out["Tier"].map(_TIER_ORDER).fillna(3)
    # Rank by empirical edge (bucket-proven), not model Win%/EV — the latter is anti-informative.
    # Win% is the tiebreaker and the fallback when there's no calibration (Emp_Edge all NaN).
    out = out.sort_values(
        ["_t", "Emp_Edge", "Win%"], ascending=[True, False, False], na_position="last"
    ).drop(columns="_t").reset_index(drop=True)
    return out
