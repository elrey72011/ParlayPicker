"""All-games lean view: the model's read on EVERY game, tiered honestly.

The games card stakes only proven +EV picks, so on an efficient slate it looks empty even
though the model has a directional read on every game. This view re-presents the SAME card
(it adds no staking and changes no guard) so a bettor who wants action across the board can
see, per game: the model's side, its confidence, and an honest tier -

  * BET   - the pick the system would actually stake (Actionable: a real, priced edge).
  * LEAN  - the model has a positive-EV side but below the stake bar, and it is not fading
            Kalshi. A read worth knowing; NOT a proven +EV bet. Bet at your own risk.
  * AVOID - negative EV at the price, or the model is fading consensus (Disagrees). The
            board the math says to stay off.

The tiers separate a best-available directional read from a production wager. Every game
remains visible, but only a BET clears the absolute price gate and receives money.
"""
from __future__ import annotations

import pandas as pd

from core.production_gate import (
    MIN_PRODUCTION_CALIBRATED_EDGE,
    evaluate_absolute_production_gate,
)


def _first_col(df: pd.DataFrame, *names: str):
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series([None] * len(df), index=df.index)


def classify_lean_tier(status: object, eff_ev: object, consensus: object,
                       *, calibrated_win: object = None, break_even: object = None) -> str:
    """BET / LEAN / AVOID for one row (see module docstring).

    When ``calibrated_win`` and ``break_even`` are supplied, a would-be LEAN is demoted to
    AVOID if its CALIBRATED win probability fails to beat the break-even price - the model's
    raw EV is overconfident in the 0.50-0.55 band (327 graded picks: predicted .53, realized
    .43), so a positive *raw* EV there is usually a negative *calibrated* EV.
    """
    ev = pd.to_numeric(pd.Series([eff_ev]), errors="coerce").iloc[0]
    cons = str(consensus or "").strip()
    cw = pd.to_numeric(pd.Series([calibrated_win]), errors="coerce").iloc[0]
    be = pd.to_numeric(pd.Series([break_even]), errors="coerce").iloc[0]
    absolute_edge = (float(cw) - float(be)) if pd.notna(cw) and pd.notna(be) else None

    # "Actionable" is necessary but no longer sufficient. A production BET must
    # also clear the offered price by an absolute safety margin and retain
    # positive model EV. This prevents a relative best-in-game candidate from
    # being funded merely because every alternative was worse.
    if (
        str(status).strip() == "Actionable"
        and pd.notna(ev)
        and float(ev) > 0.0
        and absolute_edge is not None
        and absolute_edge >= MIN_PRODUCTION_CALIBRATED_EDGE
    ):
        return "BET"
    if not (pd.notna(ev) and ev > 0 and cons != "Disagrees"):
        return "AVOID"
    if absolute_edge is not None and absolute_edge <= 0.0:
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


def score_best_picks_rows(best_picks_df: pd.DataFrame, *, calibration: object = _UNSET,
                          bucket_stats: object = _UNSET) -> pd.DataFrame:
    """Per-row lean scoring, INDEX-ALIGNED to the input frame (no sorting).

    Shared core of the Play Card and the main-card tier columns: computes the
    bucket-calibrated win probability, break-even, Emp_Edge, and BET/LEAN/AVOID
    tier for every row of a best-picks frame. build_all_games_lean_card sorts
    and formats this; the main Best Picks card joins it back by index so every
    displayed row carries a tier and a playable stake.
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
    # under:Agrees ~61%) isn't crushed below break-even by the pooled curve - same number the
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
        # The upstream effective probability is already the best available
        # calibrated value on legacy/no-artifact runs. Using it as the explicit
        # fallback keeps the gate price-aware instead of silently funding blind.
        calib_win = win.copy()
    breakeven = pd.Series([_american_breakeven(o) for o in odds], index=df.index)
    calib_num = pd.to_numeric(calib_win, errors="coerce")
    break_even_num = pd.to_numeric(breakeven, errors="coerce")
    gate = evaluate_absolute_production_gate(calib_num, break_even_num, eff_ev)

    tier = [
        classify_lean_tier(s, e, c, calibrated_win=cw, break_even=be)
        for s, e, c, cw, be in zip(status, eff_ev, consensus, calib_win, breakeven)
    ]

    # Empirical edge = bucket-aware calibrated win minus break-even. This - NOT model EV - is
    # the predictive ranking signal: across 63 graded picks the model's EV ranking was
    # inverted (its highest-EV/most-contrarian picks lost), while bucket-realized performance
    # held up. So the card is ordered by Emp_Edge, best first, within each tier.
    emp_edge = gate["absolute_production_edge"]

    # A started game is never playable at ANY size - pre-game lines are stale
    # and the shown odds may be in-game. attach_play_stakes zeroes these rows.
    started = _first_col(df, "game_already_started_flag").map(
        lambda v: bool(v) if pd.notna(v) and v is not None else False
    )
    if "status_blocker_stage" in df.columns:
        started = started | df["status_blocker_stage"].astype(str).eq("game_already_started")
    if "Play_Tier" in df.columns:
        started = started | df["Play_Tier"].astype(str).str.strip().str.upper().eq("STARTED")
    pick_text = _first_col(df, "best_pick").fillna("").astype(str)
    unavailable_line = pick_text.str.lower().str.contains(
        r"unresolved|\(no line\)|missing line|rejected", regex=True, na=False
    )
    unsafe_line_identity = pd.Series(False, index=df.index)
    if "line_consistency_flag" in df.columns:
        unsafe_line_identity = unsafe_line_identity | ~df["line_consistency_flag"].fillna(False).astype(bool)
    if "line_event_identity_match_flag" in df.columns:
        unsafe_line_identity = unsafe_line_identity | ~df["line_event_identity_match_flag"].fillna(False).astype(bool)
    if "market_line_source_detail" in df.columns:
        unsafe_line_identity = unsafe_line_identity | df["market_line_source_detail"].astype(str).eq(
            "upload_total_fallback_after_rejected_live"
        )
    playable = ~(started | unavailable_line | unsafe_line_identity)
    production_gate_pass = (
        pd.Series(tier, index=df.index).eq("BET")
        & gate["production_gate_pass"]
        & playable
    )
    production_gate_reason = gate["production_gate_reason"].copy()
    non_actionable = ~status.astype(str).str.strip().eq("Actionable")
    production_gate_reason.loc[non_actionable & gate["production_gate_pass"]] = (
        "upstream status is not Actionable"
    )

    return pd.DataFrame({
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
        "Started": started,
        # This only states that a valid pregame line is present; wager approval
        # remains exclusively in Production_Gate_Pass / Wager_Approved.
        "Line_Available": playable,
        "Selection_Mode": "Best Available",
        "Bet_Decision": production_gate_pass.map(
            {True: "BET", False: "BEST AVAILABLE - PASS"}
        ),
        "Production_Gate_Pass": production_gate_pass,
        "Production_Gate_Reason": production_gate_reason,
        "Absolute_Edge": gate["absolute_production_edge"],
        "Calibrated_EV": gate["calibrated_expected_value"],
        # Stake only on rows that pass the independent absolute price gate.
        "Suggested_Stake": kelly.where(production_gate_pass, 0.0),
    }, index=df.index)


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
    out = score_best_picks_rows(best_picks_df, calibration=calibration, bucket_stats=bucket_stats)
    if out.empty:
        return out
    out["_t"] = out["Tier"].map(_TIER_ORDER).fillna(3)
    # Rank by empirical edge (bucket-proven), not model Win%/EV - the latter is anti-informative.
    # Win% is the tiebreaker and the fallback when there's no calibration (Emp_Edge all NaN).
    out = out.sort_values(
        ["_t", "Emp_Edge", "Win%"], ascending=[True, False, False], na_position="last"
    ).drop(columns="_t").reset_index(drop=True)
    return out


# Display units by tier for attach_play_stakes. Only a genuine BET receives a
# stake. LEAN and AVOID remain useful reads, but assigning dollars to them
# contradicts the production gate and makes an abstaining card look bettable.
PLAY_UNITS_BET = 2.0
PLAY_UNITS_LEAN = 0.0
PLAY_UNITS_AVOID_NEAR = 0.0
PLAY_UNITS_AVOID_FAR = 0.0
AVOID_NEAR_EDGE = -0.05
# Hopeless-price floor: below this calibrated-win-vs-break-even edge, even the
# minimum action unit is money on fire (4 Jul: CWS +5.5 at -1718, Emp_Edge
# -0.35, drew a $2.50 play stake). No recreational stake at any size.
PLAY_NO_PRICE_EDGE = -0.20


def attach_play_stakes(card: pd.DataFrame, unit: float = 1.0) -> pd.DataFrame:
    """Attach a dollar stake only to production-quality BET rows.

    LEAN and AVOID rows remain visible so the model's full-board read is useful,
    but they receive zero dollars. This keeps Play_Stake aligned with the app's
    positive-EV production decision:

      BET    2.0u (or the pick's own Kelly stake if larger - a real edge)
      LEAN   0u
      AVOID  0u

    Pure: returns a copy with Play_Stake ($) and Play_Units columns.
    """
    if card is None or card.empty:
        return pd.DataFrame() if card is None else card.copy()
    out = card.copy()
    tier = out.get("Tier", pd.Series("", index=out.index)).astype(str)
    emp_edge = pd.to_numeric(out.get("Emp_Edge"), errors="coerce")

    units = pd.Series(0.0, index=out.index)
    units[tier.eq("BET")] = PLAY_UNITS_BET
    # Hopeless prices (deeply below break-even) get $0 at any tier except a
    # genuine BET: paying -1700 juice for "action" isn't recreation, it's a fee.
    units[emp_edge.lt(PLAY_NO_PRICE_EDGE) & ~tier.eq("BET")] = 0.0

    stake = units * float(unit)
    if "Suggested_Stake" in out.columns:
        kelly = pd.to_numeric(out["Suggested_Stake"], errors="coerce").fillna(0.0)
        stake = stake.where(~(tier.eq("BET") & kelly.gt(stake)), kelly)

    # Prefer the explicit availability name while accepting legacy cards that
    # still carry Playable. Availability is not itself wager approval.
    availability_column = (
        "Line_Available" if "Line_Available" in out.columns
        else "Playable" if "Playable" in out.columns
        else None
    )
    if availability_column is not None:
        line_available = out[availability_column].fillna(False).astype(bool)
        units = units.where(line_available, 0.0)
        stake = stake.where(line_available, 0.0)
        out.loc[~line_available, "Tier"] = "UNAVAILABLE"
        if "Bet_Decision" in out.columns:
            out.loc[~line_available, "Bet_Decision"] = "UNAVAILABLE"

    # Started games are unplayable at any size: the pre-game line is gone.
    if "Started" in out.columns:
        started = out["Started"].fillna(False).astype(bool)
        units = units.where(~started, 0.0)
        stake = stake.where(~started, 0.0)
        out.loc[started, "Tier"] = "STARTED"
        if "Bet_Decision" in out.columns:
            out.loc[started, "Bet_Decision"] = "STARTED"

    out["Play_Units"] = units
    out["Play_Stake"] = stake.round(2)
    out["All_Row_Bet"] = stake.gt(0)
    out["Wager_Approved"] = out["Play_Stake"].gt(0)
    out["Export_Role"] = "COVERAGE PICK - PASS"
    out.loc[out["Wager_Approved"], "Export_Role"] = "PRODUCTION WAGER"
    out["Wager_Instruction"] = "DO NOT BET: diagnostic best-available direction only."
    out.loc[out["Wager_Approved"], "Wager_Instruction"] = (
        "APPROVED: wager the exported Play_Stake amount."
    )
    return out
