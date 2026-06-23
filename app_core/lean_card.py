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


def classify_lean_tier(status: object, eff_ev: object, consensus: object) -> str:
    """BET / LEAN / AVOID for one row (see module docstring)."""
    if str(status).strip() == "Actionable":
        return "BET"
    ev = pd.to_numeric(pd.Series([eff_ev]), errors="coerce").iloc[0]
    cons = str(consensus or "").strip()
    if pd.notna(ev) and ev > 0 and cons != "Disagrees":
        return "LEAN"
    return "AVOID"


_TIER_ORDER = {"BET": 0, "LEAN": 1, "AVOID": 2}


def build_all_games_lean_card(best_picks_df: pd.DataFrame) -> pd.DataFrame:
    """Derive the all-games lean card from the games best-picks frame.

    Pure and side-effect-free: reads the existing per-game pick/probability/EV columns,
    assigns a tier, and returns a compact, sorted view (BET first, then by confidence).
    Tolerant of the pre- and post-export column names (home_team/Home, etc.).
    """
    if best_picks_df is None or best_picks_df.empty:
        return pd.DataFrame()

    df = best_picks_df
    home = _first_col(df, "Home", "home_team").astype(str)
    away = _first_col(df, "Away", "away_team").astype(str)
    status = _first_col(df, "Pick_Status")
    eff_ev = _first_col(df, "effective_expected_value", "expected_value")
    consensus = _first_col(df, "consensus_agreement")
    win = _first_col(df, "effective_win_probability", "WinProbability")
    edge = _first_col(df, "effective_edge", "edge")
    kelly = pd.to_numeric(_first_col(df, "Kelly_Bet_Size"), errors="coerce").fillna(0.0)

    tier = [classify_lean_tier(s, e, c) for s, e, c in zip(status, eff_ev, consensus)]

    out = pd.DataFrame({
        "League": _first_col(df, "league", "League"),
        "Matchup": (away + " @ " + home).str.strip(" @"),
        "Pick": _first_col(df, "best_pick"),
        "Win%": pd.to_numeric(win, errors="coerce"),
        "Edge": pd.to_numeric(edge, errors="coerce"),
        "EV": pd.to_numeric(eff_ev, errors="coerce"),
        "Consensus": consensus,
        "Tier": tier,
        # Stake only on the BET tier (the genuinely actionable picks); LEAN/AVOID are reads,
        # not staked, so the user decides any size themselves.
        "Suggested_Stake": kelly.where(pd.Series(tier, index=df.index) == "BET", 0.0),
    })
    out["_t"] = out["Tier"].map(_TIER_ORDER).fillna(3)
    out = out.sort_values(["_t", "Win%"], ascending=[True, False]).drop(columns="_t").reset_index(drop=True)
    return out
