"""Best Overall Pick of the Day across funded game and prop tickets.

The banner is a production recommendation, not a research-leader label. It may
therefore be empty when the model has no funded edge. Research-only props and
recreational game leans remain visible on their own cards, but cannot be
promoted into a suggested wager by this selector.

Honesty rules:
  * Hard-disqualified game rows never qualify: already-started games,
    wrong-game Kalshi matches, proven-losing buckets, unresolved/rejected lines.
  * A candidate must be production-eligible, Actionable, and carry a positive
    executable stake.
  * The selector never invents a courtesy stake for an unfunded row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app_core.weights_config import MIN_STAKE_WIN_PROBABILITY

# Retained for compatibility with older callers; production selection no longer
# creates a courtesy bet when the board has no funded edge.
BELOW_FLOOR_ACTION_STAKE = 5.0

_GAME_DISQUALIFYING_STAGES = {
    "game_already_started",
    "kalshi_wrong_game_title",
    "line_provenance_unresolved",
    "extreme_price_guard",
    "empirical_proven_losing_bucket",
}


def _num(series_like, default=np.nan, index=None) -> pd.Series:
    """Numeric Series coercion that is safe for missing optional columns."""
    if isinstance(series_like, pd.Series):
        return pd.to_numeric(series_like, errors="coerce").fillna(default)
    if series_like is None:
        return pd.Series(default, index=index, dtype=float)
    return pd.to_numeric(pd.Series(series_like, index=index), errors="coerce").fillna(default)


def _game_candidates(best_picks_df: pd.DataFrame | None) -> pd.DataFrame:
    if best_picks_df is None or best_picks_df.empty:
        return pd.DataFrame()
    df = best_picks_df.copy()

    keep = pd.Series(True, index=df.index)
    if "game_already_started_flag" in df.columns:
        keep &= ~df["game_already_started_flag"].fillna(False).astype(bool)
    if "status_blocker_stage" in df.columns:
        keep &= ~df["status_blocker_stage"].astype(str).isin(_GAME_DISQUALIFYING_STAGES)
    status = df.get("Pick_Status", pd.Series("", index=df.index)).astype(str).str.strip()
    stake = _num(df.get("Kelly_Bet_Size"), default=0.0, index=df.index)
    production_eligible = pd.Series(
        df.get("production_eligible", True), index=df.index
    ).fillna(False).astype(bool)
    keep &= status.eq("Actionable") & production_eligible & stake.gt(0)
    if "best_pick" in df.columns:
        bp = df["best_pick"].astype(str)
        keep &= bp.str.strip().ne("") & ~bp.str.contains("Unresolved|Rejected", case=False, na=False)
    df = df[keep]
    if df.empty:
        return pd.DataFrame()

    # Probability basis mirrors the stake floor: empirical -> effective -> calibrated.
    prob = pd.Series(np.nan, index=df.index)
    for col in ("empirical_win_probability", "effective_win_probability", "calibrated_probability"):
        if col in df.columns:
            prob = prob.fillna(_num(df[col]))
    matchup = (
        df.get("away_team", pd.Series("", index=df.index)).astype(str)
        + " @ "
        + df.get("home_team", pd.Series("", index=df.index)).astype(str)
    )
    edge_source = next(
        (c for c in ("empirical_edge", "effective_edge", "edge") if c in df.columns),
        None,
    )
    ev_source = next(
        (c for c in ("effective_expected_value", "expected_value") if c in df.columns),
        None,
    )
    out = pd.DataFrame(
        {
            "board": "game",
            "league": df.get("league", pd.Series("", index=df.index)).astype(str),
            "pick": df["best_pick"].astype(str) if "best_pick" in df.columns else matchup,
            "detail": matchup,
            "win_probability": prob,
            "odds_american": _num(df.get("odds_american"), index=df.index),
            "edge": _num(df[edge_source]) if edge_source else pd.Series(np.nan, index=df.index),
            "expected_value": _num(df[ev_source]) if ev_source else pd.Series(np.nan, index=df.index),
            "kelly_stake": stake.loc[df.index],
            "pick_status": status.loc[df.index],
        }
    )
    return out[out["win_probability"].notna()]


def _prop_candidates(prop_card: pd.DataFrame | None) -> pd.DataFrame:
    if prop_card is None or prop_card.empty:
        return pd.DataFrame()
    df = prop_card.copy()
    status = df.get(
        "Pick_Status", pd.Series("", index=df.index)
    ).astype(str).str.strip()
    stake = _num(df.get("Kelly_Bet_Size"), default=0.0, index=df.index)
    production_eligible = pd.Series(
        df.get("production_eligible", True), index=df.index
    ).fillna(False).astype(bool)
    if "Stake_Status" in df.columns:
        funded = df["Stake_Status"].astype(str).str.strip().eq("Funded")
    else:
        funded = status.eq("Actionable") & stake.gt(0)
    keep = funded & status.eq("Actionable") & production_eligible & stake.gt(0)
    df = df[keep]
    if df.empty:
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            "board": "prop",
            "league": df.get("league", pd.Series("MLB", index=df.index)).astype(str),
            "pick": df.get("best_pick", pd.Series("", index=df.index)).astype(str),
            "detail": df.get("matchup", pd.Series("", index=df.index)).astype(str),
            "participant": df.get(
                "player", pd.Series("", index=df.index)
            ).fillna("").astype(str).str.lower().str.split().str.join(" "),
            "win_probability": _num(df.get("WinProbability"), index=df.index),
            "edge": _num(df.get("edge"), index=df.index),
            "expected_value": _num(df.get("expected_value"), index=df.index),
            "odds_american": _num(df.get("odds_american"), index=df.index),
            "book": df.get("book", pd.Series("", index=df.index)).astype(str).str.lower().str.strip(),
            "kelly_stake": stake.loc[df.index],
            "pick_status": status.loc[df.index],
        }
    )
    return out[out["win_probability"].notna()]


def select_pick_of_the_day(
    best_picks_df: pd.DataFrame | None,
    prop_card: pd.DataFrame | None,
    min_win_probability: float = float(MIN_STAKE_WIN_PROBABILITY),
) -> dict | None:
    """The likeliest funded production ticket, or ``None`` when the board abstains.

    Returns a dict: board, league, pick, detail, win_probability, odds_american,
    stake, below_floor, runner_up (same shape, sans runner_up, or None).
    """
    pool = pd.concat(
        [_game_candidates(best_picks_df), _prop_candidates(prop_card)],
        ignore_index=True,
    )
    if pool.empty:
        return None
    pool = pool.sort_values(
        ["win_probability", "kelly_stake"], ascending=[False, False]
    ).reset_index(drop=True)

    def _row_to_dict(r: pd.Series) -> dict:
        below = float(r["win_probability"]) < float(min_win_probability)
        stake = float(r["kelly_stake"])
        return {
            "board": str(r["board"]),
            "league": str(r["league"]),
            "pick": str(r["pick"]),
            "detail": str(r["detail"]),
            "win_probability": float(r["win_probability"]),
            "odds_american": None if pd.isna(r["odds_american"]) else float(r["odds_american"]),
            "stake": round(stake, 2),
            "below_floor": below,
        }

    top = _row_to_dict(pool.iloc[0])
    top["runner_up"] = _row_to_dict(pool.iloc[1]) if len(pool) > 1 else None
    return top

