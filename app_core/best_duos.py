"""Best Duos — the likeliest 2-leg parlays across the whole board (owner, 8 Jul).

"I really want the best 2 leg parlays without overlap despite the bet":
  * Legs come from EVERYTHING bettable — game picks and pitcher props — on the
    same honest probability basis the Pick of the Day uses (empirical ->
    effective for games, model WinProbability for props), with the same hard
    disqualifiers (started games, wrong-game Kalshi, proven-losing buckets).
  * "Without overlap": two legs from the same GAME are never paired — a prop on
    a pitcher is correlated with that game's total/run-line, and same-game
    correlation quietly turns a "70% x 65%" parlay into something worse.
    Duos in the ranked list also never reuse a leg, so the top-3 are nine... six
    independent bets, not one pick wearing three hats.
  * "Despite the bet": ranked by JOINT WIN PROBABILITY (p1 x p2, independent by
    construction since legs share no game), not by payout. Combined odds are
    shown so the price is visible, never the ranking key.
"""
from __future__ import annotations

import itertools

import pandas as pd

from app_core.pick_of_day import _game_candidates, _prop_candidates

DUO_MIN_LEG_PROBABILITY = 0.55   # same bar as everything else win-prob-first
DUO_MAX_LEGS_CONSIDERED = 14     # top legs by probability; pairs are O(n^2)

# Generic tokens that appear in many team names and must not create phantom
# overlap ("New York Yankees" vs "New York Mets" DO overlap — that's the same
# city but different games are fine; the token that matters is the club token).
_GENERIC_TOKENS = {"new", "los", "las", "san", "st", "st.", "saint", "city", "bay", "blue", "red", "white", "sox"}


def _matchup_tokens(text: object) -> frozenset:
    """Distinctive tokens of a matchup string, for overlap detection.

    Both boards print matchups differently ("Toronto @ San Francisco" vs
    "Toronto Blue Jays @ San Francisco Giants"), so overlap = any shared
    distinctive token (len >= 4, non-generic), lowercased.
    """
    toks = str(text or "").lower().replace("@", " ").replace("vs", " ").split()
    return frozenset(t for t in toks if len(t) >= 4 and t not in _GENERIC_TOKENS)


def _decimal(odds: object) -> float | None:
    o = pd.to_numeric(pd.Series([odds]), errors="coerce").iloc[0]
    if pd.isna(o) or o == 0:
        return None
    o = float(o)
    return 1.0 + (o / 100.0 if o > 0 else 100.0 / -o)


def build_best_duos(
    best_picks_df: pd.DataFrame | None,
    prop_card: pd.DataFrame | None,
    max_duos: int = 3,
    min_leg_probability: float = DUO_MIN_LEG_PROBABILITY,
) -> pd.DataFrame:
    """Top ``max_duos`` two-leg parlays, no shared games, no reused legs.

    Returns columns: leg1, leg2, leg1_prob, leg2_prob, combined_probability,
    combined_decimal, payout_per_10 (None when either leg lacks a price).
    """
    pool = pd.concat(
        [_game_candidates(best_picks_df), _prop_candidates(prop_card)],
        ignore_index=True,
    )
    if pool.empty:
        return pd.DataFrame()
    pool = pool[pd.to_numeric(pool["win_probability"], errors="coerce").ge(float(min_leg_probability))]
    if len(pool) < 2:
        return pd.DataFrame()
    pool = pool.sort_values("win_probability", ascending=False).head(DUO_MAX_LEGS_CONSIDERED).reset_index(drop=True)
    pool["_toks"] = [_matchup_tokens(f"{d} {p}") for d, p in zip(pool["detail"], pool["pick"])]

    pairs = []
    for i, j in itertools.combinations(range(len(pool)), 2):
        a, b = pool.iloc[i], pool.iloc[j]
        if a["_toks"] & b["_toks"]:
            continue  # same game (or same pitcher) — correlated, skip
        p = float(a["win_probability"]) * float(b["win_probability"])
        da, db = _decimal(a["odds_american"]), _decimal(b["odds_american"])
        dec = (da * db) if (da and db) else None
        pairs.append((p, i, j, dec))
    pairs.sort(key=lambda x: x[0], reverse=True)

    used: set[int] = set()
    rows = []
    for p, i, j, dec in pairs:
        if i in used or j in used:
            continue
        a, b = pool.iloc[i], pool.iloc[j]
        rows.append({
            "leg1": f"{a['pick']} ({a['detail']})".strip(),
            "leg2": f"{b['pick']} ({b['detail']})".strip(),
            "leg1_prob": round(float(a["win_probability"]), 4),
            "leg2_prob": round(float(b["win_probability"]), 4),
            "combined_probability": round(p, 4),
            "combined_decimal": round(dec, 3) if dec else None,
            "payout_per_10": round(10.0 * dec, 2) if dec else None,
        })
        used.update((i, j))
        if len(rows) >= max_duos:
            break
    return pd.DataFrame(rows)
