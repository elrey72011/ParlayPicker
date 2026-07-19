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
from core.parlay_safety import conservative_joint_probability, row_is_untrusted, shares_game

DUO_MIN_LEG_PROBABILITY = 0.55   # same bar as everything else win-prob-first
DUO_STRICT_MIN_LEG_PROBABILITY = 0.60
DUO_MAX_LEGS_CONSIDERED = 14     # top legs by probability; pairs are O(n^2)
DUO_PROBABILITY_HAIRCUT = 0.90
DUO_STRICT_MIN_PARLAY_EV = 0.05
PARLAY_MINIMUM_BET = 1.0

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


def _pick_direction(text: object) -> str:
    value = f" {str(text or '').lower()} "
    if " over " in value:
        return "over"
    if " under " in value:
        return "under"
    return ""


def build_best_duos(
    best_picks_df: pd.DataFrame | None,
    prop_card: pd.DataFrame | None,
    max_duos: int = 3,
    min_leg_probability: float = DUO_MIN_LEG_PROBABILITY,
    require_mixed: bool = False,
    strict: bool = False,
    allow_probation: bool = False,
    min_parlay_ev: float = 0.0,
) -> pd.DataFrame:
    """Top ``max_duos`` two-leg parlays, no shared games, no reused legs.

    ``require_mixed=True`` restricts pairs to one GAME leg + one PROP leg
    (owner request, 8 Jul: "a parlay with one game and one strikeout prop").

    strict=True is the production path. It requires game rows to be
    explicitly production-eligible, excludes probationary prop markets, rejects
    fallback/degraded rows, applies a 10% model-risk haircut to the joint
    probability, and drops parlays with non-positive post-haircut EV.

    Returns columns: leg1, leg2, leg1_prob, leg2_prob, combined_probability,
    combined_decimal, parlay_ev, payout_per_10.
    """
    if strict:
        # Game candidates must have passed the main production portfolio gate.
        # Props use the stricter Actionable-only path and probationary markets
        # stay visible on the prop card but cannot enter parlays.
        safe_games = best_picks_df
        if safe_games is not None and not safe_games.empty:
            if "production_eligible" in safe_games.columns:
                safe_games = safe_games[
                    safe_games["production_eligible"].fillna(False).astype(bool)
                ]
            else:
                safe_games = safe_games.iloc[0:0]
        safe_props = prop_card
        if safe_props is not None and not safe_props.empty:
            if "production_eligible" in safe_props.columns:
                safe_props = safe_props[
                    safe_props["production_eligible"].fillna(False).astype(bool)
                ]
            if "Stake_Status" in safe_props.columns:
                safe_props = safe_props[
                    safe_props["Stake_Status"].astype(str).str.strip().eq("Funded")
                ]
            safe_props = safe_props[
                safe_props.get(
                    "Pick_Status",
                    pd.Series("Actionable", index=safe_props.index),
                ).astype(str).str.strip().eq("Actionable")
            ].copy()
            if "Market_Probation" in safe_props.columns and not allow_probation:
                safe_props = safe_props[
                    ~safe_props["Market_Probation"].fillna(False).astype(bool)
                ]
            if "Kelly_Bet_Size" in safe_props.columns:
                safe_props = safe_props[
                    pd.to_numeric(safe_props["Kelly_Bet_Size"], errors="coerce")
                    .fillna(0.0).gt(0)
                ]
            if not safe_props.empty:
                safe_props = safe_props[
                    ~safe_props.apply(row_is_untrusted, axis=1)
                ]
        game_pool = _game_candidates(safe_games)
        prop_pool = _prop_candidates(safe_props)
    else:
        game_pool = _game_candidates(best_picks_df)
        prop_pool = _prop_candidates(prop_card)

    pool = pd.concat([game_pool, prop_pool], ignore_index=True)
    if pool.empty:
        return pd.DataFrame()
    probability_floor = (
        max(float(min_leg_probability), DUO_STRICT_MIN_LEG_PROBABILITY)
        if strict else float(min_leg_probability)
    )
    pool = pool[pd.to_numeric(pool["win_probability"], errors="coerce").ge(probability_floor)]
    if strict and "edge" in pool.columns:
        pool = pool[pd.to_numeric(pool["edge"], errors="coerce").ge(0.02)]
    if len(pool) < 2:
        return pd.DataFrame()
    pool = pool.sort_values("win_probability", ascending=False).head(DUO_MAX_LEGS_CONSIDERED).reset_index(drop=True)
    # Match only on the game description. Including the pick text made generic
    # betting words such as "Under" look like a shared game and incorrectly
    # suppressed otherwise independent parlays.
    pool["_toks"] = [_matchup_tokens(d) for d in pool["detail"]]
    required_parlay_ev = max(
        float(min_parlay_ev),
        DUO_STRICT_MIN_PARLAY_EV if strict else 0.0,
    )

    pairs = []
    for i, j in itertools.combinations(range(len(pool)), 2):
        a, b = pool.iloc[i], pool.iloc[j]
        common_book = ""
        if strict:
            book_a = str(a.get("book", "") or "").strip().lower()
            book_b = str(b.get("book", "") or "").strip().lower()
            if book_a in {"", "nan", "none"} or book_a != book_b:
                continue  # cannot quote or place a cross-book parlay
            common_book = book_a
        if a["_toks"] & b["_toks"]:
            continue  # same game — correlated, skip
        if a["board"] == "prop" and b["board"] == "prop":
            participant_a = str(a.get("participant", "") or "").strip()
            participant_b = str(b.get("participant", "") or "").strip()
            if participant_a and participant_a == participant_b:
                continue  # never parlay two markets on the same player
        if allow_probation:
            # Research parlays must diversify model-direction risk too.
            direction_a, direction_b = _pick_direction(a["pick"]), _pick_direction(b["pick"])
            if direction_a and direction_a == direction_b:
                continue
        if require_mixed and a["board"] == b["board"]:
            continue  # one game leg + one prop leg only
        if strict and shares_game(a, b):
            continue
        raw_p = float(a["win_probability"]) * float(b["win_probability"])
        p = conservative_joint_probability(
            [float(a["win_probability"]), float(b["win_probability"])],
            haircut=DUO_PROBABILITY_HAIRCUT,
        ) if strict else raw_p
        da, db = _decimal(a["odds_american"]), _decimal(b["odds_american"])
        dec = (da * db) if (da and db) else None
        parlay_ev = (p * dec - 1.0) if dec else None
        if strict and (
            parlay_ev is None
            or parlay_ev <= 0
            or parlay_ev < required_parlay_ev
        ):
            continue
        pairs.append((p, i, j, dec, parlay_ev, common_book))
    pairs.sort(key=lambda x: x[0], reverse=True)

    used: set[int] = set()
    rows = []
    for p, i, j, dec, parlay_ev, common_book in pairs:
        if i in used or j in used:
            continue
        a, b = pool.iloc[i], pool.iloc[j]
        # Keep mixed-parlay exports stable and human-readable regardless of
        # probability sorting: game leg first, prop leg second.
        if a["board"] == "prop" and b["board"] == "game":
            a, b = b, a
        rows.append({
            "boards": f"{a['board']}+{b['board']}",
            "leg1": f"{a['pick']} ({a['detail']})".strip(),
            "leg2": f"{b['pick']} ({b['detail']})".strip(),
            "leg1_prob": round(float(a["win_probability"]), 4),
            "leg2_prob": round(float(b["win_probability"]), 4),
            "combined_probability": round(p, 4),
            "combined_decimal": round(dec, 3) if dec else None,
            "parlay_ev": round(parlay_ev, 4) if parlay_ev is not None else None,
            "payout_per_10": round(10.0 * dec, 2) if dec else None,
            "common_book": common_book or None,
            "production_safety_mode": bool(strict and not allow_probation),
            "probation_parlay_mode": bool(strict and allow_probation),
            "model_risk_haircut": DUO_PROBABILITY_HAIRCUT if strict else 1.0,
        })
        used.update((i, j))
        if len(rows) >= max_duos:
            break
    return pd.DataFrame(rows)


def duos_to_smart_parlays(duos: pd.DataFrame | None, bankroll: float = 1000.0) -> pd.DataFrame:
    """Map strict Best Duos into the strategic-parlay export schema."""
    if duos is None or duos.empty:
        return pd.DataFrame()
    rows = []
    for index, duo in duos.reset_index(drop=True).iterrows():
        probability = float(duo.get("combined_probability") or 0.0)
        decimal = float(duo.get("combined_decimal") or 0.0)
        ev = float(duo.get("parlay_ev") or 0.0)
        if probability <= 0 or decimal <= 1 or ev <= 0:
            continue
        full_kelly = max(0.0, (probability * decimal - 1.0) / (decimal - 1.0))
        fractional_kelly = full_kelly * 0.125
        probation_mode = bool(duo.get("probation_parlay_mode", False))
        recommended = (
            PARLAY_MINIMUM_BET if probation_mode
            else min(float(bankroll) * 0.0025, float(bankroll) * fractional_kelly)
        )
        book_value = duo.get("common_book")
        common_book = "" if pd.isna(book_value) else str(book_value).strip().lower()
        if common_book in {"", "nan", "none"} or recommended + 1e-9 < PARLAY_MINIMUM_BET:
            continue
        book_display = "Novig" if common_book == "novig" else common_book.replace("_", " ").title()
        rows.append({
            "risk_tier": "Probation / Research" if probation_mode else "Controlled",
            "group_id": f"strict_duo_{index + 1}",
            "parlay_legs": f"{duo.get('leg1')} | {duo.get('leg2')}",
            "combined_probability": probability,
            "combined_decimal_odds": decimal,
            "parlay_ev": ev,
            "legs": 2,
            "combined_market_prob": None,
            "ev_boost_pct": None,
            "is_high_correlation": False,
            "best_payout_book": book_display,
            "Conviction_Score": min(float(duo.get("leg1_prob") or 0.0), float(duo.get("leg2_prob") or 0.0)),
            "min_leg_prob": min(float(duo.get("leg1_prob") or 0.0), float(duo.get("leg2_prob") or 0.0)),
            "kelly_fraction": fractional_kelly,
            "recommended_bet": round(recommended, 2),
            "production_safety_mode": bool(duo.get("production_safety_mode", True)),
            "probation_parlay_mode": probation_mode,
            "model_risk_haircut": duo.get("model_risk_haircut"),
        })
    return pd.DataFrame(rows)



def build_tiered_prop_parlays(
    best_picks_df: pd.DataFrame | None,
    prop_card: pd.DataFrame | None,
    bankroll: float = 1000.0,
) -> pd.DataFrame:
    """Return the best independently priced production duo when available.

    Research/probation props remain gradeable on their separate card but never
    become a recommended parlay ticket.
    """
    production_duos = build_best_duos(best_picks_df, prop_card, max_duos=1, strict=True)
    return duos_to_smart_parlays(production_duos, bankroll=bankroll)
