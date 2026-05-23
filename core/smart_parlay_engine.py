from __future__ import annotations

from itertools import combinations

import pandas as pd
import hashlib
import json

# Only picks with these statuses qualify as parlay legs.
# Below Threshold explicitly failed the confidence gate and shouldn't anchor a parlay.
PARLAY_ELIGIBLE_STATUSES = {"Actionable", "High Variance/Speculative"}

# Minimum calibrated probability per leg — legs below this floor drag down every
# combo they touch. 0.54 keeps out borderline coin-flip picks.
MIN_LEG_PROBABILITY = 0.54

# Minimum edge per leg — ensures each leg genuinely beats the market.
MIN_LEG_EDGE = 0.02


def _amer_to_dec(odds: float) -> float:
    if pd.isna(odds) or odds == 0:
        return 1.0
    if odds > 0:
        return 1 + (odds / 100.0)
    return 1 + (100.0 / abs(odds))


def _best_book_odds(legs: pd.DataFrame) -> tuple[float, str]:
    """Return (best combined decimal odds, book name) across available bookmakers."""
    best_odds = float(legs["decimal_odds"].prod())
    best_book = "novig"
    for book in ("fanduel", "draftkings", "betmgm"):
        col = f"odds_american_{book}"
        if col in legs.columns and legs[col].notna().all():
            book_odds = float(legs[col].apply(_amer_to_dec).prod())
            if book_odds > best_odds:
                best_odds = book_odds
                best_book = book
    return best_odds, best_book


def _adj_prob(legs: pd.DataFrame, prob_col: str, rho: float = 0.8) -> float:
    """Combined probability with power-law correlation penalty for same-game legs."""
    result = 1.0
    for matchup_id in legs["matchup_id"].unique():
        m_legs = legs[legs["matchup_id"] == matchup_id]
        if len(m_legs) > 1:
            p_prod = m_legs[prob_col].prod()
            p_min = m_legs[prob_col].min()
            result *= (p_prod ** (1 - rho)) * (p_min ** rho)
        else:
            result *= float(m_legs[prob_col].iloc[0])
    return float(result)


def _leg_labels(legs: pd.DataFrame, label_cols: list[str]) -> list[str]:
    labels: list[str] = []
    has_teams = {"away_team", "home_team"}.issubset(set(label_cols))
    for _, row in legs.iterrows():
        game_ctx = ""
        if has_teams and pd.notna(row.get("away_team")) and pd.notna(row.get("home_team")):
            game_ctx = f"{row['away_team']} @ {row['home_team']}"

        if "best_pick" in label_cols and pd.notna(row.get("best_pick")) and str(row.get("best_pick")).strip():
            pick = str(row["best_pick"])
            labels.append(f"{game_ctx}: {pick}" if game_ctx else pick)
        elif "team" in label_cols and pd.notna(row.get("team")):
            labels.append(str(row["team"]))
        elif game_ctx:
            labels.append(game_ctx)
        elif "away_team" in label_cols:
            labels.append(str(row["away_team"]))
        else:
            labels.append("leg")
    return labels


def _build_record(legs: pd.DataFrame, label_cols: list[str], leg_count: int,
                  risk_tier: str, group_id=pd.NA) -> dict | None:
    """Build a parlay record dict; returns None if EV is non-positive."""
    combined_probability = _adj_prob(legs, "calibrated_probability")
    combined_market_prob = _adj_prob(legs, "market_probability")
    combined_decimal_odds, best_book = _best_book_odds(legs)

    parlay_ev = (combined_probability * (combined_decimal_odds - 1)) - (1 - combined_probability)
    if parlay_ev <= 0:
        return None

    ev_boost_pct = (
        (combined_probability - combined_market_prob) / combined_market_prob
        if combined_market_prob > 0 else 0.0
    )
    is_high_correlation = len(legs["matchup_id"].unique()) < leg_count
    parlay_conviction = float(legs["Conviction_Score"].mean()) if "Conviction_Score" in legs.columns else pd.NA
    min_leg_prob = float(legs["calibrated_probability"].min())

    return {
        "parlay_legs": " | ".join(_leg_labels(legs, label_cols)),
        "combined_probability": combined_probability,
        "combined_decimal_odds": combined_decimal_odds,
        "parlay_ev": parlay_ev,
        "legs": leg_count,
        "combined_market_prob": combined_market_prob,
        "ev_boost_pct": ev_boost_pct,
        "is_high_correlation": is_high_correlation,
        "risk_tier": risk_tier,
        "group_id": group_id,
        "best_payout_book": best_book.capitalize() if best_book != "novig" else "Novig",
        "Conviction_Score": parlay_conviction,
        "min_leg_prob": min_leg_prob,
    }


def generate_smart_parlays(df: pd.DataFrame, num_rr_candidates: int = 5) -> pd.DataFrame:
    """Generate +EV 2- and 3-leg parlays from quality-filtered candidates."""
    columns = [
        "parlay_legs", "combined_probability", "combined_decimal_odds", "parlay_ev",
        "legs", "combined_market_prob", "ev_boost_pct", "is_high_correlation",
        "risk_tier", "group_id", "best_payout_book", "Conviction_Score", "min_leg_prob",
    ]
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)

    needed = {"edge", "calibrated_probability", "decimal_odds", "market_probability", "matchup_id"}
    if not needed.issubset(df.columns):
        return pd.DataFrame(columns=columns)

    # Quality gate: only Actionable and High Variance picks as legs
    candidates = df.copy()
    if "Pick_Status" in candidates.columns:
        candidates = candidates[
            candidates["Pick_Status"].astype(str).isin(PARLAY_ELIGIBLE_STATUSES)
        ]

    # Per-leg probability and edge floor
    candidates = candidates[
        candidates["calibrated_probability"].ge(MIN_LEG_PROBABILITY)
        & candidates["edge"].ge(MIN_LEG_EDGE)
    ]

    candidate_bets = candidates.nlargest(20, "edge").dropna(
        subset=["calibrated_probability", "decimal_odds"]
    ).copy()

    if candidate_bets.empty:
        return pd.DataFrame(columns=columns)

    rr_candidates = candidates.nlargest(num_rr_candidates, "edge").dropna(
        subset=["calibrated_probability", "decimal_odds"]
    ).copy()

    label_cols = [c for c in ["best_pick", "team", "away_team", "home_team"] if c in candidate_bets.columns]
    records: list[dict] = []

    # 2-leg and 3-leg parlays only
    for leg_count in (2, 3):
        if len(candidate_bets) < leg_count:
            continue
        for combo in combinations(candidate_bets.index, leg_count):
            legs = candidate_bets.loc[list(combo)]
            risk_tier = "Bankroll Builder" if leg_count == 2 else "Standard"
            rec = _build_record(legs, label_cols, leg_count, risk_tier)
            if rec is not None:
                records.append(rec)

    # Round-Robin packages from the top quality candidates
    if not rr_candidates.empty and len(rr_candidates) >= 2:
        leg_names = _leg_labels(rr_candidates, label_cols)
        group_hash = hashlib.md5(json.dumps(sorted(leg_names)).encode()).hexdigest()[:8]
        rr_group_id = f"RR_{group_hash}"

        for leg_count in (2, 3):
            if len(rr_candidates) < leg_count:
                continue
            for combo in combinations(rr_candidates.index, leg_count):
                legs = rr_candidates.loc[list(combo)]
                rec = _build_record(legs, label_cols, leg_count, "Round Robin", rr_group_id)
                if rec is not None:
                    records.append(rec)

    if not records:
        return pd.DataFrame(columns=columns)

    result = pd.DataFrame(records)

    # Drop duplicate combinations (same legs can appear in both standard and RR pools)
    result = result.drop_duplicates(subset=["parlay_legs"]).copy()

    # Sort: highest EV first; use min_leg_prob as tiebreaker (stronger weakest leg wins)
    result = result.sort_values(
        ["parlay_ev", "min_leg_prob"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return result
