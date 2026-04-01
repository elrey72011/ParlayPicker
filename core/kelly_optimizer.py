from __future__ import annotations

import pandas as pd


def kelly_fraction(prob: float, decimal_odds: float) -> float:
    """Return Kelly fraction, floored at zero for no-bet scenarios."""
    if pd.isna(prob) or pd.isna(decimal_odds) or decimal_odds <= 1:
        return 0.0

    b = decimal_odds - 1
    q = 1 - prob
    f = (b * prob - q) / b
    return max(float(f), 0.0)


def add_kelly_bet_sizing(df: pd.DataFrame, bankroll: float, fraction: float = 0.25) -> pd.DataFrame:
    """Attach Kelly fractions and fractional Kelly bankroll recommendations."""
    if df is None or df.empty:
        return df

    out = df.copy()

    # We use combined_probability and combined_decimal_odds for parlays if present
    prob_col = "combined_probability" if "combined_probability" in out.columns else "calibrated_probability"
    odds_col = "combined_decimal_odds" if "combined_decimal_odds" in out.columns else "decimal_odds"

    out["kelly_fraction"] = out.apply(
        lambda r: kelly_fraction(r.get(prob_col), r.get(odds_col)),
        axis=1,
    )
    out["recommended_bet"] = float(bankroll) * out["kelly_fraction"] * float(fraction)
    return out


def apply_simultaneous_kelly(df: pd.DataFrame, bankroll: float, max_exposure: float = 0.05) -> pd.DataFrame:
    """
    Applies Simultaneous Kelly Optimization (Outcome Exposure Cap).
    Ensures that the sum of the recommended bet sizes (as a % of bankroll)
    tied to any specific specific game outcome (parlay leg) does not exceed
    `max_exposure` (default 5%).

    If it does, it linearly scales down the bet sizes for all parlays
    containing that specific leg.

    Also calculates a unified_bankroll_recommendation for Round Robin groups.
    """
    if df is None or df.empty or "recommended_bet" not in df.columns or "parlay_legs" not in df.columns:
        return df

    out = df.copy()

    # 1. Calculate Leg Exposure Map
    # Since parlay_legs is a joined string like "Team A | Team B vs Team C",
    # we can just use the exact string components to track exposure.
    # Note: If two parlays share "Team A", they share that outcome exposure.
    exposure_map = {}

    for idx, row in out.iterrows():
        rec_bet = float(row.get("recommended_bet", 0.0))
        if rec_bet <= 0:
            continue

        pct_exposure = rec_bet / bankroll
        legs_str = str(row.get("parlay_legs", ""))
        legs = [l.strip() for l in legs_str.split("|")]

        for leg in legs:
            if leg:
                exposure_map[leg] = exposure_map.get(leg, 0.0) + pct_exposure

    # 2. Identify over-exposed legs and their scaling factors
    scale_factors = {}
    for leg, total_exposure in exposure_map.items():
        if total_exposure > max_exposure:
            scale_factors[leg] = max_exposure / total_exposure

    # 3. Apply the strictest scale-down factor for each parlay
    for idx, row in out.iterrows():
        legs_str = str(row.get("parlay_legs", ""))
        legs = [l.strip() for l in legs_str.split("|")]

        min_scale = 1.0
        for leg in legs:
            if leg in scale_factors:
                min_scale = min(min_scale, scale_factors[leg])

        if min_scale < 1.0:
            out.at[idx, "recommended_bet"] = out.at[idx, "recommended_bet"] * min_scale

    # 4. Calculate Unified Bankroll Recommendation for Round Robin Groups
    out["unified_bankroll_recommendation"] = pd.NA

    if "group_id" in out.columns:
        # Group by group_id and sum the adjusted recommended bets
        rr_groups = out[out["group_id"].notna() & (out["risk_tier"] == "Round Robin")]
        if not rr_groups.empty:
            for group_id, group_df in rr_groups.groupby("group_id"):
                total_bet = group_df["recommended_bet"].sum()
                pct = (total_bet / bankroll) * 100
                # Attach the string description to every row in this RR group
                msg = f"Wager {pct:.2f}% of bankroll total across these {len(group_df)} combinations"
                out.loc[out["group_id"] == group_id, "unified_bankroll_recommendation"] = msg

    return out
