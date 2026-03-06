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
    out["kelly_fraction"] = out.apply(
        lambda r: kelly_fraction(r.get("calibrated_probability"), r.get("decimal_odds")),
        axis=1,
    )
    out["recommended_bet"] = float(bankroll) * out["kelly_fraction"] * float(fraction)
    return out
