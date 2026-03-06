from __future__ import annotations

import numpy as np
import pandas as pd


def _numeric_series(df: pd.DataFrame, col: str, default: float | int | None = None) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype="float64")
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
    else:
        s = pd.Series([pd.NA] * len(df), index=df.index, dtype="Float64")
    if default is not None:
        s = s.fillna(default)
    return s


def _default_simulation_payload(starting_bankroll: float) -> dict[str, float | list[list[float]] | int]:
    return {
        "expected_bankroll": float(starting_bankroll),
        "median_final_bankroll": float(starting_bankroll),
        "mean_final_bankroll": float(starting_bankroll),
        "risk_of_ruin": 0.0,
        "ruin_probability": 0.0,
        "max_drawdown": 0.0,
        "bankroll_curves": [],
        "curves": [],
        "inputs_used": 0,
    }


def simulate_bankroll(
    portfolio_df: pd.DataFrame,
    starting_bankroll: float,
    days: int = 1000,
    simulations: int = 1000,
) -> dict[str, float | list[list[float]]]:
    """Monte Carlo simulation of bankroll using recommended single bets."""
    if portfolio_df is None or portfolio_df.empty:
        return _default_simulation_payload(starting_bankroll)

    probs = _numeric_series(portfolio_df, "calibrated_probability", 0.5).clip(0.0, 1.0)
    odds = _numeric_series(portfolio_df, "decimal_odds", 2.0)
    stakes = _numeric_series(portfolio_df, "recommended_bet", 0.0).clip(lower=0.0)

    if stakes.empty or float(stakes.sum()) <= 0:
        return _default_simulation_payload(starting_bankroll)

    probs_arr = probs.to_numpy()
    odds_arr = odds.to_numpy()
    stakes_arr = stakes.to_numpy()

    curves: list[list[float]] = []
    ending_bankrolls: list[float] = []
    max_drawdowns: list[float] = []

    for _ in range(simulations):
        bankroll = float(starting_bankroll)
        curve = [bankroll]
        peak = bankroll
        mdd = 0.0

        for _day in range(days):
            outcomes = np.random.random(size=len(probs_arr)) < probs_arr
            pnl = np.where(outcomes, stakes_arr * (odds_arr - 1), -stakes_arr).sum()
            bankroll = max(bankroll + float(pnl), 0.0)
            curve.append(bankroll)

            peak = max(peak, bankroll)
            if peak > 0:
                drawdown = (peak - bankroll) / peak
                mdd = max(mdd, float(drawdown))

        curves.append(curve)
        ending_bankrolls.append(bankroll)
        max_drawdowns.append(mdd)

    ending_array = np.array(ending_bankrolls)
    mean_final_bankroll = float(np.mean(ending_array))
    ruin_probability = float(np.mean(ending_array <= (0.05 * starting_bankroll)))
    return {
        "expected_bankroll": mean_final_bankroll,
        "median_final_bankroll": float(np.median(ending_array)),
        "mean_final_bankroll": mean_final_bankroll,
        "risk_of_ruin": ruin_probability,
        "ruin_probability": ruin_probability,
        "max_drawdown": float(np.mean(max_drawdowns)),
        "bankroll_curves": curves,
        "curves": curves,
        "inputs_used": int(len(stakes_arr)),
    }
