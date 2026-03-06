from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_bankroll(
    portfolio_df: pd.DataFrame,
    starting_bankroll: float,
    days: int = 1000,
    simulations: int = 1000,
) -> dict[str, float | list[list[float]]]:
    """Monte Carlo simulation of bankroll using recommended single bets."""
    if portfolio_df is None or portfolio_df.empty:
        return {
            "expected_bankroll": float(starting_bankroll),
            "risk_of_ruin": 0.0,
            "max_drawdown": 0.0,
            "bankroll_curves": [],
        }

    probs = pd.to_numeric(portfolio_df.get("calibrated_probability"), errors="coerce").fillna(0.5).to_numpy()
    odds = pd.to_numeric(portfolio_df.get("decimal_odds"), errors="coerce").fillna(2.0).to_numpy()
    stakes = pd.to_numeric(portfolio_df.get("recommended_bet"), errors="coerce").fillna(0.0).to_numpy()

    curves: list[list[float]] = []
    ending_bankrolls: list[float] = []
    max_drawdowns: list[float] = []

    for _ in range(simulations):
        bankroll = float(starting_bankroll)
        curve = [bankroll]
        peak = bankroll
        mdd = 0.0

        for _day in range(days):
            outcomes = np.random.random(size=len(probs)) < probs
            pnl = np.where(outcomes, stakes * (odds - 1), -stakes).sum()
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
    return {
        "expected_bankroll": float(np.mean(ending_array)),
        "risk_of_ruin": float(np.mean(ending_array <= (0.05 * starting_bankroll))),
        "max_drawdown": float(np.mean(max_drawdowns)),
        "bankroll_curves": curves,
    }
