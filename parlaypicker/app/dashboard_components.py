from __future__ import annotations

import polars as pl
import streamlit as st

from parlaypicker.core.ev_engine import bet_signal, expected_value
from parlaypicker.core.portfolio_engine import Bet, optimize_portfolio, portfolio_metrics, simulate_portfolio


def _to_bets(df: pl.DataFrame) -> list[Bet]:
    required = {"game_id", "true_probability", "odds"}
    if not required.issubset(set(df.columns)):
        return []

    bets: list[Bet] = []
    for row in df.iter_rows(named=True):
        ev = row.get("expected_value")
        if ev is None:
            ev = expected_value(float(row["true_probability"]), float(row["odds"]))

        if bet_signal(float(ev), min_edge=0.03):
            bets.append(
                Bet(
                    str(row["game_id"]),
                    float(row["true_probability"]),
                    float(row["odds"]),
                    float(ev),
                )
            )
    return bets


def render_top_bets(df: pl.DataFrame, bankroll: float = 1000.0):
    st.subheader("Top Bets")
    st.dataframe(df.head(20).to_pandas(), width="stretch")

    bets = _to_bets(df)
    if not bets:
        st.info("No positive-EV opportunities available for portfolio optimization.")
        return

    allocations = optimize_portfolio(bets, bankroll=bankroll, max_fraction=0.05)
    metrics = portfolio_metrics(bets, allocations, bankroll)
    monte_carlo = simulate_portfolio(bets, allocations, bankroll=bankroll, seasons=10000)

    portfolio_df = pl.DataFrame(
        {
            "game_id": [bet.game_id for bet in bets],
            "probability": [bet.probability for bet in bets],
            "odds": [bet.odds for bet in bets],
            "expected_value": [bet.ev for bet in bets],
            "optimal_bet_size": allocations,
        }
    )

    st.subheader("Portfolio Optimization")
    st.dataframe(portfolio_df.to_pandas(), width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Expected portfolio return", f"{metrics['expected_portfolio_return']:.2%}")
        st.metric("Portfolio risk", f"{metrics['portfolio_risk']:.2%}")
        st.metric("Total bankroll exposure", f"{metrics['total_bankroll_exposure']:.2%}")
    with col2:
        st.metric("Portfolio variance", f"{metrics['portfolio_variance']:.6f}")
        st.metric("Sharpe ratio", f"{metrics['sharpe_ratio']:.2f}")
        st.metric("Max drawdown estimate", f"{metrics['max_drawdown_estimate']:.2%}")

    st.caption(
        "Monte Carlo (10,000 seasons): "
        f"mean={monte_carlo['mean_return']:.2%}, "
        f"p05={monte_carlo['p05_return']:.2%}, "
        f"p95={monte_carlo['p95_return']:.2%}, "
        f"P(profit)={monte_carlo['probability_of_profit']:.2%}"
    )
