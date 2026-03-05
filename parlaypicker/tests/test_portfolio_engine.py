import numpy as np

from parlaypicker.core.portfolio_engine import (
    Bet,
    expected_return,
    optimize_portfolio,
    payout_from_odds,
    portfolio_metrics,
    simulate_portfolio,
)
from parlaypicker.core.portfolio_engine.correlation_matrix import build_correlation_matrix
from parlaypicker.core.portfolio_engine.risk_model import bet_variance


def test_payout_from_odds():
    assert payout_from_odds(150) == 1.5
    assert round(payout_from_odds(-110), 6) == round(100 / 110, 6)


def test_expected_return_matches_ev_formula():
    assert round(expected_return(0.55, -110), 6) > 0


def test_bet_variance_positive():
    assert bet_variance(0.55, -110) > 0


def test_correlation_matrix_same_game_high_corr():
    bets = [Bet("g1", 0.55, -110, 0.05), Bet("g1", 0.53, -105, 0.04), Bet("g2", 0.54, 120, 0.03)]
    corr = build_correlation_matrix(bets)
    assert corr.shape == (3, 3)
    assert corr[0, 1] == 0.8
    assert corr[0, 2] == 0


def test_optimize_portfolio_respects_max_fraction():
    bets = [Bet("g1", 0.58, -110, 0.06), Bet("g2", 0.57, -105, 0.05), Bet("g3", 0.56, 120, 0.04)]
    bankroll = 1000
    allocations = optimize_portfolio(bets, bankroll=bankroll, max_fraction=0.05)
    assert len(allocations) == len(bets)
    assert np.all(allocations <= bankroll * 0.05 + 1e-9)


def test_portfolio_metrics_and_simulation():
    bets = [Bet("g1", 0.58, -110, 0.06), Bet("g2", 0.56, 120, 0.04)]
    allocations = np.array([50.0, 50.0])
    metrics = portfolio_metrics(bets, allocations, bankroll=1000)
    sim = simulate_portfolio(bets, allocations, bankroll=1000, seasons=500)
    assert metrics["portfolio_variance"] >= 0
    assert -1 <= sim["mean_return"] <= 1
