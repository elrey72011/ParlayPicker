from __future__ import annotations

from parlaypicker.simulation.monte_carlo_simulator import simulate_seasons


def run_bankroll_projection(starting_bankroll: float, win_rate: float, avg_odds: float):
    return simulate_seasons(starting_bankroll, win_rate, avg_odds)
