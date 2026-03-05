from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import numpy as np


def _simulate_once(seed: int, starting_bankroll: float, win_rate: float, avg_odds: float, bets_per_season: int) -> float:
    rng = np.random.default_rng(seed)
    wins = rng.random(bets_per_season) < win_rate
    payout = np.where(avg_odds > 0, avg_odds / 100, 100 / abs(avg_odds))
    returns = np.where(wins, payout, -1.0)
    return starting_bankroll * (1 + returns.mean())


def _simulate_once_star(args):
    return _simulate_once(*args)


def simulate_seasons(starting_bankroll: float, win_rate: float, avg_odds: float, bets_per_season: int = 300, seasons: int = 10000, workers: int = 4):
    seeds = list(range(seasons))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        args = [(s, starting_bankroll, win_rate, avg_odds, bets_per_season) for s in seeds]
        results = list(ex.map(_simulate_once_star, args))
    arr = np.array(results)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "p90": float(np.quantile(arr, 0.90)),
    }
