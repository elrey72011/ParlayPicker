import pandas as pd

from app_core.best_duos import build_best_duos
from app_core.prop_runner import apply_probation_portfolio_guard


def test_probation_guard_limits_market_and_game_concentration():
    card = pd.DataFrame({
        "matchup": ["A @ B", "C @ D", "E @ F", "A @ B", "G @ H"],
        "market_type": ["batter_hits_under", "batter_hits_over", "batter_hits_under",
                        "pitcher_walks_over", "pitcher_walks_under"],
        "WinProbability": [0.75, 0.74, 0.73, 0.72, 0.71],
        "expected_value": [0.08] * 5,
        "edge": [0.07] * 5,
        "Kelly_Bet_Size": [1.0] * 5,
        "Market_Probation": [True] * 5,
    })
    out = apply_probation_portfolio_guard(card, bankroll=500.0)
    selected = out[pd.to_numeric(out["Kelly_Bet_Size"]) > 0]
    assert len(selected) == 3
    assert selected["matchup"].nunique() == 3
    families = selected["market_type"].str.replace(r"_(over|under)$", "", regex=True)
    assert families.value_counts().max() <= 2


def test_probation_parlay_excludes_unallocated_props():
    props = pd.DataFrame({
        "league": ["MLB", "MLB"],
        "matchup": ["A @ B", "C @ D"],
        "best_pick": ["A Under 1.5 Hits", "B Over 0.5 Hits"],
        "WinProbability": [0.72, 0.70],
        "expected_value": [0.10, 0.10],
        "edge": [0.08, 0.07],
        "odds_american": [-110, -110],
        "Pick_Status": ["Actionable", "Actionable"],
        "Market_Probation": [True, True],
        "Kelly_Bet_Size": [1.0, 0.0],
    })
    assert build_best_duos(None, props, strict=True, allow_probation=True).empty
