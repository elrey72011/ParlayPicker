import pandas as pd

from app_core.best_duos import build_best_duos
from app_core.prop_runner import (
    apply_probation_portfolio_guard,
    apply_probation_share_guard,
    apply_prop_participant_guard,
)


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
    assert len(selected) == 2
    assert selected["matchup"].nunique() == 2
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


def test_participant_guard_funds_only_one_market_per_player():
    card = pd.DataFrame({
        "player": ["Shane Baz", "Shane Baz", "Matthew Boyd"],
        "market_type": [
            "pitcher_walks_over", "pitcher_strikeouts_under", "pitcher_strikeouts_under"
        ],
        "WinProbability": [0.67, 0.66, 0.65],
        "expected_value": [0.12, 0.10, 0.08],
        "edge": [0.08, 0.07, 0.06],
        "Kelly_Bet_Size": [1.0, 1.0, 1.0],
        "Market_Probation": [True, False, False],
    })
    out = apply_prop_participant_guard(card)
    funded = out[pd.to_numeric(out["Kelly_Bet_Size"]) > 0]
    assert len(funded) == 2
    assert funded["player"].str.lower().nunique() == 2
    assert funded.loc[funded["player"].eq("Shane Baz"), "market_type"].iloc[0] == (
        "pitcher_strikeouts_under"
    )


def test_probation_share_is_no_more_than_twenty_five_percent():
    card = pd.DataFrame({
        "player": [f"P{i}" for i in range(8)],
        "WinProbability": [0.80, 0.79, 0.78, 0.77, 0.76, 0.75, 0.74, 0.73],
        "expected_value": [0.08] * 8,
        "edge": [0.07] * 8,
        "Kelly_Bet_Size": [1.0] * 8,
        "Market_Probation": [False] * 6 + [True, True],
    })
    out = apply_probation_share_guard(card)
    funded = out[pd.to_numeric(out["Kelly_Bet_Size"]) > 0]
    probation_stake = pd.to_numeric(
        funded.loc[funded["Market_Probation"], "Kelly_Bet_Size"]
    ).sum()
    total_stake = pd.to_numeric(funded["Kelly_Bet_Size"]).sum()
    assert probation_stake == 2.0
    assert probation_stake / total_stake <= 0.25
